"""Model building, cross-validation, metrics and probability calibration.

Design choices (defended in INTERVIEW_PREP.md):
* All preprocessing lives inside sklearn Pipelines, so it is re-fitted on each
  CV training fold -> no leakage from validation folds through scalers/encoders.
* Class imbalance (~16% churners) is handled by stratified splits, ranking
  metrics (PR-AUC, recall@top-k) and a profit-based decision threshold —
  not by blindly resampling. Class weights are evaluated as an option.
* Final probabilities are calibrated on out-of-fold predictions because the
  business layer multiplies them by money.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.data import CATEGORICAL_COLS, ENGINEERED_COLS, FEATURES, NUMERIC_COLS, TARGET

# Heavy-tailed money columns get log1p before scaling in the linear model.
LOG_COLS = ["credit_limit", "total_revolving_bal", "avg_open_to_buy",
            "total_trans_amt", "avg_trans_amt"]


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def split(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified hold-out split. The test set is touched exactly once, at the end."""
    X, y = df[FEATURES], df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_logreg(class_weight: str | None = "balanced", seed: int = 42) -> Pipeline:
    """Baseline: one-hot + log/scale + L2 logistic regression.

    `avg_open_to_buy` is dropped: it equals credit_limit - total_revolving_bal,
    so it is perfectly collinear and makes coefficients unstable.
    """
    log_cols = [c for c in LOG_COLS if c != "avg_open_to_buy"]
    other_num = [c for c in NUMERIC_COLS + ENGINEERED_COLS
                 if c not in log_cols and c != "avg_open_to_buy"]
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), CATEGORICAL_COLS),
        ("log", Pipeline([("log1p", FunctionTransformer(np.log1p)),
                          ("scale", StandardScaler())]), log_cols),
        ("num", StandardScaler(), other_num),
    ])
    clf = LogisticRegression(C=1.0, class_weight=class_weight, max_iter=2000,
                             random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def make_catboost(seed: int = 42, class_weights: str | None = None, **params) -> CatBoostClassifier:
    """Main model. CatBoost handles categoricals natively (ordered target
    statistics), which avoids one-hot blow-up and target-encoding leakage."""
    defaults = dict(
        iterations=800,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        loss_function="Logloss",
        eval_metric="PRAUC",
        cat_features=CATEGORICAL_COLS,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
        allow_writing_files=False,
    )
    if class_weights:
        defaults["auto_class_weights"] = class_weights
    defaults.update(params)
    return CatBoostClassifier(**defaults)


def make_lightgbm(seed: int = 42) -> Pipeline:
    """Secondary GBM for comparison. Categoricals are one-hot encoded."""
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
    ], remainder="passthrough")
    clf = LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31,
                         subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                         random_state=seed, verbose=-1, n_jobs=-1)
    return Pipeline([("pre", pre), ("clf", clf)])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def recall_at_top_k(y_true, scores, k: float = 0.10) -> float:
    """Share of all churners captured if we contact the top-k% highest scores.

    This is the metric the campaign actually cares about: with budget for
    k% of the base, how many of the leavers do we reach?
    """
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(-np.asarray(scores), kind="stable")[:n_top]
    return float(y_true[top_idx].sum() / y_true.sum())


def precision_at_top_k(y_true, scores, k: float = 0.10) -> float:
    """Share of contacted customers (top-k%) who are real churners."""
    y_true = np.asarray(y_true)
    n_top = int(np.ceil(k * len(y_true)))
    top_idx = np.argsort(-np.asarray(scores), kind="stable")[:n_top]
    return float(y_true[top_idx].mean())


def evaluate(y_true, proba, k: float = 0.10) -> dict[str, float]:
    """All metrics reported in the project, for one set of predictions.

    Ranking metrics work for any score; Brier and log-loss only make sense for
    probabilities, so they are NaN when the score is outside [0, 1] (e.g. the
    rule-based flag count).
    """
    proba = np.asarray(proba)
    is_prob = proba.min() >= 0 and proba.max() <= 1
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        f"recall@top{int(k * 100)}%": recall_at_top_k(y_true, proba, k),
        f"precision@top{int(k * 100)}%": precision_at_top_k(y_true, proba, k),
        "brier": brier_score_loss(y_true, proba) if is_prob else np.nan,
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6)) if is_prob else np.nan,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def fresh_copy(model):
    """Unfitted copy of a model. sklearn's clone() rejects CatBoost because
    CatBoost copies list parameters (cat_features), so rebuild it from params."""
    if isinstance(model, CatBoostClassifier):
        return CatBoostClassifier(**model.get_params())
    return clone(model)


@dataclass
class CVResult:
    name: str
    fold_metrics: pd.DataFrame      # one row per fold
    oof_proba: np.ndarray           # out-of-fold predictions on the training set

    def summary(self) -> pd.Series:
        m = self.fold_metrics.agg(["mean", "std"]).T
        return m.apply(lambda r: f"{r['mean']:.4f} ± {r['std']:.4f}", axis=1).rename(self.name)


def cross_validate(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5,
                   seed: int = 42, k: float = 0.10, name: str = "model") -> CVResult:
    """Stratified K-fold CV returning per-fold metrics and OOF predictions.

    The model is cloned for every fold so nothing learned on one fold's
    validation data can carry over.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    rows = []
    for fold, (tr, va) in enumerate(skf.split(X, y)):
        m = fresh_copy(model)
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        rows.append({"fold": fold, **evaluate(y.iloc[va], p, k)})
    return CVResult(name, pd.DataFrame(rows).set_index("fold"), oof)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Sigmoid calibration fitted on logit of raw scores (2 parameters)."""

    def fit(self, proba, y):
        logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6)))
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(logit.reshape(-1, 1), y)
        return self

    def predict(self, proba):
        logit = np.log(np.clip(proba, 1e-6, 1 - 1e-6) / (1 - np.clip(proba, 1e-6, 1 - 1e-6)))
        return self.lr_.predict_proba(logit.reshape(-1, 1))[:, 1]


class IsotonicCalibrator:
    """Monotone, non-parametric calibration. Needs more data than Platt."""

    def fit(self, proba, y):
        self.iso_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(proba, y)
        return self

    def predict(self, proba):
        return self.iso_.predict(proba)


class IdentityCalibrator:
    """No-op calibrator, used when raw scores are already the best option."""

    def fit(self, proba, y):
        return self

    def predict(self, proba):
        return np.asarray(proba)


def fit_calibrator(oof_proba, y, method: str = "isotonic"):
    """Fit a calibrator on OUT-OF-FOLD scores (never on scores the model was
    trained on — those are over-confident and would mis-calibrate)."""
    cal = IsotonicCalibrator() if method == "isotonic" else PlattCalibrator()
    return cal.fit(np.asarray(oof_proba), np.asarray(y))


def select_calibrator(oof_proba, y, methods=("none", "platt", "isotonic"),
                      n_splits: int = 5, seed: int = 42) -> tuple[str, pd.DataFrame]:
    """Choose a calibration method WITHOUT touching the test set.

    The out-of-fold scores are split again with stratified K-fold: each
    calibrator is fitted on 4/5 of the OOF scores and scored (Brier, ECE) on the
    remaining 1/5. The method with the lowest mean Brier wins. "none" keeps the
    raw scores, so calibration is only used if it actually helps.
    """
    oof_proba, y = np.asarray(oof_proba), np.asarray(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for fold, (tr, va) in enumerate(skf.split(oof_proba.reshape(-1, 1), y)):
        for method in methods:
            if method == "none":
                p = oof_proba[va]
            else:
                p = fit_calibrator(oof_proba[tr], y[tr], method).predict(oof_proba[va])
            rows.append({"fold": fold, "method": method,
                         "brier": brier_score_loss(y[va], p),
                         "ece": expected_calibration_error(y[va], p)})
    table = pd.DataFrame(rows).groupby("method")[["brier", "ece"]].agg(["mean", "std"])
    best = table[("brier", "mean")].idxmin()
    return best, table


def reliability_table(y_true, proba, n_bins: int = 10) -> pd.DataFrame:
    """Observed churn rate vs mean predicted probability per quantile bin."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=n_bins, strategy="quantile")
    return pd.DataFrame({"mean_predicted": mean_pred, "observed_rate": frac_pos})


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Weighted mean |observed - predicted| over equal-frequency bins."""
    y_true, proba = np.asarray(y_true), np.asarray(proba)
    bins = np.quantile(proba, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(bins, proba, side="right") - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y_true[mask].mean() - proba[mask].mean())
    return float(ece)


# ---------------------------------------------------------------------------
# Leakage diagnostics
# ---------------------------------------------------------------------------

def single_feature_auc(df: pd.DataFrame, cols: list[str], target: str = TARGET) -> pd.DataFrame:
    """ROC-AUC of every numeric column used alone as a score.

    An AUC close to 1.0 for a single raw column is a red flag for leakage
    (the column probably encodes the label or was recorded after the event).
    AUC is symmetrised: max(auc, 1-auc), because direction does not matter here.
    """
    rows = []
    for c in cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            auc = roc_auc_score(df[target], df[c])
            rows.append({"feature": c, "auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
