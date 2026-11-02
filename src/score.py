"""Batch scoring CLI: customers CSV in -> campaign list with actions out.

    python -m src.score --input new_customers.csv --output scored.csv
    python -m src.score --input data/raw/BankChurners.csv --output reports/scored_customers.csv

What it does for every customer:
    1. calibrated churn probability (CatBoost + calibrator from models/)
    2. customer value (CLV proxy) and expected profit of a contact (config.yaml)
    3. contact decision: expected profit > 0
    4. primary / secondary next-best-action from SHAP driver families
It also prints a drift report (PSI) of the input batch against the training
data, so a shifted customer base is flagged before anyone trusts the scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from src import business as B
from src.config import load_config, path
from src.data import CATEGORICAL_COLS, FEATURES, NUMERIC_COLS, load_dataset, prepare_for_scoring
from src.monitoring import drift_report
from src.recommend import assign_actions


def load_model():
    model = CatBoostClassifier()
    model.load_model(str(path("models") / "catboost_churn.cbm"))
    calibrator = joblib.load(path("models") / "calibrator.joblib")
    return model, calibrator


def load_reference() -> pd.DataFrame:
    """Training-set features saved by notebook 02 (the drift baseline)."""
    ref_file = path("models") / "train_reference.csv"
    if ref_file.exists():
        return pd.read_csv(ref_file)
    print("models/train_reference.csv not found; using the full dataset as drift reference")
    return load_dataset()


def score_frame(df: pd.DataFrame, model=None, calibrator=None) -> pd.DataFrame:
    """Score a model-ready frame (output of prepare_for_scoring)."""
    if model is None:
        model, calibrator = load_model()
    a = B.Assumptions.from_config(load_config())
    pool = Pool(df[FEATURES], cat_features=CATEGORICAL_COLS)
    proba = calibrator.predict(model.predict_proba(pool)[:, 1])
    shap_values = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    clv = B.customer_value(df, a).to_numpy()
    ev = B.expected_profit(proba, clv, a)

    out = pd.DataFrame({
        "clientnum": df["clientnum"].to_numpy() if "clientnum" in df else np.arange(len(df)),
        "churn_probability": proba.round(4),
        "clv": clv.round(2),
        "expected_profit": ev.round(2),
        "contact": (ev > 0).astype(int),
    })
    actions = assign_actions(df[["total_revolving_bal"]], shap_values, FEATURES)
    for col in ["action_family", "offer", "secondary_family", "secondary_offer", "top3_drivers"]:
        out[col] = np.where(out["contact"] == 1, actions[col], "")
    return out.sort_values("expected_profit", ascending=False).reset_index(drop=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--no-drift", action="store_true", help="skip the PSI drift report")
    args = parser.parse_args(argv)

    batch = prepare_for_scoring(pd.read_csv(args.input))
    if not args.no_drift:
        report = drift_report(load_reference(), batch, NUMERIC_COLS, CATEGORICAL_COLS)
        flagged = report[report.status != "stable"]
        print(f"Drift check vs training data: {len(flagged)} of {len(report)} features shifted")
        if len(flagged):
            print(flagged.round(3).to_string(index=False))

    scored = score_frame(batch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output, index=False)
    n = scored.contact.sum()
    print(f"Scored {len(scored):,} customers -> {args.output}")
    print(f"Contact {n:,} ({n / len(scored):.1%}); expected campaign profit "
          f"${scored.loc[scored.contact == 1, 'expected_profit'].sum():,.0f}")


if __name__ == "__main__":
    main()
