"""Production monitoring: data drift (PSI) and score drift.

Labels arrive late in churn (we only know who left ~90 days later), so the
early-warning system has to watch the INPUTS and the SCORES:

* PSI per feature: has the customer base changed compared with training?
* PSI of the churn score: has the model's output distribution moved?
* Once labels arrive: calibration (predicted vs observed churn) per score band.

PSI = sum over bins of (cur% - ref%) * ln(cur% / ref%).
Rule of thumb used in credit-risk model validation:
    < 0.10 stable · 0.10–0.25 moderate shift, investigate · > 0.25 major shift
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PSI_THRESHOLDS = (0.10, 0.25)


def psi(reference, current, n_bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index for one numeric variable.

    Bins are quantiles of the REFERENCE distribution, so each reference bin
    holds ~10% of rows; discrete variables collapse duplicate edges.
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(reference, edges)[0] / len(reference)
    cur_share = np.histogram(current, edges)[0] / len(current)
    ref_share, cur_share = np.clip(ref_share, eps, None), np.clip(cur_share, eps, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def psi_categorical(reference, current, eps: float = 1e-4) -> float:
    """PSI for a categorical variable (one bin per category)."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    cats = ref.index.union(cur.index)
    r = ref.reindex(cats, fill_value=0).clip(lower=eps).to_numpy()
    c = cur.reindex(cats, fill_value=0).clip(lower=eps).to_numpy()
    return float(np.sum((c - r) * np.log(c / r)))


def status(value: float) -> str:
    low, high = PSI_THRESHOLDS
    return "stable" if value < low else ("investigate" if value < high else "ALERT")


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    """PSI and status for every feature, worst first."""
    rows = [{"feature": c, "psi": psi(reference[c], current[c])} for c in numeric]
    rows += [{"feature": c, "psi": psi_categorical(reference[c], current[c])} for c in categorical]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["status"] = out["psi"].map(status)
    return out


def calibration_by_band(y_true, proba, bands=(0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)) -> pd.DataFrame:
    """Once labels arrive: predicted vs observed churn per score band."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["band"] = pd.cut(df["p"], bands, include_lowest=True)
    return (df.groupby("band", observed=True)
              .agg(customers=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
              .reset_index())
