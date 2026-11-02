"""Business layer: customer value, expected profit, profit curve, sensitivity.

Everything monetary is driven by assumptions in config.yaml -> `business`.
The dataset has NO revenue or cost data, so all money figures are
illustrative and must be re-estimated with the bank's real unit economics.

Economics of contacting one customer i:
    benefit_i = P(churn_i) * retention_success_rate * CLV_i
    cost_i    = contact_cost + incentive_accept_rate * incentive_cost
    E[profit_i] = benefit_i - cost_i
Contact customer i only if E[profit_i] > 0, or pick the top-k% that maximises
total profit on the profit curve.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Assumptions:
    annual_interchange_rate: float
    revolving_margin_rate: float
    clv_horizon_years: float
    clv_floor: float
    contact_cost: float
    incentive_cost: float
    incentive_accept_rate: float
    retention_success_rate: float

    @classmethod
    def from_config(cls, cfg: dict) -> "Assumptions":
        b = cfg["business"]
        return cls(**{f: b[f] for f in cls.__dataclass_fields__})

    @property
    def cost_per_contact(self) -> float:
        return self.contact_cost + self.incentive_accept_rate * self.incentive_cost


def customer_value(df: pd.DataFrame, a: Assumptions) -> pd.Series:
    """Value at risk if the customer leaves (simple CLV proxy, USD).

    Interchange income on yearly spend + interest margin on revolving balance,
    over a fixed horizon, with a floor. Deliberately simple and transparent.
    """
    yearly = (a.annual_interchange_rate * df["total_trans_amt"]
              + a.revolving_margin_rate * df["total_revolving_bal"])
    return (yearly * a.clv_horizon_years).clip(lower=a.clv_floor).rename("clv")


def expected_profit(proba: np.ndarray, clv: np.ndarray, a: Assumptions) -> np.ndarray:
    """Expected profit of contacting each customer (ex-ante, uses probabilities)."""
    return proba * a.retention_success_rate * clv - a.cost_per_contact


def profit_curve(y_true: np.ndarray, rank_score: np.ndarray, clv: np.ndarray,
                 a: Assumptions) -> pd.DataFrame:
    """Realised campaign profit when contacting the top-n customers by `rank_score`.

    Uses the TRUE labels of the hold-out set: a contacted churner is saved
    with probability `retention_success_rate` (so we count the expected value
    of that), a contacted non-churner only costs money.
    Returns one row per n = 0..N.
    """
    order = np.argsort(-rank_score, kind="stable")
    y_sorted = np.asarray(y_true)[order]
    clv_sorted = np.asarray(clv)[order]
    gain = np.cumsum(y_sorted * a.retention_success_rate * clv_sorted)
    cost = np.arange(1, len(order) + 1) * a.cost_per_contact
    profit = np.concatenate([[0.0], gain - cost])
    n = np.arange(len(order) + 1)
    return pd.DataFrame({
        "n_contacted": n,
        "share_contacted": n / len(order),
        "profit": profit,
        "churners_reached": np.concatenate([[0], np.cumsum(y_sorted)]),
    })


def best_point(curve: pd.DataFrame) -> pd.Series:
    """Row of the profit curve with maximal profit."""
    return curve.loc[curve["profit"].idxmax()]


def profit_at_threshold(y_true, proba, clv, threshold: float, a: Assumptions) -> dict:
    """Realised profit when contacting everyone with proba >= threshold."""
    mask = np.asarray(proba) >= threshold
    y, v = np.asarray(y_true)[mask], np.asarray(clv)[mask]
    profit = (y * a.retention_success_rate * v).sum() - mask.sum() * a.cost_per_contact
    return {"threshold": threshold, "n_contacted": int(mask.sum()),
            "share_contacted": float(mask.mean()), "profit": float(profit)}


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

# Which assumptions to stress-test, and how a multiplier is applied.
SENSITIVITY_PARAMS = {
    "retention_success_rate": "retention_success_rate",
    "clv (value of a customer)": "clv_scale",
    "contact_cost": "contact_cost",
    "incentive_cost": "incentive_cost",
}


def sensitivity_one_way(df_test: pd.DataFrame, y_true, proba, base: Assumptions,
                        multipliers: list[float], rank_by: str = "expected_value") -> pd.DataFrame:
    """Vary one assumption at a time by each multiplier, keep the others fixed.

    For each scenario we re-optimise the campaign size (the bank would) and
    also record the profit of the ORIGINAL plan (decision chosen under base
    assumptions) — the gap shows the cost of being wrong about assumptions.
    """
    base_clv = customer_value(df_test, base).to_numpy()
    base_rank = _rank_score(proba, base_clv, base, rank_by)
    base_curve = profit_curve(y_true, base_rank, base_clv, base)
    base_n = int(best_point(base_curve)["n_contacted"])

    rows = []
    for label, field in SENSITIVITY_PARAMS.items():
        for m in multipliers:
            if field == "clv_scale":
                a, clv = base, base_clv * m
            else:
                a = replace(base, **{field: getattr(base, field) * m})
                clv = base_clv
            rank = _rank_score(proba, clv, a, rank_by)
            curve = profit_curve(y_true, rank, clv, a)
            best = best_point(curve)
            rows.append({
                "assumption": label,
                "multiplier": m,
                "optimal_share_contacted": best["share_contacted"],
                "optimal_profit": best["profit"],
                "profit_of_base_plan": curve.loc[base_n, "profit"],
            })
    return pd.DataFrame(rows)


def sensitivity_grid(df_test: pd.DataFrame, y_true, proba, base: Assumptions,
                     success_rates: list[float], clv_scales: list[float]) -> pd.DataFrame:
    """Two-way grid: the two most uncertain assumptions together."""
    base_clv = customer_value(df_test, base).to_numpy()
    rows = []
    for s in success_rates:
        for c in clv_scales:
            a = replace(base, retention_success_rate=s)
            clv = base_clv * c
            curve = profit_curve(y_true, _rank_score(proba, clv, a, "expected_value"), clv, a)
            best = best_point(curve)
            rows.append({"retention_success_rate": s, "clv_scale": c,
                         "optimal_profit": best["profit"],
                         "optimal_share_contacted": best["share_contacted"]})
    return pd.DataFrame(rows)


def _rank_score(proba, clv, a: Assumptions, rank_by: str) -> np.ndarray:
    if rank_by == "probability":
        return np.asarray(proba)
    return expected_profit(np.asarray(proba), np.asarray(clv), a)
