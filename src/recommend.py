"""Next-best-action rules for at-risk customers (cross-sell / reactivation).

Idea: the churn model tells us WHO is at risk; per-customer SHAP values tell
us WHY. SHAP values are summed into four actionable driver families, and each
family maps to an offer:

    usage     -> reactivation offers (the offer depends on WHICH usage signal)
    products  -> cross-sell a second product
    balance   -> credit offers (depends on zero vs maxed-out balance)
    service   -> service-recovery call

Why families and two layers: in this data low usage dominates almost every
at-risk customer's score, so "top single driver -> offer" sends nearly
everyone the same reactivation offer. The PRIMARY offer comes from the
dominant family; the SECONDARY offer (cross-sell, credit, service) comes from
the strongest remaining family when its contribution is material. That second
layer is where personalisation actually happens.

These rules are business hypotheses — validate every offer with an A/B test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Driver family -> features whose SHAP values are summed into it.
FAMILIES: dict[str, list[str]] = {
    "usage": ["total_trans_ct", "total_trans_amt", "avg_trans_amt", "total_ct_chng_q4_q1",
              "total_amt_chng_q4_q1", "months_inactive_12_mon"],
    "products": ["total_relationship_count", "trans_ct_per_product"],
    "balance": ["total_revolving_bal", "is_zero_revolving", "avg_utilization_ratio"],
    "service": ["contacts_count_12_mon"],
}

# Minimum family contribution (log-odds) for a SECONDARY offer to be worth it.
SECONDARY_MIN_SHAP = 0.25

# Revolving balance is capped at $2,517 in this dataset; ~5% of customers sit at
# the cap and churn ~31% (see notebook 02, SHAP dependence plot).
MAXED_BALANCE = 2500

OFFERS = {
    "usage:frequency": ("Reactivation", "Spend challenge: cashback for N transactions in 60 days",
                        "Few or falling transactions: the card is not top-of-wallet"),
    "usage:amount": ("Reactivation", "Category cashback on the customer's top spend category",
                     "Spend pattern (amount / ticket size) signals the card is losing share of wallet"),
    "usage:inactivity": ("Reactivation", "Re-engagement push + small activation bonus",
                         "Inactivity streak: reach out before the account goes dormant"),
    "products": ("Cross-sell", "Second product: savings account / debit card / insurance bundle",
                 "Few products: low switching cost; another product raises stickiness"),
    "balance:zero": ("Cross-sell", "Instalment plan / 0% balance-transfer offer",
                     "Never carries a balance: a low-value 'backup' card"),
    "balance:maxed": ("Credit", "Credit-limit review / debt-restructuring offer",
                      "Balance at the limit: possible financial stress or debt moved elsewhere"),
    "balance:other": ("Credit", "Personalised rate or limit offer",
                      "Revolving behaviour adds to risk"),
    "service": ("Service recovery", "Proactive call from a senior agent to resolve open issues",
                "Many contacts with the bank often mean unresolved problems"),
}
DEFAULT_OFFER = ("Retention", "Personal fee waiver / loyalty offer", "No dominant actionable driver")


def family_shap(shap_values: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    """Sum per-feature SHAP values into driver families (one column per family)."""
    idx = {f: i for i, f in enumerate(feature_names)}
    return pd.DataFrame({fam: shap_values[:, [idx[f] for f in feats if f in idx]].sum(axis=1)
                         for fam, feats in FAMILIES.items()})


def _offer_key(family: str, row: pd.Series, shap_row: pd.Series) -> str:
    """Pick the concrete offer inside a family from the customer's own data."""
    if family == "usage":
        freq = shap_row[["total_trans_ct", "total_ct_chng_q4_q1"]].sum()
        amount = shap_row[["total_trans_amt", "avg_trans_amt", "total_amt_chng_q4_q1"]].sum()
        inactive = shap_row["months_inactive_12_mon"]
        best = max([("usage:frequency", freq), ("usage:amount", amount), ("usage:inactivity", inactive)],
                   key=lambda t: t[1])
        return best[0]
    if family == "balance":
        if row["total_revolving_bal"] == 0:
            return "balance:zero"
        if row["total_revolving_bal"] >= MAXED_BALANCE:
            return "balance:maxed"
        return "balance:other"
    return family


def assign_actions(customers: pd.DataFrame, shap_values: np.ndarray,
                   feature_names: list[str]) -> pd.DataFrame:
    """Attach primary and secondary next-best-actions to each customer.

    `customers` must contain total_revolving_bal (used to choose the balance offer).
    Demographic features are never used for offers: we cannot change them.
    """
    fam = family_shap(shap_values, feature_names)
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    rows = []
    for i in range(len(customers)):
        row, srow, f = customers.iloc[i], shap_df.iloc[i], fam.iloc[i].sort_values(ascending=False)
        positive = f[f > 0]
        if positive.empty:
            primary = DEFAULT_OFFER
            primary_family, secondary = "none", None
        else:
            primary_family = positive.index[0]
            primary = OFFERS[_offer_key(primary_family, row, srow)]
            rest = positive.iloc[1:]
            rest = rest[rest >= SECONDARY_MIN_SHAP]
            secondary = OFFERS[_offer_key(rest.index[0], row, srow)] if len(rest) else None
        top3 = srow.sort_values(ascending=False).head(3)
        rows.append({
            "main_driver_family": primary_family,
            "action_family": primary[0], "offer": primary[1], "rationale": primary[2],
            "secondary_family": secondary[0] if secondary else "none",
            "secondary_offer": secondary[1] if secondary else "",
            "top3_drivers": ", ".join(top3[top3 > 0].index),
        })
    return pd.concat([customers.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def rules_table() -> pd.DataFrame:
    """Human-readable rule table for the README / notebook."""
    rows = []
    for key, (family, offer, why) in OFFERS.items():
        driver_family = key.split(":")[0]
        rows.append({"driver family": driver_family, "condition": key, "action family": family,
                     "offer": offer, "rationale": why,
                     "features": ", ".join(FAMILIES[driver_family])})
    return pd.DataFrame(rows)
