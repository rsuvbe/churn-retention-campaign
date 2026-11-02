"""Build the business presentation (reports/churn_retention_presentation.pptx).

Step 1 (Python): collect every number shown on the slides from reports/tables
         into reports/deck/deck_data.json — slides never contain hand-typed
         results, so they cannot drift from the notebooks.
Step 2 (Node):   reports/deck/build_deck.js renders the deck with pptxgenjs
         (native, editable charts).

Usage:  python -m src.slides
"""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config, path

DECK_DIR = PROJECT_ROOT / "reports" / "deck"


def _t(name: str) -> pd.DataFrame:
    return pd.read_csv(path("tables") / f"{name}.csv")


SHORT_OFFERS = {
    "Instalment": "instalment offer",
    "Second product": "second product",
    "Credit-limit": "credit offer",
    "Personalised rate": "credit offer",
    "Proactive call": "service call",
    "Spend challenge": "spend challenge",
}


def _short_offer(offer: str) -> str:
    return next((v for k, v in SHORT_OFFERS.items() if offer.startswith(k)), offer)


def collect() -> dict:
    cfg = load_config()
    k = int(cfg["model"]["top_k_share"] * 100)

    overview = _t("sql_q1_overview")
    act = _t("sql_q3_activity_segments")
    hyp = _t("hypotheses")
    test = _t("test_results").set_index("model")
    dec = _t("business_decisions")
    sens = _t("business_sensitivity")
    peek = _t("ab_peeking")
    power = _t("ab_power_table")
    nba = _t("nba_summary")
    sec = _t("nba_secondary_summary")
    robust = _t("business_robustness")
    shap_imp = _t("shap_importance")

    n_total = int(overview.customers.sum())
    churn_rate = float(overview.loc[overview.churned == 1, "share"].iloc[0])

    # Gain curve points (model vs rules vs random) from hold-out predictions
    preds = pd.read_csv(path("processed") / "test_with_business.csv")
    y = preds["churned"].to_numpy()
    shares = np.round(np.linspace(0, 1, 21), 2)

    def gains(score):
        order = np.argsort(-score, kind="stable")
        cum = np.concatenate([[0], np.cumsum(y[order])]) / y.sum()
        return [float(cum[int(round(s * len(y)))]) for s in shares]

    rules = ((preds.total_ct_chng_q4_q1 < 0.6).astype(int) + (preds.total_trans_ct < 45).astype(int)
             + (preds.total_revolving_bal == 0).astype(int) + (preds.total_relationship_count <= 2).astype(int)
             + (preds.months_inactive_12_mon >= 3).astype(int)).to_numpy()
    rng = np.random.default_rng(cfg["seed"])
    rules = rules + rng.uniform(0, 1e-3, len(rules))

    # Profit curve (rank by expected profit), sampled every 2.5% of customers
    from src import business as B
    A = B.Assumptions.from_config(cfg)
    curve = B.profit_curve(y, preds["expected_profit"].to_numpy(), preds["clv"].to_numpy(), A)
    pc_shares = np.round(np.arange(0, 1.0001, 0.025), 3)
    pc_profit = [float(curve.loc[int(round(s * len(y))), "profit"]) for s in pc_shares]
    scale = 10_000 / len(y)  # hold-out -> "per 10k customers"

    rule_a = dec.iloc[0]
    everyone = dec[dec.rule == "Contact everyone"].iloc[0]
    thr05 = dec[dec.rule == "P(churn) >= 0.5"].iloc[0]

    lo = sens[sens.multiplier == 0.5].set_index("assumption")
    hi = sens[sens.multiplier == 1.5].set_index("assumption")
    base_profit = float(sens[sens.multiplier == 1.0].optimal_profit.iloc[0])

    model_name = [i for i in test.index if i.startswith("CatBoost +")][0]
    m = test.loc[model_name]
    rules_m = test.loc["SQL rules (5 flags)"]

    return {
        "n_customers": n_total,
        "n_test": int(len(y)),
        "churn_rate": churn_rate,
        "n_churners": int(overview.loc[overview.churned == 1, "customers"].iloc[0]),
        "top_k": k,
        "model": {
            "roc_auc": float(m.roc_auc), "pr_auc": float(m.pr_auc),
            "recall_top_k": float(m[f"recall@top{k}%"]),
            "precision_top_k": float(m[f"precision@top{k}%"]),
            "rules_recall_top_k": float(rules_m[f"recall@top{k}%"]),
        },
        "activity": {
            "labels": [s.split(". ", 1)[1] for s in act.activity_segment],
            "share_base": act.share_of_base.round(3).tolist(),
            "share_churners": act.share_of_all_churners.round(3).tolist(),
            "churn_rate": act.churn_rate.round(3).tolist(),
        },
        "hypotheses": hyp[["hypothesis", "short_name", "slide_stat", "slide_text", "actionable"]].to_dict("records"),
        "gains": {
            "shares": shares.tolist(),
            "model": gains(preds["proba"].to_numpy()),
            "rules": gains(rules),
            "random": shares.tolist(),
        },
        "profit_curve": {"shares": pc_shares.tolist(), "profit_per_10k": [p * scale for p in pc_profit]},
        "decision": {
            "share_contacted": float(rule_a.share_contacted),
            "profit_per_10k": float(rule_a.profit_per_10k_customers),
            "churners_reached_share": float(rule_a.churners_reached / y.sum()),
            "everyone_profit_per_10k": float(everyone.profit_per_10k_customers),
            "thr05_profit_per_10k": float(thr05.profit_per_10k_customers),
        },
        "sensitivity": {
            "labels": lo.index.tolist(),
            "low": [float(v) * scale for v in lo.optimal_profit],
            "high": [float(v) * scale for v in hi.optimal_profit],
            "low_share": lo.optimal_share_contacted.round(3).tolist(),
            "high_share": hi.optimal_share_contacted.round(3).tolist(),
            "base": base_profit * scale,
        },
        "ab": {
            "mde": float(cfg["ab_test"]["mde_relative"]),
            "n_per_arm": int(power.loc[np.isclose(power.mde_relative, cfg["ab_test"]["mde_relative"]), "n_per_arm"].iloc[0]),
            "p_control": float(power.p_control.iloc[0]),
            "fpr_fixed": float(peek.iloc[0]["false positive rate (A/A)"]),
            "fpr_peeking": float(peek.iloc[1]["false positive rate (A/A)"]),
            "fpr_pocock": float(peek.iloc[2]["false positive rate (A/A)"]),
        },
        "nba_primary": nba.groupby("action_family").customers.sum().sort_values(ascending=False).to_dict(),
        "nba_secondary": {k: int(v) for k, v in sec.assign(short=sec.secondary_offer.map(_short_offer))
                          .groupby("short").customers.sum().sort_values(ascending=False).items()},
        "robust": {
            "band_low": float(robust.flat_band_low_share.iloc[0]),
            "band_high": float(robust.flat_band_high_share.iloc[0]),
            "break_even_success": float(robust.break_even_success_rate_base_plan.iloc[0]),
            "assumed_success": float(robust.assumed_success_rate.iloc[0]),
        },
        "shap_top": shap_imp.head(6).to_dict("records"),
    }


def main() -> None:
    data = collect()
    out = DECK_DIR / "deck_data.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    subprocess.run(["node", "build_deck.js"], cwd=DECK_DIR, check=True)


if __name__ == "__main__":
    main()
