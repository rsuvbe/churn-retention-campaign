"""Streamlit dashboard: score a customer, explain the score, recommend an action.

Run from the project root:
    streamlit run app/streamlit_app.py

Requires the artefacts produced by notebooks/02_modeling.ipynb (models/).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier, Pool

from src import business as B
from src.config import load_config, path
from src.data import CATEGORICAL_COLS, FEATURES, load_dataset
from src.recommend import assign_actions

st.set_page_config(page_title="Churn risk scoring", layout="wide")


@st.cache_resource
def load_artifacts():
    model = CatBoostClassifier()
    model.load_model(str(path("models") / "catboost_churn.cbm"))
    calibrator = joblib.load(path("models") / "calibrator.joblib")
    meta = joblib.load(path("models") / "model_meta.joblib")
    return model, calibrator, meta


@st.cache_data
def load_customers() -> pd.DataFrame:
    return load_dataset()


@st.cache_data
def portfolio_scores(_model, _calibrator, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Score the whole base once per session (inputs are fixed, so cache it)."""
    return score(load_customers(), _model, _calibrator)


def money(x: float) -> str:
    return f"{'−' if x < 0 else ''}${abs(x):,.0f}"


def score(df: pd.DataFrame, model, calibrator) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated churn probability and SHAP values for each row."""
    pool = Pool(df[FEATURES], cat_features=CATEGORICAL_COLS)
    proba = calibrator.predict(model.predict_proba(pool)[:, 1])
    shap_vals = model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    return proba, shap_vals


model, calibrator, meta = load_artifacts()
customers = load_customers()
cfg = load_config()
base = B.Assumptions.from_config(cfg)

st.title("Customer churn risk & retention action")
st.caption(f"CatBoost + {meta['calibration']} calibration · money figures are based on assumptions in "
           "config/config.yaml, not on real P&L data")

# --- Sidebar: business assumptions are editable -----------------------------
st.sidebar.header("Business assumptions")
success = st.sidebar.slider("Retention success rate", 0.05, 0.60, float(base.retention_success_rate), 0.05)
contact = st.sidebar.number_input("Contact cost, $", 0.0, 200.0, float(base.contact_cost), 5.0)
incentive = st.sidebar.number_input("Incentive cost, $", 0.0, 300.0, float(base.incentive_cost), 5.0)
clv_scale = st.sidebar.slider("CLV multiplier", 0.5, 2.0, 1.0, 0.25)
A = B.Assumptions(**{**base.__dict__, "retention_success_rate": success,
                     "contact_cost": contact, "incentive_cost": incentive})

tab_one, tab_portfolio = st.tabs(["Single customer", "Portfolio"])

# --- Single customer --------------------------------------------------------
with tab_one:
    p_base, _ = portfolio_scores(model, calibrator, len(customers))
    order = np.argsort(-p_base)                      # riskiest customers first
    ids = customers["clientnum"].to_numpy()[order].tolist()
    risk_of = dict(zip(customers["clientnum"], p_base))
    left, right = st.columns([1, 2])
    with left:
        cid = st.selectbox("Customer ID (sorted by churn risk)", ids, index=0,
                           format_func=lambda c: f"{c}  ·  risk {risk_of[c]:.0%}")
        row = customers.loc[customers.clientnum == cid].copy()
        st.markdown("**Edit behaviour to run a what-if:**")
        row["total_trans_ct"] = st.number_input("Transactions (12m)", 0, 200, int(row.total_trans_ct.iloc[0]))
        row["total_trans_amt"] = st.number_input("Spend (12m), $", 0, 30000, int(row.total_trans_amt.iloc[0]))
        row["total_relationship_count"] = st.number_input("Products held", 1, 6, int(row.total_relationship_count.iloc[0]))
        row["total_revolving_bal"] = st.number_input("Revolving balance, $", 0, 3000, int(row.total_revolving_bal.iloc[0]))
        row["months_inactive_12_mon"] = st.number_input("Inactive months (12m)", 0, 12, int(row.months_inactive_12_mon.iloc[0]))
        # keep engineered features consistent with edited inputs
        row["avg_trans_amt"] = row.total_trans_amt / row.total_trans_ct.clip(lower=1)
        row["is_zero_revolving"] = (row.total_revolving_bal == 0).astype(int)
        row["trans_ct_per_product"] = row.total_trans_ct / row.total_relationship_count

    proba, shap_vals = score(row, model, calibrator)
    clv = float(B.customer_value(row, A).iloc[0]) * clv_scale
    ev = float(B.expected_profit(proba, np.array([clv]), A)[0])
    action = assign_actions(row[["clientnum", "total_revolving_bal"]], shap_vals, FEATURES).iloc[0]

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn probability", f"{proba[0]:.0%}")
        c2.metric("Customer value (CLV proxy)", money(clv))
        c3.metric("Expected profit of contact", money(ev),
                  delta="contact" if ev > 0 else "do not contact",
                  delta_color="normal" if ev > 0 else "inverse")
        if ev > 0:
            st.markdown(f"**Recommended action:** {action.action_family} — {action.offer}")
            if action.secondary_offer:
                st.markdown(f"**Secondary offer:** {action.secondary_family} — {action.secondary_offer}")
            st.caption(f"Why: {action.rationale}. Top drivers: {action.top3_drivers or '—'}")
        else:
            st.markdown("**Recommended action:** none — the expected saving does not cover "
                        f"the ${A.cost_per_contact:.0f} cost of a contact.")

        contrib = pd.Series(shap_vals[0], index=FEATURES)
        top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(10)[::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(top.index, top.values, color=["#C8102E" if v > 0 else "#2E8B57" for v in top.values])
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("SHAP contribution to churn log-odds (red = raises risk)")
        ax.set_title("Why this score?")
        st.pyplot(fig)
        if "churned" in row:
            st.caption(f"Actual outcome in the dataset: {'churned' if row.churned.iloc[0] else 'stayed'} "
                       "(note: this customer may have been in the training set)")

# --- Portfolio --------------------------------------------------------------
with tab_portfolio:
    st.markdown("Scores **all 10,127 customers** (incl. training rows, so this is a "
                "deployment view, not an evaluation).")
    p_all, _ = portfolio_scores(model, calibrator, len(customers))
    clv_all = B.customer_value(customers, A).to_numpy() * clv_scale
    ev_all = B.expected_profit(p_all, clv_all, A)
    target = ev_all > 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers to contact", f"{target.sum():,} ({target.mean():.1%})")
    c2.metric("Expected campaign profit", money(ev_all[target].sum()))
    c3.metric("Avg churn prob. in campaign", f"{p_all[target].mean():.0%}" if target.any() else "—")
    out = customers.loc[target, ["clientnum", "total_trans_ct", "total_relationship_count",
                                 "total_revolving_bal"]].assign(
        churn_probability=p_all[target].round(3), expected_profit=ev_all[target].round(1))
    st.dataframe(out.sort_values("expected_profit", ascending=False), width="stretch")
    st.download_button("Download campaign list (CSV)", out.to_csv(index=False), "campaign_list.csv")
