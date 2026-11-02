"""Data loading, cleaning and the DuckDB analytical layer.

The raw Kaggle file has CamelCase column names and two columns produced by
someone else's Naive Bayes model that was trained ON THE TARGET. Those two
columns are pure target leakage and are dropped here, at the very first step,
so that no downstream code can ever see them.
"""
from __future__ import annotations

import duckdb
import pandas as pd

from src.config import path

TARGET = "churned"
ID_COL = "clientnum"

# Prefix of the two leaky columns shipped with the Kaggle dataset.
LEAKY_PREFIX = "Naive_Bayes_Classifier"

CATEGORICAL_COLS = [
    "gender",
    "education_level",
    "marital_status",
    "income_category",
    "card_category",
]

NUMERIC_COLS = [
    "customer_age",
    "dependent_count",
    "months_on_book",
    "total_relationship_count",
    "months_inactive_12_mon",
    "contacts_count_12_mon",
    "credit_limit",
    "total_revolving_bal",
    "avg_open_to_buy",
    "total_amt_chng_q4_q1",
    "total_trans_amt",
    "total_trans_ct",
    "total_ct_chng_q4_q1",
    "avg_utilization_ratio",
]


def load_raw(keep_leaky: bool = False) -> pd.DataFrame:
    """Read the raw CSV. `keep_leaky=True` is only used by the leakage demo."""
    df = pd.read_csv(path("raw_data"))
    if not keep_leaky:
        df = df.loc[:, ~df.columns.str.startswith(LEAKY_PREFIX)]
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names and encode the target as 0/1.

    'Unknown' in categorical columns is kept as its own category: in banking
    data a missing education/income field is itself informative (e.g. the
    customer skipped the form) and imputing it would hide that signal.
    """
    out = df.copy()
    out.columns = [c.lower() for c in out.columns]
    out[TARGET] = (out.pop("attrition_flag") == "Attrited Customer").astype(int)
    if out[ID_COL].duplicated().any():
        raise ValueError("Duplicate customer IDs found — check the raw file.")
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-wise engineered features.

    Every feature uses only the customer's own row, so there is no way for
    information to leak across the train/test split.
    """
    out = df.copy()
    out["avg_trans_amt"] = out["total_trans_amt"] / out["total_trans_ct"].clip(lower=1)
    out["is_zero_revolving"] = (out["total_revolving_bal"] == 0).astype(int)
    out["trans_ct_per_product"] = out["total_trans_ct"] / out["total_relationship_count"]
    return out


ENGINEERED_COLS = ["avg_trans_amt", "is_zero_revolving", "trans_ct_per_product"]
FEATURES = CATEGORICAL_COLS + NUMERIC_COLS + ENGINEERED_COLS


def prepare_for_scoring(df: pd.DataFrame) -> pd.DataFrame:
    """Raw or already-cleaned customer rows -> model-ready frame, no target needed.

    Accepts the original Kaggle column names (CamelCase) or the cleaned
    snake_case names; leaky columns are dropped if present. Used by batch
    scoring, where new customers obviously have no Attrition_Flag yet.
    """
    out = df.loc[:, ~df.columns.str.startswith(LEAKY_PREFIX)].copy()
    out.columns = [c.lower() for c in out.columns]
    if "attrition_flag" in out:
        out[TARGET] = (out.pop("attrition_flag") == "Attrited Customer").astype(int)
    missing = [c for c in CATEGORICAL_COLS + NUMERIC_COLS if c not in out]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    return add_features(out)


def load_dataset() -> pd.DataFrame:
    """Raw -> cleaned -> engineered. The single entry point for modelling."""
    return add_features(clean(load_raw()))


# ---------------------------------------------------------------------------
# SQL layer
# ---------------------------------------------------------------------------

def build_duckdb() -> str:
    """Create (or replace) the `customers` table in a local DuckDB file.

    Column names are lower-case snake_case, so the SQL in sql/ runs unchanged
    on PostgreSQL too (see sql/00_schema_postgres.sql).
    """
    df = clean(load_raw())
    db_file = str(path("duckdb"))
    with duckdb.connect(db_file) as con:
        con.register("df_view", df)
        con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM df_view")
    return db_file


def run_sql_file(name: str) -> pd.DataFrame:
    """Execute sql/<name>.sql against DuckDB and return a DataFrame."""
    from src.config import PROJECT_ROOT

    sql_path = PROJECT_ROOT / "sql" / (name if name.endswith(".sql") else f"{name}.sql")
    query = sql_path.read_text(encoding="utf-8")
    with duckdb.connect(str(path("duckdb")), read_only=True) as con:
        return con.execute(query).df()
