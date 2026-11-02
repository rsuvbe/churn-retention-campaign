"""Optional: load the cleaned data into PostgreSQL instead of DuckDB.

Usage:
    psql -d churn -f sql/00_schema_postgres.sql
    DATABASE_URL=postgresql://user@localhost/churn python -m src.load_postgres

Requires `sqlalchemy` and `psycopg[binary]` (not in the default requirements,
because the default pipeline uses DuckDB and needs no database server).
"""
from __future__ import annotations

import os

from src.data import clean, load_raw


def main() -> None:
    from sqlalchemy import create_engine

    url = os.environ.get("DATABASE_URL")
    if not url:
        raise SystemExit("Set DATABASE_URL, e.g. postgresql://user@localhost/churn")
    df = clean(load_raw())
    engine = create_engine(url)
    # Table is created by 00_schema_postgres.sql; append keeps its types/constraints.
    df.to_sql("customers", engine, if_exists="append", index=False, method="multi", chunksize=2000)
    print(f"Loaded {len(df):,} rows into customers")


if __name__ == "__main__":
    main()
