-- =============================================================================
-- Optional: PostgreSQL schema. The project runs on DuckDB by default (no server
-- needed), but every query in this folder is written in PostgreSQL-compatible
-- SQL. To run on Postgres:
--   1) psql -d <db> -f sql/00_schema_postgres.sql
--   2) DATABASE_URL=postgresql://user@host/db python -m src.load_postgres
-- =============================================================================
DROP TABLE IF EXISTS customers;

CREATE TABLE customers (
    clientnum                 BIGINT PRIMARY KEY,
    customer_age              INTEGER,
    gender                    TEXT,
    dependent_count           INTEGER,
    education_level           TEXT,
    marital_status            TEXT,
    income_category           TEXT,
    card_category             TEXT,
    months_on_book            INTEGER,
    total_relationship_count  INTEGER,
    months_inactive_12_mon    INTEGER,
    contacts_count_12_mon     INTEGER,
    credit_limit              NUMERIC,
    total_revolving_bal       INTEGER,
    avg_open_to_buy           NUMERIC,
    total_amt_chng_q4_q1      NUMERIC,
    total_trans_amt           INTEGER,
    total_trans_ct            INTEGER,
    total_ct_chng_q4_q1       NUMERIC,
    avg_utilization_ratio     NUMERIC,
    churned                   INTEGER NOT NULL CHECK (churned IN (0, 1))
);
