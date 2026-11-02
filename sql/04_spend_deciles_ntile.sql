-- Q4. NTILE(10) on yearly spend: churn rate per spend decile.
-- Decile 1 = lowest spenders. Shows whether churn concentrates in low spend
-- and how much spend (a revenue proxy) sits in each decile.
WITH ranked AS (
    SELECT
        churned,
        total_trans_amt,
        NTILE(10) OVER (ORDER BY total_trans_amt, clientnum) AS spend_decile  -- id breaks ties
    FROM customers
)
SELECT
    spend_decile,
    MIN(total_trans_amt)                           AS min_spend,
    MAX(total_trans_amt)                           AS max_spend,
    COUNT(*)                                       AS customers,
    ROUND(AVG(churned * 1.0), 4)                   AS churn_rate,
    SUM(CASE WHEN churned = 1 THEN total_trans_amt ELSE 0 END) AS spend_lost_to_churn
FROM ranked
GROUP BY spend_decile
ORDER BY spend_decile;
