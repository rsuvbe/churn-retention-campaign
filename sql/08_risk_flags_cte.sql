-- Q8. A transparent rule-based risk score: count of simple warning flags.
-- This is the "no-ML" benchmark the model has to beat. Each flag is a
-- business-readable rule taken from the EDA.
WITH flags AS (
    SELECT
        clientnum,
        churned,
        (CASE WHEN total_ct_chng_q4_q1 < 0.6      THEN 1 ELSE 0 END) AS f_declining_tx,
        (CASE WHEN total_trans_ct < 45            THEN 1 ELSE 0 END) AS f_low_tx,
        (CASE WHEN total_revolving_bal = 0        THEN 1 ELSE 0 END) AS f_zero_revolving,
        (CASE WHEN total_relationship_count <= 2  THEN 1 ELSE 0 END) AS f_few_products,
        (CASE WHEN months_inactive_12_mon >= 3    THEN 1 ELSE 0 END) AS f_inactive
    FROM customers
),
scored AS (
    SELECT
        *,
        f_declining_tx + f_low_tx + f_zero_revolving + f_few_products + f_inactive AS risk_flags
    FROM flags
)
SELECT
    risk_flags,
    COUNT(*)                                          AS customers,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4)  AS share_of_base,
    ROUND(AVG(churned * 1.0), 4)                      AS churn_rate,
    -- cumulative recall if we target everyone with >= this many flags
    ROUND(SUM(SUM(churned)) OVER (ORDER BY risk_flags DESC) * 1.0
          / SUM(SUM(churned)) OVER (), 4)             AS cum_recall_if_targeted,
    ROUND(SUM(COUNT(*)) OVER (ORDER BY risk_flags DESC) * 1.0
          / SUM(COUNT(*)) OVER (), 4)                 AS cum_share_targeted
FROM scored
GROUP BY risk_flags
ORDER BY risk_flags DESC;
