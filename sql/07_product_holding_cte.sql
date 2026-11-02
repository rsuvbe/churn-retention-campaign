-- Q7. Product holding vs churn (cross-sell angle), with chained CTEs.
-- Customers with 1-2 products are the cross-sell pool; the query sizes it and
-- compares churn with multi-product customers.
WITH holding AS (
    SELECT
        clientnum,
        churned,
        total_relationship_count,
        total_trans_amt,
        CASE
            WHEN total_relationship_count <= 2 THEN '1-2 products'
            WHEN total_relationship_count <= 4 THEN '3-4 products'
            ELSE '5-6 products'
        END AS holding_band
    FROM customers
),
band_stats AS (
    SELECT
        holding_band,
        COUNT(*)                              AS customers,
        ROUND(AVG(churned * 1.0), 4)          AS churn_rate,
        ROUND(AVG(total_trans_amt * 1.0), 0)  AS avg_trans_amt
    FROM holding
    GROUP BY holding_band
),
base AS (
    SELECT AVG(churned * 1.0) AS overall_churn FROM customers
)
SELECT
    b.holding_band,
    b.customers,
    b.churn_rate,
    ROUND(b.churn_rate / base.overall_churn, 2) AS churn_lift_vs_avg,
    b.avg_trans_amt
FROM band_stats b
CROSS JOIN base
ORDER BY b.holding_band;
