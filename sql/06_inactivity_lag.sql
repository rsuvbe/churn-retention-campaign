-- Q6. LAG(): how churn changes with each extra inactive month.
-- NOTE: the dataset is a single snapshot (no monthly panel), so LAG is applied
-- across ordered levels of months_inactive_12_mon, not across calendar time.
-- The delta shows where the "cliff" in churn risk is -> trigger point for a
-- reactivation campaign.
WITH by_inactivity AS (
    SELECT
        months_inactive_12_mon       AS inactive_months,
        COUNT(*)                     AS customers,
        ROUND(AVG(churned * 1.0), 4) AS churn_rate
    FROM customers
    GROUP BY months_inactive_12_mon
)
SELECT
    inactive_months,
    customers,
    churn_rate,
    LAG(churn_rate) OVER (ORDER BY inactive_months)              AS prev_churn_rate,
    churn_rate - LAG(churn_rate) OVER (ORDER BY inactive_months) AS delta_vs_prev,
    SUM(customers) OVER (ORDER BY inactive_months)               AS cumulative_customers
FROM by_inactivity
ORDER BY inactive_months;
