-- Q1. Portfolio overview: size, churn rate and how churners differ on the
-- key activity metrics. Used as the headline numbers in EDA and the README.
SELECT
    churned,
    COUNT(*)                                             AS customers,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4)     AS share,
    ROUND(AVG(total_trans_ct * 1.0), 1)                  AS avg_trans_ct,
    ROUND(AVG(total_trans_amt * 1.0), 0)                 AS avg_trans_amt,
    ROUND(AVG(total_revolving_bal * 1.0), 0)             AS avg_revolving_bal,
    ROUND(AVG(total_relationship_count * 1.0), 2)        AS avg_products,
    ROUND(AVG(months_inactive_12_mon * 1.0), 2)          AS avg_inactive_months
FROM customers
GROUP BY churned
ORDER BY churned;
