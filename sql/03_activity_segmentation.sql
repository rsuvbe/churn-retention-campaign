-- Q3. Rule-based activity segmentation with CASE. Business-friendly segments
-- that a CRM team can use without a model. Thresholds are round numbers close
-- to the quartiles of total_trans_ct (Q1 ~ 45, median ~ 67, Q3 ~ 81).
WITH segmented AS (
    SELECT
        clientnum,
        churned,
        total_trans_amt,
        CASE
            WHEN total_trans_ct < 45 AND months_inactive_12_mon >= 3 THEN '1. Dormant'
            WHEN total_trans_ct < 45                                 THEN '2. Low activity'
            WHEN total_ct_chng_q4_q1 < 0.6                           THEN '3. Declining'
            WHEN total_trans_ct >= 80                                THEN '5. Highly active'
            ELSE                                                          '4. Regular'
        END AS activity_segment
    FROM customers
)
SELECT
    activity_segment,
    COUNT(*)                                          AS customers,
    ROUND(COUNT(*) * 1.0 / SUM(COUNT(*)) OVER (), 4)  AS share_of_base,
    ROUND(AVG(churned * 1.0), 4)                      AS churn_rate,
    SUM(churned)                                      AS churners,
    ROUND(SUM(churned) * 1.0 / SUM(SUM(churned)) OVER (), 4) AS share_of_all_churners,
    ROUND(AVG(total_trans_amt * 1.0), 0)              AS avg_trans_amt
FROM segmented
GROUP BY activity_segment
ORDER BY activity_segment;
