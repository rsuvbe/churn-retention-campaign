-- Q2. Churn rate by demographic / product segment in one pass using
-- GROUPING SETS. Answers "which segments leak the most customers?"
-- Segments with < 100 customers are flagged as statistically unreliable.
SELECT
    CASE
        WHEN GROUPING(gender) = 0          THEN 'gender'
        WHEN GROUPING(income_category) = 0 THEN 'income_category'
        WHEN GROUPING(card_category) = 0   THEN 'card_category'
        WHEN GROUPING(education_level) = 0 THEN 'education_level'
        WHEN GROUPING(marital_status) = 0  THEN 'marital_status'
    END                                                   AS dimension,
    COALESCE(gender, income_category, card_category,
             education_level, marital_status)             AS segment,
    COUNT(*)                                              AS customers,
    SUM(churned)                                          AS churners,
    ROUND(AVG(churned * 1.0), 4)                          AS churn_rate,
    COUNT(*) < 100                                        AS small_sample
FROM customers
GROUP BY GROUPING SETS (
    (gender), (income_category), (card_category),
    (education_level), (marital_status)
)
ORDER BY dimension, churn_rate DESC, segment;  -- segment breaks ties deterministically
