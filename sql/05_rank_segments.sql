-- Q5. RANK() and DENSE_RANK(): rank income x card micro-segments by churn
-- rate, but only those with enough customers (>= 50) to be trusted.
-- The global rank feeds the "where to focus first" slide; the within-card
-- rank shows the riskiest income band inside each card tier.
WITH seg AS (
    SELECT
        card_category,
        income_category,
        COUNT(*)                     AS customers,
        ROUND(AVG(churned * 1.0), 4) AS churn_rate
    FROM customers
    GROUP BY card_category, income_category
    HAVING COUNT(*) >= 50
)
SELECT
    card_category,
    income_category,
    customers,
    churn_rate,
    RANK()       OVER (ORDER BY churn_rate DESC)                            AS global_risk_rank,  -- ties share a rank
    DENSE_RANK() OVER (PARTITION BY card_category ORDER BY churn_rate DESC) AS rank_within_card
FROM seg
ORDER BY global_risk_rank, card_category, income_category;
