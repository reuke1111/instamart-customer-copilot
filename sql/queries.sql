-- Q1: WHERE filters individual completed orders before HAVING filters totals.
SELECT customer_id, ROUND(SUM(order_amount), 2) AS qualifying_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY qualifying_spend DESC, customer_id;

-- Q2: Completed revenue across the three-table join.
SELECT c.city, p.category, ROUND(SUM(o.order_amount), 2) AS revenue
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC, c.city, p.category;

-- Q3: Scalar subquery over the whole product catalogue.
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4: Keep the completed filter in ON so unmatched customers survive.
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5: Average only over customers with at least one completed order.
WITH customer_spend AS (
    SELECT customer_id, ROUND(SUM(order_amount), 2) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS average_spend FROM customer_spend
)
SELECT c.customer_id, c.total_spend, a.average_spend
FROM customer_spend AS c
CROSS JOIN average_spend AS a
WHERE c.total_spend > a.average_spend
ORDER BY c.total_spend DESC, c.customer_id;

-- Q6: Round monetary totals to cents before ranking, preserving real ties.
WITH customer_spend AS (
    SELECT c.city, c.customer_id, ROUND(SUM(o.order_amount), 2) AS total_spend
    FROM customers AS c
    JOIN orders AS o ON o.customer_id = c.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.city, c.customer_id
)
SELECT city, customer_id, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC) AS row_number
FROM customer_spend
ORDER BY city, row_number;

-- Q7: One daily row, with an explicit cumulative window.
WITH daily_revenue AS (
    SELECT SUBSTR(order_ts, 1, 10) AS day,
           ROUND(SUM(order_amount), 2) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY SUBSTR(order_ts, 1, 10)
)
SELECT day, revenue,
       ROUND(SUM(revenue) OVER (
           ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
       ), 2) AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
