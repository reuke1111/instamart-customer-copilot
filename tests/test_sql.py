"""Verify SQL semantics with independent, deliberately small examples."""

import sqlite3
import unittest

import pandas as pd

from scripts.sql_analysis import load_queries


class SQLSemanticsTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)
        self.queries = load_queries()
        pd.DataFrame({
            "customer_id": ["A", "B", "C", "D", "E"],
            "city": ["Pune", "Pune", "Pune", "Mumbai", "Pune"],
        }).to_sql("customers", self.connection, index=False)
        pd.DataFrame({
            "product_id": ["P1", "P2", "P3"],
            "product_name": ["Low", "Mid", "High"],
            "category": ["Grocery", "Grocery", "Electronics"],
            "price_inr": [10, 20, 30],
        }).to_sql("products", self.connection, index=False)

    def orders(self, rows):
        pd.DataFrame(rows, columns=[
            "order_id", "customer_id", "product_id", "status", "order_amount", "order_ts",
        ]).to_sql("orders", self.connection, index=False)

    def query(self, name):
        return pd.read_sql_query(self.queries[name], self.connection)

    def test_where_precedes_having_and_boundaries_are_strict(self):
        self.orders([
            ["1", "A", "P1", "completed", 8000, "2025-04-01 10:00:00"],
            ["2", "A", "P1", "completed", 500, "2025-04-01 10:00:00"],
            ["3", "B", "P1", "completed", 8000.01, "2025-04-01 10:00:00"],
            ["4", "C", "P1", "cancelled", 10000, "2025-04-01 10:00:00"],
        ])
        self.assertEqual(self.query("Q1")["customer_id"].tolist(), ["B"])

    def test_left_join_includes_cancelled_only_and_no_orders(self):
        self.orders([
            ["1", "A", "P1", "completed", 100, "2025-04-01 10:00:00"],
            ["2", "B", "P1", "cancelled", 100, "2025-04-01 10:00:00"],
        ])
        self.assertEqual(self.query("Q4")["customer_id"].tolist(), ["B", "C", "D", "E"])

    def test_average_excludes_nonbuyers_and_catalogue_subquery_is_strict(self):
        self.orders([
            ["1", "A", "P1", "completed", 100, "2025-04-01 10:00:00"],
            ["2", "B", "P1", "completed", 300, "2025-04-01 10:00:00"],
        ])
        result = self.query("Q5")
        self.assertEqual(result["customer_id"].tolist(), ["B"])
        self.assertEqual(result["average_spend"].tolist(), [200])
        self.assertEqual(self.query("Q3")["product_id"].tolist(), ["P3"])

    def test_ranking_ties_and_city_partitions(self):
        self.orders([
            ["1", "A", "P1", "completed", 100, "2025-04-01 10:00:00"],
            ["2", "B", "P1", "completed", 100, "2025-04-01 10:00:00"],
            ["3", "C", "P1", "completed", 50, "2025-04-01 10:00:00"],
            ["4", "D", "P1", "completed", 10, "2025-04-01 10:00:00"],
        ])
        result = self.query("Q6")
        pune = result.loc[result["city"].eq("Pune")]
        self.assertEqual(pune["spend_rank"].tolist(), [1, 1, 3])
        self.assertEqual(pune["dense_rank"].tolist(), [1, 1, 2])
        self.assertEqual(pune["row_number"].tolist(), [1, 2, 3])
        self.assertEqual(result.loc[result["city"].eq("Mumbai"), "spend_rank"].tolist(), [1])

    def test_join_and_daily_running_total_exclude_cancelled(self):
        self.orders([
            ["1", "A", "P1", "completed", 20, "2025-04-02 10:00:00"],
            ["2", "B", "P3", "completed", 10, "2025-04-01 11:00:00"],
            ["3", "A", "P1", "completed", 5, "2025-04-01 10:00:00"],
            ["4", "D", "P1", "cancelled", 999, "2025-04-01 10:00:00"],
        ])
        q2 = self.query("Q2")
        self.assertEqual(dict(zip(q2["category"], q2["revenue"])), {"Grocery": 25, "Electronics": 10})
        daily = self.query("Q7")
        self.assertEqual(daily["day"].tolist(), ["2025-04-01", "2025-04-02"])
        self.assertEqual(daily["revenue"].tolist(), [15, 20])
        self.assertEqual(daily["cumulative_revenue"].tolist(), [15, 35])


if __name__ == "__main__":
    unittest.main()
