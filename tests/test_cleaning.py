"""Small independent fixtures exercise cleaning rules, rather than mirroring code."""

from contextlib import closing
import sqlite3
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from scripts.clean_data import clean_customers, clean_orders, persist
from scripts.common import ROOT


def customer_fixture():
    return pd.DataFrame({
        "customer_id": ["C1", "C2", "C3", "C4", "C5"],
        "city": [" pune ", "PUNE", "pune", "mumbai", " MUMBAI "],
        "signup_date": ["2024-01-01"] * 5,
        "age": [20, 40, -4, 60, np.nan],
        "gender": ["M", " Female ", np.nan, "f", "unknown-spelling"],
    })


def product_fixture():
    return pd.DataFrame({
        "product_id": ["P1", "P2"],
        "product_name": ["First", "Second"],
        "category": ["A", "B"],
        "price_inr": [100.00, 19.99],
    })


def order_fixture():
    columns = [
        "order_id", "customer_id", "product_id", "quantity", "order_ts",
        "status", "discount_pct", "order_amount", "delivery_minutes",
    ]
    def row(oid, product="P1", amount="100", duration=10, status="completed",
            quantity=1, discount=0, timestamp="2025-04-01 11:00:00"):
        return [oid, "C1", product, quantity, timestamp, status, discount, amount, duration]
    latest = row("O1", amount="error", quantity=2, discount=10)
    return pd.DataFrame([
        latest,
        row("O1", status="pending", quantity=9, timestamp="2025-04-01 10:00:00", duration=np.nan),
        list(latest),  # Exact duplicate of the final snapshot.
        row("O2", duration=30),
        row("O3", amount="N/A", duration=-10),
        row("O4", duration=np.nan),
        row("O5", product="P2", duration=80),
        row("O6", product="P2", amount="error", quantity=3, discount=15, duration=np.nan),
        row("O7", duration=999, status="cancelled"),
    ], columns=columns)


class CustomerCleaningTests(unittest.TestCase):
    def test_city_specific_imputation_mapping_and_input_preservation(self):
        raw = customer_fixture()
        original = raw.copy(deep=True)
        clean, log = clean_customers(raw)
        self.assertEqual(clean["age"].tolist(), [20, 40, 30, 60, 60])
        self.assertEqual(clean["city"].tolist(), ["Pune"] * 3 + ["Mumbai"] * 2)
        self.assertEqual(clean["gender"].tolist(), ["male", "female", "unknown", "female", "unknown"])
        self.assertEqual(clean["customer_id"].tolist(), raw["customer_id"].tolist())
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(clean["signup_date"]))
        self.assertEqual(log[2]["changed"], 1)
        self.assertEqual(log[3]["changed"], 2)
        pd.testing.assert_frame_equal(raw, original)

    def test_inclusive_age_bounds_and_fractional_median(self):
        raw = customer_fixture()
        raw["city"] = "Pune"
        raw["age"] = [10, 100, 37, 38, np.nan]
        clean, _ = clean_customers(raw)
        self.assertEqual(clean["age"].tolist(), [10, 100, 37, 38, 37.5])

    def test_city_with_no_valid_age_raises(self):
        raw = customer_fixture()
        raw.loc[3:4, "age"] = np.nan
        with self.assertRaisesRegex(ValueError, "city has no valid"):
            clean_customers(raw)

    def test_duplicate_customer_id_raises(self):
        raw = customer_fixture()
        raw.loc[1, "customer_id"] = "C1"
        with self.assertRaisesRegex(ValueError, "identifiers"):
            clean_customers(raw)


class OrderCleaningTests(unittest.TestCase):
    def test_latest_snapshot_amount_recovery_and_category_imputation(self):
        raw = order_fixture()
        original = raw.copy(deep=True)
        clean, log = clean_orders(raw, product_fixture())
        indexed = clean.set_index("order_id")
        self.assertEqual(len(clean), 7)
        self.assertEqual(log[1]["changed"], 1)
        self.assertEqual(log[2]["changed"], 1)
        self.assertEqual(indexed.loc["O1", "quantity"], 2)
        self.assertEqual(indexed.loc["O1", "order_amount"], 180.00)
        self.assertEqual(indexed.loc["O3", "order_amount"], 100.00)
        self.assertEqual(indexed.loc["O6", "order_amount"], 50.97)
        self.assertEqual(indexed.loc["O3", "delivery_minutes"], 20)
        self.assertEqual(indexed.loc["O4", "delivery_minutes"], 20)
        self.assertEqual(indexed.loc["O6", "delivery_minutes"], 80)
        self.assertTrue(pd.isna(indexed.loc["O7", "delivery_minutes"]))
        self.assertEqual(log[-2]["changed"], 3)
        self.assertEqual(log[-1]["changed"], 1)
        pd.testing.assert_frame_equal(raw, original)

    def test_missing_completed_category_has_no_fallback(self):
        raw = order_fixture()
        raw.loc[raw["product_id"].eq("P2"), "delivery_minutes"] = np.nan
        with self.assertRaisesRegex(ValueError, "category has no valid"):
            clean_orders(raw, product_fixture())

    def test_unknown_product_raises(self):
        raw = order_fixture()
        raw.loc[raw["order_id"].eq("O2"), "product_id"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "unknown product"):
            clean_orders(raw, product_fixture())

    def test_pending_final_snapshot_raises(self):
        raw = order_fixture()
        raw.loc[raw["order_id"].eq("O2"), "status"] = "pending"
        with self.assertRaisesRegex(ValueError, "completed or cancelled"):
            clean_orders(raw, product_fixture())

    def test_conflicting_same_time_snapshots_raise(self):
        raw = order_fixture()
        raw.loc[1, "order_ts"] = "2025-04-01 11:00:00"
        with self.assertRaisesRegex(ValueError, "Conflicting snapshots"):
            clean_orders(raw, product_fixture())

    def test_bad_timestamp_raises(self):
        raw = order_fixture()
        raw.loc[0, "order_ts"] = "not a timestamp"
        with self.assertRaises(ValueError):
            clean_orders(raw, product_fixture())

    def test_storage_roundtrip_nulls_format_and_repeatability(self):
        customers, _ = clean_customers(customer_fixture())
        products = product_fixture()
        orders, _ = clean_orders(order_fixture(), products)
        tables = {"customers": customers, "products": products, "orders": orders}
        # Keep temporary work inside the workspace; TemporaryDirectory removes
        # only this directory after all SQLite handles have been closed.
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as temporary:
            root = Path(temporary)
            persist(tables, root)
            persist(tables, root)
            with closing(sqlite3.connect(root / "data" / "instamart.db")) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0], 7)
                self.assertIsNone(connection.execute(
                    "SELECT delivery_minutes FROM orders WHERE order_id='O7'"
                ).fetchone()[0])
                self.assertEqual(connection.execute(
                    "SELECT order_ts FROM orders WHERE order_id='O1'"
                ).fetchone()[0], "2025-04-01 11:00:00")
                self.assertEqual(connection.execute(
                    "SELECT signup_date FROM customers WHERE customer_id='C1'"
                ).fetchone()[0], "2024-01-01 00:00:00")
                self.assertFalse(connection.execute("PRAGMA foreign_key_check").fetchall())
                connection.execute("PRAGMA foreign_keys=ON")
                with self.assertRaises(sqlite3.IntegrityError):
                    connection.execute("UPDATE orders SET customer_id='missing' WHERE order_id='O1'")


if __name__ == "__main__":
    unittest.main()
