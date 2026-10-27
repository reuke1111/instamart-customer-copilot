"""Regression coverage for environment, money and report reproducibility."""

from decimal import Decimal
from contextlib import closing
from io import StringIO
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.audit_data import audit_tables
from scripts.build_readme import render
from scripts.common import ROOT, load_json, read_tables, save_json
from scripts.clean_data import persist
from scripts.lapse_features import build_features
from scripts.model_artifact import load_model
from scripts.money import customer_spend, paise, write_money_csv
from scripts.runtime import check_environment
from scripts.segmentation import build_rfm


class ReproducibilityTests(unittest.TestCase):
    def test_rfm_money_equals_decimal_oracle_even_after_shuffle(self):
        orders = read_tables(clean=True)["orders"]
        completed = orders.loc[orders.status.eq("completed")]
        expected = completed.groupby("customer_id").order_amount.apply(
            lambda values: float(sum((Decimal(str(v)) for v in values), Decimal(0))))
        for shuffled in [orders, orders.sample(frac=1, random_state=17)]:
            actual = build_rfm(shuffled).set_index("customer_id").monetary
            np.testing.assert_array_equal(actual, expected)

    def test_lapse_money_equals_historical_decimal_oracle(self):
        tables = read_tables(clean=True)
        orders = tables["orders"]
        history = orders.loc[orders.status.eq("completed") & orders.order_ts.lt("2025-09-01")]
        expected = history.groupby("customer_id").order_amount.apply(
            lambda values: float(sum((Decimal(str(v)) for v in values), Decimal(0))))
        np.testing.assert_array_equal(build_features(tables).monetary, expected)

    def test_decimal_totals_and_csv_have_exact_two_place_currency(self):
        frame = pd.DataFrame({"customer_id": ["A"] * 100, "order_amount": [0.1] * 100})
        self.assertEqual(customer_spend(frame).loc["A"], 10.0)
        result = pd.DataFrame({"customer_id": ["A", "B"], "monetary": [11195.10, 0.30],
                               "probability": [0.12345678912345678, 0.9876543219876543]})
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as tmp:
            path = Path(tmp) / "money.csv"
            write_money_csv(result, path, ["monetary"])
            text = path.read_text()
            self.assertIn("A,11195.10,", text)
            self.assertIn("B,0.30,", text)
            loaded = pd.read_csv(StringIO(text), float_precision="round_trip")
            np.testing.assert_array_equal(loaded.monetary, result.monetary)
            np.testing.assert_array_equal(loaded.probability, result.probability)

    def test_money_rejects_sub_paise_nonfinite_and_negative_values(self):
        for value in [0.001, -1, float("inf"), float("nan"), None, True]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                paise(value)
        self.assertEqual(paise("19.99"), 1999)

    def test_raw_audit_schema_is_independent_of_string_inference(self):
        with pd.option_context("future.infer_string", False):
            original = audit_tables(read_tables())
        with pd.option_context("future.infer_string", True):
            inferred = audit_tables(read_tables())
        self.assertEqual(original, inferred)
        columns = original["tables"]["customers"]["column_audit"]
        self.assertEqual({x["dtype"] for x in columns if x["column"] != "age"}, {"object"})

    def test_readme_does_not_depend_on_reader_runtime_strings(self):
        expected = render()
        with patch("platform.python_version", return_value="0.0.0"), \
             patch("sqlite3.sqlite_version", "0.0.0"), \
             patch("sklearn.__version__", "0.0.0"):
            self.assertEqual(render(), expected)

    def test_environment_mismatch_is_actionable(self):
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as tmp:
            root = Path(tmp)
            (root / "requirements.txt").write_text("scikit-learn==0.0.0\n")
            with self.assertRaisesRegex(RuntimeError, "pip install -r requirements.txt"):
                check_environment(root)

    def test_model_metadata_mismatch_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as tmp:
            root = Path(tmp)
            shutil.copy2(ROOT / "requirements.txt", root / "requirements.txt")
            metadata = load_json(ROOT / "outputs/part_b/model_metadata.json")
            metadata["dependencies"]["scikit-learn"] = "1.7.2"
            save_json(root / "outputs/part_b/model_metadata.json", metadata)
            with patch("scripts.model_artifact.joblib.load") as unpickle:
                with self.assertRaisesRegex(RuntimeError, "Model training dependencies"):
                    load_model(root)
                unpickle.assert_not_called()

    def test_model_checksum_rejected_before_unpickling(self):
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as tmp:
            root = Path(tmp)
            shutil.copy2(ROOT / "requirements.txt", root / "requirements.txt")
            save_json(root / "outputs/part_b/model_metadata.json",
                      load_json(ROOT / "outputs/part_b/model_metadata.json"))
            (root / "outputs/lapse_model.joblib").write_bytes(b"damaged model")
            with patch("scripts.model_artifact.joblib.load") as unpickle:
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    load_model(root)
                unpickle.assert_not_called()

    def test_database_noop_preserves_bytes_but_changed_data_is_rebuilt(self):
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as tmp:
            root = Path(tmp)
            (root / "data").mkdir()
            database = root / "data/instamart.db"
            shutil.copy2(ROOT / "data/instamart.db", database)
            tables = read_tables(clean=True)
            before = database.read_bytes()
            persist(tables, root)
            self.assertEqual(database.read_bytes(), before)
            changed = {name: frame.copy() for name, frame in tables.items()}
            changed["customers"].loc[0, "age"] += 1
            persist(changed, root)
            self.assertNotEqual(database.read_bytes(), before)
            with closing(sqlite3.connect(database)) as connection:
                age = connection.execute("SELECT age FROM customers WHERE customer_id = ?",
                                         (changed["customers"].loc[0, "customer_id"],)).fetchone()[0]
            self.assertEqual(age, changed["customers"].loc[0, "age"])


if __name__ == "__main__":
    unittest.main()
