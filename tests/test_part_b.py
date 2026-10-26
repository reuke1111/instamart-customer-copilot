"""Part B numerical, temporal, split, preprocessing and reproduction checks."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import StandardScaler

from scripts.common import ROOT, load_json, read_tables
from scripts.lapse_features import (
    CUTOFF, LABEL_END, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    build_dataset, build_features, build_labels, split_dataset,
)
from scripts.lapse_models import (
    LOGISTIC_GRID, FOREST_GRID, evaluate, model_pipeline, oversample_training, retention_list,
)
from scripts.part_b_integrity import assert_preserved, fingerprint
from scripts.segmentation import RFM_COLUMNS, SEGMENT_COLUMNS, build_rfm, compute, segment_profiles
from scripts.validate_part_b import K_REFERENCE, validate


class SegmentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = read_tables(clean=True)
        cls.summary, cls.segments, cls.profiles, cls.comparison, cls.projection = compute(cls.tables["orders"])
        cls.rfm = build_rfm(cls.tables["orders"])
        cls.scaled = StandardScaler().fit_transform(cls.rfm[RFM_COLUMNS])

    def test_rfm_counts_and_statistics(self):
        self.assertEqual(len(self.rfm), 1175)
        self.assertEqual(self.rfm["recency"].agg(["median", "min", "max"]).tolist(), [19, 0, 180])
        self.assertAlmostEqual(self.rfm["frequency"].mean(), 9.46, delta=0.005)
        self.assertEqual(self.rfm["frequency"].max(), 33)
        self.assertAlmostEqual(self.rfm["monetary"].mean(), 2552.06, delta=0.005)
        self.assertAlmostEqual(self.rfm["monetary"].max(), 14766.55)
        self.assertTrue(self.rfm["customer_id"].is_monotonic_increasing)

    def test_kmeans_determinism_and_actual_objective(self):
        first = KMeans(n_clusters=4, n_init=10, random_state=42).fit(self.scaled)
        second = KMeans(n_clusters=4, n_init=10, random_state=42).fit(self.scaled)
        np.testing.assert_array_equal(first.labels_, second.labels_)
        np.testing.assert_array_equal(first.labels_, self.segments["segment_id"])
        squared_distance = np.square(self.scaled - first.cluster_centers_[first.labels_]).sum()
        self.assertAlmostEqual(first.inertia_, squared_distance)
        score = silhouette_score(self.scaled, first.labels_)
        self.assertAlmostEqual(score, self.comparison.set_index("k").loc[4, "silhouette"])

    def test_all_k_metrics_and_choice(self):
        self.assertEqual(self.comparison["k"].tolist(), list(range(2, 9)))
        for row in self.comparison.itertuples():
            expected_inertia, expected_silhouette = K_REFERENCE[row.k]
            self.assertAlmostEqual(row.inertia, expected_inertia, delta=expected_inertia * 0.05)
            self.assertAlmostEqual(row.silhouette, expected_silhouette, delta=0.05)
        eligible = self.comparison.loc[self.comparison["k"].isin([3, 4, 5])]
        self.assertEqual(self.summary["selected_k"], int(eligible.loc[eligible["silhouette"].idxmax(), "k"]))

    def test_profiles_and_premium_are_independent_of_cluster_label_number(self):
        self.assertEqual(sorted(self.profiles["customers"]), [138, 182, 419, 436])
        premium_id = self.profiles.loc[self.profiles["mean_monetary"].idxmax(), "segment_id"]
        np.testing.assert_array_equal(self.segments["is_premium"], self.segments["segment_id"].eq(premium_id))
        remapped = self.segments["segment_id"].map({0: 9, 1: 7, 2: 6, 3: 8})
        renamed, _ = segment_profiles(self.rfm, remapped)
        np.testing.assert_array_equal(renamed["is_premium"], self.segments["is_premium"])
        self.assertEqual(int(renamed["is_premium"].sum()), 138)
        self.assertTrue(renamed.loc[renamed["is_premium"], "segment_name"].eq("Premium").all())

    def test_pca_variance_and_segment_schema(self):
        np.testing.assert_allclose(self.summary["pca_variance"], [0.634, 0.218], rtol=0, atol=0.02)
        self.assertAlmostEqual(self.summary["pca_variance_total"], 0.852, delta=0.02)
        self.assertEqual(self.projection.shape, (1175, 4))
        self.assertEqual(self.segments.columns.tolist(), SEGMENT_COLUMNS)


class LapseFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = read_tables(clean=True)
        cls.features, cls.labels = build_dataset(cls.tables)
        cls.train, cls.test, cls.train_y, cls.test_y = split_dataset(cls.features, cls.labels)

    def test_cohort_and_september_label_from_independent_sets(self):
        orders = self.tables["orders"]
        dates = pd.to_datetime(orders["order_ts"])
        before = set(orders.loc[orders["status"].eq("completed") & dates.lt(CUTOFF), "customer_id"])
        active = set(orders.loc[
            orders["status"].eq("completed") & dates.ge(CUTOFF) & dates.lt(LABEL_END), "customer_id",
        ])
        self.assertEqual(set(self.features.index), before)
        self.assertEqual(set(self.labels.index[self.labels.eq(1)]), before - active)
        self.assertEqual((len(self.features), int(self.labels.sum())), (1156, 443))
        self.assertAlmostEqual(self.labels.mean() * 100, 38.32, delta=0.005)

    def test_future_order_changes_cannot_change_any_feature(self):
        altered = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
        future = pd.to_datetime(altered["orders"]["order_ts"]).ge(CUTOFF)
        altered["orders"].loc[future, ["order_amount", "delivery_minutes", "discount_pct"]] = 999999.0
        altered["orders"].loc[future, "status"] = "cancelled"
        altered["orders"].loc[future, "product_id"] = "FUTURE_ONLY_UNKNOWN_PRODUCT"
        # Segments and labels in a joined customer table must not enter features.
        altered["customers"]["is_premium"] = True
        altered["customers"]["segment_id"] = 999
        altered["customers"]["lapsed"] = 1
        pd.testing.assert_frame_equal(build_features(altered), self.features)
        pre_only = {name: frame.copy(deep=True) for name, frame in self.tables.items()}
        pre_only["orders"] = pre_only["orders"].loc[~future]
        pd.testing.assert_frame_equal(build_features(pre_only), self.features)

    def test_cutoff_label_boundaries_and_cancelled_activity(self):
        orders = pd.DataFrame({
            "customer_id": ["A", "B", "C", "D"],
            "order_ts": ["2025-09-01 00:00:00", "2025-10-01 00:00:00",
                         "2025-09-30 23:59:59", "2025-09-30 23:59:59"],
            "status": ["completed", "completed", "cancelled", "completed"],
        })
        labels = build_labels(orders, pd.Index(["A", "B", "C", "D"], name="customer_id"))
        self.assertEqual(labels.tolist(), [0, 1, 1, 0])

    def test_stratified_split_and_no_missing_or_target_features(self):
        self.assertEqual((len(self.train), len(self.test)), (924, 232))
        self.assertEqual(self.train_y.value_counts().to_dict(), {0: 570, 1: 354})
        self.assertEqual(self.test_y.value_counts().to_dict(), {0: 143, 1: 89})
        self.assertTrue(set(self.train.index).isdisjoint(self.test.index))
        self.assertEqual(set(self.train.index) | set(self.test.index), set(self.features.index))
        self.assertEqual(self.features.columns.tolist(), FEATURES)
        self.assertFalse(self.features.isna().any().any())
        self.assertFalse(set(FEATURES) & {"customer_id", "lapsed", "is_premium", "segment_id", "segment_name"})

    def test_cancellation_rate_and_favourite_tie_break(self):
        tables = {
            "customers": pd.DataFrame({
                "customer_id": ["A"], "city": ["Pune"], "age": [30],
                "gender": ["male"], "signup_date": ["2025-01-01"],
            }),
            "products": pd.DataFrame({"product_id": ["P1", "P2"], "category": ["Zebra", "Alpha"]}),
            "orders": pd.DataFrame({
                "order_id": ["O1", "O2", "O3", "O4"], "customer_id": ["A"] * 4,
                "product_id": ["P1", "P2", "P1", "P1"],
                "order_ts": ["2025-08-01 00:00:00"] * 3 + ["2025-09-01 00:00:00"],
                "status": ["completed", "completed", "cancelled", "completed"],
                "order_amount": [100, 200, 999, 100000],
                "discount_pct": [10, 20, 99, 50], "delivery_minutes": [20, 30, np.nan, 1000],
            }),
        }
        row = build_features(tables).loc["A"]
        self.assertEqual((row["frequency"], row["monetary"], row["average_discount"]), (2, 300, 15))
        self.assertEqual(row["average_delivery"], 25)
        self.assertEqual(row["favourite_category"], "Alpha")
        self.assertAlmostEqual(row["cancellation_rate"], 1 / 3)
        self.assertEqual(row["recency"], 31)

    def test_training_only_oversampling_preserves_originals(self):
        balanced_x, balanced_y, added = oversample_training(self.train, self.train_y)
        self.assertEqual(balanced_y.value_counts().to_dict(), {0: 570, 1: 570})
        self.assertEqual(len(added), 216)
        self.assertTrue(self.train_y.loc[added].eq(1).all())
        self.assertTrue(set(balanced_x.index).isdisjoint(self.test.index))
        pd.testing.assert_frame_equal(balanced_x.iloc[:len(self.train)], self.train)
        pd.testing.assert_series_equal(balanced_y.iloc[:len(self.train)], self.train_y)
        self.assertEqual(added, oversample_training(self.train, self.train_y)[2])

    def test_preprocessing_fits_only_training_and_handles_unknowns(self):
        pipeline = model_pipeline(LogisticRegression(max_iter=1000)).fit(self.train, self.train_y)
        preprocess = pipeline.named_steps["preprocess"]
        scaler = preprocess.named_transformers_["numeric"].named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, self.train[NUMERIC_FEATURES].mean(), rtol=0, atol=1e-10)
        self.assertEqual(scaler.n_samples_seen_, 924)
        self.assertNotAlmostEqual(scaler.mean_[1], self.features["frequency"].mean())
        encoder = preprocess.named_transformers_["categorical"].named_steps["encoder"]
        heldout = self.test.copy()
        heldout["city"] = "HELD_OUT_ONLY_CITY"
        heldout.loc[heldout.index[0], "average_delivery"] = np.nan
        with warnings.catch_warnings(record=True):
            transformed = preprocess.transform(heldout)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertNotIn("HELD_OUT_ONLY_CITY", encoder.categories_[CATEGORICAL_FEATURES.index("city")])
        self.assertEqual(scaler.n_samples_seen_, 924)

    def test_preprocessing_is_refitted_inside_each_cv_fold(self):
        results = cross_validate(
            model_pipeline(LogisticRegression(max_iter=1000)), self.train, self.train_y,
            cv=5, scoring="f1", return_estimator=True, return_indices=True,
        )
        for pipeline, indices in zip(results["estimator"], results["indices"]["train"]):
            fold = self.train.iloc[indices]
            scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
            self.assertEqual(scaler.n_samples_seen_, len(fold))
            np.testing.assert_allclose(scaler.mean_, fold[NUMERIC_FEATURES].mean(), rtol=0, atol=1e-10)
            self.assertTrue(set(fold.index).isdisjoint(self.test.index))
            encoder = pipeline.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encoder"]
            for column, learned in zip(CATEGORICAL_FEATURES, encoder.categories_):
                self.assertEqual(learned.tolist(), sorted(fold[column].unique()))


class ModelArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_json(ROOT / "outputs/part_b/models.json")
        cls.features, cls.labels = build_dataset(read_tables(clean=True))
        cls.train, cls.test, cls.train_y, cls.test_y = split_dataset(cls.features, cls.labels)
        cls.model = joblib.load(ROOT / "outputs/lapse_model.joblib")

    def test_exact_hyperparameter_grids(self):
        self.assertEqual(LOGISTIC_GRID, {"model__C": [0.01, 0.1, 1, 10, 100]})
        self.assertEqual(FOREST_GRID, {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]})
        for name, count in [("logistic", 5), ("forest", 9)]:
            self.assertEqual(len(pd.read_csv(ROOT / f"outputs/part_b/{name}_cv.csv")), count)

    def test_model_selected_by_cv_and_required_holdout_f1(self):
        selected = max(self.report["models"], key=lambda name: self.report["models"][name]["cv_f1"])
        self.assertEqual(self.report["selected_model"], selected)
        actual = evaluate(self.test_y, self.model.predict(self.test))
        self.assertEqual(actual, self.report["models"][selected]["test"])
        self.assertGreaterEqual(actual["f1"], 0.55)
        self.assertEqual(self.report["baseline"]["f1"], 0)
        self.assertAlmostEqual(self.report["baseline"]["accuracy"], 0.616, delta=0.001)

    def test_saved_predictions_and_second_save_reload_are_identical(self):
        stored = pd.read_csv(ROOT / "outputs/part_b/test_predictions.csv")
        selected = self.report["selected_model"]
        np.testing.assert_array_equal(self.model.predict(self.test), stored[f"{selected}_prediction"])
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as temporary:
            path = Path(temporary) / "roundtrip.joblib"
            joblib.dump(self.model, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.predict(self.test), self.model.predict(self.test))
            np.testing.assert_array_equal(loaded.predict_proba(self.test), self.model.predict_proba(self.test))

    def test_retention_exact_top_ten_heldout_premium_and_schema(self):
        segments = pd.read_csv(ROOT / "outputs/customer_segments.csv")
        actual = pd.read_csv(ROOT / "outputs/retention_list.csv")
        expected = retention_list(self.model, self.test, self.test_y, segments)
        self.assertEqual(actual.columns.tolist(), ["customer_id", "lapse_probability", "actually_lapsed"])
        self.assertEqual(len(actual), 10)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=0, atol=1e-14)
        self.assertTrue(set(actual["customer_id"]) <= set(self.test.index))
        premium = set(segments.loc[segments["is_premium"], "customer_id"])
        self.assertTrue(set(actual["customer_id"]) <= premium)
        self.assertEqual(int(actual["actually_lapsed"].sum()), self.report["retention"]["actually_lapsed"])

    def test_all_checkpoint_artifacts_and_part_a_preservation(self):
        validate()
        assert_preserved()


class FullReproductionTests(unittest.TestCase):
    def test_fresh_source_pipeline_a_through_b_preserves_a(self):
        original = fingerprint()
        with tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT) as temporary:
            root = Path(temporary)
            for name in [
                "generate_data.py", "mock_llm.py", "source_hashes.json",
                "requirements.txt", ".gitignore", ".gitattributes",
            ]:
                shutil.copy2(ROOT / name, root / name)
            for name in ["scripts", "sql"]:
                shutil.copytree(ROOT / name, root / name, ignore=shutil.ignore_patterns("__pycache__"))
            first = subprocess.run(
                [sys.executable, "-m", "scripts.run_part_a"], cwd=root,
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            before_b = fingerprint(root)
            second = subprocess.run(
                [sys.executable, "-m", "scripts.run_part_b"], cwd=root,
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(second.returncode, 0, second.stdout + second.stderr)
            self.assertIn("ALL PART B CHECKPOINTS PASSED", second.stdout)
            self.assertEqual(before_b, fingerprint(root))
            self.assertEqual(original, fingerprint())
            for relative in [
                "customer_segments.csv", "retention_list.csv", "part_b/test_predictions.csv",
                "part_b/lapse_dataset.csv", "part_b/split_assignments.csv",
            ]:
                pd.testing.assert_frame_equal(
                    pd.read_csv(ROOT / "outputs" / relative),
                    pd.read_csv(root / "outputs" / relative),
                    check_exact=False, rtol=0, atol=1e-12,
                )
            features, labels = build_dataset(read_tables(root, clean=True))
            _, test, _, _ = split_dataset(features, labels)
            recreated = joblib.load(root / "outputs/lapse_model.joblib")
            existing = joblib.load(ROOT / "outputs/lapse_model.joblib")
            np.testing.assert_array_equal(existing.predict(test), recreated.predict(test))
            np.testing.assert_allclose(existing.predict_proba(test), recreated.predict_proba(test), rtol=0, atol=1e-14)
            # Extend the existing source-only test through Part C. All original
            # A/B assertions above remain intact, as does full README equality.
            from scripts.part_c_integrity import fingerprint as fingerprint_ab
            protected_ab = fingerprint_ab(root)
            third = subprocess.run(
                [sys.executable, "-m", "scripts.run_part_c"], cwd=root,
                capture_output=True, text=True, timeout=120,
            )
            self.assertEqual(third.returncode, 0, third.stdout + third.stderr)
            self.assertIn("ALL PART C CHECKPOINTS PASSED", third.stdout)
            self.assertEqual(protected_ab, fingerprint_ab(root))
            for relative in ["triage_results.csv", "triage_budget_demo.csv"]:
                self.assertEqual(
                    (root / "outputs" / relative).read_bytes(),
                    (ROOT / "outputs" / relative).read_bytes(),
                )
            for path in (root / "outputs/part_c").iterdir():
                if path.name != "parts_a_b_baseline.json":
                    self.assertEqual(path.read_bytes(), (ROOT / "outputs/part_c" / path.name).read_bytes())
            self.assertEqual((root / "README.md").read_bytes(), (ROOT / "README.md").read_bytes())


if __name__ == "__main__":
    unittest.main()
