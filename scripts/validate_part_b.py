"""Check all mandatory Part B results without refitting the predictive models."""

from pathlib import Path
import ast
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.build_readme import render
from scripts.common import ROOT, load_json, read_tables, verify_sources
from scripts.lapse_features import (
    FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, build_dataset, split_dataset,
)
from scripts.lapse_models import LOGISTIC_GRID, FOREST_GRID, evaluate, oversample_training, retention_list
from scripts.part_b_integrity import assert_preserved
from scripts.segmentation import SEGMENT_COLUMNS, compute
from scripts.model_artifact import load_model
from scripts.validate_checkpoints import close, same_frame

K_REFERENCE = {
    2: (1820.8, 0.465), 3: (1268.2, 0.445), 4: (743.1, 0.526),
    5: (617.4, 0.481), 6: (503.2, 0.482), 7: (408.5, 0.416), 8: (366.5, 0.412),
}


def validate(root=ROOT):
    if not __debug__:
        raise RuntimeError("Part B validation requires assertions; do not use -O")
    root = Path(root)
    folder = root / "outputs/part_b"
    verify_sources(root)
    assert_preserved(root)
    for relative in [
        "customer_segments.csv", "lapse_model.joblib", "retention_list.csv",
        *[f"part_b/{name}" for name in [
            "segmentation.json", "segment_profiles.csv", "kmeans_comparison.csv", "pca_projection.csv",
            "cohort.json", "lapse_dataset.csv", "split_assignments.csv",
            "models.json", "model_metadata.json", "logistic_cv.csv", "forest_cv.csv", "test_predictions.csv",
            "forest_importances.csv", "logistic_coefficients.csv", "part_a_baseline.json",
        ]],
    ]:
        assert (root / "outputs" / relative).is_file(), f"Missing {relative}"
    tables = read_tables(root, clean=True)
    summary, expected_segments, profiles, comparison, projection = compute(tables["orders"])
    stored_summary = load_json(folder / "segmentation.json")
    for key, value in summary.items():
        if isinstance(value, float):
            close(stored_summary[key], value)
        elif key == "pca_variance":
            np.testing.assert_allclose(stored_summary[key], value, rtol=0, atol=1e-12)
        else:
            assert stored_summary[key] == value, key
    segments = pd.read_csv(root / "outputs/customer_segments.csv")
    assert segments.columns.tolist() == SEGMENT_COLUMNS
    assert segments["customer_id"].is_unique
    assert segments["customer_id"].is_monotonic_increasing
    same_frame(segments, expected_segments)
    same_frame(pd.read_csv(folder / "segment_profiles.csv"), profiles)
    same_frame(pd.read_csv(folder / "kmeans_comparison.csv"), comparison)
    same_frame(pd.read_csv(folder / "pca_projection.csv"), projection)
    assert summary["customers"] == 1175
    assert (summary["recency_median"], summary["recency_min"], summary["recency_max"]) == (19, 0, 180)
    close(round(summary["frequency_mean"], 2), 9.46)
    assert summary["frequency_max"] == 33
    close(round(summary["monetary_mean"], 2), 2552.06)
    close(summary["monetary_max"], 14766.55)
    for row in comparison.itertuples():
        inertia, silhouette = K_REFERENCE[row.k]
        close(row.inertia, inertia, tolerance=0.05 * inertia)
        close(row.silhouette, silhouette, tolerance=0.05)
    assert summary["selected_k"] in {3, 4, 5}
    assert summary["selected_k"] == 4
    assert sorted(profiles["customers"]) == [138, 182, 419, 436]
    premium = profiles.loc[profiles["mean_monetary"].idxmax()]
    assert premium["customers"] == 138
    close(premium["mean_monetary"], 7130, tolerance=50)
    frequent = profiles.loc[profiles["mean_frequency"].idxmax()]
    assert frequent["customers"] == 419
    close(frequent["mean_frequency"], 19.4, tolerance=0.2)
    assert segments["is_premium"].equals(segments["segment_id"].eq(premium["segment_id"]))
    np.testing.assert_allclose(summary["pca_variance"], [0.634, 0.218], rtol=0, atol=0.02)
    close(summary["pca_variance_total"], 0.852, tolerance=0.02)

    features, labels = build_dataset(tables)
    train, test, train_y, test_y = split_dataset(features, labels)
    assert len(features) == 1156 and int(labels.sum()) == 443
    close(round(labels.mean() * 100, 2), 38.32)
    assert (len(train), len(test)) == (924, 232)
    assert train_y.value_counts().to_dict() == {0: 570, 1: 354}
    assert test_y.value_counts().to_dict() == {0: 143, 1: 89}
    assert set(train.index).isdisjoint(test.index)
    assert set(train.index) | set(test.index) == set(features.index)
    assert features.columns.tolist() == FEATURES
    assert not features.isna().any().any()
    assert not set(features) & {"lapsed", "is_premium", "segment_id", "segment_name", "customer_id"}
    same_frame(pd.read_csv(folder / "lapse_dataset.csv"), features.join(labels).reset_index())
    assigned = pd.read_csv(folder / "split_assignments.csv").set_index("customer_id")["split"]
    assert set(assigned.index[assigned.eq("train")]) == set(train.index)
    assert set(assigned.index[assigned.eq("test")]) == set(test.index)
    cohort = load_json(folder / "cohort.json")
    for key, actual in {
        "cohort": len(features), "lapsed": int(labels.sum()), "lapse_rate": float(labels.mean()),
        "train_rows": len(train), "test_rows": len(test), "missing_feature_values": 0,
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
    }.items():
        assert cohort[key] == actual, key
    close(cohort["full_cohort_frequency_mean"], features["frequency"].mean())
    close(cohort["training_frequency_mean"], train["frequency"].mean())

    report = load_json(folder / "models.json")
    predictions = pd.read_csv(folder / "test_predictions.csv")
    assert predictions["customer_id"].tolist() == test.index.tolist()
    np.testing.assert_array_equal(predictions["actually_lapsed"], test_y)
    for name in ["baseline", "logistic", "forest"]:
        measured = evaluate(test_y, predictions[f"{name}_prediction"])
        expected = report["baseline"] if name == "baseline" else report["models"][name]["test"]
        assert measured == expected, name
    close(report["baseline"]["accuracy"], 0.616, tolerance=0.001)
    assert report["baseline"]["f1"] == 0
    assert predictions["baseline_prediction"].eq(0).all()
    for name, grid in [("logistic", LOGISTIC_GRID), ("forest", FOREST_GRID)]:
        cv = pd.read_csv(folder / f"{name}_cv.csv")
        assert len(cv) == (5 if name == "logistic" else 9)
        params = [ast.literal_eval(value) for value in cv["params"]]
        from sklearn.model_selection import ParameterGrid
        assert params == list(ParameterGrid(grid))
        chosen = int(cv["mean_test_score"].idxmax())
        model_report = report["models"][name]
        assert params[chosen] == model_report["best_params"]
        close(cv.loc[chosen, "mean_test_score"], model_report["cv_f1"])
        assert model_report["scaler_rows_seen"] == len(train)
        close(model_report["learned_frequency_mean"], train["frequency"].mean())
    selected = max(report["models"], key=lambda name: report["models"][name]["cv_f1"])
    assert report["selected_model"] == selected
    assert report["models"][selected]["test"]["f1"] >= 0.55

    model = load_model(root)
    np.testing.assert_array_equal(model.predict(test), predictions[f"{selected}_prediction"])
    np.testing.assert_allclose(model.predict_proba(test)[:, 1], predictions[f"{selected}_probability"], rtol=0, atol=1e-14)
    assert report["reload_predictions_identical"] and report["reload_probabilities_identical"]
    assert model.named_steps["preprocess"].get_feature_names_out().tolist() == report["models"][selected]["transformed_features"]
    for key, value in report["models"][selected]["best_params"].items():
        assert model.get_params()[key] == value
    scaler = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
    assert int(scaler.n_samples_seen_) == 924
    np.testing.assert_allclose(scaler.mean_, train[NUMERIC_FEATURES].mean().to_numpy(), rtol=0, atol=1e-10)
    encoder = model.named_steps["preprocess"].named_transformers_["categorical"].named_steps["encoder"]
    assert encoder.drop == "first" and encoder.handle_unknown == "ignore"
    for column, values in zip(CATEGORICAL_FEATURES, encoder.categories_):
        assert values.tolist() == sorted(train[column].unique())
    assert np.isfinite(model.named_steps["preprocess"].transform(test)).all()
    balanced_x, balanced_y, added = oversample_training(train, train_y)
    assert balanced_y.value_counts().to_dict() == {0: 570, 1: 570}
    assert set(balanced_x.index).isdisjoint(test.index)
    assert report["oversampling"]["sampled_customer_ids"] == added
    assert report["oversampling"]["added_rows"] == 216
    assert report["oversampling"]["after"] == evaluate(test_y, predictions["balanced_logistic_prediction"])
    assert report["oversampling"]["before"] == report["models"]["logistic"]["test"]
    for metric in ["precision", "recall", "f1"]:
        close(
            report["oversampling"]["metric_deltas"][metric],
            report["oversampling"]["after"][metric] - report["oversampling"]["before"][metric],
        )

    importances = pd.read_csv(folder / "forest_importances.csv")
    close(importances["importance"].sum(), 1.0)
    assert importances["importance"].is_monotonic_decreasing
    same_frame(importances.head(5), pd.DataFrame(report["forest_top_five"]))
    if selected == "forest":
        mapped = dict(zip(model.named_steps["preprocess"].get_feature_names_out(),
                          model.named_steps["model"].feature_importances_))
        np.testing.assert_allclose(
            importances["importance"], importances["feature"].map(mapped), rtol=0, atol=1e-14,
        )
    coefficients = pd.read_csv(folder / "logistic_coefficients.csv").set_index("feature")
    for row in report["logistic_interpretation"]:
        close(row["coefficient"], coefficients.loc[row["feature"], "coefficient"])
        assert row["sign"] == ("positive" if row["coefficient"] >= 0 else "negative")
    expected_retention = retention_list(model, test, test_y, segments)
    retained = pd.read_csv(root / "outputs/retention_list.csv")
    assert retained.columns.tolist() == ["customer_id", "lapse_probability", "actually_lapsed"]
    assert len(retained) == 10 and retained["customer_id"].is_unique
    assert retained["lapse_probability"].is_monotonic_decreasing
    same_frame(retained, expected_retention)
    assert report["retention"]["listed"] == 10
    assert report["retention"]["actually_lapsed"] == int(retained["actually_lapsed"].sum())
    assert report["retention"]["post_hoc"]
    assert (root / "README.md").read_text(encoding="utf-8") == render(root), "README is stale"


def main():
    validate()
    print("ALL PART B CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
