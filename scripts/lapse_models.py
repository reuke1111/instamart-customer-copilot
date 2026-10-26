"""Training-only preprocessing, fixed CV grids, evaluation and persistence."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.common import ROOT, read_tables, save_json
from scripts.lapse_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_dataset, split_dataset

LOGISTIC_GRID = {"model__C": [0.01, 0.1, 1, 10, 100]}
FOREST_GRID = {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]}


def preprocessing():
    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ], remainder="drop")


def model_pipeline(model):
    return Pipeline([("preprocess", preprocessing()), ("model", model)])


def evaluate(labels, predictions):
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, pos_label=1, zero_division=0)),
        "recall": float(recall_score(labels, predictions, pos_label=1, zero_division=0)),
        "f1": float(f1_score(labels, predictions, pos_label=1, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def oversample_training(train, labels):
    """Keep every original row and append only sampled minority training rows."""
    counts = labels.value_counts()
    if set(counts.index) != {0, 1} or counts[1] > counts[0]:
        raise ValueError("This experiment expects lapsed=1 to be the training minority")
    added = labels.loc[labels.eq(1)].sample(
        n=int(counts[0] - counts[1]), replace=True, random_state=42,
    ).index
    balanced_x = pd.concat([train, train.loc[added]])
    balanced_y = pd.concat([labels, labels.loc[added]])
    assert set(balanced_x.index) <= set(train.index)
    return balanced_x, balanced_y, added.tolist()


def retention_list(model, test, labels, segments):
    premium_ids = segments.loc[segments["is_premium"].eq(True), "customer_id"]
    eligible = test.loc[test.index.isin(premium_ids)]
    result = pd.DataFrame({
        "customer_id": eligible.index,
        "lapse_probability": model.predict_proba(eligible)[:, 1],
        "actually_lapsed": labels.reindex(eligible.index).to_numpy(),
    })
    return result.sort_values(
        ["lapse_probability", "customer_id"], ascending=[False, True]
    ).head(10).reset_index(drop=True)


def run(root=ROOT):
    root = Path(root)
    folder = root / "outputs" / "part_b"
    folder.mkdir(parents=True, exist_ok=True)
    features, labels = build_dataset(read_tables(root, clean=True))
    train, test, train_y, test_y = split_dataset(features, labels)
    baseline = DummyClassifier(strategy="most_frequent").fit(train, train_y)
    predictions = pd.DataFrame({"customer_id": test.index, "actually_lapsed": test_y.to_numpy()})
    predictions["baseline_prediction"] = baseline.predict(test)
    report = {"baseline": evaluate(test_y, baseline.predict(test)), "models": {}}
    searches = {}
    specs = [
        ("logistic", LogisticRegression(max_iter=1000), LOGISTIC_GRID),
        ("forest", RandomForestClassifier(n_estimators=100, random_state=42), FOREST_GRID),
    ]
    # Both grids finish using only training rows before either sees the test set.
    for name, estimator, grid in specs:
        print(f"Tuning {name}: {len(list(ParameterGrid(grid)))} candidates x 5 folds", flush=True)
        search = GridSearchCV(model_pipeline(estimator), param_grid=grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(train, train_y)
        searches[name] = search
        cv = pd.DataFrame(search.cv_results_)
        cv[["params", "mean_test_score", "std_test_score", "rank_test_score"]].to_csv(
            folder / f"{name}_cv.csv", index=False, lineterminator="\n",
        )
    selected = max(searches, key=lambda name: searches[name].best_score_)
    # Model family selection is fixed by CV, not by looking at holdout scores.
    report["selected_model"] = selected
    report["selection_rule"] = "Highest best mean 5-fold training CV F1; holdout evaluated only afterwards"
    for name, search in searches.items():
        pipeline = search.best_estimator_
        predicted = pipeline.predict(test)
        predictions[f"{name}_prediction"] = predicted
        predictions[f"{name}_probability"] = pipeline.predict_proba(test)[:, 1]
        transformed = pipeline.named_steps["preprocess"].transform(test)
        assert np.isfinite(np.asarray(transformed)).all()
        scaler = pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["scaler"]
        report["models"][name] = {
            "best_params": search.best_params_, "cv_f1": float(search.best_score_),
            "test": evaluate(test_y, predicted), "scaler_rows_seen": int(scaler.n_samples_seen_),
            "learned_frequency_mean": float(scaler.mean_[NUMERIC_FEATURES.index("frequency")]),
            "transformed_features": pipeline.named_steps["preprocess"].get_feature_names_out().tolist(),
        }

    balanced_x, balanced_y, added = oversample_training(train, train_y)
    balanced = clone(searches["logistic"].best_estimator_).fit(balanced_x, balanced_y)
    balanced_predictions = balanced.predict(test)
    predictions["balanced_logistic_prediction"] = balanced_predictions
    before = report["models"]["logistic"]["test"]
    after = evaluate(test_y, balanced_predictions)
    deltas = {metric: after[metric] - before[metric] for metric in ["precision", "recall", "f1"]}
    report["oversampling"] = {
        "before": before, "after": after, "metric_deltas": deltas,
        "largest_change": max(deltas, key=lambda metric: abs(deltas[metric])),
        "balanced_class_counts": {str(k): int(v) for k, v in balanced_y.value_counts().sort_index().items()},
        "original_training_rows": len(train), "added_rows": len(added),
        "sampled_customer_ids": added, "test_rows": len(test),
    }
    forest = searches["forest"].best_estimator_
    importances = pd.DataFrame({
        "feature": forest.named_steps["preprocess"].get_feature_names_out(),
        "importance": forest.named_steps["model"].feature_importances_,
    }).sort_values(["importance", "feature"], ascending=[False, True]).reset_index(drop=True)
    importances.to_csv(folder / "forest_importances.csv", index=False, lineterminator="\n")
    logistic = searches["logistic"].best_estimator_
    coefficients = pd.DataFrame({
        "feature": logistic.named_steps["preprocess"].get_feature_names_out(),
        "coefficient": logistic.named_steps["model"].coef_[0],
    })
    coefficients["sign"] = np.where(coefficients["coefficient"].ge(0), "positive", "negative")
    coefficients.to_csv(folder / "logistic_coefficients.csv", index=False, lineterminator="\n")
    report["forest_top_five"] = importances.head(5).to_dict("records")
    report["logistic_interpretation"] = coefficients.loc[coefficients["feature"].isin([
        "numeric__recency", "numeric__frequency", "numeric__monetary",
    ])].to_dict("records")

    final_model = searches[selected].best_estimator_
    model_path = root / "outputs" / "lapse_model.joblib"
    joblib.dump(final_model, model_path, compress=3)
    reloaded = joblib.load(model_path)  # Only our own locally created model.
    before_prediction = final_model.predict(test)
    after_prediction = reloaded.predict(test)
    np.testing.assert_array_equal(before_prediction, after_prediction)
    np.testing.assert_array_equal(final_model.predict_proba(test), reloaded.predict_proba(test))
    report["reload_predictions_identical"] = True
    report["reload_probabilities_identical"] = True
    segments = pd.read_csv(root / "outputs" / "customer_segments.csv")
    retention = retention_list(reloaded, test, test_y, segments)
    retention.to_csv(root / "outputs" / "retention_list.csv", index=False, lineterminator="\n")
    report["retention"] = {
        "eligible_premium_test_customers": int(test.index.isin(segments.loc[segments["is_premium"], "customer_id"]).sum()),
        "listed": len(retention), "actually_lapsed": int(retention["actually_lapsed"].sum()),
        "post_hoc": True,
    }
    predictions.to_csv(folder / "test_predictions.csv", index=False, lineterminator="\n")
    save_json(folder / "models.json", report)
    assert report["models"][selected]["test"]["f1"] >= 0.55, "Selected model is below required test F1"
    return report


if __name__ == "__main__":
    print(run())
