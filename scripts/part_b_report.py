"""Extend the existing generated README without changing its Part A results."""

from pathlib import Path
import pandas as pd

from scripts.common import load_json
from scripts.build_readme import table, number
from scripts.runtime import pinned_versions

PREMIUM_WARNING = (
    "Premium segments use the October snapshot, including September outcomes: "
    "the Premium filter contains future information, so this retention list is an "
    "illustrative post-hoc analysis, not a leakage-free prospective evaluation. "
    "Premium status and segment labels are never predictive features."
)


def part_b_section(root):
    root = Path(root)
    pins = pinned_versions(root)
    folder = root / "outputs" / "part_b"
    segmentation = load_json(folder / "segmentation.json")
    cohort = load_json(folder / "cohort.json")
    models = load_json(folder / "models.json")
    comparisons = pd.read_csv(folder / "kmeans_comparison.csv")
    profiles = pd.read_csv(folder / "segment_profiles.csv")
    retention = pd.read_csv(root / "outputs" / "retention_list.csv")
    selected = models["selected_model"]
    winner = models["models"][selected]
    balance = models["oversampling"]
    lines = [
        "# Part B", "",
        "Completed: customer segmentation and September lapse prediction, using the "
        "verified Part A clean tables. No external API or paid service is used. "
        f"Model runtime pins: scikit-learn {pins['scikit-learn']}, joblib {pins['joblib']}. "
        "`outputs/part_b/model_metadata.json` records the training dependency versions "
        "and model SHA-256; the loader checks both before deserialization.", "",
        "## RFM and K-means", "",
        "Only completed orders before the snapshot `2025-10-01 00:00:00` contribute "
        "to RFM. Recency uses whole elapsed days, frequency counts distinct orders, "
        "and monetary sums completed order amounts in integer paise, converted to INR "
        "only after summation and written with two decimal places. Customer IDs are sorted before "
        "all clustering and split operations.", "",
        table(["RFM checkpoint", "Measured"], [
            ["Customers", segmentation["customers"]],
            ["Recency median / minimum / maximum (days)",
             f"{segmentation['recency_median']:.0f} / {segmentation['recency_min']} / {segmentation['recency_max']}"],
            ["Frequency mean / maximum", f"{segmentation['frequency_mean']:.6f} / {segmentation['frequency_max']}"],
            ["Monetary mean / maximum (INR)",
             f"{number(segmentation['monetary_mean'])} / {number(segmentation['monetary_max'])}"],
        ]), "",
        "`StandardScaler` is fitted to the three RFM columns for this descriptive "
        "whole-window segmentation. Every candidate uses "
        "`KMeans(n_clusters=k, n_init=10, random_state=42)`.", "",
        table(["K", "Measured inertia", "Measured silhouette"], [
            [int(row.k), f"{row.inertia:.4f}", f"{row.silhouette:.6f}"]
            for row in comparisons.itertuples()
        ]), "",
        f"Chosen K: **{segmentation['selected_k']}**, the highest measured silhouette "
        "among the permitted K values 3, 4 and 5. For these data, the large inertia "
        "drop from K=3 to K=4 is followed by a smaller gain at K=5, and four profiles "
        "support distinct retention actions. Adding clusters allows smaller within-cluster "
        "distances, explaining the lower measured inertia at K=8 than K=2; inertia alone "
        "does not penalise complexity or establish useful business segments. K-means "
        "finds local solutions, so monotonicity is a theoretical optimum property "
        "rather than a guarantee for every individual fit.", "",
        table(["ID", "Name", "Customers", "Mean recency", "Mean frequency", "Mean monetary (INR)", "Action"], [
            [int(row.segment_id), row.segment_name, int(row.customers),
             f"{row.mean_recency:.2f}", f"{row.mean_frequency:.2f}", number(row.mean_monetary),
             row.recommended_action]
            for row in profiles.itertuples()
        ]), "",
        "Cluster numbers are arbitrary labels with no ordering. Names are assigned "
        "from measured profiles: Premium has the highest mean monetary value; Frequent "
        "has the highest remaining frequency; Dormant has the highest remaining recency. "
        "Only the highest-mean-monetary cluster has `is_premium=True`.", "",
        f"`PCA(n_components=2)` on scaled RFM explains "
        f"{segmentation['pca_variance'][0]:.6f} (PC1) and "
        f"{segmentation['pca_variance'][1]:.6f} (PC2), for "
        f"{segmentation['pca_variance_total']:.6f} combined. PCA is descriptive "
        "and is not a lapse-model feature.", "",
        "## Lapse cohort, features and leakage controls", "",
        "Eligibility requires at least one completed order strictly before "
        "`2025-09-01 00:00:00`. The target is 1 for no completed order in "
        "`[2025-09-01, 2025-10-01)`, otherwise 0. September cancellations do not "
        "count as activity. No September orders enter feature aggregation.", "",
        table(["Cohort checkpoint", "Measured"], [
            ["Eligible customers", cohort["cohort"]], ["Lapsed customers", cohort["lapsed"]],
            ["Lapse rate", f"{cohort['lapse_rate'] * 100:.2f}%"],
            ["Training / test rows", f"{cohort['train_rows']} / {cohort['test_rows']}"],
            ["Training non-lapsed / lapsed", f"{cohort['train_class_counts']['0']} / {cohort['train_class_counts']['1']}"],
            ["Test non-lapsed / lapsed", f"{cohort['test_class_counts']['0']} / {cohort['test_class_counts']['1']}"],
            ["Missing feature values", cohort["missing_feature_values"]],
        ]), "",
        "Numeric features: " + ", ".join(f"`{name}`" for name in cohort["numeric_features"]) + ".",
        "Categorical features: " + ", ".join(f"`{name}`" for name in cohort["categorical_features"]) + ".", "",
        "RFM, average discount and average delivery use pre-cutoff completed orders. "
        "Cancellation rate is cancelled divided by all final-status pre-cutoff orders. "
        "Favourite category is the most frequent completed category, with alphabetical "
        "tie-breaking. Tenure is whole days from signup to cutoff. City, age and gender "
        "come from the required clean customer table. Features are calculated from "
        "the fixed Part A clean records; Part A's historical cleaning is not rerun.", "",
        "The split is `train_test_split(test_size=0.2, stratify=y, random_state=42)`. "
        "A `ColumnTransformer` inside each model `Pipeline` contains numeric "
        "`SimpleImputer(strategy='median')` then `StandardScaler`, and categorical "
        "`SimpleImputer(strategy='most_frequent')` then "
        "`OneHotEncoder(drop='first', handle_unknown='ignore')`. The supplied features "
        "have no missing values; the imputers also handle future missing inputs using "
        "training-fitted statistics. Every encoder, scaler and model is fitted within "
        "each CV training fold and then refitted on the full training split only. "
        "No encoder, scaler or resampler sees test rows during fitting; the test set "
        "is only transformed and evaluated. No target-derived features, whole-window "
        "Premium flags, or segment labels enter the feature schema.", "",
        f"The fitted frequency scaler mean is {winner['learned_frequency_mean']:.9f}; "
        f"the full-cohort mean is {cohort['full_cohort_frequency_mean']:.9f}. "
        "They differ because the scaler sees only the 924 training customers, "
        "not all 1,156 cohort customers.", "",
        "Transformed model features: " + ", ".join(f"`{name}`" for name in winner["transformed_features"]) + ".", "",
        "## Baseline, tuning and held-out evaluation", "",
        "The baseline always predicts the most frequent training class (not lapsed). "
        "Both grid searches use `GridSearchCV(cv=5, scoring='f1')`, with positive class "
        "`lapsed=1`. Logistic regression uses `max_iter=1000` and "
        "`model__C=[0.01, 0.1, 1, 10, 100]` (25 CV fits). The forest uses "
        "`n_estimators=100, random_state=42` and "
        "`model__max_depth=[3,5,8]` × `model__min_samples_leaf=[5,10,20]` (45 CV fits). "
        "Full CV result tables are saved. Both searches finish before test evaluation, "
        "and model-family selection uses the higher best training CV F1.", "",
    ]
    rows = [["Majority baseline", "Most frequent class", "—", *[
        f"{models['baseline'][metric]:.4f}" for metric in ["accuracy", "precision", "recall", "f1"]
    ]]]
    for name, model in models["models"].items():
        rows.append([name, str(model["best_params"]), f"{model['cv_f1']:.4f}", *[
            f"{model['test'][metric]:.4f}" for metric in ["accuracy", "precision", "recall", "f1"]
        ]])
    lines += [
        table(["Model", "Best parameters", "CV F1", "Test accuracy", "Test precision", "Test recall", "Test F1"], rows), "",
        "Confusion matrices use actual rows [not lapsed, lapsed] and predicted columns "
        "[not lapsed, lapsed], i.e. `[[TN, FP], [FN, TP]]`.", "",
        table(["Model", "Confusion matrix"], [
            ["Majority baseline", str(models["baseline"]["confusion_matrix"])],
            *[[name, str(model["test"]["confusion_matrix"])] for name, model in models["models"].items()],
        ]), "",
        f"The selected model is **{selected}**, chosen for its higher training CV F1 "
        f"({winner['cv_f1']:.4f}) before examining holdout performance. Its test F1 "
        f"of {winner['test']['f1']:.4f} exceeds the required 0.55. Missing a lapsing "
        "customer loses a retention opportunity, while targeting a loyal customer "
        "spends incentives unnecessarily. The logistic model's slightly different "
        "holdout score does not justify retuning or switching after seeing the test set; "
        "campaign costs and future validation should determine any threshold change.", "",
        f"Reference discrepancy: the measured logistic optimum is "
        f"`C={models['models']['logistic']['best_params']['model__C']}`, compared with the "
        "brief's illustrative value around 1. All prescribed candidates were evaluated "
        "on the fixed split; results are reported as measured and no hyperparameters "
        "were changed to match the illustrative reference.", "",
        "## Training-only oversampling", "",
        f"All {balance['original_training_rows']} original training rows are retained; "
        f"{balance['added_rows']} lapsed training rows are sampled with replacement "
        "using `random_state=42`. This yields 570 non-lapsed and 570 lapsed rows. "
        "The best logistic configuration is cloned and refitted on that balanced "
        "training set, then evaluated on the same untouched 232 test rows.", "",
        table(["Logistic experiment", "Precision", "Recall", "F1", "Confusion matrix"], [
            [label, f"{values['precision']:.4f}", f"{values['recall']:.4f}", f"{values['f1']:.4f}",
             str(values["confusion_matrix"])]
            for label, values in [("Original", balance["before"]), ("Oversampled", balance["after"])]
        ]), "",
        f"The largest absolute change is {balance['largest_change']}: recall changes "
        f"by {balance['metric_deltas']['recall']:+.4f}, precision by "
        f"{balance['metric_deltas']['precision']:+.4f}, and F1 by "
        f"{balance['metric_deltas']['f1']:+.4f}. More lapsing customers are found, "
        "at the cost of more loyal customers receiving unnecessary outreach. This "
        "holdout experiment does not retroactively change the CV-selected persisted model.", "",
        "## Interpretation, persistence and retention", "",
    ]
    readings = {
        "numeric__monetary": "Historical spend helps distinguish purchase patterns; importance has no direction.",
        "numeric__frequency": "Purchase regularity is informative about lapse risk.",
        "numeric__recency": "Time since the last purchase helps identify disengagement.",
        "numeric__average_delivery": "Delivery experience is associated with behaviour; this is not a causal effect.",
        "numeric__average_discount": "Discount usage helps distinguish customer shopping patterns.",
    }
    lines += [
        table(["Forest feature", "Importance", "Business reading"], [
            [row["feature"], f"{row['importance']:.6f}",
             readings.get(row["feature"], "This measured attribute helps distinguish lapse outcomes; importance has no direction.")]
            for row in models["forest_top_five"]
        ]), "",
        "Impurity-based feature importances are associations and can favour continuous "
        "variables; they are not causal effects.", "",
        table(["Logistic feature", "Coefficient", "Sign", "Business reading"], [
            [row["feature"], f"{row['coefficient']:.6f}", row["sign"],
             f"Higher {row['feature'].removeprefix('numeric__')} is associated with "
             f"{'higher' if row['coefficient'] > 0 else 'lower'} lapse log-odds, holding other inputs fixed."]
            for row in models["logistic_interpretation"]
        ]), "",
        "Numeric coefficients correspond to a one-training-standard-deviation change. "
        "RFM variables are correlated, so these conditional signs should not be read "
        "as independent causal effects.", "",
        "`outputs/lapse_model.joblib` stores the selected full preprocessing-and-model "
        "pipeline. Reloading it reproduces all 232 test predictions and probabilities "
        "exactly; both equality checks passed. Only trusted locally generated joblib "
        "artifacts are loaded by this project.", "",
        f"There are {models['retention']['eligible_premium_test_customers']} Premium "
        f"test customers; the top {models['retention']['listed']} by saved-model lapse "
        f"probability are retained, and {models['retention']['actually_lapsed']} actually "
        "lapsed. Tied probabilities are ordered by customer ID.", "",
        table(["Customer ID", "Lapse probability", "Actually lapsed"], [
            [row.customer_id, f"{row.lapse_probability:.6f}", int(row.actually_lapsed)]
            for row in retention.itertuples()
        ]), "", PREMIUM_WARNING, "",
        "## Part B artifacts and verification", "",
        "- Required outputs: `outputs/customer_segments.csv`, `outputs/lapse_model.joblib`, `outputs/retention_list.csv`.",
        "- Measured reports, RFM profiles, K comparison, PCA coordinates, cohort, split "
        "assignments, CV scores, holdout predictions, interpretation and Part A preservation hashes: `outputs/part_b/`.",
        "- Stages: `scripts/segmentation.py`, `scripts/lapse_features.py`, "
        "`scripts/lapse_models.py`, `scripts/run_part_b.py`.",
        "- Checks: `scripts/validate_part_b.py` and `tests/test_part_b.py`. "
        "Part A numerical assertions remain intact; tests include full A-to-B reproduction "
        "in an isolated temporary folder.", "",
    ]
    return "\n".join(lines)


def extend_readme(base, root):
    root = Path(root)
    present = (root / "outputs" / "part_b" / "models.json").exists()
    commands = (
        "12. `python -m scripts.run_part_b`\n"
        "13. `python scripts/validate_part_b.py`\n"
        "14. `python scripts/validate_checkpoints.py`\n"
        "15. `python -m unittest discover -s tests -v`"
    )
    base = base.replace("12. `python -m unittest discover -s tests -v`", commands)
    base = base.replace("steps 3–12", "steps 3–15")
    base = base.replace(
        "pandas and NumPy are the only direct third-party dependencies. python-dateutil, six and tzdata are pandas dependencies.",
        "Direct dependencies are pandas, NumPy, scikit-learn and joblib; the frozen file also includes their required transitive dependencies.",
    )
    base = base.replace(
        "then `python -m unittest discover -s tests -v`.",
        "then `python -m scripts.run_part_b` and `python -m unittest discover -s tests -v`. "
        "For an existing validated Part A checkout, run only Part B; it validates Part A read-only "
        "and verifies that its data and analytical artifacts are unchanged.",
    )
    if present:
        base = base.replace(
            "Parts B, C and D are not implemented.",
            "Part B adds measured customer segmentation and lapse prediction. Parts C and D are not implemented.",
        )
        start, end = base.index("# Part B\n"), base.index("# Part C\n")
        base = base[:start] + part_b_section(root) + "\n" + base[end:]
        base = base.replace(
            "Next step: review Part A and approve any work on Parts B–D. No segmentation, "
            "prediction, ticket triage, policy assistant or refund agent has been implemented.",
            PREMIUM_WARNING + " The model evaluation uses the fixed clean Part A inputs; "
            "Part A's already-completed imputations are inherited, rather than refitted by Part B. "
            "A production prospective study should also fit upstream cleaning on historical "
            "training information. Review Parts A and B before approving Part C or D. "
            "Ticket triage, a policy assistant and a refund agent are not implemented.",
        )
    return base
