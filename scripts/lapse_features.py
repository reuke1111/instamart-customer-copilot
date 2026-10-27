"""Point-in-time feature aggregation, September labels and a fixed holdout."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from scripts.common import ROOT, read_tables, save_json

CUTOFF = pd.Timestamp("2025-09-01 00:00:00")
LABEL_END = pd.Timestamp("2025-10-01 00:00:00")
NUMERIC_FEATURES = [
    "recency", "frequency", "monetary", "average_discount",
    "cancellation_rate", "average_delivery", "age", "tenure",
]
CATEGORICAL_FEATURES = ["favourite_category", "city", "gender"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(tables, cutoff=CUTOFF):
    """Read only pre-cutoff order rows; customer_id is the index, not a feature."""
    orders = tables["orders"].copy()
    orders["order_ts"] = pd.to_datetime(orders["order_ts"])
    history = orders.loc[
        orders["order_ts"].lt(cutoff) & orders["status"].isin(["completed", "cancelled"])
    ].copy()
    completed = history.loc[history["status"].eq("completed")]
    if history["order_id"].isna().any() or history["order_id"].duplicated().any():
        raise ValueError("Feature history requires unique nonmissing order identifiers")
    if not completed["product_id"].isin(tables["products"]["product_id"]).all():
        raise ValueError("Feature history refers to an unknown product")
    features = completed.groupby("customer_id", sort=True).agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"), average_discount=("discount_pct", "mean"),
        average_delivery=("delivery_minutes", "mean"),
    )
    features["recency"] = (cutoff - features.pop("last_order")).dt.days
    cancelled = history["status"].eq("cancelled").groupby(history["customer_id"]).mean()
    features["cancellation_rate"] = cancelled.reindex(features.index)
    categories = completed.merge(
        tables["products"][["product_id", "category"]], on="product_id", validate="many_to_one"
    ).groupby(["customer_id", "category"]).size().rename("orders").reset_index()
    # Stable alphabetical tie break; no future orders resolve tied favourites.
    favourite = categories.sort_values(
        ["customer_id", "orders", "category"], ascending=[True, False, True]
    ).drop_duplicates("customer_id").set_index("customer_id")["category"]
    features["favourite_category"] = favourite.reindex(features.index)
    customers = tables["customers"].set_index("customer_id")
    features = features.join(customers[["city", "age", "gender", "signup_date"]], validate="one_to_one")
    features["tenure"] = (cutoff - pd.to_datetime(features.pop("signup_date"))).dt.days
    features = features[FEATURES].sort_index()
    if features.isna().any().any():
        raise ValueError("The supplied clean data unexpectedly produced missing customer features")
    if not np.isfinite(features[NUMERIC_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError("Numeric model features must be finite")
    return features


def build_labels(orders, customer_ids, cutoff=CUTOFF, label_end=LABEL_END):
    dates = pd.to_datetime(orders["order_ts"])
    active = orders.loc[
        orders["status"].eq("completed") & dates.ge(cutoff) & dates.lt(label_end), "customer_id"
    ]
    return pd.Series(
        (~customer_ids.isin(active)).astype(int), index=customer_ids, name="lapsed", dtype="int64",
    )


def build_dataset(tables):
    features = build_features(tables)
    labels = build_labels(tables["orders"], features.index)
    return features, labels


def split_dataset(features, labels):
    return train_test_split(features, labels, test_size=0.2, stratify=labels, random_state=42)


def run(root=ROOT):
    root = Path(root)
    features, labels = build_dataset(read_tables(root, clean=True))
    train, test, train_y, test_y = split_dataset(features, labels)
    folder = root / "outputs" / "part_b"
    folder.mkdir(parents=True, exist_ok=True)
    features.join(labels).to_csv(folder / "lapse_dataset.csv", index=True, lineterminator="\n")
    assignments = pd.Series("train", index=train.index, name="split")
    assignments = pd.concat([assignments, pd.Series("test", index=test.index, name="split")]).sort_index()
    assignments.to_csv(folder / "split_assignments.csv", lineterminator="\n")
    summary = {
        "cutoff": str(CUTOFF), "label_end_exclusive": str(LABEL_END),
        "cohort": len(features), "lapsed": int(labels.sum()), "lapse_rate": float(labels.mean()),
        "train_rows": len(train), "test_rows": len(test),
        "train_class_counts": {str(k): int(v) for k, v in train_y.value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in test_y.value_counts().sort_index().items()},
        "numeric_features": NUMERIC_FEATURES, "categorical_features": CATEGORICAL_FEATURES,
        "missing_feature_values": int(features.isna().sum().sum()),
        "full_cohort_frequency_mean": float(features["frequency"].mean()),
        "training_frequency_mean": float(train["frequency"].mean()),
    }
    save_json(folder / "cohort.json", summary)
    return features, labels, summary


if __name__ == "__main__":
    print(run()[2])
