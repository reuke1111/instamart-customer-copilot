"""Completed-order RFM, measured K selection, business profiles and PCA."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from scripts.common import ROOT, read_tables, save_json

SNAPSHOT = pd.Timestamp("2025-10-01 00:00:00")
RFM_COLUMNS = ["recency", "frequency", "monetary"]
SEGMENT_COLUMNS = [
    "customer_id", "recency", "frequency", "monetary",
    "segment_id", "segment_name", "is_premium",
]


def build_rfm(orders, snapshot=SNAPSHOT):
    frame = orders.copy()
    frame["order_ts"] = pd.to_datetime(frame["order_ts"])
    completed = frame.loc[frame["status"].eq("completed") & frame["order_ts"].lt(snapshot)]
    grouped = completed.groupby("customer_id", sort=True).agg(
        last_order=("order_ts", "max"), frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum"),
    )
    grouped["recency"] = (snapshot - grouped.pop("last_order")).dt.days
    return grouped[RFM_COLUMNS].reset_index()


def segment_profiles(rfm, labels):
    frame = rfm.assign(segment_id=labels)
    profiles = frame.groupby("segment_id", sort=True).agg(
        customers=("customer_id", "size"), mean_recency=("recency", "mean"),
        mean_frequency=("frequency", "mean"), mean_monetary=("monetary", "mean"),
    )
    premium = int(profiles["mean_monetary"].idxmax())
    names = {premium: ("Premium", "Offer priority service and personalised high-value retention outreach.")}
    remaining = profiles.drop(index=premium)
    if len(remaining):
        frequent = int(remaining["mean_frequency"].idxmax())
        names[frequent] = ("Frequent", "Reward repeat grocery shopping with replenishment reminders and loyalty benefits.")
        remaining = remaining.drop(index=frequent)
    if len(remaining):
        inactive = int(remaining["mean_recency"].idxmax())
        names[inactive] = ("Dormant", "Test a limited win-back offer and ask what stopped repeat purchases.")
        remaining = remaining.drop(index=inactive)
    # Labels are assigned by measured profile order, never by K-means ID.
    for index, segment_id in enumerate(remaining.sort_values("mean_recency").index):
        name = "Occasional" if index == 0 else f"Occasional {index + 1}"
        names[int(segment_id)] = (name, "Encourage a second shopping occasion with relevant basket suggestions.")
    profiles["segment_name"] = [names[int(index)][0] for index in profiles.index]
    profiles["recommended_action"] = [names[int(index)][1] for index in profiles.index]
    profiles["is_premium"] = profiles.index == premium
    frame["segment_name"] = frame["segment_id"].map(profiles["segment_name"])
    frame["is_premium"] = frame["segment_id"].eq(premium)
    return frame[SEGMENT_COLUMNS], profiles.reset_index()


def compute(orders):
    rfm = build_rfm(orders)
    scaled = StandardScaler().fit_transform(rfm[RFM_COLUMNS])
    comparisons, estimators = [], {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = model.fit_predict(scaled)
        comparisons.append({
            "k": k, "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(scaled, labels)),
        })
        estimators[k] = model
    comparison = pd.DataFrame(comparisons)
    # Operational constraint: 3-5 actionable groups; choose their best silhouette.
    candidates = comparison.loc[comparison["k"].isin([3, 4, 5])]
    chosen = int(candidates.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    segments, profiles = segment_profiles(rfm, estimators[chosen].labels_)
    pca = PCA(n_components=2)
    components = pca.fit_transform(scaled)
    projection = pd.DataFrame(components, columns=["pc1", "pc2"])
    projection.insert(0, "customer_id", rfm["customer_id"])
    projection["segment_id"] = segments["segment_id"]
    summary = {
        "snapshot": str(SNAPSHOT), "customers": len(rfm),
        "recency_median": float(rfm["recency"].median()),
        "recency_min": int(rfm["recency"].min()), "recency_max": int(rfm["recency"].max()),
        "frequency_mean": float(rfm["frequency"].mean()), "frequency_max": int(rfm["frequency"].max()),
        "monetary_mean": float(rfm["monetary"].mean()), "monetary_max": float(rfm["monetary"].max()),
        "selected_k": chosen, "selection_rule": "Highest silhouette among K=3,4,5; ties prefer smaller K",
        "pca_variance": pca.explained_variance_ratio_.tolist(),
        "pca_variance_total": float(pca.explained_variance_ratio_.sum()),
        "premium_segment_id": int(profiles.loc[profiles["is_premium"], "segment_id"].iloc[0]),
    }
    return summary, segments, profiles, comparison, projection


def run(root=ROOT):
    root = Path(root)
    folder = root / "outputs" / "part_b"
    folder.mkdir(parents=True, exist_ok=True)
    summary, segments, profiles, comparisons, projection = compute(read_tables(root, clean=True)["orders"])
    segments.to_csv(root / "outputs" / "customer_segments.csv", index=False, lineterminator="\n")
    for name, frame in [
        ("segment_profiles", profiles), ("kmeans_comparison", comparisons), ("pca_projection", projection),
    ]:
        frame.to_csv(folder / f"{name}.csv", index=False, lineterminator="\n")
    save_json(folder / "segmentation.json", summary)
    return summary


if __name__ == "__main__":
    print(run())
