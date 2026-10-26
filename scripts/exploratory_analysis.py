"""Pandas cross-check, named city aggregations, and completed-order EDA."""

import json

import numpy as np
import pandas as pd

from scripts.common import ROOT, output_dir, read_tables, save_json
from scripts.sql_analysis import execute_queries

CORRELATION_COLUMNS = ["order_amount", "discount_pct", "quantity", "delivery_minutes"]


def analyze(tables, q2):
    orders = tables["orders"]
    completed = orders.loc[orders["status"].eq("completed")].merge(
        tables["customers"][["customer_id", "city"]], on="customer_id", validate="many_to_one"
    ).merge(
        tables["products"][["product_id", "category"]], on="product_id", validate="many_to_one"
    )
    pivot = completed.pivot_table(
        index="city", columns="category", values="order_amount", aggfunc="sum", margins=True
    )
    sql_cells = q2.pivot(index="city", columns="category", values="revenue").sort_index().sort_index(axis=1)
    pandas_cells = pivot.drop(index="All", columns="All").sort_index().sort_index(axis=1)
    pd.testing.assert_index_equal(sql_cells.index, pandas_cells.index)
    pd.testing.assert_index_equal(sql_cells.columns, pandas_cells.columns)
    np.testing.assert_allclose(sql_cells.to_numpy(), pandas_cells.to_numpy(), rtol=0, atol=1e-8)
    np.testing.assert_allclose(pivot.loc["All", "All"], q2["revenue"].sum(), rtol=0, atol=1e-8)
    city = completed.groupby("city").agg(
        revenue_sum=("order_amount", "sum"),
        revenue_mean=("order_amount", "mean"),
        completed_order_count=("order_amount", "count"),
    )
    amount = completed["order_amount"]
    q1, median, q3 = amount.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    corr = completed[CORRELATION_COLUMNS].corr()
    pairs = [
        (left, right, float(corr.loc[left, right]))
        for index, left in enumerate(CORRELATION_COLUMNS)
        for right in CORRELATION_COLUMNS[index + 1:]
    ]
    left, right, coefficient = max(pairs, key=lambda entry: abs(entry[2]))
    report = {
        "completed_orders": len(completed),
        "pivot_grand_total": float(pivot.loc["All", "All"]),
        "sql_grand_total": float(q2["revenue"].sum()),
        "all_30_cells_match": True,
        "statistics": {
            "mean": float(amount.mean()), "median": float(median),
            "skewness": float(amount.skew()), "q1": float(q1), "q3": float(q3),
            "iqr": float(iqr), "lower_fence": float(lower), "upper_fence": float(upper),
            "lower_outliers": int(amount.lt(lower).sum()),
            "upper_outliers": int(amount.gt(upper).sum()),
        },
        "largest_absolute_correlation": {"left": left, "right": right, "coefficient": coefficient},
    }
    return report, pivot, city, corr


def run(root=ROOT):
    report, pivot, city, corr = analyze(read_tables(root, clean=True), execute_queries(root)["Q2"])
    folder = output_dir(root)
    pivot.to_csv(folder / "revenue_pivot.csv", lineterminator="\n")
    city.to_csv(folder / "city_aggregates.csv", lineterminator="\n")
    corr.to_csv(folder / "correlations.csv", lineterminator="\n")
    save_json(folder / "eda.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
