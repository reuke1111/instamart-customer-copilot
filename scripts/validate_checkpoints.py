"""Read-only verification of every Part A reference and persisted artifact."""

from pathlib import Path
import re
import sys

# Also support the specifically requested direct-script invocation.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.audit_data import audit_tables
from scripts.build_readme import eda_answers, render
from scripts.checkpoints import (
    CITY_AGGREGATES, CITY_COUNTS, GENDER_COUNTS, Q6_LEADERS,
    RAW_METRICS, REVENUE, SQL_ROW_COUNTS,
)
from scripts.clean_data import clean_customers, clean_orders, cleaning_summary
from scripts.common import ROOT, TABLES, TIME_FORMAT, connect_readonly, load_json, read_tables, verify_sources
from scripts.exploratory_analysis import analyze
from scripts.sql_analysis import execute_queries, summarize


def close(actual, expected, tolerance=1e-8):
    assert np.isclose(actual, expected, rtol=0, atol=tolerance), (
        f"Expected {expected}, got {actual} (absolute tolerance {tolerance})"
    )


def same_frame(actual, expected):
    pd.testing.assert_frame_equal(
        actual.reset_index(drop=True), expected.reset_index(drop=True),
        # CSV does not retain dtype or the name of a column axis.
        check_dtype=False, check_names=False, check_exact=False, rtol=0, atol=1e-8,
    )


def validate(root=ROOT):
    if not __debug__:
        raise RuntimeError("Validation requires assertions; do not run Python with -O")
    root = Path(root)
    verify_sources(root)
    folder = root / "outputs" / "part_a"
    required = [
        ".gitignore", ".gitattributes", "requirements.txt", "README.md",
        "sql/queries.sql", "data/instamart.db",
        *[f"data/clean/{name}.csv" for name in TABLES],
        *[f"outputs/part_a/q{index}.csv" for index in range(1, 8)],
        *[f"outputs/part_a/{name}" for name in (
            "audit.json", "cleaning.json", "sql_summary.json", "eda.json",
            "revenue_pivot.csv", "city_aggregates.csv", "correlations.csv",
        )],
    ]
    for relative in required:
        assert (root / relative).is_file(), f"Missing artifact: {relative}"
    raw, clean = read_tables(root), read_tables(root, clean=True)
    audit = audit_tables(raw)
    assert audit["metrics"] == RAW_METRICS, audit["metrics"]
    assert audit == load_json(folder / "audit.json"), "Stale audit report"
    assert {name: frame.shape for name, frame in raw.items()} == {
        "customers": (1200, 5), "products": (40, 4), "orders": (13084, 9),
    }
    assert len(pd.read_csv(root / "data" / "raw" / "tickets.csv", keep_default_na=False)) == 164

    customers, products, orders = (clean[name] for name in TABLES)
    assert customers["customer_id"].tolist() == raw["customers"]["customer_id"].tolist()
    assert customers["customer_id"].is_unique and customers["customer_id"].notna().all()
    assert customers["city"].value_counts().to_dict() == CITY_COUNTS
    assert customers["gender"].value_counts().to_dict() == GENDER_COUNTS
    assert customers["age"].isna().sum() == 0
    assert customers["age"].between(18, 60).all()
    assert customers["signup_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()
    assert pd.to_datetime(customers["signup_date"], format=TIME_FORMAT).notna().all()
    same_frame(products, raw["products"])
    assert products["product_id"].is_unique
    assert products["category"].nunique() == 5

    assert len(orders) == 12309
    assert orders["order_id"].is_unique and orders["order_id"].notna().all()
    assert set(orders["order_id"]) == set(raw["orders"]["order_id"])
    assert orders["status"].value_counts().to_dict() == {"completed": 11110, "cancelled": 1199}
    assert orders["status"].eq("pending").sum() == 0
    assert orders["order_amount"].isna().sum() == 0
    assert orders["customer_id"].isin(customers["customer_id"]).all()
    assert orders["product_id"].isin(products["product_id"]).all()
    completed, cancelled = orders["status"].eq("completed"), orders["status"].eq("cancelled")
    assert orders.loc[completed, "delivery_minutes"].isna().sum() == 0
    assert orders.loc[completed, "delivery_minutes"].ge(0).all()
    assert orders.loc[cancelled, "delivery_minutes"].isna().sum() == 1199
    close(orders.loc[completed, "order_amount"].sum(), REVENUE)
    assert orders["order_ts"].str.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}").all()
    assert pd.to_datetime(orders["order_ts"], format=TIME_FORMAT).notna().all()
    latest = raw["orders"].groupby("order_id")["order_ts"].max()
    assert orders.set_index("order_id")["order_ts"].sort_index().equals(latest.sort_index())

    # Recompute transformations and compare every persisted value, not just totals.
    expected_customers, customer_log = clean_customers(raw["customers"])
    expected_orders, order_log = clean_orders(raw["orders"], raw["products"])
    expected_customers["signup_date"] = expected_customers["signup_date"].dt.strftime(TIME_FORMAT)
    expected_orders["order_ts"] = expected_orders["order_ts"].dt.strftime(TIME_FORMAT)
    same_frame(customers, expected_customers)
    same_frame(orders, expected_orders)
    cleaning = load_json(folder / "cleaning.json")
    assert cleaning["customer_rules"] == customer_log
    assert cleaning["order_rules"] == order_log
    assert cleaning["summary"] == cleaning_summary(customers, orders)
    assert cleaning["city_age_medians"] == customers.groupby("city")["age"].median().to_dict()
    medians = orders.loc[completed].merge(
        products[["product_id", "category"]], on="product_id", validate="many_to_one"
    ).groupby("category")["delivery_minutes"].median().to_dict()
    assert cleaning["category_delivery_medians"] == medians

    with connect_readonly(root) as connection:
        names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert names == set(TABLES), names
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        for name in TABLES:
            frame = pd.read_sql_query(f"SELECT * FROM {name} ORDER BY rowid", connection)
            same_frame(frame, clean[name])
        assert connection.execute(
            "SELECT COUNT(*) FROM orders WHERE status='cancelled' AND delivery_minutes IS NULL"
        ).fetchone()[0] == 1199
        assert connection.execute(
            "SELECT COUNT(*) FROM orders WHERE typeof(order_ts) != 'text' OR length(order_ts) != 19"
        ).fetchone()[0] == 0

    queries = execute_queries(root)
    sql = summarize(queries)
    assert sql["row_counts"] == SQL_ROW_COUNTS, sql["row_counts"]
    close(sql["q2_total"], REVENUE)
    assert sql["q2_highest"] == {
        "city": "Bengaluru", "category": "Electronics", "revenue": 233909.45,
    }
    close(round(sql["q5_average"], 2), 2552.06)
    leaders = {row["city"]: (row["customer_id"], row["total_spend"]) for row in sql["q6_leaders"]}
    assert leaders == Q6_LEADERS, leaders
    assert sql["q6_rank_row_number_differences"] == 8
    ranked = queries["Q6"]
    for _, group in ranked.groupby("city"):
        same_frame(
            group[["spend_rank", "dense_rank", "row_number"]],
            pd.DataFrame({
                "spend_rank": group["total_spend"].rank(method="min", ascending=False),
                "dense_rank": group["total_spend"].rank(method="dense", ascending=False),
                "row_number": range(1, len(group) + 1),
            }),
        )
    close(sql["q7_final_total"], REVENUE)
    daily = queries["Q7"]
    assert daily["day"].tolist() == pd.date_range("2025-04-01", "2025-09-30").strftime("%Y-%m-%d").tolist()
    np.testing.assert_allclose(daily["revenue"].cumsum(), daily["cumulative_revenue"], rtol=0, atol=1e-8)
    assert sql == load_json(folder / "sql_summary.json"), "Stale SQL summary"
    for name, frame in queries.items():
        same_frame(pd.read_csv(folder / f"{name.lower()}.csv"), frame)

    eda, pivot, cities, correlations = analyze(clean, queries["Q2"])
    assert eda == load_json(folder / "eda.json"), "Stale EDA summary"
    same_frame(pd.read_csv(folder / "revenue_pivot.csv"), pivot.reset_index())
    same_frame(pd.read_csv(folder / "city_aggregates.csv"), cities.reset_index())
    pd.testing.assert_frame_equal(
        pd.read_csv(folder / "correlations.csv", index_col=0), correlations,
        check_dtype=False, rtol=0, atol=1e-12,
    )
    close(eda["pivot_grand_total"], REVENUE)
    assert eda["completed_orders"] == 11110 and eda["all_30_cells_match"]
    for city, (revenue, mean, count) in CITY_AGGREGATES.items():
        close(cities.loc[city, "revenue_sum"], revenue)
        close(round(cities.loc[city, "revenue_mean"], 2), mean)
        assert cities.loc[city, "completed_order_count"] == count
    stats = eda["statistics"]
    for key, expected in {
        "median": 135.00, "q1": 64.80, "q3": 297.00, "iqr": 232.20,
        "lower_fence": -283.50, "upper_fence": 645.30,
    }.items():
        close(stats[key], expected)
    close(round(stats["mean"], 2), 269.91)
    close(stats["skewness"], 4.61, tolerance=0.01)
    assert stats["upper_outliers"] == 897 and stats["lower_outliers"] == 0
    pair = eda["largest_absolute_correlation"]
    assert (pair["left"], pair["right"]) == ("order_amount", "discount_pct")
    close(pair["coefficient"], -0.095, tolerance=0.001)
    close(correlations.loc["order_amount", "quantity"], 0.087, tolerance=0.001)
    assert sum(len(answer.split()) for answer in eda_answers(stats, pair)) <= 200

    readme = (root / "README.md").read_text(encoding="utf-8")
    assert readme == render(root), "README does not match measured outputs; rebuild it"
    assert re.findall(r"(?m)^# (.+)$", readme) == [
        "Overview", "Reproduce", "Part A", "Part B", "Part C", "Part D", "Limitations and Next Steps",
    ]
    assert chr(36) not in readme, "Use INR for monetary amounts"
    assert "correlation of this size does not establish causation" in readme.lower()


def main():
    validate()
    print("ALL CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
