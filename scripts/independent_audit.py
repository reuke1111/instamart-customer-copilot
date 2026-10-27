"""Recompute rubric evidence independently of the production A/B transforms.

Uses CSV/Decimal dictionaries for cleaning and SQL oracles, independent sklearn
fits for ML, observed mock calls for triage, and fresh agent state for Part D.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

from collections import Counter, defaultdict
from contextlib import closing
from copy import deepcopy
import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import statistics
import tempfile

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mock_llm

from scripts.common import ROOT, load_json, save_json, verify_sources
from scripts.model_artifact import load_model


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def numeric(value):
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def independent_tables(root):
    customers, products, raw = [read_csv(root / f"data/raw/{name}.csv") for name in ("customers", "products", "orders")]
    exact = len(raw) - len({tuple(row.items()) for row in raw})
    latest = {}
    for row in raw:
        if row["order_id"] not in latest or row["order_ts"] > latest[row["order_id"]]["order_ts"]:
            latest[row["order_id"]] = dict(row)
    price = {p["product_id"]: Decimal(p["price_inr"]) for p in products}
    categories = {p["product_id"]: p["category"] for p in products}
    durations = defaultdict(list)
    for row in latest.values():
        value = numeric(row["delivery_minutes"])
        if row["status"] == "completed" and value is not None and value >= 0:
            durations[categories[row["product_id"]]].append(value)
    medians = {category: statistics.median(values) for category, values in durations.items()}
    orders = []
    for row in latest.values():
        amount = numeric(row["order_amount"])
        if amount is None:
            amount = round(price[row["product_id"]] * int(row["quantity"]) * (1 - Decimal(row["discount_pct"]) / 100), 2)
        duration = numeric(row["delivery_minutes"])
        if row["status"] == "cancelled":
            duration = None
        elif duration is None or duration < 0:
            duration = medians[categories[row["product_id"]]]
        orders.append({**row, "order_amount": float(amount), "delivery_minutes": float(duration) if duration is not None else np.nan,
                       "quantity": int(row["quantity"]), "discount_pct": int(row["discount_pct"])})
    ages = defaultdict(list)
    for row in customers:
        age = numeric(row["age"])
        if age is not None and 10 <= age <= 100:
            ages[row["city"].strip().title()].append(age)
    clean_customers = []
    for row in customers:
        city, age = row["city"].strip().title(), numeric(row["age"])
        if age is None or not 10 <= age <= 100:
            age = statistics.median(ages[city])
        gender = {"m": "male", "male": "male", "f": "female", "female": "female"}.get(row["gender"].lower().strip(), "unknown")
        clean_customers.append({**row, "city": city, "age": float(age), "gender": gender, "signup_date": row["signup_date"] + " 00:00:00"})
    tables = {"customers": pd.DataFrame(clean_customers), "products": pd.DataFrame(products),
              "orders": pd.DataFrame(orders).sort_values("order_id").reset_index(drop=True)}
    tables["products"]["price_inr"] = tables["products"]["price_inr"].astype(int)
    for name, frame in tables.items():
        pd.testing.assert_frame_equal(frame, pd.read_csv(root / f"data/clean/{name}.csv"), check_dtype=False, rtol=0, atol=1e-10)
    report = {"customers": len(customers), "products": len(products), "raw_orders": len(raw), "exact_duplicates": exact,
              "pending_snapshots": sum(r["status"] == "pending" for r in raw),
              "nonnumeric_amounts": sum(numeric(r["order_amount"]) is None for r in raw),
              "final_orders": len(orders), "completed": sum(r["status"] == "completed" for r in orders),
              "cancelled": sum(r["status"] == "cancelled" for r in orders),
              "completed_revenue": float(sum(Decimal(str(r["order_amount"])) for r in orders if r["status"] == "completed"))}
    assert list(report.values()) == [1200, 40, 13084, 119, 656, 120, 12309, 11110, 1199, 2998668.8]
    return tables, report


def audit_sql_eda(root, tables):
    queries = dict(re.findall(r"-- (Q\d):[^\n]*\n(.*?);", (root / "sql/queries.sql").read_text(), re.S))
    with closing(sqlite3.connect((root / "data/instamart.db").resolve().as_uri() + "?mode=ro", uri=True)) as db:
        assert db.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        results = {name: pd.read_sql_query(query, db) for name, query in queries.items()}
        for name, expected in tables.items():
            actual = pd.read_sql_query(f"SELECT * FROM {name}", db)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=0, atol=1e-10)
        assert db.execute("SELECT COUNT(*) FROM orders WHERE status='cancelled' AND delivery_minutes IS NULL").fetchone()[0] == 1199
    completed = tables["orders"].query("status == 'completed'")
    spend = completed.groupby("customer_id").order_amount.sum().round(2)
    qualifying = completed.loc[completed.order_amount.gt(500)].groupby("customer_id").order_amount.sum()
    assert set(results["Q1"].customer_id) == set(qualifying[qualifying.gt(8000)].index)
    assert set(results["Q3"].product_id) == set(tables["products"].loc[tables["products"].price_inr.gt(tables["products"].price_inr.mean()), "product_id"])
    assert set(results["Q4"].customer_id) == set(tables["customers"].customer_id) - set(spend.index)
    assert set(results["Q5"].customer_id) == set(spend[spend.gt(spend.mean())].index)
    joined = completed.merge(tables["customers"][["customer_id", "city"]]).merge(tables["products"][["product_id", "category"]])
    cells = joined.groupby(["city", "category"]).order_amount.sum().sort_index()
    np.testing.assert_allclose(results["Q2"].set_index(["city", "category"]).revenue.sort_index(), cells, rtol=0, atol=1e-8)
    for city, rows in results["Q6"].groupby("city"):
        amounts = rows.total_spend.tolist()
        assert rows.spend_rank.tolist() == [1 + sum(v > value for v in amounts) for value in amounts]
        assert rows.dense_rank.tolist() == [1 + len({v for v in amounts if v > value}) for value in amounts]
        assert rows.row_number.tolist() == list(range(1, len(rows) + 1))
    daily = completed.assign(day=completed.order_ts.str[:10]).groupby("day").order_amount.sum()
    np.testing.assert_allclose(results["Q7"].revenue, daily, rtol=0, atol=1e-8)
    np.testing.assert_allclose(results["Q7"].cumulative_revenue, daily.cumsum(), rtol=0, atol=1e-8)
    amount = completed.order_amount
    q1, q2, q3 = amount.quantile([.25, .5, .75])
    low, high = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    corr = completed[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr) for b in list(corr)[i + 1:]]
    report = {"sql_rows": {k: len(v) for k, v in results.items()},
              "q6_leaders": results["Q6"].query("row_number == 1")[["city", "customer_id", "total_spend"]].to_dict("records"),
              "q6_rank_differences": int(results["Q6"].spend_rank.ne(results["Q6"].row_number).sum()),
              "q2_revenue": float(cells.sum()), "q7_final_revenue": float(daily.sum()),
              "eda": {"mean": float(amount.mean()), "median": float(q2), "skewness": float(amount.skew()),
                      "q1": float(q1), "q3": float(q3), "lower_fence": float(low), "upper_fence": float(high),
                      "lower_outliers": int(amount.lt(low).sum()), "upper_outliers": int(amount.gt(high).sum())},
              "largest_absolute_correlation": list(max(pairs, key=lambda row: abs(row[2]))),
              "city_aggregates": joined.groupby("city").order_amount.agg(["sum", "mean", "count"]).reset_index().to_dict("records")}
    assert report["sql_rows"] == dict(zip([f"Q{i}" for i in range(1, 8)], [30, 30, 11, 25, 550, 1175, 183]))
    assert report["q6_rank_differences"] == 8
    saved = load_json(root / "outputs/part_a/eda.json")["statistics"]
    for key, value in report["eda"].items():
        assert math.isclose(saved[key], value, abs_tol=1e-10)
    return report


def audit_ml(root, tables):
    orders = tables["orders"].copy()
    orders["date"] = pd.to_datetime(orders.order_ts)
    complete = orders.loc[orders.status.eq("completed")]
    rfm = complete.groupby("customer_id").agg(last=("date", "max"), frequency=("order_id", "nunique"), monetary=("order_amount", "sum"))
    rfm["recency"] = (pd.Timestamp("2025-10-01") - rfm.pop("last")).dt.days
    scaled = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
    km, comparison = {}, []
    for k in range(2, 9):
        km[k] = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled)
        comparison.append({"k": k, "inertia": km[k].inertia_, "silhouette": silhouette_score(scaled, km[k].labels_)})
    pca = PCA(n_components=2).fit(scaled).explained_variance_ratio_
    segments = pd.read_csv(root / "outputs/customer_segments.csv").set_index("customer_id")
    np.testing.assert_array_equal(km[4].labels_, segments.loc[rfm.index, "segment_id"])
    profiles = rfm.assign(cluster=km[4].labels_).groupby("cluster").monetary.mean()
    assert segments.loc[rfm.index, "is_premium"].tolist() == (km[4].labels_ == profiles.idxmax()).tolist()
    history = orders.loc[orders.date.lt("2025-09-01")]
    customer_info = tables["customers"].set_index("customer_id")
    category = tables["products"].set_index("product_id").category.to_dict()
    rows = []
    for cid, all_rows in history.groupby("customer_id", sort=True):
        done = all_rows.loc[all_rows.status.eq("completed")]
        if done.empty:
            continue
        count = Counter(category[p] for p in done.product_id)
        person = customer_info.loc[cid]
        rows.append({"customer_id": cid, "recency": (pd.Timestamp("2025-09-01") - done.date.max()).days,
                     "frequency": done.order_id.nunique(), "monetary": math.fsum(done.order_amount),
                     "average_discount": statistics.mean(done.discount_pct), "average_delivery": statistics.mean(done.delivery_minutes),
                     "cancellation_rate": sum(all_rows.status.eq("cancelled")) / len(all_rows),
                     "age": person.age, "tenure": (pd.Timestamp("2025-09-01") - pd.Timestamp(person.signup_date)).days,
                     "favourite_category": min(count, key=lambda key: (-count[key], key)), "city": person.city, "gender": person.gender})
    numeric_columns = ["recency", "frequency", "monetary", "average_discount", "cancellation_rate", "average_delivery", "age", "tenure"]
    category_columns = ["favourite_category", "city", "gender"]
    features = pd.DataFrame(rows).set_index("customer_id")[numeric_columns + category_columns]
    active = set(complete.loc[complete.date.ge("2025-09-01") & complete.date.lt("2025-10-01"), "customer_id"])
    y = pd.Series([int(cid not in active) for cid in features.index], index=features.index, name="lapsed")
    saved_features = pd.read_csv(root / "outputs/part_b/lapse_dataset.csv").set_index("customer_id")
    pd.testing.assert_frame_equal(features.join(y), saved_features, check_dtype=False, rtol=0, atol=1e-9)
    train, test, train_y, test_y = train_test_split(features, y, test_size=.2, stratify=y, random_state=42)
    searches, metrics = {}, {}
    for name, estimator, grid in [("logistic", LogisticRegression(max_iter=1000), {"model__C": [.01, .1, 1, 10, 100]}),
                                 ("forest", RandomForestClassifier(n_estimators=100, random_state=42),
                                  {"model__max_depth": [3, 5, 8], "model__min_samples_leaf": [5, 10, 20]})]:
        prep = ColumnTransformer([("numeric", StandardScaler(), numeric_columns),
                                  ("categorical", OneHotEncoder(drop="first", handle_unknown="ignore"), category_columns)])
        search = GridSearchCV(Pipeline([("preprocess", prep), ("model", estimator)]), grid, cv=5, scoring="f1", n_jobs=1)
        search.fit(train, train_y)
        searches[name] = search
    selected = max(searches, key=lambda name: searches[name].best_score_)
    def scores(pred):
        return {"precision": precision_score(test_y, pred, zero_division=0), "recall": recall_score(test_y, pred, zero_division=0),
                "f1": f1_score(test_y, pred, zero_division=0), "confusion_matrix": confusion_matrix(test_y, pred).tolist()}
    for name, search in searches.items():
        metrics[name] = {"best_params": search.best_params_, "cv_f1": search.best_score_, "test": scores(search.predict(test))}
    positive = train_y.loc[train_y.eq(1)].sample(n=int(train_y.eq(0).sum() - train_y.eq(1).sum()), replace=True, random_state=42).index
    balanced = clone(searches["logistic"].best_estimator_).fit(pd.concat([train, train.loc[positive]]), pd.concat([train_y, train_y.loc[positive]]))
    reloaded = load_model(root)
    np.testing.assert_array_equal(reloaded.predict(test), searches[selected].predict(test))
    np.testing.assert_allclose(reloaded.predict_proba(test), searches[selected].predict_proba(test), rtol=0, atol=1e-12)
    eligible = test.loc[test.index.isin(segments.index[segments.is_premium])]
    retention = pd.DataFrame({"customer_id": eligible.index, "lapse_probability": searches[selected].predict_proba(eligible)[:, 1],
                              "actually_lapsed": test_y.loc[eligible.index].values}).sort_values(["lapse_probability", "customer_id"], ascending=[False, True]).head(10).reset_index(drop=True)
    pd.testing.assert_frame_equal(retention, pd.read_csv(root / "outputs/retention_list.csv"), rtol=0, atol=1e-12)
    report = {"rfm_customers": len(rfm), "recency_median": rfm.recency.median(), "frequency_mean": rfm.frequency.mean(),
              "monetary_mean": rfm.monetary.mean(), "kmeans": comparison, "cluster_sizes": dict(Counter(map(int, km[4].labels_))),
              "pca": pca.tolist(), "cohort": len(features), "lapsed": int(y.sum()), "train": len(train), "test": len(test),
              "models": metrics, "selected": selected, "balanced_logistic": scores(balanced.predict(test)),
              "reload_predictions_equal": True, "retention_rows": len(retention), "retention_lapsed": int(retention.actually_lapsed.sum())}
    saved_models = load_json(root / "outputs/part_b/models.json")
    for name, values in metrics.items():
        assert values["best_params"] == saved_models["models"][name]["best_params"]
        assert math.isclose(values["cv_f1"], saved_models["models"][name]["cv_f1"], abs_tol=1e-12)
        assert values["test"]["confusion_matrix"] == saved_models["models"][name]["test"]["confusion_matrix"]
    return report


def audit_triage(root):
    from scripts.triage import run_batch, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT
    tickets = pd.read_csv(root / "data/raw/tickets.csv", keep_default_na=False)
    patterns = [(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]"), (r"\b[6-9]\d{9}\b", "[PHONE]"), (r"ORD-\d{6}", "[ORDER_ID]")]
    expected, counts = {}, [0, 0, 0]
    for row in tickets.itertuples():
        text = row.text
        for i, (pattern, replacement) in enumerate(patterns):
            text, n = re.subn(pattern, replacement, text)
            counts[i] += n
        expected[row.ticket_id] = text
    class Spy:
        def reset(self):
            self.calls, self.responses, self.failures = [], [], 0
            mock_llm.reset()
        def chat(self, messages):
            self.calls.append(deepcopy(messages))
            assert all(not re.search(pattern, message["content"]) for pattern, _ in patterns for message in messages)
            try:
                result = mock_llm.chat(messages)
            except mock_llm.TransientLLMError:
                self.failures += 1
                raise
            self.responses.append(result)
            return result
    results = {}
    for name, prompt, budget in [("zero_shot", ZERO_SHOT_PROMPT, None), ("few_shot", FEW_SHOT_PROMPT, None),
                                 ("budget_demo", ZERO_SHOT_PROMPT, 6000), ("final", FEW_SHOT_PROMPT, 100000)]:
        spy = Spy()
        frame, summary, ledger = run_batch(tickets, pd.read_csv(root / "outputs/customer_segments.csv"), prompt, budget, client=spy, sleeper=lambda _: None)
        for messages, attempt in zip(spy.calls, ledger.attempts):
            assert messages[1]["content"] == expected[attempt["ticket_id"]]
        assert summary["total_tokens"] == sum(response.usage.total_tokens for response in spy.responses)
        assert summary["transient_errors"] == spy.failures
        results[name] = {key: summary[key] for key in ["tickets", "empty", "long", "eligible", "status_counts", "classified", "correct",
                         "total_tokens", "transient_errors", "routing_counts", "redactions"]}
        saved = root / (f"outputs/part_c/{name}.csv" if name in {"zero_shot", "few_shot"} else
                        "outputs/triage_budget_demo.csv" if name == "budget_demo" else "outputs/triage_results.csv")
        pd.testing.assert_frame_equal(frame, pd.read_csv(saved, keep_default_na=False), check_dtype=False)
    assert counts == [55, 91, 132]
    return results


def run(root=ROOT):
    root = Path(root)
    if not __debug__:
        raise RuntimeError("Independent audit requires assertions")
    hashes = verify_sources(root)
    tables, a = independent_tables(root)
    a.update(audit_sql_eda(root, tables))
    print("Independent raw-data reconstruction, seven SQL queries and EDA passed", flush=True)
    b = audit_ml(root, tables)
    # The brief requires Part B to consume Part A's fixed cleaned inputs. Audit
    # the upstream temporal limitation separately; do not alter those inputs.
    from scripts.clean_data import clean_orders
    from scripts.lapse_features import build_features
    raw_orders = pd.read_csv(root / "data/raw/orders.csv")
    historical = raw_orders.loc[raw_orders.order_ts.lt("2025-09-01")].copy()
    before = build_features(tables)
    historical_clean, _ = clean_orders(historical, tables["products"])
    historical_features = build_features({**tables, "orders": historical_clean})
    actual_difference = int((before.average_delivery - historical_features.average_delivery).abs().gt(1e-12).sum())
    future = raw_orders.order_ts.ge("2025-09-01") & raw_orders.status.eq("completed")
    raw_orders.loc[future, "delivery_minutes"] = 1000
    counterfactual_clean, _ = clean_orders(raw_orders, tables["products"])
    counterfactual = build_features({**tables, "orders": counterfactual_clean})
    b["upstream_imputation_audit"] = {
        "actual_customers_differing_from_pre_cutoff_cleaning": actual_difference,
        "counterfactual_future_change_affected_customers": int((before.average_delivery - counterfactual.average_delivery).abs().gt(1e-12).sum()),
        "interpretation": "Feature aggregation excludes September; inherited Part A whole-window imputation is not temporally isolated.",
    }
    print("Independent K-means/PCA and both complete model grids passed", flush=True)
    c = audit_triage(root)
    from scripts.policy_rag import evaluate
    from scripts.agent_analysis import run as run_agents
    with tempfile.TemporaryDirectory(prefix=".part-a-test-audit-", dir=root) as temporary:
        fresh = Path(temporary)
        shutil.copytree(root / "data", fresh / "data")
        shutil.copy2(root / "mock_llm.py", fresh / "mock_llm.py")
        summary, demos = run_agents(fresh)
        assert summary == load_json(root / "outputs/part_d/agent_summary.json")
        assert demos == load_json(root / "outputs/part_d/demonstrations.json")
    d = {str(overlap): {key: value for key, value in evaluate(root, overlap).items() if key != "details"} for overlap in (10, 0)}
    d["scenarios"] = summary["scenarios"]
    report = {"source_hashes": hashes, "part_a": a, "part_b": b, "part_c": c, "part_d": d}
    save_json(root / "outputs/final_audit/independent_checks.json", report)
    print("INDEPENDENT AUDIT CHECKS PASSED", flush=True)
    return report


if __name__ == "__main__":
    run()
