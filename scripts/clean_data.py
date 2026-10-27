"""Clean in the prescribed order, account for changes, and persist typed data."""

import json
from contextlib import closing
from pathlib import Path
import sqlite3
from itertools import zip_longest

import numpy as np
import pandas as pd

from scripts.common import ROOT, TIME_FORMAT, output_dir, read_tables, save_json


def changed(before, after):
    """Count unequal values while considering two missing values equal."""
    equal = before.eq(after).fillna(False) | (before.isna() & after.isna())
    return int((~equal).sum())


def clean_customers(raw):
    frame = raw.copy(deep=True)
    log = []
    city = frame["city"].str.strip().str.title()
    log.append({"rule": "Strip and title-case city", "changed": changed(frame["city"], city)})
    frame["city"] = city
    if city.isna().any() or city.eq("").any():
        raise ValueError("Customer cities must not be missing")
    gender = frame["gender"].str.strip().str.lower().map(
        {"m": "male", "male": "male", "f": "female", "female": "female"}
    ).fillna("unknown")
    log.append({"rule": "Map gender to male/female/unknown", "changed": changed(frame["gender"], gender)})
    frame["gender"] = gender
    age = pd.to_numeric(frame["age"], errors="raise")
    invalid = age.notna() & ~age.between(10, 100)
    log.append({"rule": "Replace ages outside 10-100 with missing", "changed": int(invalid.sum())})
    frame["age"] = age.mask(invalid)
    missing = frame["age"].isna()
    medians = frame.groupby("city")["age"].transform("median")
    frame["age"] = frame["age"].fillna(medians)
    if frame["age"].isna().any():
        raise ValueError("Cannot impute ages: a city has no valid age observations")
    log.append({"rule": "Impute missing ages with city median", "changed": int(missing.sum())})
    frame["signup_date"] = pd.to_datetime(frame["signup_date"], format="%Y-%m-%d", errors="raise")
    if frame["signup_date"].isna().any():
        raise ValueError("Signup dates must not be missing")
    log.append({"rule": "Parse signup_date to datetime", "changed": int(frame["signup_date"].notna().sum())})
    if frame["customer_id"].isna().any() or frame["customer_id"].duplicated().any():
        raise ValueError("Customer identifiers must be nonmissing and unique")
    return frame, log


def clean_orders(raw, products):
    frame = raw.copy(deep=True)
    log = []
    if frame[["order_id", "customer_id", "product_id"]].isna().any().any():
        raise ValueError("Order, customer and product identifiers must not be missing")
    frame["order_ts"] = pd.to_datetime(frame["order_ts"], format=TIME_FORMAT, errors="raise")
    if frame["order_ts"].isna().any():
        raise ValueError("Order timestamps must not be missing")
    log.append({"rule": "Parse order_ts to datetime", "changed": len(frame)})
    before = len(frame)
    frame = frame.drop_duplicates().copy()
    log.append({"rule": "Remove exact duplicate rows", "changed": before - len(frame)})
    # A tie between different snapshots has no well-defined latest record.
    if frame.duplicated(["order_id", "order_ts"]).any():
        raise ValueError("Conflicting snapshots share an order ID and timestamp")
    before = len(frame)
    frame = frame.sort_values("order_ts", kind="stable").drop_duplicates("order_id", keep="last").copy()
    log.append({"rule": "Keep latest snapshot per order ID", "changed": before - len(frame)})
    if not frame["status"].isin(["completed", "cancelled"]).all():
        raise ValueError("Latest snapshots must be completed or cancelled")
    if products["product_id"].duplicated().any():
        raise ValueError("Product identifiers must be unique")
    product_lookup = products.set_index("product_id")
    if not frame["product_id"].isin(product_lookup.index).all():
        raise ValueError("Order refers to an unknown product")
    original = frame["order_amount"]
    amount = pd.to_numeric(original, errors="coerce")
    log.append({
        "rule": "Convert order_amount to numeric",
        "changed": int(original.notna().sum()),
        "new_missing": int((original.notna() & amount.isna()).sum()),
        "already_missing": int(original.isna().sum()),
        "invalid_total": int(amount.isna().sum()),
    })
    missing = amount.isna()
    recovered = (
        frame["product_id"].map(product_lookup["price_inr"])
        * frame["quantity"] * (1 - frame["discount_pct"] / 100)
    ).round(2)
    frame["order_amount"] = amount.fillna(recovered)
    log.append({"rule": "Recover amounts from price x quantity x (1-discount/100)", "changed": int(missing.sum())})
    frame["delivery_minutes"] = pd.to_numeric(frame["delivery_minutes"], errors="raise")
    negative = frame["delivery_minutes"].lt(0)
    frame.loc[negative, "delivery_minutes"] = np.nan
    log.append({"rule": "Replace negative delivery durations with missing", "changed": int(negative.sum())})
    completed = frame["status"].eq("completed")
    category = frame["product_id"].map(product_lookup["category"])
    category_medians = frame.loc[completed, "delivery_minutes"].groupby(category.loc[completed]).transform("median")
    missing = completed & frame["delivery_minutes"].isna()
    frame.loc[completed, "delivery_minutes"] = frame.loc[completed, "delivery_minutes"].fillna(category_medians)
    if frame.loc[completed, "delivery_minutes"].isna().any():
        raise ValueError("Cannot impute delivery: a completed category has no valid durations")
    log.append({"rule": "Impute completed durations with completed category median", "changed": int(missing.sum())})
    cancelled = frame["status"].eq("cancelled")
    cleared = int(frame.loc[cancelled, "delivery_minutes"].notna().sum())
    frame.loc[cancelled, "delivery_minutes"] = np.nan
    log.append({
        "rule": "Keep cancelled delivery durations missing",
        "changed": cleared,
        "left_missing": int(cancelled.sum()),
    })
    return frame.sort_values("order_id").reset_index(drop=True), log


def cleaning_summary(customers, orders):
    completed = orders["status"].eq("completed")
    cancelled = orders["status"].eq("cancelled")
    return {
        "customers": len(customers),
        "city_counts": {str(k): int(v) for k, v in customers["city"].value_counts().items()},
        "gender_counts": {str(k): int(v) for k, v in customers["gender"].value_counts().items()},
        "missing_ages": int(customers["age"].isna().sum()),
        "min_age": float(customers["age"].min()),
        "max_age": float(customers["age"].max()),
        "orders": len(orders),
        "completed": int(completed.sum()),
        "cancelled": int(cancelled.sum()),
        "pending": int(orders["status"].eq("pending").sum()),
        "missing_amounts": int(orders["order_amount"].isna().sum()),
        "missing_completed_delivery": int(orders.loc[completed, "delivery_minutes"].isna().sum()),
        "missing_cancelled_delivery": int(orders.loc[cancelled, "delivery_minutes"].isna().sum()),
        "completed_revenue": round(float(orders.loc[completed, "order_amount"].sum()), 2),
    }


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE customers (
    customer_id TEXT PRIMARY KEY NOT NULL,
    city TEXT NOT NULL,
    signup_date TEXT NOT NULL,
    age REAL NOT NULL CHECK(age BETWEEN 10 AND 100),
    gender TEXT NOT NULL CHECK(gender IN ('male', 'female', 'unknown'))
);
CREATE TABLE products (
    product_id TEXT PRIMARY KEY NOT NULL,
    product_name TEXT NOT NULL,
    category TEXT NOT NULL,
    price_inr REAL NOT NULL CHECK(price_inr >= 0)
);
CREATE TABLE orders (
    order_id TEXT PRIMARY KEY NOT NULL,
    customer_id TEXT NOT NULL REFERENCES customers(customer_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK(quantity > 0),
    order_ts TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('completed', 'cancelled')),
    discount_pct REAL NOT NULL CHECK(discount_pct BETWEEN 0 AND 100),
    order_amount REAL NOT NULL CHECK(order_amount >= 0),
    delivery_minutes REAL,
    CHECK((status = 'cancelled' AND delivery_minutes IS NULL)
       OR (status = 'completed' AND delivery_minutes IS NOT NULL AND delivery_minutes >= 0))
);
CREATE INDEX orders_customer_status ON orders(customer_id, status);
CREATE INDEX orders_timestamp ON orders(order_ts);
"""


def same_database(first, second):
    """Compare full SQL schema and contents, not SQLite's writer-version header."""
    if not first.exists():
        return False
    try:
        with closing(sqlite3.connect(first.resolve().as_uri() + "?mode=ro", uri=True)) as left, \
             closing(sqlite3.connect(second.resolve().as_uri() + "?mode=ro", uri=True)) as right:
            return all(a == b for a, b in zip_longest(left.iterdump(), right.iterdump()))
    except sqlite3.DatabaseError:
        return False


def persist(tables, root=ROOT):
    root = Path(root)
    clean_dir = root / "data" / "clean"
    clean_dir.mkdir(parents=True, exist_ok=True)
    serialized = {name: frame.copy() for name, frame in tables.items()}
    serialized["customers"]["signup_date"] = pd.to_datetime(
        serialized["customers"]["signup_date"]
    ).dt.strftime(TIME_FORMAT)
    serialized["orders"]["order_ts"] = pd.to_datetime(
        serialized["orders"]["order_ts"]
    ).dt.strftime(TIME_FORMAT)
    temporary_db = root / "data" / "instamart.db.tmp"
    # The temporary file belongs only to this stage; the previous database
    # remains intact until a complete replacement is ready.
    temporary_db.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary_db)) as connection:
            connection.executescript(SCHEMA)
            for name, frame in serialized.items():
                frame.to_sql(name, connection, if_exists="append", index=False)
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        for name, frame in serialized.items():
            frame.to_csv(clean_dir / f"{name}.csv", index=False, lineterminator="\n")
        database = root / "data" / "instamart.db"
        # Keep valid checkpoints' byte fingerprint on no-op rebuilds, even
        # when Python ships another SQLite version. Changed data/schema still
        # replaces the database and is rejected by existing checkpoint guards.
        if not same_database(database, temporary_db):
            temporary_db.replace(database)
    finally:
        temporary_db.unlink(missing_ok=True)


def run(root=ROOT):
    raw = read_tables(root)
    customers, customer_log = clean_customers(raw["customers"])
    orders, order_log = clean_orders(raw["orders"], raw["products"])
    tables = {"customers": customers, "products": raw["products"].copy(), "orders": orders}
    persist(tables, root)
    report = {
        "customer_rules": customer_log,
        "order_rules": order_log,
        "summary": cleaning_summary(customers, orders),
        "city_age_medians": {str(k): float(v) for k, v in customers.groupby("city")["age"].median().items()},
        "category_delivery_medians": {
            str(k): float(v) for k, v in orders.loc[orders["status"].eq("completed")].merge(
                raw["products"][["product_id", "category"]], on="product_id", validate="many_to_one"
            ).groupby("category")["delivery_minutes"].median().items()
        },
    }
    save_json(output_dir(root) / "cleaning.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
