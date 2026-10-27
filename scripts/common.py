"""Paths, serialization, and source integrity shared by Part A stages."""

import hashlib
import json
from contextlib import contextmanager, closing
from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ("customers", "products", "orders")
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
CSV_DTYPES = {
    "customers": {"customer_id": "object", "city": "object", "signup_date": "object",
                  "age": "float64", "gender": "object"},
    "products": {"product_id": "object", "product_name": "object", "category": "object",
                 "price_inr": "int64"},
    "orders": {"order_id": "object", "customer_id": "object", "product_id": "object",
               "quantity": "int64", "order_ts": "object", "status": "object",
               "discount_pct": "int64", "order_amount": "object", "delivery_minutes": "float64"},
}


def read_tables(root=ROOT, clean=False):
    folder = Path(root) / "data" / ("clean" if clean else "raw")
    # pandas default NA handling includes literal N/A in order_amount.
    tables = {}
    for name in TABLES:
        dtype = CSV_DTYPES[name].copy()
        if clean and name == "orders":
            dtype["order_amount"] = "float64"
        # Explicit object text preserves NumPy NaN semantics on pandas 2 and 3.
        # Round-trip parsing preserves the decimal serialization of clean floats.
        tables[name] = pd.read_csv(folder / f"{name}.csv", dtype=dtype, float_precision="round_trip")
    return tables


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify_sources(root=ROOT):
    root = Path(root)
    expected = load_json(root / "source_hashes.json")
    assert set(expected) == {"generate_data.py", "mock_llm.py"}
    for name, digest in expected.items():
        actual = hashlib.sha256((root / name).read_bytes()).hexdigest()
        assert actual == digest, f"Supplied source changed: {name}"
    return expected


@contextmanager
def connect_readonly(root=ROOT):
    database = (Path(root) / "data" / "instamart.db").resolve()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        yield connection


def output_dir(root=ROOT):
    path = Path(root) / "outputs" / "part_a"
    path.mkdir(parents=True, exist_ok=True)
    return path
