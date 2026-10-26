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


def read_tables(root=ROOT, clean=False):
    folder = Path(root) / "data" / ("clean" if clean else "raw")
    # pandas default NA handling includes literal N/A in order_amount.
    return {name: pd.read_csv(folder / f"{name}.csv") for name in TABLES}


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
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
