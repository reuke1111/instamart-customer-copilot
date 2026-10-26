"""Audit all raw columns and the assignment's twelve data-quality metrics."""

import json
import pandas as pd

from scripts.common import ROOT, output_dir, read_tables, save_json


def audit_tables(tables):
    report = {"tables": {}}
    for name, frame in tables.items():
        report["tables"][name] = {
            "rows": len(frame),
            "columns": len(frame.columns),
            "column_audit": [
                {
                    "column": column,
                    "dtype": str(frame[column].dtype),
                    "missing_count": int(frame[column].isna().sum()),
                    "missing_pct": float(frame[column].isna().mean() * 100),
                }
                for column in frame
            ],
        }
    customers, orders = tables["customers"], tables["orders"]
    age = pd.to_numeric(customers["age"], errors="coerce")
    duration = pd.to_numeric(orders["delivery_minutes"], errors="coerce")
    report["metrics"] = {
        "raw_orders": len(orders),
        "exact_duplicates": int(orders.duplicated().sum()),
        "pending_rows": int(orders["status"].eq("pending").sum()),
        "distinct_order_ids": int(orders["order_id"].nunique()),
        "non_numeric_amounts": int(pd.to_numeric(orders["order_amount"], errors="coerce").isna().sum()),
        "negative_delivery_times": int(duration.lt(0).sum()),
        "blank_delivery_times": int(orders["delivery_minutes"].isna().sum()),
        "blank_ages": int(customers["age"].isna().sum()),
        "invalid_ages": int((age.notna() & ~age.between(10, 100)).sum()),
        "raw_city_spellings": int(customers["city"].nunique()),
        "nonblank_gender_spellings": int(customers["gender"].nunique(dropna=True)),
        "blank_genders": int(customers["gender"].isna().sum()),
    }
    return report


def run(root=ROOT):
    report = audit_tables(read_tables(root))
    save_json(output_dir(root) / "audit.json", report)
    return report


def main():
    print(json.dumps(run(), indent=2))


if __name__ == "__main__":
    main()
