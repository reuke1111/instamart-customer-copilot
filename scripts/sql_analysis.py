"""Execute the seven saved SQL statements with pandas and retain all rows."""

import json
from pathlib import Path
import re

import pandas as pd

from scripts.common import ROOT, connect_readonly, output_dir, save_json


def load_queries(root=ROOT):
    text = (Path(root) / "sql" / "queries.sql").read_text(encoding="utf-8")
    parts = re.split(r"(?m)^-- (Q[1-7]):[^\n]*\n", text)
    queries = {parts[index]: parts[index + 1].strip() for index in range(1, len(parts), 2)}
    assert list(queries) == [f"Q{i}" for i in range(1, 8)], "Expected exactly Q1-Q7"
    return queries


def execute_queries(root=ROOT):
    with connect_readonly(root) as connection:
        return {
            name: pd.read_sql_query(query, connection)
            for name, query in load_queries(root).items()
        }


def summarize(results):
    q2, q5, q6, q7 = (results[name] for name in ("Q2", "Q5", "Q6", "Q7"))
    return {
        "row_counts": {name: len(frame) for name, frame in results.items()},
        "q2_total": round(float(q2["revenue"].sum()), 2),
        "q2_highest": q2.iloc[0].to_dict(),
        "q5_average": float(q5["average_spend"].iloc[0]),
        "q6_leaders": q6.loc[q6["spend_rank"].eq(1)].to_dict("records"),
        "q6_rank_row_number_differences": int(q6["spend_rank"].ne(q6["row_number"]).sum()),
        "q7_final_total": float(q7["cumulative_revenue"].iloc[-1]),
    }


def run(root=ROOT):
    results = execute_queries(root)
    folder = output_dir(root)
    for name, frame in results.items():
        frame.to_csv(folder / f"{name.lower()}.csv", index=False, lineterminator="\n")
    report = summarize(results)
    save_json(folder / "sql_summary.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
