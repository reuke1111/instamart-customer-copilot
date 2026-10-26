"""Four independent mock runs with privacy-safe results and usage audit trails."""

import json
from pathlib import Path
import time
import pandas as pd

from scripts.common import ROOT, save_json, verify_sources
from scripts.triage import (
    ATTEMPT_COLUMNS, CATEGORIES, FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT, assert_no_pii, run_batch,
)

RUN_NAMES = ["zero_shot", "few_shot", "budget_demo", "final"]


def run(root=ROOT, *, sleeper=time.sleep):
    root = Path(root)
    verify_sources(root)
    tickets = pd.read_csv(root / "data/raw/tickets.csv", keep_default_na=False)
    segments = pd.read_csv(root / "outputs/customer_segments.csv")
    folder = root / "outputs/part_c"
    folder.mkdir(parents=True, exist_ok=True)
    report = {"runs": {}, "usage_policy": (
        "Sum actual response usage across every attempt, including successful retries and invalid outputs. "
        "The supplied mock's transient exceptions carry no usage, so they add zero reported tokens. "
        "Check the cumulative total before each call; the last response can cross the threshold."
    )}
    configurations = [
        ("zero_shot", ZERO_SHOT_PROMPT, None),
        ("few_shot", FEW_SHOT_PROMPT, None),
        ("budget_demo", ZERO_SHOT_PROMPT, 6000),
        ("final", FEW_SHOT_PROMPT, 100000),
    ]
    for name, prompt, budget in configurations:
        results, summary, ledger = run_batch(tickets, segments, prompt, budget, sleeper=sleeper)
        if name == "budget_demo" and not results["status"].eq("budget_exhausted").any():
            results, summary, ledger = run_batch(tickets, segments, prompt, 5000, sleeper=sleeper)
        if name == "budget_demo":
            assert results["status"].eq("budget_exhausted").any(), "Budget demo did not exhaust"
        output = {
            "budget_demo": root / "outputs/triage_budget_demo.csv",
            "final": root / "outputs/triage_results.csv",
        }.get(name, folder / f"{name}.csv")
        results.to_csv(output, index=False, lineterminator="\n")
        pd.DataFrame(ledger.attempts, columns=ATTEMPT_COLUMNS).to_csv(
            folder / f"{name}_attempts.csv", index=False, lineterminator="\n",
        )
        # Exactly one PII-free JSON line for each retry, with no wall-clock
        # timestamp or ticket content, so replay remains reproducible.
        retry_log = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ledger.retries)
        assert_no_pii(retry_log)
        (folder / f"{name}_retries.jsonl").write_text(retry_log, encoding="utf-8", newline="\n")
        report["runs"][name] = summary
    zero = report["runs"]["zero_shot"]
    pd.DataFrame(zero["confusion_matrix"], index=CATEGORIES, columns=CATEGORIES).rename_axis("actual").to_csv(
        folder / "zero_shot_confusion.csv", lineterminator="\n",
    )
    for name, prompt in [("zero_shot", ZERO_SHOT_PROMPT), ("few_shot", FEW_SHOT_PROMPT)]:
        assert_no_pii(prompt)
        (folder / f"{name}_prompt.txt").write_text(prompt.rstrip("\n") + "\n", encoding="utf-8", newline="\n")
    assert_no_pii(json.dumps(report))
    save_json(folder / "triage_summary.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({name: {
        key: values[key] for key in ["classified", "accuracy", "total_tokens", "transient_errors", "routing_counts"]
    } for name, values in result["runs"].items()}, indent=2))
