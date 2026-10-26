"""Read-only validation of Part C checkpoints, privacy and token ledgers."""

from pathlib import Path
import json
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from scripts.build_readme import render
from scripts.common import ROOT, load_json, verify_sources
from scripts.part_c_integrity import assert_preserved
from scripts.triage import (
    CATEGORIES, CLASSIFIED, OUTPUT_COLUMNS, FEW_SHOT_PROMPT, ZERO_SHOT_PROMPT,
    assert_no_pii, prepare_tickets, routing, run_batch, segment_lookup,
)
from scripts.validate_checkpoints import same_frame

FULL_STATUSES = {
    "classified_json": 135, "classified_keyword": 12,
    "flagged_invalid_output": 9, "flagged_long": 4, "skipped_empty": 4,
}


def validate(root=ROOT):
    if not __debug__:
        raise RuntimeError("Part C validation requires assertions; do not use -O")
    root = Path(root)
    verify_sources(root)
    assert_preserved(root)
    folder = root / "outputs/part_c"
    tickets = pd.read_csv(root / "data/raw/tickets.csv", keep_default_na=False)
    segments = pd.read_csv(root / "outputs/customer_segments.csv")
    prepared, counts = prepare_tickets(tickets)
    assert len(prepared) == 164
    assert counts == {"email": 55, "phone": 91, "order_id": 132, "total": 278, "residual_matches": 0}
    guarded_ids = {row["ticket_id"] for row in prepared if row["guard_status"] is not None}
    assert len(guarded_ids) == 8
    report = load_json(folder / "triage_summary.json")
    budget = report["runs"]["budget_demo"]["budget"]
    assert budget in {6000, 5000}
    lookup = segment_lookup(segments)
    specifications = [
        ("zero_shot", ZERO_SHOT_PROMPT, None, folder / "zero_shot.csv"),
        ("few_shot", FEW_SHOT_PROMPT, None, folder / "few_shot.csv"),
        ("budget_demo", ZERO_SHOT_PROMPT, budget, root / "outputs/triage_budget_demo.csv"),
        ("final", FEW_SHOT_PROMPT, 100000, root / "outputs/triage_results.csv"),
    ]
    for name, prompt, limit, path in specifications:
        expected, summary, ledger = run_batch(tickets, segments, prompt, limit, sleeper=lambda _: None)
        assert summary == report["runs"][name], f"Stale {name} summary"
        actual = pd.read_csv(path, keep_default_na=False)
        assert actual.columns.tolist() == OUTPUT_COLUMNS
        assert len(actual) == 164
        assert actual["ticket_id"].tolist() == tickets["ticket_id"].tolist()
        assert actual["customer_id"].tolist() == tickets["customer_id"].tolist()
        assert actual["ticket_id"].is_unique
        same_frame(actual, expected)
        assert_no_pii(path.read_text(encoding="utf-8"))
        assert summary["empty"] == 4 and summary["long"] == 4 and summary["eligible"] == 156
        assert summary["long_ticket_tokens"] == [734] * 4
        assert summary["redactions"] == counts
        if name != "budget_demo":
            assert summary["status_counts"] == FULL_STATUSES
            assert summary["classified"] == 147
            assert summary["correct"] == (126 if name == "zero_shot" else 147)
            assert summary["accuracy"] == (126 / 147 if name == "zero_shot" else 1.0)
            assert summary["transient_errors"] == 19
            assert summary["retry_attempts"] == 19
            assert summary["model_attempts"] == 175
            assert summary["successful_responses"] == 156
        assert summary["permanent_failures"] == 0
        attempts = pd.read_csv(folder / f"{name}_attempts.csv")
        same_frame(attempts, pd.DataFrame(ledger.attempts))
        assert not set(attempts["ticket_id"]) & guarded_ids
        assert (attempts["prompt_tokens"] + attempts["completion_tokens"]).equals(attempts["total_tokens"])
        np.testing.assert_array_equal(attempts["total_tokens"].cumsum(), attempts["running_total"])
        assert int(attempts["total_tokens"].sum()) == summary["total_tokens"]
        assert int(actual["tokens_used"].sum()) == summary["total_tokens"]
        assert attempts.loc[attempts["outcome"].eq("transient_error"), "total_tokens"].eq(0).all()
        assert not attempts.loc[attempts["outcome"].eq("transient_error"), "usage_reported"].any()
        if limit is not None:
            # A response may cross the threshold; no subsequent attempt may start.
            assert (attempts["running_total"] - attempts["total_tokens"]).lt(limit).all()
        retry_lines = (folder / f"{name}_retries.jsonl").read_text(encoding="utf-8").splitlines()
        retries = [json.loads(line) for line in retry_lines]
        assert retries == ledger.retries
        assert len(retries) == summary["retry_attempts"]
        for entry in retries:
            assert set(entry) == {"ticket_id", "retry_number", "wait_seconds"}
            assert entry["retry_number"] == 0
            assert 0.045 <= entry["wait_seconds"] <= 0.0575
        for row in actual.itertuples(index=False):
            segment, premium = lookup.get(row.customer_id, ("", False))
            assert row.segment_name == segment
            assert (row.route, row.priority) == routing(row.status, row.category, premium)
            if row.status not in CLASSIFIED:
                assert row.category == ""
            if row.status in {"skipped_empty", "flagged_long", "budget_exhausted"}:
                assert row.tokens_used == 0
        classified = actual.loc[actual["status"].isin(CLASSIFIED)]
        true = tickets.set_index("ticket_id")["true_category"]
        assert int(classified["category"].eq(classified["ticket_id"].map(true)).sum()) == summary["correct"]
        assert summary["tokens_per_classified"] == summary["total_tokens"] / summary["classified"]
    assert report["runs"]["budget_demo"]["status_counts"]["budget_exhausted"] > 0
    assert report["runs"]["final"]["total_tokens"] < 100000
    assert report["runs"]["final"]["routing_counts"] == {
        "human_review": 17, "premium_desk": 11, "security_team": 34, "support_queue": 102,
    }
    expected_confusion = pd.DataFrame(
        [[38, 0, 0, 0], [0, 36, 0, 0], [10, 0, 29, 0], [11, 0, 0, 23]],
        index=pd.Index(CATEGORIES, name="actual"), columns=CATEGORIES,
    )
    same_frame(pd.read_csv(folder / "zero_shot_confusion.csv"), expected_confusion.reset_index())
    for name, prompt in [("zero_shot", ZERO_SHOT_PROMPT), ("few_shot", FEW_SHOT_PROMPT)]:
        assert (folder / f"{name}_prompt.txt").read_text(encoding="utf-8") == prompt.rstrip("\n") + "\n"
        assert_no_pii(prompt)
    assert "EXAMPLES:" not in ZERO_SHOT_PROMPT
    assert "\nEXAMPLES:\n" in FEW_SHOT_PROMPT
    # Scan every new Part C artifact, including logs and preservation metadata.
    for path in folder.iterdir():
        if path.is_file():
            assert_no_pii(path.read_text(encoding="utf-8"))
    assert (root / "README.md").read_text(encoding="utf-8") == render(root), "README is stale"


def main():
    validate()
    print("ALL PART C CHECKPOINTS PASSED")


if __name__ == "__main__":
    main()
