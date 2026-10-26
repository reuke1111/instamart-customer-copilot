"""Read-only validation of measured RAG, agent, recovery and preservation evidence."""

from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.agent_storage import read_jsonl
from scripts.build_readme import render
from scripts.common import ROOT, load_json, verify_sources
from scripts.part_d_integrity import assert_preserved
from scripts.policy_rag import PolicyRetriever, evaluate
from scripts.refund_tools import TOOLS_SPEC, TOOL_MAP, get_order, money

COUNTS = {"delivery_policy.txt": 3, "escalation_policy.txt": 3,
          "payment_policy.txt": 3, "privacy_policy.txt": 2, "refund_policy.txt": 4}
EXPECTED = [
    ("S1", "REFUND", 128.25), ("S2", "REFUND", 135.0), ("S3", "ESCALATE", 1999.0),
    ("S4", "DENIED", 0.0), ("S5", "DENIED", 0.0), ("S6", "NOT_FOUND", 0.0),
    ("S7", "FALLBACK_MAX_ITERATIONS", 0.0),
]
REFUND_SEQUENCE = ["get_order", "search_policy", "calculate_refund", "issue_refund"]


def validate_result(row):
    assert set(row) == {"run_id", "scenario_id", "status", "final_answer", "steps",
                        "llm_calls", "tokens_used", "refund_inr"}
    assert isinstance(row["final_answer"], str) and row["final_answer"]
    assert isinstance(row["llm_calls"], int) and 0 <= row["llm_calls"] <= 6
    assert isinstance(row["tokens_used"], int) and row["tokens_used"] >= 0
    ids = set()
    for step in row["steps"]:
        assert set(step) == {"tool_call_id", "tool", "arguments", "observation"}
        assert step["tool"] in TOOL_MAP and step["tool_call_id"] not in ids
        ids.add(step["tool_call_id"])
        assert isinstance(step["arguments"], dict) and isinstance(step["observation"], dict)


def validate(root=ROOT):
    if not __debug__:
        raise RuntimeError("Part D validation requires assertions; do not use -O")
    root = Path(root)
    verify_sources(root)
    assert_preserved(root)
    folder = root / "outputs/part_d"
    for overlap in (10, 0):
        report = load_json(folder / f"rag_overlap_{overlap}.json")
        assert report == evaluate(root, overlap), "Saved RAG results differ from fresh evaluation"
        retriever = PolicyRetriever(root, overlap)
        assert load_json(folder / f"chunks_overlap_{overlap}.json") == retriever.chunks
        assert report["questions"] == 14 and report["answerable"] == 12
        assert report["source_correct"] == 12
        assert report["unanswerable"] == report["unsupported_fallbacks"] == 2
        if overlap == 10:
            assert report["chunk_counts"] == COUNTS and report["chunks"] == 15
            assert report["fact_correct"] == 12
        else:
            assert report["fact_correct"] < 12, "Expected an actual measured boundary failure"
    example = load_json(folder / "retrieval_example.json")
    assert example == PolicyRetriever(root).retrieve("refund policy late delivery")
    assert len(example) == 3 and all(row["source"] == "refund_policy.txt" for row in example)
    assert len(TOOLS_SPEC) == len(TOOL_MAP) == 5
    assert {item["name"] for item in TOOLS_SPEC} == set(TOOL_MAP)

    results = read_jsonl(root / "outputs/agent_runs.jsonl")
    assert len(results) == 7
    assert [(row["scenario_id"], row["status"], row["refund_inr"]) for row in results] == EXPECTED
    summary = load_json(folder / "agent_summary.json")
    assert summary["scenarios"] == [{
        **{key: row[key] for key in ("scenario_id", "run_id", "status", "refund_inr", "llm_calls", "tokens_used")},
        "tools": [step["tool"] for step in row["steps"]],
    } for row in results]
    for row in results:
        validate_result(row)
        assert row["tokens_used"] <= 6000
        state = load_json(root / "state" / f"{row['run_id']}.json")
        assert state["execution_state"] == "finished" and state["result"] == row
        assert state["step_number"] == len(row["steps"])
        assert state["tools_executed"] == [step["tool"] for step in row["steps"]]
        observations = [message for message in state["messages"] if message["role"] == "tool"]
        assert len(observations) == len(row["steps"])
        for message, step in zip(observations, row["steps"]):
            assert message["tool_call_id"] == step["tool_call_id"] and message["name"] == step["tool"]
            assert json.loads(message["content"]) == step["observation"]
    assert results[0]["llm_calls"] == 5
    assert [step["tool"] for step in results[0]["steps"]] == REFUND_SEQUENCE
    assert results[-1]["llm_calls"] == 6
    assert [step["tool"] for step in results[-1]["steps"]] == ["search_policy"] * 6

    demos = load_json(folder / "demonstrations.json")
    token, timeout, crash = [demos[key] for key in ("token_limit", "timeout", "crash_resume")]
    for row in (token, timeout, crash["result"]):
        validate_result(row)
    assert token["status"] == "FALLBACK_TOKEN_LIMIT" and token["tokens_used"] > 200
    assert [step["tool"] for step in token["steps"]] == REFUND_SEQUENCE[:2]
    assert timeout["status"] == "FALLBACK_TIMEOUT"
    assert timeout["llm_calls"] == timeout["tokens_used"] == len(timeout["steps"]) == 0
    assert crash["checkpointed_tools"] == REFUND_SEQUENCE[:2]
    assert crash["resumed_tools"] == REFUND_SEQUENCE[2:] and crash["combined_tools"] == REFUND_SEQUENCE
    assert crash["result"]["status"] == "REFUND" and crash["result"]["refund_inr"] == 128.25
    assert crash["ledger_entries_for_run"] == 1 and crash["repeat_resume_unchanged"]
    interrupted = load_json(folder / "crash_checkpoint.json")
    assert interrupted["execution_state"] == "running" and interrupted["step_number"] == 2
    assert interrupted["tools_executed"] == REFUND_SEQUENCE[:2] and interrupted["llm_calls"] == 2
    assert len([message for message in interrupted["messages"] if message["role"] == "tool"]) == 2
    assert interrupted["steps"] == crash["result"]["steps"][:2]

    ledger = read_jsonl(root / "outputs/refund_ledger.jsonl")
    assert len(ledger) == len({event["action_id"] for event in ledger}) == 4
    assert {event["run_id"] for event in ledger} == {"scenario-S1", "scenario-S2", "scenario-S3", "demo-crash-S1"}
    for event in ledger:
        assert set(event) == {"timestamp", "action_id", "run_id", "scenario_id", "action", "tool_name", "order_id", "refund_inr"}
        assert datetime.fromisoformat(event["timestamp"]).utcoffset() == timedelta(0)
        assert event["action_id"] == f"{event['run_id']}:{event['action']}:{event['order_id']}"
        assert event["tool_name"] == event["action"]
        order = get_order(event["order_id"], root=root)
        assert order["status"] == "completed" and order["delivery_minutes"] > 45
        assert money(event["refund_inr"]) == money(order["order_amount"])
        assert (event["action"] == "issue_refund") == (event["refund_inr"] <= 1000)
    assert summary["ledger"] == {
        "events": 4, "normal_events": 3, "crash_demo_events": 1,
        "issued_events": 3, "escalated_events": 1, "issued_inr": 391.5,
        "escalated_inr": 1999.0, "unique_action_ids": 4,
    }
    assert (root / "README.md").read_text(encoding="utf-8") == render(root)


if __name__ == "__main__":
    validate()
    print("ALL PART D CHECKPOINTS PASSED")
