"""Run the seven scenarios and separate safety/recovery demonstrations."""

import csv
from pathlib import Path

from scripts.agent_storage import atomic_json, read_jsonl, write_jsonl
from scripts.common import ROOT, load_json
from scripts.refund_agent import SimulatedCrash, resume_agent, run_agent


def run(root=ROOT):
    root = Path(root)
    with (root / "data/raw/agent_scenarios.csv").open(encoding="utf-8", newline="") as stream:
        scenarios = list(csv.DictReader(stream))
    results = [
        run_agent(row["message"], row["scenario_id"], root=root, run_id=f"scenario-{row['scenario_id']}")
        for row in scenarios
    ]
    # Exactly seven normal rows, replaced atomically rather than appended again
    # on repeat pipeline runs. Safety demos are stored separately.
    write_jsonl(root / "outputs/agent_runs.jsonl", results)
    s1 = scenarios[0]
    token = run_agent(s1["message"], "S1", token_limit=200, root=root, run_id="demo-token-S1")
    timeout = run_agent(s1["message"], "S1", time_limit_seconds=0, root=root, run_id="demo-timeout-S1")
    run_id = "demo-crash-S1"
    evidence_path = root / "outputs/part_d/crash_checkpoint.json"
    checkpoint = root / "state" / f"{run_id}.json"
    if not checkpoint.exists() or load_json(checkpoint)["execution_state"] != "finished":
        try:
            run_agent(s1["message"], "S1", root=root, run_id=run_id, crash_after_tools=2)
        except SimulatedCrash:
            saved = load_json(checkpoint)
            assert saved["step_number"] == 2
            assert saved["tools_executed"] == ["get_order", "search_policy"]
            atomic_json(evidence_path, saved)
        resumed = resume_agent(run_id, root=root)
    else:
        resumed = resume_agent(run_id, root=root)
    saved = load_json(evidence_path)
    ledger_before = (root / "outputs/refund_ledger.jsonl").read_bytes()
    repeated = resume_agent(run_id, root=root)
    assert repeated == resumed
    assert (root / "outputs/refund_ledger.jsonl").read_bytes() == ledger_before
    sequence = [step["tool"] for step in resumed["steps"]]
    assert sequence == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND" and resumed["refund_inr"] == 128.25
    ledger = read_jsonl(root / "outputs/refund_ledger.jsonl")
    assert sum(event["run_id"] == run_id for event in ledger) == 1
    demonstrations = {
        "token_limit": token, "timeout": timeout,
        "crash_resume": {
            "run_id": run_id, "checkpointed_tools": saved["tools_executed"],
            "resumed_tools": sequence[saved["step_number"]:],
            "combined_tools": sequence, "result": resumed,
            "ledger_entries_for_run": sum(event["run_id"] == run_id for event in ledger),
            "repeat_resume_unchanged": repeated == resumed,
        },
    }
    atomic_json(root / "outputs/part_d/demonstrations.json", demonstrations)
    summary = {
        "scenarios": [{
            "scenario_id": row["scenario_id"], "run_id": row["run_id"], "status": row["status"],
            "refund_inr": row["refund_inr"], "llm_calls": row["llm_calls"],
            "tokens_used": row["tokens_used"], "tools": [step["tool"] for step in row["steps"]],
        } for row in results],
        "ledger": {
            "events": len(ledger),
            "normal_events": sum(event["run_id"].startswith("scenario-") for event in ledger),
            "crash_demo_events": sum(event["run_id"] == run_id for event in ledger),
            "issued_events": sum(event["action"] == "issue_refund" for event in ledger),
            "escalated_events": sum(event["action"] == "escalate_to_human" for event in ledger),
            "issued_inr": round(sum(event["refund_inr"] for event in ledger if event["action"] == "issue_refund"), 2),
            "escalated_inr": round(sum(event["refund_inr"] for event in ledger if event["action"] == "escalate_to_human"), 2),
            "unique_action_ids": len({event["action_id"] for event in ledger}),
        },
    }
    atomic_json(root / "outputs/part_d/agent_summary.json", summary)
    return summary, demonstrations


if __name__ == "__main__":
    summary, demonstrations = run()
    for row in summary["scenarios"]:
        print(row)
    print("Token demo:", demonstrations["token_limit"]["status"])
    print("Timeout demo:", demonstrations["timeout"]["status"])
    print("Crash/resume:", demonstrations["crash_resume"]["combined_tools"])
