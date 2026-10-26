"""Render Part D from measured artifacts without changing Parts A-C sections."""

from pathlib import Path
from scripts.build_readme import table
from scripts.common import load_json


def part_d_section(root):
    folder = Path(root) / "outputs/part_d"
    overlap, zero = [load_json(folder / f"rag_overlap_{n}.json") for n in (10, 0)]
    summary = load_json(folder / "agent_summary.json")
    demos = load_json(folder / "demonstrations.json")
    example = load_json(folder / "retrieval_example.json")
    token, timeout, crash = [demos[key] for key in ("token_limit", "timeout", "crash_resume")]
    ledger = summary["ledger"]
    failures = [row for row in zero["details"] if row["answerable"] and not row["fact_correct"]]
    lines = [
        "# Part D", "",
        "Completed: local policy retrieval and a bounded ReAct refund agent using the unchanged "
        "`mock_llm.py`, SQLite, scikit-learn and standard-library persistence. No external API "
        "or real payment service is used.", "",
        "## Policy corpus, chunks and retrieval", "",
        "The five synthetic policies cover refunds, delivery, escalation, payment and privacy. "
        "Whitespace splitting uses 40-word chunks with 10-word overlap (step 30), stopping "
        "as soon as a chunk reaches the document end. Files sort by filename; chunk IDs "
        "are `filename:000`, `filename:001`, etc. Word offsets are zero-based and end-exclusive.", "",
        table(["Source", "Overlap 10", "Overlap 0"], [
            [name, count, zero["chunk_counts"][name]] for name, count in overlap["chunk_counts"].items()
        ] + [["Total", overlap["chunks"], zero["chunks"]]]), "",
        "`TfidfVectorizer()` is fitted to chunk text only. Cosine similarity ranks chunks; "
        "ties sort by chunk ID. `retrieve(query, k=3)` returns source, score and text. "
        "There are no query-to-document mappings. For `refund policy late delivery`:", "",
        table(["Rank", "Source", "Cosine score", "Opening words"], [
            [index, row["source"], f"{row['score']:.9f}", " ".join(row["text"].split()[:12])]
            for index, row in enumerate(example, 1)
        ]), "",
        "`answer(question)` sends exactly three filename-labelled chunks under `CONTEXT:`, "
        "then the original question under `QUESTION:`; its system prompt contains POLICY ASSISTANT. "
        "Only the supplied mock answers. Returned text must be present in the concatenated "
        "retrieved context, or the answer becomes an unsupported-information fallback. "
        "Evaluation references are used only for scoring, never supplied to the model.", "",
        "## Measured RAG evaluation", "",
        table(["Overlap", "Questions", "Source retrieval", "Expected facts", "Unsupported fallbacks", "Tokens"], [
            [report["overlap"], report["questions"], f"{report['source_correct']}/{report['answerable']}",
             f"{report['fact_correct']}/{report['answerable']}",
             f"{report['unsupported_fallbacks']}/{report['unanswerable']}", report["total_tokens"]]
            for report in (overlap, zero)
        ]), "",
        "Without overlap, fact coverage falls by two questions (16.67 percentage points). "
        "The measured failures are:", "",
        table(["Question", "Missing expected fact", "Actual answer"], [
            [row["question"], row["expected_fact"], " ".join(row["answer"].split())] for row in failures
        ]), "",
        "In refund_policy.txt, the 40-word boundary separates the late-delivery condition "
        "from its refund consequence. In payment_policy.txt it separates 'returned automatically' "
        "from 'within 48 hours'. Retrieval still finds the right documents, but isolated "
        "sentence fragments lose the required facts; concatenation can even join unrelated "
        "fragments from different documents. The ten repeated words restore complete facts "
        "for this corpus. The two unsupported questions fall back in both configurations.", "",
        "## Tools and execution rules", "",
        table(["Tool", "Parameters", "Behavior"], [
            ["get_order", "order_id", "Read-only, parameterized SQLite lookup; four fields or a structured missing-order error."],
            ["search_policy", "query", "Reuse TF-IDF retrieval; return three chunks."],
            ["calculate_refund", "order_amount, delivery_minutes", "Verify inputs against the authorized order; return full amount only after 45 minutes."],
            ["issue_refund", "order_id, refund_inr", "Record eligible amount up to and including INR 1,000."],
            ["escalate_to_human", "order_id, refund_inr", "Record eligible amount above INR 1,000 for approval; do not issue it."],
        ]), "",
        "`TOOLS_SPEC` includes descriptions and object schemas with required fields and no "
        "additional properties; `TOOL_MAP` is an explicit allowlist. Runtime validation rejects "
        "unknown tools, malformed arguments, booleans masquerading as numbers, nonfinite or "
        "negative values, fractional paise and order IDs outside the original request. "
        "Each financial execution independently re-reads the completed order, its delivery "
        "duration and exact amount. Decimal checks enforce the amount and approval boundary.", "",
        "Missing orders yield NOT_FOUND; cancelled orders and completed deliveries at or below "
        "45 minutes are denied. Completed deliveries strictly above 45 minutes receive the "
        "full amount through the appropriate route. As instructed, all supplied scenarios "
        "are considered within the seven-day policy window; date eligibility is not inferred "
        "from today's date or enforced for arbitrary production requests.", "",
        "## Seven normal scenarios", "",
        table(["Scenario", "Status", "Refund / escalation INR", "LLM calls", "Tokens", "Tool sequence"], [
            [row["scenario_id"], row["status"], f"{row['refund_inr']:.2f}", row["llm_calls"],
             row["tokens_used"], " → ".join(row["tools"])] for row in summary["scenarios"]
        ]), "",
        "Exactly seven normal results are saved in `outputs/agent_runs.jsonl`. Each contains "
        "status, final answer, full tool steps/observations, model calls, tokens and identifiers. "
        "The default limits are six model calls, 6,000 reported tokens and ten seconds. "
        "Monotonic time is checked at the top of each loop. Every response is charged before "
        "its tools execute; usage strictly above the limit prevents those tools. Model final "
        "claims are checked against actual observations and recorded actions.", "",
        "## Fallback and interruption demonstrations", "",
        table(["Demonstration", "Status", "LLM calls", "Tokens", "Completed tools"], [
            ["S1: token limit 200", token["status"], token["llm_calls"], token["tokens_used"], len(token["steps"])],
            ["S1: timeout 0 seconds", timeout["status"], timeout["llm_calls"], timeout["tokens_used"], len(timeout["steps"])],
            ["S7: default iteration limit", summary["scenarios"][-1]["status"],
             summary["scenarios"][-1]["llm_calls"], summary["scenarios"][-1]["tokens_used"],
             len(summary["scenarios"][-1]["tools"])],
        ]), "",
        "The token demonstration stops before calculate_refund; its violating response is "
        "counted, but its tool does not run. A post-response budget cannot prevent that "
        "response's token overshoot. Every fallback supplies a human-handoff summary with "
        "completed steps, calls and usage. Timeout zero invokes neither model nor tool.", "",
        "Checkpoints under `state/` preserve full messages, tool-call IDs, finalized observations, "
        "counters, limits, elapsed active time, source hashes, pending calls and execution state. "
        "A pending intent is saved before execution and every observation is saved immediately "
        "after completion. Atomic replacement and OS file locks prevent partial writes and "
        "concurrent writers. Resume retains elapsed active time while excluding process downtime; "
        "changed source inputs or repurposed run IDs are rejected.", "",
        f"Crash simulation saves {' → '.join(crash['checkpointed_tools'])}, then raises an "
        f"interruption. Resume executes {' → '.join(crash['resumed_tools'])}. The combined "
        f"sequence is {' → '.join(crash['combined_tools'])}; final status is "
        f"{crash['result']['status']} for INR {crash['result']['refund_inr']:.2f}. "
        "No completed tool repeats, and repeat resume returns the saved result without "
        "another model call or ledger action. A separate test interrupts immediately after "
        "the financial write but before its observation: replay reconciles the pending call "
        "using the existing action ID, preventing a duplicate ledger event.", "",
        f"The ledger has {ledger['events']} unique actions: {ledger['normal_events']} normal "
        f"scenario events and {ledger['crash_demo_events']} separate crash-demo event. "
        f"There are {ledger['issued_events']} simulated issued refunds totalling INR "
        f"{ledger['issued_inr']:.2f} (including the separate INR 128.25 crash demo), and "
        f"{ledger['escalated_events']} escalation totalling INR {ledger['escalated_inr']:.2f}. "
        "Normal S1/S2 refunds total INR 263.25. Each entry has an actual UTC timestamp, action, "
        "order, amount, scenario, run and idempotency key. Keys use run/action/order; "
        "replays retain the original timestamp. Cross-run order deduplication is outside this "
        "simulation: the crash demo deliberately uses a different run ID for S1.", "",
        "## ReAct versus plan-and-execute (under 250 words)", "",
        "ReAct observes each tool result before deciding the next action. Here S1 needs "
        "five model calls and 894 tokens; a cancelled order (S5) stops after three calls "
        "and 359 tokens, while a missing order (S6) stops after two calls and 112 tokens. "
        "These measurements show useful early exits, but repeated histories add overhead. "
        "The looping S7 consumes six calls and 2,781 tokens before the iteration guard stops it.", "",
        "Plan-and-execute would create a plan first and then run its steps, potentially reducing "
        "repeated model decisions. Unexpected missing or cancelled orders require explicit "
        "branches or replanning; a rigid plan must never continue to a refund after either "
        "observation. That architecture was not implemented, so no comparative token or call "
        "savings are claimed. Both approaches need validated tools, authoritative order checks, "
        "budgets, human escalation, durable checkpoints and idempotent actions. ReAct resumes "
        "the saved conversation and pending action here; a planner would need to persist plan "
        "position and observations as well. ReAct is preferable for this assignment's short, "
        "observation-dependent workflow and supplied mock. A deterministic execution plan "
        "could be attractive for a stable production process once its branches are proven.", "",
        "## Artifacts and reproducibility", "",
        "- `scripts/policy_rag.py`: chunking, retrieval, answering and both evaluations.",
        "- `scripts/refund_tools.py`, `refund_agent.py`, `agent_storage.py`: tool boundaries, "
        "agent loop and durable storage; `agent_analysis.py`: scenarios and demonstrations.",
        "- `scripts/run_part_d.py`, `validate_part_d.py`, `part_d_integrity.py`, "
        "`part_d_report.py`: orchestration, validation, A-C preservation and this report.",
        "- `outputs/part_d/`: chunks, retrieval example, per-question evaluations, summary, "
        "demonstrations, interrupted checkpoint evidence and A-C preservation hashes.",
        "- `outputs/agent_runs.jsonl`, `outputs/refund_ledger.jsonl`, `state/*.json`: "
        "results, simulated financial audit events and recoverable conversations.",
        "- `tests/test_part_d.py` and the extended source-only A-through-D reproduction test.", "",
        "Rerunning fixed run IDs returns finished checkpoints and preserves ledger bytes; "
        "exactly seven normal results remain. Clean reproduction uses a new source-only "
        "folder and independently executes all four parts. It compares analytical artifacts, "
        "agent decisions, counts and README; real timestamps and elapsed checkpoint durations "
        "are excluded from cross-run equality. No earlier A-C data or outputs are modified.", "",
    ]
    return "\n".join(lines)


def extend_readme(base, root):
    base = base.replace(
        "19. `python -m unittest discover -s tests -v`",
        "19. `python -m scripts.run_part_d`\n20. `python scripts/validate_part_d.py`\n"
        "21. `python scripts/validate_checkpoints.py`\n22. `python scripts/validate_part_b.py`\n"
        "23. `python scripts/validate_part_c.py`\n24. `python -m unittest discover -s tests -v`",
    ).replace("steps 3–19", "steps 3–24")
    base = base.replace(
        "`python -m scripts.run_part_c` and `python -m unittest discover -s tests -v`.",
        "`python -m scripts.run_part_c`, `python -m scripts.run_part_d` and "
        "`python -m unittest discover -s tests -v`.",
    ).replace(
        "For an existing validated Parts A/B checkout, run only Part C; it validates "
        "both earlier Parts read-only and verifies that their artifacts are unchanged.",
        "For an existing validated Parts A/B/C checkout, run only Part D; it validates "
        "all earlier Parts read-only and verifies that their artifacts are unchanged.",
    )
    if (Path(root) / "outputs/part_d/agent_summary.json").exists():
        base = base.replace("Part D is not implemented.", "Part D adds policy RAG and a resumable refund agent.")
        start, end = base.index("# Part D\n"), base.index("# Limitations and Next Steps\n")
        base = base[:start] + part_d_section(root) + "\n" + base[end:]
        base = base.replace(
            "Review Part C before approving Part D. A policy assistant and refund agent are not implemented.",
            "Part D is complete. TF-IDF matches vocabulary rather than meaning; an extractive "
            "context check does not prove semantic correctness, especially across chunk boundaries. "
            "Synthetic policies and a deterministic mock do not establish real-model accuracy or "
            "eliminate hallucination risk. Production refunds require authenticated customer ownership, "
            "observed delivery evidence, date eligibility and payment authorization. Local run/action "
            "idempotency needs a transactional, globally deduplicated payment ledger in production; "
            "human escalation must connect to an authenticated approval workflow. Checkpoint conversations "
            "and audit logs contain synthetic order IDs and requests; production use needs access controls, "
            "encryption, retention limits, PII minimization and tamper-evident audit storage. "
            "The time guard checks between operations and cannot interrupt a blocking model or tool call. "
            "No real payments or external messages are sent. Review the completed capstone before publication.",
        )
    return base
