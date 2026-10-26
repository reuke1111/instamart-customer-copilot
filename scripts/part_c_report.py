"""Render measured Part C results while retaining Parts A and B verbatim."""

from pathlib import Path
import pandas as pd
from scripts.common import load_json
from scripts.build_readme import table


def part_c_section(root):
    root = Path(root)
    folder = root / "outputs/part_c"
    report = load_json(folder / "triage_summary.json")
    zero, few, budget, final = (report["runs"][name] for name in ["zero_shot", "few_shot", "budget_demo", "final"])
    confusion = pd.read_csv(folder / "zero_shot_confusion.csv").set_index("actual")
    extra = few["total_tokens"] - zero["total_tokens"]
    lines = [
        "# Part C", "",
        "Completed: offline support ticket triage with input guardrails, PII redaction, "
        "retries, response validation, run-wide token budgets and Part B-aware routing. "
        "The supplied `mock_llm.py` is unchanged; no external model, account or API is used.", "",
        "## Inputs, guardrails and redaction", "",
        "`data/raw/tickets.csv` is read with `keep_default_na=False` and processed "
        "in its original row order. Empty or whitespace-only tickets are skipped; "
        "tickets exceeding `MAX_INPUT_TOKENS=400` using `mock_llm.est_tokens` are "
        "flagged before any model call. Guard decisions use original input length, "
        "so redaction cannot make a long ticket bypass the limit.", "",
        table(["Checkpoint", "Measured"], [
            ["Total tickets", zero["tickets"]], ["Empty / skipped", zero["empty"]],
            ["Long / flagged", zero["long"]], ["Tokens per long ticket", str(zero["long_ticket_tokens"])],
            ["Eligible tickets", zero["eligible"]],
            ["Email replacements", zero["redactions"]["email"]],
            ["Phone replacements", zero["redactions"]["phone"]],
            ["Order ID replacements", zero["redactions"]["order_id"]],
            ["Total replacements", zero["redactions"]["total"]],
            ["Remaining PII pattern matches", zero["redactions"]["residual_matches"]],
        ]), "",
        "Redaction applies the assignment's patterns in exactly this order: "
        "`[\\w.+-]+@[\\w-]+\\.[\\w.]+` → `[EMAIL]`; "
        "`\\b[6-9]\\d{9}\\b` → `[PHONE]`; "
        "`ORD-\\d{6}` → `[ORDER_ID]`. Counts cover all 164 input tickets once, "
        "including guarded tickets. Reapplying the patterns finds zero matches. "
        "Every user message equals the redacted ticket text alone; ground-truth labels, "
        "customer segments and identifiers are never added to prompts.", "",
        "## Prompts, retries and response validation", "",
        "Both system prompts contain CLASSIFY, define billing/delivery/technical/fraud "
        "and request JSON-only category/reason fields. The zero-shot prompt has no "
        "EXAMPLES: marker. The few-shot prompt includes labelled fraud and technical "
        "examples. Exact prompts are saved under `outputs/part_c/`, and the mock is "
        "reset immediately before each independent run.", "",
        "Each ticket permits an initial attempt plus at most three retries for "
        "`TransientLLMError`. Retry n uses `0.05 * 2**n * uniform(0.9, 1.15)` for "
        "n=0,1,2; a per-run random seed of 42 makes the jitter reproducible. One JSON "
        "log line per retry records only ticket ID, retry number and wait. Exception "
        "messages, ticket bodies and model replies are never persisted.", "",
        table(["Run", "Model attempts", "Transient failures", "Retries", "Permanent failures"], [
            [name, run["model_attempts"], run["transient_errors"], run["retry_attempts"], run["permanent_failures"]]
            for name, run in report["runs"].items()
        ]), "",
        "For each full run, all 19 transient failures recover on retry 0; every wait "
        "is between 0.045 and 0.0575 seconds. Exhausting retries would flag the ticket "
        "for human review; this does not occur with the supplied dataset.", "",
        "Valid JSON is accepted only when its category is one of the four allowed "
        "strings. Non-JSON replies use `re.findall(r'[a-z]+', text.lower())` and "
        "are accepted only when exactly one distinct allowed category occurs. "
        "Thus hyphenated category words are recognised, while valid JSON with an "
        "invalid category never gets keyword salvage.", "",
        table(["Full-run parsing outcome", "Count"], [
            ["classified_json", final["status_counts"]["classified_json"]],
            ["classified_keyword", final["status_counts"]["classified_keyword"]],
            ["flagged_invalid_output", final["status_counts"]["flagged_invalid_output"]],
            ["Successfully classified", final["classified"]],
        ]), "",
        "## Accuracy and token comparison", "",
        table(["Prompt", "Correct / classified", "Accuracy", "Total tokens", "Tokens / classified", "System prompt tokens"], [
            [name, f"{run['correct']}/{run['classified']}", f"{run['accuracy'] * 100:.2f}%",
             run["total_tokens"], f"{run['tokens_per_classified']:.4f}", run["system_prompt_tokens"]]
            for name, run in [("Zero-shot", zero), ("Few-shot", few)]
        ]), "",
        "Zero-shot confusion matrix: rows are actual categories, columns are predictions. "
        "Only the 147 classified tickets enter this matrix and the accuracy denominator.", "",
        table(["Actual / predicted", *confusion.columns], [
            [name, *[int(value) for value in row]] for name, row in confusion.iterrows()
        ]), "",
        f"The few-shot prompt raises measured accuracy from {zero['accuracy'] * 100:.2f}% "
        f"to {few['accuracy'] * 100:.2f}%, correcting 11 fraud-to-billing and 10 "
        f"technical-to-billing errors. It consumes {extra:,} extra estimated tokens "
        f"({extra / zero['total_tokens'] * 100:.2f}% more), increasing tokens per "
        f"classified ticket from {zero['tokens_per_classified']:.2f} to "
        f"{few['tokens_per_classified']:.2f}. That increase is worthwhile when missing "
        "fraud or sending app failures to billing costs more than the added processing. "
        "These are offline token estimates, not monetary charges or evidence of real-model accuracy.", "",
        "## Budget accounting", "",
        report["usage_policy"], "",
        "The ledger charges successful retries and invalid replies exactly once. "
        "The budget is shared across all tickets and checked again before every retry; "
        "empty and long tickets retain their input-guard status even after exhaustion. "
        "Failures without returned usage are counted as failures, not assigned fabricated token estimates.", "",
        table(["Budget demonstration", "Measured"], [
            ["Configured budget", budget["budget"]], ["Actual reported usage", budget["total_tokens"]],
            ["Budget-exhausted tickets", budget["status_counts"].get("budget_exhausted", 0)],
            ["Classified tickets", budget["classified"]],
            ["Invalid replies", budget["status_counts"].get("flagged_invalid_output", 0)],
            ["Retry attempts", budget["retry_attempts"]],
        ]), "",
        f"The last permitted response takes usage to {budget['total_tokens']:,}, "
        f"which is {budget['total_tokens'] - budget['budget']} above the "
        f"{budget['budget']:,} threshold; no subsequent model call is made. "
        "This implements the specified pre-call threshold check, rather than a "
        "reservation of unknown future response tokens.", "",
        "## Final routing and artifacts", "",
        f"The final independent few-shot run uses a {final['budget']:,}-token budget, "
        f"consumes {final['total_tokens']:,} tokens, and produces {final['tickets']} rows. "
        "Guarded, invalid, exhausted or permanently failed tickets go to human_review/NORMAL. "
        "Otherwise fraud takes precedence over Premium and goes to security_team/HIGH; "
        "other Premium tickets go to premium_desk/HIGH; remaining tickets go to "
        "support_queue/NORMAL. Unknown customers are not Premium; unavailable category "
        "and segment names remain blank.", "",
        table(["Final route", "Tickets"], final["routing_counts"].items()), "",
        table(["Priority", "Tickets"], final["priority_counts"].items()), "",
        "The 135 JSON and 12 keyword classifications passed rule-based output checks; "
        "this is not human validation. Human review receives 9 invalid replies, 4 empty "
        "tickets and 4 long tickets. Premium membership is read from the unchanged "
        "Part B segment file.", "",
        "- Required outputs: `outputs/triage_results.csv` and `outputs/triage_budget_demo.csv`.",
        "- Supporting outputs: `outputs/part_c/triage_summary.json`, zero/few-shot results, "
        "zero-shot confusion CSV, exact system prompts, per-attempt token ledgers, "
        "retry JSONL logs and Parts A/B preservation hashes.",
        "- Implementation: `scripts/triage.py`, `scripts/triage_analysis.py`, "
        "`scripts/run_part_c.py`, `scripts/validate_part_c.py`, "
        "`scripts/part_c_integrity.py` and `scripts/part_c_report.py`.",
        "- Tests: `tests/test_part_c.py` plus an extended fresh-folder A-through-C "
        "reproduction test. All earlier numerical assertions are retained.", "",
        "## Privacy safeguards and limitations", "",
        "The new Part C outputs contain no ticket bodies, model reply bodies, original "
        "email addresses, phone numbers or order IDs. Only the required synthetic ticket "
        "and customer identifiers remain for routing; these are not full anonymisation. "
        "Regex assertions scan every prompt and all new Part C artifacts. The fixed "
        "patterns cover this assignment's planted formats, not arbitrary real-world PII. "
        "The classifier is a deterministic keyword mock, so its few-shot gains do not "
        "establish production accuracy. The Part B Premium flag uses the whole-window "
        "segment and is illustrative for this synthetic routing exercise.", "",
    ]
    return "\n".join(lines)


def extend_readme(base, root):
    root = Path(root)
    base = base.replace(
        "15. `python -m unittest discover -s tests -v`",
        "15. `python -m scripts.run_part_c`\n"
        "16. `python scripts/validate_part_c.py`\n"
        "17. `python scripts/validate_checkpoints.py`\n"
        "18. `python scripts/validate_part_b.py`\n"
        "19. `python -m unittest discover -s tests -v`",
    ).replace("steps 3–15", "steps 3–19")
    base = base.replace(
        "then `python -m scripts.run_part_b` and `python -m unittest discover -s tests -v`.",
        "then `python -m scripts.run_part_b`, `python -m scripts.run_part_c` and "
        "`python -m unittest discover -s tests -v`.",
    ).replace(
        "For an existing validated Part A checkout, run only Part B; it validates Part A read-only "
        "and verifies that its data and analytical artifacts are unchanged.",
        "For an existing validated Parts A/B checkout, run only Part C; it validates "
        "both earlier Parts read-only and verifies that their artifacts are unchanged.",
    )
    if (root / "outputs/part_c/triage_summary.json").exists():
        base = base.replace(
            "Parts C and D are not implemented.",
            "Part C adds PII-safe offline ticket triage. Part D is not implemented.",
        )
        start, end = base.index("# Part C\n"), base.index("# Part D\n")
        base = base[:start] + part_c_section(root) + "\n" + base[end:]
        base = base.replace(
            "# Part D\n\nNot yet implemented. Requires separate approval after Part A.",
            "# Part D\n\nNot yet implemented. Requires separate approval after Part C.",
        )
        base = base.replace(
            "Review Parts A and B before approving Part C or D. "
            "Ticket triage, a policy assistant and a refund agent are not implemented.",
            "Parts A, B and C are complete. Part C regex redaction covers the specified "
            "synthetic PII formats; the mock and token estimator are not production evaluations. "
            "Review Part C before approving Part D. A policy assistant and refund agent are not implemented.",
        )
    return base
