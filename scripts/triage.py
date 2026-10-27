"""Part C: guarded, redacted, budgeted calls to the supplied offline mock."""

from collections import Counter
from dataclasses import dataclass, field
import json
import random
import re
import time

import pandas as pd
import mock_llm

EMAIL = r"[\w.+-]+@[\w-]+\.[\w.]+"
PHONE = r"\b[6-9]\d{9}\b"
ORDER = r"ORD-\d{6}"
REDACTIONS = [
    ("email", re.compile(EMAIL), "[EMAIL]"),
    ("phone", re.compile(PHONE), "[PHONE]"),
    ("order_id", re.compile(ORDER), "[ORDER_ID]"),
]
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
CLASSIFIED = {"classified_json", "classified_keyword"}
MAX_INPUT_TOKENS = 400
MAX_RETRIES = 3
BASE_DELAY = 0.05
OUTPUT_COLUMNS = [
    "ticket_id", "customer_id", "status", "category",
    "segment_name", "priority", "route", "tokens_used",
]
ATTEMPT_COLUMNS = [
    "ticket_id", "attempt", "outcome", "usage_reported",
    "prompt_tokens", "completion_tokens", "total_tokens", "running_total",
]
ZERO_SHOT_PROMPT = (
    'CLASSIFY the support ticket. Categories: billing (charges or invoices), '
    'delivery (arrival or rider issues), technical (app or login problems), '
    'fraud (unauthorized account activity). Reply only with JSON containing '
    '"category" and "reason".'
)
FEW_SHOT_PROMPT = ZERO_SHOT_PROMPT + """
EXAMPLES:
Ticket: Someone hacked my account and charged my wallet.
Label: {"category": "fraud", "reason": "unauthorized account activity"}
Ticket: The app freezes on the payment page.
Label: {"category": "technical", "reason": "app failure"}
"""


def assert_no_pii(text):
    assert not any(pattern.search(text) for _, pattern, _ in REDACTIONS), "PII pattern remained in protected content"


def redact(text):
    counts = {}
    for name, pattern, replacement in REDACTIONS:
        text, counts[name] = pattern.subn(replacement, text)
    assert_no_pii(text)
    return text, counts


def guard(text):
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return None


def prepare_tickets(tickets):
    required = {"ticket_id", "customer_id", "text", "true_category"}
    if not required <= set(tickets):
        raise ValueError("Ticket columns are incomplete")
    if not tickets["ticket_id"].is_unique:
        raise ValueError("Ticket identifiers must be unique")
    prepared, counts = [], Counter()
    for row in tickets.itertuples(index=False):
        # Only these synthetic identifiers may appear in metadata and retry logs.
        if not re.fullmatch(r"T\d+", row.ticket_id) or not re.fullmatch(r"C\d+", row.customer_id):
            raise ValueError("Unexpected synthetic identifier format")
        guarded = guard(row.text)
        redacted, substitutions = redact(row.text)
        counts.update(substitutions)
        prepared.append({
            "ticket_id": row.ticket_id, "customer_id": row.customer_id,
            "redacted_text": redacted, "guard_status": guarded,
            "input_tokens": mock_llm.est_tokens(row.text),
            "true_category": row.true_category,
        })
    return prepared, {**dict(counts), "total": sum(counts.values()), "residual_matches": 0}


def parse_response(text):
    if not isinstance(text, str):
        return "flagged_invalid_output", ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        allowed = set(re.findall(r"[a-z]+", text.lower())) & set(CATEGORIES)
        if len(allowed) == 1:
            return "classified_keyword", allowed.pop()
        return "flagged_invalid_output", ""
    if isinstance(parsed, dict) and isinstance(parsed.get("category"), str) and parsed["category"] in CATEGORIES:
        return "classified_json", parsed["category"]
    # Valid JSON with an invalid category never gets keyword salvage.
    return "flagged_invalid_output", ""


def routing(status, category, premium):
    if status not in CLASSIFIED:
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def segment_lookup(segments):
    if not segments["customer_id"].is_unique:
        raise ValueError("Segment customer identifiers must be unique")
    lookup = {}
    for row in segments.itertuples(index=False):
        if not isinstance(row.is_premium, bool):
            raise ValueError("Premium membership must contain booleans, not truthy strings")
        assert_no_pii(str(row.segment_name))
        lookup[row.customer_id] = (str(row.segment_name), row.is_premium)
    return lookup


@dataclass
class TokenLedger:
    limit: int | None = None
    total: int = 0
    attempts: list = field(default_factory=list)
    retries: list = field(default_factory=list)

    def __post_init__(self):
        if self.limit is not None and (type(self.limit) is not int or self.limit < 0):
            raise ValueError("Token budget must be a nonnegative integer or None")

    @property
    def exhausted(self):
        return self.limit is not None and self.total >= self.limit

    def record(self, ticket_id, attempt, outcome, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            if outcome == "response":
                raise ValueError("Successful response omitted usage; stop rather than undercount")
            prompt, completion, total = 0, 0, 0
        else:
            prompt = getattr(usage, "prompt_tokens", None)
            completion = getattr(usage, "completion_tokens", None)
            total = getattr(usage, "total_tokens", None)
            if any(type(value) is not int or value < 0 for value in (prompt, completion, total)) or prompt + completion != total:
                raise ValueError("Invalid reported token usage; stop rather than undercount")
        self.total += total
        self.attempts.append({
            "ticket_id": ticket_id, "attempt": attempt, "outcome": outcome,
            "usage_reported": usage is not None, "prompt_tokens": prompt,
            "completion_tokens": completion, "total_tokens": total, "running_total": self.total,
        })


def call_with_retries(ticket_id, messages, ledger, client, rng, sleeper):
    for message in messages:
        assert_no_pii(message["content"])
    for attempt in range(MAX_RETRIES + 1):
        # The same run-wide ledger is checked again before every retry.
        if ledger.exhausted:
            return "budget_exhausted", None
        try:
            response = client.chat(messages)
        except mock_llm.TransientLLMError as error:
            # Never log exception text; it could contain echoed input.
            ledger.record(ticket_id, attempt, "transient_error", error)
            if ledger.exhausted:
                return "budget_exhausted", None
            if attempt == MAX_RETRIES:
                return "flagged_permanent_error", None
            delay = BASE_DELAY * (2 ** attempt) * rng.uniform(0.9, 1.15)
            ledger.retries.append({
                "ticket_id": ticket_id, "retry_number": attempt, "wait_seconds": delay,
            })
            sleeper(delay)
            continue
        ledger.record(ticket_id, attempt, "response", response)
        return "response", response
    raise AssertionError("Retry loop should always return")


def run_batch(tickets, segments, system_prompt, token_budget=None, *, client=None, sleeper=time.sleep):
    if not __debug__:
        raise RuntimeError("Triage requires privacy assertions; do not use -O")
    client = mock_llm if client is None else client
    prepared, redaction_counts = prepare_tickets(tickets)
    lookup = segment_lookup(segments)
    assert_no_pii(system_prompt)
    if "CLASSIFY" not in system_prompt or not all(category in system_prompt for category in CATEGORIES):
        raise ValueError("Classification system prompt is incomplete")
    ledger = TokenLedger(limit=token_budget)
    rng = random.Random(42)
    rows, correct, classified, confusion = [], 0, 0, Counter()
    client.reset()
    for ticket in prepared:
        before = ledger.total
        status, category = ticket["guard_status"], ""
        if status is None:
            # No IDs, labels, segments or wrapper text enter the user message.
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ticket["redacted_text"]},
            ]
            assert messages[1]["content"] == ticket["redacted_text"]
            status, response = call_with_retries(
                ticket["ticket_id"], messages, ledger, client, rng, sleeper,
            )
            if status == "response":
                try:
                    content = response.choices[0].message.content
                except (AttributeError, IndexError, TypeError):
                    content = None
                status, category = parse_response(content)
        name, premium = lookup.get(ticket["customer_id"], ("", False))
        route, priority = routing(status, category, premium)
        rows.append({
            "ticket_id": ticket["ticket_id"], "customer_id": ticket["customer_id"],
            "status": status, "category": category, "segment_name": name,
            "priority": priority, "route": route, "tokens_used": ledger.total - before,
        })
        if status in CLASSIFIED:
            classified += 1
            correct += int(category == ticket["true_category"])
            confusion[(ticket["true_category"], category)] += 1
    results = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # No raw text, response reason or original order identifier is exported.
    assert_no_pii(results.to_csv(index=False))
    assert results["ticket_id"].tolist() == tickets["ticket_id"].tolist()
    assert int(results["tokens_used"].sum()) == ledger.total
    summary = {
        "tickets": len(tickets),
        "empty": sum(ticket["guard_status"] == "skipped_empty" for ticket in prepared),
        "long": sum(ticket["guard_status"] == "flagged_long" for ticket in prepared),
        "eligible": sum(ticket["guard_status"] is None for ticket in prepared),
        "long_ticket_tokens": [ticket["input_tokens"] for ticket in prepared if ticket["guard_status"] == "flagged_long"],
        "redactions": redaction_counts, "budget": token_budget,
        "status_counts": {str(k): int(v) for k, v in results["status"].value_counts().sort_index().items()},
        "classified": classified, "correct": correct,
        "accuracy": correct / classified if classified else None,
        "total_tokens": ledger.total,
        "tokens_per_classified": ledger.total / classified if classified else None,
        "system_prompt_tokens": mock_llm.est_tokens(system_prompt),
        "model_attempts": len(ledger.attempts),
        "successful_responses": sum(row["outcome"] == "response" for row in ledger.attempts),
        "transient_errors": sum(row["outcome"] == "transient_error" for row in ledger.attempts),
        "retry_attempts": len(ledger.retries),
        "permanent_failures": int(results["status"].eq("flagged_permanent_error").sum()),
        "routing_counts": {str(k): int(v) for k, v in results["route"].value_counts().sort_index().items()},
        "priority_counts": {str(k): int(v) for k, v in results["priority"].value_counts().sort_index().items()},
        "confusion_matrix": [[confusion[(actual, predicted)] for predicted in CATEGORIES] for actual in CATEGORIES],
        "confusion_labels": CATEGORIES,
    }
    return results, summary, ledger
