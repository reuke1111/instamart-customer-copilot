"""Guardrail, privacy, retry, budget, parsing, routing and artifact tests."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import pandas as pd
import mock_llm

from scripts.common import ROOT, load_json
from scripts.part_c_integrity import assert_preserved
from scripts.triage import (
    BASE_DELAY, MAX_INPUT_TOKENS, MAX_RETRIES, EMAIL, PHONE, ORDER, REDACTIONS,
    OUTPUT_COLUMNS, ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT, TokenLedger,
    assert_no_pii, call_with_retries, guard, parse_response, prepare_tickets,
    redact, routing, run_batch,
)
from scripts.validate_part_c import validate


def response(content='{"category": "billing", "reason": "payment issue"}', tokens=10):
    return NS(
        choices=[NS(message=NS(content=content))],
        usage=NS(prompt_tokens=tokens - 2, completion_tokens=2, total_tokens=tokens),
    )


def segments():
    return pd.DataFrame({
        "customer_id": ["C1"], "segment_name": ["Premium"], "is_premium": [True],
    })


def tickets(texts, customer="C1"):
    return pd.DataFrame({
        "ticket_id": [f"T{index + 1}" for index in range(len(texts))],
        "customer_id": [customer] * len(texts),
        "text": texts, "true_category": ["billing"] * len(texts),
    })


class SpyClient:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1
        self.messages, self.responses = [], []
        self.failures = 0
        mock_llm.reset()

    def chat(self, messages):
        self.messages.append(deepcopy(messages))
        try:
            result = mock_llm.chat(messages)
        except mock_llm.TransientLLMError:
            self.failures += 1
            raise
        self.responses.append(result)
        return result


class InputAndPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/raw/tickets.csv", keep_default_na=False)

    def test_counts_long_lengths_and_redaction_totals(self):
        prepared, counts = prepare_tickets(self.raw)
        self.assertEqual(len(prepared), 164)
        self.assertEqual(sum(item["guard_status"] == "skipped_empty" for item in prepared), 4)
        self.assertEqual(sum(item["guard_status"] == "flagged_long" for item in prepared), 4)
        self.assertEqual(sum(item["guard_status"] is None for item in prepared), 156)
        self.assertEqual([item["input_tokens"] for item in prepared if item["guard_status"] == "flagged_long"], [734] * 4)
        self.assertEqual(counts, {"email": 55, "phone": 91, "order_id": 132, "total": 278, "residual_matches": 0})
        for item in prepared:
            assert_no_pii(item["redacted_text"])

    def test_exact_regex_order_and_idempotent_replacement(self):
        self.assertEqual(EMAIL, r"[\w.+-]+@[\w-]+\.[\w.]+")
        self.assertEqual(PHONE, r"\b[6-9]\d{9}\b")
        self.assertEqual(ORDER, r"ORD-\d{6}")
        self.assertEqual([name for name, _, _ in REDACTIONS], ["email", "phone", "order_id"])
        cleaned, counts = redact("person@example.invalid 9123456789 ORD-123456")
        self.assertEqual(cleaned, "[EMAIL] [PHONE] [ORDER_ID]")
        self.assertEqual(counts, {"email": 1, "phone": 1, "order_id": 1})
        again, counts_again = redact(cleaned)
        self.assertEqual(again, cleaned)
        self.assertEqual(sum(counts_again.values()), 0)

    def test_guard_threshold_and_whitespace(self):
        self.assertEqual(MAX_INPUT_TOKENS, 400)
        self.assertEqual(guard(" \t\n"), "skipped_empty")
        self.assertEqual(mock_llm.est_tokens("word " * 307), 400)
        self.assertIsNone(guard("word " * 307))
        self.assertEqual(guard("word " * 308), "flagged_long")
        self.assertIsNone(guard("N/A"))

    def test_empty_and_long_never_call_model_even_at_zero_budget(self):
        client = NS(chat=Mock(side_effect=AssertionError("Guarded ticket reached model")), reset=Mock())
        result, summary, ledger = run_batch(
            tickets([" \n", "word " * 308]), segments(), ZERO_SHOT_PROMPT,
            token_budget=0, client=client, sleeper=lambda _: None,
        )
        self.assertEqual(result["status"].tolist(), ["skipped_empty", "flagged_long"])
        self.assertTrue(result["tokens_used"].eq(0).all())
        client.chat.assert_not_called()
        client.reset.assert_called_once()
        self.assertEqual(ledger.total, 0)

    def test_user_prompts_are_exact_redacted_text_and_no_metadata(self):
        client = SpyClient()
        segment_table = pd.read_csv(ROOT / "outputs/customer_segments.csv")
        result, summary, ledger = run_batch(
            self.raw, segment_table, FEW_SHOT_PROMPT, client=client, sleeper=lambda _: None,
        )
        prepared, _ = prepare_tickets(self.raw)
        by_id = {item["ticket_id"]: item for item in prepared}
        self.assertEqual(client.reset_count, 1)
        self.assertEqual(len(client.messages), len(ledger.attempts))
        for messages, attempt in zip(client.messages, ledger.attempts):
            expected = by_id[attempt["ticket_id"]]
            self.assertIsNone(expected["guard_status"])
            self.assertEqual(messages, [
                {"role": "system", "content": FEW_SHOT_PROMPT},
                {"role": "user", "content": expected["redacted_text"]},
            ])
            for message in messages:
                assert_no_pii(message["content"])
        self.assertEqual(summary["total_tokens"], sum(item.usage.total_tokens for item in client.responses))
        self.assertEqual(summary["transient_errors"], client.failures)
        self.assertEqual(result["ticket_id"].tolist(), self.raw["ticket_id"].tolist())

    def test_shuffled_input_order_is_preserved(self):
        shuffled = self.raw.sample(frac=1, random_state=17)
        result, _, _ = run_batch(
            shuffled, pd.read_csv(ROOT / "outputs/customer_segments.csv"),
            ZERO_SHOT_PROMPT, sleeper=lambda _: None,
        )
        self.assertEqual(result["ticket_id"].tolist(), shuffled["ticket_id"].tolist())

    def test_every_new_artifact_has_zero_pii_pattern_matches(self):
        paths = list((ROOT / "outputs/part_c").iterdir()) + [
            ROOT / "outputs/triage_results.csv", ROOT / "outputs/triage_budget_demo.csv",
        ]
        for path in paths:
            if path.is_file():
                with self.subTest(artifact=path.name):
                    assert_no_pii(path.read_text(encoding="utf-8"))


class ResponseAndRoutingTests(unittest.TestCase):
    def test_json_category_validation_without_keyword_salvage(self):
        self.assertEqual(parse_response('{"category": "fraud", "reason": "security"}'), ("classified_json", "fraud"))
        self.assertEqual(parse_response('{"category": "billing"}'), ("classified_json", "billing"))
        for content in [
            '{"category": "shipping", "reason": "billing fraud"}',
            '{"category": ["billing"]}', '{"category": null}', '["billing"]', '"billing"', 'null',
            '{"category": "Billing"}', '{"reason": "billing"}',
        ]:
            with self.subTest(content=content):
                self.assertEqual(parse_response(content), ("flagged_invalid_output", ""))

    def test_hyphenated_keyword_fallback_and_ambiguity(self):
        self.assertEqual(
            parse_response("The customer appears to have a billing-related problem."),
            ("classified_keyword", "billing"),
        )
        self.assertEqual(parse_response("FRAUD fraud"), ("classified_keyword", "fraud"))
        for content in ["billing or fraud", "shipping-related", "", None]:
            self.assertEqual(parse_response(content), ("flagged_invalid_output", ""))

    def test_routing_precedence_and_all_human_statuses(self):
        for status in [
            "skipped_empty", "flagged_long", "flagged_invalid_output",
            "budget_exhausted", "flagged_permanent_error",
        ]:
            self.assertEqual(routing(status, "fraud", True), ("human_review", "NORMAL"))
        self.assertEqual(routing("classified_json", "fraud", True), ("security_team", "HIGH"))
        self.assertEqual(routing("classified_keyword", "fraud", False), ("security_team", "HIGH"))
        self.assertEqual(routing("classified_json", "billing", True), ("premium_desk", "HIGH"))
        self.assertEqual(routing("classified_json", "billing", False), ("support_queue", "NORMAL"))

    def test_unknown_customer_is_not_premium(self):
        client = NS(chat=Mock(return_value=response()), reset=Mock())
        result, _, _ = run_batch(
            tickets(["charged twice"], customer="C9999"), segments(),
            ZERO_SHOT_PROMPT, client=client, sleeper=lambda _: None,
        )
        self.assertEqual(result.loc[0, "segment_name"], "")
        self.assertEqual(result.loc[0, "route"], "support_queue")
        self.assertEqual(result.loc[0, "priority"], "NORMAL")

    def test_response_reason_and_exception_text_are_not_exported(self):
        private = "person@example.invalid 9123456789 ORD-123456"
        error = mock_llm.TransientLLMError(private)
        content = json.dumps({"category": "billing", "reason": private})
        client = NS(chat=Mock(side_effect=[error, response(content)]), reset=Mock())
        results, summary, ledger = run_batch(
            tickets(["payment failed"]), segments(), ZERO_SHOT_PROMPT,
            client=client, sleeper=lambda _: None,
        )
        assert_no_pii(results.to_csv(index=False))
        assert_no_pii(json.dumps(summary))
        assert_no_pii(json.dumps(ledger.retries))
        assert_no_pii(json.dumps(ledger.attempts))

    def test_system_prompt_contracts(self):
        for prompt in [ZERO_SHOT_PROMPT, FEW_SHOT_PROMPT]:
            self.assertIn("CLASSIFY", prompt)
            for word in ["billing", "delivery", "technical", "fraud", "JSON", "category", "reason"]:
                self.assertIn(word, prompt)
            assert_no_pii(prompt)
        self.assertNotIn("EXAMPLES:", ZERO_SHOT_PROMPT)
        self.assertIn("\nEXAMPLES:\n", FEW_SHOT_PROMPT)
        self.assertGreaterEqual(FEW_SHOT_PROMPT.count("Label:"), 2)


class RetryAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.messages = [{"role": "system", "content": ZERO_SHOT_PROMPT}, {"role": "user", "content": "payment failed"}]
        self.rng = NS(uniform=Mock(return_value=1.0))
        self.delays = []

    def test_three_retries_exponential_backoff_and_final_success(self):
        self.assertEqual((MAX_RETRIES, BASE_DELAY), (3, 0.05))
        errors = [mock_llm.TransientLLMError("temporary") for _ in range(3)]
        client = NS(chat=Mock(side_effect=errors + [response(tokens=10)]))
        ledger = TokenLedger()
        status, _ = call_with_retries("T1", self.messages, ledger, client, self.rng, self.delays.append)
        self.assertEqual(status, "response")
        self.assertEqual(client.chat.call_count, 4)
        self.assertEqual(self.delays, [0.05, 0.1, 0.2])
        self.assertEqual([row["retry_number"] for row in ledger.retries], [0, 1, 2])
        self.assertEqual(ledger.total, 10)
        self.rng.uniform.assert_called_with(0.9, 1.15)

    def test_permanent_failure_is_bounded_and_routes_to_human(self):
        client = NS(chat=Mock(side_effect=mock_llm.TransientLLMError("temporary")), reset=Mock())
        results, summary, ledger = run_batch(
            tickets(["payment failed"]), segments(), ZERO_SHOT_PROMPT,
            client=client, sleeper=self.delays.append,
        )
        self.assertEqual(client.chat.call_count, 4)
        self.assertEqual(len(self.delays), 3)
        self.assertEqual(summary["permanent_failures"], 1)
        self.assertEqual(results.loc[0, "status"], "flagged_permanent_error")
        self.assertEqual(results.loc[0, "route"], "human_review")
        self.assertEqual(results.loc[0, "tokens_used"], 0)

    def test_shared_budget_crossing_blocks_every_later_call(self):
        client = NS(chat=Mock(return_value=response(tokens=7)), reset=Mock())
        results, summary, _ = run_batch(
            tickets(["payment failed"] * 3), segments(), ZERO_SHOT_PROMPT,
            token_budget=10, client=client, sleeper=lambda _: None,
        )
        self.assertEqual(client.chat.call_count, 2)
        self.assertEqual(results["status"].tolist(), ["classified_json", "classified_json", "budget_exhausted"])
        self.assertEqual(results["tokens_used"].tolist(), [7, 7, 0])
        self.assertEqual(summary["total_tokens"], 14)

    def test_zero_budget_never_calls_eligible_tickets(self):
        client = NS(chat=Mock(), reset=Mock())
        results, _, _ = run_batch(
            tickets(["payment failed"]), segments(), ZERO_SHOT_PROMPT,
            token_budget=0, client=client, sleeper=lambda _: None,
        )
        client.chat.assert_not_called()
        self.assertEqual(results.loc[0, "status"], "budget_exhausted")

    def test_budget_applies_to_retry_when_error_reports_usage(self):
        # The supplied mock has no exception usage. This explicit fixture tests
        # that reported retry costs would never escape the shared budget.
        error = mock_llm.TransientLLMError("temporary")
        error.usage = NS(prompt_tokens=3, completion_tokens=2, total_tokens=5)
        client = NS(chat=Mock(side_effect=[error, response()]))
        ledger = TokenLedger(limit=5)
        status, _ = call_with_retries("T1", self.messages, ledger, client, self.rng, self.delays.append)
        self.assertEqual(status, "budget_exhausted")
        self.assertEqual(client.chat.call_count, 1)
        self.assertEqual(ledger.total, 5)
        self.assertEqual(self.delays, [])

    def test_invalid_model_response_still_consumes_tokens(self):
        client = NS(chat=Mock(return_value=response('{"category":"shipping"}', 8)), reset=Mock())
        results, summary, _ = run_batch(
            tickets(["payment failed"]), segments(), ZERO_SHOT_PROMPT,
            client=client, sleeper=lambda _: None,
        )
        self.assertEqual(results.loc[0, "status"], "flagged_invalid_output")
        self.assertEqual(results.loc[0, "tokens_used"], 8)
        self.assertEqual(summary["total_tokens"], 8)
        self.assertEqual(summary["classified"], 0)


class FullRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data/raw/tickets.csv", keep_default_na=False)
        cls.segments = pd.read_csv(ROOT / "outputs/customer_segments.csv")

    def test_independent_full_run_counts_accuracy_and_actual_usage(self):
        for prompt, correct in [(ZERO_SHOT_PROMPT, 126), (FEW_SHOT_PROMPT, 147)]:
            client = SpyClient()
            _, summary, ledger = run_batch(
                self.raw, self.segments, prompt, client=client, sleeper=lambda _: None,
            )
            self.assertEqual(summary["status_counts"], {
                "classified_json": 135, "classified_keyword": 12,
                "flagged_invalid_output": 9, "flagged_long": 4, "skipped_empty": 4,
            })
            self.assertEqual((summary["classified"], summary["correct"]), (147, correct))
            self.assertEqual(summary["accuracy"], correct / 147)
            self.assertEqual((summary["transient_errors"], summary["permanent_failures"]), (19, 0))
            self.assertEqual((client.reset_count, len(ledger.attempts)), (1, 175))
            self.assertEqual(summary["total_tokens"], sum(row.usage.total_tokens for row in client.responses))
            self.assertEqual(summary["tokens_per_classified"], summary["total_tokens"] / 147)
            for entry in ledger.retries:
                self.assertEqual(entry["retry_number"], 0)
                self.assertGreaterEqual(entry["wait_seconds"], 0.045)
                self.assertLessEqual(entry["wait_seconds"], 0.0575)

    def test_budget_exhaustion_final_schema_and_routing_totals(self):
        budget = pd.read_csv(ROOT / "outputs/triage_budget_demo.csv", keep_default_na=False)
        final = pd.read_csv(ROOT / "outputs/triage_results.csv", keep_default_na=False)
        self.assertTrue(budget["status"].eq("budget_exhausted").any())
        self.assertEqual(final.columns.tolist(), OUTPUT_COLUMNS)
        self.assertEqual(len(final), 164)
        self.assertEqual(final["ticket_id"].tolist(), self.raw["ticket_id"].tolist())
        self.assertEqual(final["route"].value_counts().to_dict(), {
            "support_queue": 102, "security_team": 34, "human_review": 17, "premium_desk": 11,
        })

    def test_all_part_c_checkpoints_and_preservation(self):
        validate()
        assert_preserved()


if __name__ == "__main__":
    unittest.main()
