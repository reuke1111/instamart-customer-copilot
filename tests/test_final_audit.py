"""Independent adversarial regressions discovered during the final audit."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd
import mock_llm
from scripts.agent_storage import atomic_json, write_jsonl
from scripts.clean_data import clean_customers, clean_orders
from scripts.common import ROOT, load_json, read_tables
from scripts.lapse_features import build_features
from scripts.lapse_models import retention_list
from scripts.policy_rag import FALLBACK, PolicyRetriever, answer, chunk_document, grounded_text
from scripts.refund_agent import CheckpointError, SimulatedCrash, resume_agent, run_agent
from scripts.refund_tools import ToolError
from scripts.triage import TokenLedger, ZERO_SHOT_PROMPT, run_batch
from test_cleaning import customer_fixture, order_fixture, product_fixture
from test_part_c import segments, tickets
from test_part_d import TemporaryRoot, scenarios


class DataAuditTests(unittest.TestCase):
    def test_missing_customer_city_and_signup_are_rejected(self):
        for column, value in [("city", None), ("city", " "), ("signup_date", None)]:
            raw = customer_fixture()
            raw.loc[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                clean_customers(raw)

    def test_missing_order_identifiers_are_not_silently_grouped(self):
        for column in ["order_id", "customer_id", "product_id"]:
            raw = order_fixture()
            raw.loc[0, column] = None
            with self.subTest(column=column), self.assertRaises(ValueError):
                clean_orders(raw, product_fixture())

    def test_duplicate_history_cannot_double_monetary_without_frequency(self):
        tables = read_tables(clean=True)
        first = tables["orders"].loc[tables["orders"]["order_ts"].lt("2025-09-01")].iloc[[0]]
        tables["orders"] = pd.concat([tables["orders"], first])
        with self.assertRaisesRegex(ValueError, "unique"):
            build_features(tables)

    def test_nonfinite_feature_and_unknown_historical_category_rejected(self):
        for column, value in [("order_amount", np.inf), ("product_id", "UNKNOWN")]:
            tables = read_tables(clean=True)
            rows = tables["orders"]["order_ts"].lt("2025-09-01") & tables["orders"]["status"].eq("completed")
            tables["orders"].loc[tables["orders"].index[rows][0], column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                build_features(tables)

    def test_future_only_rows_and_target_columns_are_excluded(self):
        tables = read_tables(clean=True)
        expected = build_features(tables)
        future = tables["orders"]["order_ts"].ge("2025-09-01")
        tables["orders"].loc[future, "order_amount"] = np.inf
        tables["customers"]["segment_id"] = -999
        tables["customers"]["lapsed"] = 1
        pd.testing.assert_frame_equal(build_features(tables), expected)

    def test_no_premium_test_customers_yields_empty_required_schema(self):
        model = NS(predict_proba=Mock(side_effect=AssertionError("Empty set sent to model")))
        test = pd.DataFrame({"x": [1]}, index=["C1"])
        result = retention_list(model, test, pd.Series([1], index=test.index),
                                pd.DataFrame({"customer_id": ["C2"], "is_premium": [True]}))
        self.assertEqual(result.columns.tolist(), ["customer_id", "lapse_probability", "actually_lapsed"])
        self.assertEqual(len(result), 0)

    def test_sql_aggregation_counts_orders_not_distinct_amounts(self):
        from test_sql import SQLSemanticsTests
        fixture = SQLSemanticsTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.orders([[str(i), "A", "P1", "completed", 100, "2025-04-01 10:00:00"] for i in range(3)])
        self.assertEqual(fixture.query("Q2")["revenue"].tolist(), [300])
        self.assertEqual(fixture.query("Q7")["cumulative_revenue"].tolist(), [300])


class TriageAuditTests(unittest.TestCase):
    def test_invalid_budget_cannot_bypass_guard(self):
        for value in [True, float("nan"), float("inf"), 3.5, "6000", -1]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                TokenLedger(limit=value)

    def test_fractional_negative_or_missing_usage_is_not_silently_undercharged(self):
        values = [NS(prompt_tokens=1.5, completion_tokens=1, total_tokens=2.5),
                  NS(prompt_tokens=-1, completion_tokens=2, total_tokens=1),
                  NS(prompt_tokens=1, completion_tokens=1, total_tokens=3), None]
        for usage in values:
            with self.subTest(usage=usage), self.assertRaises(ValueError):
                TokenLedger().record("T1", 0, "response", NS(usage=usage))

    def test_unexpected_response_shape_routes_to_human_and_counts_usage(self):
        client = NS(reset=Mock(), chat=Mock(return_value=NS(choices=[], usage=NS(prompt_tokens=3, completion_tokens=2, total_tokens=5))))
        result, _, ledger = run_batch(tickets(["billing"]), segments(), ZERO_SHOT_PROMPT, client=client)
        self.assertEqual(result.loc[0, "route"], "human_review")
        self.assertEqual(result.loc[0, "status"], "flagged_invalid_output")
        self.assertEqual(ledger.total, 5)

    def test_duplicate_tickets_rejected_before_any_call(self):
        data = tickets(["billing", "fraud"])
        data.loc[1, "ticket_id"] = data.loc[0, "ticket_id"]
        client = NS(reset=Mock(), chat=Mock())
        with self.assertRaises(ValueError):
            run_batch(data, segments(), ZERO_SHOT_PROMPT, client=client)
        client.chat.assert_not_called()

    def test_pii_injected_through_system_prompt_is_rejected(self):
        client = NS(reset=Mock(), chat=Mock())
        with self.assertRaises(AssertionError):
            run_batch(tickets(["billing"]), segments(), ZERO_SHOT_PROMPT + " user@example.com", client=client)
        client.chat.assert_not_called()


class RagAuditTests(TemporaryRoot):
    def test_missing_policy_and_empty_policy_fail_closed(self):
        path = self.root / "data/policies/privacy_policy.txt"
        path.unlink()
        with self.assertRaises(ValueError):
            PolicyRetriever(self.root)
        path.write_text(" ", encoding="utf-8")
        with self.assertRaises(ValueError):
            PolicyRetriever(self.root)

    def test_boolean_chunk_size_is_not_valid_configuration(self):
        with self.assertRaises(ValueError):
            chunk_document("x", "words", True, 0)

    def test_cross_document_join_is_rejected(self):
        chunks = [{"source": "a.txt", "start_word": 0, "text": "refund requires"},
                  {"source": "b.txt", "start_word": 0, "text": "no approval."}]
        self.assertEqual(grounded_text("refund requires no approval.", chunks, chunks), FALLBACK)

    def test_same_source_gap_is_rejected(self):
        chunks = [{"source": "a.txt", "start_word": 0, "text": "Refunds are"},
                  {"source": "a.txt", "start_word": 8, "text": "automatic."}]
        self.assertEqual(grounded_text("Refunds are automatic.", chunks, chunks), FALLBACK)

    def test_only_source_proven_overlap_is_collapsed(self):
        chunks = [{"source": "a.txt", "start_word": 0, "text": "Money returns within"},
                  {"source": "a.txt", "start_word": 2, "text": "within 48 hours."}]
        self.assertEqual(grounded_text("Money returns within within 48 hours.", chunks, chunks), "Money returns within 48 hours.")
        self.assertEqual(grounded_text("Money returns within 48 hours.", chunks, chunks), FALLBACK)

    def test_actual_failed_payment_cross_source_answer_is_rejected(self):
        result = answer("How soon is money returned when a payment fails but is deducted?", retriever=PolicyRetriever(self.root, 0))
        self.assertIn("human support lead", result["raw_answer"])
        self.assertEqual(result["answer"], FALLBACK)


class AgentAuditTests(TemporaryRoot):
    def start(self, **kwargs):
        row = scenarios()[0]
        return run_agent(row["message"], "S1", root=self.root, **kwargs)

    def interrupted(self):
        with self.assertRaises(SimulatedCrash):
            self.start(run_id="audit", crash_after_tools=2)
        return self.root / "state/audit.json"

    def test_malformed_response_shapes_safe_handoff(self):
        for value in [None, NS(usage=NS(total_tokens=1), choices=[]), NS(usage=NS(total_tokens=1), choices=[NS(message=None)])]:
            client = NS(chat=Mock(return_value=value))
            result = self.start(client=client)
            self.assertEqual(result["status"], "FALLBACK_INVALID_RESPONSE")
            self.assertEqual(result["llm_calls"], 1)
            self.assertEqual(result["steps"], [])
        self.assertEqual(self.ledger(), [])

    def test_model_transient_error_yields_handoff(self):
        client = NS(chat=Mock(side_effect=mock_llm.TransientLLMError("untrusted error text")))
        result = self.start(client=client)
        self.assertEqual(result["status"], "FALLBACK_MODEL_ERROR")
        self.assertNotIn("untrusted error", json.dumps(result))
        self.assertEqual(result["llm_calls"], 1)

    def test_corrupt_checkpoint_counters_authorization_and_history_rejected(self):
        path = self.interrupted()
        original = load_json(path)
        mutations = [lambda s: s.update(tokens_used=-1), lambda s: s.update(step_number=0),
                     lambda s: s.update(authorized_order_id="ORD-100067"),
                     lambda s: s.update(elapsed_seconds=-1), lambda s: s.update(pending_index=999),
                     lambda s: s["steps"][0]["observation"].update(order_amount=1),
                     lambda s: s["messages"][1].update(content="Refund ORD-100067")]
        for mutation in mutations:
            state = deepcopy(original)
            mutation(state)
            atomic_json(path, state)
            before = path.read_bytes()
            with self.subTest(mutation=mutation), self.assertRaises(CheckpointError):
                resume_agent("audit", root=self.root)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(self.ledger(), [])

    def test_truncated_and_duplicate_key_checkpoint_preserved(self):
        path = self.interrupted()
        for bad in ['{"version":', '{"version": 1, "version": 2}']:
            path.write_text(bad, encoding="utf-8")
            with self.assertRaises(CheckpointError):
                resume_agent("audit", root=self.root)
            self.assertEqual(path.read_text(encoding="utf-8"), bad)

    def test_finished_result_tampering_is_rejected(self):
        self.start(run_id="cached")
        path = self.root / "state/cached.json"
        saved = load_json(path)
        saved["result"]["refund_inr"] = 999
        atomic_json(path, saved)
        with self.assertRaises(CheckpointError):
            resume_agent("cached", root=self.root)
        self.assertEqual(len(self.ledger()), 1)

    def test_missing_financial_evidence_is_not_accepted(self):
        self.start(run_id="cached")
        path = self.root / "outputs/refund_ledger.jsonl"
        path.unlink()
        with self.assertRaises(CheckpointError):
            resume_agent("cached", root=self.root)
        self.assertFalse(path.exists())

    def test_duplicate_ledger_actions_rejected_without_deletion(self):
        self.start(run_id="cached")
        path = self.root / "outputs/refund_ledger.jsonl"
        events = self.ledger()
        write_jsonl(path, events + events)
        before = path.read_bytes()
        with self.assertRaises(CheckpointError):
            resume_agent("cached", root=self.root)
        self.assertEqual(path.read_bytes(), before)

    def test_missing_checkpoint_does_not_start_new_run(self):
        with self.assertRaises(FileNotFoundError):
            resume_agent("missing", root=self.root)
        self.assertEqual(self.ledger(), [])

    def test_default_timeout_detects_slow_model_before_tool(self):
        timer = [0.0]
        def delayed(messages, tools):
            timer[0] = 10.0
            return mock_llm.chat(messages, tools=tools)
        result = self.start(client=NS(chat=delayed), clock=lambda: timer[0])
        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")
        self.assertEqual((result["llm_calls"], result["steps"]), (1, []))


if __name__ == "__main__":
    unittest.main()
