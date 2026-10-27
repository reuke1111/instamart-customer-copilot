"""Part D retrieval, execution-boundary, budget and durable-recovery tests."""

from collections import Counter
from contextlib import closing
from copy import deepcopy
import csv
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mock_llm
from scripts import agent_analysis, policy_rag
from scripts.agent_storage import atomic_json, file_lock, read_jsonl
from scripts.common import ROOT, load_json
from scripts.part_d_integrity import fingerprint
from scripts.refund_agent import SimulatedCrash, resume_agent, run_agent
from scripts.refund_tools import (
    TOOLS_SPEC, TOOL_MAP, ToolError, ToolExecutor, calculate_refund, get_order, issue_refund, validate_call,
)
from scripts.validate_part_d import COUNTS, EXPECTED, REFUND_SEQUENCE, validate, validate_result


def scenarios():
    with (ROOT / "data/raw/agent_scenarios.csv").open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def response(*, name=None, arguments=None, content=None, tokens=1, calls=None):
    if calls is None:
        calls = [] if name is None else [NS(id="call_1", name=name, arguments=arguments)]
    return NS(choices=[NS(message=NS(content=content, tool_calls=calls))], usage=NS(total_tokens=tokens))


class SpyClient:
    def __init__(self):
        self.messages = []
        self.usage = []

    def chat(self, messages, tools=None):
        self.messages.append(deepcopy(messages))
        result = mock_llm.chat(messages, tools=tools)
        self.usage.append(result.usage.total_tokens)
        return result


class RagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = policy_rag.PolicyRetriever()

    def test_policy_file_discovery(self):
        self.assertEqual([path.name for path in self.retriever.paths], sorted(COUNTS))

    def test_exact_chunk_counts(self):
        self.assertEqual(Counter(chunk["source"] for chunk in self.retriever.chunks), COUNTS)
        self.assertEqual(len(self.retriever.chunks), 15)

    def test_overlap_boundaries_and_end_condition(self):
        for source in COUNTS:
            chunks = [chunk for chunk in self.retriever.chunks if chunk["source"] == source]
            words = (ROOT / "data/policies" / source).read_text(encoding="utf-8").split()
            for previous, current in zip(chunks, chunks[1:]):
                self.assertEqual(previous["text"].split()[-10:], current["text"].split()[:10])
                self.assertEqual(current["start_word"] - previous["start_word"], 30)
            self.assertEqual(chunks[-1]["end_word"], len(words))
            self.assertTrue(all(chunk["end_word"] < len(words) for chunk in chunks[:-1]))
        self.assertEqual(len(policy_rag.chunk_document("x", "word " * 40)), 1)
        self.assertEqual(policy_rag.chunk_document("empty", ""), [])

    def test_source_attribution_ids_and_text(self):
        for path in self.retriever.paths:
            words = path.read_text(encoding="utf-8").split()
            chunks = [chunk for chunk in self.retriever.chunks if chunk["source"] == path.name]
            for index, chunk in enumerate(chunks):
                self.assertEqual(chunk["chunk_id"], f"{path.name}:{index:03d}")
                self.assertEqual(chunk["text"], " ".join(words[chunk["start_word"]:chunk["end_word"]]))

    def test_refund_retrieval_ranking(self):
        result = self.retriever.retrieve("refund policy late delivery")
        self.assertEqual(len(result), 3)
        self.assertEqual([row["source"] for row in result], ["refund_policy.txt"] * 3)
        self.assertEqual([row["score"] for row in result], sorted([row["score"] for row in result], reverse=True))
        self.assertTrue(all(set(row) == {"source", "score", "text"} for row in result))

    def test_tie_breaking_is_chunk_id_order(self):
        result = self.retriever.retrieve("zzzznonexistentvocabulary", k=15)
        expected = sorted(self.retriever.chunks, key=lambda row: row["chunk_id"])
        self.assertEqual([row["text"] for row in result], [row["text"] for row in expected])
        self.assertTrue(all(row["score"] == 0 for row in result))

    def test_exact_prompt_structure(self):
        question = "How soon are approved refunds credited?"
        client = SpyClient()
        result = policy_rag.answer(question, retriever=self.retriever, client=client)
        prompt = "CONTEXT:\n" + "\n".join(f"[{row['source']}]\n{row['text']}" for row in result["retrieved"])
        self.assertEqual(client.messages[0][-1], {"role": "user", "content": prompt + "\nQUESTION:\n" + question})
        self.assertIn("POLICY ASSISTANT", client.messages[0][0]["content"])
        self.assertEqual(len(result["sources"]), 3)

    def test_all_fourteen_questions(self):
        report = policy_rag.evaluate()
        self.assertEqual((report["questions"], report["answerable"], report["source_correct"], report["fact_correct"]), (14, 12, 12, 12))
        self.assertEqual((report["unanswerable"], report["unsupported_fallbacks"]), (2, 2))

    def test_unsupported_question_fallbacks(self):
        report = policy_rag.evaluate()
        for row in report["details"]:
            if not row["answerable"]:
                self.assertEqual(row["answer"], policy_rag.FALLBACK)

    def test_out_of_context_or_empty_model_answer_is_rejected(self):
        for text in ["Refunds are always INR 9000.", "", None, "I could not find it, but refunds are unlimited."]:
            with self.subTest(text=text):
                client = NS(chat=lambda messages: response(content=text))
                result = policy_rag.answer("Refund?", retriever=self.retriever, client=client)
                self.assertEqual(result["answer"], policy_rag.FALLBACK)

    def test_zero_overlap_measured_failures(self):
        report = policy_rag.evaluate(overlap=0)
        self.assertEqual(report["chunks"], 13)
        # The original raw mock still hits 10 facts. Source-boundary validation
        # additionally rejects two misleading cross-document answers.
        self.assertEqual(report["raw_fact_correct"], 10)
        self.assertEqual(report["fact_correct"], 8)
        self.assertEqual(report["source_correct"], 12)
        failures = [row["expected_fact"] for row in report["details"] if row["answerable"] and not row["fact_correct"]]
        self.assertEqual(set(failures), {"more than 45 minutes", "48 hours", "INR 25", "180 days"})

    def test_invalid_chunking_and_retrieval_options(self):
        for size, overlap in [(0, 0), (40, 40), (40, -1)]:
            with self.assertRaises(ValueError):
                policy_rag.chunk_document("x", "text", size, overlap)
        for k in [0, -1, True, "3"]:
            with self.assertRaises(ValueError):
                self.retriever.retrieve("refund", k)


class TemporaryRoot(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix=".part-a-test-", dir=ROOT)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        shutil.copytree(ROOT / "data/policies", self.root / "data/policies")
        (self.root / "data/raw").mkdir()
        for name in ("agent_scenarios.csv", "policy_eval.csv"):
            shutil.copy2(ROOT / "data/raw" / name, self.root / "data/raw" / name)
        shutil.copy2(ROOT / "data/instamart.db", self.root / "data/instamart.db")
        shutil.copy2(ROOT / "mock_llm.py", self.root / "mock_llm.py")

    def ledger(self):
        return read_jsonl(self.root / "outputs/refund_ledger.jsonl")


class ToolTests(TemporaryRoot):
    def setUp(self):
        super().setUp()
        self.retriever = policy_rag.PolicyRetriever(self.root)

    def executor(self, order_id="ORD-100048", run_id="tool-test"):
        return ToolExecutor(self.root, run_id, "test", order_id, self.retriever)

    def change_order(self, *, amount=128.25, minutes=65, status="completed"):
        with closing(sqlite3.connect(self.root / "data/instamart.db")) as connection:
            with connection:
                connection.execute("UPDATE orders SET order_amount=?, delivery_minutes=?, status=? WHERE order_id=?",
                                   (amount, None if status == "cancelled" else minutes, status, "ORD-100048"))

    def test_tool_schemas_and_map(self):
        self.assertEqual({item["name"] for item in TOOLS_SPEC}, set(TOOL_MAP))
        for spec in TOOLS_SPEC:
            self.assertTrue(spec["description"])
            self.assertFalse(spec["parameters"]["additionalProperties"])
            self.assertEqual(set(spec["parameters"]["required"]), set(spec["parameters"]["properties"]))

    def test_missing_order_structured_error(self):
        self.assertEqual(get_order("ORD-999999", root=self.root), {"error": "order not found"})

    def test_order_lookup_fields_and_policy_reuse(self):
        executor = self.executor()
        order = executor.execute("get_order", {"order_id": "ORD-100048"})
        self.assertEqual(set(order), {"order_id", "status", "order_amount", "delivery_minutes"})
        self.assertEqual(order["order_amount"], 128.25)
        self.assertEqual(executor.execute("search_policy", {"query": "refund"})["results"], self.retriever.retrieve("refund"))

    def test_cancelled_order_cannot_receive_financial_action(self):
        self.change_order(status="cancelled")
        with self.assertRaises(ToolError):
            self.executor().execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25})
        self.assertEqual(self.ledger(), [])

    def test_exactly_45_minutes_denied(self):
        self.change_order(minutes=45)
        executor = self.executor()
        self.assertEqual(executor.execute("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 45}), {"refund_inr": 0.0})
        with self.assertRaises(ToolError):
            executor.execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25})

    def test_more_than_45_minutes_eligible(self):
        self.change_order(minutes=45.01)
        self.assertEqual(calculate_refund(128.25, 45.01), {"refund_inr": 128.25})
        self.executor().execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25})
        self.assertEqual(len(self.ledger()), 1)

    def test_exactly_1000_automatically_issued(self):
        self.change_order(amount=1000)
        executor = self.executor()
        with self.assertRaises(ToolError):
            executor.execute("escalate_to_human", {"order_id": "ORD-100048", "refund_inr": 1000})
        self.assertEqual(executor.execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 1000}), {"refund_inr": 1000.0})

    def test_above_1000_requires_escalation(self):
        self.change_order(amount=1000.01)
        executor = self.executor()
        with self.assertRaises(ToolError):
            executor.execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 1000.01})
        executor.execute("escalate_to_human", {"order_id": "ORD-100048", "refund_inr": 1000.01})
        self.assertEqual(self.ledger()[0]["action"], "escalate_to_human")

    def test_forged_amount_and_calculation_inputs_rejected(self):
        executor = self.executor()
        for name, arguments in [
            ("issue_refund", {"order_id": "ORD-100048", "refund_inr": 999}),
            ("calculate_refund", {"order_amount": 999, "delivery_minutes": 65}),
            ("calculate_refund", {"order_amount": 128.25, "delivery_minutes": 999}),
        ]:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ToolError):
                executor.execute(name, arguments)
        self.assertEqual(self.ledger(), [])

    def test_order_outside_user_request_rejected(self):
        with self.assertRaises(ToolError):
            self.executor().execute("issue_refund", {"order_id": "ORD-100067", "refund_inr": 135})
        with self.assertRaises(ToolError):
            self.executor().execute("get_order", {"order_id": "ORD-100067"})
        with self.assertRaises(ToolError):
            issue_refund("ORD-100048", 128.25)
        self.assertEqual(self.ledger(), [])

    def test_invalid_tool_names_arguments_and_numbers(self):
        invalid = [
            ("delete_database", {}), ("get_order", "{}"), ("get_order", {}),
            ("get_order", {"order_id": "ORD-100048", "extra": 1}),
            ("get_order", {"order_id": "ORD-100048' OR 1=1"}),
            ("search_policy", {"query": " "}),
        ]
        invalid += [("issue_refund", {"order_id": "ORD-100048", "refund_inr": value})
                    for value in [True, -1, float("nan"), float("inf"), "128.25", 128.251]]
        for name, arguments in invalid:
            with self.subTest(name=name, arguments=arguments), self.assertRaises(ToolError):
                validate_call(name, arguments)
        self.assertEqual(self.ledger(), [])

    def test_idempotent_action_retains_timestamp_and_ledger_bytes(self):
        executor = self.executor()
        arguments = {"order_id": "ORD-100048", "refund_inr": 128.25}
        first = executor.execute("issue_refund", arguments)
        before = (self.root / "outputs/refund_ledger.jsonl").read_bytes()
        self.assertEqual(executor.execute("issue_refund", arguments), first)
        self.assertEqual((self.root / "outputs/refund_ledger.jsonl").read_bytes(), before)
        self.assertEqual(len(self.ledger()), 1)

    def test_reused_action_key_with_changed_payload_fails_closed(self):
        executor = self.executor()
        executor.execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 128.25})
        self.change_order(amount=200)
        with self.assertRaises(ToolError):
            executor.execute("issue_refund", {"order_id": "ORD-100048", "refund_inr": 200})
        self.assertEqual(self.ledger()[0]["refund_inr"], 128.25)

    def test_os_lock_rejects_competing_writer(self):
        path = self.root / "test.lock"
        with file_lock(path):
            with self.assertRaises(OSError):
                with file_lock(path):
                    self.fail("A competing writer acquired the same lock")


class AgentTests(TemporaryRoot):
    def run_scenario(self, number=1, **kwargs):
        row = scenarios()[number - 1]
        return run_agent(row["message"], row["scenario_id"], root=self.root, **kwargs)

    def test_all_seven_statuses_refunds_and_call_counts(self):
        results = [self.run_scenario(number) for number in range(1, 8)]
        self.assertEqual([(row["scenario_id"], row["status"], row["refund_inr"]) for row in results], EXPECTED)
        self.assertEqual([row["llm_calls"] for row in results], [5, 5, 5, 4, 3, 2, 6])
        self.assertEqual(len(self.ledger()), 3)
        self.assertEqual([step["tool"] for step in results[0]["steps"]], REFUND_SEQUENCE)

    def test_token_limit_blocks_violating_response_tool(self):
        client = SpyClient()
        result = self.run_scenario(token_limit=200, client=client)
        self.assertEqual(result["status"], "FALLBACK_TOKEN_LIMIT")
        self.assertEqual(result["tokens_used"], sum(client.usage))
        self.assertLessEqual(sum(client.usage[:-1]), 200)
        self.assertGreater(sum(client.usage), 200)
        self.assertEqual([step["tool"] for step in result["steps"]], REFUND_SEQUENCE[:2])
        self.assertEqual(self.ledger(), [])

    def test_zero_timeout_has_no_calls_or_tools(self):
        client = NS(chat=lambda *args, **kwargs: self.fail("Timeout called the model"))
        result = self.run_scenario(time_limit_seconds=0, client=client)
        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")
        self.assertEqual((result["llm_calls"], result["tokens_used"], result["steps"]), (0, 0, []))

    def test_maximum_iteration_fallback_under_default_budget(self):
        result = self.run_scenario(7)
        self.assertEqual(result["status"], "FALLBACK_MAX_ITERATIONS")
        self.assertEqual(result["llm_calls"], 6)
        self.assertLess(result["tokens_used"], 6000)
        self.assertEqual([step["tool"] for step in result["steps"]], ["search_policy"] * 6)

    def test_invalid_model_tool_name_or_arguments_never_execute(self):
        for name, arguments in [("shell", {}), ("get_order", {"wrong": "ORD-100048"}), ("issue_refund", "not JSON")]:
            with self.subTest(name=name):
                client = NS(chat=lambda *args, **kwargs: response(name=name, arguments=arguments))
                result = self.run_scenario(client=client)
                self.assertEqual(result["status"], "FALLBACK_INVALID_TOOL")
                self.assertEqual(result["steps"], [])
        self.assertEqual(self.ledger(), [])

    def test_all_calls_validated_before_any_in_batch_execute(self):
        calls = [NS(id="one", name="get_order", arguments={"order_id": "ORD-100048"}),
                 NS(id="two", name="unknown", arguments={})]
        client = NS(chat=lambda *args, **kwargs: response(calls=calls))
        result = self.run_scenario(client=client)
        self.assertEqual(result["status"], "FALLBACK_INVALID_TOOL")
        self.assertEqual(result["steps"], [])

    def test_malformed_tool_container_is_rejected(self):
        client = NS(chat=lambda *args, **kwargs: response(calls=7))
        self.assertEqual(self.run_scenario(client=client)["status"], "FALLBACK_INVALID_TOOL")

    def test_hallucinated_financial_final_is_rejected(self):
        client = NS(chat=lambda *args, **kwargs: response(content="REFUND: INR 128.25 issued."))
        result = self.run_scenario(client=client)
        self.assertEqual(result["status"], "FALLBACK_INVALID_RESPONSE")
        self.assertEqual(self.ledger(), [])

    def test_tool_boundary_failure_returns_safe_handoff(self):
        client = NS(chat=lambda *args, **kwargs: response(name="issue_refund", arguments={"order_id": "ORD-100048", "refund_inr": 999}))
        result = self.run_scenario(client=client)
        self.assertEqual(result["status"], "FALLBACK_TOOL_ERROR")
        self.assertIn("Human review", result["final_answer"])
        self.assertEqual(self.ledger(), [])

    def test_checkpoint_serialization_and_resume_without_repeated_tools(self):
        executed = []
        original = ToolExecutor.execute
        def track(executor, name, arguments):
            executed.append(name)
            return original(executor, name, arguments)
        with patch.object(ToolExecutor, "execute", track):
            with self.assertRaises(SimulatedCrash):
                self.run_scenario(run_id="resume-test", crash_after_tools=2)
            saved = load_json(self.root / "state/resume-test.json")
            self.assertEqual(saved["step_number"], 2)
            self.assertEqual(saved["tools_executed"], REFUND_SEQUENCE[:2])
            self.assertEqual(saved["execution_state"], "running")
            self.assertEqual(saved["llm_calls"], 2)
            self.assertGreater(saved["tokens_used"], 0)
            self.assertEqual(json.loads(json.dumps(saved)), saved)
            client = SpyClient()
            result = resume_agent("resume-test", root=self.root, client=client)
        self.assertEqual(client.messages[0], saved["messages"])
        self.assertEqual(executed, REFUND_SEQUENCE)
        self.assertEqual(result["status"], "REFUND")
        self.assertEqual(result["refund_inr"], 128.25)
        self.assertEqual(result["llm_calls"], 5)
        self.assertEqual(len(self.ledger()), 1)

    def test_crash_after_ledger_write_before_observation_is_idempotent(self):
        def crash_after_effect(name, observation):
            if name == "issue_refund":
                raise SimulatedCrash("Ledger persisted, observation not yet persisted")
        with self.assertRaises(SimulatedCrash):
            self.run_scenario(run_id="effect-crash", after_effect=crash_after_effect)
        saved = load_json(self.root / "state/effect-crash.json")
        self.assertEqual(saved["tools_executed"], REFUND_SEQUENCE[:3])
        self.assertEqual(saved["pending_calls"][saved["pending_index"]]["name"], "issue_refund")
        before = (self.root / "outputs/refund_ledger.jsonl").read_bytes()
        result = resume_agent("effect-crash", root=self.root)
        self.assertEqual(result["status"], "REFUND")
        self.assertEqual([step["tool"] for step in result["steps"]], REFUND_SEQUENCE)
        self.assertEqual((self.root / "outputs/refund_ledger.jsonl").read_bytes(), before)

    def test_finished_resume_has_no_model_call_or_new_action(self):
        first = self.run_scenario(run_id="finished")
        before = (self.root / "outputs/refund_ledger.jsonl").read_bytes()
        client = NS(chat=lambda *args, **kwargs: self.fail("Finished run called the model"))
        self.assertEqual(resume_agent("finished", root=self.root, client=client), first)
        self.assertEqual((self.root / "outputs/refund_ledger.jsonl").read_bytes(), before)

    def test_changed_source_rejects_resume(self):
        with self.assertRaises(SimulatedCrash):
            self.run_scenario(run_id="changed", crash_after_tools=2)
        path = self.root / "data/policies/refund_policy.txt"
        path.write_text(path.read_text(encoding="utf-8") + " changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            resume_agent("changed", root=self.root)
        self.assertEqual(self.ledger(), [])

    def test_same_run_id_cannot_be_repurposed(self):
        self.run_scenario(run_id="fixed")
        with self.assertRaises(ValueError):
            self.run_scenario(2, run_id="fixed")
        with self.assertRaises(ValueError):
            self.run_scenario(run_id="fixed", token_limit=200)
        self.assertEqual(len(self.ledger()), 1)

    def test_invalid_run_ids_and_limits(self):
        for run_id in ["../outside", "a/b", "a\\b", "a" * 81]:
            with self.assertRaises(ValueError):
                self.run_scenario(run_id=run_id)
        for kwargs in [{"max_iterations": -1}, {"token_limit": True}, {"time_limit_seconds": float("nan")}]:
            with self.assertRaises(ValueError):
                self.run_scenario(**kwargs)

    def test_active_time_is_retained_across_resume(self):
        with self.assertRaises(SimulatedCrash):
            self.run_scenario(run_id="elapsed", crash_after_tools=2)
        path = self.root / "state/elapsed.json"
        state = load_json(path)
        state["elapsed_seconds"] = state["limits"]["time_limit_seconds"]
        atomic_json(path, state)
        client = NS(chat=lambda *args, **kwargs: self.fail("Exhausted active-time budget called model"))
        result = resume_agent("elapsed", root=self.root, client=client)
        self.assertEqual(result["status"], "FALLBACK_TIMEOUT")
        self.assertEqual(result["llm_calls"], 2)
        self.assertEqual(self.ledger(), [])

    def test_deterministic_full_part_d_reruns(self):
        first_rag = policy_rag.run(self.root)
        first_agents = agent_analysis.run(self.root)
        paths = [self.root / "outputs/agent_runs.jsonl", self.root / "outputs/refund_ledger.jsonl"]
        before = [path.read_bytes() for path in paths]
        self.assertEqual(policy_rag.run(self.root), first_rag)
        self.assertEqual(agent_analysis.run(self.root), first_agents)
        self.assertEqual([path.read_bytes() for path in paths], before)
        self.assertEqual(len(read_jsonl(paths[0])), 7)
        self.assertEqual(len(self.ledger()), 4)
        for row in read_jsonl(paths[0]):
            validate_result(row)


class ArtifactTests(unittest.TestCase):
    def test_all_checkpoints_and_original_artifact_preservation(self):
        before = fingerprint()
        validate()
        self.assertEqual(fingerprint(), before)

    def test_jsonl_schema_exact_seven_normal_results(self):
        results = read_jsonl(ROOT / "outputs/agent_runs.jsonl")
        self.assertEqual(len(results), 7)
        self.assertEqual({row["scenario_id"] for row in results}, {f"S{n}" for n in range(1, 8)})
        for row in results:
            validate_result(row)
        ledger = read_jsonl(ROOT / "outputs/refund_ledger.jsonl")
        self.assertEqual(len(ledger), len({row["action_id"] for row in ledger}))
        for row in ledger:
            self.assertTrue({"timestamp", "action", "order_id", "refund_inr", "run_id", "scenario_id"} <= set(row))


if __name__ == "__main__":
    unittest.main()
