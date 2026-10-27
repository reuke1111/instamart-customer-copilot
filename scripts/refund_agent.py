"""Bounded ReAct loop with durable pending calls, checkpoints and resumability."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
import uuid

import mock_llm
from scripts.agent_storage import atomic_json, file_lock
from scripts.common import ROOT
from scripts.policy_rag import default_retriever
from scripts.refund_tools import TOOLS_SPEC, ToolError, ToolExecutor, checked_ledger, money, validate_call

SYSTEM_PROMPT = (
    "AGENT. Use the provided tools to process this simulated refund. "
    "Never invent observations. Follow verified eligibility and amount checks. "
    "Return a final status and concise explanation."
)


class SimulatedCrash(RuntimeError):
    """Test/demo interruption after a successfully persisted observation."""


class CheckpointError(ValueError):
    """Invalid saved state is preserved and never executed."""


def authorized_order(user_message):
    orders = sorted(set(re.findall(r"ORD-\d{6}", user_message)))
    if len(orders) > 1:
        raise ValueError("Each refund run must authorize at most one order")
    return orders[0] if orders else None


def load_checkpoint(path, root, run_id):
    """Validate state identity, history and financial evidence before any resume.

    This detects corruption and inconsistent local edits; it is not a signature
    or protection against an attacker who can coherently rewrite every file.
    """
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_object)
        if state["version"] != 1 or state["run_id"] != run_id or state["system_prompt"] != SYSTEM_PROMPT:
            raise ValueError("Checkpoint identity mismatch")
        if state["source_fingerprint"] != source_fingerprint(root):
            raise ValueError("Checkpoint source version mismatch")
        if not isinstance(state["user_message"], str) or not isinstance(state["scenario_id"], str) or not state["scenario_id"]:
            raise ValueError("Invalid saved request")
        if state["authorized_order_id"] != authorized_order(state["user_message"]):
            raise ValueError("Saved authorization differs from original request")
        _validate_limits(**state["limits"])
        for key in ("llm_calls", "tokens_used", "step_number", "pending_index"):
            if type(state[key]) is not int or state[key] < 0:
                raise ValueError("Invalid saved counter")
        if type(state["elapsed_seconds"]) not in (int, float) or not math.isfinite(state["elapsed_seconds"]) or state["elapsed_seconds"] < 0:
            raise ValueError("Invalid elapsed time")
        if state["execution_state"] not in {"running", "finished"}:
            raise ValueError("Invalid execution state")
        if state["llm_calls"] > state["limits"]["max_iterations"] or (
            state["execution_state"] == "running" and state["tokens_used"] > state["limits"]["token_limit"]
        ):
            raise ValueError("Running checkpoint exceeds its budget")
        messages, steps, pending = state["messages"], state["steps"], state["pending_calls"]
        if not all(isinstance(item, list) for item in (messages, steps, pending)):
            raise ValueError("Checkpoint histories must be lists")
        if messages[:2] != [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": state["user_message"]}]:
            raise ValueError("Original messages changed")
        if state["step_number"] != len(steps) or state["tools_executed"] != [s["tool"] for s in steps]:
            raise ValueError("Completed tool sequence mismatch")
        queue, ids, observed, assistant_count, last_batch = [], set(), [], 0, []
        for index, message in enumerate(messages[2:], 2):
            if message["role"] == "assistant":
                assistant_count += 1
                if queue:
                    raise ValueError("New assistant message before pending observations")
                if message.get("tool_calls"):
                    if message["content"] is not None or not isinstance(message["tool_calls"], list):
                        raise ValueError("Invalid assistant call batch")
                    for call in message["tool_calls"]:
                        if set(call) != {"id", "name", "arguments"} or safe_run_id(call["id"]) in ids:
                            raise ValueError("Invalid or duplicate call ID")
                        validate_call(call["name"], call["arguments"])
                        if "order_id" in call["arguments"] and call["arguments"]["order_id"] != state["authorized_order_id"]:
                            raise ValueError("Saved call is outside the request")
                        ids.add(call["id"])
                    last_batch = message["tool_calls"]
                    queue = list(last_batch)
                elif state["execution_state"] != "finished" or index != len(messages) - 1:
                    raise ValueError("Unexpected final message")
            elif message["role"] == "tool":
                if not queue:
                    raise ValueError("Orphan tool observation")
                call = queue.pop(0)
                if message["tool_call_id"] != call["id"] or message["name"] != call["name"]:
                    raise ValueError("Tool observation does not match its call")
                observation = json.loads(message["content"], object_pairs_hook=unique_object)
                if not isinstance(observation, dict):
                    raise ValueError("Observation must be an object")
                observed.append({"tool_call_id": call["id"], "tool": call["name"], "arguments": call["arguments"], "observation": observation})
            else:
                raise ValueError("Unexpected message role")
        if observed != steps or not assistant_count <= state["llm_calls"] <= assistant_count + 1:
            raise ValueError("History, steps or call count disagree")
        if state["pending_index"] > len(pending) or pending[state["pending_index"]:] != queue or (pending and pending != last_batch):
            raise ValueError("Pending execution state disagrees with history")
        financial = [step for step in steps if step["tool"] in {"issue_refund", "escalate_to_human"}]
        if financial:
            ledger = checked_ledger(Path(root) / "outputs/refund_ledger.jsonl")
            for step in financial:
                if not any(event["run_id"] == run_id and event["scenario_id"] == state["scenario_id"] and
                           event["action"] == step["tool"] and event["order_id"] == step["arguments"]["order_id"] and
                           event["refund_inr"] == step["arguments"]["refund_inr"] == step["observation"]["refund_inr"] for event in ledger):
                    raise ValueError("Financial observation has no matching ledger event")
        if state["execution_state"] == "finished":
            result = state["result"]
            for key in ("run_id", "scenario_id", "steps", "llm_calls", "tokens_used"):
                if result[key] != state[key]:
                    raise ValueError("Cached result disagrees with checkpoint")
            if result["refund_inr"] != _refund_from_steps(steps) or not isinstance(result["final_answer"], str):
                raise ValueError("Cached result amount or answer is invalid")
            fallbacks = {"FALLBACK_TIMEOUT", "FALLBACK_TOKEN_LIMIT", "FALLBACK_MAX_ITERATIONS", "FALLBACK_TOOL_ERROR",
                         "FALLBACK_INVALID_RESPONSE", "FALLBACK_INVALID_TOOL", "FALLBACK_MODEL_ERROR"}
            if result["status"] not in fallbacks and _verified_final(result["final_answer"], steps) != result["status"]:
                raise ValueError("Unsupported cached final result")
        return state
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        raise CheckpointError("Invalid checkpoint; preserved for human review") from error


def safe_run_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError("Run ID must use 1-80 letters, digits, underscores or hyphens")
    return value


def source_fingerprint(root):
    root = Path(root)
    paths = [root / "data/instamart.db", root / "mock_llm.py",
             *sorted((root / "data/policies").glob("*.txt"))]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _validate_limits(max_iterations, token_limit, time_limit_seconds):
    for value in [max_iterations, token_limit]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Iteration and token limits must be nonnegative integers")
    if isinstance(time_limit_seconds, bool) or not isinstance(time_limit_seconds, (int, float)) or (
        not math.isfinite(time_limit_seconds) or time_limit_seconds < 0
    ):
        raise ValueError("Time limit must be finite and nonnegative")


def _refund_from_steps(steps):
    return next((
        float(step["observation"]["refund_inr"]) for step in reversed(steps)
        if step["tool"] in {"issue_refund", "escalate_to_human"}
    ), 0.0)


def _verified_final(text, steps):
    if not isinstance(text, str) or ":" not in text:
        raise ToolError("Model did not return a supported final status")
    status = text.split(":", 1)[0]
    observations = {step["tool"]: step["observation"] for step in steps}
    if status == "REFUND":
        if "issue_refund" not in observations:
            raise ToolError("Model claimed an unexecuted refund")
        stated = re.search(r"INR\s+([0-9]+(?:\.[0-9]+)?)", text)
        if not stated or money(float(stated.group(1))) != money(observations["issue_refund"]["refund_inr"]):
            raise ToolError("Model final amount disagrees with the recorded refund")
    elif status == "ESCALATE":
        if "escalate_to_human" not in observations:
            raise ToolError("Model claimed an unexecuted escalation")
    elif status == "NOT_FOUND":
        if observations.get("get_order") != {"error": "order not found"}:
            raise ToolError("Missing-order claim has no supporting observation")
    elif status == "DENIED":
        order = observations.get("get_order", {})
        if _refund_from_steps(steps) or not (
            order.get("status") == "cancelled" or observations.get("calculate_refund") == {"refund_inr": 0.0}
        ):
            raise ToolError("Denial has no supporting observation")
    elif status == "NEEDS_INFO":
        if steps:
            raise ToolError("Unexpected request for more information after tool execution")
    else:
        raise ToolError("Unsupported model final status")
    return status


def _continue(state, path, root, client, clock, crash_after_tools=None, after_effect=None):
    if state["execution_state"] == "finished":
        return deepcopy(state["result"])
    started = clock()
    prior_elapsed = state["elapsed_seconds"]
    executor = ToolExecutor(
        Path(root), state["run_id"], state["scenario_id"],
        state["authorized_order_id"], default_retriever(Path(root)),
    )

    def elapsed():
        return prior_elapsed + clock() - started

    def checkpoint():
        state["elapsed_seconds"] = elapsed()
        state["step_number"] = len(state["steps"])
        state["tools_executed"] = [step["tool"] for step in state["steps"]]
        atomic_json(path, state)

    def finish(status, text=None):
        if text is None:
            text = (
                f"Human review required ({status}). Completed {len(state['steps'])} tools "
                f"after {state['llm_calls']} model calls and {state['tokens_used']} tokens. "
                "Review the saved observations, any pending call, and the refund ledger before continuing."
            )
        result = {
            "run_id": state["run_id"], "scenario_id": state["scenario_id"],
            "status": status, "final_answer": text, "steps": deepcopy(state["steps"]),
            "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"],
            "refund_inr": _refund_from_steps(state["steps"]),
        }
        state["execution_state"] = "finished"
        state["result"] = result
        checkpoint()
        return deepcopy(result)

    while True:
        if elapsed() >= state["limits"]["time_limit_seconds"]:
            return finish("FALLBACK_TIMEOUT")
        # Pending calls are completed before another model invocation. The intent
        # was checkpointed before execution, so an interrupted financial write can
        # be reconciled through its durable ledger idempotency key.
        if state["pending_index"] < len(state["pending_calls"]):
            call = state["pending_calls"][state["pending_index"]]
            try:
                observation = executor.execute(call["name"], call["arguments"])
            except (ToolError, OSError, ValueError, sqlite3.Error) as error:
                state["last_error_type"] = type(error).__name__
                return finish("FALLBACK_TOOL_ERROR")
            if after_effect is not None:
                after_effect(call["name"], observation)
            state["messages"].append({
                "role": "tool", "tool_call_id": call["id"], "name": call["name"],
                "content": json.dumps(observation, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            })
            state["steps"].append({
                "tool_call_id": call["id"], "tool": call["name"],
                "arguments": deepcopy(call["arguments"]), "observation": observation,
            })
            state["pending_index"] += 1
            checkpoint()
            if crash_after_tools is not None and len(state["steps"]) == crash_after_tools:
                raise SimulatedCrash("Simulated crash after the saved tool observation")
            continue
        state["pending_calls"], state["pending_index"] = [], 0
        if state["llm_calls"] >= state["limits"]["max_iterations"]:
            return finish("FALLBACK_MAX_ITERATIONS")
        state["llm_calls"] += 1
        try:
            response = client.chat(state["messages"], tools=TOOLS_SPEC)
        except mock_llm.TransientLLMError:
            return finish("FALLBACK_MODEL_ERROR")
        usage = getattr(getattr(response, "usage", None), "total_tokens", None)
        if isinstance(usage, bool) or not isinstance(usage, int) or usage < 0:
            return finish("FALLBACK_INVALID_RESPONSE")
        state["tokens_used"] += usage
        # Strictly exceeds, as required; equality alone is not a violation.
        if state["tokens_used"] > state["limits"]["token_limit"]:
            return finish("FALLBACK_TOKEN_LIMIT")
        try:
            message = response.choices[0].message
            calls = message.tool_calls
            content = message.content
        except (AttributeError, IndexError, TypeError):
            return finish("FALLBACK_INVALID_RESPONSE")
        calls = [] if calls is None else calls
        if not isinstance(calls, list):
            return finish("FALLBACK_INVALID_TOOL")
        if not calls:
            try:
                status = _verified_final(content, state["steps"])
            except ToolError:
                return finish("FALLBACK_INVALID_RESPONSE")
            state["messages"].append({"role": "assistant", "content": message.content})
            return finish(status, message.content)
        validated = []
        used_ids = {step["tool_call_id"] for step in state["steps"]}
        try:
            for call in calls:
                if not isinstance(call.id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", call.id) or call.id in used_ids:
                    raise ToolError("Invalid or repeated tool-call identifier")
                validate_call(call.name, call.arguments)
                validated.append({"id": call.id, "name": call.name, "arguments": deepcopy(call.arguments)})
                used_ids.add(call.id)
        except (ToolError, AttributeError):
            return finish("FALLBACK_INVALID_TOOL")
        state["messages"].append({"role": "assistant", "content": None, "tool_calls": validated})
        state["pending_calls"] = validated
        checkpoint()


def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10,
              *, run_id=None, root=ROOT, client=mock_llm, clock=time.monotonic,
              crash_after_tools=None, after_effect=None):
    _validate_limits(max_iterations, token_limit, time_limit_seconds)
    if not isinstance(user_message, str) or not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("User message and scenario ID must be strings")
    run_id = safe_run_id(run_id or uuid.uuid4().hex)
    root = Path(root)
    path = root / "state" / f"{run_id}.json"
    limits = {"max_iterations": max_iterations, "token_limit": token_limit, "time_limit_seconds": time_limit_seconds}
    order_id = authorized_order(user_message)
    inputs = source_fingerprint(root)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            state = load_checkpoint(path, root, run_id)
            if state["user_message"] != user_message or state["scenario_id"] != scenario_id or (
                state["limits"] != limits or state["source_fingerprint"] != inputs or state["system_prompt"] != SYSTEM_PROMPT
            ):
                raise ValueError("Run ID already belongs to a different request, configuration or input version")
        else:
            state = {
                "version": 1, "run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
                "system_prompt": SYSTEM_PROMPT, "authorized_order_id": order_id,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message}],
                "steps": [], "step_number": 0, "tools_executed": [], "llm_calls": 0, "tokens_used": 0,
                "elapsed_seconds": 0.0, "limits": limits, "source_fingerprint": inputs,
                "pending_calls": [], "pending_index": 0, "execution_state": "running",
            }
            atomic_json(path, state)
        return _continue(state, path, root, client, clock, crash_after_tools, after_effect)


def resume_agent(run_id, *, root=ROOT, client=mock_llm, clock=time.monotonic, after_effect=None):
    root = Path(root)
    run_id = safe_run_id(run_id)
    path = root / "state" / f"{run_id}.json"
    with file_lock(path.with_suffix(".lock")):
        state = load_checkpoint(path, root, run_id)
        return _continue(state, path, root, client, clock, after_effect=after_effect)
