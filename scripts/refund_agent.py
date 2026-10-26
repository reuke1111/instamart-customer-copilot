"""Bounded ReAct loop with durable pending calls, checkpoints and resumability."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import time
import uuid

import mock_llm
from scripts.agent_storage import atomic_json, file_lock
from scripts.common import ROOT, load_json
from scripts.policy_rag import default_retriever
from scripts.refund_tools import TOOLS_SPEC, ToolError, ToolExecutor, money, validate_call

SYSTEM_PROMPT = (
    "AGENT. Use the provided tools to process this simulated refund. "
    "Never invent observations. Follow verified eligibility and amount checks. "
    "Return a final status and concise explanation."
)


class SimulatedCrash(RuntimeError):
    """Test/demo interruption after a successfully persisted observation."""


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
            except (ToolError, OSError, ValueError) as error:
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
        response = client.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1
        usage = response.usage.total_tokens
        if isinstance(usage, bool) or not isinstance(usage, int) or usage < 0:
            return finish("FALLBACK_INVALID_RESPONSE")
        state["tokens_used"] += usage
        # Strictly exceeds, as required; equality alone is not a violation.
        if state["tokens_used"] > state["limits"]["token_limit"]:
            return finish("FALLBACK_TOKEN_LIMIT")
        message = response.choices[0].message
        calls = message.tool_calls or []
        if not isinstance(calls, list):
            return finish("FALLBACK_INVALID_TOOL")
        if not calls:
            try:
                status = _verified_final(message.content, state["steps"])
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
    orders = sorted(set(re.findall(r"ORD-\d{6}", user_message)))
    if len(orders) > 1:
        raise ValueError("Each refund run must authorize at most one order")
    inputs = source_fingerprint(root)
    with file_lock(path.with_suffix(".lock")):
        if path.exists():
            state = load_json(path)
            if state["user_message"] != user_message or state["scenario_id"] != scenario_id or (
                state["limits"] != limits or state["source_fingerprint"] != inputs or state["system_prompt"] != SYSTEM_PROMPT
            ):
                raise ValueError("Run ID already belongs to a different request, configuration or input version")
        else:
            state = {
                "version": 1, "run_id": run_id, "scenario_id": scenario_id, "user_message": user_message,
                "system_prompt": SYSTEM_PROMPT, "authorized_order_id": orders[0] if orders else None,
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
        state = load_json(path)
        if state["version"] != 1 or state["run_id"] != run_id or (
            state["source_fingerprint"] != source_fingerprint(root) or state["system_prompt"] != SYSTEM_PROMPT
        ):
            raise ValueError("Checkpoint identity or source version does not match")
        _validate_limits(**state["limits"])
        return _continue(state, path, root, client, clock, after_effect=after_effect)
