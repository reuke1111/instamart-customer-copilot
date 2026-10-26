"""Allowlisted refund tools with database-backed authorization and idempotency."""

from contextvars import ContextVar
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import re

from scripts.agent_storage import file_lock, read_jsonl, utc_now, write_jsonl
from scripts.common import ROOT, connect_readonly
from scripts.policy_rag import PolicyRetriever, default_retriever

_ACTIVE = ContextVar("refund_execution_context", default=None)


class ToolError(ValueError):
    """A tool request failed validation; no unsafe action should be attempted."""


def money(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ToolError("Amount must be a finite nonnegative number")
    try:
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or number != number.quantize(Decimal("0.01")):
            raise ToolError("Amount must be finite, nonnegative and expressed in whole paise")
    except InvalidOperation as error:
        raise ToolError("Invalid amount") from error
    return number


def duration(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ToolError("Delivery duration must be finite and nonnegative")
    return value


def get_order(order_id, *, root=ROOT):
    if not isinstance(order_id, str) or not re.fullmatch(r"ORD-\d{6}", order_id):
        raise ToolError("Invalid order identifier")
    context = _ACTIVE.get()
    if context is not None:
        context.authorize(order_id)
        root = context.root
    with connect_readonly(root) as connection:
        row = connection.execute(
            "SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?",
            (order_id,),
        ).fetchone()
    if row is None:
        return {"error": "order not found"}
    return dict(zip(["order_id", "status", "order_amount", "delivery_minutes"], row))


def search_policy(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 2000:
        raise ToolError("Policy query must be nonempty text of at most 2000 characters")
    context = _ACTIVE.get()
    retriever = context.retriever if context is not None else default_retriever()
    return {"results": retriever.retrieve(query, k=3)}


def calculate_refund(order_amount, delivery_minutes):
    amount, minutes = money(order_amount), duration(delivery_minutes)
    context = _ACTIVE.get()
    if context is not None:
        verified = context.eligible_order()
        if amount != money(verified["order_amount"]) or minutes != verified["delivery_minutes"]:
            raise ToolError("Calculation inputs differ from the verified order")
    return {"refund_inr": float(amount) if minutes > 45 else 0.0}


def issue_refund(order_id, refund_inr):
    context = _ACTIVE.get()
    if context is None:
        raise ToolError("Financial actions require an authorized run context")
    return context.record_action("issue_refund", order_id, refund_inr)


def escalate_to_human(order_id, refund_inr):
    context = _ACTIVE.get()
    if context is None:
        raise ToolError("Financial actions require an authorized run context")
    return context.record_action("escalate_to_human", order_id, refund_inr)


def spec(name, description, properties):
    return {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(properties), "additionalProperties": False},
    }


TOOLS_SPEC = [
    spec("get_order", "Read the requested order from the verified database.", {"order_id": {"type": "string"}}),
    spec("search_policy", "Retrieve the three most relevant policy chunks.", {"query": {"type": "string"}}),
    spec("calculate_refund", "Compute the verified order's full late-delivery refund or zero.",
         {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    spec("issue_refund", "Record an eligible simulated refund of at most INR 1000.",
         {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    spec("escalate_to_human", "Record an eligible refund above INR 1000 for human approval.",
         {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {function.__name__: function for function in [
    get_order, search_policy, calculate_refund, issue_refund, escalate_to_human,
]}


def validate_call(name, arguments):
    if not isinstance(name, str) or name not in TOOL_MAP:
        raise ToolError("Unknown tool")
    if not isinstance(arguments, dict):
        raise ToolError("Tool arguments must be an object")
    schema = next(item["parameters"] for item in TOOLS_SPEC if item["name"] == name)
    if set(arguments) != set(schema["required"]):
        raise ToolError("Missing or additional tool arguments")
    for key, value in arguments.items():
        kind = schema["properties"][key]["type"]
        if kind == "string" and not isinstance(value, str):
            raise ToolError("Tool text argument has the wrong type")
        if kind == "number" and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        ):
            raise ToolError("Tool numeric argument must be finite")
    if "order_id" in arguments and not re.fullmatch(r"ORD-\d{6}", arguments["order_id"]):
        raise ToolError("Invalid order identifier")
    if name == "search_policy" and (not arguments["query"].strip() or len(arguments["query"]) > 2000):
        raise ToolError("Invalid policy query")
    for key in ("order_amount", "refund_inr"):
        if key in arguments:
            money(arguments[key])
    if "delivery_minutes" in arguments:
        duration(arguments["delivery_minutes"])


@dataclass
class ToolExecutor:
    root: Path
    run_id: str
    scenario_id: str
    authorized_order_id: str | None
    retriever: PolicyRetriever

    def authorize(self, order_id):
        if self.authorized_order_id is None or order_id != self.authorized_order_id:
            raise ToolError("Order is outside this run's authorized request")

    def eligible_order(self):
        if self.authorized_order_id is None:
            raise ToolError("No order is authorized for this run")
        order = get_order(self.authorized_order_id, root=self.root)
        if "error" in order:
            raise ToolError("Order not found")
        if order["status"] != "completed":
            raise ToolError("Only completed orders can be refunded")
        duration(order["delivery_minutes"])
        return order

    def execute(self, name, arguments):
        validate_call(name, arguments)
        token = _ACTIVE.set(self)
        try:
            return TOOL_MAP[name](**arguments)
        finally:
            _ACTIVE.reset(token)

    def record_action(self, name, order_id, refund_inr):
        self.authorize(order_id)
        order = self.eligible_order()
        expected = money(order["order_amount"])
        requested = money(refund_inr)
        if order["delivery_minutes"] <= 45 or expected <= 0 or requested != expected:
            raise ToolError("Refund eligibility or amount does not match the verified order")
        if (name == "issue_refund") != (expected <= Decimal("1000.00")):
            raise ToolError("Refund must use the correct automatic or human approval route")
        path = Path(self.root) / "outputs/refund_ledger.jsonl"
        action_id = f"{self.run_id}:{name}:{order_id}"
        payload = {
            "action_id": action_id, "run_id": self.run_id, "scenario_id": self.scenario_id,
            "action": name, "tool_name": name, "order_id": order_id, "refund_inr": float(expected),
        }
        # Read, compare and write under one OS lock; atomic replacement prevents
        # partial JSONL records if the process exits during persistence.
        with file_lock(path.with_suffix(path.suffix + ".lock")):
            existing = read_jsonl(path)
            for event in existing:
                if event["action_id"] == action_id:
                    if any(event[key] != value for key, value in payload.items()):
                        raise ToolError("Idempotency key conflicts with an existing action")
                    return {"refund_inr": float(expected)}
                if event["run_id"] == self.run_id and event["order_id"] == order_id:
                    raise ToolError("This run already recorded a different financial action")
            write_jsonl(path, [*existing, {"timestamp": utc_now(), **payload}])
        return {"refund_inr": float(expected)}
