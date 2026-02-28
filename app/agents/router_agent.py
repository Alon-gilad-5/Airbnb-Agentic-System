"""Routing component that selects which domain agent should execute the request."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, TypedDict
import operator

from langgraph.graph import END, StateGraph

from app.schemas import StepLog


@dataclass
class RouteDecision:
    """Router output with selected agent and explanation."""

    agent_name: str
    reason: str


class RouterState(TypedDict, total=False):
    prompt: str
    selected_agent: str
    reason: str
    steps: Annotated[list[StepLog], operator.add]


class RouterAgent:
    """Keyword-based router (cheap and deterministic; no LLM call needed).

    Exposed as both a callable class (for backward-compatible main.py usage)
    and a compiled LangGraph StateGraph for structural consistency.
    """

    name = "router_agent"

    _mail_keywords = {
        "email", "emails", "inbox", "mail", "mailbox", "gmail",
        "unread", "draft", "reply", "respond to review",
        "guest message", "leave review",
    }
    _reviews_keywords = {
        "review", "reviews", "guest", "guests", "wifi", "internet",
        "clean", "cleanliness", "check-in", "service", "noise",
        "host", "feedback", "rating",
        "neighbor", "neighbours", "neighbors", "neighbour",
        "comparison",
    }
    _analyst_keywords = {
        "benchmark",
        "competitive analysis",
        "analyze my scores",
        "analyse my scores",
        "analyze my property specs",
        "analyse my property specs",
        "compare my property specs",
        "compare my review scores",
        "how do i compare to neighbors",
        "how do i benchmark",
    }
    _market_watch_keywords = {
        "market", "intel", "weather", "forecast", "storm", "snow",
        "rain", "event", "events", "concert", "conference", "festival",
        "demand", "pricing", "price", "holiday", "nearby",
    }

    def __init__(self) -> None:
        self._graph = self._build_graph()

    def route(self, prompt: str) -> tuple[RouteDecision, StepLog]:
        """Select an agent and return a trace step aligned with architecture modules."""

        lowered = prompt.lower()
        if any(keyword in lowered for keyword in self._mail_keywords):
            decision = RouteDecision(
                agent_name="mail_agent",
                reason="Matched mail/inbox intent keywords.",
            )
        elif any(keyword in lowered for keyword in self._market_watch_keywords):
            decision = RouteDecision(
                agent_name="market_watch_agent",
                reason="Matched market/weather/event intent keywords.",
            )
        elif any(keyword in lowered for keyword in self._analyst_keywords):
            decision = RouteDecision(
                agent_name="analyst_agent",
                reason="Matched analyst/benchmark intent keywords.",
            )
        elif any(keyword in lowered for keyword in self._reviews_keywords):
            decision = RouteDecision(
                agent_name="reviews_agent",
                reason="Matched hospitality/review intent keywords.",
            )
        else:
            decision = RouteDecision(
                agent_name="reviews_agent",
                reason="Fallback route: defaulting to reviews_agent for unknown intent.",
            )

        step = StepLog(
            module=self.name,
            prompt={"user_prompt": prompt},
            response={"selected_agent": decision.agent_name, "reason": decision.reason},
        )
        return decision, step

    def _build_graph(self) -> Any:
        """Build a single-node LangGraph wrapper around keyword routing."""

        router_self = self

        def route_node(state: RouterState) -> dict[str, Any]:
            prompt = state["prompt"]
            decision, step = router_self.route(prompt)
            return {
                "selected_agent": decision.agent_name,
                "reason": decision.reason,
                "steps": [step],
            }

        builder = StateGraph(RouterState)
        builder.add_node("route_intent", route_node)
        builder.set_entry_point("route_intent")
        builder.add_edge("route_intent", END)
        return builder.compile()

    def route_via_graph(self, prompt: str) -> tuple[RouteDecision, StepLog]:
        """Alternative entry point that invokes routing through the compiled graph."""

        result = self._graph.invoke({"prompt": prompt, "steps": []})
        decision = RouteDecision(
            agent_name=result["selected_agent"],
            reason=result["reason"],
        )
        steps = result.get("steps", [])
        step = steps[0] if steps else StepLog(
            module=self.name,
            prompt={"user_prompt": prompt},
            response={"selected_agent": decision.agent_name, "reason": decision.reason},
        )
        return decision, step
