"""Routing component that selects which domain agent should execute the request."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas import StepLog


@dataclass
class RouteDecision:
    """Router output with selected agent and explanation."""

    agent_name: str
    reason: str


class RouterAgent:
    """Keyword-based router (cheap and deterministic; no LLM call needed)."""

    name = "router_agent"

    _reviews_keywords = {
        "review",
        "reviews",
        "guest",
        "guests",
        "wifi",
        "internet",
        "clean",
        "cleanliness",
        "check-in",
        "service",
        "noise",
        "host",
        "feedback",
        "rating",
    }
    _market_watch_keywords = {
        "market",
        "intel",
        "weather",
        "forecast",
        "storm",
        "snow",
        "rain",
        "event",
        "events",
        "concert",
        "conference",
        "festival",
        "demand",
        "pricing",
        "price",
        "holiday",
        "nearby",
    }

    def route(self, prompt: str) -> tuple[RouteDecision, StepLog]:
        """Select an agent and return a trace step aligned with architecture modules."""

        lowered = prompt.lower()
        if any(keyword in lowered for keyword in self._market_watch_keywords):
            decision = RouteDecision(
                agent_name="market_watch_agent",
                reason="Matched market/weather/event intent keywords.",
            )
        elif any(keyword in lowered for keyword in self._reviews_keywords):
            decision = RouteDecision(
                agent_name="reviews_agent",
                reason="Matched hospitality/review intent keywords.",
            )
        else:
            # Keep reviews as deterministic default for unknown prompts.
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
