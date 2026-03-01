from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

from app.agents.base import AgentResult
from app.schemas import ExecuteRequest, StepLog
import app.main as main_module


@dataclass
class _DummyChatService:
    is_available: bool


class _DummyAgent:
    def __init__(self, label: str) -> None:
        self._label = label

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        return AgentResult(response=self._label, steps=[])


def _patch_router(monkeypatch, *, agent_name: str) -> None:
    def _route(prompt: str):
        decision = SimpleNamespace(agent_name=agent_name, reason="test route")
        step = StepLog(
            module="router_agent",
            prompt={"prompt": prompt},
            response={"selected_agent": agent_name, "reason": "test route"},
        )
        return decision, step

    monkeypatch.setattr(main_module.router_agent, "route", _route)


def _patch_provider_state(monkeypatch, *, openrouter_available: bool = True) -> None:
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(
        main_module,
        "chat_services_by_provider",
        {
            "llmod": _DummyChatService(is_available=True),
            "openrouter": _DummyChatService(is_available=openrouter_available),
        },
    )
    monkeypatch.setattr(
        main_module,
        "reviews_agents_by_provider",
        {
            "llmod": _DummyAgent("reviews:llmod"),
            "openrouter": _DummyAgent("reviews:openrouter"),
        },
    )
    monkeypatch.setattr(
        main_module,
        "mail_agents_by_provider",
        {
            "llmod": _DummyAgent("mail:llmod"),
            "openrouter": _DummyAgent("mail:openrouter"),
        },
    )
    monkeypatch.setattr(
        main_module,
        "analysis_agents_by_provider",
        {
            "llmod": _DummyAgent("analysis:llmod"),
            "openrouter": _DummyAgent("analysis:openrouter"),
        },
    )
    monkeypatch.setattr(
        main_module,
        "pricing_agents_by_provider",
        {
            "llmod": _DummyAgent("pricing:llmod"),
            "openrouter": _DummyAgent("pricing:openrouter"),
        },
    )
    monkeypatch.setattr(
        main_module,
        "agent_registry",
        {
            "reviews_agent": _DummyAgent("reviews:default"),
            "market_watch_agent": _DummyAgent("market_watch:deterministic"),
            "mail_agent": _DummyAgent("mail:default"),
            "analyst_agent": _DummyAgent("analysis:default"),
            "pricing_agent": _DummyAgent("pricing:default"),
        },
    )
    monkeypatch.setattr(main_module.settings, "market_watch_enabled", True)
    monkeypatch.setattr(main_module.settings, "mail_enabled", True)
    monkeypatch.setattr(main_module.settings, "pricing_enabled", True)


def test_execute_uses_default_provider_when_no_override(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)
    _patch_router(monkeypatch, agent_name="reviews_agent")

    result = main_module.execute(ExecuteRequest(prompt="wifi"))

    assert result.status == "ok"
    assert result.response == "reviews:llmod"


def test_execute_uses_openrouter_when_explicit_override(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)
    _patch_router(monkeypatch, agent_name="reviews_agent")

    result = main_module.execute(
        ExecuteRequest(prompt="wifi", llm_provider="openrouter")
    )

    assert result.status == "ok"
    assert result.response == "reviews:openrouter"


def test_execute_returns_error_when_explicit_provider_unavailable(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=False)
    _patch_router(monkeypatch, agent_name="reviews_agent")

    result = main_module.execute(
        ExecuteRequest(prompt="wifi", llm_provider="openrouter")
    )

    assert result.status == "error"
    assert result.error is not None
    assert "openrouter" in result.error


def test_execute_market_watch_ignores_provider_override(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=False)
    _patch_router(monkeypatch, agent_name="market_watch_agent")

    result = main_module.execute(
        ExecuteRequest(prompt="events", llm_provider="openrouter")
    )

    assert result.status == "ok"
    assert result.response == "market_watch:deterministic"


def test_execute_uses_openrouter_when_analyst_override_is_explicit(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)
    _patch_router(monkeypatch, agent_name="analyst_agent")

    result = main_module.execute(
        ExecuteRequest(prompt="benchmark my scores", llm_provider="openrouter")
    )

    assert result.status == "ok"
    assert result.response == "analysis:openrouter"


def test_execute_uses_openrouter_when_pricing_override_is_explicit(monkeypatch) -> None:
    _patch_provider_state(monkeypatch, openrouter_available=True)
    _patch_router(monkeypatch, agent_name="pricing_agent")

    result = main_module.execute(
        ExecuteRequest(prompt="What should I charge next weekend?", llm_provider="openrouter")
    )

    assert result.status == "ok"
    assert result.response == "pricing:openrouter"
