from app.agents.analyst_agent import AnalystRunOutcome
import app.main as main_module
from app.schemas import (
    AnalysisCategoryBucket,
    AnalysisCategoricalItem,
    AnalysisNumericItem,
    AnalysisRequest,
    StepLog,
)


class _DummyAnalystAgent:
    def __init__(self, *, error: str | None = None, narrative: str = "Benchmark summary.") -> None:
        self.error = error
        self.narrative = narrative
        self.last_prompt: str | None = None
        self.last_context: dict[str, object] | None = None

    def analyze(self, prompt: str, context: dict[str, object] | None = None) -> AnalystRunOutcome:
        self.last_prompt = prompt
        self.last_context = context
        return AnalystRunOutcome(
            narrative=self.narrative,
            error=self.error,
            numeric_comparison=[
                AnalysisNumericItem(
                    column="review_scores_rating",
                    owner_value=4.5,
                    neighbor_avg=4.2,
                    neighbor_min=4.0,
                    neighbor_max=4.4,
                    neighbor_count=8,
                )
            ],
            categorical_comparison=[
                AnalysisCategoricalItem(
                    column="room_type",
                    owner_value="Private room",
                    neighbor_count=8,
                    buckets=[AnalysisCategoryBucket(value="Private room", count=5, pct=62.5)],
                )
            ],
            steps=[
                StepLog(
                    module="analyst_agent.answer_generation",
                    prompt={"status": "ok"},
                    response={"text": self.narrative},
                )
            ],
        )


def _patch_default_analysis_agent(monkeypatch, agent: _DummyAnalystAgent) -> None:
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(
        main_module,
        "analysis_agents_by_provider",
        {"llmod": agent},
    )


def test_run_analysis_returns_ok_payload(monkeypatch) -> None:
    dummy = _DummyAnalystAgent()
    _patch_default_analysis_agent(monkeypatch, dummy)
    monkeypatch.setattr(main_module.settings.active_owner, "property_id", "42409434")

    response = main_module.run_analysis(AnalysisRequest(category="review_scores"))

    assert response.status == "ok"
    assert response.error is None
    assert response.response == "Benchmark summary."
    assert len(response.numeric_comparison) == 1
    assert len(response.categorical_comparison) == 1
    assert dummy.last_context is not None
    assert dummy.last_context["property_id"] == "42409434"
    assert response.status == "ok"


def test_run_analysis_uses_prompt_and_provider_override(monkeypatch) -> None:
    llmod_dummy = _DummyAnalystAgent(narrative="llmod summary")
    openrouter_dummy = _DummyAnalystAgent(narrative="openrouter summary")

    monkeypatch.setattr(
        main_module,
        "chat_services_by_provider",
        {
            "llmod": type("DummyChatService", (), {"is_available": True})(),
            "openrouter": type("DummyChatService", (), {"is_available": True})(),
        },
    )
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(
        main_module,
        "analysis_agents_by_provider",
        {
            "llmod": llmod_dummy,
            "openrouter": openrouter_dummy,
        },
    )

    response = main_module.run_analysis(
        AnalysisRequest(
            property_id="10046908",
            prompt="How do I compare on cleanliness and value?",
            llm_provider="openrouter",
        )
    )

    assert response.status == "ok"
    assert response.response == "openrouter summary"
    assert openrouter_dummy.last_prompt == "How do I compare on cleanliness and value?"
    assert openrouter_dummy.last_context == {"property_id": "10046908"}
    assert llmod_dummy.last_prompt is None


def test_run_analysis_returns_error_when_explicit_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "chat_services_by_provider",
        {
            "llmod": type("DummyChatService", (), {"is_available": True})(),
            "openrouter": type("DummyChatService", (), {"is_available": False})(),
        },
    )
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")

    response = main_module.run_analysis(
        AnalysisRequest(prompt="Benchmark my scores", llm_provider="openrouter")
    )

    assert response.status == "error"
    assert response.error is not None
    assert "openrouter" in response.error

def test_run_analysis_maps_agent_error_to_error_response(monkeypatch) -> None:
    dummy = _DummyAnalystAgent(error="Listing store unavailable.")
    _patch_default_analysis_agent(monkeypatch, dummy)

    response = main_module.run_analysis(
        AnalysisRequest(property_id="10046908", category="property_specs")
    )

    assert response.status == "error"
    assert response.error == "Listing store unavailable."
    assert response.response is None
    assert dummy.last_context is not None
    assert dummy.last_context["analysis_category"] == "property_specs"
