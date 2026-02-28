from app.agents.analyst_agent import AnalystRunOutcome
import app.main as main_module
from app.schemas import (
    AnalysisCategoryBucket,
    AnalysisCategoricalItem,
    AnalysisExplainSelectionRequest,
    AnalysisNeighborPoint,
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
            analysis_category="review_scores",
            narrative=self.narrative,
            error=self.error,
            numeric_comparison=[
                AnalysisNumericItem(
                    label="Overall Rating",
                    column="review_scores_rating",
                    owner_value=4.5,
                    neighbor_avg=4.2,
                    neighbor_min=4.0,
                    neighbor_max=4.4,
                    neighbor_count=8,
                    neighbor_points=[
                        AnalysisNeighborPoint(listing_id="n1", listing_name="Neighbor One", value=4.0),
                        AnalysisNeighborPoint(listing_id="n2", listing_name="Neighbor Two", value=4.4),
                    ],
                    neighbor_min_points=[
                        AnalysisNeighborPoint(listing_id="n1", listing_name="Neighbor One", value=4.0),
                    ],
                    neighbor_max_points=[
                        AnalysisNeighborPoint(listing_id="n2", listing_name="Neighbor Two", value=4.4),
                    ],
                )
            ],
            categorical_comparison=[
                AnalysisCategoricalItem(
                    label="Room Type",
                    column="room_type",
                    owner_value="Private room",
                    neighbor_count=8,
                    buckets=[
                        AnalysisCategoryBucket(
                            value="Private room",
                            count=5,
                            pct=62.5,
                            listing_ids=["n1", "n2"],
                            listing_names=["Neighbor One", "Neighbor Two"],
                        )
                    ],
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

    def explain_selection(
        self,
        *,
        prompt: str,
        property_id: str,
        category: str,
        selection_type: str,
        metric_column: str,
        selection_payload: dict[str, object],
    ):
        self.last_prompt = prompt
        self.last_context = {
            "property_id": property_id,
            "category": category,
            "selection_type": selection_type,
            "metric_column": metric_column,
            "selection_payload": selection_payload,
        }
        return type(
            "ExplainResult",
            (),
            {
                "response": "Selection explanation.",
                "steps": [
                    StepLog(
                        module="analyst_agent.selection_explanation",
                        prompt={"metric_column": metric_column},
                        response={"text": "Selection explanation."},
                    )
                ],
            },
        )()


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
    assert response.analysis_category == "review_scores"
    assert len(response.numeric_comparison) == 1
    assert len(response.categorical_comparison) == 1
    assert response.numeric_comparison[0].neighbor_points[0].listing_id == "n1"
    assert response.categorical_comparison[0].buckets[0].listing_names == ["Neighbor One", "Neighbor Two"]
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


def test_explain_analysis_selection_returns_ok_payload(monkeypatch) -> None:
    openrouter_dummy = _DummyAnalystAgent()
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
        {"llmod": _DummyAnalystAgent(), "openrouter": openrouter_dummy},
    )

    response = main_module.explain_analysis_selection(
        AnalysisExplainSelectionRequest(
            property_id="42409434",
            prompt="How are my review scores compared to my neighbors?",
            category="review_scores",
            selection_type="numeric_extreme",
            metric_column="review_scores_rating",
            selection_payload={"extreme_type": "max", "selected_value": 5.0},
            llm_provider="openrouter",
        )
    )

    assert response.status == "ok"
    assert response.response == "Selection explanation."
    assert openrouter_dummy.last_context is not None
    assert openrouter_dummy.last_context["metric_column"] == "review_scores_rating"
    assert openrouter_dummy.last_context["selection_type"] == "numeric_extreme"


def test_explain_analysis_selection_returns_error_when_provider_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "chat_services_by_provider",
        {
            "llmod": type("DummyChatService", (), {"is_available": True})(),
            "openrouter": type("DummyChatService", (), {"is_available": False})(),
        },
    )
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")

    response = main_module.explain_analysis_selection(
        AnalysisExplainSelectionRequest(
            property_id="42409434",
            prompt="How are my review scores compared to my neighbors?",
            category="review_scores",
            selection_type="numeric_point",
            metric_column="review_scores_rating",
            selection_payload={"listing_id": "n1"},
            llm_provider="openrouter",
        )
    )

    assert response.status == "error"
    assert response.error is not None
    assert "openrouter" in response.error
