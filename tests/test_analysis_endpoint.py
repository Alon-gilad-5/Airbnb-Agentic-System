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
    def __init__(self, *, error: str | None = None) -> None:
        self.error = error
        self.last_prompt: str | None = None
        self.last_context: dict[str, object] | None = None

    def analyze(self, prompt: str, context: dict[str, object] | None = None) -> AnalystRunOutcome:
        self.last_prompt = prompt
        self.last_context = context
        return AnalystRunOutcome(
            narrative="Benchmark summary.",
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
                    response={"text": "Benchmark summary."},
                )
            ],
        )


def test_run_analysis_returns_ok_payload(monkeypatch) -> None:
    dummy = _DummyAnalystAgent()
    monkeypatch.setattr(main_module, "analyst_agent", dummy)
    monkeypatch.setattr(main_module.settings.active_owner, "property_id", "42409434")

    response = main_module.run_analysis(AnalysisRequest(category="review_scores"))

    assert response.status == "ok"
    assert response.error is None
    assert response.response == "Benchmark summary."
    assert len(response.numeric_comparison) == 1
    assert len(response.categorical_comparison) == 1
    assert dummy.last_context is not None
    assert dummy.last_context["property_id"] == "42409434"
    assert dummy.last_context["analysis_category"] == "review_scores"


def test_run_analysis_maps_agent_error_to_error_response(monkeypatch) -> None:
    dummy = _DummyAnalystAgent(error="Listing store unavailable.")
    monkeypatch.setattr(main_module, "analyst_agent", dummy)

    response = main_module.run_analysis(
        AnalysisRequest(property_id="10046908", category="property_specs")
    )

    assert response.status == "error"
    assert response.error == "Listing store unavailable."
    assert response.response is None
