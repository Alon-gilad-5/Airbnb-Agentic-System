from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import app.main as main_module
from app.agents.pricing_agent import PricingAgent, PricingAgentConfig, PricingRunOutcome
from app.schemas import PricingRecommendation, PricingRequest, PricingSignalSummary, StepLog


@dataclass
class _DummyChatService:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "Pricing narrative."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_available:
            raise RuntimeError("Chat service unavailable")
        return self._response


class _DummyProviders:
    def __init__(self, *, events: list[object] | None = None, weather_days: list[object] | None = None) -> None:
        self._events = events or []
        self._weather_days = weather_days or []

    def fetch_weather_forecast(self, **kwargs):
        return self._weather_days, {"status": "ok"}

    def fetch_ticketmaster_events(self, **kwargs):
        return self._events, {"status": "ok"}

    def fetch_us_public_holidays(self, **kwargs):
        return [], {"status": "ok"}


class _DummyNeighborStore:
    def __init__(self, neighbors: list[str] | None = None) -> None:
        self._neighbors = neighbors or ["n1", "n2", "n3"]

    def get_neighbors(self, property_id: str) -> list[str] | None:
        return list(self._neighbors)


class _DummyListingStore:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows

    def get_listings_by_ids(self, listing_ids: list[str], columns: list[str]) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        for row in self._rows:
            if row["id"] not in listing_ids:
                continue
            out.append({column: row.get(column) for column in ["id", "name", *columns]})
        return out


def _row(
    *,
    listing_id: str,
    price: str,
    rating: float,
    total_reviews: int | None,
    recent_reviews: int | None,
    reviews_per_month: float | None = None,
) -> dict[str, object]:
    return {
        "id": listing_id,
        "name": listing_id,
        "price": price,
        "review_scores_rating": rating,
        "review_scores_accuracy": rating,
        "review_scores_cleanliness": rating,
        "review_scores_checkin": rating,
        "review_scores_communication": rating,
        "review_scores_location": rating,
        "review_scores_value": rating,
        "property_type": "Apartment",
        "room_type": "Entire home/apt",
        "accommodates": 2,
        "bathrooms": 1,
        "bedrooms": 1,
        "beds": 1,
        "host_is_superhost": "t",
        "number_of_reviews": total_reviews,
        "number_of_reviews_ltm": total_reviews,
        "number_of_reviews_l30d": recent_reviews,
        "reviews_per_month": reviews_per_month,
    }


def _make_agent(rows: list[dict[str, object]], *, events: list[object] | None = None) -> PricingAgent:
    return PricingAgent(
        listing_store=_DummyListingStore(rows),
        neighbor_store=_DummyNeighborStore(neighbors=[row["id"] for row in rows[1:]]),
        market_data_providers=_DummyProviders(events=events),
        chat_service=_DummyChatService(is_available=False),
        config=PricingAgentConfig(),
    )


def test_pricing_review_volume_boosts_raise_when_quality_and_demand_support_it() -> None:
    agent = _make_agent(
        [
            _row(listing_id="owner", price="$100.00", rating=4.9, total_reviews=150, recent_reviews=5),
            _row(listing_id="n1", price="$120.00", rating=4.6, total_reviews=40, recent_reviews=1),
            _row(listing_id="n2", price="$118.00", rating=4.5, total_reviews=50, recent_reviews=1),
            _row(listing_id="n3", price="$122.00", rating=4.6, total_reviews=45, recent_reviews=1),
        ],
        events=[SimpleNamespace(popularity_hint="high")],
    )

    outcome = agent.recommend("What should I charge next weekend?", context={"property_id": "owner", "latitude": 1, "longitude": 1})

    assert outcome.error is None
    assert outcome.recommendation is not None
    assert outcome.recommendation.price_action == "raise"
    assert outcome.recommendation.price_change_pct is not None
    assert outcome.recommendation.price_change_pct > 6.0
    assert outcome.signals is not None
    assert outcome.signals.review_volume_strength == "above_market"


def test_pricing_review_volume_dampens_raise_when_review_base_is_thin() -> None:
    agent = _make_agent(
        [
            _row(listing_id="owner", price="$100.00", rating=4.9, total_reviews=4, recent_reviews=0, reviews_per_month=0.0),
            _row(listing_id="n1", price="$120.00", rating=4.6, total_reviews=80, recent_reviews=3),
            _row(listing_id="n2", price="$118.00", rating=4.5, total_reviews=90, recent_reviews=2),
            _row(listing_id="n3", price="$122.00", rating=4.6, total_reviews=70, recent_reviews=2),
        ],
        events=[SimpleNamespace(popularity_hint="high")],
    )

    outcome = agent.recommend("What should I charge next weekend?", context={"property_id": "owner", "latitude": 1, "longitude": 1})

    assert outcome.recommendation is not None
    assert outcome.recommendation.price_action == "raise"
    assert outcome.recommendation.price_change_pct is not None
    assert outcome.recommendation.price_change_pct < 6.0
    assert outcome.recommendation.confidence == "low"
    assert outcome.signals is not None
    assert outcome.signals.review_volume_strength == "below_market"


def test_pricing_strong_review_base_softens_lower_without_reversing_it() -> None:
    agent = _make_agent(
        [
            _row(listing_id="owner", price="$140.00", rating=4.4, total_reviews=160, recent_reviews=4),
            _row(listing_id="n1", price="$100.00", rating=4.5, total_reviews=40, recent_reviews=1),
            _row(listing_id="n2", price="$102.00", rating=4.6, total_reviews=50, recent_reviews=1),
            _row(listing_id="n3", price="$104.00", rating=4.5, total_reviews=45, recent_reviews=1),
        ],
    )

    outcome = agent.recommend("Should I lower my prices?", context={"property_id": "owner"})

    assert outcome.recommendation is not None
    assert outcome.recommendation.price_action == "lower"
    assert outcome.recommendation.price_change_pct is not None
    assert outcome.recommendation.price_change_pct > -6.0
    assert outcome.recommendation.price_change_pct < 0.0


def test_pricing_request_validation_accepts_horizon_and_mode() -> None:
    payload = PricingRequest(property_id="owner", horizon_days=7, price_mode="conservative", llm_provider="openrouter")
    assert payload.horizon_days == 7
    assert payload.price_mode == "conservative"
    assert payload.llm_provider == "openrouter"


class _DummyPricingAgent:
    def __init__(self) -> None:
        self.last_prompt: str | None = None
        self.last_context: dict[str, object] | None = None

    def recommend(self, prompt: str, *, context: dict[str, object] | None = None) -> PricingRunOutcome:
        self.last_prompt = prompt
        self.last_context = context
        return PricingRunOutcome(
            narrative="Raise nightly price from $100 to $106.",
            error=None,
            recommendation=PricingRecommendation(
                current_price=100.0,
                recommended_price=106.0,
                price_change_abs=6.0,
                price_change_pct=6.0,
                price_action="raise",
                confidence="medium",
                primary_reason="Below-market pricing with stronger review volume.",
                risk_note=None,
            ),
            signals=PricingSignalSummary(
                neighbor_avg_price=108.0,
                neighbor_min_price=100.0,
                neighbor_max_price=112.0,
                price_position_pct=25.0,
                review_score_gap=0.1,
                strongest_review_metric="review_scores_cleanliness",
                weakest_review_metric="review_scores_value",
                demand_signal_count=2,
                high_severity_signal_count=1,
                market_pressure="strong",
                owner_number_of_reviews=100,
                neighbor_avg_number_of_reviews=60.0,
                owner_recent_reviews_30d=3,
                neighbor_avg_recent_reviews_30d=1.0,
                review_volume_strength="above_market",
            ),
            steps=[StepLog(module="pricing_agent.answer_generation", prompt={"status": "ok"}, response={"text": "Raise nightly price from $100 to $106."})],
        )


def test_run_pricing_returns_review_volume_fields(monkeypatch) -> None:
    dummy = _DummyPricingAgent()
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(main_module, "pricing_agents_by_provider", {"llmod": dummy})
    monkeypatch.setattr(main_module, "pricing_agent", dummy)
    monkeypatch.setattr(main_module, "chat_services_by_provider", {"llmod": SimpleNamespace(is_available=True)})
    monkeypatch.setattr(main_module.settings, "pricing_enabled", True)
    monkeypatch.setattr(main_module.settings.active_owner, "property_id", "owner")

    response = main_module.run_pricing(PricingRequest(property_id="owner"))

    assert response.status == "ok"
    assert response.signals is not None
    assert response.signals.owner_number_of_reviews == 100
    assert response.signals.review_volume_strength == "above_market"
    assert dummy.last_context is not None
    assert dummy.last_context["property_id"] == "owner"


def test_run_pricing_rejects_horizon_above_configured_max(monkeypatch) -> None:
    dummy = _DummyPricingAgent()
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(main_module, "pricing_agents_by_provider", {"llmod": dummy})
    monkeypatch.setattr(main_module, "pricing_agent", dummy)
    monkeypatch.setattr(main_module, "chat_services_by_provider", {"llmod": SimpleNamespace(is_available=True)})
    monkeypatch.setattr(main_module.settings, "pricing_enabled", True)
    monkeypatch.setattr(main_module.settings, "pricing_max_horizon_days", 7)

    response = main_module.run_pricing(PricingRequest(property_id="owner", horizon_days=14))

    assert response.status == "error"
    assert response.error == "horizon_days must be <= configured PRICING_MAX_HORIZON_DAYS (7)."
    assert dummy.last_context is None


def test_run_pricing_does_not_apply_active_owner_coordinates_to_unknown_property(monkeypatch) -> None:
    dummy = _DummyPricingAgent()
    monkeypatch.setattr(main_module, "default_chat_provider", "llmod")
    monkeypatch.setattr(main_module, "pricing_agents_by_provider", {"llmod": dummy})
    monkeypatch.setattr(main_module, "pricing_agent", dummy)
    monkeypatch.setattr(main_module, "chat_services_by_provider", {"llmod": SimpleNamespace(is_available=True)})
    monkeypatch.setattr(main_module.settings, "pricing_enabled", True)
    monkeypatch.setattr(
        main_module,
        "property_profiles",
        {
            "primary": SimpleNamespace(
                property_id="owner",
                property_name="Known Property",
                latitude=32.0,
                longitude=34.0,
            )
        },
    )

    response = main_module.run_pricing(PricingRequest(property_id="unknown-property"))

    assert response.status == "ok"
    assert dummy.last_context is not None
    assert dummy.last_context["property_id"] == "unknown-property"
    assert "latitude" not in dummy.last_context
    assert "longitude" not in dummy.last_context
