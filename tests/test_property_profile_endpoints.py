from __future__ import annotations

from types import SimpleNamespace

import app.main as main_module
from app.config import ActiveOwnerContext
from app.schemas import MarketWatchRunRequest
from app.services.market_alert_store import MarketAlertRecord


class _DummyMarketAlertStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.count_since_calls: list[dict[str, object]] = []

    def list_latest_alerts(
        self,
        *,
        owner_id: str | None,
        property_id: str | None,
        limit: int,
    ) -> list[MarketAlertRecord]:
        self.calls.append(
            {
                "owner_id": owner_id,
                "property_id": property_id,
                "limit": limit,
            }
        )
        return []

    def count_since(
        self,
        since_utc: str,
        *,
        owner_id: str | None = None,
        property_id: str | None = None,
    ) -> int:
        self.count_since_calls.append(
            {
                "since_utc": since_utc,
                "owner_id": owner_id,
                "property_id": property_id,
            }
        )
        return 11


class _DummyMarketWatchAgent:
    def __init__(self) -> None:
        self.last_context: dict[str, object] | None = None

    def run_autonomous(self, context: dict[str, object] | None = None):
        self.last_context = context
        return SimpleNamespace(
            response="ok",
            inserted_count=0,
            steps=[],
        )


def _active_owner(
    *,
    owner_id: str = "owner-1",
    owner_name: str = "Owner One",
    property_id: str = "property-1",
) -> ActiveOwnerContext:
    return ActiveOwnerContext(
        owner_id=owner_id,
        owner_name=owner_name,
        property_id=property_id,
        property_name="Primary Property",
        city="Los Angeles",
        region="los angeles",
        latitude=34.05,
        longitude=-118.24,
        google_maps_url=None,
        tripadvisor_url=None,
        default_max_scrape_reviews=5,
    )


def test_property_profiles_endpoint_returns_primary_and_secondary(monkeypatch) -> None:
    monkeypatch.setattr(
        main_module,
        "property_profiles",
        {
            "primary": _active_owner(property_id="42409434"),
            "secondary": ActiveOwnerContext(
                owner_id="owner-1",
                owner_name="Owner One",
                property_id="10046908",
                property_name="Cozy Vintage-Styled Unit w/ Patio",
                city="Los Angeles",
                region="LOS ANGELES",
                latitude=34.01542,
                longitude=-118.29229,
                google_maps_url=None,
                tripadvisor_url=None,
                default_max_scrape_reviews=5,
            ),
        },
    )

    response = main_module.property_profiles_endpoint()

    assert response.default_profile_id == "primary"
    assert [p.profile_id for p in response.profiles] == ["primary", "secondary"]
    assert response.profiles[1].region == "los angeles"


def test_market_watch_run_uses_payload_override_context(monkeypatch) -> None:
    dummy_agent = _DummyMarketWatchAgent()

    monkeypatch.setattr(main_module.settings, "market_watch_enabled", True)
    monkeypatch.setattr(main_module.settings, "active_owner", _active_owner())
    monkeypatch.setattr(main_module, "market_watch_agent", dummy_agent)
    monkeypatch.setattr(main_module, "_assert_market_watch_trigger_authorized", lambda **kwargs: None)

    result = main_module.market_watch_run(
        payload=MarketWatchRunRequest(
            property_id="10046908",
            property_name="Cozy Vintage-Styled Unit w/ Patio",
            city="Los Angeles",
            region="LOS ANGELES",
            latitude=34.01542,
            longitude=-118.29229,
        ),
        x_market_watch_secret=None,
        authorization=None,
    )

    assert result.status == "ok"
    assert dummy_agent.last_context is not None
    assert dummy_agent.last_context["property_id"] == "10046908"
    assert dummy_agent.last_context["region"] == "los angeles"
    assert dummy_agent.last_context["owner_id"] == "owner-1"


def test_market_watch_alerts_respects_optional_scope_overrides(monkeypatch) -> None:
    dummy_store = _DummyMarketAlertStore()

    monkeypatch.setattr(main_module.settings, "active_owner", _active_owner())
    monkeypatch.setattr(main_module, "market_alert_store", dummy_store)

    main_module.market_watch_alerts(
        limit=10,
        owner_id="owner-override",
        property_id="property-override",
    )
    main_module.market_watch_alerts(limit=5, owner_id=None, property_id=None)

    assert dummy_store.calls[0]["owner_id"] == "owner-override"
    assert dummy_store.calls[0]["property_id"] == "property-override"
    assert dummy_store.calls[1]["owner_id"] == "owner-1"
    assert dummy_store.calls[1]["property_id"] == "property-1"


def test_nav_badges_passes_property_scope_to_market_count(monkeypatch) -> None:
    dummy_store = _DummyMarketAlertStore()

    monkeypatch.setattr(
        main_module,
        "notification_store",
        SimpleNamespace(count_pending=lambda: 4),
    )
    monkeypatch.setattr(main_module, "market_alert_store", dummy_store)

    response = main_module.nav_badges(
        market_since="2026-02-26T00:00:00Z",
        property_id=" 10046908 ",
    )

    assert response == {"mail_count": 4, "market_count": 11}
    assert dummy_store.count_since_calls == [
        {
            "since_utc": "2026-02-26T00:00:00Z",
            "owner_id": None,
            "property_id": "10046908",
        }
    ]
