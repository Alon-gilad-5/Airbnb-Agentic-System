from __future__ import annotations

from app.config import load_settings


def test_secondary_profile_absent_when_property_id_missing(monkeypatch) -> None:
    monkeypatch.delenv("SECONDARY_PROPERTY_ID", raising=False)
    monkeypatch.delenv("SECONDARY_PROPERTY_REGION", raising=False)
    monkeypatch.delenv("SECONDARY_PROPERTY_LAT", raising=False)
    monkeypatch.delenv("SECONDARY_PROPERTY_LON", raising=False)
    monkeypatch.delenv("SECONDARY_MAX_SCRAPE_REVIEWS", raising=False)

    settings = load_settings()

    assert settings.secondary_owner is None


def test_secondary_profile_parses_and_normalizes_region(monkeypatch) -> None:
    monkeypatch.setenv("ACTIVE_OWNER_ID", "owner-1")
    monkeypatch.setenv("ACTIVE_OWNER_NAME", "Primary Owner")
    monkeypatch.setenv("SECONDARY_PROPERTY_ID", "10046908")
    monkeypatch.setenv("SECONDARY_PROPERTY_NAME", "Cozy Vintage-Styled Unit w/ Patio")
    monkeypatch.setenv("SECONDARY_PROPERTY_CITY", "Los Angeles")
    monkeypatch.setenv("SECONDARY_PROPERTY_REGION", "LOS ANGELES")
    monkeypatch.setenv("SECONDARY_PROPERTY_LAT", "34.01542")
    monkeypatch.setenv("SECONDARY_PROPERTY_LON", "-118.29229")
    monkeypatch.setenv("SECONDARY_MAX_SCRAPE_REVIEWS", "5")

    settings = load_settings()

    assert settings.secondary_owner is not None
    assert settings.secondary_owner.owner_id == "owner-1"
    assert settings.secondary_owner.owner_name == "Primary Owner"
    assert settings.secondary_owner.property_id == "10046908"
    assert settings.secondary_owner.region == "los angeles"
    assert settings.secondary_owner.latitude == 34.01542
    assert settings.secondary_owner.longitude == -118.29229
    assert settings.secondary_owner.default_max_scrape_reviews == 5


def test_secondary_profile_invalid_numeric_values_fall_back(monkeypatch) -> None:
    monkeypatch.setenv("SECONDARY_PROPERTY_ID", "10046908")
    monkeypatch.setenv("SECONDARY_PROPERTY_LAT", "not-a-float")
    monkeypatch.setenv("SECONDARY_PROPERTY_LON", "not-a-float")
    monkeypatch.setenv("SECONDARY_MAX_SCRAPE_REVIEWS", "not-an-int")

    settings = load_settings()

    assert settings.secondary_owner is not None
    assert settings.secondary_owner.latitude is None
    assert settings.secondary_owner.longitude is None
    assert settings.secondary_owner.default_max_scrape_reviews is None
