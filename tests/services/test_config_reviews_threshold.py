from __future__ import annotations

from app.config import load_settings


def test_load_settings_reads_reviews_relevance_threshold_from_env(monkeypatch) -> None:
    monkeypatch.setenv("REVIEWS_RELEVANCE_SCORE_THRESHOLD", "0.52")
    settings = load_settings()
    assert settings.reviews_relevance_score_threshold == 0.52

