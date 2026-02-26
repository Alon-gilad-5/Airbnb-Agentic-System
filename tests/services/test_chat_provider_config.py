from __future__ import annotations

from app.config import load_settings


def test_load_settings_chat_provider_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LLM_CHAT_PROVIDER", raising=False)
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_CHAT_MODEL", raising=False)

    settings = load_settings()

    assert settings.llm_chat_provider == "llmod"
    assert settings.openrouter_base_url == "https://openrouter.ai/api/v1"
    assert settings.openrouter_chat_model == "openai/gpt-4o-mini"


def test_load_settings_invalid_chat_provider_falls_back_to_llmod(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CHAT_PROVIDER", "not-a-real-provider")

    settings = load_settings()

    assert settings.llm_chat_provider == "llmod"


def test_load_settings_reads_openrouter_values(monkeypatch) -> None:
    monkeypatch.setenv("LLM_CHAT_PROVIDER", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")
    monkeypatch.setenv("OPENROUTER_BASE_URL", "https://openrouter.example/v1")
    monkeypatch.setenv("OPENROUTER_CHAT_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("OPENROUTER_HTTP_REFERER", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_TITLE", "Airbnb Business Agent")

    settings = load_settings()

    assert settings.llm_chat_provider == "openrouter"
    assert settings.openrouter_api_key == "or-test-key"
    assert settings.openrouter_base_url == "https://openrouter.example/v1"
    assert settings.openrouter_chat_model == "openai/gpt-4o-mini"
    assert settings.openrouter_http_referer == "https://example.com"
    assert settings.openrouter_app_title == "Airbnb Business Agent"
