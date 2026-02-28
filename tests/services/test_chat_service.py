from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService


def test_chat_service_builds_http_client_with_trust_env_disabled(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyChatOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(chat_service_module, "ChatOpenAI", DummyChatOpenAI)

    service = ChatService(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        model="gpt-test",
    )

    assert service.is_available is True
    http_client = captured.get("http_client")
    assert http_client is not None
    assert getattr(http_client, "_trust_env", None) is False
