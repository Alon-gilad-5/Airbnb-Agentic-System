"""Tests for the mail agent: classification, policy rules, and HITL flows.

Covers Airbnb-only filtering, email classification accuracy, threshold-based
policy decisions, and owner action handling -- all with dummy services.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from app.agents.mail_agent import (
    CATEGORY_GUEST_MESSAGE,
    CATEGORY_LEAVE_REVIEW,
    CATEGORY_NEW_PROPERTY_REVIEW,
    CATEGORY_NON_AIRBNB,
    CATEGORY_UNSUPPORTED_AIRBNB,
    NO_MAIL_RESPONSE,
    MailAgent,
    MailAgentConfig,
    _classify_email,
    _extract_guest_name,
    _extract_rating,
    _score_importance,
)
from app.services.gmail_service import EmailMessage, GmailService

# -- Dummy services -----------------------------------------------------------


@dataclass
class DummyChatService:
    is_available: bool = True
    model: str = "test-model"
    _response: str = "Test draft reply."

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        if not self.is_available:
            raise RuntimeError("Chat unavailable")
        return self._response


def _make_gmail(enabled: bool = True) -> GmailService:
    return GmailService(
        enabled=enabled,
        gauth_path="__nonexistent_test_gauth__.json",
        accounts_path="__nonexistent_test_accounts__.json",
    )


def _make_agent(
    *,
    gmail_enabled: bool = True,
    chat_available: bool = True,
    chat_response: str = "Test draft reply.",
    bad_review_threshold: int = 3,
) -> MailAgent:
    return MailAgent(
        gmail_service=_make_gmail(enabled=gmail_enabled),
        chat_service=DummyChatService(is_available=chat_available, _response=chat_response),
        config=MailAgentConfig(bad_review_threshold=bad_review_threshold),
    )


def _make_email(
    *,
    id: str = "test-001",
    sender: str = "no-reply@airbnb.com",
    subject: str = "Test email",
    body: str = "Test body",
) -> EmailMessage:
    return EmailMessage(
        id=id,
        thread_id="thread-001",
        sender=sender,
        recipient="owner@example.com",
        subject=subject,
        snippet=body[:80],
        body=body,
        date="2026-02-20T10:00:00Z",
    )


# -- Module-level helper tests ------------------------------------------------


MAIL_MODULE_NAMES = {
    "mail_agent.fetch_inbox",
    "mail_agent.airbnb_filter",
    "mail_agent.classify",
    "mail_agent.policy",
    "mail_agent.guest_message_policy",
    "mail_agent.leave_review_policy",
    "mail_agent.property_review_policy",
    "mail_agent.answer_generation",
}


class TestRatingExtraction:
    def test_slash_notation(self) -> None:
        assert _extract_rating("Rating: 3/5 stars") == 3

    def test_star_notation(self) -> None:
        assert _extract_rating("Guest left a 2 star review") == 2

    def test_colon_notation(self) -> None:
        assert _extract_rating("Rating: 4") == 4

    def test_no_rating(self) -> None:
        assert _extract_rating("No rating information here") is None

    def test_out_of_range(self) -> None:
        assert _extract_rating("Rating: 7/5 stars") is None


class TestGuestNameExtraction:
    def test_from_pattern(self) -> None:
        assert _extract_guest_name("New message from guest John") == "John"

    def test_possessive_pattern(self) -> None:
        assert _extract_guest_name("Sarah's stay has ended") == "Sarah"

    def test_left_review_pattern(self) -> None:
        assert _extract_guest_name("Mike left a review") == "Mike"

    def test_no_name(self) -> None:
        assert _extract_guest_name("General notification") is None


class TestClassification:
    def test_non_airbnb_filtered(self) -> None:
        gmail = _make_gmail()
        email = _make_email(sender="promo@newsletter.com", subject="Buy stuff")
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NON_AIRBNB
        assert cls.confidence == 1.0

    def test_guest_message(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New message from guest Alex",
            body="Hi, I have a question about parking.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_GUEST_MESSAGE
        assert cls.extracted_guest_name == "Alex"

    def test_leave_review_request(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Leave a review for your guest Sarah",
            body="Sarah's stay has ended. Share your experience hosting them.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_LEAVE_REVIEW

    def test_new_property_review_good(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New review from guest Lisa: 5 stars",
            body="Lisa left a review of your property. Rating: 5/5 stars. Amazing!",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NEW_PROPERTY_REVIEW
        assert cls.extracted_rating == 5
        assert cls.extracted_guest_name == "Lisa"

    def test_new_property_review_bad(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="New review from guest Mike: 2 stars",
            body="Mike left a review. Rating: 2/5 stars. Not clean.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_NEW_PROPERTY_REVIEW
        assert cls.extracted_rating == 2

    def test_unsupported_airbnb_email(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Your monthly earnings summary",
            body="Here is your payment report for February.",
        )
        cls = _classify_email(email, gmail)
        assert cls.category == CATEGORY_UNSUPPORTED_AIRBNB


class TestImportanceScoring:
    def test_high_importance_urgent(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Urgent message from guest",
            body="I'm locked out and can't get in!",
        )
        cls = _classify_email(email, gmail)
        assert _score_importance(cls) == "high"

    def test_low_importance_thanks(self) -> None:
        gmail = _make_gmail()
        email = _make_email(
            subject="Message from guest Bob",
            body="Just wanted to say thanks for the wonderful stay!",
        )
        cls = _classify_email(email, gmail)
        assert _score_importance(cls) == "low"


# -- Agent-level integration tests ---------------------------------------------


class TestMailAgentDisabled:
    def test_disabled_returns_config_message(self) -> None:
        agent = _make_agent(gmail_enabled=False)
        result = agent.run("check my inbox")
        assert "not configured" in result.response.lower() or "not configured" in result.response
        assert len(result.steps) >= 1
        assert result.steps[0].module == "mail_agent.fetch_inbox"

    def test_inbox_summary_empty_when_disabled(self) -> None:
        agent = _make_agent(gmail_enabled=False)
        items = agent.get_inbox_summary()
        assert items == []


class TestMailAgentDemoMode:
    def test_processes_demo_inbox(self) -> None:
        agent = _make_agent()
        result = agent.run("check my inbox")
        assert result.response is not None
        assert len(result.response) > 0
        assert len(result.steps) >= 1

        module_names = {s.module for s in result.steps}
        assert "mail_agent.fetch_inbox" in module_names
        assert "mail_agent.airbnb_filter" in module_names
        assert "mail_agent.classify" in module_names

    def test_step_shapes_are_valid(self) -> None:
        agent = _make_agent()
        result = agent.run("check my inbox")
        for step in result.steps:
            assert step.module in MAIL_MODULE_NAMES, f"Unexpected module: {step.module}"
            assert isinstance(step.prompt, dict)
            assert isinstance(step.response, dict)

    def test_non_airbnb_filtered_out(self) -> None:
        agent = _make_agent()
        items = agent.get_inbox_summary()
        categories = [item["category"] for item in items]
        assert CATEGORY_NON_AIRBNB in categories
        airbnb_categories = [c for c in categories if c != CATEGORY_NON_AIRBNB]
        assert len(airbnb_categories) > 0

    def test_inbox_summary_returns_classified_items(self) -> None:
        agent = _make_agent()
        items = agent.get_inbox_summary()
        assert len(items) > 0
        for item in items:
            assert "email_id" in item
            assert "category" in item
            assert "confidence" in item
            assert item["confidence"] > 0


class TestMailAgentPolicies:
    def test_leave_review_awaits_rating_without_action(self) -> None:
        agent = _make_agent()
        result = agent.run("check inbox")
        found_leave_review = False
        for step in result.steps:
            if step.module == "mail_agent.leave_review_policy":
                found_leave_review = True
                assert step.response.get("action") == "awaiting_owner_rating"
        assert found_leave_review

    def test_leave_review_with_positive_rating(self) -> None:
        agent = _make_agent()
        owner_action = {"email_id": "demo-002", "rating": 4}
        result = agent.run_with_action("check inbox", owner_action=owner_action)
        found_review_draft = False
        for step in result.steps:
            if (
                step.module == "mail_agent.leave_review_policy"
                and step.response.get("action") == "review_draft_ready"
            ):
                found_review_draft = True
                assert step.response.get("rating_tier") == "positive"
        assert found_review_draft

    def test_leave_review_with_negative_rating(self) -> None:
        agent = _make_agent()
        owner_action = {"email_id": "demo-002", "rating": 2, "issues": ["Noise complaints"]}
        result = agent.run_with_action("check inbox", owner_action=owner_action)
        found_review_draft = False
        for step in result.steps:
            if (
                step.module == "mail_agent.leave_review_policy"
                and step.response.get("action") == "review_draft_ready"
            ):
                found_review_draft = True
                assert step.response.get("rating_tier") == "negative"
        assert found_review_draft

    def test_bad_property_review_requires_owner_consult(self) -> None:
        agent = _make_agent(bad_review_threshold=3)
        result = agent.run("check inbox")
        found_bad_review = False
        for step in result.steps:
            if (
                step.module == "mail_agent.property_review_policy"
                and step.response.get("is_bad_review") is True
            ):
                found_bad_review = True
                assert step.response.get("requires_owner") is True
        assert found_bad_review

    def test_good_property_review_gets_draft(self) -> None:
        agent = _make_agent()
        result = agent.run("check inbox")
        found_good_review = False
        for step in result.steps:
            if (
                step.module == "mail_agent.property_review_policy"
                and step.response.get("is_bad_review") is False
            ):
                found_good_review = True
                assert step.response.get("has_draft") is True
        assert found_good_review

    def test_chat_unavailable_uses_fallbacks(self) -> None:
        agent = _make_agent(chat_available=False)
        result = agent.run("check inbox")
        assert result.response is not None
        assert len(result.response) > 0


# -- Router integration --------------------------------------------------------


class TestRouterMailKeywords:
    def test_inbox_routes_to_mail(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, step = router.route("check my email inbox")
        assert decision.agent_name == "mail_agent"

    def test_mail_keyword(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, _ = router.route("any new mail?")
        assert decision.agent_name == "mail_agent"

    def test_gmail_keyword(self) -> None:
        from app.agents.router_agent import RouterAgent

        router = RouterAgent()
        decision, _ = router.route("check gmail for messages")
        assert decision.agent_name == "mail_agent"
