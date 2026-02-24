"""Compatibility wrapper for the mail pipeline.

Delegates to the LangChain-first MailPipeline defined in mail_agent
while preserving the original build/invoke interface for existing callers
and tests.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from app.agents.mail_agent import (
    ClassifiedEmail,
    MailPipeline,
    NO_MAIL_RESPONSE,
)
from app.schemas import StepLog
from app.services.gmail_service import EmailMessage

__all__ = [
    "NO_MAIL_RESPONSE",
    "MailState",
    "build_mail_graph",
]


class MailState(TypedDict, total=False):
    """Pipeline state schema kept for type-annotation compatibility."""

    prompt: str
    context: dict[str, Any]
    owner_action: dict[str, Any] | None
    raw_messages: list[EmailMessage]
    airbnb_messages: list[EmailMessage]
    classified_emails: list[ClassifiedEmail]
    mail_actions: list[dict[str, Any]]
    answer: str
    steps: Annotated[list[StepLog], operator.add]


def build_mail_graph(
    *,
    gmail_service: Any,
    chat_service: Any,
    config: Any,
) -> MailPipeline:
    """Build the mail pipeline with injected services.

    Returns a MailPipeline whose ``.invoke()`` method accepts and returns
    the same dict shape as the original StateGraph for full backward
    compatibility.
    """

    return MailPipeline(
        gmail_service=gmail_service,
        chat_service=chat_service,
        config=config,
    )
