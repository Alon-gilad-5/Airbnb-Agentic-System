"""Gmail adapter service for email ingestion and draft management.

Supports three modes:
- **MCP**: connects to ``mcp-server-google-workspace`` via stdio and
  calls Gmail tools through the Model Context Protocol.
- **Demo**: returns sample Airbnb emails for development/testing.
- **Disabled**: returns empty results when mail feature is off.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AIRBNB_SENDER_DOMAINS = ["airbnb.com", "airbnbmail.com", "airbnb.co"]

_DEMO_EMAILS: list[dict[str, Any]] = [
    {
        "id": "demo-001",
        "thread_id": "thread-001",
        "from": "automated@airbnb.com",
        "to": "owner@example.com",
        "subject": "New message from guest John",
        "snippet": "Hi, I wanted to ask about early check-in options for my stay next week...",
        "body": (
            "Hi,\n\n"
            "I wanted to ask about early check-in options for my stay next week. "
            "We're arriving on an early flight and wondering if the property would "
            "be available before the standard 3pm check-in time.\n\n"
            "Thanks,\nJohn"
        ),
        "date": "2026-02-22T10:30:00Z",
        "labels": ["INBOX", "UNREAD"],
    },
    {
        "id": "demo-002",
        "thread_id": "thread-002",
        "from": "no-reply@airbnb.com",
        "to": "owner@example.com",
        "subject": "Leave a review for your guest Sarah",
        "snippet": "Sarah's stay has ended. Share your experience hosting them...",
        "body": (
            "Sarah's stay at your property has ended.\n\n"
            "Please leave a review to help the Airbnb community. "
            "Your review helps other hosts know what to expect.\n\n"
            "Rate your experience and share feedback about your guest."
        ),
        "date": "2026-02-21T14:00:00Z",
        "labels": ["INBOX", "UNREAD"],
    },
    {
        "id": "demo-003",
        "thread_id": "thread-003",
        "from": "no-reply@airbnb.com",
        "to": "owner@example.com",
        "subject": "New review from guest Mike: 2 stars",
        "snippet": "Mike left a 2-star review of your property. The place was not clean...",
        "body": (
            "Mike left a review of your property.\n\n"
            "Rating: 2/5 stars\n\n"
            '"The place was not clean when we arrived. Found dirty towels in the '
            "bathroom and the kitchen had unwashed dishes. Location was okay but "
            'the cleanliness issues ruined our stay."\n\n'
            "You can respond to this review within 30 days."
        ),
        "date": "2026-02-20T09:15:00Z",
        "labels": ["INBOX", "UNREAD"],
    },
    {
        "id": "demo-004",
        "thread_id": "thread-004",
        "from": "no-reply@airbnb.com",
        "to": "owner@example.com",
        "subject": "New review from guest Lisa: 5 stars",
        "snippet": "Lisa left a 5-star review: Amazing place! Everything was perfect...",
        "body": (
            "Lisa left a review of your property.\n\n"
            "Rating: 5/5 stars\n\n"
            '"Amazing place! Everything was perfect, from the cozy decor to the '
            "spotless kitchen. The host was incredibly responsive and helpful. "
            'Would definitely book again!"\n\n'
            "You can respond to this review within 30 days."
        ),
        "date": "2026-02-19T16:45:00Z",
        "labels": ["INBOX", "UNREAD"],
    },
    {
        "id": "demo-005",
        "thread_id": "thread-005",
        "from": "promo@newsletter.com",
        "to": "owner@example.com",
        "subject": "Weekly deals just for you!",
        "snippet": "Check out this week's best deals on travel accessories...",
        "body": "Check out this week's best deals on travel accessories and hosting supplies!",
        "date": "2026-02-18T08:00:00Z",
        "labels": ["INBOX", "UNREAD"],
    },
]


@dataclass
class EmailMessage:
    """Standardized email representation."""

    id: str
    thread_id: str
    sender: str
    recipient: str
    subject: str
    snippet: str
    body: str
    date: str
    labels: list[str] = field(default_factory=list)


@dataclass
class DraftResult:
    """Result of creating a draft email."""

    draft_id: str
    thread_id: str | None
    subject: str
    body: str
    status: str


# ---------------------------------------------------------------------------
# MCP Client wrapper (async internals, sync public API)
# ---------------------------------------------------------------------------


class _McpGmailClient:
    """Persistent MCP client that spawns ``mcp-server-google-workspace``
    as a subprocess and communicates via stdio transport.

    Runs an asyncio event loop on a background daemon thread so the rest
    of the synchronous codebase can call ``call_tool()`` without changes.
    """

    def __init__(self, *, gauth_path: str, accounts_path: str, credentials_dir: str) -> None:
        self._gauth_path = gauth_path
        self._accounts_path = accounts_path
        self._credentials_dir = credentials_dir
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()
        self._session: Any = None
        self._exit_stack: AsyncExitStack | None = None
        self._ready = threading.Event()
        self._init_error: str | None = None
        asyncio.run_coroutine_threadsafe(self._connect(), self._loop)

    async def _connect(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        npx_path = shutil.which("npx")
        if not npx_path:
            self._init_error = "npx not found on PATH — Node.js is required for Gmail MCP"
            self._ready.set()
            return

        server_params = StdioServerParameters(
            command=npx_path,
            args=[
                "mcp-server-google-workspace",
                "--gauth-file", self._gauth_path,
                "--accounts-file", self._accounts_path,
                "--credentials-dir", self._credentials_dir,
            ],
            env={**os.environ},
        )

        try:
            self._exit_stack = AsyncExitStack()
            transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = transport
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await self._session.initialize()
            logger.info("Gmail MCP session initialized successfully")
        except Exception as exc:
            self._init_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Gmail MCP client connection failed: %s", self._init_error)
        finally:
            self._ready.set()

    @property
    def is_connected(self) -> bool:
        self._ready.wait(timeout=60)
        return self._session is not None and self._init_error is None

    @property
    def init_error(self) -> str | None:
        self._ready.wait(timeout=60)
        return self._init_error

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float = 30) -> Any:
        """Call an MCP tool synchronously (blocks until result)."""
        if not self.is_connected:
            raise RuntimeError(f"Gmail MCP not connected: {self._init_error}")
        future = asyncio.run_coroutine_threadsafe(
            self._session.call_tool(name, arguments), self._loop
        )
        return future.result(timeout=timeout)

    def close(self) -> None:
        if self._exit_stack:
            try:
                asyncio.run_coroutine_threadsafe(
                    self._exit_stack.aclose(), self._loop
                ).result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)


# ---------------------------------------------------------------------------
# GmailService
# ---------------------------------------------------------------------------


class GmailService:
    """Adapter for Gmail email operations.

    Supports three modes:
    - **MCP mode**: connects to ``mcp-server-google-workspace`` via the
      Model Context Protocol when ``.gauth.json`` and ``.accounts.json``
      are present in the configured directory.
    - **Demo mode**: returns sample Airbnb emails (default fallback).
    - **Disabled**: ``enabled=False`` → all methods return empty results.
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        gauth_path: str | None = None,
        accounts_path: str | None = None,
        credentials_dir: str | None = None,
        airbnb_sender_domains: list[str] | None = None,
    ) -> None:
        self._enabled = enabled
        self._airbnb_domains = airbnb_sender_domains or list(AIRBNB_SENDER_DOMAINS)
        self._mcp_client: _McpGmailClient | None = None
        self._demo_mode = True

        if not enabled:
            return

        gauth = gauth_path or ".gauth.json"
        accounts = accounts_path or ".accounts.json"
        creds_dir = credentials_dir or str(Path(gauth).parent)

        if Path(gauth).exists() and Path(accounts).exists():
            try:
                self._mcp_client = _McpGmailClient(
                    gauth_path=gauth,
                    accounts_path=accounts,
                    credentials_dir=creds_dir,
                )
                if self._mcp_client.is_connected:
                    self._demo_mode = False
                    logger.info("GmailService running in MCP mode")
                else:
                    logger.warning(
                        "Gmail MCP connection failed (%s), falling back to demo mode",
                        self._mcp_client.init_error,
                    )
                    self._mcp_client = None
            except Exception as exc:
                logger.warning("Gmail MCP init error, falling back to demo mode: %s", exc)
                self._mcp_client = None
        else:
            missing = []
            if not Path(gauth).exists():
                missing.append(gauth)
            if not Path(accounts).exists():
                missing.append(accounts)
            logger.info(
                "Gmail MCP config files not found (%s), running in demo mode",
                ", ".join(missing),
            )

    @property
    def is_available(self) -> bool:
        return self._enabled

    @property
    def is_demo_mode(self) -> bool:
        return self._demo_mode

    def list_unread_messages(self, max_results: int = 20) -> list[EmailMessage]:
        """Fetch unread messages from inbox."""
        if not self._enabled:
            return []
        if self._demo_mode:
            return self._demo_list_unread(max_results)
        return self._mcp_list_unread(max_results)

    def get_message(self, message_id: str) -> EmailMessage | None:
        """Fetch full message by ID."""
        if not self._enabled:
            return None
        if self._demo_mode:
            return self._demo_get_message(message_id)
        return self._mcp_get_message(message_id)

    def create_draft(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
    ) -> DraftResult:
        """Create a draft email (never auto-sends by default)."""
        if self._demo_mode:
            draft_id = f"draft-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"
            return DraftResult(
                draft_id=draft_id,
                thread_id=thread_id,
                subject=subject,
                body=body,
                status="created",
            )
        return self._mcp_create_draft(to=to, subject=subject, body=body, thread_id=thread_id)

    def is_airbnb_sender(self, sender: str) -> bool:
        """Check if sender email domain belongs to Airbnb."""
        email_match = re.search(r"@([\w.-]+)", sender)
        if not email_match:
            return False
        domain = email_match.group(1).lower()
        return any(domain.endswith(d) for d in self._airbnb_domains)

    def close(self) -> None:
        """Shut down the MCP client subprocess and background thread."""
        if self._mcp_client:
            self._mcp_client.close()
            self._mcp_client = None

    # -- MCP mode implementations ---------------------------------------------------

    def _mcp_list_unread(self, max_results: int) -> list[EmailMessage]:
        assert self._mcp_client is not None
        try:
            result = self._mcp_client.call_tool(
                "gmail_query_emails",
                {"query": "is:unread", "max_results": max_results},
            )
            return self._parse_mcp_email_list(result)
        except Exception as exc:
            logger.error("MCP gmail_query_emails failed: %s", exc)
            return self._demo_list_unread(max_results)

    def _mcp_get_message(self, message_id: str) -> EmailMessage | None:
        assert self._mcp_client is not None
        try:
            result = self._mcp_client.call_tool(
                "gmail_get_email",
                {"email_id": message_id},
            )
            return self._parse_mcp_single_email(result)
        except Exception as exc:
            logger.error("MCP gmail_get_email failed: %s", exc)
            return None

    def _mcp_create_draft(
        self, *, to: str, subject: str, body: str, thread_id: str | None
    ) -> DraftResult:
        assert self._mcp_client is not None
        try:
            args: dict[str, Any] = {"to": to, "subject": subject, "body": body}
            if thread_id:
                args["thread_id"] = thread_id
            result = self._mcp_client.call_tool("gmail_create_draft", args)
            content = self._extract_mcp_text(result)
            return DraftResult(
                draft_id=content.get("id", f"draft-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}"),
                thread_id=thread_id,
                subject=subject,
                body=body,
                status="created",
            )
        except Exception as exc:
            logger.error("MCP gmail_create_draft failed: %s", exc)
            return DraftResult(
                draft_id="error",
                thread_id=thread_id,
                subject=subject,
                body=body,
                status=f"error: {exc}",
            )

    # -- MCP response parsers -------------------------------------------------------

    @staticmethod
    def _extract_mcp_text(result: Any) -> dict[str, Any]:
        """Extract text content from an MCP tool result and parse as JSON if possible."""
        if hasattr(result, "content"):
            for block in result.content:
                text = getattr(block, "text", None)
                if text:
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        return {"raw": text}
        return {}

    def _parse_mcp_email_list(self, result: Any) -> list[EmailMessage]:
        """Parse the MCP tool result from gmail_query_emails into EmailMessage list."""
        data = self._extract_mcp_text(result)

        if isinstance(data, dict) and "emails" in data:
            emails_raw = data["emails"]
        elif isinstance(data, list):
            emails_raw = data
        elif isinstance(data, dict) and "raw" in data:
            try:
                emails_raw = json.loads(data["raw"])
                if isinstance(emails_raw, dict):
                    emails_raw = emails_raw.get("emails", [])
            except (json.JSONDecodeError, TypeError):
                emails_raw = []
        else:
            emails_raw = []

        messages: list[EmailMessage] = []
        for raw in emails_raw:
            if not isinstance(raw, dict):
                continue
            messages.append(EmailMessage(
                id=str(raw.get("id", "")),
                thread_id=str(raw.get("threadId", raw.get("thread_id", ""))),
                sender=str(raw.get("from", raw.get("sender", ""))),
                recipient=str(raw.get("to", raw.get("recipient", ""))),
                subject=str(raw.get("subject", "")),
                snippet=str(raw.get("snippet", ""))[:200],
                body=str(raw.get("body", raw.get("text", raw.get("snippet", "")))),
                date=str(raw.get("date", raw.get("internalDate", ""))),
                labels=raw.get("labels", raw.get("labelIds", [])),
            ))
        return messages

    def _parse_mcp_single_email(self, result: Any) -> EmailMessage | None:
        """Parse the MCP tool result from gmail_get_email into a single EmailMessage."""
        data = self._extract_mcp_text(result)
        if not data or "raw" in data and not data["raw"]:
            return None

        raw = data if "id" in data else data.get("email", data)
        if not isinstance(raw, dict) or "id" not in raw:
            return None

        return EmailMessage(
            id=str(raw.get("id", "")),
            thread_id=str(raw.get("threadId", raw.get("thread_id", ""))),
            sender=str(raw.get("from", raw.get("sender", ""))),
            recipient=str(raw.get("to", raw.get("recipient", ""))),
            subject=str(raw.get("subject", "")),
            snippet=str(raw.get("snippet", ""))[:200],
            body=str(raw.get("body", raw.get("text", raw.get("snippet", "")))),
            date=str(raw.get("date", raw.get("internalDate", ""))),
            labels=raw.get("labels", raw.get("labelIds", [])),
        )

    # -- Demo mode implementations --------------------------------------------------

    def _demo_list_unread(self, max_results: int) -> list[EmailMessage]:
        return [self._raw_to_message(raw) for raw in _DEMO_EMAILS[:max_results]]

    def _demo_get_message(self, message_id: str) -> EmailMessage | None:
        for raw in _DEMO_EMAILS:
            if raw["id"] == message_id:
                return self._raw_to_message(raw)
        return None

    @staticmethod
    def _raw_to_message(raw: dict[str, Any]) -> EmailMessage:
        return EmailMessage(
            id=raw["id"],
            thread_id=raw["thread_id"],
            sender=raw["from"],
            recipient=raw["to"],
            subject=raw["subject"],
            snippet=raw["snippet"],
            body=raw["body"],
            date=raw["date"],
            labels=raw.get("labels", []),
        )
