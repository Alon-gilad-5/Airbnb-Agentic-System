"""Gmail adapter service for email ingestion and draft management.

Supports four modes:
- **MCP**: connects to ``mcp-server-google-workspace`` via stdio and
  calls Gmail tools through the Model Context Protocol (local dev).
- **Gmail API**: direct REST API calls with OAuth2 refresh token (Render/production).
- **Demo**: returns sample Airbnb emails for development/testing.
- **Disabled**: returns empty results when mail feature is off.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import shutil
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.mime.text import MIMEText
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
    message_id_header: str | None = None  # RFC Message-ID for reply threading
    references: str | None = None  # RFC References for reply threading


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
# Direct Gmail API client (OAuth2 refresh token, no MCP)
# ---------------------------------------------------------------------------


def _decode_gmail_body(payload: dict[str, Any]) -> str:
    """Extract and decode body text from Gmail API message payload."""
    if not payload:
        return ""
    body = payload.get("body", {})
    data = body.get("data")
    if data:
        try:
            return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
        except Exception:
            return ""
    parts = payload.get("parts", [])
    for part in parts:
        if part.get("mimeType") == "text/plain":
            part_data = part.get("body", {}).get("data")
            if part_data:
                try:
                    return base64.urlsafe_b64decode(part_data).decode("utf-8", errors="replace")
                except Exception:
                    pass
    for part in parts:
        if part.get("mimeType") == "text/html":
            part_data = part.get("body", {}).get("data")
            if part_data:
                try:
                    return base64.urlsafe_b64decode(part_data).decode("utf-8", errors="replace")
                except Exception:
                    pass
    return ""


def _gmail_header(payload: dict[str, Any], name: str) -> str:
    """Get a header value from Gmail API message payload."""
    for h in payload.get("headers", []):
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


class _DirectGmailClient:
    """Gmail v1 API client using OAuth2 refresh token (headless, no browser).

    Used as fallback when MCP is not available (e.g. on Render).
    """

    def __init__(self, *, client_id: str, client_secret: str, refresh_token: str) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._refresh_token = refresh_token
        self._service: Any = None

    def _get_service(self):  # noqa: ANN201
        if self._service is not None:
            return self._service
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        creds = Credentials(
            token=None,
            refresh_token=self._refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=self._client_id,
            client_secret=self._client_secret,
            scopes=["https://www.googleapis.com/auth/gmail.readonly", "https://www.googleapis.com/auth/gmail.compose", "https://www.googleapis.com/auth/gmail.modify"],
        )
        self._service = build("gmail", "v1", credentials=creds, cache_discovery=False)
        return self._service

    def _message_to_email(self, msg: dict[str, Any]) -> EmailMessage:
        payload = msg.get("payload", {})
        return EmailMessage(
            id=msg.get("id", ""),
            thread_id=msg.get("threadId", ""),
            sender=_gmail_header(payload, "From"),
            recipient=_gmail_header(payload, "To"),
            subject=_gmail_header(payload, "Subject"),
            snippet=(msg.get("snippet") or "")[:200],
            body=_decode_gmail_body(payload) or (msg.get("snippet") or ""),
            date=msg.get("internalDate", ""),
            labels=msg.get("labelIds", []),
            message_id_header=_gmail_header(payload, "Message-ID") or None,
            references=_gmail_header(payload, "References") or None,
        )

    def list_unread(self, max_results: int) -> list[EmailMessage]:
        from googleapiclient.errors import HttpError

        service = self._get_service()
        try:
            list_result = (
                service.users()
                .messages()
                .list(userId="me", q="is:unread", maxResults=max_results)
                .execute()
            )
        except HttpError as exc:
            logger.error("Gmail API list messages failed: %s", exc)
            return []
        message_list = list_result.get("messages", [])
        result: list[EmailMessage] = []
        for m in message_list:
            mid = m.get("id")
            if not mid:
                continue
            try:
                full = (
                    service.users()
                    .messages()
                    .get(userId="me", id=mid, format="full")
                    .execute()
                )
                result.append(self._message_to_email(full))
            except HttpError as exc:
                logger.warning("Gmail API get message %s failed: %s", mid, exc)
        return result

    def get_message(self, message_id: str) -> EmailMessage | None:
        from googleapiclient.errors import HttpError

        service = self._get_service()
        try:
            msg = (
                service.users()
                .messages()
                .get(userId="me", id=message_id, format="full")
                .execute()
            )
            return self._message_to_email(msg)
        except HttpError as exc:
            logger.error("Gmail API get message %s failed: %s", message_id, exc)
            return None

    def create_draft(
        self,
        *,
        to: str,
        subject: str,
        body: str,
        thread_id: str | None = None,
    ) -> DraftResult:
        from googleapiclient.errors import HttpError

        mime = MIMEText(body, "plain", "utf-8")
        mime["To"] = to
        mime["Subject"] = subject
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii").rstrip("=")
        draft_body: dict[str, Any] = {"message": {"raw": raw}}
        if thread_id:
            draft_body["message"]["threadId"] = thread_id
        try:
            service = self._get_service()
            draft = (
                service.users()
                .drafts()
                .create(userId="me", body=draft_body)
                .execute()
            )
            draft_id = draft.get("id", "")
            return DraftResult(
                draft_id=str(draft_id),
                thread_id=thread_id,
                subject=subject,
                body=body,
                status="created",
            )
        except HttpError as exc:
            logger.error("Gmail API create draft failed: %s", exc)
            return DraftResult(
                draft_id="error",
                thread_id=thread_id,
                subject=subject,
                body=body,
                status=f"error: {exc}",
            )

    def list_messages_since(self, start_history_id: str) -> list[EmailMessage]:
        """Return messages added since the given history ID. Returns empty on 404 (history too old)."""
        from googleapiclient.errors import HttpError

        service = self._get_service()
        try:
            result = (
                service.users()
                .history()
                .list(
                    userId="me",
                    startHistoryId=start_history_id,
                    historyTypes="messageAdded",
                    maxResults=100,
                )
                .execute()
            )
        except HttpError as exc:
            if exc.resp and exc.resp.status == 404:
                logger.info("Gmail history.list 404 (history too old), returning empty")
                return []
            logger.error("Gmail API history.list failed: %s", exc)
            return []

        message_ids: list[str] = []
        for rec in result.get("history", []):
            for added in rec.get("messagesAdded", []):
                msg = added.get("message")
                if msg and msg.get("id"):
                    message_ids.append(msg["id"])

        out: list[EmailMessage] = []
        for mid in message_ids:
            try:
                full = (
                    service.users()
                    .messages()
                    .get(userId="me", id=mid, format="full")
                    .execute()
                )
                out.append(self._message_to_email(full))
            except HttpError as exc:
                logger.warning("Gmail API get message %s failed: %s", mid, exc)
        return out

    def watch(self, topic_name: str) -> dict[str, Any]:
        """Register mailbox for push notifications. Returns {historyId, expiration} (expiration in ms)."""
        from googleapiclient.errors import HttpError

        service = self._get_service()
        try:
            body = {"topicName": topic_name}
            resp = (
                service.users()
                .watch(userId="me", body=body)
                .execute()
            )
            return {
                "historyId": str(resp.get("historyId", "")),
                "expiration": int(resp.get("expiration", 0)),
            }
        except HttpError as exc:
            logger.error("Gmail API watch failed: %s", exc)
            raise

    def send_message(self, to: str, subject: str, body: str) -> bool:
        """Send a plain-text email (e.g. notification to owner). Returns True on success."""
        from googleapiclient.errors import HttpError

        mime = MIMEText(body, "plain", "utf-8")
        mime["To"] = to
        mime["Subject"] = subject
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii").rstrip("=")
        try:
            service = self._get_service()
            service.users().messages().send(
                userId="me",
                body={"raw": raw},
            ).execute()
            return True
        except HttpError as exc:
            logger.error("Gmail API send_message failed: %s", exc)
            return False

    def send_reply(
        self,
        *,
        thread_id: str,
        to: str,
        subject: str,
        body: str,
        in_reply_to: str | None = None,
        references: str | None = None,
    ) -> bool:
        """Send a reply in the same thread with In-Reply-To/References for threading."""
        from googleapiclient.errors import HttpError

        mime = MIMEText(body, "plain", "utf-8")
        mime["To"] = to
        mime["Subject"] = subject
        if in_reply_to:
            mime["In-Reply-To"] = in_reply_to
        if references:
            mime["References"] = references
        raw = base64.urlsafe_b64encode(mime.as_bytes()).decode("ascii").rstrip("=")
        try:
            service = self._get_service()
            service.users().messages().send(
                userId="me",
                body={"raw": raw, "threadId": thread_id},
            ).execute()
            return True
        except HttpError as exc:
            logger.error("Gmail API send_reply failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# GmailService
# ---------------------------------------------------------------------------


class GmailService:
    """Adapter for Gmail email operations.

    Supports four modes:
    - **MCP mode**: connects to ``mcp-server-google-workspace`` via the
      Model Context Protocol when ``.gauth.json`` and ``.accounts.json``
      are present in the configured directory (local dev).
    - **Gmail API mode**: direct REST API with OAuth2 refresh token from
      env vars (Render/production).
    - **Demo mode**: returns sample Airbnb emails (tests/fallback).
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
        gmail_client_id: str | None = None,
        gmail_client_secret: str | None = None,
        gmail_refresh_token: str | None = None,
    ) -> None:
        self._enabled = enabled
        self._airbnb_domains = airbnb_sender_domains or list(AIRBNB_SENDER_DOMAINS)
        self._mcp_client: _McpGmailClient | None = None
        self._api_client: _DirectGmailClient | None = None
        self._demo_mode = True

        if not enabled:
            return

        gauth = gauth_path or ".gauth.json"
        accounts = accounts_path or ".accounts.json"
        creds_dir = credentials_dir or str(Path(gauth).parent)

        # 1) Try MCP first (local dev)
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
                        "Gmail MCP connection failed (%s), trying Gmail API / demo",
                        self._mcp_client.init_error,
                    )
                    self._mcp_client = None
            except Exception as exc:
                logger.warning("Gmail MCP init error, trying Gmail API / demo: %s", exc)
                self._mcp_client = None

        # 2) If MCP not active, try Gmail API (Render/production)
        if self._demo_mode and gmail_client_id and gmail_client_secret and gmail_refresh_token:
            try:
                self._api_client = _DirectGmailClient(
                    client_id=gmail_client_id.strip(),
                    client_secret=gmail_client_secret.strip(),
                    refresh_token=gmail_refresh_token.strip(),
                )
                self._api_client._get_service()
                self._demo_mode = False
                logger.info("GmailService running in Gmail API mode")
            except Exception as exc:
                logger.warning("Gmail API init error, falling back to demo mode: %s", exc)
                self._api_client = None

        # 3) Otherwise stay in demo mode
        if self._demo_mode:
            logger.info("GmailService running in demo mode")

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
        if self._api_client:
            return self._api_list_unread(max_results)
        return self._mcp_list_unread(max_results)

    def get_message(self, message_id: str) -> EmailMessage | None:
        """Fetch full message by ID."""
        if not self._enabled:
            return None
        if self._demo_mode:
            return self._demo_get_message(message_id)
        if self._api_client:
            return self._api_get_message(message_id)
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
        if self._api_client:
            return self._api_create_draft(to=to, subject=subject, body=body, thread_id=thread_id)
        return self._mcp_create_draft(to=to, subject=subject, body=body, thread_id=thread_id)

    def list_messages_since_history(self, start_history_id: str) -> list[EmailMessage]:
        """Return messages added since the given history ID (Gmail API mode only)."""
        if not self._enabled:
            return []
        if self._api_client:
            return self._api_client.list_messages_since(start_history_id)
        return []

    def setup_watch(self, topic_name: str) -> dict[str, Any] | None:
        """Register mailbox for push notifications (Gmail API mode only). Returns {historyId, expiration} or None."""
        if not self._enabled or not self._api_client:
            return None
        try:
            return self._api_client.watch(topic_name)
        except Exception as exc:
            logger.warning("Gmail setup_watch failed: %s", exc)
            return None

    def send_message(self, to: str, subject: str, body: str) -> bool:
        """Send a plain-text email (e.g. notification). Demo/MCP: no-op returns True."""
        if not self._enabled:
            return False
        if self._demo_mode:
            logger.debug("Demo mode: would send message to %s", to)
            return True
        if self._api_client:
            return self._api_client.send_message(to, subject, body)
        logger.debug("MCP mode: send_message no-op")
        return True

    def send_reply(
        self,
        *,
        thread_id: str,
        to: str,
        subject: str,
        body: str,
        in_reply_to: str | None = None,
        references: str | None = None,
    ) -> bool:
        """Send a reply in thread. Demo: no-op returns True. MCP: no-op returns False."""
        if not self._enabled:
            return False
        if self._demo_mode:
            logger.debug("Demo mode: would send reply in thread %s", thread_id)
            return True
        if self._api_client:
            return self._api_client.send_reply(
                thread_id=thread_id,
                to=to,
                subject=subject,
                body=body,
                in_reply_to=in_reply_to,
                references=references,
            )
        logger.warning("MCP mode: send_reply not available")
        return False

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

    # -- Gmail API mode implementations --------------------------------------------

    def _api_list_unread(self, max_results: int) -> list[EmailMessage]:
        assert self._api_client is not None
        try:
            return self._api_client.list_unread(max_results)
        except Exception as exc:
            logger.error("Gmail API list_unread failed: %s", exc)
            return self._demo_list_unread(max_results)

    def _api_get_message(self, message_id: str) -> EmailMessage | None:
        assert self._api_client is not None
        return self._api_client.get_message(message_id)

    def _api_create_draft(
        self, *, to: str, subject: str, body: str, thread_id: str | None
    ) -> DraftResult:
        assert self._api_client is not None
        return self._api_client.create_draft(to=to, subject=subject, body=body, thread_id=thread_id)

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
