"""Supabase-backed notification store with SSE fan-out for the mail agent UI."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class NotificationStore:
    """Persists mail notifications in Postgres/Supabase and broadcasts to SSE clients."""

    def __init__(self, database_url: str | None) -> None:
        self._db_url = database_url
        self._clients: list[asyncio.Queue[dict[str, Any]]] = []
        self._table_ready = False

    def _ensure_table(self) -> None:
        if self._table_ready or not self._db_url:
            return
        try:
            import psycopg
        except ImportError:
            logger.warning("psycopg not available; notification store disabled")
            return
        try:
            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS mail_notifications (
                            id TEXT PRIMARY KEY,
                            email_id TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                            category TEXT,
                            subject TEXT,
                            sender TEXT,
                            guest_name TEXT,
                            rating INTEGER,
                            snippet TEXT,
                            action_data JSONB,
                            status TEXT NOT NULL DEFAULT 'pending'
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_mail_notif_status
                        ON mail_notifications (status)
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_mail_notif_email_id
                        ON mail_notifications (email_id)
                        """
                    )
                    conn.commit()
            self._table_ready = True
        except Exception as e:
            logger.warning("Failed to create mail_notifications table: %s", e)

    def has_notification_for_email(self, email_id: str) -> bool:
        """Check whether a pending notification already exists for this email."""
        if not self._db_url:
            return False
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT 1 FROM mail_notifications WHERE email_id = %s AND status = 'pending' LIMIT 1",
                        (email_id,),
                    )
                    return cur.fetchone() is not None
        except Exception as e:
            logger.warning("has_notification_for_email failed: %s", e)
            return False

    def add_notification(self, action: dict[str, Any]) -> dict[str, Any] | None:
        """Insert a notification from a mail pipeline action dict. Returns the notification or None."""
        if not self._db_url:
            return None
        self._ensure_table()

        email_id = action.get("email_id", "")
        if not email_id:
            return None

        if self.has_notification_for_email(email_id):
            return None

        notif_id = f"notif_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        notif = {
            "id": notif_id,
            "email_id": email_id,
            "created_at": now.isoformat(),
            "category": action.get("category", ""),
            "subject": action.get("subject", ""),
            "sender": action.get("sender", ""),
            "guest_name": action.get("guest_name"),
            "rating": action.get("rating"),
            "snippet": action.get("snippet", ""),
            "action_data": action,
            "status": "pending",
        }

        try:
            import psycopg
            from psycopg.types.json import Jsonb

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO mail_notifications
                            (id, email_id, created_at, category, subject, sender,
                             guest_name, rating, snippet, action_data, status)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (
                            notif_id,
                            email_id,
                            now,
                            notif["category"],
                            notif["subject"],
                            notif["sender"],
                            notif["guest_name"],
                            notif["rating"],
                            notif["snippet"],
                            Jsonb(action),
                            "pending",
                        ),
                    )
                    conn.commit()
        except Exception as e:
            logger.warning("add_notification failed: %s", e)
            return None

        self._broadcast({"type": "new", "notification": notif})
        return notif

    def get_pending(self) -> list[dict[str, Any]]:
        """Return all pending notifications, newest first."""
        if not self._db_url:
            return []
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT id, email_id, created_at, category, subject, sender,
                               guest_name, rating, snippet, action_data, status
                        FROM mail_notifications
                        WHERE status = 'pending'
                        ORDER BY created_at DESC
                        """
                    )
                    rows = cur.fetchall()
                    return [
                        {
                            "id": r[0],
                            "email_id": r[1],
                            "created_at": r[2].isoformat() if r[2] else None,
                            "category": r[3],
                            "subject": r[4],
                            "sender": r[5],
                            "guest_name": r[6],
                            "rating": r[7],
                            "snippet": r[8],
                            "action_data": r[9] if isinstance(r[9], dict) else {},
                            "status": r[10],
                        }
                        for r in rows
                    ]
        except Exception as e:
            logger.warning("get_pending failed: %s", e)
            return []

    def mark_handled(self, notification_id: str) -> bool:
        """Set a notification status to 'handled'. Returns True on success."""
        return self._update_status(notification_id, "handled")

    def mark_handled_by_email(self, email_id: str) -> bool:
        """Mark all pending notifications for an email_id as handled."""
        if not self._db_url:
            return False
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE mail_notifications
                        SET status = 'handled'
                        WHERE email_id = %s AND status = 'pending'
                        RETURNING id
                        """,
                        (email_id,),
                    )
                    ids = [row[0] for row in cur.fetchall()]
                    conn.commit()
            for nid in ids:
                self._broadcast({"type": "handled", "notification_id": nid})
            return bool(ids)
        except Exception as e:
            logger.warning("mark_handled_by_email failed: %s", e)
            return False

    def dismiss(self, notification_id: str) -> bool:
        """Set a notification status to 'dismissed'. Returns True on success."""
        return self._update_status(notification_id, "dismissed")

    def _update_status(self, notification_id: str, new_status: str) -> bool:
        if not self._db_url:
            return False
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE mail_notifications SET status = %s WHERE id = %s AND status = 'pending'",
                        (new_status, notification_id),
                    )
                    updated = cur.rowcount > 0
                    conn.commit()
            if updated:
                self._broadcast({"type": new_status, "notification_id": notification_id})
            return updated
        except Exception as e:
            logger.warning("_update_status(%s) failed: %s", new_status, e)
            return False

    def dismiss_all(self) -> list[str]:
        """Dismiss all pending notifications. Returns list of dismissed IDs."""
        if not self._db_url:
            return []
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE mail_notifications
                        SET status = 'dismissed'
                        WHERE status = 'pending'
                        RETURNING id
                        """
                    )
                    ids = [row[0] for row in cur.fetchall()]
                    conn.commit()
            for nid in ids:
                self._broadcast({"type": "dismissed", "notification_id": nid})
            return ids
        except Exception as e:
            logger.warning("dismiss_all failed: %s", e)
            return []

    def count_pending(self) -> int:
        """Return the number of pending notifications."""
        if not self._db_url:
            return 0
        self._ensure_table()
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT COUNT(*) FROM mail_notifications WHERE status = 'pending'"
                    )
                    row = cur.fetchone()
                    return row[0] if row else 0
        except Exception as e:
            logger.warning("count_pending failed: %s", e)
            return 0

    # ------------------------------------------------------------------
    # SSE fan-out
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        """Register a new SSE client. Returns the queue to read events from."""
        q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._clients.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[dict[str, Any]]) -> None:
        """Remove an SSE client queue."""
        try:
            self._clients.remove(q)
        except ValueError:
            pass

    def _broadcast(self, event: dict[str, Any]) -> None:
        """Push an event to all connected SSE clients."""
        for q in list(self._clients):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
