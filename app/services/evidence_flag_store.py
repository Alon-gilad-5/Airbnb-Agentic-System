"""Postgres-backed store for evidence relevance flags (future retrieval tuning)."""

from __future__ import annotations

import logging
import uuid

logger = logging.getLogger(__name__)


class EvidenceFlagStore:
    """Persists user flags on review-evidence items for future retrieval tuning."""

    def __init__(self, database_url: str | None) -> None:
        self._db_url = database_url
        self._table_ready = False

    def _ensure_table(self) -> None:
        if self._table_ready or not self._db_url:
            return
        try:
            import psycopg
        except ImportError:
            logger.warning("psycopg not available; evidence flag store disabled")
            return
        try:
            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS evidence_flags (
                            id TEXT PRIMARY KEY,
                            vector_id TEXT NOT NULL,
                            query_text TEXT NOT NULL,
                            flag TEXT NOT NULL DEFAULT 'irrelevant',
                            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    )
                    cur.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_evidence_flags_vid
                        ON evidence_flags (vector_id)
                        """
                    )
                    conn.commit()
            self._table_ready = True
        except Exception as e:
            logger.warning("Failed to create evidence_flags table: %s", e)

    def add_flag(self, vector_id: str, query_text: str, flag: str = "irrelevant") -> bool:
        """Insert a flag record. Returns True on success."""
        if not self._db_url:
            return False
        self._ensure_table()
        flag_id = f"eflag_{uuid.uuid4().hex[:12]}"
        try:
            import psycopg

            with psycopg.connect(self._db_url, prepare_threshold=None, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO evidence_flags (id, vector_id, query_text, flag)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                        """,
                        (flag_id, vector_id, query_text, flag),
                    )
                    conn.commit()
            return True
        except Exception as e:
            logger.warning("add_flag failed: %s", e)
            return False
