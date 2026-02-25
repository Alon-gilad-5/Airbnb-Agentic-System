"""Persistence layer for market-watch alerts (SQLite locally, Postgres on Vercel)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sqlite3
import uuid
from typing import Any, Protocol


@dataclass
class MarketAlertRecord:
    """Canonical alert shape persisted to inbox storage."""

    id: str
    created_at_utc: str
    owner_id: str | None
    property_id: str | None
    property_name: str | None
    city: str | None
    region: str | None
    alert_type: str
    severity: str
    title: str
    summary: str
    start_at_utc: str | None
    end_at_utc: str | None
    source_name: str
    source_url: str | None
    evidence: dict[str, Any]


class MarketAlertStore(Protocol):
    """Store contract used by market-watch flows and APIs."""

    def insert_alerts(self, alerts: list[MarketAlertRecord]) -> int:
        """Insert alert list and return inserted row count."""

    def list_latest_alerts(
        self,
        *,
        owner_id: str | None,
        property_id: str | None,
        limit: int,
    ) -> list[MarketAlertRecord]:
        """Return latest alerts for owner/property scope."""


class SqliteMarketAlertStore:
    """SQLite-backed implementation for local development and testing."""

    def __init__(self, *, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection with Row access for named column parsing."""

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """Create table/index when missing."""

        ddl = """
        CREATE TABLE IF NOT EXISTS market_watch_alerts (
            id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            owner_id TEXT,
            property_id TEXT,
            property_name TEXT,
            city TEXT,
            region TEXT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            start_at_utc TEXT,
            end_at_utc TEXT,
            source_name TEXT NOT NULL,
            source_url TEXT,
            evidence_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_market_watch_scope_time
        ON market_watch_alerts(owner_id, property_id, created_at_utc DESC);
        """
        with self._connect() as conn:
            conn.executescript(ddl)
            conn.commit()

    def insert_alerts(self, alerts: list[MarketAlertRecord]) -> int:
        """Insert alerts with `INSERT OR REPLACE` to keep deterministic IDs idempotent."""

        if not alerts:
            return 0
        sql = """
        INSERT OR REPLACE INTO market_watch_alerts (
            id, created_at_utc, owner_id, property_id, property_name, city, region,
            alert_type, severity, title, summary, start_at_utc, end_at_utc,
            source_name, source_url, evidence_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        rows = [
            (
                a.id,
                a.created_at_utc,
                a.owner_id,
                a.property_id,
                a.property_name,
                a.city,
                a.region,
                a.alert_type,
                a.severity,
                a.title,
                a.summary,
                a.start_at_utc,
                a.end_at_utc,
                a.source_name,
                a.source_url,
                json.dumps(a.evidence, ensure_ascii=True),
            )
            for a in alerts
        ]
        with self._connect() as conn:
            conn.executemany(sql, rows)
            conn.commit()
        return len(rows)

    def list_latest_alerts(
        self,
        *,
        owner_id: str | None,
        property_id: str | None,
        limit: int,
    ) -> list[MarketAlertRecord]:
        """Fetch latest alerts constrained by active owner/property when provided."""

        limit = max(1, int(limit))
        where: list[str] = []
        params: list[Any] = []
        if owner_id:
            where.append("owner_id = ?")
            params.append(owner_id)
        if property_id:
            where.append("property_id = ?")
            params.append(property_id)

        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
        SELECT id, created_at_utc, owner_id, property_id, property_name, city, region,
               alert_type, severity, title, summary, start_at_utc, end_at_utc,
               source_name, source_url, evidence_json
        FROM market_watch_alerts
        {where_sql}
        ORDER BY created_at_utc DESC
        LIMIT ?
        """
        params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._from_row(row) for row in rows]

    def _from_row(self, row: sqlite3.Row) -> MarketAlertRecord:
        """Convert sqlite row to strongly-typed record."""

        evidence_raw = row["evidence_json"] or "{}"
        return MarketAlertRecord(
            id=row["id"],
            created_at_utc=row["created_at_utc"],
            owner_id=row["owner_id"],
            property_id=row["property_id"],
            property_name=row["property_name"],
            city=row["city"],
            region=row["region"],
            alert_type=row["alert_type"],
            severity=row["severity"],
            title=row["title"],
            summary=row["summary"],
            start_at_utc=row["start_at_utc"],
            end_at_utc=row["end_at_utc"],
            source_name=row["source_name"],
            source_url=row["source_url"],
            evidence=json.loads(evidence_raw),
        )


class PostgresMarketAlertStore:
    """Postgres-backed implementation intended for Vercel/serverless deployment."""

    def __init__(self, *, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_schema()

    def _connect(self):  # type: ignore[no-untyped-def]
        """Connect with psycopg3 only when this backend is used."""

        try:
            import psycopg
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Postgres backend requires psycopg. Add `psycopg[binary]` dependency."
            ) from exc
        return psycopg.connect(self.database_url, prepare_threshold=0)

    def _ensure_schema(self) -> None:
        """Create table/index if missing."""

        ddl = """
        CREATE TABLE IF NOT EXISTS market_watch_alerts (
            id TEXT PRIMARY KEY,
            created_at_utc TIMESTAMPTZ NOT NULL,
            owner_id TEXT,
            property_id TEXT,
            property_name TEXT,
            city TEXT,
            region TEXT,
            alert_type TEXT NOT NULL,
            severity TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            start_at_utc TIMESTAMPTZ,
            end_at_utc TIMESTAMPTZ,
            source_name TEXT NOT NULL,
            source_url TEXT,
            evidence_json JSONB NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_market_watch_scope_time
        ON market_watch_alerts(owner_id, property_id, created_at_utc DESC);
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def insert_alerts(self, alerts: list[MarketAlertRecord]) -> int:
        """Insert alerts via UPSERT for idempotent run retries."""

        if not alerts:
            return 0

        sql = """
        INSERT INTO market_watch_alerts (
            id, created_at_utc, owner_id, property_id, property_name, city, region,
            alert_type, severity, title, summary, start_at_utc, end_at_utc,
            source_name, source_url, evidence_json
        ) VALUES (
            %(id)s, %(created_at_utc)s, %(owner_id)s, %(property_id)s, %(property_name)s, %(city)s, %(region)s,
            %(alert_type)s, %(severity)s, %(title)s, %(summary)s, %(start_at_utc)s, %(end_at_utc)s,
            %(source_name)s, %(source_url)s, %(evidence_json)s
        )
        ON CONFLICT (id) DO UPDATE SET
            created_at_utc = EXCLUDED.created_at_utc,
            owner_id = EXCLUDED.owner_id,
            property_id = EXCLUDED.property_id,
            property_name = EXCLUDED.property_name,
            city = EXCLUDED.city,
            region = EXCLUDED.region,
            alert_type = EXCLUDED.alert_type,
            severity = EXCLUDED.severity,
            title = EXCLUDED.title,
            summary = EXCLUDED.summary,
            start_at_utc = EXCLUDED.start_at_utc,
            end_at_utc = EXCLUDED.end_at_utc,
            source_name = EXCLUDED.source_name,
            source_url = EXCLUDED.source_url,
            evidence_json = EXCLUDED.evidence_json
        """
        rows = [
            {
                "id": a.id,
                "created_at_utc": a.created_at_utc,
                "owner_id": a.owner_id,
                "property_id": a.property_id,
                "property_name": a.property_name,
                "city": a.city,
                "region": a.region,
                "alert_type": a.alert_type,
                "severity": a.severity,
                "title": a.title,
                "summary": a.summary,
                "start_at_utc": a.start_at_utc,
                "end_at_utc": a.end_at_utc,
                "source_name": a.source_name,
                "source_url": a.source_url,
                "evidence_json": json.dumps(a.evidence, ensure_ascii=True),
            }
            for a in alerts
        ]
        with self._connect() as conn:
            with conn.cursor() as cur:
                for row in rows:
                    cur.execute(sql, row)
            conn.commit()
        return len(rows)

    def list_latest_alerts(
        self,
        *,
        owner_id: str | None,
        property_id: str | None,
        limit: int,
    ) -> list[MarketAlertRecord]:
        """Fetch latest alerts from Postgres with optional scope filtering."""

        limit = max(1, int(limit))
        where: list[str] = []
        params: list[Any] = []
        if owner_id:
            where.append("owner_id = %s")
            params.append(owner_id)
        if property_id:
            where.append("property_id = %s")
            params.append(property_id)
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        sql = f"""
        SELECT id, created_at_utc, owner_id, property_id, property_name, city, region,
               alert_type, severity, title, summary, start_at_utc, end_at_utc,
               source_name, source_url, evidence_json
        FROM market_watch_alerts
        {where_sql}
        ORDER BY created_at_utc DESC
        LIMIT %s
        """
        params.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

        out: list[MarketAlertRecord] = []
        for row in rows:
            evidence_obj = row[15] if isinstance(row[15], dict) else json.loads(row[15] or "{}")
            out.append(
                MarketAlertRecord(
                    id=row[0],
                    created_at_utc=self._to_utc_text(row[1]),
                    owner_id=row[2],
                    property_id=row[3],
                    property_name=row[4],
                    city=row[5],
                    region=row[6],
                    alert_type=row[7],
                    severity=row[8],
                    title=row[9],
                    summary=row[10],
                    start_at_utc=self._to_utc_text(row[11]),
                    end_at_utc=self._to_utc_text(row[12]),
                    source_name=row[13],
                    source_url=row[14],
                    evidence=evidence_obj,
                )
            )
        return out

    def _to_utc_text(self, value: Any) -> str | None:
        """Normalize date/datetime values into ISO UTC strings."""

        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=UTC)
            return value.astimezone(UTC).isoformat()
        return str(value)


def create_market_alert_store(
    *,
    database_url: str | None,
    sqlite_path: str,
) -> MarketAlertStore:
    """Factory selecting Postgres when DATABASE_URL is configured, otherwise SQLite."""

    if database_url:
        return PostgresMarketAlertStore(database_url=database_url)
    return SqliteMarketAlertStore(db_path=sqlite_path)


def build_alert_id(
    *,
    owner_id: str | None,
    property_id: str | None,
    alert_type: str,
    title: str,
    start_at_utc: str | None,
) -> str:
    """Build deterministic-ish IDs to reduce duplicates across repeated autonomous runs."""

    seed = "|".join(
        [
            owner_id or "unknown-owner",
            property_id or "unknown-property",
            alert_type.strip().lower(),
            title.strip().lower(),
            start_at_utc or "no-start-time",
        ]
    )
    # UUID5 keeps IDs stable for identical alert content.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(tz=UTC).isoformat()
