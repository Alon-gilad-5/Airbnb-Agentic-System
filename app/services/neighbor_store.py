"""Persistence layer for property-neighbor mappings (Postgres)."""

from __future__ import annotations


class PostgresNeighborStore:
    """Postgres-backed store for property → neighbor ID mappings."""

    def __init__(self, *, database_url: str) -> None:
        self.database_url = database_url
        self._ensure_schema()

    def _connect(self):  # type: ignore[no-untyped-def]
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError(
                "Postgres backend requires psycopg. Add `psycopg[binary]` dependency."
            ) from exc
        return psycopg.connect(self.database_url, prepare_threshold=None, connect_timeout=5)

    def _ensure_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS property_neighbors (
            property_id TEXT PRIMARY KEY,
            neighbor_ids TEXT[] NOT NULL
        )
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl)
            conn.commit()

    def bulk_load(self, rows: list[tuple[str, list[str]]]) -> int:
        """Insert or update neighbor mappings. Returns number of rows upserted."""
        if not rows:
            return 0
        sql = """
        INSERT INTO property_neighbors (property_id, neighbor_ids)
        VALUES (%s, %s)
        ON CONFLICT (property_id) DO UPDATE SET neighbor_ids = EXCLUDED.neighbor_ids
        """
        params = [(pid, nids) for pid, nids in rows]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.executemany(sql, params)
            conn.commit()
        return len(params)

    def get_neighbors(self, property_id: str) -> list[str] | None:
        """Return neighbor IDs for a property, or None if not found."""
        sql = "SELECT neighbor_ids FROM property_neighbors WHERE property_id = %(pid)s"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, {"pid": property_id})
                row = cur.fetchone()
        if row is None:
            return None
        return list(row[0])


def create_neighbor_store(database_url: str | None) -> PostgresNeighborStore | None:
    """Factory: returns None when DATABASE_URL is not configured."""
    if not database_url:
        return None
    try:
        return PostgresNeighborStore(database_url=database_url)
    except Exception:
        return None
