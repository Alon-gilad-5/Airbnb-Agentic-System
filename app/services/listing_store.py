"""Read-only access layer for listing-level structured data in Supabase."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

REVIEW_SCORE_COLUMNS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

PROPERTY_SPEC_NUMERIC_COLUMNS = [
    "accommodates",
    "bathrooms",
    "bedrooms",
    "beds",
    "price",
]

PROPERTY_SPEC_CATEGORICAL_COLUMNS = [
    "property_type",
    "room_type",
    "host_is_superhost",
]

PROPERTY_SPEC_COLUMNS = PROPERTY_SPEC_NUMERIC_COLUMNS + PROPERTY_SPEC_CATEGORICAL_COLUMNS

BASE_COLUMNS = ["id", "name"]

ALLOWED_COLUMNS = set(BASE_COLUMNS + REVIEW_SCORE_COLUMNS + PROPERTY_SPEC_COLUMNS)


class ListingStore:
    """Read-only store over the Supabase listing table."""

    table_name = "large_dataset_table"

    def __init__(self, *, database_url: str) -> None:
        self.database_url = database_url

    def _connect(self):  # type: ignore[no-untyped-def]
        try:
            import psycopg
        except Exception as exc:
            raise RuntimeError(
                "Postgres backend requires psycopg. Add `psycopg[binary]` dependency."
            ) from exc
        return psycopg.connect(self.database_url, prepare_threshold=None, connect_timeout=5)

    def get_listings_by_ids(
        self,
        listing_ids: Sequence[str],
        columns: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Fetch a whitelisted column subset for the requested listing IDs."""

        ids = [str(listing_id).strip() for listing_id in listing_ids if str(listing_id).strip()]
        if not ids:
            return []

        requested_columns = ["id"]
        for column in columns:
            clean_column = str(column).strip()
            if not clean_column or clean_column == "id":
                continue
            if clean_column not in ALLOWED_COLUMNS:
                raise ValueError(f"Unsupported listing column requested: {clean_column}")
            requested_columns.append(clean_column)

        # Preserve order while deduplicating.
        seen: set[str] = set()
        select_columns: list[str] = []
        for column in requested_columns:
            if column in seen:
                continue
            seen.add(column)
            select_columns.append(column)

        try:
            from psycopg import sql
            from psycopg.rows import dict_row
        except Exception as exc:
            raise RuntimeError(
                "Postgres backend requires psycopg. Add `psycopg[binary]` dependency."
            ) from exc

        query = sql.SQL(
            "SELECT {fields} FROM {table} WHERE id = ANY(%s)"
        ).format(
            fields=sql.SQL(", ").join(sql.Identifier(column) for column in select_columns),
            table=sql.Identifier(self.table_name),
        )

        with self._connect() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(query, (ids,))
                rows = cur.fetchall()
        return [dict(row) for row in rows]


def create_listing_store(database_url: str | None) -> ListingStore | None:
    """Return a listing store when database access is configured and reachable."""

    if not database_url:
        return None
    try:
        return ListingStore(database_url=database_url)
    except Exception:
        return None
