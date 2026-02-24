"""Shared helpers for region normalization across ingest and retrieval."""

from __future__ import annotations


def canonicalize_region(raw: str | None) -> str | None:
    """Return normalized region string or None when missing/blank."""

    if raw is None:
        return None
    stripped = raw.strip()
    if not stripped:
        return None
    return stripped.lower()

