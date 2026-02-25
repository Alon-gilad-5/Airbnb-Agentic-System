"""Persistence for Gmail push state (historyId, watch expiration) and owner choices (don't reply, etc.)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_PUSH_STATE_FILE = Path("data/mail_push_state.json")
DEFAULT_OWNER_CHOICES_FILE = Path("data/mail_owner_choices.json")


def _ensure_data_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Push state (historyId, expiration_ts)
# ---------------------------------------------------------------------------


def get_push_state(database_url: str | None) -> dict[str, Any] | None:
    """Return stored push state: {history_id, expiration_ts} or None if never set."""
    if database_url and database_url.strip():
        return _get_push_state_postgres(database_url)
    return _get_push_state_file()


def set_push_state(
    database_url: str | None,
    history_id: str,
    expiration_ts: int | None = None,
) -> None:
    """Persist history_id and optional expiration_ts (epoch ms)."""
    if database_url and database_url.strip():
        _set_push_state_postgres(database_url, history_id, expiration_ts)
    else:
        _set_push_state_file(history_id, expiration_ts)


def _get_push_state_file() -> dict[str, Any] | None:
    path = DEFAULT_PUSH_STATE_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict) and data.get("history_id"):
            return data
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read mail push state file %s: %s", path, e)
        return None


def _set_push_state_file(history_id: str, expiration_ts: int | None) -> None:
    path = DEFAULT_PUSH_STATE_FILE
    _ensure_data_dir(path)
    data: dict[str, Any] = {"history_id": history_id}
    if expiration_ts is not None:
        data["expiration_ts"] = expiration_ts
    try:
        path.write_text(json.dumps(data, indent=0), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write mail push state file %s: %s", path, e)


def _get_push_state_postgres(database_url: str) -> dict[str, Any] | None:
    try:
        import psycopg
    except ImportError:
        logger.warning("psycopg not available, falling back to file for push state")
        return _get_push_state_file()

    try:
        with psycopg.connect(database_url, prepare_threshold=0) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mail_push_state (
                        id TEXT PRIMARY KEY DEFAULT 'default',
                        history_id TEXT,
                        expiration_ts BIGINT,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "SELECT history_id, expiration_ts FROM mail_push_state WHERE id = 'default'"
                )
                row = cur.fetchone()
                if row and row[0]:
                    return {"history_id": row[0], "expiration_ts": row[1]}
                return None
    except Exception as e:
        logger.warning("Failed to get push state from Postgres: %s", e)
        return None


def _set_push_state_postgres(
    database_url: str,
    history_id: str,
    expiration_ts: int | None,
) -> None:
    try:
        import psycopg
    except ImportError:
        _set_push_state_file(history_id, expiration_ts)
        return

    try:
        with psycopg.connect(database_url, prepare_threshold=0) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mail_push_state (
                        id TEXT PRIMARY KEY DEFAULT 'default',
                        history_id TEXT,
                        expiration_ts BIGINT,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO mail_push_state (id, history_id, expiration_ts, updated_at)
                    VALUES ('default', %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE SET
                        history_id = EXCLUDED.history_id,
                        expiration_ts = EXCLUDED.expiration_ts,
                        updated_at = NOW()
                    """,
                    (history_id, expiration_ts),
                )
                conn.commit()
    except Exception as e:
        logger.warning("Failed to set push state in Postgres: %s", e)
        _set_push_state_file(history_id, expiration_ts)


# ---------------------------------------------------------------------------
# Owner choices (don_t_reply, reply_style, etc.)
# ---------------------------------------------------------------------------


def get_owner_choice(database_url: str | None, email_id: str) -> str | None:
    """Return stored choice for this email_id (e.g. 'don_t_reply', 'apologetic') or None."""
    if database_url and database_url.strip():
        return _get_owner_choice_postgres(database_url, email_id)
    return _get_owner_choice_file(email_id)


def set_owner_choice(
    database_url: str | None,
    email_id: str,
    choice: str,
) -> None:
    """Persist owner choice for this email_id."""
    if database_url and database_url.strip():
        _set_owner_choice_postgres(database_url, email_id, choice)
    else:
        _set_owner_choice_file(email_id, choice)


def _get_owner_choice_file(email_id: str) -> str | None:
    path = DEFAULT_OWNER_CHOICES_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data.get(email_id)
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read owner choices file %s: %s", path, e)
        return None


def _set_owner_choice_file(email_id: str, choice: str) -> None:
    path = DEFAULT_OWNER_CHOICES_FILE
    _ensure_data_dir(path)
    data: dict[str, str] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (json.JSONDecodeError, OSError):
            pass
    data[email_id] = choice
    try:
        path.write_text(json.dumps(data, indent=0), encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write owner choices file %s: %s", path, e)


def _get_owner_choice_postgres(database_url: str, email_id: str) -> str | None:
    try:
        import psycopg
    except ImportError:
        return _get_owner_choice_file(email_id)

    try:
        with psycopg.connect(database_url, prepare_threshold=0) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mail_owner_choices (
                        email_id TEXT PRIMARY KEY,
                        choice TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    "SELECT choice FROM mail_owner_choices WHERE email_id = %s",
                    (email_id,),
                )
                row = cur.fetchone()
                return row[0] if row else None
    except Exception as e:
        logger.warning("Failed to get owner choice from Postgres: %s", e)
        return _get_owner_choice_file(email_id)


def _set_owner_choice_postgres(database_url: str, email_id: str, choice: str) -> None:
    try:
        import psycopg
    except ImportError:
        _set_owner_choice_file(email_id, choice)
        return

    try:
        with psycopg.connect(database_url, prepare_threshold=0) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS mail_owner_choices (
                        email_id TEXT PRIMARY KEY,
                        choice TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO mail_owner_choices (email_id, choice, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (email_id) DO UPDATE SET
                        choice = EXCLUDED.choice,
                        updated_at = NOW()
                    """,
                    (email_id, choice),
                )
                conn.commit()
    except Exception as e:
        logger.warning("Failed to set owner choice in Postgres: %s", e)
        _set_owner_choice_file(email_id, choice)
