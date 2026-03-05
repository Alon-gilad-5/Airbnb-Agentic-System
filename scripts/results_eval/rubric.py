"""Helpers for optional manual rubric scoring files."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


RUBRIC_COLUMNS = [
    "case_id",
    "split",
    "grounding",
    "actionability",
    "tone_policy_safety",
    "notes",
]


def ensure_rubric_template(
    *,
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Create or augment a rubric template CSV with missing cases."""

    existing: list[dict[str, Any]] = []
    file_exists = path.exists()
    if path.exists():
        with path.open("r", encoding="utf-8", newline="") as handle:
            existing = list(csv.DictReader(handle))

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in existing:
        key = (str(row.get("case_id", "")).strip(), str(row.get("split", "")).strip().lower())
        if key[0]:
            by_key[key] = {
                "case_id": key[0],
                "split": key[1],
                "grounding": str(row.get("grounding", "")).strip(),
                "actionability": str(row.get("actionability", "")).strip(),
                "tone_policy_safety": str(row.get("tone_policy_safety", "")).strip(),
                "notes": str(row.get("notes", "")).strip(),
            }

    added_new = False
    for row in rows:
        key = (str(row.get("case_id", "")).strip(), str(row.get("split", "")).strip().lower())
        if not key[0] or key in by_key:
            continue
        added_new = True
        by_key[key] = {
            "case_id": key[0],
            "split": key[1],
            "grounding": "",
            "actionability": "",
            "tone_policy_safety": "",
            "notes": "",
        }

    if file_exists and not added_new:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUBRIC_COLUMNS)
        writer.writeheader()
        for key in sorted(by_key.keys()):
            writer.writerow(by_key[key])


def load_rubric_rows(path: Path) -> list[dict[str, Any]]:
    """Load rubric CSV rows; return empty list when not found."""

    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
