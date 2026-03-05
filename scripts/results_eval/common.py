"""Shared contracts and utility functions for results evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class EvalCase:
    """One evaluation case loaded from JSONL."""

    case_id: str
    split: str
    prompt: str
    context: dict[str, Any]
    expected: dict[str, Any]
    tags: list[str]
    raw: dict[str, Any]


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(tz=UTC).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    """Load a UTF-8 JSON document."""

    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a UTF-8 JSON document."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL rows."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write deterministic UTF-8 JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def parse_cases(path: Path, *, split: str | None = None) -> list[EvalCase]:
    """Load and validate common case schema from JSONL."""

    rows = load_jsonl(path)
    out: list[EvalCase] = []
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        row_split = str(row.get("split", "")).strip().lower()
        prompt = str(row.get("prompt", "")).strip()
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        expected = row.get("expected") if isinstance(row.get("expected"), dict) else {}
        tags_raw = row.get("tags") if isinstance(row.get("tags"), list) else []
        tags = [str(tag).strip() for tag in tags_raw if str(tag).strip()]
        if not case_id or row_split not in {"dev", "test"} or not prompt:
            raise ValueError(f"Malformed evaluation case row in {path}: {row}")
        if split is not None and row_split != split:
            continue
        out.append(
            EvalCase(
                case_id=case_id,
                split=row_split,
                prompt=prompt,
                context=context,
                expected=expected,
                tags=tags,
                raw=row,
            )
        )
    out.sort(key=lambda item: item.case_id)
    return out


def percentile(values: list[float], p: float) -> float:
    """Linear percentile without numpy dependency."""

    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    sorted_vals = sorted(values)
    idx = (len(sorted_vals) - 1) * p
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    if lower == upper:
        return sorted_vals[lower]
    fraction = idx - lower
    return sorted_vals[lower] + ((sorted_vals[upper] - sorted_vals[lower]) * fraction)


def compute_reliability(case_results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute reliability metrics required by the results tables."""

    if not case_results:
        return {
            "crash_free_rate": 0.0,
            "contract_pass_rate": 0.0,
            "step_trace_completeness": 0.0,
            "p95_latency_ms": 0.0,
        }
    crash_free = mean(0.0 if bool(row.get("metadata", {}).get("exception")) else 1.0 for row in case_results)
    contract_pass = mean(1.0 if bool(row.get("metadata", {}).get("contract_ok")) else 0.0 for row in case_results)
    trace_ok = mean(1.0 if bool(row.get("metadata", {}).get("step_trace_ok")) else 0.0 for row in case_results)
    latencies = [float(row.get("latency_ms", 0.0)) for row in case_results]
    return {
        "crash_free_rate": round(crash_free, 4),
        "contract_pass_rate": round(contract_pass, 4),
        "step_trace_completeness": round(trace_ok, 4),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
    }


def compute_task_success_rate(case_results: list[dict[str, Any]]) -> float:
    """Compute share of passing cases."""

    if not case_results:
        return 0.0
    return round(mean(1.0 if bool(row.get("pass")) else 0.0 for row in case_results), 4)


def summarize_rubric_scores(
    *,
    rubric_rows: list[dict[str, Any]],
    split: str,
) -> dict[str, Any]:
    """Aggregate optional manual rubric scores (0-6 total)."""

    filtered = [row for row in rubric_rows if str(row.get("split", "")).strip().lower() == split]
    totals: list[int] = []
    for row in filtered:
        try:
            grounding = int(row["grounding"])
            actionability = int(row["actionability"])
            tone = int(row["tone_policy_safety"])
        except Exception:
            continue
        if not all(0 <= value <= 2 for value in (grounding, actionability, tone)):
            continue
        totals.append(grounding + actionability + tone)

    distribution = {str(score): 0 for score in range(7)}
    for total in totals:
        distribution[str(total)] += 1

    if not totals:
        return {
            "status": "not_scored",
            "scored_cases": 0,
            "mean_total_score": None,
            "distribution": distribution,
        }

    return {
        "status": "scored",
        "scored_cases": len(totals),
        "mean_total_score": round(mean(totals), 4),
        "distribution": distribution,
    }

