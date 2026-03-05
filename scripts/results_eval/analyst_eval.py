"""Offline evaluator for analyst-agent numeric/categorical benchmark logic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from app.agents.analyst_agent import AnalystAgent
from app.schemas import AnalysisCategoricalItem, AnalysisNumericItem, StepLog
from scripts.results_eval.common import compute_reliability, compute_task_success_rate, parse_cases


@dataclass
class _DummyChatService:
    is_available: bool = False
    model: str = "offline-eval"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        raise RuntimeError("Chat unavailable in offline evaluation mode.")


class _DummyNeighborStore:
    def __init__(self, neighbors: list[str] | None = None) -> None:
        self._neighbors = neighbors or []

    def get_neighbors(self, property_id: str) -> list[str] | None:
        return list(self._neighbors)


class _DummyListingStore:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self._rows = rows or []

    def get_listings_by_ids(self, listing_ids: list[str], columns: list[str]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for row in self._rows:
            if str(row.get("id")) not in listing_ids:
                continue
            out.append({column: row.get(column) for column in ["id", "name", *columns]})
        return out


VALID_MODULES = {
    "analyst_agent.neighbor_lookup",
    "analyst_agent.data_fetch",
    "analyst_agent.comparison_compute",
    "analyst_agent.answer_generation",
}


def _is_step_log(step: Any) -> bool:
    return isinstance(step, StepLog) and isinstance(step.module, str) and isinstance(step.prompt, dict) and isinstance(step.response, dict)


def _float_eq(actual: Any, expected: Any, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(actual) - float(expected)) <= tolerance
    except Exception:
        return False


def _check_numeric(
    items: list[AnalysisNumericItem],
    expected_rows: list[dict[str, Any]],
) -> bool:
    by_column = {item.column: item for item in items}
    for row in expected_rows:
        column = str(row.get("column", "")).strip()
        item = by_column.get(column)
        if item is None:
            return False
        if "owner_value" in row and not _float_eq(item.owner_value, row["owner_value"]):
            return False
        if "neighbor_avg" in row and not _float_eq(item.neighbor_avg, row["neighbor_avg"]):
            return False
        if "neighbor_min" in row and not _float_eq(item.neighbor_min, row["neighbor_min"]):
            return False
        if "neighbor_max" in row and not _float_eq(item.neighbor_max, row["neighbor_max"]):
            return False
        if "neighbor_count" in row and int(item.neighbor_count) != int(row["neighbor_count"]):
            return False
    return True


def _check_categorical(
    items: list[AnalysisCategoricalItem],
    expected_rows: list[dict[str, Any]],
) -> bool:
    by_column = {item.column: item for item in items}
    for row in expected_rows:
        column = str(row.get("column", "")).strip()
        item = by_column.get(column)
        if item is None:
            return False
        if "owner_value" in row and str(item.owner_value) != str(row["owner_value"]):
            return False
        if "top_bucket" in row:
            if not item.buckets:
                return False
            if str(item.buckets[0].value) != str(row["top_bucket"]):
                return False
        if "top_bucket_pct" in row:
            if not item.buckets or not _float_eq(item.buckets[0].pct, row["top_bucket_pct"], tolerance=1e-3):
                return False
    return True


def evaluate_analyst(
    *,
    repo_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate analyst agent against deterministic oracle fixtures."""

    cases = parse_cases(repo_root / "eval" / "cases" / "analyst_cases.jsonl", split=split)
    case_results: list[dict[str, Any]] = []
    numeric_ok_hits = 0
    numeric_ok_total = 0

    for case in cases:
        started = time.perf_counter()
        exception = None
        contract_ok = False
        trace_ok = False
        passed = False
        failure_reason = "uninitialized"
        try:
            fixtures = case.raw.get("fixtures") if isinstance(case.raw.get("fixtures"), dict) else {}
            agent = AnalystAgent(
                listing_store=_DummyListingStore(rows=fixtures.get("rows", [])),
                neighbor_store=_DummyNeighborStore(neighbors=fixtures.get("neighbors", [])),
                chat_service=_DummyChatService(),
            )
            outcome = agent.analyze(case.prompt, context=case.context)
            modules = [step.module for step in outcome.steps]
            contract_ok = (
                isinstance(outcome.narrative, str)
                and isinstance(outcome.error, (str, type(None)))
                and all(_is_step_log(step) for step in outcome.steps)
            )
            required_modules = set(case.expected.get("required_modules", ["analyst_agent.answer_generation"]))
            trace_ok = set(modules).issubset(VALID_MODULES) and required_modules.issubset(set(modules))

            expected_status = str(case.expected.get("status", "ok"))
            if expected_status == "error":
                error_contains = str(case.expected.get("error_contains", "")).strip().lower()
                passed = bool(outcome.error) and (error_contains in outcome.error.lower())
                failure_reason = None if passed else "error_expectation_mismatch"
            else:
                numeric_ok_total += 1
                numeric_ok = _check_numeric(
                    outcome.numeric_comparison,
                    case.expected.get("numeric_expectations", []),
                )
                categorical_ok = _check_categorical(
                    outcome.categorical_comparison,
                    case.expected.get("categorical_expectations", []),
                )
                numeric_ok_hits += 1 if numeric_ok else 0
                passed = (
                    outcome.error is None
                    and numeric_ok
                    and categorical_ok
                )
                failure_reason = None if passed else "oracle_mismatch"
        except Exception as exc:  # pragma: no cover - defensive guard
            exception = f"{type(exc).__name__}: {exc}"
            failure_reason = "exception"

        latency_ms = (time.perf_counter() - started) * 1000.0
        case_results.append(
            {
                "agent": "analyst",
                "split": split,
                "case_id": case.case_id,
                "pass": passed,
                "failure_reason": failure_reason,
                "latency_ms": round(latency_ms, 3),
                "metadata": {
                    "contract_ok": contract_ok,
                    "step_trace_ok": trace_ok,
                    "exception": exception,
                },
            }
        )

    reliability = compute_reliability(case_results)
    numeric_accuracy = (numeric_ok_hits / numeric_ok_total) if numeric_ok_total else 0.0
    task_success_rate = compute_task_success_rate(case_results)
    return {
        "agent": "analyst",
        "split": split,
        "primary_metric_name": "numeric_computation_accuracy",
        "primary_metric": round(numeric_accuracy, 4),
        "metrics": {
            "case_count": len(case_results),
            "numeric_computation_accuracy": round(numeric_accuracy, 4),
            "task_success_rate": task_success_rate,
            "primary_metric_hits": numeric_ok_hits,
            "primary_metric_total": numeric_ok_total,
            "task_success_passes": sum(1 for row in case_results if bool(row.get("pass"))),
            "task_success_total": len(case_results),
        },
        "reliability": reliability,
        "task_success_rate": task_success_rate,
        "case_results": case_results,
    }
