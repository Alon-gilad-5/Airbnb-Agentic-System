"""Offline reviews-agent evaluation over labeled retrieval benchmark cases."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Any

from scripts.optimize_reviews_threshold import (
    CaseLabel,
    build_case_labels,
    choose_winner,
    dedupe_label_pool_rows,
    evaluate_threshold,
    load_gold_csv,
    load_jsonl as load_pool_jsonl,
)
from scripts.results_eval.common import compute_reliability, compute_task_success_rate, write_jsonl


def _pick_existing_path(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of the candidate paths exists: {candidates}")


def _load_case_metadata(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_pool_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if case_id:
            out[case_id] = row
    return out


def _assign_split(case_ids: list[str], dev_ratio: float = 0.7) -> dict[str, str]:
    ordered = sorted(case_ids)
    dev_count = max(1, int(round(len(ordered) * dev_ratio)))
    if dev_count >= len(ordered):
        dev_count = len(ordered) - 1
    out: dict[str, str] = {}
    for index, case_id in enumerate(ordered):
        out[case_id] = "dev" if index < dev_count else "test"
    return out


def _build_case_files(
    *,
    labels: list[CaseLabel],
    split_map: dict[str, str],
    metadata_by_case: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    rows: list[dict[str, Any]] = []
    for label in sorted(labels, key=lambda item: item.case_id):
        meta = metadata_by_case.get(label.case_id, {})
        rows.append(
            {
                "case_id": label.case_id,
                "split": split_map[label.case_id],
                "prompt": str(meta.get("prompt", "What do guests say?")).strip() or "What do guests say?",
                "context": {
                    "property_id": str(meta.get("property_id", "")).strip() or None,
                    "region": str(meta.get("region", "")).strip() or None,
                },
                "expected": {
                    "should_answer": bool(label.should_answer),
                    "relevant_vector_count": len(label.relevant_vector_ids),
                },
                "tags": [
                    "reviews",
                    str(meta.get("topic", "unknown")),
                    str(meta.get("tier", "unknown")),
                ],
            }
        )
    write_jsonl(output_path, rows)


def _evaluate_case(label: CaseLabel, threshold: float) -> tuple[bool, dict[str, Any]]:
    selected = {vector_id for vector_id, score in label.candidates if score >= threshold}
    predicted_answer = bool(selected)
    decision_ok = predicted_answer == label.should_answer
    relevant_selected = len(selected & label.relevant_vector_ids)
    evidence_hit = (not label.should_answer) or (relevant_selected > 0)
    passed = decision_ok and evidence_hit
    return passed, {
        "threshold": round(threshold, 2),
        "should_answer": label.should_answer,
        "predicted_answer": predicted_answer,
        "decision_ok": decision_ok,
        "selected_total": len(selected),
        "relevant_selected": relevant_selected,
        "evidence_hit": evidence_hit,
        "selected_vector_ids_preview": sorted(selected)[:6],
        "candidate_max_score": round(max((score for _, score in label.candidates), default=0.0), 4),
    }


def _score_split(
    *,
    labels: list[CaseLabel],
    threshold: float,
    split_map: dict[str, str],
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decision_hits = 0
    selected_total = 0
    relevant_selected_total = 0
    for label in sorted(labels, key=lambda item: item.case_id):
        if split_map.get(label.case_id) != split:
            continue
        started = time.perf_counter()
        exception = None
        try:
            passed, payload = _evaluate_case(label, threshold)
        except Exception as exc:  # pragma: no cover - defensive guard
            passed = False
            payload = {}
            exception = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - started) * 1000.0
        decision_ok = bool(payload.get("decision_ok"))
        decision_hits += 1 if decision_ok else 0
        selected_total += int(payload.get("selected_total", 0))
        relevant_selected_total += int(payload.get("relevant_selected", 0))
        rows.append(
            {
                "agent": "reviews",
                "split": split,
                "case_id": label.case_id,
                "pass": passed,
                "failure_reason": None if passed else ("exception" if exception else "decision_or_evidence_mismatch"),
                "latency_ms": round(latency_ms, 3),
                "metadata": {
                    **payload,
                    "contract_ok": exception is None,
                    "step_trace_ok": exception is None,
                    "exception": exception,
                },
            }
        )
    total = len(rows)
    evidence_precision = (relevant_selected_total / selected_total) if selected_total else 0.0
    metrics = {
        "case_count": total,
        "answer_decision_accuracy": round((decision_hits / total) if total else 0.0, 4),
        "primary_metric_hits": decision_hits,
        "primary_metric_total": total,
        "evidence_precision": round(evidence_precision, 4),
        "task_success_rate": compute_task_success_rate(rows),
        "task_success_passes": sum(1 for row in rows if bool(row.get("pass"))),
        "task_success_total": total,
    }
    return rows, metrics


def evaluate_reviews(
    *,
    repo_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate reviews benchmark on selected split with strict dev/test holdout."""

    if split not in {"dev", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    pool_path = _pick_existing_path(
        [
            repo_root / "outputs" / "reviews_threshold_label_pool.jsonl",
            repo_root / "outputs" / "reviews_threshold_label_pool_labeled20.jsonl",
        ]
    )
    gold_path = _pick_existing_path(
        [
            repo_root / "eval" / "cases" / "reviews_threshold_gold.csv",
            repo_root / "outputs" / "reviews_threshold_gold.csv",
            repo_root / "outputs" / "reviews_threshold_gold_labeled20.csv",
        ]
    )
    cases_path = repo_root / "outputs" / "reviews_threshold_cases.jsonl"

    label_pool_rows = load_pool_jsonl(pool_path)
    label_pool_rows, dropped = dedupe_label_pool_rows(label_pool_rows)
    gold_rows = load_gold_csv(gold_path)
    labels = build_case_labels(label_pool_rows=label_pool_rows, gold_rows=gold_rows)

    split_map = _assign_split([label.case_id for label in labels], dev_ratio=0.7)
    metadata_by_case = _load_case_metadata(cases_path) if cases_path.exists() else {}
    _build_case_files(
        labels=labels,
        split_map=split_map,
        metadata_by_case=metadata_by_case,
        output_path=repo_root / "eval" / "cases" / "reviews_cases.jsonl",
    )

    dev_labels = [label for label in labels if split_map[label.case_id] == "dev"]
    dev_threshold_metrics: list[dict[str, Any]] = []
    for threshold in [round(v, 2) for v in [0.1 + (i * 0.02) for i in range(41)]]:
        dev_threshold_metrics.append(
            evaluate_threshold(cases=dev_labels, threshold=threshold, weights=(0.5, 0.3, 0.2))
        )
    winner, constrained = choose_winner(metrics=dev_threshold_metrics, fp_answer_rate_max=0.15)
    tuned_threshold = float(winner["threshold"])
    baseline_threshold = 0.40

    case_results, split_metrics = _score_split(
        labels=labels,
        threshold=tuned_threshold,
        split_map=split_map,
        split=split,
    )
    _, baseline_metrics = _score_split(
        labels=labels,
        threshold=baseline_threshold,
        split_map=split_map,
        split=split,
    )
    reliability = compute_reliability(case_results)

    return {
        "agent": "reviews",
        "split": split,
        "primary_metric_name": "answer_decision_accuracy",
        "primary_metric": split_metrics["answer_decision_accuracy"],
        "metrics": {
            **split_metrics,
            "baseline_answer_decision_accuracy": baseline_metrics["answer_decision_accuracy"],
            "baseline_evidence_precision": baseline_metrics["evidence_precision"],
        },
        "reliability": reliability,
        "task_success_rate": split_metrics["task_success_rate"],
        "ablation": {
            "baseline_threshold": baseline_threshold,
            "tuned_threshold": tuned_threshold,
            "winner_metrics_on_dev": winner,
            "constraint_satisfied": constrained,
            "duplicate_case_rows_dropped": dropped,
            "baseline_on_split": baseline_metrics,
            "tuned_on_split": split_metrics,
        },
        "case_results": case_results,
    }
