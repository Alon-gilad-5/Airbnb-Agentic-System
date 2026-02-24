#!/usr/bin/env python3
"""Optimize reviews relevance threshold from labeled benchmark data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CaseLabel:
    """Gold label + candidate scores for one benchmark case."""

    case_id: str
    should_answer: bool
    relevant_vector_ids: set[str]
    candidates: list[tuple[str, float]]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL rows."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                out.append(json.loads(stripped))
    return out


def dedupe_label_pool_rows(label_pool_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Drop duplicate property-topic-prompt cases while preserving stable first occurrence."""

    deduped: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    dropped = 0
    for row in label_pool_rows:
        key = (
            str(row.get("property_id", "")).strip(),
            str(row.get("topic", "")).strip(),
            str(row.get("prompt", "")).strip(),
        )
        if key in seen_keys:
            dropped += 1
            continue
        seen_keys.add(key)
        deduped.append(row)
    return deduped, dropped


def load_gold_csv(path: Path) -> dict[str, dict[str, Any]]:
    """Load manual gold labels keyed by case_id."""

    if not path.exists():
        raise FileNotFoundError(f"Gold CSV file not found: {path}")
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"case_id", "should_answer", "relevant_vector_ids"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Gold CSV missing required columns: {required}")
        for row in reader:
            case_id = str(row.get("case_id", "")).strip()
            if not case_id:
                continue
            should_answer_raw = str(row.get("should_answer", "")).strip()
            if should_answer_raw not in {"0", "1"}:
                raise ValueError(
                    f"Invalid should_answer for case {case_id}: '{should_answer_raw}'. "
                    "Fill outputs/reviews_threshold_gold.csv with 0/1 values before optimization."
                )
            relevant_raw = str(row.get("relevant_vector_ids", "")).strip()
            relevant_ids = {x.strip() for x in relevant_raw.split("|") if x.strip()}
            out[case_id] = {
                "should_answer": should_answer_raw == "1",
                "relevant_vector_ids": relevant_ids,
            }
    return out


def build_case_labels(
    *,
    label_pool_rows: list[dict[str, Any]],
    gold_rows: dict[str, dict[str, Any]],
) -> list[CaseLabel]:
    """Join label pool and manual gold rows into case-level eval records."""

    out: list[CaseLabel] = []
    for row in label_pool_rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        if case_id not in gold_rows:
            raise ValueError(f"Gold labels missing case_id: {case_id}")
        gold = gold_rows[case_id]
        candidates_raw = row.get("candidates", []) or []
        candidates: list[tuple[str, float]] = []
        for c in candidates_raw:
            vector_id = str(c.get("vector_id", "")).strip()
            if not vector_id:
                continue
            score = float(c.get("score", 0.0))
            candidates.append((vector_id, score))
        out.append(
            CaseLabel(
                case_id=case_id,
                should_answer=bool(gold["should_answer"]),
                relevant_vector_ids=set(gold["relevant_vector_ids"]),
                candidates=candidates,
            )
        )
    if not out:
        raise ValueError("No joined cases available for optimization.")
    return out


def safe_div(num: float, den: float) -> float:
    """Return 0.0 when denominator is zero."""

    if den == 0:
        return 0.0
    return num / den


def evaluate_threshold(
    *,
    cases: list[CaseLabel],
    threshold: float,
    weights: tuple[float, float, float],
) -> dict[str, Any]:
    """Compute threshold metrics + weighted objective."""

    tp_answer = 0
    fp_answer = 0
    fn_answer = 0
    tn_answer = 0
    selected_total = 0
    relevant_selected = 0
    predicted_answer_count = 0

    for case in cases:
        selected_ids = {vector_id for vector_id, score in case.candidates if score >= threshold}
        predicted_answer = bool(selected_ids)
        if predicted_answer:
            predicted_answer_count += 1
        if case.should_answer and predicted_answer:
            tp_answer += 1
        elif (not case.should_answer) and predicted_answer:
            fp_answer += 1
        elif case.should_answer and (not predicted_answer):
            fn_answer += 1
        else:
            tn_answer += 1

        selected_total += len(selected_ids)
        relevant_selected += len(selected_ids & case.relevant_vector_ids)

    answer_precision = safe_div(tp_answer, tp_answer + fp_answer)
    answer_recall = safe_div(tp_answer, tp_answer + fn_answer)
    evidence_precision = safe_div(relevant_selected, selected_total)
    fp_answer_rate = safe_div(fp_answer, predicted_answer_count)
    w1, w2, w3 = weights
    objective_j = (w1 * answer_precision) + (w2 * answer_recall) + (w3 * evidence_precision)

    return {
        "threshold": round(threshold, 2),
        "objective_j": objective_j,
        "answer_precision": answer_precision,
        "answer_recall": answer_recall,
        "evidence_precision": evidence_precision,
        "fp_answer_rate": fp_answer_rate,
        "tp_answer": tp_answer,
        "fp_answer": fp_answer,
        "fn_answer": fn_answer,
        "tn_answer": tn_answer,
        "predicted_answer_count": predicted_answer_count,
        "selected_total": selected_total,
        "relevant_selected": relevant_selected,
        "case_count": len(cases),
    }


def build_threshold_grid(start: float, stop: float, step: float) -> list[float]:
    """Build rounded inclusive grid."""

    if step <= 0:
        raise ValueError("threshold_step must be > 0")
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 2))
        current += step
    return values


def choose_winner(
    *,
    metrics: list[dict[str, Any]],
    fp_answer_rate_max: float,
) -> tuple[dict[str, Any], bool]:
    """Choose winner under safety constraint with explicit tie-breaks."""

    valid = [m for m in metrics if float(m["fp_answer_rate"]) <= fp_answer_rate_max]
    constrained = True
    pool = valid
    if not pool:
        constrained = False
        pool = metrics

    # Primary: objective desc. Tie-breaker: answer precision desc. Tie-breaker: threshold desc.
    winner = sorted(
        pool,
        key=lambda m: (float(m["objective_j"]), float(m["answer_precision"]), float(m["threshold"])),
        reverse=True,
    )[0]
    return winner, constrained


def percentile(values: list[float], p: float) -> float:
    """Linear percentile without numpy."""

    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    idx = (len(sorted_vals) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_vals[lo]
    fraction = idx - lo
    return sorted_vals[lo] + ((sorted_vals[hi] - sorted_vals[lo]) * fraction)


def bootstrap_ci(
    *,
    cases: list[CaseLabel],
    threshold: float,
    weights: tuple[float, float, float],
    iterations: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Bootstrap mean + 95% CI for key metrics at winner threshold."""

    rng = random.Random(seed)
    j_vals: list[float] = []
    prec_vals: list[float] = []
    rec_vals: list[float] = []
    for _ in range(iterations):
        sample = [rng.choice(cases) for _ in range(len(cases))]
        metric = evaluate_threshold(cases=sample, threshold=threshold, weights=weights)
        j_vals.append(float(metric["objective_j"]))
        prec_vals.append(float(metric["answer_precision"]))
        rec_vals.append(float(metric["answer_recall"]))
    return {
        "objective_j": {
            "mean": safe_div(sum(j_vals), len(j_vals)),
            "ci95_low": percentile(j_vals, 0.025),
            "ci95_high": percentile(j_vals, 0.975),
        },
        "answer_precision": {
            "mean": safe_div(sum(prec_vals), len(prec_vals)),
            "ci95_low": percentile(prec_vals, 0.025),
            "ci95_high": percentile(prec_vals, 0.975),
        },
        "answer_recall": {
            "mean": safe_div(sum(rec_vals), len(rec_vals)),
            "ci95_low": percentile(rec_vals, 0.025),
            "ci95_high": percentile(rec_vals, 0.975),
        },
    }


def write_curve_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write threshold curve CSV for plotting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "threshold",
        "objective_j",
        "answer_precision",
        "answer_recall",
        "evidence_precision",
        "fp_answer_rate",
        "tp_answer",
        "fp_answer",
        "fn_answer",
        "tn_answer",
        "predicted_answer_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_recommendation_md(
    *,
    path: Path,
    winner: dict[str, Any],
    baseline_row: dict[str, Any] | None,
    constrained: bool,
    fp_answer_rate_max: float,
) -> None:
    """Write concise threshold recommendation markdown."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Reviews Threshold Recommendation",
        "",
        f"Recommended `REVIEWS_RELEVANCE_SCORE_THRESHOLD`: **{winner['threshold']:.2f}**",
        "",
        "## Selection rationale",
        f"- Objective `J = 0.50*answer_precision + 0.30*answer_recall + 0.20*evidence_precision` maximized at this threshold.",
        f"- Safety constraint: `fp_answer_rate <= {fp_answer_rate_max:.2f}` "
        + ("(satisfied)." if constrained else "(not satisfiable by any grid candidate; best unconstrained value chosen)."),
        f"- Tie-breakers applied: higher answer_precision, then higher threshold.",
        "",
        "## Winner metrics",
        f"- objective_j: {winner['objective_j']:.6f}",
        f"- answer_precision: {winner['answer_precision']:.6f}",
        f"- answer_recall: {winner['answer_recall']:.6f}",
        f"- evidence_precision: {winner['evidence_precision']:.6f}",
        f"- fp_answer_rate: {winner['fp_answer_rate']:.6f}",
    ]
    if baseline_row:
        lines.extend(
            [
                "",
                "## Comparison vs baseline (0.40)",
                f"- baseline objective_j: {baseline_row['objective_j']:.6f}",
                f"- baseline answer_precision: {baseline_row['answer_precision']:.6f}",
                f"- baseline answer_recall: {baseline_row['answer_recall']:.6f}",
                f"- baseline evidence_precision: {baseline_row['evidence_precision']:.6f}",
                f"- baseline fp_answer_rate: {baseline_row['fp_answer_rate']:.6f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """CLI for threshold optimization from labeled benchmark data."""

    parser = argparse.ArgumentParser(description="Optimize ReviewsAgent relevance threshold.")
    parser.add_argument("--label-pool-path", default="outputs/reviews_threshold_label_pool.jsonl")
    parser.add_argument("--gold-path", default="outputs/reviews_threshold_gold.csv")
    parser.add_argument("--report-path", default="outputs/reviews_threshold_report.json")
    parser.add_argument("--curve-path", default="outputs/reviews_threshold_curve.csv")
    parser.add_argument("--recommendation-path", default="outputs/reviews_threshold_recommendation.md")
    parser.add_argument("--threshold-start", type=float, default=0.10)
    parser.add_argument("--threshold-stop", type=float, default=0.80)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--fp-answer-rate-max", type=float, default=0.15)
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    """Run threshold optimization and write report artifacts."""

    args = build_parser().parse_args()
    weights = (0.50, 0.30, 0.20)

    label_pool_raw = load_jsonl(Path(args.label_pool_path))
    label_pool, duplicate_case_rows_dropped = dedupe_label_pool_rows(label_pool_raw)
    gold_rows = load_gold_csv(Path(args.gold_path))
    cases = build_case_labels(label_pool_rows=label_pool, gold_rows=gold_rows)

    grid = build_threshold_grid(args.threshold_start, args.threshold_stop, args.threshold_step)
    metrics = [
        evaluate_threshold(cases=cases, threshold=t, weights=weights)
        for t in grid
    ]
    winner, constrained = choose_winner(metrics=metrics, fp_answer_rate_max=args.fp_answer_rate_max)
    bootstrap = bootstrap_ci(
        cases=cases,
        threshold=float(winner["threshold"]),
        weights=weights,
        iterations=args.bootstrap_iters,
        seed=args.seed,
    )

    report = {
        "summary": {
            "winner_threshold": winner["threshold"],
            "constraint_satisfied": constrained,
            "fp_answer_rate_max": args.fp_answer_rate_max,
            "weights": {"answer_precision": 0.50, "answer_recall": 0.30, "evidence_precision": 0.20},
            "tie_breakers": ["higher answer_precision", "higher threshold"],
            "case_count": len(cases),
            "duplicate_case_rows_dropped": duplicate_case_rows_dropped,
        },
        "winner_metrics": winner,
        "bootstrap_ci": bootstrap,
        "threshold_metrics": metrics,
    }
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    write_curve_csv(Path(args.curve_path), metrics)
    baseline = next((m for m in metrics if abs(float(m["threshold"]) - 0.40) < 1e-9), None)
    write_recommendation_md(
        path=Path(args.recommendation_path),
        winner=winner,
        baseline_row=baseline,
        constrained=constrained,
        fp_answer_rate_max=args.fp_answer_rate_max,
    )
    print(
        "[done] threshold optimization completed: "
        f"winner={winner['threshold']:.2f} objective_j={winner['objective_j']:.6f} "
        f"constraint_satisfied={constrained}"
    )


if __name__ == "__main__":
    main()
