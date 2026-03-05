"""Reporting helpers for results-evaluation outputs."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from scripts.results_eval.common import load_json


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Return Wilson score 95% interval for a binomial proportion."""

    if total <= 0:
        return 0.0, 0.0
    p_hat = successes / total
    z2 = z * z
    denom = 1.0 + (z2 / total)
    center = (p_hat + (z2 / (2.0 * total))) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _primary_counts(metrics: dict[str, Any]) -> tuple[int | None, int | None]:
    hits = _coerce_int(metrics.get("primary_metric_hits"))
    total = _coerce_int(metrics.get("primary_metric_total"))
    if hits is None or total is None or total <= 0:
        return None, None
    return hits, total


def _task_counts(item: dict[str, Any], metrics: dict[str, Any]) -> tuple[int | None, int | None]:
    hits = _coerce_int(metrics.get("task_success_passes"))
    total = _coerce_int(metrics.get("task_success_total"))
    if hits is not None and total is not None and total > 0:
        return hits, total

    case_results = item.get("case_results")
    if isinstance(case_results, list) and case_results:
        total = len(case_results)
        hits = sum(1 for row in case_results if bool(row.get("pass")))
        return hits, total

    total = _coerce_int(metrics.get("case_count"))
    if total is None or total <= 0:
        return None, None
    rate = float(item.get("task_success_rate", 0.0))
    hits = int(round(rate * total))
    return hits, total


def _format_rate_with_ci(rate: float, hits: int | None, total: int | None) -> str:
    if hits is None or total is None or total <= 0:
        return f"{rate:.4f} (n/a)"
    low, high = _wilson_ci(hits, total)
    return f"{rate:.4f} ({hits}/{total}; 95% CI {low:.3f}-{high:.3f})"


def write_per_agent_csv(*, summary: dict[str, Any], output_path: Path) -> None:
    """Export per-agent metrics to flat CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    agents = summary.get("agents", {})
    for agent_name in sorted(agents.keys()):
        item = agents[agent_name]
        reliability = item.get("reliability", {})
        metrics = item.get("metrics", {})
        primary_hits, primary_total = _primary_counts(metrics)
        task_hits, task_total = _task_counts(item, metrics)
        primary_low, primary_high = _wilson_ci(primary_hits, primary_total) if primary_hits is not None and primary_total is not None else (None, None)
        task_low, task_high = _wilson_ci(task_hits, task_total) if task_hits is not None and task_total is not None else (None, None)
        rows.append(
            {
                "agent": agent_name,
                "split": summary.get("split"),
                "primary_metric_name": item.get("primary_metric_name"),
                "primary_metric": item.get("primary_metric"),
                "primary_metric_hits": primary_hits,
                "primary_metric_total": primary_total,
                "primary_metric_ci95_low": None if primary_low is None else round(primary_low, 4),
                "primary_metric_ci95_high": None if primary_high is None else round(primary_high, 4),
                "task_success_rate": item.get("task_success_rate"),
                "task_success_passes": task_hits,
                "task_success_total": task_total,
                "task_success_ci95_low": None if task_low is None else round(task_low, 4),
                "task_success_ci95_high": None if task_high is None else round(task_high, 4),
                "case_count": metrics.get("case_count"),
                "crash_free_rate": reliability.get("crash_free_rate"),
                "contract_pass_rate": reliability.get("contract_pass_rate"),
                "step_trace_completeness": reliability.get("step_trace_completeness"),
                "p95_latency_ms": reliability.get("p95_latency_ms"),
            }
        )
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "agent",
                "split",
                "primary_metric_name",
                "primary_metric",
                "primary_metric_hits",
                "primary_metric_total",
                "primary_metric_ci95_low",
                "primary_metric_ci95_high",
                "task_success_rate",
                "task_success_passes",
                "task_success_total",
                "task_success_ci95_low",
                "task_success_ci95_high",
                "case_count",
                "crash_free_rate",
                "contract_pass_rate",
                "step_trace_completeness",
                "p95_latency_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _dataset_table(repo_root: Path) -> str:
    overall = load_json(repo_root / "eda_output" / "reviews_overall_summary.json")
    return (
        "| Regions | Total Reviews | Unique Properties | Unique Reviewers | Duplicate Rows |\n"
        "|---:|---:|---:|---:|---:|\n"
        f"| {overall.get('regions', 0)} | {overall.get('total_reviews', 0)} | "
        f"{overall.get('unique_properties_global', 0)} | {overall.get('unique_reviewers_global', 0)} | "
        f"{overall.get('duplicate_review_rows_total', 0)} |\n"
    )


def _per_agent_table(summary: dict[str, Any]) -> str:
    agents = summary.get("agents", {})
    lines = [
        "| Agent | Primary Metric (x/n, 95% CI) | Task Success (x/n, 95% CI) | Crash-Free | Contract | Step Trace | P95 Latency (ms) | Cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_name in sorted(agents.keys()):
        item = agents[agent_name]
        reliability = item.get("reliability", {})
        metrics = item.get("metrics", {})
        primary_hits, primary_total = _primary_counts(metrics)
        task_hits, task_total = _task_counts(item, metrics)
        primary_display = _format_rate_with_ci(
            float(item.get("primary_metric", 0.0)),
            primary_hits,
            primary_total,
        )
        task_display = _format_rate_with_ci(
            float(item.get("task_success_rate", 0.0)),
            task_hits,
            task_total,
        )
        lines.append(
            "| {agent} | {primary} | {task} | {crash:.4f} | {contract:.4f} | {trace:.4f} | {p95:.3f} | {cases} |".format(
                agent=agent_name,
                primary=primary_display,
                task=task_display,
                crash=float(reliability.get("crash_free_rate", 0.0)),
                contract=float(reliability.get("contract_pass_rate", 0.0)),
                trace=float(reliability.get("step_trace_completeness", 0.0)),
                p95=float(reliability.get("p95_latency_ms", 0.0)),
                cases=int(metrics.get("case_count", 0)),
            )
        )
    return "\n".join(lines) + "\n"


def _reviews_ablation_table(summary: dict[str, Any]) -> str:
    reviews = summary.get("agents", {}).get("reviews", {})
    ablation = reviews.get("ablation", {})
    baseline = ablation.get("baseline_on_split", {})
    tuned = ablation.get("tuned_on_split", {})
    return (
        "| Setting | Threshold | Answer Decision Accuracy | Evidence Precision | Task Success |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Baseline | {ablation.get('baseline_threshold', 0.40)} | "
        f"{float(baseline.get('answer_decision_accuracy', 0.0)):.4f} | "
        f"{float(baseline.get('evidence_precision', 0.0)):.4f} | "
        f"{float(baseline.get('task_success_rate', 0.0)):.4f} |\n"
        f"| Tuned (dev-selected) | {ablation.get('tuned_threshold', 0.40)} | "
        f"{float(tuned.get('answer_decision_accuracy', 0.0)):.4f} | "
        f"{float(tuned.get('evidence_precision', 0.0)):.4f} | "
        f"{float(tuned.get('task_success_rate', 0.0)):.4f} |\n"
    )


def build_results_markdown(*, repo_root: Path, summary: dict[str, Any]) -> str:
    """Build paper-ready markdown tables and limitations notes."""

    reviews_rubric = summary.get("rubric", {}).get("reviews", {})
    mail_rubric = summary.get("rubric", {}).get("mail", {})
    return (
        "# Results Tables\n\n"
        "## Table 1. Dataset profile\n\n"
        + _dataset_table(repo_root)
        + "\n## Table 2. Per-agent quality and reliability metrics\n\n"
        + _per_agent_table(summary)
        + "\n## Table 3. Reviews threshold ablation\n\n"
        + _reviews_ablation_table(summary)
        + "\n## Manual rubric scoring status\n\n"
        f"- Reviews rubric status: `{reviews_rubric.get('status', 'not_scored')}` "
        f"(scored_cases={reviews_rubric.get('scored_cases', 0)}).\n"
        f"- Mail rubric status: `{mail_rubric.get('status', 'not_scored')}` "
        f"(scored_cases={mail_rubric.get('scored_cases', 0)}).\n\n"
        "## Limitations\n\n"
        "- Manual rubric scoring is single-annotator in this cycle.\n"
        "- Evaluation is offline-first with synthetic fixtures for analyst/pricing/market-watch.\n"
        "- Reported results support task quality + reliability claims, not direct business-impact claims.\n"
    )
