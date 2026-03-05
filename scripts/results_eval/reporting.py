"""Reporting helpers for results-evaluation outputs."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from scripts.results_eval.common import load_json


def write_per_agent_csv(*, summary: dict[str, Any], output_path: Path) -> None:
    """Export per-agent metrics to flat CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    agents = summary.get("agents", {})
    for agent_name in sorted(agents.keys()):
        item = agents[agent_name]
        reliability = item.get("reliability", {})
        metrics = item.get("metrics", {})
        rows.append(
            {
                "agent": agent_name,
                "split": summary.get("split"),
                "primary_metric_name": item.get("primary_metric_name"),
                "primary_metric": item.get("primary_metric"),
                "task_success_rate": item.get("task_success_rate"),
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
                "task_success_rate",
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
        "| Agent | Primary Metric | Task Success | Crash-Free | Contract | Step Trace | P95 Latency (ms) | Cases |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for agent_name in sorted(agents.keys()):
        item = agents[agent_name]
        reliability = item.get("reliability", {})
        metrics = item.get("metrics", {})
        lines.append(
            "| {agent} | {primary:.4f} | {task:.4f} | {crash:.4f} | {contract:.4f} | {trace:.4f} | {p95:.3f} | {cases} |".format(
                agent=agent_name,
                primary=float(item.get("primary_metric", 0.0)),
                task=float(item.get("task_success_rate", 0.0)),
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

