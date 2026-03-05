from __future__ import annotations

from pathlib import Path

from scripts.results_eval.reporting import build_results_markdown, write_per_agent_csv


def _summary_fixture() -> dict:
    return {
        "split": "test",
        "agents": {
            "reviews": {
                "primary_metric_name": "answer_decision_accuracy",
                "primary_metric": 0.9,
                "task_success_rate": 0.8,
                "metrics": {"case_count": 5},
                "reliability": {
                    "crash_free_rate": 1.0,
                    "contract_pass_rate": 1.0,
                    "step_trace_completeness": 1.0,
                    "p95_latency_ms": 10.0,
                },
                "ablation": {
                    "baseline_threshold": 0.4,
                    "tuned_threshold": 0.46,
                    "baseline_on_split": {
                        "answer_decision_accuracy": 0.8,
                        "evidence_precision": 0.6,
                        "task_success_rate": 0.7,
                    },
                    "tuned_on_split": {
                        "answer_decision_accuracy": 0.9,
                        "evidence_precision": 0.8,
                        "task_success_rate": 0.8,
                    },
                },
            },
            "mail": {
                "primary_metric_name": "classification_accuracy",
                "primary_metric": 0.95,
                "task_success_rate": 0.9,
                "metrics": {"case_count": 4},
                "reliability": {
                    "crash_free_rate": 1.0,
                    "contract_pass_rate": 1.0,
                    "step_trace_completeness": 0.75,
                    "p95_latency_ms": 15.0,
                },
            },
        },
        "rubric": {
            "reviews": {"status": "not_scored", "scored_cases": 0},
            "mail": {"status": "not_scored", "scored_cases": 0},
        },
    }


def test_reporting_exports_csv_and_markdown() -> None:
    summary = _summary_fixture()
    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "outputs" / "tmp_test_results_eval_reporting"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "per_agent_metrics.csv"
    write_per_agent_csv(summary=summary, output_path=csv_path)
    content = csv_path.read_text(encoding="utf-8")
    assert "agent,split,primary_metric_name" in content
    assert "reviews,test,answer_decision_accuracy" in content

    markdown = build_results_markdown(repo_root=repo_root, summary=summary)
    assert "## Table 1. Dataset profile" in markdown
    assert "## Table 2. Per-agent quality and reliability metrics" in markdown
    assert "## Table 3. Reviews threshold ablation" in markdown
