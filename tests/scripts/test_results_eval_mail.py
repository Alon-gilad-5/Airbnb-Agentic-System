from __future__ import annotations

from pathlib import Path

from scripts.results_eval.mail_eval import evaluate_mail


def test_mail_evaluator_reports_classification_and_policy_metrics() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = evaluate_mail(repo_root=repo_root, split="test")
    assert result["agent"] == "mail"
    assert result["primary_metric_name"] == "classification_accuracy"
    assert result["metrics"]["case_count"] >= 1
    assert 0.0 <= result["metrics"]["classification_accuracy"] <= 1.0
    assert 0.0 <= result["metrics"]["policy_action_accuracy"] <= 1.0
    assert "reliability" in result

