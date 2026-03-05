from __future__ import annotations

from pathlib import Path

from scripts.results_eval.pricing_eval import evaluate_pricing


def test_pricing_evaluator_returns_expected_shape() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    result = evaluate_pricing(repo_root=repo_root, split="test")
    assert result["agent"] == "pricing"
    assert result["primary_metric_name"] == "recommendation_direction_accuracy"
    assert result["metrics"]["case_count"] >= 1
    assert 0.0 <= result["primary_metric"] <= 1.0
    assert "reliability" in result
    assert len(result["case_results"]) == result["metrics"]["case_count"]

