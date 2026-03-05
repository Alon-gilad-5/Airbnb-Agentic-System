from __future__ import annotations

from scripts.optimize_reviews_threshold import CaseLabel
from scripts.results_eval.reviews_eval import _evaluate_case, _score_split


def test_evaluate_case_requires_relevant_evidence_hit() -> None:
    label = CaseLabel(
        case_id="case-1",
        should_answer=True,
        relevant_vector_ids={"v1"},
        candidates=[("v1", 0.6), ("v2", 0.4)],
    )
    passed, payload = _evaluate_case(label, threshold=0.5)
    assert passed is True
    assert payload["predicted_answer"] is True
    assert payload["relevant_selected"] == 1


def test_score_split_computes_metrics() -> None:
    labels = [
        CaseLabel(
            case_id="case-1",
            should_answer=True,
            relevant_vector_ids={"v1"},
            candidates=[("v1", 0.6)],
        ),
        CaseLabel(
            case_id="case-2",
            should_answer=True,
            relevant_vector_ids={"v2"},
            candidates=[("v3", 0.6)],
        ),
    ]
    split_map = {"case-1": "test", "case-2": "test"}
    rows, metrics = _score_split(labels=labels, threshold=0.5, split_map=split_map, split="test")
    assert len(rows) == 2
    assert metrics["case_count"] == 2
    assert metrics["answer_decision_accuracy"] == 1.0
    assert metrics["evidence_precision"] == 0.5
    assert metrics["task_success_rate"] == 0.5

