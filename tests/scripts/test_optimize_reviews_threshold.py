from __future__ import annotations

from scripts.optimize_reviews_threshold import (
    CaseLabel,
    choose_winner,
    dedupe_label_pool_rows,
    evaluate_threshold,
)


def test_optimizer_metrics_and_constraint_choice() -> None:
    cases = [
        CaseLabel(
            case_id="c1",
            should_answer=True,
            relevant_vector_ids={"a"},
            candidates=[("a", 0.70), ("b", 0.30)],
        ),
        CaseLabel(
            case_id="c2",
            should_answer=False,
            relevant_vector_ids=set(),
            candidates=[("c", 0.60)],
        ),
        CaseLabel(
            case_id="c3",
            should_answer=True,
            relevant_vector_ids={"d"},
            candidates=[("d", 0.20)],
        ),
    ]
    weights = (0.50, 0.30, 0.20)
    m050 = evaluate_threshold(cases=cases, threshold=0.50, weights=weights)
    m070 = evaluate_threshold(cases=cases, threshold=0.70, weights=weights)

    assert m050["fp_answer_rate"] == 0.5
    assert m070["fp_answer_rate"] == 0.0
    assert m070["objective_j"] > m050["objective_j"]

    winner, constrained = choose_winner(metrics=[m050, m070], fp_answer_rate_max=0.15)
    assert constrained is True
    assert winner["threshold"] == 0.7


def test_dedupe_label_pool_rows_by_property_topic_prompt() -> None:
    rows = [
        {"case_id": "case_001", "property_id": "p1", "topic": "wifi", "prompt": "wifi?", "candidates": []},
        {"case_id": "case_002", "property_id": "p1", "topic": "wifi", "prompt": "wifi?", "candidates": []},
        {"case_id": "case_003", "property_id": "p1", "topic": "noise", "prompt": "noise?", "candidates": []},
    ]
    deduped, dropped = dedupe_label_pool_rows(rows)
    assert dropped == 1
    assert [row["case_id"] for row in deduped] == ["case_001", "case_003"]
