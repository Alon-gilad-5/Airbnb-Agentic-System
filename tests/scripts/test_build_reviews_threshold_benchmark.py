from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

from scripts.build_reviews_threshold_benchmark import (
    build_benchmark_cases,
    collect_property_stats,
    load_selection_state_property_ids,
)


def _write_reviews_gz(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_benchmark_cases_is_deterministic(tmp_path: Path) -> None:
    reviews_dir = tmp_path / "reviews"
    _write_reviews_gz(
        reviews_dir / "los angels reviews.gz",
        [
            {"listing_id": "p1", "id": "1", "date": "2025-01-01", "reviewer_id": "r1", "reviewer_name": "a", "comments": "good"},
            {"listing_id": "p1", "id": "2", "date": "2025-01-02", "reviewer_id": "r2", "reviewer_name": "b", "comments": "good"},
            {"listing_id": "p2", "id": "3", "date": "2025-01-03", "reviewer_id": "r3", "reviewer_name": "c", "comments": "ok"},
            {"listing_id": "p3", "id": "4", "date": "2025-01-04", "reviewer_id": "r4", "reviewer_name": "d", "comments": "ok"},
        ],
    )
    _write_reviews_gz(
        reviews_dir / "san diego reviews.gz",
        [
            {"listing_id": "p4", "id": "5", "date": "2025-01-05", "reviewer_id": "r5", "reviewer_name": "e", "comments": "ok"},
            {"listing_id": "p5", "id": "6", "date": "2025-01-06", "reviewer_id": "r6", "reviewer_name": "f", "comments": "ok"},
        ],
    )
    selection_state = tmp_path / "state.json"
    selection_state.write_text(
        json.dumps(
            {
                "namespace": "airbnb-reviews-test",
                "selected_property_ids": ["p1", "p2", "p3", "p4", "p5"],
            }
        ),
        encoding="utf-8",
    )
    selected = load_selection_state_property_ids(
        selection_state_path=selection_state,
        expected_namespace="airbnb-reviews-test",
    )
    props = collect_property_stats(reviews_dir=reviews_dir, allowed_property_ids=selected)

    first = build_benchmark_cases(
        properties=props,
        total_cases=6,
        high_cases=3,
        low_cases=3,
        seed=42,
    )
    second = build_benchmark_cases(
        properties=props,
        total_cases=6,
        high_cases=3,
        low_cases=3,
        seed=42,
    )
    assert first == second
    assert len(first) == 6
    assert sum(1 for c in first if c["tier"] == "high") == 3
    assert sum(1 for c in first if c["tier"] == "low") == 3
    pair_keys = {(str(c["property_id"]), str(c["topic"])) for c in first}
    assert len(pair_keys) == len(first)
