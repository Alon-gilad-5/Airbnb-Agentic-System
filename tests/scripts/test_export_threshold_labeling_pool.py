from __future__ import annotations

from scripts.export_threshold_labeling_pool import build_case_candidates, metadata_filter_for_case


class _FakeResponse:
    def __init__(self, matches):
        self.matches = matches


class _FakeIndex:
    def query(self, **kwargs):
        return _FakeResponse(
            [
                {"id": "v1", "score": 0.61, "metadata": {"review_text": "A", "review_id": "r1", "property_id": "p1", "region": "x", "review_date": "2025-01-01"}},
                {"id": "v2", "score": 0.55, "metadata": {"review_text": "B", "review_id": "r2", "property_id": "p1", "region": "x", "review_date": "2025-01-02"}},
                {"id": "v3", "score": 0.33, "metadata": {"review_text": "C", "review_id": "r3", "property_id": "p1", "region": "x", "review_date": "2025-01-03"}},
            ]
        )


def test_metadata_filter_for_case_property_and_region() -> None:
    filt = metadata_filter_for_case({"property_id": "p1", "region": " Los Angeles "})
    assert filt == {
        "$and": [
            {"property_id": {"$eq": "p1"}},
            {"region": {"$eq": "los angeles"}},
        ]
    }


def test_build_case_candidates_limits_to_top_k_and_includes_score_and_vector_id() -> None:
    row = build_case_candidates(
        case={
            "case_id": "case_001",
            "property_id": "p1",
            "region": "x",
            "tier": "high",
            "topic": "wifi",
            "prompt": "wifi?",
        },
        embedding=[0.1, 0.2],
        index=_FakeIndex(),
        namespace="ns",
        top_k=2,
        max_review_text_chars=20,
    )
    assert row["candidate_count"] == 2
    assert len(row["candidates"]) == 2
    assert row["candidates"][0]["vector_id"] == "v1"
    assert isinstance(row["candidates"][0]["score"], float)

