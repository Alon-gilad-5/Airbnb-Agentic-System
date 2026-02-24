from app.services.region_utils import canonicalize_region
from app.services.web_review_ingest import WebReviewIngestService


def test_canonicalize_region_none_and_blank() -> None:
    assert canonicalize_region(None) is None
    assert canonicalize_region("") is None
    assert canonicalize_region("   ") is None


def test_canonicalize_region_normalizes_case_and_spaces() -> None:
    assert canonicalize_region("  San Francisco  ") == "san francisco"


def test_clean_metadata_omits_none_region() -> None:
    service = object.__new__(WebReviewIngestService)
    cleaned = service._clean_metadata(
        {
            "property_id": "123",
            "region": None,
            "review_text": "Good stay.",
        }
    )
    assert "region" not in cleaned
    assert cleaned["property_id"] == "123"

