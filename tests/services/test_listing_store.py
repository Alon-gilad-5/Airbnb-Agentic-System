from app.services.listing_store import ListingStore, create_listing_store


def test_listing_store_returns_empty_for_empty_ids() -> None:
    store = ListingStore(database_url="postgresql://example")

    assert store.get_listings_by_ids([], ["review_scores_rating"]) == []


def test_listing_store_rejects_unsupported_columns() -> None:
    store = ListingStore(database_url="postgresql://example")

    try:
        store.get_listings_by_ids(["123"], ["drop table users"])
    except ValueError as exc:
        assert "Unsupported listing column" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported columns")


def test_create_listing_store_returns_none_without_database_url() -> None:
    assert create_listing_store(None) is None
