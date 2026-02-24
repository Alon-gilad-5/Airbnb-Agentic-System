import pytest

from app.services.web_review_scraper import PlaywrightReviewScraper


def _scraper(*, require_source_selectors: bool = True) -> PlaywrightReviewScraper:
    return PlaywrightReviewScraper(
        enabled=True,
        allowlist=["google_maps", "tripadvisor"],
        default_max_reviews=5,
        timeout_seconds=30,
        require_source_selectors=require_source_selectors,
        min_review_chars=40,
        min_token_count=8,
        reject_private_use_ratio=0.10,
    )


def test_looks_like_review_accepts_valid_text() -> None:
    scraper = _scraper()
    text = (
        "We stayed three nights and the apartment was very clean. "
        "The host responded fast and the location was quiet at night."
    )
    assert scraper._looks_like_review(text) is True


def test_looks_like_review_rejects_ui_noise() -> None:
    scraper = _scraper()
    text = (
        "Sign in to continue and accept cookies before you read more. "
        "Open the main menu, sort by newest, and filter reviews."
    )
    assert scraper._looks_like_review(text) is False


def test_looks_like_review_rejects_private_use_unicode() -> None:
    scraper = _scraper()
    text = (
        "Great location and friendly host. " + ("\ue123" * 25) +
        " Would stay again because check-in was simple."
    )
    assert scraper._looks_like_review(text) is False


def test_extract_reviews_source_selector_hit() -> None:
    scraper = _scraper()
    html = """
    <html><body>
      <span class="wiI7pd">
        We loved the place and the bed was comfortable. The kitchen had everything we needed.
      </span>
    </body></html>
    """
    reviews, meta = scraper._extract_reviews(
        html=html,
        source="google_maps",
        source_url="https://maps.google.com/place/example",
        max_count=3,
    )
    assert len(reviews) == 1
    assert meta["selector_miss"] is False
    assert meta["used_generic_fallback"] is False


def test_extract_reviews_selector_miss_returns_empty_when_required() -> None:
    scraper = _scraper(require_source_selectors=True)
    html = """
    <html><body>
      <p>
        This is long enough to look review-like and includes punctuation.
        It should still be rejected because selectors do not match.
      </p>
    </body></html>
    """
    reviews, meta = scraper._extract_reviews(
        html=html,
        source="google_maps",
        source_url="https://www.google.com/maps/search/example",
        max_count=3,
    )
    assert reviews == []
    assert meta["selector_miss"] is True
    assert meta["selector_miss_reason"] == "no_source_specific_selector_hit"
    assert meta["used_generic_fallback"] is False


def test_extract_reviews_selector_miss_uses_generic_when_disabled() -> None:
    scraper = _scraper(require_source_selectors=False)
    html = """
    <html><body>
      <p>
        The apartment was spotless and very quiet at night. We had excellent sleep and smooth check-in.
      </p>
    </body></html>
    """
    reviews, meta = scraper._extract_reviews(
        html=html,
        source="google_maps",
        source_url="https://www.google.com/maps/search/example",
        max_count=3,
    )
    assert len(reviews) == 1
    assert meta["selector_miss"] is True
    assert meta["used_generic_fallback"] is True


def test_build_targets_google_maps_uses_property_name_only() -> None:
    scraper = _scraper()
    targets = scraper._build_targets(
        prompt="wifi",
        property_name="Your Beach Front Home",
        city="Los Angeles",
        region="los angels",
        source_urls=None,
    )
    google = next(t for t in targets if t["source"] == "google_maps")
    assert google["url"] == "https://www.google.com/maps/search/Your+Beach+Front+Home"
    assert "reviews" not in google["url"].lower()
    assert "los+angeles" not in google["url"].lower()


# ---------------------------------------------------------------------------
# _is_search_shell_url tests
# ---------------------------------------------------------------------------


def test_is_search_shell_url_google_maps_search() -> None:
    scraper = _scraper()
    assert scraper._is_search_shell_url(source="google_maps", url="https://www.google.com/maps/search/Hotel+reviews") is True


def test_is_search_shell_url_google_maps_place() -> None:
    scraper = _scraper()
    assert scraper._is_search_shell_url(source="google_maps", url="https://www.google.com/maps/place/Hotel+ABC") is False


def test_is_search_shell_url_tripadvisor_search() -> None:
    scraper = _scraper()
    assert scraper._is_search_shell_url(source="tripadvisor", url="https://www.tripadvisor.com/Search?q=hotel") is True


def test_is_search_shell_url_tripadvisor_detail() -> None:
    scraper = _scraper()
    assert scraper._is_search_shell_url(source="tripadvisor", url="https://www.tripadvisor.com/Hotel_Review-g123-d456") is False


# ---------------------------------------------------------------------------
# _navigate_to_detail_page diagnostics shape (no real browser — unsupported source)
# ---------------------------------------------------------------------------


def test_navigate_to_detail_page_unknown_source_returns_error() -> None:
    scraper = _scraper()

    class _FakePage:
        url = "https://example.com"

    diag = scraper._navigate_to_detail_page(
        page=_FakePage(),
        source="unknown_source",
        property_name="Test Hotel",
    )
    assert diag["navigated"] is False
    assert diag["error"] == "no_click_strategy_for_source:unknown_source"
    assert diag["detail_url"] is None
    assert diag["navigation_method"] is None


class _FakeElement:
    def __init__(self, *, text: str = "", attrs: dict[str, str] | None = None, visible: bool = True, on_click=None) -> None:
        self._text = text
        self._attrs = attrs or {}
        self._visible = visible
        self._on_click = on_click

    def click(self) -> None:
        if self._on_click:
            self._on_click()

    def inner_text(self) -> str:
        return self._text

    def get_attribute(self, name: str) -> str | None:
        return self._attrs.get(name)

    def is_visible(self, timeout: int | None = None) -> bool:
        return self._visible


class _FakeLocatorGroup:
    def __init__(self, elements: list[object] | None = None) -> None:
        self._elements = elements or []

    @property
    def first(self) -> object:
        if self._elements:
            return self._elements[0]
        return _FakeElement(visible=False)

    def all(self) -> list[object]:
        return list(self._elements)


class _FakeMouse:
    def wheel(self, dx: int, dy: int) -> None:
        return None


class _FakeContainer:
    def __init__(self, selector_map: dict[str, _FakeElement]) -> None:
        self._selector_map = selector_map

    def query_selector(self, selector: str):
        return self._selector_map.get(selector)


class _FakePageForNavigation:
    def __init__(self, selector_to_element: dict[str, _FakeElement] | None = None) -> None:
        self.url = "https://www.google.com/maps/search/example"
        self._selector_to_element = selector_to_element or {}

    def wait_for_selector(self, selector: str, timeout: int) -> None:
        if selector not in self._selector_to_element:
            raise RuntimeError("selector not found")

    def query_selector(self, selector: str):
        return self._selector_to_element.get(selector)

    def wait_for_url(self, pattern, timeout: int) -> None:
        if "/maps/place/" not in self.url:
            raise RuntimeError("not on place url")

    def wait_for_load_state(self, state: str, timeout: int) -> None:
        return None

    def wait_for_timeout(self, timeout: int) -> None:
        return None

    def locator(self, *args, **kwargs):
        return _FakeLocatorGroup([])


class _FakeCookiePage:
    def get_by_role(self, role: str, name: str):
        raise RuntimeError("button missing")


class _FakePageForLiveExtraction:
    def __init__(self, containers: list[_FakeContainer]) -> None:
        self.mouse = _FakeMouse()
        self._containers = containers

    def wait_for_selector(self, selector: str, timeout: int) -> None:
        raise RuntimeError("not found")

    def query_selector(self, selector: str):
        return None

    def query_selector_all(self, selector: str):
        if selector in {"div[data-review-id]", "div.jftiEf"}:
            return self._containers
        return []

    def wait_for_timeout(self, timeout: int) -> None:
        return None


def test_handle_cookie_consent_no_button_is_safe() -> None:
    scraper = _scraper()
    scraper._handle_cookie_consent(_FakeCookiePage())


def test_navigate_google_maps_diagnostics_on_miss() -> None:
    scraper = _scraper()
    diag = scraper._navigate_to_detail_page(
        page=_FakePageForNavigation(selector_to_element={}),
        source="google_maps",
        property_name="Hotel",
    )
    assert diag["navigated"] is False
    assert diag["navigation_attempts"] >= 1
    assert diag["navigation_error_code"] == "no_google_maps_result_element_found"
    assert isinstance(diag["navigation_elapsed_ms"], int)


def test_navigate_google_maps_diagnostics_on_success() -> None:
    scraper = _scraper()
    page = _FakePageForNavigation()
    page._selector_to_element["a[href*=\"/maps/place/\"]"] = _FakeElement(
        on_click=lambda: setattr(page, "url", "https://www.google.com/maps/place/hotel"),
    )
    diag = scraper._navigate_to_detail_page(
        page=page,
        source="google_maps",
        property_name="Hotel",
    )
    assert diag["navigated"] is True
    assert diag["navigation_method"] == "place_href_anchor"
    assert diag["navigation_error_code"] is None
    assert diag["detail_url"] == "https://www.google.com/maps/place/hotel"


def test_extract_google_maps_reviews_from_live_containers() -> None:
    scraper = _scraper()
    container = _FakeContainer(
        {
            "span.wiI7pd": _FakeElement(
                text=(
                    "Great stay overall. The room was spotless, wifi was stable all week, "
                    "and check-in was very smooth."
                )
            ),
            "span[aria-label*=\"star\"]": _FakeElement(attrs={"aria-label": "5 stars"}),
            "span.rsqaWe": _FakeElement(text="2 weeks ago"),
            "div.d4r55": _FakeElement(text="Alice"),
        }
    )
    reviews, meta = scraper._extract_google_maps_reviews_from_page(
        page=_FakePageForLiveExtraction([container]),
        source_url="https://www.google.com/maps/place/hotel",
        max_count=5,
    )
    assert len(reviews) == 1
    assert reviews[0].reviewer_name == "Alice"
    assert reviews[0].review_date == "2 weeks ago"
    assert reviews[0].rating == pytest.approx(5.0)
    assert meta["container_count"] == 1
    assert meta["accepted_review_count"] == 1


def test_extract_google_maps_reviews_rejects_noise_even_in_container() -> None:
    scraper = _scraper()
    noisy = _FakeContainer(
        {
            "span.wiI7pd": _FakeElement(
                text=(
                    "Sign in to continue and accept cookies before using main menu controls. "
                    "Read more and filter reviews from navigation."
                )
            ),
        }
    )
    reviews, meta = scraper._extract_google_maps_reviews_from_page(
        page=_FakePageForLiveExtraction([noisy]),
        source_url="https://www.google.com/maps/place/hotel",
        max_count=5,
    )
    assert reviews == []
    assert meta["container_text_found"] == 1
    assert meta["accepted_review_count"] == 0
    assert meta["rejected_review_count"] == 1


def test_looks_like_review_rejects_no_punctuation() -> None:
    scraper = _scraper()
    text = (
        "Great location amazing host perfect cleanliness five stars "
        "highly recommended excellent value for money"
    )
    assert scraper._looks_like_review(text) is False


def test_looks_like_review_rejects_short_text() -> None:
    scraper = _scraper()
    assert scraper._looks_like_review("Nice place!") is False
