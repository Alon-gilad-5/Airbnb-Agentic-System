"""Playwright-based web review scraping service for fallback evidence collection."""

from __future__ import annotations

from dataclasses import dataclass
import random
import re
import time
from typing import Any
from urllib.parse import quote_plus

from bs4 import BeautifulSoup


@dataclass
class ScrapedReview:
    """Normalized scraped review payload used by the reviews agent."""

    source: str
    source_url: str
    review_text: str
    reviewer_name: str | None = None
    review_date: str | None = None
    rating: float | None = None


_NOISE_MARKERS = [
    "log in",
    "log out",
    "sign in",
    "sign up",
    "sign out",
    "create account",
    "forgot password",
    "reset password",
    "cookie",
    "cookies",
    "cookie policy",
    "accept cookies",
    "we use cookies",
    "cookie settings",
    "privacy",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "all rights reserved",
    "copyright",
    "enable javascript",
    "javascript",
    "map data",
    "main menu",
    "skip to content",
    "skip navigation",
    "breadcrumb",
    "back to top",
    "write a review",
    "add a review",
    "post a review",
    "read more",
    "show more",
    "load more",
    "see all reviews",
    "sort by",
    "filter by",
    "filter reviews",
    "download the app",
    "get the app",
    "powered by",
    "newsletter",
    "follow us",
]

_DEFAULT_GMAPS_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
]


class PlaywrightReviewScraper:
    """Best-effort review scraper with configurable source allowlist and review cap."""

    def __init__(
        self,
        *,
        enabled: bool,
        allowlist: list[str],
        default_max_reviews: int,
        timeout_seconds: int,
        require_source_selectors: bool = True,
        min_review_chars: int = 40,
        min_token_count: int = 8,
        reject_private_use_ratio: float = 0.10,
        navigation_click_timeout_ms: int | None = None,
        gmaps_locale: str = "en-US",
        gmaps_viewport_width: int = 1280,
        gmaps_viewport_height: int = 900,
        gmaps_user_agents: list[str] | None = None,
        gmaps_scroll_passes: int = 3,
        gmaps_scroll_pause_ms: int = 900,
        gmaps_nav_timeout_ms: int | None = None,
        headless: bool = True,
    ) -> None:
        self.enabled = enabled
        self.allowlist = set(x.strip().lower() for x in allowlist if x.strip())
        self.default_max_reviews = max(1, default_max_reviews)
        self.timeout_ms = max(5, timeout_seconds) * 1000
        self.require_source_selectors = require_source_selectors
        self.min_review_chars = max(10, min_review_chars)
        self.min_token_count = max(3, min_token_count)
        self.reject_private_use_ratio = max(0.0, min(1.0, reject_private_use_ratio))
        self.navigation_click_timeout_ms: int | None = (
            int(navigation_click_timeout_ms) if navigation_click_timeout_ms is not None else None
        )
        self.gmaps_locale = (gmaps_locale or "en-US").strip() or "en-US"
        self.gmaps_viewport_width = max(800, int(gmaps_viewport_width))
        self.gmaps_viewport_height = max(600, int(gmaps_viewport_height))
        cleaned_user_agents = [ua.strip() for ua in (gmaps_user_agents or []) if ua and ua.strip()]
        self.gmaps_user_agents = cleaned_user_agents or list(_DEFAULT_GMAPS_USER_AGENTS)
        self.gmaps_scroll_passes = max(1, int(gmaps_scroll_passes))
        self.gmaps_scroll_pause_ms = max(200, int(gmaps_scroll_pause_ms))
        self.gmaps_nav_timeout_ms: int | None = (
            int(gmaps_nav_timeout_ms) if gmaps_nav_timeout_ms is not None else None
        )
        self.headless = headless

    @property
    def is_available(self) -> bool:
        """Return True only when feature is enabled and Playwright is importable."""

        if not self.enabled:
            return False
        try:
            import playwright.sync_api  # noqa: F401
        except Exception:
            return False
        return True

    def scrape_reviews(
        self,
        *,
        prompt: str,
        property_name: str | None,
        city: str | None,
        region: str | None,
        source_urls: dict[str, str] | None,
        max_reviews: int | None,
    ) -> tuple[list[ScrapedReview], dict[str, Any]]:
        """Scrape reviews from configured sources with total-cap enforcement."""

        if not self.enabled:
            return [], {"status": "disabled", "reason": "SCRAPING_ENABLED is false"}

        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            return [], {"status": "unavailable", "reason": f"Playwright import failed: {exc}"}

        max_total = max(1, int(max_reviews or self.default_max_reviews))
        targets = self._build_targets(
            prompt=prompt,
            property_name=property_name,
            city=city,
            region=region,
            source_urls=source_urls,
        )
        if not targets:
            return [], {
                "status": "no_targets",
                "reason": (
                    "No allowed source targets were provided/built. "
                    "Provide source_urls or set ACTIVE_PROPERTY_NAME for search fallback."
                ),
            }

        reviews: list[ScrapedReview] = []
        errors: list[str] = []
        attempted_targets: list[dict[str, Any]] = []
        noise_rejection_stats: dict[str, dict[str, int]] = {}

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            try:
                for target in targets:
                    if len(reviews) >= max_total:
                        break
                    target_diag: dict[str, Any] = {
                        "source": target["source"],
                        "url": target["url"],
                        "is_search_shell": self._is_search_shell_url(
                            source=target["source"],
                            url=target["url"],
                        ),
                    }
                    context = None
                    page = None
                    try:
                        context = browser.new_context(**self._context_kwargs_for_source(target["source"]))
                        page = context.new_page()
                        page.goto(target["url"], wait_until="domcontentloaded", timeout=self.timeout_ms)
                        if target["source"] == "google_maps":
                            self._handle_cookie_consent(page)
                        # Give dynamic pages a brief chance to finish initial hydration.
                        try:
                            page.wait_for_load_state("networkidle", timeout=min(6000, self.timeout_ms))
                        except Exception:
                            pass

                        # If this is a search-shell page, try to click through to the
                        # actual property detail page before attempting extraction.
                        if self._is_search_shell_url(source=target["source"], url=target["url"]):
                            nav_diag = self._navigate_to_detail_page(
                                page=page,
                                source=target["source"],
                                property_name=property_name,
                            )
                            target_diag["navigation"] = nav_diag
                            if nav_diag["navigated"]:
                                target["url"] = nav_diag["detail_url"]
                            else:
                                target_diag["status"] = "navigation_failed"
                                attempted_targets.append(target_diag)
                                continue

                        if target["source"] == "google_maps":
                            extracted, extract_meta = self._extract_google_maps_reviews_from_page(
                                page=page,
                                source_url=target["url"],
                                max_count=max_total - len(reviews),
                            )
                            # Fallback to static HTML extraction when live DOM path yields nothing.
                            if not extracted:
                                html = page.content()
                                fallback_extracted, fallback_meta = self._extract_reviews(
                                    html=html,
                                    source=target["source"],
                                    source_url=target["url"],
                                    max_count=max_total - len(reviews),
                                )
                                extracted = fallback_extracted
                                extract_meta["html_fallback"] = fallback_meta
                                extract_meta["html_fallback_extracted_count"] = len(fallback_extracted)
                                extract_meta["selector_miss"] = fallback_meta.get("selector_miss", True)
                                extract_meta["selector_miss_reason"] = fallback_meta.get("selector_miss_reason")
                        else:
                            # Prefer source-specific selectors; if none appear we still parse HTML once.
                            for selector in self._selectors_for_source(target["source"]):
                                try:
                                    page.wait_for_selector(selector, timeout=1000)
                                    target_diag["selector_hint_hit"] = selector
                                    break
                                except Exception:
                                    continue
                            # Trigger lazy-loaded review sections on dynamic pages.
                            for _ in range(3):
                                page.mouse.wheel(0, 4000)
                                page.wait_for_timeout(500)
                            html = page.content()
                            extracted, extract_meta = self._extract_reviews(
                                html=html,
                                source=target["source"],
                                source_url=target["url"],
                                max_count=max_total - len(reviews),
                            )

                        target_diag.update(extract_meta)
                        target_diag["extracted_count"] = len(extracted)
                        stats = extract_meta.get("noise_rejection_stats")
                        if isinstance(stats, dict):
                            src_stats = noise_rejection_stats.setdefault(
                                target["source"],
                                {"examined": 0, "accepted": 0, "rejected": 0},
                            )
                            src_stats["examined"] += int(stats.get("examined", 0))
                            src_stats["accepted"] += int(stats.get("accepted", 0))
                            src_stats["rejected"] += int(stats.get("rejected", 0))
                        reviews.extend(extracted)
                    except Exception as exc:
                        errors.append(f"{target['source']}::{target['url']}::{type(exc).__name__}: {exc}")
                        target_diag["status"] = "error"
                        target_diag["error"] = f"{type(exc).__name__}: {exc}"
                    finally:
                        if page is not None:
                            try:
                                page.close()
                            except Exception:
                                pass
                        if context is not None:
                            try:
                                context.close()
                            except Exception:
                                pass
                    attempted_targets.append(target_diag)
            finally:
                browser.close()

        # Deduplicate by normalized review text to avoid repeated content from dynamic rendering.
        deduped: list[ScrapedReview] = []
        seen = set()
        for review in reviews:
            normalized = self._normalize_text(review.review_text)
            if normalized and normalized not in seen:
                seen.add(normalized)
                deduped.append(review)
            if len(deduped) >= max_total:
                break

        return deduped, {
            "status": "ok",
            "attempted_targets": attempted_targets,
            "errors": errors,
            "max_total": max_total,
            "raw_count": len(reviews),
            "deduped_count": len(deduped),
            "noise_rejection_stats": noise_rejection_stats,
            "require_source_selectors": self.require_source_selectors,
        }

    def _build_targets(
        self,
        *,
        prompt: str,
        property_name: str | None,
        city: str | None,
        region: str | None,
        source_urls: dict[str, str] | None,
    ) -> list[dict[str, str]]:
        """Build ordered source targets from known URLs, then search URLs as fallback."""

        targets: list[dict[str, str]] = []
        if source_urls:
            for source, url in source_urls.items():
                key = source.strip().lower()
                if key in self.allowlist and url:
                    targets.append({"source": key, "url": url})

        if targets:
            return targets

        # Search fallback should use stable entity context, not full user question.
        # Without a known property name/url, scraping usually yields noisy/non-actionable pages.
        if not property_name:
            return targets

        if "google_maps" in self.allowlist:
            gmaps_query = property_name.strip()
            if gmaps_query:
                encoded_gmaps = quote_plus(gmaps_query)
                targets.append(
                    {
                        "source": "google_maps",
                        "url": f"https://www.google.com/maps/search/{encoded_gmaps}",
                    }
                )
        if "tripadvisor" in self.allowlist:
            query_parts = [x for x in [property_name, city, region] if x]
            query = " ".join(query_parts).strip()
            if not query:
                return targets
            encoded = quote_plus(query + " reviews")
            targets.append(
                {
                    "source": "tripadvisor",
                    "url": f"https://www.tripadvisor.com/Search?q={encoded}",
                }
            )
        return targets

    def _extract_reviews(
        self,
        *,
        html: str,
        source: str,
        source_url: str,
        max_count: int,
    ) -> tuple[list[ScrapedReview], dict[str, Any]]:
        """Extract likely review snippets with source-first selectors."""

        soup = BeautifulSoup(html, "html.parser")
        selectors = self._selectors_for_source(source)

        texts: list[str] = []
        examined = 0
        accepted = 0
        source_specific_hits = 0
        for selector in selectors:
            for node in soup.select(selector):
                source_specific_hits += 1
                text = self._normalize_text(node.get_text(" ", strip=True))
                if not text:
                    continue
                examined += 1
                if self._looks_like_review(text):
                    texts.append(text)
                    accepted += 1

        selector_miss_reason: str | None = None
        used_generic_fallback = False
        if source_specific_hits == 0:
            if selectors:
                selector_miss_reason = "no_source_specific_selector_hit"
            else:
                selector_miss_reason = "no_source_specific_selectors_configured"
        if source_specific_hits == 0 and not self.require_source_selectors:
            used_generic_fallback = True
            for node in soup.select("article, p, div, span"):
                text = self._normalize_text(node.get_text(" ", strip=True))
                if not text:
                    continue
                examined += 1
                if self._looks_like_review(text):
                    texts.append(text)
                    accepted += 1

        out: list[ScrapedReview] = []
        seen = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            out.append(ScrapedReview(source=source, source_url=source_url, review_text=text))
            if len(out) >= max_count:
                break
        return out, {
            "status": "ok",
            "selector_miss": source_specific_hits == 0,
            "selector_miss_reason": selector_miss_reason,
            "used_generic_fallback": used_generic_fallback,
            "noise_rejection_stats": {
                "examined": examined,
                "accepted": accepted,
                "rejected": max(0, examined - accepted),
            },
        }

    def _context_kwargs_for_source(self, source: str) -> dict[str, Any]:
        """Build browser context kwargs for source-specific stability settings."""

        if source != "google_maps":
            return {}
        return {
            "viewport": {"width": self.gmaps_viewport_width, "height": self.gmaps_viewport_height},
            "locale": self.gmaps_locale,
            "user_agent": random.choice(self.gmaps_user_agents),
        }

    def _handle_cookie_consent(self, page: Any) -> None:
        """Best-effort cookie consent handling for regions that gate content."""

        for text in ["Reject all", "Accept all", "I agree", "Agree", "Accept"]:
            try:
                button = page.get_by_role("button", name=text)
                if button.is_visible(timeout=1200):
                    button.click()
                    page.wait_for_timeout(400)
                    return
            except Exception:
                continue

    def _open_google_maps_reviews_tab(self, page: Any) -> None:
        """Best-effort click on Reviews tab to expose review containers."""

        selectors = [
            'button[aria-label*="Reviews"]',
            'button[aria-label*="reviews"]',
            "button:has-text('Reviews')",
            "[role='tab']:has-text('Reviews')",
        ]
        for selector in selectors:
            try:
                page.wait_for_selector(selector, timeout=1200)
                tab = page.query_selector(selector)
                if tab is None:
                    continue
                tab.click()
                page.wait_for_timeout(900)
                return
            except Exception:
                continue

    def _extract_google_maps_reviews_from_page(
        self,
        *,
        page: Any,
        source_url: str,
        max_count: int,
    ) -> tuple[list[ScrapedReview], dict[str, Any]]:
        """Extract reviews from live Google Maps review containers."""

        self._open_google_maps_reviews_tab(page)
        for _ in range(self.gmaps_scroll_passes):
            try:
                page.mouse.wheel(0, 850)
                page.wait_for_timeout(self.gmaps_scroll_pause_ms)
            except Exception:
                break

        containers = page.query_selector_all("div[data-review-id]") or []
        container_selector = "div[data-review-id]"
        if not containers:
            containers = page.query_selector_all("div.jftiEf") or []
            container_selector = "div.jftiEf"

        out: list[ScrapedReview] = []
        seen_texts: set[str] = set()
        container_text_found = 0
        rejected = 0

        for container in containers:
            text = self._read_text_from_node(container, "span.wiI7pd")
            if not text:
                text = self._read_text_from_node(container, "div.MyEned")
            text = self._normalize_text(text)
            if not text:
                continue
            container_text_found += 1
            if not self._looks_like_review(text):
                rejected += 1
                continue
            if text in seen_texts:
                continue

            seen_texts.add(text)
            out.append(
                ScrapedReview(
                    source="google_maps",
                    source_url=source_url,
                    review_text=text,
                    reviewer_name=self._read_text_from_node(container, "div.d4r55") or None,
                    review_date=self._read_text_from_node(container, "span.rsqaWe") or None,
                    rating=self._extract_rating_from_node(container),
                )
            )
            if len(out) >= max_count:
                break

        examined = container_text_found
        accepted = len(out)
        selector_miss = len(containers) == 0
        return out, {
            "status": "ok",
            "source_extractor": "google_maps_live_dom",
            "selector_miss": selector_miss,
            "selector_miss_reason": (
                "no_google_maps_review_containers_found" if selector_miss else None
            ),
            "used_generic_fallback": False,
            "container_selector": container_selector,
            "container_count": len(containers),
            "container_text_found": container_text_found,
            "accepted_review_count": accepted,
            "rejected_review_count": rejected,
            "noise_rejection_stats": {
                "examined": examined,
                "accepted": accepted,
                "rejected": max(0, examined - accepted),
            },
        }

    def _read_text_from_node(self, node: Any, selector: str) -> str:
        """Read inner text from node.query_selector(selector), best-effort."""

        try:
            elem = node.query_selector(selector)
            if elem is None:
                return ""
            return self._normalize_text(elem.inner_text())
        except Exception:
            return ""

    def _extract_rating_from_node(self, node: Any) -> float | None:
        """Extract star rating from aria-label text like '5 stars'."""

        try:
            elem = node.query_selector('span[aria-label*="star"]')
            if elem is None:
                return None
            aria = elem.get_attribute("aria-label") or ""
            match = re.search(r"(\d+(?:\.\d+)?)", aria)
            if not match:
                return None
            return float(match.group(1))
        except Exception:
            return None

    def _looks_like_review(self, text: str) -> bool:
        """Heuristic filter to keep likely review paragraphs and drop UI noise."""

        if len(text) < self.min_review_chars:
            return False
        tokens = text.split()
        if len(tokens) < self.min_token_count:
            return False
        if len(set(token.lower() for token in tokens)) < max(4, int(self.min_token_count * 0.5)):
            return False
        if not any(ch in text for ch in ".!?"):
            return False
        # Reject low-information strings dominated by symbols/digits.
        letter_count = sum(1 for ch in text if ch.isalpha())
        if letter_count < 20:
            return False
        ratio = letter_count / max(1, len(text))
        if ratio < 0.35:
            return False
        private_use_count = sum(1 for ch in text if "\ue000" <= ch <= "\uf8ff")
        private_use_ratio = private_use_count / max(1, len(text))
        if private_use_ratio > self.reject_private_use_ratio:
            return False
        lowered = text.lower()
        if any(marker in lowered for marker in _NOISE_MARKERS):
            return False
        if self._has_repeated_nav_patterns(lowered):
            return False
        return True

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and collapse noisy line breaks."""

        return re.sub(r"\s+", " ", text or "").strip()

    def _selectors_for_source(self, source: str) -> list[str]:
        """Return source-specific selectors in descending preference order."""

        selectors_by_source = {
            "google_maps": [
                "span.wiI7pd",
                "div.MyEned",
                "div.jftiEf span",
            ],
            "tripadvisor": [
                "div[data-test-target='review-body'] span",
                "div[data-test-target='review-body']",
                "span[class*='JguWG']",
            ],
        }
        return selectors_by_source.get(source, [])

    def _navigate_to_detail_page(
        self,
        *,
        page: Any,
        source: str,
        property_name: str | None,
    ) -> dict[str, Any]:
        """Click from a search-results page to the property detail page.

        Returns a diagnostic dict — never raises.
        """

        diag: dict[str, Any] = {
            "navigated": False,
            "detail_url": None,
            "navigation_method": None,
            "navigation_attempts": 0,
            "navigation_elapsed_ms": 0,
            "navigation_error_code": None,
            "error": None,
        }
        if source == "google_maps":
            click_timeout = self.gmaps_nav_timeout_ms or min(20_000, self.timeout_ms // 2)
        else:
            click_timeout = self.navigation_click_timeout_ms or min(15_000, self.timeout_ms // 2)

        started = time.perf_counter()
        try:
            if source == "google_maps":
                diag = self._navigate_google_maps(page, click_timeout)
            elif source == "tripadvisor":
                diag = self._navigate_tripadvisor(page, click_timeout)
            else:
                diag["error"] = f"no_click_strategy_for_source:{source}"
                diag["navigation_error_code"] = "no_click_strategy_for_source"
        except Exception as exc:
            diag["error"] = f"{type(exc).__name__}: {exc}"
            diag["navigation_error_code"] = "navigation_exception"
        diag.setdefault("navigation_attempts", 0)
        diag.setdefault("navigation_error_code", None)
        diag["navigation_elapsed_ms"] = int((time.perf_counter() - started) * 1000)
        return diag

    def _navigate_google_maps(
        self, page: Any, click_timeout: int,
    ) -> dict[str, Any]:
        """Google Maps: /maps/search/ → /maps/place/ click-through."""

        diag: dict[str, Any] = {
            "navigated": False,
            "detail_url": None,
            "navigation_method": None,
            "navigation_attempts": 0,
            "navigation_error_code": None,
            "error": None,
        }
        selectors = [
            ("place_href_anchor", 'a[href*="/maps/place/"]'),
            ("hfpxzc_anchor", "a.hfpxzc"),
            ("Nv2PK_card", "div.Nv2PK"),
        ]

        for method_name, sel in selectors:
            diag["navigation_attempts"] += 1
            try:
                try:
                    page.wait_for_selector(sel, timeout=min(4000, click_timeout))
                except Exception:
                    pass
                element = page.query_selector(sel)
                if element is None:
                    continue
                element.click()
                page.wait_for_url("**/maps/place/**", timeout=click_timeout)
                diag["navigation_method"] = method_name
                diag["navigated"] = True
                diag["detail_url"] = page.url
                break
            except Exception:
                continue

        if not diag["navigated"]:
            diag["navigation_error_code"] = "no_google_maps_result_element_found"
            diag["error"] = "no_google_maps_result_element_found"
            return diag

        # Brief settle for the detail panel.
        try:
            page.wait_for_load_state("networkidle", timeout=min(4000, click_timeout))
        except Exception:
            pass

        # Best-effort: click the Reviews tab if present.
        self._open_google_maps_reviews_tab(page)

        return diag

    def _navigate_tripadvisor(
        self, page: Any, click_timeout: int,
    ) -> dict[str, Any]:
        """TripAdvisor: /Search?q= → /Hotel_Review or /VacationRentalReview click-through."""

        diag: dict[str, Any] = {
            "navigated": False,
            "detail_url": None,
            "navigation_method": None,
            "error": None,
        }
        selectors = [
            ("hotel_review_anchor", "a[href*='Hotel_Review']"),
            ("vacation_rental_anchor", "a[href*='VacationRentalReview']"),
            ("attraction_review_anchor", "a[href*='Attraction_Review']"),
        ]

        for method_name, sel in selectors:
            try:
                element = page.query_selector(sel)
                if element is None:
                    continue
                element.click()
                page.wait_for_url(
                    re.compile(r"/Hotel_Review|/VacationRentalReview|/Attraction_Review"),
                    timeout=click_timeout,
                )
                diag["navigation_method"] = method_name
                diag["navigated"] = True
                diag["detail_url"] = page.url
                break
            except Exception:
                continue

        if not diag["navigated"]:
            diag["error"] = "no_tripadvisor_result_element_found"
            return diag

        # Give lazy review fragments time to load.
        try:
            page.wait_for_load_state("networkidle", timeout=min(4000, click_timeout))
        except Exception:
            pass

        return diag

    def _is_search_shell_url(self, *, source: str, url: str) -> bool:
        """Detect likely search shell pages where review text is often absent."""

        lowered = url.strip().lower()
        if source == "google_maps":
            return "/maps/search/" in lowered
        if source == "tripadvisor":
            return "/search?" in lowered
        return False

    def _has_repeated_nav_patterns(self, lowered_text: str) -> bool:
        """Reject text that looks like concatenated navigation/action labels."""

        nav_markers = ["menu", "filter", "sort", "share", "report", "helpful", "read more"]
        hits = sum(1 for marker in nav_markers if marker in lowered_text)
        return hits >= 3
