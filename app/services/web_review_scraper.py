"""Playwright-based web review scraping service for fallback evidence collection."""

from __future__ import annotations

from dataclasses import dataclass
import re
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


class PlaywrightReviewScraper:
    """Best-effort review scraper with configurable source allowlist and review cap."""

    def __init__(
        self,
        *,
        enabled: bool,
        allowlist: list[str],
        default_max_reviews: int,
        timeout_seconds: int,
        headless: bool = True,
    ) -> None:
        self.enabled = enabled
        self.allowlist = set(x.strip().lower() for x in allowlist if x.strip())
        self.default_max_reviews = max(1, default_max_reviews)
        self.timeout_ms = max(5, timeout_seconds) * 1000
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

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=self.headless)
            try:
                context = browser.new_context()
                for target in targets:
                    if len(reviews) >= max_total:
                        break
                    try:
                        page = context.new_page()
                        page.goto(target["url"], wait_until="domcontentloaded", timeout=self.timeout_ms)
                        # Trigger lazy-loaded review sections on dynamic pages.
                        for _ in range(3):
                            page.mouse.wheel(0, 4000)
                            page.wait_for_timeout(500)
                        html = page.content()
                        page.close()

                        extracted = self._extract_reviews(
                            html=html,
                            source=target["source"],
                            source_url=target["url"],
                            max_count=max_total - len(reviews),
                        )
                        reviews.extend(extracted)
                    except Exception as exc:
                        errors.append(f"{target['source']}::{target['url']}::{type(exc).__name__}: {exc}")
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
            "attempted_targets": targets,
            "errors": errors,
            "max_total": max_total,
            "raw_count": len(reviews),
            "deduped_count": len(deduped),
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

        query_parts = [x for x in [property_name, city, region] if x]
        query = " ".join(query_parts).strip()
        if not query:
            return targets
        encoded = quote_plus(query + " reviews")

        if "google_maps" in self.allowlist:
            targets.append(
                {
                    "source": "google_maps",
                    "url": f"https://www.google.com/maps/search/{encoded}",
                }
            )
        if "tripadvisor" in self.allowlist:
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
    ) -> list[ScrapedReview]:
        """Extract likely review snippets from source-specific selectors + generic fallback."""

        soup = BeautifulSoup(html, "html.parser")
        selectors_by_source = {
            "google_maps": [
                "span.wiI7pd",  # Common Maps review body selector
                "div.MyEned",
                "div.jftiEf span",
            ],
            "tripadvisor": [
                "div[data-test-target='review-body'] span",
                "div[data-test-target='review-body']",
                "span[class*='JguWG']",
            ],
        }
        selectors = selectors_by_source.get(source, [])

        texts: list[str] = []
        for selector in selectors:
            for node in soup.select(selector):
                text = self._normalize_text(node.get_text(" ", strip=True))
                if self._looks_like_review(text):
                    texts.append(text)

        if not texts:
            # Generic fallback extraction for when source-specific selectors fail.
            for node in soup.select("article, p, div, span"):
                text = self._normalize_text(node.get_text(" ", strip=True))
                if self._looks_like_review(text):
                    texts.append(text)

        out: list[ScrapedReview] = []
        seen = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            out.append(ScrapedReview(source=source, source_url=source_url, review_text=text))
            if len(out) >= max_count:
                break
        return out

    def _looks_like_review(self, text: str) -> bool:
        """Heuristic filter to keep likely review paragraphs and drop UI noise."""

        if len(text) < 40:
            return False
        if len(text.split()) < 8:
            return False
        # Reject low-information strings dominated by symbols/digits.
        letter_count = sum(1 for ch in text if ch.isalpha())
        if letter_count < 20:
            return False
        ratio = letter_count / max(1, len(text))
        if ratio < 0.35:
            return False
        noise_markers = ["cookie", "sign in", "privacy", "terms", "javascript", "map data"]
        lowered = text.lower()
        if any(marker in lowered for marker in noise_markers):
            return False
        return True

    def _normalize_text(self, text: str) -> str:
        """Normalize whitespace and collapse noisy line breaks."""

        return re.sub(r"\s+", " ", text or "").strip()
