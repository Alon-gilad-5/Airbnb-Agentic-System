# Scraping Roadmap

## Execution Order

1. **Plan 1 first** — merge and verify acceptance criteria
2. **Plan 2 second** — depends on clean Playwright fallback from Plan 1

## Decisions log

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Integration target is **Bright Data Web Scraper API** (SERP endpoint), not MCP protocol. All naming uses `BRIGHTDATA_*` prefix, no `_MCP_`. | Avoids conflation with Model Context Protocol; matches actual HTTP/JSON integration. |
| D2 | Config prefix is `BRIGHTDATA_*` everywhere (config.py, main.py, docs, smoke script). | Single convention, no drift. |
| D3 | Region canonicalization: extract shared `_canonicalize_region()` → `strip().lower()`, use in both ingest and filter, never store `"unknown"`. | Current code: ingest writes `"unknown"` on None; filter never emits `"unknown"` → silent mismatch. Fix: skip region from filter/metadata when absent instead of writing sentinel. |
| D4 | Upsert relevance gating uses positional index (enumerate), not review_text as dict key. | Duplicate review texts can collide in a text→score dict. Zip scored list with scraped list by index. |
| D5 | Test framework: **pytest**. Add `pytest` + `pytest-asyncio` to dev dependencies. Layout: `tests/services/`, `tests/agents/`. | Repo currently has zero tests. pytest is standard, supports fixtures and parametrize. |
| D6 | In-memory circuit breaker and daily cap are **per-process, best-effort** on serverless. Documented as a known limitation. | Vercel serverless may run multiple cold-start instances. True distributed state would require Redis/external store — out of scope for Plan 2; acceptable because Playwright fallback is safe. |

---

## Plan 1: Stabilize Current Playwright Scraper

### Goal
Reduce UI/noise capture to near-zero and ensure zero-relevance content never reaches the quarantine namespace, without changing API response contracts.

### Root cause analysis

1. **Generic DOM fallback is the primary noise amplifier.** In `_extract_reviews` (lines 224-229), when source-specific selectors miss, the code falls through to `soup.select("article, p, div, span")` — harvesting nav bars, footers, cookie banners, page titles. This fires frequently because `domcontentloaded` fires before JS-rendered review content populates.

2. **`_looks_like_review` noise markers list covers only 6 patterns.** The current list is `["cookie", "sign in", "privacy", "terms", "javascript", "map data"]`. Dozens of other UI patterns pass all numeric checks (40 chars, 8 words, 35% letter ratio).

3. **No lexical relevance gate before quarantine upsert.** Every `ScrapedReview` that exits `_extract_reviews` is upserted unconditionally. The `_score_scraped_relevance` function scores each review, but scores are only used for live-response evidence selection — never consulted during upsert.

4. **Metadata filter has no `property_id` dimension.** `_build_metadata_filter` only emits a `region` filter. Multi-property indexes mix results, inflating `top_score` with semi-relevant cross-property matches.

5. **Private-use unicode and non-sentence text pass all checks.** Google Maps font-substituted icon strings contain chars in U+E000–U+F8FF. Rating labels and address fragments pass the letter-ratio check despite containing no sentences.

6. **Region canonicalization mismatch.** Ingest (`web_review_ingest.py:74`) writes `(region or "unknown").lower()` into vector metadata. Filter (`reviews_agent.py:238`) uses `str(context["region"]).lower()` — never emits `"unknown"`. Vectors ingested with `region="unknown"` are unreachable by any filter query.

7. **Search-shell targeting yields zero reviews.** `_build_targets` Phase 2 generates search URLs (`/maps/search/`, `/Search?q=`) not property detail pages. Source-specific review selectors never match on search result pages. With `require_source_selectors=True`, the result is zero reviews (no noise, but also no data).

### Code locations to change
- `app/services/web_review_scraper.py`
  - `_navigate_to_detail_page` (NEW) — click-through from search to detail page
  - `_navigate_google_maps`, `_navigate_tripadvisor` (NEW) — source-specific strategies
  - `scrape_reviews` — insert navigation call before selector-wait loop
  - `_extract_reviews` — remove generic fallback, add noise rejection stats, add `selector_miss` reason code
  - `_looks_like_review` — expand noise markers, add unicode/punctuation guards
- `app/agents/reviews_agent.py`
  - `_build_metadata_filter` — add `property_id` compound filter, use shared region canonicalization
  - `_scrape_fallback` — add relevance gate before upsert (by index, not text key)
- `app/services/web_review_ingest.py`
  - Region handling — use shared canonicalization, skip `"unknown"` sentinel
- `app/services/region_utils.py` — shared `canonicalize_region()` helper
- `app/config.py` — add config knobs including `SCRAPING_NAVIGATION_CLICK_TIMEOUT_MS`

### Implementation steps (ordered by impact)

#### Step 0 — Add click-through navigation from search to detail pages (CRITICAL)
**File:** `app/services/web_review_scraper.py`

**Problem:** Without direct URLs, the scraper hits search result pages where review selectors never match.

**Solution:** New `_navigate_to_detail_page` method called in `scrape_reviews` when `_is_search_shell_url` returns `True`.

**Google Maps** (`/maps/search/` → `/maps/place/`):
1. Try selectors: `a.hfpxzc`, `a[href*="/maps/place/"]`, `div.Nv2PK`
2. Click first match, wait for URL to contain `/maps/place/`
3. Brief `networkidle` settle (4s cap)
4. Best-effort Reviews tab click

**TripAdvisor** (`/Search?q=` → `/Hotel_Review-` or `/VacationRentalReview-`):
1. Try selectors: `a[href*="Hotel_Review"]`, `a[href*="VacationRentalReview"]`, `a[href*="Attraction_Review"]`
2. Click first match, wait for URL pattern match
3. Brief `networkidle` settle (4s cap)

**On success:** patches `target["url"]` to detail URL so `ScrapedReview.source_url` reflects the real page.
**On failure:** logs `navigation_failed` with error reason, skips target cleanly (no extraction attempt).
**Timeout:** `min(15_000, self.timeout_ms // 2)`, overridable via `SCRAPING_NAVIGATION_CLICK_TIMEOUT_MS`.

#### Step 1 — Remove generic DOM fallback (QUICK WIN, highest impact)
**File:** `app/services/web_review_scraper.py` → `_extract_reviews` (lines 224-229)

When source-specific selectors (`span.wiI7pd` etc. for google_maps, `div[data-test-target='review-body']` etc. for tripadvisor) find nothing, return `[]` instead of falling through to `soup.select("article, p, div, span")`.

Add `selector_miss=True` and `selector_miss_reason="no_source_specific_selector_hit"` to per-target diagnostics in `scraper_meta["attempted_targets"]`. This surfaces in the `reviews_agent.web_scrape` step response as an explicit reason code.

Add env escape hatch `SCRAPING_REQUIRE_SOURCE_SELECTORS` (default `True`); set to `False` to re-enable generic fallback for debugging.

#### Step 2 — Expand noise markers (QUICK WIN)
**File:** `app/services/web_review_scraper.py` → `_looks_like_review` (line 256)

Replace the 6-entry list with ~42 entries:

```python
_NOISE_MARKERS = [
    # Auth / account UI
    "log in", "log out", "sign in", "sign up", "sign out",
    "create account", "forgot password", "reset password",
    # Legal / consent
    "cookie", "cookies", "cookie policy", "accept cookies",
    "we use cookies", "cookie settings", "privacy", "privacy policy",
    "terms of service", "terms and conditions", "all rights reserved",
    "copyright", "©",
    # Technical
    "javascript", "enable javascript", "map data",
    # Navigation / chrome
    "navigation", "main menu", "skip to content", "skip navigation",
    "breadcrumb", "back to top", "scroll to top",
    # Action / CTA buttons
    "write a review", "add a review", "post a review",
    "read more", "show more", "load more", "see all reviews",
    "sort by", "filter by", "filter reviews", "helpful",
    "share", "report", "translate", "original language",
    # App promotion
    "download the app", "get the app", "powered by",
    # Footer
    "subscribe", "newsletter", "follow us",
]
```

#### Step 3 — Add private-use unicode rejection
**File:** `app/services/web_review_scraper.py` → `_looks_like_review`

Count chars in U+E000–U+F8FF range. Reject if ratio > `SCRAPING_REJECT_PRIVATE_USE_RATIO` (default 0.10). Google Maps font-substituted icon strings hit this; real reviews never do.

```python
private_use_count = sum(1 for ch in text if "\ue000" <= ch <= "\uf8ff")
if private_use_count / max(1, len(text)) > self._reject_private_use_ratio:
    return False
```

#### Step 4 — Require sentence-ending punctuation
**File:** `app/services/web_review_scraper.py` → `_looks_like_review`

Reject text without any `. ! ?` character. Eliminates rating-label strings like `"4.7 stars 2,341 reviews Boutique Hotel Pacific Grove California"`.

```python
if not any(ch in text for ch in ".!?"):
    return False
```

#### Step 5 — Fix region canonicalization and add property_id filter
**Files:** `app/agents/reviews_agent.py` → `_build_metadata_filter`, `app/services/web_review_ingest.py`

**Region fix:** Extract a shared helper (or inline the same logic in both places):
```python
def _canonicalize_region(raw: str | None) -> str | None:
    """Return normalized region or None. Never returns sentinel like 'unknown'."""
    if not raw or not raw.strip():
        return None
    return raw.strip().lower()
```

In `web_review_ingest.py:74`, replace `(region or "unknown").lower()` — if canonicalized region is `None`, omit `region` key from metadata entirely (existing `_clean_metadata` already strips None values).

In `_build_metadata_filter`, use the same canonicalization. Add `property_id` via Pinecone `$and` compound filter:

```python
region = _canonicalize_region(self._context_str(context, "region"))
property_id = self._context_str(context, "property_id")

# ... (existing prompt-based region detection fallback) ...

if property_id and region:
    return {"$and": [
        {"property_id": {"$eq": property_id}},
        {"region": {"$eq": region}},
    ]}
if property_id:
    return {"property_id": {"$eq": property_id}}
if region:
    return {"region": {"$eq": region}}
return None
```

#### Step 6 — Gate quarantine upsert on lexical relevance (by index)
**File:** `app/agents/reviews_agent.py` → `_scrape_fallback` (lines 256-302)

Before calling `_upsert_scraped_reviews`, filter scraped reviews using positional pairing with scored converted matches. Gate at `score >= SCRAPING_MIN_LEXICAL_RELEVANCE_FOR_UPSERT` (default 0.15). Log `rejected_by_relevance_gate` count in step response.

```python
converted = self._convert_scraped_to_matches(
    scraped_reviews=scraped, prompt=prompt, context=context,
)
min_relevance = self.config.min_lexical_relevance_for_upsert

# Gate by positional index — avoids text-key collisions on duplicate review texts
reviews_for_upsert = [
    review for review, match in zip(scraped, converted)
    if match.score >= min_relevance
]
rejected_count = len(scraped) - len(reviews_for_upsert)
```

Note: `_convert_scraped_to_matches` returns items in the same order as the input `scraped` list (1-indexed `web:{source}:{idx}`), so `zip(scraped, converted)` is safe.

#### Step 7 — Add noise rejection logging
**File:** `app/services/web_review_scraper.py` → `_extract_reviews`

Track `candidates_examined` vs `candidates_accepted` per source. Emit as `noise_rejection_stats` in `scraper_meta`:

```python
"noise_rejection_stats": {
    source: {"examined": N, "accepted": M, "rejected": N - M}
}
```

### Config additions

| Env var | Default | Purpose |
|---------|---------|---------|
| `SCRAPING_REQUIRE_SOURCE_SELECTORS` | `True` | Disable generic fallback |
| `SCRAPING_MIN_LEXICAL_RELEVANCE_FOR_UPSERT` | `0.15` | Upsert quality gate |
| `SCRAPING_REJECT_PRIVATE_USE_RATIO` | `0.10` | Unicode noise threshold |
| `SCRAPING_NAVIGATION_CLICK_TIMEOUT_MS` | None (auto) | Override click-through navigation timeout |

No API schema changes (`ExecuteRequest`/`ExecuteResponse` unchanged).

### Test framework setup

**New dev dependency:** `pytest` (add to `requirements-dev.txt` or `pyproject.toml` `[project.optional-dependencies]`)

**Directory layout:**
```
tests/
  conftest.py           # shared fixtures (scraper instances, HTML fixtures, mock contexts)
  services/
    __init__.py
    test_web_review_scraper.py        # _looks_like_review, _extract_reviews
    test_web_review_ingest.py         # region canonicalization, metadata correctness
  agents/
    __init__.py
    test_reviews_agent.py             # filter build, relevance gate, regression
```

**Run:** `pytest tests/ -v`

### Tests

1. **`_looks_like_review` unit tests** (`tests/services/test_web_review_scraper.py`):
   - valid review text → passes
   - UI/menu text → rejected by noise markers
   - cookie banner text → rejected
   - private-use unicode strings (ratio > 0.10) → rejected
   - no-punctuation strings → rejected
   - short text (< 40 chars) → rejected

2. **Selector-hit/miss parser fixtures** (`tests/services/test_web_review_scraper.py`):
   - google_maps HTML with `span.wiI7pd` → extracts review
   - google_maps HTML without selectors → returns `[]` (not generic noise)
   - google_maps selector miss → `selector_miss_reason == "no_source_specific_selector_hit"` in meta
   - tripadvisor HTML without `data-test-target='review-body'` → returns `[]`
   - noise marker content in valid selectors → filtered out

3. **Region canonicalization** (`tests/services/test_web_review_ingest.py`):
   - `None` → `None` (not `"unknown"`)
   - `"  San Francisco  "` → `"san francisco"`
   - `""` → `None`

4. **Agent integration tests** (`tests/agents/test_reviews_agent.py`):
   - `property_id + region` → `$and` compound Pinecone filter
   - `property_id` only → `{"property_id": {...}}` filter
   - region only → `{"region": {...}}` filter
   - 3 scraped reviews (scores 0.0, 0.0, 0.3) → only score >= 0.15 upserted (by index)
   - all-zero-relevance scraped → upsert called with empty list
   - duplicate review texts with different scores → both independently gated (no dict-key collision)

5. **Regression** (`tests/agents/test_reviews_agent.py`):
   - step module name strings unchanged
   - no-evidence response exact string preserved

### Acceptance criteria
- Generic fallback disabled by default; selector-miss returns empty list with explicit reason code
- Quarantine never receives zero-relevance reviews
- `property_id` filter active when present in context
- Region `"unknown"` sentinel never written to vector metadata
- `noise_rejection_stats` appears in all non-disabled scrape runs
- `selector_miss_reason` appears in step response when selectors miss
- No API contract or step module name changes
- `pytest tests/ -v` passes

---

## Plan 2: Migrate to Bright Data Web Scraper API (Dual-Mode)

### Prerequisite
**Plan 1 must be merged first.** Playwright fallback must be clean before it serves as a safety net — otherwise `auto` mode just falls back to garbage.

### Goal
Adopt Bright Data Web Scraper API (SERP endpoint) for higher extraction reliability while keeping service continuity via Playwright fallback, with circuit breaker protection, cost guardrails, and a hard per-request timeout budget.

### Naming convention
All Bright Data config, classes, and env vars use the `BRIGHTDATA_*` prefix. No `_MCP_` suffix — the integration is a direct HTTP/JSON API call, not Model Context Protocol.

### Why dual-mode

- **No hard cutover risk:** quality and cost validated in production before disabling Playwright.
- **Availability guarantee:** API key/quota issues don't take scraping offline.
- **Objective comparison:** A/B quality and cost metrics before full switch.
- **Instant rollback:** single env var change (`SCRAPING_PROVIDER=playwright`), no code deploy.
- **Bright Data advantage:** returns pre-rendered structured JSON with `reviewer_name`, `review_date`, `rating` (always None in Playwright path), eliminating JS-render failures.

### Target architecture

```
ReviewsAgent
  └─ ReviewScraperProvider (protocol)
       ├─ DualModeReviewScraperProvider (SCRAPING_PROVIDER=auto)
       │    ├─ primary:  BrightDataReviewScraper
       │    │    ├─ _CircuitBreaker (5 consecutive OR >50% rate in 20-window)
       │    │    └─ _DailyRequestCounter (default cap: 200/day)
       │    └─ fallback: PlaywrightReviewScraper
       ├─ BrightDataReviewScraper (SCRAPING_PROVIDER=brightdata)
       └─ PlaywrightReviewScraper (SCRAPING_PROVIDER=playwright)
```

### Serverless / multi-instance limitation

Circuit breaker and daily request counter are **in-memory, per-process**. On Vercel serverless (or any multi-instance deployment), each cold-start instance maintains its own counters. This means:
- Circuit breaker may trip independently per instance — one instance may fall back to Playwright while another still tries Bright Data.
- Daily cap is per-instance, not global — actual daily request count across all instances may exceed the cap by a factor of (concurrent instances).

**This is acceptable because:** (a) Playwright fallback is safe (Plan 1 ensures it's clean), (b) the cap's purpose is cost guardrail, not hard budget enforcement, (c) true distributed state (Redis, DynamoDB) is out of scope for Plan 2 and would add infrastructure complexity disproportionate to the risk. If precise global caps become necessary, a future Plan 3 can introduce an external counter.

### Code locations to change

**New files:**
- `app/services/review_scraper_provider.py` — `ReviewScraperProvider` protocol + `DualModeReviewScraperProvider`
- `app/services/brightdata_review_scraper.py` — `BrightDataReviewScraper` + `_CircuitBreaker` + `_DailyRequestCounter`
- `scripts/run_brightdata_smoke_test.py` — mandatory live smoke test

**Modified files:**
- `app/agents/reviews_agent.py` — type `web_scraper` as `ReviewScraperProvider`
- `app/services/web_review_ingest.py` — include `provider_selected` and `provider_version` in quarantine vector metadata
- `app/config.py` — add Bright Data config fields
- `app/main.py` — provider selection/instantiation logic

### Implementation steps

#### Step 1 — Provider protocol
**New file:** `app/services/review_scraper_provider.py`

```python
@runtime_checkable
class ReviewScraperProvider(Protocol):
    @property
    def is_available(self) -> bool: ...

    def scrape_reviews(
        self, *, prompt: str, property_name: str | None, city: str | None,
        region: str | None, source_urls: dict[str, str] | None,
        max_reviews: int | None,
    ) -> tuple[list[ScrapedReview], dict[str, Any]]: ...
```

`PlaywrightReviewScraper` already satisfies this signature — add `provider: "playwright"` to its meta dict.

#### Step 2 — Bright Data client wrapper
**New file:** `app/services/brightdata_review_scraper.py`

- HTTP client using `httpx` (sync, matching Playwright's sync pattern)
- Maps Bright Data SERP API JSON to `ScrapedReview` — populates `reviewer_name`, `review_date`, `rating`
- Exponential backoff retry: 3 attempts, 2^n second delays
- 4xx (except 429) fails fast; 429 and 5xx retry
- **Hard total timeout budget:** `BRIGHTDATA_TOTAL_TIMEOUT_SECONDS` (default 60) enforced across all retries and all targets in a single `scrape_reviews` call. Individual per-target timeout is `BRIGHTDATA_TIMEOUT_SECONDS` (default 30). If the total budget expires mid-iteration, remaining targets are skipped and partial results returned with `status=partial_timeout`.

#### Step 3 — Circuit breaker
**In:** `app/services/brightdata_review_scraper.py`

States: CLOSED → OPEN → HALF-OPEN → CLOSED

- **Trip conditions (OR'd):** 5 consecutive failures OR >50% failure rate in last 20 requests
- **Auto-reset:** after 300s cooldown, transitions to HALF-OPEN; next success closes it
- **When open:** returns `status=circuit_open`, triggers Playwright fallback
- Thread-safe via `threading.Lock`
- **Scope:** per-process (see serverless limitation section above)

#### Step 4 — Daily cost cap
**In:** `app/services/brightdata_review_scraper.py`

`_DailyRequestCounter`: thread-safe counter resetting at UTC midnight. When `BRIGHTDATA_DAILY_REQUEST_CAP` (default 200) reached, returns `status=daily_cap_reached`, triggers Playwright fallback.

**Scope:** per-process, best-effort (see serverless limitation section above).

#### Step 5 — Dual-mode routing
**In:** `app/services/review_scraper_provider.py`

`DualModeReviewScraperProvider`:
- Tries Bright Data first
- On any non-ok status (error, circuit_open, daily_cap_reached, partial_timeout), falls through to Playwright
- Emits in meta: `provider_selected`, `provider_attempts`, `brightdata_failed_reason`

#### Step 6 — Provider metadata in quarantine vectors
**File:** `app/services/web_review_ingest.py`

Add two metadata fields to every quarantine-upserted vector:
```python
"provider_selected": meta.get("provider", "unknown"),    # "playwright" or "brightdata"
"provider_version": "1.0",                                # bumped on provider logic changes
```

These enable later analysis of per-provider quality in the quarantine namespace without requiring re-scraping.

#### Step 7 — Provider instantiation
**File:** `app/main.py`

Based on `SCRAPING_PROVIDER`:
- `auto` → `DualModeReviewScraperProvider(brightdata=..., playwright=...)`
- `brightdata` → `BrightDataReviewScraper` only
- `playwright` (default) → `PlaywrightReviewScraper` only

### Config additions

All env vars use `BRIGHTDATA_*` prefix consistently.

| Env var | Default | Purpose |
|---------|---------|---------|
| `SCRAPING_PROVIDER` | `auto` | Provider routing: `playwright`, `brightdata`, or `auto` |
| `BRIGHTDATA_API_KEY` | None | API authentication |
| `BRIGHTDATA_BASE_URL` | `https://api.brightdata.com` | API endpoint |
| `BRIGHTDATA_TIMEOUT_SECONDS` | `30` | Per-target request timeout |
| `BRIGHTDATA_TOTAL_TIMEOUT_SECONDS` | `60` | Hard budget across all targets in one scrape call |
| `BRIGHTDATA_MAX_RESULTS` | `10` | Max reviews per request |
| `BRIGHTDATA_DAILY_REQUEST_CAP` | `200` | Cost guardrail (per-process) |
| `BRIGHTDATA_CIRCUIT_FAILURE_THRESHOLD` | `5` | Consecutive failures to trip breaker |
| `BRIGHTDATA_CIRCUIT_WINDOW_SIZE` | `20` | Sliding window size |
| `BRIGHTDATA_CIRCUIT_FAILURE_RATE` | `0.50` | Failure rate threshold to trip |
| `BRIGHTDATA_CIRCUIT_COOLDOWN_SECONDS` | `300` | Cooldown before probe |

No API schema changes. Step logging enhanced with additive keys only (`provider_selected`, `provider_attempts`, `brightdata_failed_reason`).

### Rollback plan

Set `SCRAPING_PROVIDER=playwright` — single env var change, no code deploy needed. Bypasses all Bright Data and DualMode code entirely. Vercel serverless re-instantiates on next cold start.

For gradual quality degradation (responses that don't error but produce irrelevant content): set the env var and the stabilized Playwright path from Plan 1 takes over.

### Tests

All tests use **pytest** (set up in Plan 1).

1. **Provider selection matrix** (`tests/services/test_review_scraper_provider.py`):
   - `SCRAPING_PROVIDER=playwright`, no BD key → `PlaywrightReviewScraper` instance
   - `SCRAPING_PROVIDER=brightdata`, BD key set → `BrightDataReviewScraper` instance
   - `SCRAPING_PROVIDER=auto`, BD available → BD primary, PW not called
   - `SCRAPING_PROVIDER=auto`, BD timeout → PW fallback with `brightdata_failed=True`

2. **Circuit breaker** (`tests/services/test_circuit_breaker.py`):
   - 5 consecutive `record_failure()` → `is_open == True`
   - 11 failures in 20-request window → `is_open == True`
   - Open + time past cooldown → HALF-OPEN
   - HALF-OPEN + `record_success()` → CLOSED

3. **Daily cap** (`tests/services/test_daily_request_counter.py`):
   - Counter at cap → `try_increment()` returns `False`
   - Day rolls over → counter resets

4. **Response mapping** (`tests/services/test_brightdata_review_scraper.py`):
   - BD JSON fixture → `ScrapedReview` with non-None `reviewer_name`, `review_date`, `rating`
   - BD timeout → `meta["status"] == "error"`, circuit breaker increments
   - Total timeout budget exceeded → `meta["status"] == "partial_timeout"`

5. **Provider metadata in quarantine** (`tests/services/test_web_review_ingest.py`):
   - Upserted vector metadata includes `provider_selected` and `provider_version`

6. **Live smoke test (MANDATORY)** (`scripts/run_brightdata_smoke_test.py`):
   - Against real BD endpoint for one known property
   - Assert >= 1 review returned, all pass `_looks_like_review`
   - Print comparison table: provider | raw count | accepted | avg text len | has name/date/rating %
   - Gated on `BRIGHTDATA_API_KEY` being set; skips cleanly if absent

7. **Regression:**
   - Step module name strings unchanged
   - `SCRAPING_PROVIDER=playwright` identical to pre-Plan-2 behavior

### Acceptance criteria
- `SCRAPING_PROVIDER=auto` works end-to-end with deterministic fallback
- Circuit breaker prevents cascading BD failures (verified by unit test)
- Daily cap prevents runaway costs (verified by unit test)
- BD path populates reviewer_name/date/rating at >= 80% rate in smoke test
- `SCRAPING_PROVIDER=playwright` produces identical behavior to pre-Plan-2
- Quarantine vectors include `provider_selected` and `provider_version` metadata
- Total timeout budget enforced across multi-target scrape calls
- No API contract or step module name changes
- `pytest tests/ -v` passes

---

## Assumptions and defaults
- Keep response schemas unchanged.
- Keep quarantine namespace flow unchanged.
- Default rollout mode: `SCRAPING_PROVIDER=auto` (Bright Data primary, Playwright fallback).
- Test framework: pytest, layout under `tests/`.
- In-memory counters are per-process; acceptable for current scale.
