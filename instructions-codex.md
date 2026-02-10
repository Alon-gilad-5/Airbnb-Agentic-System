# Codex Working Instructions

## Communication and Code Quality

- Do not use emojis in responses, code comments, file content, or generated outputs unless the user explicitly asks for them.
- Add concise comments/docstrings in code where logic is non-obvious, with emphasis on assumptions, tradeoffs, and operational behavior.
- Keep API contracts stable when adding features; avoid breaking required response schemas.

## Runtime and Environment Rules

- Use Python 3.12 explicitly in this project (`py -3.12 ...`) to avoid interpreter/package mismatch.
- Ensure required env vars are present before running live flows:
- `PINECONE_API_KEY`
- `LLMOD_API_KEY`
- `BASE_URL`
- Prefer settings-driven behavior (env vars) over hardcoded runtime values.

## LLMOD and Model Compatibility

- Default chat model for this project should be `RPRTHPB-gpt-5-mini` unless user changes it.
- LLMOD keys may be restricted to specific model groups; if 401 model access errors appear, switch to an allowed model.
- For gpt-5 model groups, set `temperature=1` (other values can fail in this environment).
- Always set an output cap (`max_tokens`) for cost control and concise answers.
- If model output is empty, produce a deterministic fallback answer instead of returning blank text.

## Retrieval and Evidence Guardrails

- If evidence count is zero, return exactly: `I couldn't find enough data to answer your question.`
- For thin evidence (1-2 reviews), add an explicit low-evidence disclaimer.
- Always add citations to final answers.
- Run hallucination-risk phrase checks and log warning signals in steps and server logs.

## Scraping and Playwright

- Playwright fallback must be feature-flagged (`SCRAPING_ENABLED`) and capped (`max_scrape_reviews`).
- Installing the Python package is not enough; browser binaries must also be installed (`py -3.12 -m playwright install chromium`).
- Use source allowlist controls (Google Maps and TripAdvisor by default).
- Scraped reviews should be upserted to a quarantine namespace, not mixed blindly into curated internal vectors.

## Pinecone and Metadata Safety

- Pinecone metadata values must be valid scalar/list values; do not send null for fields like `rating`.
- Sanitize optional metadata fields before upsert (omit or convert nulls).
- Use deterministic IDs for scraped content to reduce duplicate upserts.

## Debugging Lessons from This Project

- If live query output printing fails on Windows terminal encoding, print ASCII-safe text for diagnostics.
- If network-reliant commands fail in sandbox, rerun with approved escalated permissions.
- Keep explicit step modules for new logic (`retrieval`, `web_scrape`, `web_quarantine_upsert`, `evidence_guard`, `answer_generation`, `hallucination_guard`) so behavior is auditable.
