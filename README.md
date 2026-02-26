# Airbnb Business Agent (Course Project Scaffold)

This repository includes a multi-agent-ready API scaffold for hospitality intelligence.

## Implemented endpoints

- `GET /api/team_info`
- `GET /api/active_owner_context`
- `GET /api/property_profiles`
- `GET /api/agent_info`
- `GET /api/model_architecture` (PNG)
- `POST /api/execute`
- `GET /api/market_watch/alerts`
- `POST /api/market_watch/run`

These endpoint names and response shapes are aligned with the course PDF requirements.

## Current architecture

- `router_agent` decides which domain agent to run.
- `reviews_agent` performs:
  - `reviews_agent.retrieval` (embeddings + Pinecone search)
  - `reviews_agent.answer_generation` (LLMOD chat synthesis)
- `market_watch_agent` performs:
  - `market_watch_agent.signal_collection` (weather/events/holidays)
  - `market_watch_agent.weather_analysis`
  - `market_watch_agent.event_analysis`
  - `market_watch_agent.demand_analysis`
  - `market_watch_agent.alert_decision`
  - `market_watch_agent.inbox_write`
  - `market_watch_agent.answer_generation`
- Each execution returns `steps` with `module`, `prompt`, and `response`.

## Project structure

- `app/main.py`: FastAPI app + required endpoints.
- `app/config.py`: environment-driven settings.
- `app/schemas.py`: request/response schemas.
- `app/agents/`: router + domain agents.
- `app/services/`: embeddings/chat/pinecone wrappers.
- `app/services/market_data_providers.py`: weather/events/holiday provider clients.
- `app/services/market_alert_store.py`: SQLite/Postgres alert inbox backends.
- `app/services/market_watch_scheduler.py`: autonomous run scheduler.
- `app/architecture.py`: generates architecture PNG.
- `app/templates/index.html`: minimal required GUI.

## Environment variables

Minimum for real retrieval and synthesis:

- `PINECONE_API_KEY`
- `LLMOD_API_KEY`
- `BASE_URL`

Recommended:

- `PINECONE_INDEX_NAME=airbnb-reviews`
- `PINECONE_NAMESPACE=airbnb-reviews`
- `CHAT_MODEL=gpt-4.1-mini`
- `EMBEDDING_MODEL=RPRTHPB-text-embedding-3-small`
- `EMBEDDING_DEPLOYMENT=RPRTHPB-text-embedding-3-small`
- `SCRAPING_ENABLED=false`
- `SCRAPING_ALLOWLIST=google_maps,tripadvisor`
- `SCRAPING_DEFAULT_MAX_REVIEWS=5`
- `SCRAPING_TIMEOUT_SECONDS=45`
- `REVIEWS_RELEVANCE_SCORE_THRESHOLD=0.40`
- `SCRAPING_QUARANTINE_UPSERT_ENABLED=true`
- `SCRAPING_QUARANTINE_NAMESPACE=airbnb-reviews-web-quarantine`
- `TICKETMASTER_API_KEY=` (required for event signals)

Chat provider selection (chat-completions only):

- `LLM_CHAT_PROVIDER=llmod` (`llmod` or `openrouter`)
- `OPENROUTER_API_KEY=` (required when using `openrouter`)
- `OPENROUTER_BASE_URL=https://openrouter.ai/api/v1`
- `OPENROUTER_CHAT_MODEL=openai/gpt-4o-mini`
- `OPENROUTER_HTTP_REFERER=` (optional attribution header)
- `OPENROUTER_APP_TITLE=` (optional attribution header)

Notes:

- Embeddings and Pinecone retrieval stay on the existing LLMOD embedding configuration.
- You can override chat provider per request with `llm_provider` in `POST /api/execute`.

Active owner/property defaults (used when request omits context fields):

- `ACTIVE_OWNER_ID`
- `ACTIVE_OWNER_NAME`
- `ACTIVE_PROPERTY_ID`
- `ACTIVE_PROPERTY_NAME`
- `ACTIVE_PROPERTY_CITY`
- `ACTIVE_PROPERTY_REGION`
- `ACTIVE_PROPERTY_LAT`
- `ACTIVE_PROPERTY_LON`
- `ACTIVE_PROPERTY_GOOGLE_MAPS_URL`
- `ACTIVE_PROPERTY_TRIPADVISOR_URL`
- `ACTIVE_MAX_SCRAPE_REVIEWS`

Optional secondary property profile (shown in UI selector):

- `SECONDARY_PROPERTY_ID`
- `SECONDARY_PROPERTY_NAME`
- `SECONDARY_PROPERTY_CITY`
- `SECONDARY_PROPERTY_REGION`
- `SECONDARY_PROPERTY_LAT`
- `SECONDARY_PROPERTY_LON`
- `SECONDARY_PROPERTY_GOOGLE_MAPS_URL`
- `SECONDARY_PROPERTY_TRIPADVISOR_URL`
- `SECONDARY_MAX_SCRAPE_REVIEWS`

Market-watch settings:

- `MARKET_WATCH_ENABLED=true`
- `MARKET_WATCH_AUTONOMOUS_ENABLED=true`
- `MARKET_WATCH_AUTONOMOUS_MODE=internal` (`internal` or `external_cron`)
- `MARKET_WATCH_INTERVAL_HOURS=24`
- `MARKET_WATCH_LOOKAHEAD_DAYS=14`
- `MARKET_WATCH_EVENT_RADIUS_KM=15`
- `MARKET_WATCH_MAX_ALERTS_PER_RUN=8`
- `MARKET_WATCH_ALERTS_DB_PATH=data/market_watch_alerts.db`
- `MARKET_WATCH_STORM_WIND_KPH_THRESHOLD=45`
- `MARKET_WATCH_HEAVY_RAIN_MM_THRESHOLD=20`
- `MARKET_WATCH_SNOW_CM_THRESHOLD=4`
- `MARKET_WATCH_CRON_SECRET=` (required in `external_cron` mode)
- `DATABASE_URL=` (set to Postgres URL on Vercel)

Team info values:

- `GROUP_BATCH_ORDER_NUMBER`
- `TEAM_NAME`
- `STUDENT_1_NAME`, `STUDENT_1_EMAIL`
- `STUDENT_2_NAME`, `STUDENT_2_EMAIL`
- `STUDENT_3_NAME`, `STUDENT_3_EMAIL`

## Run locally

```bash
py -m pip install -r requirements.txt
py -m playwright install chromium
py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open:

- UI: `http://localhost:8000/`
- Docs: `http://localhost:8000/docs`

## Market-watch usage

Trigger one cycle manually:

```bash
curl -X POST http://localhost:8000/api/market_watch/run
```

Read latest alerts:

```bash
curl "http://localhost:8000/api/market_watch/alerts?limit=20"
```

Read alerts for a specific profile scope:

```bash
curl "http://localhost:8000/api/market_watch/alerts?limit=20&owner_id=owner_001&property_id=10046908"
```

Run market-watch for an explicit property context:

```bash
curl -X POST http://localhost:8000/api/market_watch/run \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "owner_001",
    "property_id": "10046908",
    "property_name": "Cozy Vintage-Styled Unit w/ Patio",
    "city": "Los Angeles",
    "region": "los angeles",
    "latitude": 34.01542,
    "longitude": -118.29229
  }'
```

Ask via `/api/execute` (router sends market-intel prompts to `market_watch_agent`):

```json
{
  "prompt": "Are there any nearby events next week that could increase demand?"
}
```

Force chat provider for one request:

```json
{
  "prompt": "What do guests think about wifi reliability?",
  "llm_provider": "openrouter"
}
```

## Sample properties for validation

Based on current local review archives:

- High-review sample: `region=los angels`, `property_id=42409434` (`3151` reviews)
- Low-review sample: `region=los angels`, `property_id=290761` (`1` review)

Example execute request:

```json
{
  "prompt": "What do guests think about wifi reliability?",
  "property_id": "42409434",
  "region": "los angels",
  "max_scrape_reviews": 5
}
```

Generate sample properties and run smoke tests:

```bash
py -3.12 scripts/select_sample_properties.py
py -3.12 scripts/run_property_smoke_tests.py
```

Targeted one-property ingest into test namespace (low-review profile example):

```bash
py -3.12 scripts/pinecone_reviews_ingest.py \
  --mode upsert \
  --index-name airbnb-reviews \
  --namespace airbnb-reviews-test \
  --selection-mode manual \
  --property-ids-file data/property_ids_low_review.txt \
  --max-properties 1 \
  --max-reviews 20 \
  --no-resume \
  --checkpoint-path ingest_state/reviews_ingest_low_property_checkpoint.json
```

Smoke test output is saved to:

- `outputs/sample_properties.json`
- `outputs/property_smoke_test_results.json`

## Relevance-threshold calibration workflow

Build deterministic benchmark cases (60 total by default, balanced high/low properties):

```bash
py -3.12 scripts/build_reviews_threshold_benchmark.py --namespace airbnb-reviews-test
```

Export candidate evidence pool (`top_k=20`) for manual labeling:

```bash
py -3.12 scripts/export_threshold_labeling_pool.py --namespace airbnb-reviews-test
```

Initialize manual gold CSV template:

```bash
py -3.12 scripts/init_reviews_threshold_gold.py --overwrite
```

After filling `outputs/reviews_threshold_gold.csv`, optimize threshold and produce artifacts:

```bash
py -3.12 scripts/optimize_reviews_threshold.py
```

Run fixed-subset validation smoke (scraping disabled to isolate VDB behavior):

```bash
py -3.12 scripts/run_reviews_threshold_validation.py --namespace airbnb-reviews-test --threshold 0.40
```

## Vercel transition notes

- `vercel.json` is included with daily cron calling `POST /api/market_watch/run`.
- `api/index.py` exports the FastAPI app for Vercel runtime.
- For Vercel:
  - set `MARKET_WATCH_AUTONOMOUS_MODE=external_cron`
  - set `MARKET_WATCH_CRON_SECRET`
  - set `DATABASE_URL` to Postgres (SQLite is not durable in serverless)
  - set the same cron secret in your Vercel cron auth flow

## Notes for future agents

To add another agent later:

1. Create a new agent in `app/agents/`.
2. Register it in `agent_registry` in `app/main.py`.
3. Update `router_agent` routing logic.
4. Add matching module name(s) to the architecture diagram generator.
