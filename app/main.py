"""FastAPI entrypoint implementing required course endpoints and minimal GUI."""

from __future__ import annotations

import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.agents.base import Agent
from app.agents.market_watch_agent import MarketWatchAgent, MarketWatchAgentConfig
from app.agents.reviews_agent import ReviewsAgent, ReviewsAgentConfig
from app.agents.router_agent import RouterAgent
from app.architecture import ensure_architecture_png
from app.config import load_settings
from app.schemas import (
    ActiveOwnerContextResponse,
    AgentInfoResponse,
    AgentPromptExample,
    AgentPromptTemplate,
    ExecuteRequest,
    ExecuteResponse,
    MarketAlertResponse,
    MarketWatchAlertsResponse,
    MarketWatchRunResponse,
    StepLog,
    TeamInfoResponse,
    TeamStudentResponse,
)
from app.services.market_alert_store import MarketAlertRecord, create_market_alert_store
from app.services.market_data_providers import MarketDataProviders
from app.services.market_watch_scheduler import MarketWatchScheduler
from app.services.chat_service import ChatService
from app.services.embeddings import EmbeddingService
from app.services.pinecone_retriever import PineconeRetriever
from app.services.web_review_ingest import WebReviewIngestService
from app.services.web_review_scraper import PlaywrightReviewScraper


load_dotenv()
settings = load_settings()
app = FastAPI(title="Airbnb Business Agent", version="0.1.0")
logger = logging.getLogger(__name__)

# Paths are relative to this file, making local + Render deployment consistent.
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
ARCH_PATH = STATIC_DIR / "model_architecture.png"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# Shared services are initialized once and reused by all agents.
embedding_service = EmbeddingService(
    api_key=settings.llmod_api_key,
    azure_endpoint=settings.base_url,
    model=settings.embedding_model,
    deployment=settings.embedding_deployment,
)
retriever = PineconeRetriever(
    api_key=settings.pinecone_api_key,
    index_name=settings.pinecone_index_name,
    namespace=settings.pinecone_namespace,
)
chat_service = ChatService(
    api_key=settings.llmod_api_key,
    base_url=settings.base_url,
    model=settings.chat_model,
    max_output_tokens=settings.chat_max_output_tokens,
)
web_scraper = PlaywrightReviewScraper(
    enabled=settings.scraping_enabled,
    allowlist=settings.scraping_allowlist,
    default_max_reviews=settings.scraping_default_max_reviews,
    timeout_seconds=settings.scraping_timeout_seconds,
)
web_ingest_service = WebReviewIngestService(
    enabled=settings.scraping_quarantine_upsert_enabled,
    pinecone_api_key=settings.pinecone_api_key,
    index_name=settings.pinecone_index_name,
    namespace=settings.scraping_quarantine_namespace,
    embedding_service=embedding_service,
)
market_data_providers = MarketDataProviders(
    ticketmaster_api_key=settings.ticketmaster_api_key,
    timeout_seconds=20,
)
market_alert_store = create_market_alert_store(
    database_url=settings.database_url,
    sqlite_path=settings.market_watch_alerts_db_path,
)
market_watch_agent = MarketWatchAgent(
    providers=market_data_providers,
    alert_store=market_alert_store,
    config=MarketWatchAgentConfig(
        lookahead_days=settings.market_watch_lookahead_days,
        event_radius_km=settings.market_watch_event_radius_km,
        max_alerts_per_run=settings.market_watch_max_alerts_per_run,
        storm_wind_kph_threshold=settings.market_watch_storm_wind_kph_threshold,
        heavy_rain_mm_threshold=settings.market_watch_heavy_rain_mm_threshold,
        snow_cm_threshold=settings.market_watch_snow_cm_threshold,
    ),
)

router_agent = RouterAgent()
agent_registry: dict[str, Agent] = {
    "reviews_agent": ReviewsAgent(
        embedding_service=embedding_service,
        retriever=retriever,
        chat_service=chat_service,
        web_scraper=web_scraper,
        web_ingest_service=web_ingest_service,
        config=ReviewsAgentConfig(top_k=settings.pinecone_top_k),
    ),
    "market_watch_agent": market_watch_agent,
}


def _build_effective_context(payload: ExecuteRequest) -> dict[str, object]:
    """Merge request context over active-owner defaults from environment settings."""

    owner = settings.active_owner
    source_urls: dict[str, str] = {}

    if owner.google_maps_url:
        source_urls["google_maps"] = owner.google_maps_url
    if owner.tripadvisor_url:
        source_urls["tripadvisor"] = owner.tripadvisor_url
    if payload.source_urls:
        source_urls.update(payload.source_urls)

    max_scrape_reviews = (
        payload.max_scrape_reviews
        if payload.max_scrape_reviews is not None
        else owner.default_max_scrape_reviews
    )

    return {
        "owner_id": owner.owner_id,
        "owner_name": owner.owner_name,
        "property_id": payload.property_id or owner.property_id,
        "property_name": payload.property_name or owner.property_name,
        "city": payload.city or owner.city,
        "region": payload.region or owner.region,
        "latitude": payload.latitude if payload.latitude is not None else owner.latitude,
        "longitude": payload.longitude if payload.longitude is not None else owner.longitude,
        "source_urls": source_urls or None,
        "max_scrape_reviews": max_scrape_reviews,
    }


def _build_active_owner_context() -> ActiveOwnerContextResponse:
    """Expose default owner/property context without requiring an execute payload."""

    owner = settings.active_owner
    source_urls: dict[str, str] = {}
    if owner.google_maps_url:
        source_urls["google_maps"] = owner.google_maps_url
    if owner.tripadvisor_url:
        source_urls["tripadvisor"] = owner.tripadvisor_url

    return ActiveOwnerContextResponse(
        owner_id=owner.owner_id,
        owner_name=owner.owner_name,
        property_id=owner.property_id,
        property_name=owner.property_name,
        city=owner.city,
        region=owner.region,
        latitude=owner.latitude,
        longitude=owner.longitude,
        source_urls=source_urls or None,
        max_scrape_reviews=owner.default_max_scrape_reviews,
    )


def _build_autonomous_context() -> dict[str, object]:
    """Build context for autonomous market-watch runs from active owner defaults."""

    owner = settings.active_owner
    return {
        "owner_id": owner.owner_id,
        "owner_name": owner.owner_name,
        "property_id": owner.property_id,
        "property_name": owner.property_name,
        "city": owner.city,
        "region": owner.region,
        "latitude": owner.latitude,
        "longitude": owner.longitude,
    }


def _serialize_alert(record: MarketAlertRecord) -> MarketAlertResponse:
    """Convert internal alert record to API response schema."""

    return MarketAlertResponse(
        id=record.id,
        created_at_utc=record.created_at_utc,
        owner_id=record.owner_id,
        property_id=record.property_id,
        property_name=record.property_name,
        city=record.city,
        region=record.region,
        alert_type=record.alert_type,
        severity=record.severity,
        title=record.title,
        summary=record.summary,
        start_at_utc=record.start_at_utc,
        end_at_utc=record.end_at_utc,
        source_name=record.source_name,
        source_url=record.source_url,
        evidence=record.evidence,
    )


def _extract_bearer_token(authorization: str | None) -> str | None:
    """Parse Authorization header as Bearer token, returning None if absent/invalid."""

    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _assert_market_watch_trigger_authorized(
    *,
    x_market_watch_secret: str | None,
    authorization: str | None,
) -> None:
    """Enforce secret check in external cron mode while allowing local internal-mode runs."""

    if settings.market_watch_autonomous_mode != "external_cron":
        return
    expected = settings.market_watch_cron_secret
    if not expected:
        raise HTTPException(
            status_code=500,
            detail="MARKET_WATCH_CRON_SECRET is required when MARKET_WATCH_AUTONOMOUS_MODE=external_cron.",
        )
    bearer = _extract_bearer_token(authorization)
    provided = x_market_watch_secret or bearer
    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized market_watch run trigger.")


def _run_market_watch_cycle() -> None:
    """Background scheduler callback that executes one autonomous market-watch cycle."""

    if not settings.market_watch_enabled:
        return
    outcome = market_watch_agent.run_autonomous(context=_build_autonomous_context())
    logger.info(
        "market_watch autonomous run finished: inserted_alerts=%s response_preview=%s",
        outcome.inserted_count,
        outcome.response[:120],
    )


market_watch_scheduler = MarketWatchScheduler(
    enabled=settings.market_watch_enabled and settings.market_watch_autonomous_enabled,
    mode=settings.market_watch_autonomous_mode,
    interval_hours=settings.market_watch_interval_hours,
    run_job=_run_market_watch_cycle,
    logger=logger,
)


@app.on_event("startup")
def startup() -> None:
    """Ensure static assets exist and start autonomous scheduler when configured."""

    ensure_architecture_png(ARCH_PATH)
    market_watch_scheduler.start()


@app.on_event("shutdown")
def shutdown() -> None:
    """Stop background scheduler cleanly on process shutdown."""

    market_watch_scheduler.stop()


@app.get("/", response_class=HTMLResponse)
def web_ui(request: Request) -> HTMLResponse:
    """Minimal UI for running the agent and inspecting `steps` trace."""

    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/team_info", response_model=TeamInfoResponse)
def team_info() -> TeamInfoResponse:
    """Required endpoint: returns team metadata."""

    return TeamInfoResponse(
        group_batch_order_number=settings.group_batch_order_number,
        team_name=settings.team_name,
        students=[
            TeamStudentResponse(name=s.name, email=s.email) for s in settings.students
        ],
    )


@app.get("/api/active_owner_context", response_model=ActiveOwnerContextResponse)
def active_owner_context() -> ActiveOwnerContextResponse:
    """Return currently configured default owner/property context."""

    return _build_active_owner_context()


@app.get("/api/agent_info", response_model=AgentInfoResponse)
def agent_info() -> AgentInfoResponse:
    """Required endpoint: returns purpose, template, and full prompt examples."""

    example_steps = [
        StepLog(
            module="router_agent",
            prompt={"user_prompt": "What do guests think about wifi in Santa Clara?"},
            response={
                "selected_agent": "reviews_agent",
                "reason": "Matched hospitality/review intent keywords.",
            },
        ),
        StepLog(
            module="reviews_agent.retrieval",
            prompt={"top_k": 8, "metadata_filter": {"region": {"$eq": "santa clara"}}},
            response={"match_count": 8, "top_match_ids": ["santa-clara:123", "santa-clara:456"]},
        ),
        StepLog(
            module="reviews_agent.evidence_guard",
            prompt={"relevance_score_threshold": 0.4, "thin_evidence_min": 1, "thin_evidence_max": 2},
            response={"decision": "answer_normal", "relevant_evidence_count": 6},
        ),
        StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": 3},
            response={"status": "ok", "upserted": 3, "namespace": "airbnb-reviews-web-quarantine"},
        ),
        StepLog(
            module="reviews_agent.answer_generation",
            prompt={"model": settings.chat_model, "system_prompt": "...", "user_prompt": "..."},
            response={"text": "Guests mostly describe wifi as stable and fast, with a few weak-signal cases."},
        ),
        StepLog(
            module="reviews_agent.hallucination_guard",
            prompt={"checked_phrases": ["many guests", "a lot of reviews", "most guests", "guests generally", "everyone"]},
            response={"risk_flag": False, "matched_phrases": [], "evidence_count": 6, "action": "flag_only"},
        ),
    ]

    return AgentInfoResponse(
        description=(
            "Multi-agent-ready hospitality insights API. "
            "Enabled domain agents: reviews_agent and market_watch_agent."
        ),
        purpose=(
            "Answer business questions from guest reviews and provide proactive market intelligence "
            "from weather/events/holiday signals."
        ),
        prompt_template=AgentPromptTemplate(
            template=(
                "Question: {business_question}\n"
                "Optional filters: {region/property/date}\n"
                "Goal: actionable summary with evidence and confidence"
            )
        ),
        prompt_examples=[
            AgentPromptExample(
                prompt="What do guests say about wifi quality in Santa Clara?",
                full_response=(
                    "Guests generally report good wifi stability and speed, with occasional weaker signal "
                    "in specific rooms. Confidence: medium."
                ),
                steps=example_steps,
            )
        ],
    )


@app.get("/api/model_architecture")
def model_architecture() -> FileResponse:
    """Required endpoint: returns architecture diagram PNG."""

    if not ARCH_PATH.exists():
        ensure_architecture_png(ARCH_PATH)
    return FileResponse(path=str(ARCH_PATH), media_type="image/png", filename="model_architecture.png")


@app.get("/api/market_watch/alerts", response_model=MarketWatchAlertsResponse)
def market_watch_alerts(limit: int = Query(default=20, ge=1, le=100)) -> MarketWatchAlertsResponse:
    """Return latest stored market-watch alerts for current active owner/property scope."""

    owner = settings.active_owner
    try:
        records = market_alert_store.list_latest_alerts(
            owner_id=owner.owner_id,
            property_id=owner.property_id,
            limit=limit,
        )
        alerts = [_serialize_alert(record) for record in records]
        return MarketWatchAlertsResponse(status="ok", error=None, alerts=alerts)
    except Exception as exc:
        return MarketWatchAlertsResponse(status="error", error=f"{type(exc).__name__}: {exc}", alerts=[])


@app.post("/api/market_watch/run", response_model=MarketWatchRunResponse)
def market_watch_run(
    x_market_watch_secret: str | None = Header(default=None),
    authorization: str | None = Header(default=None),
) -> MarketWatchRunResponse:
    """Trigger one market-watch cycle (manual local use or secured external cron use)."""

    _assert_market_watch_trigger_authorized(
        x_market_watch_secret=x_market_watch_secret,
        authorization=authorization,
    )
    if not settings.market_watch_enabled:
        return MarketWatchRunResponse(
            status="error",
            error="market_watch is disabled (MARKET_WATCH_ENABLED=false).",
            response=None,
            inserted_alerts=0,
            steps=[],
        )

    try:
        outcome = market_watch_agent.run_autonomous(context=_build_autonomous_context())
        return MarketWatchRunResponse(
            status="ok",
            error=None,
            response=outcome.response,
            inserted_alerts=outcome.inserted_count,
            steps=outcome.steps,
        )
    except Exception as exc:
        return MarketWatchRunResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            inserted_alerts=0,
            steps=[],
        )


@app.post("/api/execute", response_model=ExecuteResponse)
def execute(payload: ExecuteRequest) -> ExecuteResponse:
    """Required endpoint: route request, run selected agent, return full trace."""

    steps: list[StepLog] = []
    try:
        decision, route_step = router_agent.route(payload.prompt)
        if decision.agent_name == "market_watch_agent" and not settings.market_watch_enabled:
            decision.agent_name = "reviews_agent"
            decision.reason += " market_watch is disabled, rerouted to reviews_agent."
            route_step.response["selected_agent"] = decision.agent_name
            route_step.response["reason"] = decision.reason
        steps.append(route_step)

        target_agent = agent_registry.get(decision.agent_name)
        if target_agent is None:
            raise HTTPException(status_code=500, detail=f"No agent registered as '{decision.agent_name}'")

        context = _build_effective_context(payload)
        result = target_agent.run(payload.prompt, context=context)
        steps.extend(result.steps)
        for step in result.steps:
            if step.module == "reviews_agent.hallucination_guard" and step.response.get("risk_flag") is True:
                logger.warning(
                    "Hallucination-risk flag raised by reviews_agent. matched_phrases=%s evidence_count=%s",
                    step.response.get("matched_phrases"),
                    step.response.get("evidence_count"),
                )
        return ExecuteResponse(status="ok", error=None, response=result.response, steps=steps)
    except Exception as exc:
        # Keep response format exactly aligned with project error schema.
        return ExecuteResponse(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            response=None,
            steps=steps,
        )
