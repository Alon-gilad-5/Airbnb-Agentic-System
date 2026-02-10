"""Centralized runtime settings loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class TeamStudent:
    """Represents one student entry for `/api/team_info`."""

    name: str
    email: str


@dataclass
class ActiveOwnerContext:
    """Default owner/property context used when request payload omits these fields."""

    owner_id: str | None
    owner_name: str | None
    property_id: str | None
    property_name: str | None
    city: str | None
    region: str | None
    latitude: float | None
    longitude: float | None
    google_maps_url: str | None
    tripadvisor_url: str | None
    default_max_scrape_reviews: int | None


@dataclass
class Settings:
    """Application settings used across API, agent routing, and services."""

    group_batch_order_number: str
    team_name: str
    students: list[TeamStudent]
    llmod_api_key: str | None
    base_url: str | None
    chat_model: str
    chat_max_output_tokens: int
    embedding_model: str
    embedding_deployment: str
    pinecone_api_key: str | None
    pinecone_index_name: str
    pinecone_namespace: str
    pinecone_top_k: int
    scraping_enabled: bool
    scraping_allowlist: list[str]
    scraping_default_max_reviews: int
    scraping_timeout_seconds: int
    scraping_quarantine_upsert_enabled: bool
    scraping_quarantine_namespace: str
    ticketmaster_api_key: str | None
    database_url: str | None
    market_watch_enabled: bool
    market_watch_autonomous_enabled: bool
    market_watch_autonomous_mode: str
    market_watch_interval_hours: int
    market_watch_lookahead_days: int
    market_watch_event_radius_km: int
    market_watch_max_alerts_per_run: int
    market_watch_alerts_db_path: str
    market_watch_cron_secret: str | None
    market_watch_storm_wind_kph_threshold: float
    market_watch_heavy_rain_mm_threshold: float
    market_watch_snow_cm_threshold: float
    active_owner: ActiveOwnerContext


def load_settings() -> Settings:
    """Load settings from environment with safe defaults for local development."""

    def parse_bool(value: str | None, default: bool) -> bool:
        if value is None:
            return default
        lowered = value.strip().lower()
        return lowered in {"1", "true", "yes", "on"}

    def parse_float(value: str | None, default: float | None = None) -> float | None:
        if value is None or value.strip() == "":
            return default
        try:
            return float(value)
        except ValueError:
            return default

    allowlist_raw = os.getenv("SCRAPING_ALLOWLIST", "google_maps,tripadvisor")
    allowlist = [x.strip().lower() for x in allowlist_raw.split(",") if x.strip()]
    active_owner = ActiveOwnerContext(
        owner_id=os.getenv("ACTIVE_OWNER_ID"),
        owner_name=os.getenv("ACTIVE_OWNER_NAME"),
        property_id=os.getenv("ACTIVE_PROPERTY_ID"),
        property_name=os.getenv("ACTIVE_PROPERTY_NAME"),
        city=os.getenv("ACTIVE_PROPERTY_CITY"),
        region=os.getenv("ACTIVE_PROPERTY_REGION"),
        latitude=parse_float(os.getenv("ACTIVE_PROPERTY_LAT")),
        longitude=parse_float(os.getenv("ACTIVE_PROPERTY_LON")),
        google_maps_url=os.getenv("ACTIVE_PROPERTY_GOOGLE_MAPS_URL"),
        tripadvisor_url=os.getenv("ACTIVE_PROPERTY_TRIPADVISOR_URL"),
        default_max_scrape_reviews=(
            int(os.getenv("ACTIVE_MAX_SCRAPE_REVIEWS"))
            if os.getenv("ACTIVE_MAX_SCRAPE_REVIEWS")
            else None
        ),
    )

    students = [
        TeamStudent(
            name=os.getenv("STUDENT_1_NAME", "Student A"),
            email=os.getenv("STUDENT_1_EMAIL", "student.a@example.com"),
        ),
        TeamStudent(
            name=os.getenv("STUDENT_2_NAME", "Student B"),
            email=os.getenv("STUDENT_2_EMAIL", "student.b@example.com"),
        ),
        TeamStudent(
            name=os.getenv("STUDENT_3_NAME", "Student C"),
            email=os.getenv("STUDENT_3_EMAIL", "student.c@example.com"),
        ),
    ]

    embedding_model = os.getenv("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")
    return Settings(
        group_batch_order_number=os.getenv("GROUP_BATCH_ORDER_NUMBER", "00_00"),
        team_name=os.getenv("TEAM_NAME", "Reviews Agent Team"),
        students=students,
        llmod_api_key=os.getenv("LLMOD_API_KEY"),
        base_url=os.getenv("BASE_URL"),
        chat_model=os.getenv("CHAT_MODEL", "RPRTHPB-gpt-5-mini"),
        chat_max_output_tokens=int(os.getenv("CHAT_MAX_OUTPUT_TOKENS", "180")),
        embedding_model=embedding_model,
        embedding_deployment=os.getenv("EMBEDDING_DEPLOYMENT", embedding_model),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "airbnb-reviews"),
        pinecone_namespace=os.getenv("PINECONE_NAMESPACE", "airbnb-reviews"),
        pinecone_top_k=int(os.getenv("PINECONE_TOP_K", "8")),
        scraping_enabled=parse_bool(os.getenv("SCRAPING_ENABLED"), False),
        scraping_allowlist=allowlist,
        scraping_default_max_reviews=int(os.getenv("SCRAPING_DEFAULT_MAX_REVIEWS", "5")),
        scraping_timeout_seconds=int(os.getenv("SCRAPING_TIMEOUT_SECONDS", "45")),
        scraping_quarantine_upsert_enabled=parse_bool(
            os.getenv("SCRAPING_QUARANTINE_UPSERT_ENABLED"),
            True,
        ),
        scraping_quarantine_namespace=os.getenv(
            "SCRAPING_QUARANTINE_NAMESPACE",
            "airbnb-reviews-web-quarantine",
        ),
        ticketmaster_api_key=os.getenv("TICKETMASTER_API_KEY"),
        database_url=os.getenv("DATABASE_URL"),
        market_watch_enabled=parse_bool(os.getenv("MARKET_WATCH_ENABLED"), True),
        market_watch_autonomous_enabled=parse_bool(
            os.getenv("MARKET_WATCH_AUTONOMOUS_ENABLED"),
            True,
        ),
        market_watch_autonomous_mode=os.getenv("MARKET_WATCH_AUTONOMOUS_MODE", "internal").strip().lower(),
        market_watch_interval_hours=int(os.getenv("MARKET_WATCH_INTERVAL_HOURS", "24")),
        market_watch_lookahead_days=int(os.getenv("MARKET_WATCH_LOOKAHEAD_DAYS", "14")),
        market_watch_event_radius_km=int(os.getenv("MARKET_WATCH_EVENT_RADIUS_KM", "15")),
        market_watch_max_alerts_per_run=int(os.getenv("MARKET_WATCH_MAX_ALERTS_PER_RUN", "8")),
        market_watch_alerts_db_path=os.getenv(
            "MARKET_WATCH_ALERTS_DB_PATH",
            "data/market_watch_alerts.db",
        ),
        market_watch_cron_secret=os.getenv("MARKET_WATCH_CRON_SECRET"),
        market_watch_storm_wind_kph_threshold=float(
            os.getenv("MARKET_WATCH_STORM_WIND_KPH_THRESHOLD", "45")
        ),
        market_watch_heavy_rain_mm_threshold=float(
            os.getenv("MARKET_WATCH_HEAVY_RAIN_MM_THRESHOLD", "20")
        ),
        market_watch_snow_cm_threshold=float(os.getenv("MARKET_WATCH_SNOW_CM_THRESHOLD", "4")),
        active_owner=active_owner,
    )
