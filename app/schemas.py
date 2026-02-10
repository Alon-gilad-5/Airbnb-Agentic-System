"""Pydantic schemas for API request/response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TeamStudentResponse(BaseModel):
    """One team member entry."""

    name: str
    email: str


class TeamInfoResponse(BaseModel):
    """Response schema for `GET /api/team_info`."""

    group_batch_order_number: str
    team_name: str
    students: list[TeamStudentResponse]


class StepLog(BaseModel):
    """Standard execution step shape required by project guidelines."""

    module: str
    prompt: dict[str, Any]
    response: dict[str, Any]


class ExecuteRequest(BaseModel):
    """Input schema for `POST /api/execute`."""

    prompt: str = Field(min_length=1, description="User request text")
    property_id: str | None = Field(default=None, description="Known internal property identifier")
    property_name: str | None = Field(default=None, description="Property display name")
    city: str | None = Field(default=None, description="Property city")
    region: str | None = Field(default=None, description="Property region")
    latitude: float | None = Field(default=None, description="Optional property latitude override")
    longitude: float | None = Field(default=None, description="Optional property longitude override")
    source_urls: dict[str, str] | None = Field(
        default=None,
        description="Optional source URL map, e.g. {'google_maps': '...', 'tripadvisor': '...'}",
    )
    max_scrape_reviews: int | None = Field(
        default=None,
        ge=1,
        description="Optional per-request override for future scrape fallback cap",
    )


class ExecuteResponse(BaseModel):
    """Output schema for `POST /api/execute`."""

    status: Literal["ok", "error"]
    error: str | None
    response: str | None
    steps: list[StepLog]


class AgentPromptTemplate(BaseModel):
    """Wrapper for prompt template returned by `/api/agent_info`."""

    template: str


class AgentPromptExample(BaseModel):
    """Example prompt with full expected output style."""

    prompt: str
    full_response: str
    steps: list[StepLog]


class AgentInfoResponse(BaseModel):
    """Response schema for `GET /api/agent_info`."""

    description: str
    purpose: str
    prompt_template: AgentPromptTemplate
    prompt_examples: list[AgentPromptExample]


class ActiveOwnerContextResponse(BaseModel):
    """Runtime owner/property context currently applied by default in `/api/execute`."""

    owner_id: str | None
    owner_name: str | None
    property_id: str | None
    property_name: str | None
    city: str | None
    region: str | None
    latitude: float | None
    longitude: float | None
    source_urls: dict[str, str] | None
    max_scrape_reviews: int | None


class MarketAlertResponse(BaseModel):
    """Serialized alert item exposed by market-watch inbox endpoint."""

    id: str
    created_at_utc: str
    owner_id: str | None
    property_id: str | None
    property_name: str | None
    city: str | None
    region: str | None
    alert_type: str
    severity: str
    title: str
    summary: str
    start_at_utc: str | None
    end_at_utc: str | None
    source_name: str
    source_url: str | None
    evidence: dict[str, Any]


class MarketWatchAlertsResponse(BaseModel):
    """Response schema for reading latest proactive market-watch alerts."""

    status: Literal["ok", "error"]
    error: str | None
    alerts: list[MarketAlertResponse]


class MarketWatchRunResponse(BaseModel):
    """Response schema for manual/cron market-watch cycle triggers."""

    status: Literal["ok", "error"]
    error: str | None
    response: str | None
    inserted_alerts: int
    steps: list[StepLog]
