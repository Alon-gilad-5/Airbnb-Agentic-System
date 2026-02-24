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


class ThresholdLabelCandidate(BaseModel):
    """Single candidate review entry for manual threshold labeling."""

    vector_id: str
    score: float
    review_text: str
    review_id: str | None
    property_id: str | None
    region: str | None
    review_date: str | None


class ThresholdLabelCase(BaseModel):
    """Labeling case composed of one prompt and its candidate evidence rows."""

    case_id: str
    property_id: str
    region: str
    tier: str
    topic: str
    prompt: str
    candidate_count: int
    candidates: list[ThresholdLabelCandidate]
    should_answer: bool | None
    relevant_vector_ids: list[str]
    labeled: bool


class ThresholdLabelingDataResponse(BaseModel):
    """Response payload for threshold labeling UI bootstrap data."""

    status: Literal["ok", "error"]
    error: str | None
    total_cases: int
    labeled_cases: int
    cases: list[ThresholdLabelCase]


class ThresholdLabelSaveRequest(BaseModel):
    """Request payload for saving one case label decision."""

    case_id: str = Field(min_length=1)
    should_answer: bool
    relevant_vector_ids: list[str] = Field(default_factory=list)


class ThresholdLabelSaveResponse(BaseModel):
    """Result payload for one label write operation."""

    status: Literal["ok", "error"]
    error: str | None
    case_id: str


# ---------------------------------------------------------------------------
# Mail agent schemas
# ---------------------------------------------------------------------------


class MailInboxItemResponse(BaseModel):
    """Single classified inbox email item."""

    email_id: str
    subject: str
    sender: str
    date: str
    category: str
    confidence: float
    guest_name: str | None
    rating: int | None
    snippet: str


class MailInboxResponse(BaseModel):
    """Response schema for mail inbox summary."""

    status: Literal["ok", "error"]
    error: str | None
    items: list[MailInboxItemResponse]
    demo_mode: bool
    mail_actions: list[dict[str, Any]] | None = None


class MailActionRequest(BaseModel):
    """Owner action input for HITL mail workflows."""

    email_id: str = Field(min_length=1, description="Target email message ID")
    action_type: str = Field(
        description="Type of action: 'rate_guest', 'approve_response', 'edit_draft', 'don_t_reply', etc.",
    )
    rating: int | None = Field(default=None, ge=1, le=5, description="Guest rating (1-5)")
    issues: list[str] | None = Field(default=None, description="Issue checklist items for negative reviews")
    free_text: str | None = Field(default=None, description="Free-text notes from owner")
    owner_instructions: str | None = Field(
        default=None,
        description="Optional instructions for how the agent should reply (e.g. 'emphasize we fixed the cleaning')",
    )
    reply_style: str | None = Field(
        default=None,
        description="Preset reply style key when multiple options offered (e.g. 'apologetic', 'neutral', 'brief_thanks')",
    )
    don_t_reply: bool | None = Field(
        default=None,
        description="If True, owner chose not to reply to this email",
    )
    approve_and_send: bool | None = Field(
        default=None,
        description="If True with approve_response, send the reply (not just save as draft)",
    )
    thread_id: str | None = Field(default=None, description="For approve_and_send: Gmail thread ID")
    reply_to: str | None = Field(default=None, description="For approve_and_send: To address for the reply")
    subject: str | None = Field(default=None, description="For approve_and_send: Reply subject (e.g. Re: ...)")
    in_reply_to: str | None = Field(default=None, description="For approve_and_send: In-Reply-To header")
    references: str | None = Field(default=None, description="For approve_and_send: References header")
    approved: bool | None = Field(default=None, description="Whether owner approves the draft")
    edited_draft: str | None = Field(default=None, description="Owner-edited draft text")


class MailActionResponse(BaseModel):
    """Response schema for mail action execution."""

    status: Literal["ok", "error"]
    error: str | None
    response: str | None
    steps: list[StepLog]
    mail_actions: list[dict[str, Any]] | None = Field(
        default=None,
        description="Actions from pipeline (reply_options, draft, thread_id, etc.) for UI",
    )
