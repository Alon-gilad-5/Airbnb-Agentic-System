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
    llm_provider: Literal["default", "llmod", "openrouter"] | None = Field(
        default=None,
        description=(
            "Optional chat provider override for this request. "
            "Use 'default' or omit to use deployment default."
        ),
    )
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


class PropertyProfileResponse(BaseModel):
    """One selectable property profile used by Reviews and Market Watch UIs."""

    profile_id: Literal["primary", "secondary"]
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
    review_volume_label: str | None = None


class PropertyProfilesResponse(BaseModel):
    """Available selectable property profiles plus default selection marker."""

    default_profile_id: Literal["primary", "secondary"]
    profiles: list[PropertyProfileResponse]


class AnalysisRequest(BaseModel):
    """Request payload for structured competitive analysis runs."""

    prompt: str | None = Field(
        default=None,
        description="Optional freeform analyst prompt",
    )
    llm_provider: Literal["default", "llmod", "openrouter"] | None = Field(
        default=None,
        description=(
            "Optional chat provider override for this analysis request. "
            "Use 'default' or omit to use deployment default."
        ),
    )
    property_id: str | None = None
    category: Literal["review_scores", "property_specs"] | None = Field(
        default=None,
        description="Optional category shortcut or explicit override",
    )


class AnalysisNeighborPoint(BaseModel):
    """One neighbor listing point used by numeric distribution charts."""

    listing_id: str
    listing_name: str | None
    value: float


class AnalysisNumericItem(BaseModel):
    """One numeric comparison row for owner vs. neighbors."""

    label: str
    column: str
    owner_value: float | int | None
    neighbor_avg: float | None
    neighbor_min: float | None
    neighbor_max: float | None
    neighbor_count: int
    neighbor_points: list[AnalysisNeighborPoint]
    neighbor_min_points: list[AnalysisNeighborPoint]
    neighbor_max_points: list[AnalysisNeighborPoint]


class AnalysisCategoryBucket(BaseModel):
    """One categorical distribution bucket among neighbor listings."""

    value: str
    count: int
    pct: float
    listing_ids: list[str]
    listing_names: list[str]


class AnalysisCategoricalItem(BaseModel):
    """One categorical comparison item for owner vs. neighbor distribution."""

    label: str
    column: str
    owner_value: str | None
    neighbor_count: int
    buckets: list[AnalysisCategoryBucket]


class AnalysisResponse(BaseModel):
    """Response payload for analyst-agent runs."""

    status: Literal["ok", "error"]
    error: str | None
    response: str | None
    analysis_category: Literal["review_scores", "property_specs"] | None = None
    numeric_comparison: list[AnalysisNumericItem]
    categorical_comparison: list[AnalysisCategoricalItem]
    steps: list[StepLog]


class AnalysisExplainSelectionRequest(BaseModel):
    """Request payload for explaining one selected visualization item."""

    property_id: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    category: Literal["review_scores", "property_specs"]
    selection_type: Literal["numeric_point", "numeric_extreme", "categorical_bucket"]
    metric_column: str = Field(min_length=1)
    selection_payload: dict[str, Any]
    llm_provider: Literal["default", "llmod", "openrouter"] | None = Field(
        default=None,
        description=(
            "Optional chat provider override for this selection explanation request. "
            "Use 'default' or omit to use deployment default."
        ),
    )


class AnalysisExplainSelectionResponse(BaseModel):
    """Response payload for selection-specific analyst explanations."""

    status: Literal["ok", "error"]
    error: str | None
    response: str | None
    steps: list[StepLog]


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


class MarketWatchRunRequest(BaseModel):
    """Optional context override for manual market-watch runs."""

    owner_id: str | None = None
    owner_name: str | None = None
    property_id: str | None = None
    property_name: str | None = None
    city: str | None = None
    region: str | None = None
    latitude: float | None = None
    longitude: float | None = None


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


# ---------------------------------------------------------------------------
# Mail notification schemas
# ---------------------------------------------------------------------------


class NotificationItem(BaseModel):
    """A single mail notification requiring owner attention."""

    id: str
    email_id: str
    created_at: str | None
    category: str
    subject: str
    sender: str
    guest_name: str | None = None
    rating: int | None = None
    snippet: str
    action_data: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"


class NotificationsResponse(BaseModel):
    """Response schema for mail notifications list."""

    status: Literal["ok", "error"]
    error: str | None = None
    notifications: list[NotificationItem] = Field(default_factory=list)
    count: int = 0


# ---------------------------------------------------------------------------
# Mail settings (persisted preferences)
# ---------------------------------------------------------------------------


class MailSettingsResponse(BaseModel):
    """Response schema for GET /api/mail/settings."""

    status: Literal["ok", "error"]
    error: str | None = None
    auto_send_good_reviews: bool = False


class MailSettingsUpdateRequest(BaseModel):
    """Request body for POST /api/mail/settings."""

    auto_send_good_reviews: bool = Field(description="If True, agent sends replies for good reviews (rating > 3) without owner approval.")


class EvidenceFlagRequest(BaseModel):
    """Request body for POST /api/evidence/flag."""

    vector_id: str = Field(description="Pinecone vector ID of the flagged evidence item.")
    query_text: str = Field(description="The user query that surfaced this evidence.")
    flag: str = Field(default="irrelevant", description="Flag type (default: irrelevant).")
