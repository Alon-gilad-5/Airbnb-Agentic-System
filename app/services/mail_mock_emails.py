"""
Canonical mock-up structure for each mail type the agent handles.

Use for tests, demos, and documentation. Each mock is a dict compatible with
GmailService._raw_to_message() / EmailMessage (id, thread_id, sender, recipient,
subject, snippet, body, date, labels). Optional keys: message_id_header, references
for reply threading.

Categories (from mail_agent):
- guest_message       Guest sent a message (importance: high / low / uncertain)
- leave_review_request  Prompt to leave a review for a guest
- new_property_review   A guest left a review of the property (good 4–5 / bad 1–3)
- unsupported_airbnb  From Airbnb but not one of the above
- non_airbnb         Not from an Airbnb sender domain
"""

from __future__ import annotations

from typing import Any

# Sender that passes is_airbnb_sender (airbnb.com, airbnbmail.com, airbnb.co)
FROM_AIRBNB = "no-reply@airbnb.com"
FROM_AIRBNB_AUTOMATED = "automated@airbnb.com"
TO_OWNER = "owner@example.com"
DEFAULT_DATE = "2026-02-22T10:00:00Z"
DEFAULT_LABELS = ["INBOX", "UNREAD"]


def _base(
    id: str,
    thread_id: str,
    subject: str,
    body: str,
    *,
    from_addr: str = FROM_AIRBNB,
    snippet: str | None = None,
    date: str = DEFAULT_DATE,
) -> dict[str, Any]:
    return {
        "id": id,
        "thread_id": thread_id,
        "from": from_addr,
        "to": TO_OWNER,
        "subject": subject,
        "snippet": snippet or (body[:200] + "..." if len(body) > 200 else body),
        "body": body,
        "date": date,
        "labels": DEFAULT_LABELS.copy(),
    }


# ---------------------------------------------------------------------------
# guest_message
# ---------------------------------------------------------------------------

GUEST_MESSAGE_HIGH_IMPORTANCE = _base(
    "mock-guest-high-001",
    "thread-guest-high",
    "New message from guest Alex – urgent",
    """Hi,

We're locked out – the keypad code isn't working and it's raining. Can you send the backup code or call us? We've been trying for 20 minutes.

Thanks,
Alex""",
    from_addr=FROM_AIRBNB_AUTOMATED,
    snippet="We're locked out – the keypad code isn't working...",
)
GUEST_MESSAGE_HIGH_IMPORTANCE["_category"] = "guest_message"
GUEST_MESSAGE_HIGH_IMPORTANCE["_importance"] = "high"

GUEST_MESSAGE_LOW_IMPORTANCE = _base(
    "mock-guest-low-001",
    "thread-guest-low",
    "New message from guest Sam",
    """Hi!

Just wanted to say thanks again for the quick check-in instructions. We're really looking forward to our stay next week.

Sam""",
    from_addr=FROM_AIRBNB_AUTOMATED,
    snippet="Just wanted to say thanks again...",
)
GUEST_MESSAGE_LOW_IMPORTANCE["_category"] = "guest_message"
GUEST_MESSAGE_LOW_IMPORTANCE["_importance"] = "low"

GUEST_MESSAGE_UNCERTAIN = _base(
    "mock-guest-mid-001",
    "thread-guest-mid",
    "New message from guest Jordan",
    """Hi, I had a question about early check-in options for my stay next week. Our flight lands at noon.

Jordan""",
    from_addr=FROM_AIRBNB_AUTOMATED,
    snippet="I had a question about early check-in options...",
)
GUEST_MESSAGE_UNCERTAIN["_category"] = "guest_message"
GUEST_MESSAGE_UNCERTAIN["_importance"] = "uncertain"


# ---------------------------------------------------------------------------
# leave_review_request
# ---------------------------------------------------------------------------

LEAVE_REVIEW_REQUEST = _base(
    "mock-leave-001",
    "thread-leave-001",
    "Leave a review for your guest Taylor",
    """Taylor's stay at your property has ended.

Please leave a review to help the Airbnb community. Your review helps other hosts know what to expect.

Rate your experience and share feedback about your guest.""",
    snippet="Taylor's stay has ended. Share your experience hosting them...",
)
LEAVE_REVIEW_REQUEST["_category"] = "leave_review_request"
LEAVE_REVIEW_REQUEST["_extracted_guest_name"] = "Taylor"


# ---------------------------------------------------------------------------
# new_property_review – good (4–5 stars)
# ---------------------------------------------------------------------------

NEW_PROPERTY_REVIEW_GOOD = _base(
    "mock-review-good-001",
    "thread-review-good",
    "New review from guest Casey: 5 stars",
    """Casey left a review of your property.

Rating: 5/5 stars

"Amazing place! Everything was perfect, from the cozy decor to the spotless kitchen. The host was incredibly responsive and helpful. Would definitely book again!"

You can respond to this review within 30 days.""",
    snippet="Casey left a 5-star review: Amazing place!...",
)
NEW_PROPERTY_REVIEW_GOOD["_category"] = "new_property_review"
NEW_PROPERTY_REVIEW_GOOD["_rating"] = 5
NEW_PROPERTY_REVIEW_GOOD["_extracted_guest_name"] = "Casey"

NEW_PROPERTY_REVIEW_NEUTRAL = _base(
    "mock-review-mid-001",
    "thread-review-mid",
    "New review from guest Morgan: 3 stars",
    """Morgan left a review of your property.

Rating: 3/5 stars

"Nice location and the place was clean. A few small things could be improved but overall fine stay."

You can respond to this review within 30 days.""",
    snippet="Morgan left a 3-star review...",
)
NEW_PROPERTY_REVIEW_NEUTRAL["_category"] = "new_property_review"
NEW_PROPERTY_REVIEW_NEUTRAL["_rating"] = 3
NEW_PROPERTY_REVIEW_NEUTRAL["_extracted_guest_name"] = "Morgan"


# ---------------------------------------------------------------------------
# new_property_review – bad (1–3 stars, triggers reply options)
# ---------------------------------------------------------------------------

NEW_PROPERTY_REVIEW_BAD = _base(
    "mock-review-bad-001",
    "thread-review-bad",
    "New review from guest Riley: 2 stars",
    """Riley left a review of your property.

Rating: 2/5 stars

"The place was not clean when we arrived. Found dirty towels in the bathroom and the kitchen had unwashed dishes. Location was okay but the cleanliness issues ruined our stay."

You can respond to this review within 30 days.""",
    snippet="Riley left a 2-star review. The place was not clean...",
)
NEW_PROPERTY_REVIEW_BAD["_category"] = "new_property_review"
NEW_PROPERTY_REVIEW_BAD["_rating"] = 2
NEW_PROPERTY_REVIEW_BAD["_extracted_guest_name"] = "Riley"


# ---------------------------------------------------------------------------
# unsupported_airbnb (Airbnb sender, no leave-review / new-review / guest-message signals)
# ---------------------------------------------------------------------------

UNSUPPORTED_AIRBNB = _base(
    "mock-unsupported-001",
    "thread-unsupported",
    "Your payout summary for February",
    """Your payout summary is ready.

View your earnings and transaction history in the Resolution Center.""",
    snippet="Your payout summary is ready...",
)
UNSUPPORTED_AIRBNB["_category"] = "unsupported_airbnb"


# ---------------------------------------------------------------------------
# non_airbnb (non-Airbnb sender)
# ---------------------------------------------------------------------------

NON_AIRBNB = _base(
    "mock-non-airbnb-001",
    "thread-external",
    "Weekly deals just for you!",
    "Check out this week's best deals on travel accessories and hosting supplies!",
    from_addr="promo@newsletter.com",
    snippet="Check out this week's best deals...",
)
NON_AIRBNB["_category"] = "non_airbnb"


# ---------------------------------------------------------------------------
# Export: one canonical mock per category (and key variants)
# ---------------------------------------------------------------------------

def get_mock_for_category(category: str, variant: str | None = None) -> dict[str, Any]:
    """Return the canonical mock dict for a category. Optional variant for guest_message / new_property_review."""
    if category == "guest_message":
        if variant == "high":
            return {k: v for k, v in GUEST_MESSAGE_HIGH_IMPORTANCE.items() if not k.startswith("_")}
        if variant == "low":
            return {k: v for k, v in GUEST_MESSAGE_LOW_IMPORTANCE.items() if not k.startswith("_")}
        return {k: v for k, v in GUEST_MESSAGE_UNCERTAIN.items() if not k.startswith("_")}
    if category == "leave_review_request":
        return {k: v for k, v in LEAVE_REVIEW_REQUEST.items() if not k.startswith("_")}
    if category == "new_property_review":
        if variant == "bad":
            return {k: v for k, v in NEW_PROPERTY_REVIEW_BAD.items() if not k.startswith("_")}
        if variant == "good":
            return {k: v for k, v in NEW_PROPERTY_REVIEW_GOOD.items() if not k.startswith("_")}
        return {k: v for k, v in NEW_PROPERTY_REVIEW_NEUTRAL.items() if not k.startswith("_")}
    if category == "unsupported_airbnb":
        return {k: v for k, v in UNSUPPORTED_AIRBNB.items() if not k.startswith("_")}
    if category == "non_airbnb":
        return {k: v for k, v in NON_AIRBNB.items() if not k.startswith("_")}
    raise ValueError(f"Unknown category: {category}")


def all_demo_mocks() -> list[dict[str, Any]]:
    """Return one mock per type for demo inbox (no _* keys)."""
    return [
        get_mock_for_category("guest_message", "uncertain"),
        get_mock_for_category("leave_review_request"),
        get_mock_for_category("new_property_review", "bad"),
        get_mock_for_category("new_property_review", "good"),
        get_mock_for_category("non_airbnb"),
    ]


# Structure summary for documentation
MAIL_STRUCTURE_FIELDS = [
    "id", "thread_id", "from", "to", "subject", "snippet", "body", "date", "labels",
]
OPTIONAL_FIELDS = ["message_id_header", "references"]
