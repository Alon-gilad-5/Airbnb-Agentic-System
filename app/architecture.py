"""Utilities to build the architecture diagram as a crisp SVG asset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class BoxStyle:
    fill: str
    stroke: str
    accent: str
    title_fill: str = "#0f172a"
    body_fill: str = "#475569"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: tuple[str, ...]
    style: BoxStyle

    @property
    def left(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 2)

    @property
    def right(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 2)

    @property
    def top(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y)

    @property
    def bottom(self) -> tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h)

    @property
    def left_upper(self) -> tuple[int, int]:
        return (self.x, self.y + self.h // 3)

    @property
    def left_lower(self) -> tuple[int, int]:
        return (self.x, self.y + (2 * self.h) // 3)

    @property
    def right_upper(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + self.h // 3)

    @property
    def right_lower(self) -> tuple[int, int]:
        return (self.x + self.w, self.y + (2 * self.h) // 3)


SVG_WIDTH = 1340
SVG_HEIGHT = 1680

API_STYLE = BoxStyle(fill="#fff7ed", stroke="#fb923c", accent="#f59e0b", title_fill="#7c2d12")
AGENT_STYLE = BoxStyle(fill="#eff6ff", stroke="#60a5fa", accent="#2563eb")
INFRA_STYLE = BoxStyle(fill="#ffffff", stroke="#cbd5e1", accent="#94a3b8")
SCHED_STYLE = BoxStyle(fill="#ecfdf5", stroke="#4ade80", accent="#16a34a", title_fill="#14532d")
INTEGRATION_STYLE = BoxStyle(fill="#f0fdfa", stroke="#5eead4", accent="#0f766e", title_fill="#134e4a")
DATA_STYLE = BoxStyle(fill="#f8fafc", stroke="#cbd5e1", accent="#475569")


def _polyline(points: list[tuple[int, int]]) -> str:
    return " ".join(f"{x},{y}" for x, y in points)


def _text_block(*, x: int, y: int, title: str, lines: tuple[str, ...], style: BoxStyle) -> str:
    title_svg = (
        f'<text x="{x}" y="{y}" font-size="22" font-weight="700" fill="{style.title_fill}">'
        f"{escape(title)}</text>"
    )
    body_lines = []
    for idx, line in enumerate(lines):
        body_lines.append(
            f'<tspan x="{x}" dy="{0 if idx == 0 else 23}">{escape(line)}</tspan>'
        )
    body_svg = (
        f'<text x="{x}" y="{y + 34}" font-size="17" font-weight="500" fill="{style.body_fill}">'
        + "".join(body_lines)
        + "</text>"
    )
    return title_svg + body_svg


def _box_svg(box: Box) -> str:
    return f"""
    <g>
      <rect x="{box.x}" y="{box.y}" width="{box.w}" height="{box.h}" rx="24"
            fill="{box.style.fill}" stroke="{box.style.stroke}" stroke-width="2"
            filter="url(#softShadow)"/>
      <rect x="{box.x}" y="{box.y}" width="{box.w}" height="12" rx="24"
            fill="{box.style.accent}" opacity="0.95"/>
      <rect x="{box.x}" y="{box.y + 12}" width="{box.w}" height="18"
            fill="{box.style.accent}" opacity="0.08"/>
      {_text_block(x=box.x + 22, y=box.y + 46, title=box.title, lines=box.lines, style=box.style)}
    </g>
    """


def _panel(*, x: int, y: int, w: int, h: int, label: str, subtitle: str, accent: str) -> str:
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="32"
            fill="#ffffff" fill-opacity="0.62" stroke="#dbe4ee" stroke-width="1.5"/>
      <rect x="{x + 24}" y="{y + 24}" width="10" height="48" rx="5" fill="{accent}"/>
      <text x="{x + 50}" y="{y + 46}" font-size="15" font-weight="800" letter-spacing="2"
            fill="#334155">{escape(label)}</text>
      <text x="{x + 50}" y="{y + 72}" font-size="16" font-weight="500" fill="#64748b">
        {escape(subtitle)}
      </text>
    </g>
    """


def _route(points: list[tuple[int, int]], *, stroke: str, marker_id: str, width: int = 3) -> str:
    return (
        f'<polyline points="{_polyline(points)}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round" '
        f'marker-end="url(#{marker_id})"/>'
    )


def render_architecture_svg() -> str:
    """Return the architecture diagram SVG markup."""
    left_panel = (36, 214, 360, 1290)
    center_panel = (434, 214, 392, 1290)
    right_panel = (864, 214, 440, 1290)

    reviews_ui = Box(
        x=74,
        y=318,
        w=284,
        h=108,
        title="Reviews Console",
        lines=("POST /api/execute", "prompt + owner context"),
        style=API_STYLE,
    )
    market_ui = Box(
        x=74,
        y=538,
        w=284,
        h=108,
        title="Market Watch",
        lines=("POST /api/market_watch/run", "manual or autonomous run"),
        style=API_STYLE,
    )
    analysis_ui = Box(
        x=74,
        y=758,
        w=284,
        h=108,
        title="Analysis Console",
        lines=("POST /api/analysis", "comparison request + filters"),
        style=API_STYLE,
    )
    pricing_ui = Box(
        x=74,
        y=978,
        w=284,
        h=108,
        title="Pricing Console",
        lines=("POST /api/pricing", "property + horizon + mode"),
        style=API_STYLE,
    )
    mail_ui = Box(
        x=74,
        y=1198,
        w=284,
        h=108,
        title="Mail Console",
        lines=("/api/mail/* + SSE", "owner review + live updates"),
        style=API_STYLE,
    )

    reviews = Box(
        x=480,
        y=312,
        w=300,
        h=120,
        title="reviews_agent",
        lines=("review Q&A", "retrieval + evidence guardrails"),
        style=AGENT_STYLE,
    )
    market = Box(
        x=480,
        y=532,
        w=300,
        h=120,
        title="market_watch_agent",
        lines=("demand signal scan", "weather / events / holidays"),
        style=AGENT_STYLE,
    )
    analyst = Box(
        x=480,
        y=752,
        w=300,
        h=120,
        title="analyst_agent",
        lines=("neighbor benchmarking", "structured metric comparison"),
        style=AGENT_STYLE,
    )
    pricing = Box(
        x=480,
        y=972,
        w=300,
        h=120,
        title="pricing_agent",
        lines=("nightly rate recommendation", "comp + market signal blend"),
        style=AGENT_STYLE,
    )
    mail = Box(
        x=480,
        y=1192,
        w=300,
        h=120,
        title="mail_agent",
        lines=("inbox triage + drafts", "human-in-the-loop send flow"),
        style=AGENT_STYLE,
    )

    pinecone = Box(
        x=918,
        y=312,
        w=328,
        h=120,
        title="Pinecone",
        lines=("review embeddings", "web quarantine namespace"),
        style=DATA_STYLE,
    )
    market_apis = Box(
        x=918,
        y=532,
        w=328,
        h=120,
        title="External Market APIs",
        lines=("Open-Meteo + Ticketmaster", "Nager.Date holidays"),
        style=INTEGRATION_STYLE,
    )
    supabase = Box(
        x=918,
        y=752,
        w=328,
        h=120,
        title="Supabase Listings",
        lines=("large_dataset_table", "benchmark + pricing inputs"),
        style=DATA_STYLE,
    )
    gmail = Box(
        x=918,
        y=1192,
        w=328,
        h=120,
        title="Gmail API",
        lines=("OAuth2 inbox fetch + send", "push notifications"),
        style=INTEGRATION_STYLE,
    )
    llm = Box(
        x=480,
        y=1384,
        w=300,
        h=120,
        title="LLM Gateway",
        lines=("Azure OpenAI compatible", "shared by reviews / analysis / pricing / mail"),
        style=INFRA_STYLE,
    )
    automation = Box(
        x=918,
        y=1384,
        w=328,
        h=120,
        title="Automation & Outputs",
        lines=("scheduler, alerts inbox, SSE", "traceable JSON response steps"),
        style=SCHED_STYLE,
    )

    ui_boxes = (reviews_ui, market_ui, analysis_ui, pricing_ui, mail_ui)
    agent_boxes = (reviews, market, analyst, pricing, mail)

    control_routes = [
        _route([reviews_ui.right, reviews.left], stroke="#2563eb", marker_id="arrowControl", width=4),
        _route([market_ui.right, market.left], stroke="#2563eb", marker_id="arrowControl", width=4),
        _route([analysis_ui.right, analyst.left], stroke="#2563eb", marker_id="arrowControl", width=4),
        _route([pricing_ui.right, pricing.left], stroke="#2563eb", marker_id="arrowControl", width=4),
        _route([mail_ui.right, mail.left], stroke="#2563eb", marker_id="arrowControl", width=4),
    ]

    data_routes = [
        _route([reviews.right, pinecone.left], stroke="#64748b", marker_id="arrowData"),
        _route([market.right, market_apis.left], stroke="#64748b", marker_id="arrowData"),
        _route([analyst.right, supabase.left], stroke="#64748b", marker_id="arrowData"),
        _route([pricing.right_upper, supabase.left_lower], stroke="#64748b", marker_id="arrowData"),
        _route(
            [pricing.right_lower, (840, pricing.right_lower[1]), (840, market_apis.left_lower[1]), market_apis.left_lower],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route([mail.right, gmail.left], stroke="#64748b", marker_id="arrowData"),
        _route(
            [reviews.bottom, (reviews.bottom[0], 1352), (llm.top[0] - 46, 1352), (llm.top[0] - 46, llm.top[1]), llm.top],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route([analyst.bottom, (analyst.bottom[0], llm.top[1]), llm.top], stroke="#64748b", marker_id="arrowData"),
        _route(
            [pricing.bottom, (pricing.bottom[0], llm.top[1]), (llm.top[0] + 44, llm.top[1]), llm.top],
            stroke="#64748b",
            marker_id="arrowData",
        ),
        _route(
            [mail.bottom, (mail.bottom[0], 1352), (llm.top[0] + 88, 1352), (llm.top[0] + 88, llm.top[1]), llm.top],
            stroke="#64748b",
            marker_id="arrowData",
        ),
    ]

    automation_routes = [
        _route(
            [market.bottom, (market.bottom[0], 1352), (automation.left[0] + 72, 1352), (automation.left[0] + 72, automation.top[1]), automation.top],
            stroke="#16a34a",
            marker_id="arrowSchedule",
            width=4,
        ),
        _route(
            [mail.right_lower, (840, mail.right_lower[1]), (840, automation.left_lower[1]), automation.left_lower],
            stroke="#16a34a",
            marker_id="arrowSchedule",
            width=4,
        ),
    ]

    boxes_svg = "".join(_box_svg(box) for box in (
        *ui_boxes,
        *agent_boxes,
        pinecone,
        market_apis,
        supabase,
        gmail,
        llm,
        automation,
    ))

    panels_svg = (
        _panel(
            x=left_panel[0],
            y=left_panel[1],
            w=left_panel[2],
            h=left_panel[3],
            label="INTERFACES",
            subtitle="Each page calls its own endpoint directly",
            accent="#f59e0b",
        )
        + _panel(
            x=center_panel[0],
            y=center_panel[1],
            w=center_panel[2],
            h=center_panel[3],
            label="DOMAIN AGENTS",
            subtitle="Five focused agents; no central router",
            accent="#2563eb",
        )
        + _panel(
            x=right_panel[0],
            y=right_panel[1],
            w=right_panel[2],
            h=left_panel[3],
            label="DATA & OPERATIONS",
            subtitle="Domain data, shared services, and outbound channels",
            accent="#0f766e",
        )
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{SVG_WIDTH}" height="{SVG_HEIGHT}"
viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" role="img" aria-labelledby="title desc">
  <title id="title">Airbnb Business Agent system architecture</title>
  <desc id="desc">A direct-endpoint multi-agent architecture where each console calls its dedicated agent, with shared LLM infrastructure and domain-specific data integrations.</desc>
  <defs>
    <linearGradient id="bgGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#f8fbff"/>
      <stop offset="55%" stop-color="#f7f7fb"/>
      <stop offset="100%" stop-color="#eef6ff"/>
    </linearGradient>
    <linearGradient id="haloGradient" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#dbeafe" stop-opacity="0.76"/>
      <stop offset="100%" stop-color="#dbeafe" stop-opacity="0"/>
    </linearGradient>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="14" stdDeviation="18" flood-color="#94a3b8" flood-opacity="0.18"/>
    </filter>
    <marker id="arrowControl" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#2563eb"/>
    </marker>
    <marker id="arrowData" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/>
    </marker>
    <marker id="arrowSchedule" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#16a34a"/>
    </marker>
    <pattern id="dotGrid" width="28" height="28" patternUnits="userSpaceOnUse">
      <circle cx="2" cy="2" r="1.3" fill="#cbd5e1" opacity="0.24"/>
    </pattern>
  </defs>

  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="url(#bgGradient)"/>
  <rect width="{SVG_WIDTH}" height="{SVG_HEIGHT}" fill="url(#dotGrid)"/>
  <ellipse cx="980" cy="170" rx="360" ry="150" fill="url(#haloGradient)" opacity="0.92"/>

  <g>
    <text x="52" y="74" font-size="38" font-weight="800" fill="#0f172a">Airbnb Business Agent</text>
    <text x="52" y="114" font-size="20" font-weight="500" fill="#475569">
      Direct endpoint flow: page -> agent -> data and operations
    </text>
  </g>

  <g transform="translate(52 140)">
    <rect x="0" y="0" width="640" height="46" rx="18" fill="#ffffff" fill-opacity="0.78" stroke="#dbe4ee"/>
    <circle cx="20" cy="23" r="6" fill="#2563eb"/>
    <text x="36" y="28" font-size="17" font-weight="600" fill="#334155">No router layer. Each console owns its endpoint and talks to one domain agent.</text>
  </g>

  <g transform="translate(932 138)">
    <rect x="0" y="0" width="336" height="62" rx="20" fill="#ffffff" fill-opacity="0.76" stroke="#dbe4ee"/>
    <line x1="22" y1="22" x2="74" y2="22" stroke="#2563eb" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowControl)"/>
    <text x="88" y="27" font-size="15" font-weight="500" fill="#475569">direct request path</text>
    <line x1="22" y1="42" x2="74" y2="42" stroke="#64748b" stroke-width="4" stroke-linecap="round" marker-end="url(#arrowData)"/>
    <text x="88" y="47" font-size="15" font-weight="500" fill="#475569">data dependency / shared service</text>
  </g>

  {panels_svg}

  <g opacity="0.98">
    {"".join(control_routes)}
    {"".join(data_routes)}
    {"".join(automation_routes)}
  </g>

  {boxes_svg}
</svg>
"""


def ensure_architecture_svg(output_path: Path) -> None:
    """Write the current architecture diagram to an SVG file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_architecture_svg(), encoding="utf-8")
