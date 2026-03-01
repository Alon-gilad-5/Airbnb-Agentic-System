"""Utilities to build and serve the architecture diagram required by the course."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Use default font fallback to avoid platform-specific font dependency issues."""

    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def ensure_architecture_png(output_path: Path) -> None:
    """Generate a clear PNG architecture diagram for current multi-agent flows."""

    # Always regenerate so diagram stays in sync as architecture evolves.
    output_path.parent.mkdir(parents=True, exist_ok=True)

    width, height = 1760, 1120
    image = Image.new("RGB", (width, height), "#f4f7fb")
    draw = ImageDraw.Draw(image)

    title_font = _font(34)
    box_title_font = _font(24)
    text_font = _font(17)

    draw.text((32, 24), "Airbnb Business Agent Architecture", fill="#1f2f39", font=title_font)

    def box(x: int, y: int, w: int, h: int, title: str, subtitle: str) -> tuple[int, int]:
        draw.rounded_rectangle((x, y, x + w, y + h), radius=14, fill="#ffffff", outline="#cfd9e2", width=3)
        draw.text((x + 18, y + 16), title, fill="#0f4c81", font=box_title_font)
        draw.text((x + 18, y + 58), subtitle, fill="#395161", font=text_font)
        return x + w // 2, y + h // 2

    execute_api = box(
        60,
        110,
        330,
        140,
        "Input API",
        "POST /api/execute\n- prompt",
    )
    router = box(
        460,
        110,
        360,
        140,
        "router_agent",
        "Keyword intent routing\nreviews vs pricing vs market_watch vs analyst",
    )
    reviews = box(
        460,
        300,
        360,
        140,
        "reviews_agent",
        "retrieval + web fallback\nanswer + guardrails",
    )
    market_watch = box(
        460,
        500,
        360,
        140,
        "market_watch_agent",
        "weather/events/holidays\nsignal scoring + alerts",
    )
    analyst = box(
        460,
        700,
        360,
        140,
        "analyst_agent",
        "neighbor benchmarking\nstructured listing comparisons",
    )
    pricing = box(
        460,
        900,
        360,
        140,
        "pricing_agent",
        "comp-based price recommendation\nmarket signals + review volume",
    )
    execute_output = box(
        1040,
        260,
        560,
        140,
        "Output API",
        "status/error/response/steps\n(JSON)",
    )

    pinecone = box(
        60,
        320,
        330,
        140,
        "Pinecone Index",
        "internal reviews vectors\n+ web quarantine namespace",
    )
    llmod = box(
        60,
        520,
        330,
        140,
        "LLMOD Gateway",
        "Azure/OpenAI-compatible\nembeddings + chat",
    )
    market_apis = box(
        1040,
        500,
        560,
        140,
        "External Market APIs",
        "Open-Meteo + Ticketmaster\n+ Nager.Date holidays",
    )
    alerts_inbox = box(
        1040,
        680,
        560,
        140,
        "Market Alerts Inbox",
        "SQLite locally / Postgres on Vercel\nGET /api/market_watch/alerts",
    )
    supabase_listings = box(
        1040,
        870,
        560,
        140,
        "Supabase Listing Store",
        "large_dataset_table\nbenchmark + pricing source data",
    )
    cron = box(
        60,
        720,
        360,
        140,
        "Autonomous Scheduler",
        "internal thread OR external cron\nPOST /api/market_watch/run",
    )

    def arrow(a: tuple[int, int], b: tuple[int, int], color: str = "#4b6a7e") -> None:
        draw.line((a[0], a[1], b[0], b[1]), fill=color, width=4)
        # Arrowhead
        hx, hy = b
        draw.polygon([(hx, hy), (hx - 12, hy - 7), (hx - 12, hy + 7)], fill=color)

    arrow((execute_api[0] + 165, execute_api[1]), (router[0] - 180, router[1]))
    arrow((router[0], router[1] + 85), (reviews[0], reviews[1] - 85))
    arrow((router[0], router[1] + 100), (market_watch[0], market_watch[1] - 100))
    arrow((router[0] + 120, router[1] + 120), (analyst[0] + 10, analyst[1] - 120))
    arrow((router[0] + 150, router[1] + 120), (pricing[0] + 40, pricing[1] - 120))
    arrow((reviews[0] - 180, reviews[1]), (pinecone[0] + 165, pinecone[1]))
    arrow((reviews[0] - 180, reviews[1] + 20), (llmod[0] + 165, llmod[1]))
    arrow((market_watch[0] + 180, market_watch[1]), (market_apis[0] - 250, market_apis[1]))
    arrow((market_watch[0] + 180, market_watch[1] + 30), (alerts_inbox[0] - 250, alerts_inbox[1]))
    arrow((market_watch[0] + 180, market_watch[1] - 110), (execute_output[0] - 250, execute_output[1] + 10))
    arrow((reviews[0] + 180, reviews[1] - 40), (execute_output[0] - 250, execute_output[1] - 10))
    arrow((analyst[0] + 180, analyst[1]), (supabase_listings[0] - 280, supabase_listings[1]))
    arrow((analyst[0] + 180, analyst[1] - 20), (execute_output[0] - 280, execute_output[1] + 40))
    arrow((pricing[0] + 180, pricing[1] - 20), (supabase_listings[0] - 280, supabase_listings[1] + 10))
    arrow((pricing[0] + 180, pricing[1] - 55), (market_apis[0] - 280, market_apis[1] + 80))
    arrow((pricing[0] - 180, pricing[1] - 65), (llmod[0] + 165, llmod[1] + 60))
    arrow((pricing[0] + 180, pricing[1] - 85), (execute_output[0] - 280, execute_output[1] + 80))
    arrow((cron[0], cron[1] - 85), (market_watch[0], market_watch[1] + 85), color="#2d7f5f")
    arrow((cron[0] + 180, cron[1]), (alerts_inbox[0] - 250, alerts_inbox[1] + 20), color="#2d7f5f")

    image.save(output_path, format="PNG")
