#!/usr/bin/env python3
"""Run the mail agent on canonical mock emails to see how it reacts.

Use this to verify agent behavior before sending real emails to your inbox.

  python scripts/run_mail_agent_on_mocks.py

Uses .env / load_settings() for ChatService (LLMOD_API_KEY). If not set,
draft text will be fallback strings. Gmail is always in demo mode for this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.agents.mail_agent import MailAgent, MailAgentConfig
from app.config import load_settings
from app.services.chat_service import ChatService
from app.services.gmail_service import GmailService
from app.services.mail_mock_emails import all_demo_mocks


def main() -> None:
    settings = load_settings()

    gmail_service = GmailService(
        enabled=True,
        gauth_path="__script_demo__.json",
        accounts_path="__script_demo__.json",
        airbnb_sender_domains=settings.mail_airbnb_sender_domains,
        gmail_client_id=None,
        gmail_client_secret=None,
        gmail_refresh_token=None,
    )

    chat_service = ChatService(
        api_key=settings.llmod_api_key,
        base_url=settings.base_url,
        model=settings.chat_model,
        max_output_tokens=settings.chat_max_output_tokens,
    )

    mail_agent = MailAgent(
        gmail_service=gmail_service,
        chat_service=chat_service,
        config=MailAgentConfig(
            bad_review_threshold=settings.mail_bad_review_threshold,
            max_inbox_fetch=settings.mail_max_inbox_fetch,
            auto_send_enabled=False,
        ),
    )

    raws = all_demo_mocks()
    messages = [GmailService._raw_to_message(r) for r in raws]

    print("Running mail agent on", len(messages), "mock emails (push-style run_on_messages)...")
    print()
    result = mail_agent.run_on_messages(messages)

    print("--- Summary ---")
    print(result.response)
    print()

    if result.mail_actions:
        print("--- Actions ---")
        for i, action in enumerate(result.mail_actions, 1):
            cat = action.get("category", "?")
            act = action.get("action", "?")
            req = action.get("requires_owner", False)
            draft_preview = ""
            if action.get("draft"):
                draft_preview = " draft=" + str(action["draft"])[:80] + ("..." if len(str(action["draft"])) > 80 else "")
            opts = ""
            if action.get("reply_options"):
                opts = f" reply_options={len(action['reply_options'])} styles"
            print(f"  {i}. [{cat}] {act} requires_owner={req}{draft_preview}{opts}")
        print()
        print("--- Full mail_actions (JSON) ---")
        print(json.dumps(result.mail_actions, indent=2, default=str))
    else:
        print("No mail_actions (e.g. only non-Airbnb mocks).")


if __name__ == "__main__":
    main()
    sys.exit(0)
