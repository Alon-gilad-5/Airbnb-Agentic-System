"""One-time OAuth2 setup to obtain a Gmail refresh token for Render (or any headless env).

Run locally (browser will open). Copy the printed GMAIL_REFRESH_TOKEN into your
Render environment variables.

Usage:
    export GMAIL_CLIENT_ID="your-client-id"
    export GMAIL_CLIENT_SECRET="your-client-secret"
    python -m scripts.gmail_oauth_setup

Or pass as arguments:
    python -m scripts.gmail_oauth_setup --client-id ID --client-secret SECRET
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Obtain Gmail OAuth2 refresh token for use in Render env vars."
    )
    parser.add_argument(
        "--client-id",
        default=os.getenv("GMAIL_CLIENT_ID"),
        help="OAuth2 client ID (or set GMAIL_CLIENT_ID)",
    )
    parser.add_argument(
        "--client-secret",
        default=os.getenv("GMAIL_CLIENT_SECRET"),
        help="OAuth2 client secret (or set GMAIL_CLIENT_SECRET)",
    )
    args = parser.parse_args()

    client_id = (args.client_id or "").strip()
    client_secret = (args.client_secret or "").strip()
    if not client_id or not client_secret:
        print("Error: GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET are required.", file=sys.stderr)
        print("Set env vars or pass --client-id and --client-secret.", file=sys.stderr)
        sys.exit(1)

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(
            "Error: google-auth-oauthlib is required. Run: pip install google-auth-oauthlib",
            file=sys.stderr,
        )
        sys.exit(1)

    scopes = [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/gmail.modify",
    ]
    client_config = {
        "installed": {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uris": ["http://localhost:8080/"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, scopes=scopes)
    creds = flow.run_local_server(
        port=8080,
        access_type="offline",
        prompt="consent",
    )

    refresh_token = getattr(creds, "refresh_token", None)
    if not refresh_token:
        print("Error: No refresh_token in response. Ensure prompt=consent and try again.", file=sys.stderr)
        sys.exit(1)

    print("\nAdd this to your Render (or .env) environment variables:\n")
    print(f"GMAIL_REFRESH_TOKEN={refresh_token}\n")
    print("Keep GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET set as well.")
    print("In Google Cloud Console, add http://localhost:8080/ to OAuth redirect URIs if needed.\n")


if __name__ == "__main__":
    main()
