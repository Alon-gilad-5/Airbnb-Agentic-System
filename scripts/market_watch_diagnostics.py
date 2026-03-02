"""Inspect Market Watch property scoping against a running backend.

Usage examples:
  py scripts/market_watch_diagnostics.py
  py scripts/market_watch_diagnostics.py --base-url http://127.0.0.1:8000
  py scripts/market_watch_diagnostics.py --base-url https://airbnb-agentic-system.onrender.com
  py scripts/market_watch_diagnostics.py --legacy-property-id 290761
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _fetch_json(url: str, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _alerts_url(base_url: str, *, owner_id: str | None, property_id: str | None, limit: int) -> str:
    params = urllib.parse.urlencode(
        {
            "limit": str(limit),
            "owner_id": owner_id or "",
            "property_id": property_id or "",
        }
    )
    return f"{base_url.rstrip('/')}/api/market_watch/alerts?{params}"


def _print_section(title: str) -> None:
    print(f"\n=== {title} ===")


def _summarize_alerts(name: str, property_id: str | None, data: dict[str, Any]) -> None:
    alerts = data.get("alerts") or []
    returned_property_ids = sorted({str(a.get("property_id")) for a in alerts if a.get("property_id") is not None})
    alert_types: dict[str, int] = {}
    for alert in alerts:
        key = str(alert.get("alert_type") or "unknown")
        alert_types[key] = alert_types.get(key, 0) + 1

    print(f"{name}: property_id={property_id or '<none>'}")
    print(f"  alert_count={len(alerts)}")
    print(f"  returned_property_ids={returned_property_ids or ['<none>']}")
    print(f"  alert_types={alert_types or {'<none>': 0}}")

    for idx, alert in enumerate(alerts[:3], start=1):
        print(
            "  sample_{}: property_id={} type={} title={!r} start_at_utc={}".format(
                idx,
                alert.get("property_id"),
                alert.get("alert_type"),
                alert.get("title"),
                alert.get("start_at_utc"),
            )
        )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Inspect Market Watch property scoping.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("APP_BASE_URL") or "http://127.0.0.1:8000",
        help="Backend base URL to inspect.",
    )
    parser.add_argument(
        "--legacy-property-id",
        default="290761",
        help="Optional legacy property ID to compare against.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max alerts to fetch per property.",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    print(f"Backend: {base_url}")
    print(f"CWD: {Path.cwd()}")

    try:
        profiles_data = _fetch_json(f"{base_url}/api/property_profiles")
    except urllib.error.URLError as exc:
        print(f"Failed to fetch /api/property_profiles: {exc}", file=sys.stderr)
        return 1

    profiles = profiles_data.get("profiles") or []
    default_profile_id = profiles_data.get("default_profile_id")

    _print_section("Property Profiles")
    print(f"default_profile_id={default_profile_id!r}")
    if not profiles:
        print("No property profiles returned.")
        return 1

    for profile in profiles:
        print(
            "profile_id={} property_id={} property_name={!r} owner_id={}".format(
                profile.get("profile_id"),
                profile.get("property_id"),
                profile.get("property_name"),
                profile.get("owner_id"),
            )
        )

    _print_section("Scoped Alert Checks")
    for profile in profiles:
        name = str(profile.get("profile_id") or "unknown-profile")
        property_id = profile.get("property_id")
        owner_id = profile.get("owner_id")
        url = _alerts_url(base_url, owner_id=owner_id, property_id=property_id, limit=args.limit)
        try:
            data = _fetch_json(url)
        except urllib.error.URLError as exc:
            print(f"{name}: failed to fetch alerts: {exc}")
            continue
        _summarize_alerts(name, property_id, data)

    if args.legacy_property_id:
        _print_section("Legacy Property Check")
        owner_id = None
        if profiles:
            owner_id = profiles[0].get("owner_id")
        url = _alerts_url(
            base_url,
            owner_id=owner_id,
            property_id=args.legacy_property_id,
            limit=args.limit,
        )
        try:
            data = _fetch_json(url)
            _summarize_alerts("legacy", args.legacy_property_id, data)
        except urllib.error.URLError as exc:
            print(f"legacy: failed to fetch alerts: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
