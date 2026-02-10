#!/usr/bin/env python3
"""Run API smoke tests for selected high/low review-count properties."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient

# Ensure imports work when running from repository root via scripts path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import app


def run_case(client: TestClient, *, label: str, region: str, property_id: str) -> dict[str, object]:
    """Execute one prompt and validate top-level response contract."""

    payload = {
        "prompt": "What do guests think about wifi reliability?",
        "property_id": property_id,
        "region": region,
        "max_scrape_reviews": 5,
    }
    response = client.post("/api/execute", json=payload)
    body = response.json()

    required_keys = {"status", "error", "response", "steps"}
    missing = sorted(required_keys - set(body.keys()))
    if response.status_code != 200 or missing:
        raise RuntimeError(
            f"{label} failed contract check: status={response.status_code}, missing_keys={missing}, body={body}"
        )

    modules = [step.get("module") for step in body.get("steps", [])]
    return {
        "label": label,
        "status_code": response.status_code,
        "status": body.get("status"),
        "response_preview": str(body.get("response", ""))[:180],
        "step_modules": modules,
    }


def main() -> None:
    sample_path = Path("outputs/sample_properties.json")
    if not sample_path.exists():
        raise SystemExit("Missing outputs/sample_properties.json. Run scripts/select_sample_properties.py first.")

    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    high = sample["high_review_property"]
    low = sample["low_review_property"]

    client = TestClient(app)
    results = [
        run_case(
            client,
            label="high_review_property",
            region=str(high["region"]),
            property_id=str(high["property_id"]),
        ),
        run_case(
            client,
            label="low_review_property",
            region=str(low["region"]),
            property_id=str(low["property_id"]),
        ),
    ]

    payload = {"results": results}
    out_path = Path("outputs/property_smoke_test_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
