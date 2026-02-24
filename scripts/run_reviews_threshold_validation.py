#!/usr/bin/env python3
"""Run fixed-case validation smoke for tuned review relevance threshold."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

# Ensure imports work when running from repository root via scripts path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL rows."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                out.append(json.loads(stripped))
    return out


def pick_fixed_subset(cases: list[dict[str, Any]], subset_size: int) -> list[dict[str, Any]]:
    """Pick deterministic fixed subset by case_id order."""

    ordered = sorted(cases, key=lambda c: str(c.get("case_id", "")))
    return ordered[:subset_size]


def run_validation(
    *,
    cases: list[dict[str, Any]],
    namespace: str,
    threshold: float | None,
) -> dict[str, Any]:
    """Run execute endpoint calls with scraping disabled to isolate VDB behavior."""

    os.environ["DATABASE_URL"] = ""
    os.environ["MARKET_WATCH_ENABLED"] = "false"
    os.environ["SCRAPING_ENABLED"] = "false"
    os.environ["PINECONE_NAMESPACE"] = namespace
    if threshold is not None:
        os.environ["REVIEWS_RELEVANCE_SCORE_THRESHOLD"] = str(threshold)

    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    case_results: list[dict[str, Any]] = []
    for case in cases:
        payload = {
            "prompt": str(case["prompt"]),
            "property_id": str(case["property_id"]),
            "region": str(case.get("region", "")),
            "max_scrape_reviews": 5,
        }
        response = client.post("/api/execute", json=payload)
        body = response.json()
        steps = body.get("steps", [])
        retrieval = next((s for s in steps if s.get("module") == "reviews_agent.retrieval"), None)
        evidence = next((s for s in steps if s.get("module") == "reviews_agent.evidence_guard"), None)
        web_scrape = next((s for s in steps if s.get("module") == "reviews_agent.web_scrape"), None)
        case_results.append(
            {
                "case_id": case.get("case_id"),
                "status_code": response.status_code,
                "status": body.get("status"),
                "error": body.get("error"),
                "retrieval_match_count": (
                    retrieval.get("response", {}).get("match_count") if retrieval else None
                ),
                "evidence_decision": (
                    evidence.get("response", {}).get("decision") if evidence else None
                ),
                "web_scrape_status": (
                    web_scrape.get("response", {}).get("status") if web_scrape else None
                ),
                "response_preview": str(body.get("response") or "")[:220],
            }
        )

    ok_cases = [r for r in case_results if r["status"] == "ok" and r["status_code"] == 200]
    no_evidence_cases = [
        r
        for r in case_results
        if str(r.get("response_preview", "")).startswith("I couldn't find enough data")
    ]
    return {
        "summary": {
            "namespace": namespace,
            "threshold": threshold,
            "total_cases": len(case_results),
            "ok_cases": len(ok_cases),
            "error_cases": len(case_results) - len(ok_cases),
            "no_evidence_cases": len(no_evidence_cases),
        },
        "cases": case_results,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for fixed-case threshold validation smoke."""

    parser = argparse.ArgumentParser(description="Validate tuned relevance threshold on fixed benchmark subset.")
    parser.add_argument("--cases-path", default="outputs/reviews_threshold_cases.jsonl")
    parser.add_argument("--namespace", default="airbnb-reviews-test")
    parser.add_argument("--subset-size", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-path", default="outputs/reviews_threshold_validation.json")
    return parser


def main() -> None:
    """Run threshold validation and persist summary."""

    args = build_parser().parse_args()
    cases = load_jsonl(Path(args.cases_path))
    subset = pick_fixed_subset(cases, args.subset_size)
    result = run_validation(cases=subset, namespace=args.namespace, threshold=args.threshold)
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        "[done] threshold validation completed: "
        f"path={args.output_path} ok={result['summary']['ok_cases']}/{result['summary']['total_cases']}"
    )


if __name__ == "__main__":
    main()
