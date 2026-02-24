#!/usr/bin/env python3
"""Build a deterministic prompt-case benchmark for threshold calibration."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REGION_FIXES = {
    "los angels": "los angeles",
    "okland": "oakland",
    "san fransisco": "san francisco",
}

PROMPT_TEMPLATES = [
    ("cleanliness", "What do guests say about cleanliness at this property?"),
    ("noise", "What do guests say about noise levels at this property?"),
    ("wifi", "What do guests say about wifi reliability at this property?"),
    ("checkin_host", "What do guests say about check-in and host communication?"),
    ("value_for_money", "What do guests say about value for money?"),
]


@dataclass(frozen=True)
class PropertyStats:
    """Property-level stats used for deterministic tiered benchmark sampling."""

    property_id: str
    region: str
    review_count: int


def canonical_region(raw_region: str) -> str:
    """Normalize known misspellings to stable region names."""

    lowered = raw_region.strip().lower()
    return REGION_FIXES.get(lowered, lowered)


def iter_review_files(reviews_dir: Path) -> list[Path]:
    """Return deterministic list of review archives."""

    files = sorted(reviews_dir.glob("*reviews.gz"))
    if not files:
        raise FileNotFoundError(f"No '*reviews.gz' files found in: {reviews_dir}")
    return files


def load_selection_state_property_ids(
    *,
    selection_state_path: Path,
    expected_namespace: str,
) -> set[str] | None:
    """Load selected property IDs from ingestion state (namespace-guarded)."""

    if not selection_state_path.exists():
        return None
    state = json.loads(selection_state_path.read_text(encoding="utf-8"))
    state_ns = str(state.get("namespace", "")).strip()
    if state_ns and state_ns != expected_namespace:
        raise ValueError(
            f"Selection-state namespace mismatch: file has '{state_ns}', expected '{expected_namespace}'."
        )
    raw_ids = state.get("selected_property_ids", [])
    if not isinstance(raw_ids, list):
        return None
    return {str(v).strip() for v in raw_ids if str(v).strip()}


def collect_property_stats(
    *,
    reviews_dir: Path,
    allowed_property_ids: set[str] | None,
) -> list[PropertyStats]:
    """Aggregate review counts per property from local archives."""

    counts: Counter[str] = Counter()
    region_by_property: dict[str, str] = {}
    for file_path in iter_review_files(reviews_dir):
        region = canonical_region(file_path.name.replace(" reviews.gz", ""))
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                property_id = (row.get("listing_id") or "").strip()
                if not property_id:
                    continue
                if allowed_property_ids is not None and property_id not in allowed_property_ids:
                    continue
                counts[property_id] += 1
                region_by_property.setdefault(property_id, region)

    out = [
        PropertyStats(
            property_id=property_id,
            region=region_by_property.get(property_id, "unknown"),
            review_count=review_count,
        )
        for property_id, review_count in counts.items()
    ]
    out.sort(key=lambda x: (-x.review_count, x.property_id))
    return out


def split_high_low_tiers(properties: list[PropertyStats]) -> tuple[list[PropertyStats], list[PropertyStats]]:
    """Split sorted properties into high and low halves by review count rank."""

    if len(properties) < 2:
        raise ValueError("Need at least two properties to build high/low benchmark tiers.")
    midpoint = len(properties) // 2
    high = properties[:midpoint]
    low = properties[midpoint:]
    if not high or not low:
        raise ValueError("Unable to split properties into high/low tiers.")
    return high, low


def choose_tier_property_topic_pairs(
    *,
    pool: list[PropertyStats],
    case_count: int,
    rng: random.Random,
) -> list[tuple[PropertyStats, str, str]]:
    """Choose unique (property, topic) pairs for one tier without replacement."""

    if not pool:
        raise ValueError("Case property pool is empty.")
    if case_count <= 0:
        return []

    pair_pool: list[tuple[PropertyStats, str, str]] = []
    for prop in pool:
        for topic, prompt in PROMPT_TEMPLATES:
            pair_pool.append((prop, topic, prompt))

    if case_count > len(pair_pool):
        raise ValueError(
            f"Requested {case_count} cases from pool with only {len(pair_pool)} unique property-topic pairs."
        )
    return rng.sample(pair_pool, case_count)


def build_benchmark_cases(
    *,
    properties: list[PropertyStats],
    total_cases: int,
    high_cases: int,
    low_cases: int,
    seed: int,
) -> list[dict[str, object]]:
    """Build deterministic benchmark cases balanced by high/low review tiers."""

    if high_cases + low_cases != total_cases:
        raise ValueError("high_cases + low_cases must equal total_cases.")
    rng = random.Random(seed)
    high_pool, low_pool = split_high_low_tiers(properties)
    high_choices = choose_tier_property_topic_pairs(pool=high_pool, case_count=high_cases, rng=rng)
    low_choices = choose_tier_property_topic_pairs(pool=low_pool, case_count=low_cases, rng=rng)

    rows: list[tuple[str, PropertyStats, str, str]] = [("high", p, t, prompt) for p, t, prompt in high_choices] + [
        ("low", p, t, prompt) for p, t, prompt in low_choices
    ]
    rng.shuffle(rows)

    cases: list[dict[str, object]] = []
    for idx, (tier, prop, topic, prompt) in enumerate(rows, start=1):
        cases.append(
            {
                "case_id": f"case_{idx:03d}",
                "property_id": prop.property_id,
                "region": prop.region,
                "prompt": prompt,
                "topic": topic,
                "tier": tier,
                "property_review_count": prop.review_count,
            }
        )
    return cases


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL in deterministic order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def build_parser() -> argparse.ArgumentParser:
    """CLI for deterministic benchmark generation."""

    parser = argparse.ArgumentParser(description="Build deterministic benchmark cases for threshold calibration.")
    parser.add_argument("--reviews-dir", default="airbnb_reviews")
    parser.add_argument("--selection-state-path", default="ingest_state/test_property_selection.json")
    parser.add_argument("--namespace", default="airbnb-reviews-test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-cases", type=int, default=60)
    parser.add_argument("--high-cases", type=int, default=30)
    parser.add_argument("--low-cases", type=int, default=30)
    parser.add_argument("--output-path", default="outputs/reviews_threshold_cases.jsonl")
    return parser


def main() -> None:
    """Generate and persist benchmark cases."""

    args = build_parser().parse_args()
    selected_ids = load_selection_state_property_ids(
        selection_state_path=Path(args.selection_state_path),
        expected_namespace=args.namespace,
    )
    properties = collect_property_stats(
        reviews_dir=Path(args.reviews_dir),
        allowed_property_ids=selected_ids,
    )
    cases = build_benchmark_cases(
        properties=properties,
        total_cases=args.total_cases,
        high_cases=args.high_cases,
        low_cases=args.low_cases,
        seed=args.seed,
    )
    write_jsonl(Path(args.output_path), cases)
    print(
        "[done] wrote benchmark cases: "
        f"path={args.output_path} total={len(cases)} "
        f"high={sum(1 for c in cases if c['tier'] == 'high')} "
        f"low={sum(1 for c in cases if c['tier'] == 'low')}"
    )


if __name__ == "__main__":
    main()
