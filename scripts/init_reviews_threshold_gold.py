#!/usr/bin/env python3
"""Initialize manual gold-label CSV from exported labeling pool."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from file."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def build_parser() -> argparse.ArgumentParser:
    """CLI for gold CSV template initialization."""

    parser = argparse.ArgumentParser(description="Initialize manual gold CSV for threshold optimization.")
    parser.add_argument("--label-pool-path", default="outputs/reviews_threshold_label_pool.jsonl")
    parser.add_argument("--output-path", default="outputs/reviews_threshold_gold.csv")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing output file.",
    )
    return parser


def main() -> None:
    """Create CSV skeleton with required gold columns."""

    args = build_parser().parse_args()
    output_path = Path(args.output_path)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace.")

    rows = load_jsonl(Path(args.label_pool_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "should_answer", "relevant_vector_ids"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.get("case_id", ""),
                    "should_answer": "",
                    "relevant_vector_ids": "",
                }
            )
    print(f"[done] initialized gold CSV template: {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()

