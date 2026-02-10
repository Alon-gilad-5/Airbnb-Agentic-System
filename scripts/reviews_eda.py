#!/usr/bin/env python3
"""EDA for Airbnb regional review archives (*.reviews.gz)."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import cast


BINS = [
    ("1", lambda x: x == 1),
    ("2-5", lambda x: 2 <= x <= 5),
    ("6-20", lambda x: 6 <= x <= 20),
    ("21-50", lambda x: 21 <= x <= 50),
    ("51-100", lambda x: 51 <= x <= 100),
    ("101-250", lambda x: 101 <= x <= 250),
    ("251+", lambda x: x >= 251),
]


def pctl(sorted_values: list[int], p: float) -> int:
    if not sorted_values:
        return 0
    idx = int((len(sorted_values) - 1) * p)
    return sorted_values[idx]


def region_name_from_file(path: Path) -> str:
    return path.name.replace(" reviews.gz", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EDA on gzipped reviews CSV files.")
    parser.add_argument(
        "--input-dir",
        default="airbnb_reviews",
        help="Folder that contains '*reviews.gz' files",
    )
    parser.add_argument(
        "--output-dir",
        default="eda_output",
        help="Output folder for CSV/JSON summary files",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*reviews.gz"))
    if not files:
        raise SystemExit(f"No matching files found in: {input_dir}")

    region_stats: dict[str, dict[str, float | int | str]] = {}
    region_bins_count: dict[str, dict[str, int]] = {}
    region_bins_pct: dict[str, dict[str, float]] = {}
    region_duplicate_rows: dict[str, int] = {}

    global_unique_review_ids: set[str] = set()
    global_unique_reviewer_ids: set[str] = set()
    global_listing_regions: dict[str, set[str]] = defaultdict(set)
    total_reviews = 0

    for file_path in files:
        region = region_name_from_file(file_path)
        listing_counts: Counter[str] = Counter()
        review_ids: set[str] = set()
        reviewer_ids: set[str] = set()
        dates: list[str] = []
        review_rows = 0

        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"}
            if not reader.fieldnames or set(reader.fieldnames) != expected:
                raise SystemExit(f"Unexpected schema in {file_path}: {reader.fieldnames}")

            for input_row in reader:
                review_rows += 1
                listing_id = cast(str, input_row["listing_id"]).strip()
                review_id = cast(str, input_row["id"]).strip()
                reviewer_id = cast(str, input_row["reviewer_id"]).strip()
                review_date = cast(str, input_row["date"]).strip()

                listing_counts[listing_id] += 1
                review_ids.add(review_id)
                reviewer_ids.add(reviewer_id)
                dates.append(review_date)

                global_unique_review_ids.add(review_id)
                global_unique_reviewer_ids.add(reviewer_id)
                global_listing_regions[listing_id].add(region)
                total_reviews += 1

        property_count = len(listing_counts)
        per_property_counts = sorted(listing_counts.values())
        region_duplicate_rows[region] = review_rows - len(review_ids)
        top_sorted = sorted(per_property_counts, reverse=True)
        top_k = max(1, int(property_count * 0.10)) if property_count else 0
        top10_share = (
            (sum(top_sorted[:top_k]) / review_rows * 100.0) if review_rows and top_k else 0.0
        )

        stats = {
            "region": region,
            "properties": property_count,
            "reviews": review_rows,
            "reviewers": len(reviewer_ids),
            "avg_reviews_per_property": (review_rows / property_count) if property_count else 0.0,
            "median_reviews_per_property": float(median(per_property_counts)) if per_property_counts else 0.0,
            "p90_reviews_per_property": pctl(per_property_counts, 0.9),
            "max_reviews_per_property": max(per_property_counts) if per_property_counts else 0,
            "top10pct_properties_review_share_pct": top10_share,
            "min_date": min(dates) if dates else "",
            "max_date": max(dates) if dates else "",
        }
        region_stats[region] = stats

        bin_counts = {}
        bin_pcts = {}
        for label, cond in BINS:
            c = sum(1 for x in per_property_counts if cond(x))
            bin_counts[label] = c
            bin_pcts[label] = (c / property_count * 100.0) if property_count else 0.0
        region_bins_count[region] = bin_counts
        region_bins_pct[region] = bin_pcts

    total_properties = sum(int(v["properties"]) for v in region_stats.values())
    total_reviews_check = sum(int(v["reviews"]) for v in region_stats.values())
    for s in region_stats.values():
        s["property_share_pct"] = (float(s["properties"]) / total_properties * 100.0) if total_properties else 0.0
        s["review_share_pct"] = (float(s["reviews"]) / total_reviews_check * 100.0) if total_reviews_check else 0.0

    overall = {
        "regions": len(region_stats),
        "total_reviews": total_reviews,
        "unique_review_ids": len(global_unique_review_ids),
        "unique_properties_global": len(global_listing_regions),
        "unique_reviewers_global": len(global_unique_reviewer_ids),
        "listings_in_multiple_regions": sum(1 for v in global_listing_regions.values() if len(v) > 1),
        "duplicate_review_rows_total": total_reviews - len(global_unique_review_ids),
        "regions_with_duplicate_review_rows": {
            k: v for k, v in sorted(region_duplicate_rows.items()) if v > 0
        },
    }

    summary_csv = output_dir / "reviews_region_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "region",
            "properties",
            "reviews",
            "reviewers",
            "avg_reviews_per_property",
            "median_reviews_per_property",
            "p90_reviews_per_property",
            "max_reviews_per_property",
            "top10pct_properties_review_share_pct",
            "property_share_pct",
            "review_share_pct",
            "min_date",
            "max_date",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for summary_row in sorted(region_stats.values(), key=lambda x: float(x["properties"]), reverse=True):
            writer.writerow(summary_row)

    bins_count_csv = output_dir / "reviews_property_distribution_bins_count.csv"
    with bins_count_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["region", "properties"] + [label for label, _ in BINS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for region in sorted(region_bins_count.keys()):
            count_row = {"region": region, "properties": int(region_stats[region]["properties"])}
            count_row.update(region_bins_count[region])
            writer.writerow(count_row)

    bins_pct_csv = output_dir / "reviews_property_distribution_bins_pct.csv"
    with bins_pct_csv.open("w", encoding="utf-8", newline="") as f:
        fields = ["region"] + [label for label, _ in BINS]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for region in sorted(region_bins_pct.keys()):
            pct_row: dict[str, str | float] = {"region": region}
            pct_row.update(region_bins_pct[region])
            writer.writerow(pct_row)

    overall_json = output_dir / "reviews_overall_summary.json"
    overall_json.write_text(json.dumps(overall, indent=2), encoding="utf-8")

    print("Overall")
    print(json.dumps(overall, indent=2))
    print()
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {bins_count_csv}")
    print(f"Wrote: {bins_pct_csv}")
    print(f"Wrote: {overall_json}")


if __name__ == "__main__":
    main()
