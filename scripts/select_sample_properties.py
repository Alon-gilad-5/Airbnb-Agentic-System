#!/usr/bin/env python3
"""Select representative high/low review-count properties from local archives."""

from __future__ import annotations

import csv
import gzip
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    review_dir = Path("airbnb_reviews")
    files = sorted(review_dir.glob("*reviews.gz"))
    if not files:
        raise SystemExit("No review archives found")

    counts: list[dict[str, object]] = []
    for fp in files:
        region = fp.name.replace(" reviews.gz", "")
        per_listing = Counter()
        with gzip.open(fp, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                listing_id = row["listing_id"].strip()
                if listing_id:
                    per_listing[listing_id] += 1
        for listing_id, review_count in per_listing.items():
            counts.append(
                {
                    "region": region,
                    "property_id": listing_id,
                    "review_count": review_count,
                }
            )

    counts.sort(key=lambda x: int(x["review_count"]), reverse=True)
    high = counts[0]
    lows = [x for x in counts if int(x["review_count"]) == 1]
    same_region_low = [x for x in lows if x["region"] == high["region"]]
    low = same_region_low[0] if same_region_low else lows[0]

    output = {
        "high_review_property": high,
        "low_review_property": low,
    }
    out_path = Path("outputs/sample_properties.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

