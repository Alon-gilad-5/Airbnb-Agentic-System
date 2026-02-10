#!/usr/bin/env python3
"""Generate SVG visualizations from EDA output files."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path


def format_int(value: int) -> str:
    return f"{value:,}"


def pct(value: float) -> str:
    return f"{value * 100:.4f}%"


def svg_wrap(width: int, height: int, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="chart">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#f5f8fb"/>'
        f"{body}</svg>"
    )


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.write_text(svg_wrap(width, height, body), encoding="utf-8")


def load_region_summary(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_bins_pct(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def color_scale(value: float, vmax: float) -> str:
    # light to dark teal scale
    ratio = 0.0 if vmax <= 0 else max(0.0, min(1.0, value / vmax))
    r0, g0, b0 = (232, 244, 246)
    r1, g1, b1 = (15, 118, 110)
    r = int(r0 + (r1 - r0) * ratio)
    g = int(g0 + (g1 - g0) * ratio)
    b = int(b0 + (b1 - b0) * ratio)
    return f"rgb({r},{g},{b})"


def chart_overall(overall: dict, output_path: Path) -> None:
    width, height = 1200, 760
    total_reviews = int(overall["total_reviews"])
    unique_review_ids = int(overall["unique_review_ids"])
    unique_properties = int(overall["unique_properties_global"])
    unique_reviewers = int(overall["unique_reviewers_global"])
    regions = int(overall["regions"])
    duplicate_rows = int(overall["duplicate_review_rows_total"])
    multi_region = int(overall["listings_in_multiple_regions"])
    duplicate_regions = dict(overall.get("regions_with_duplicate_review_rows", {}))

    unique_ratio = unique_review_ids / total_reviews if total_reviews else 0.0
    dup_ratio = duplicate_rows / total_reviews if total_reviews else 0.0
    multi_ratio = multi_region / unique_properties if unique_properties else 0.0

    cards = [
        ("Regions", format_int(regions)),
        ("Total Review Rows", format_int(total_reviews)),
        ("Unique Review IDs", format_int(unique_review_ids)),
        ("Unique Properties", format_int(unique_properties)),
        ("Unique Reviewers", format_int(unique_reviewers)),
        ("Duplicate Rows", format_int(duplicate_rows)),
    ]

    card_body = []
    x0, y0 = 40, 80
    card_w, card_h = 350, 95
    gap_x, gap_y = 25, 18
    for i, (label, value) in enumerate(cards):
        r = i // 3
        c = i % 3
        x = x0 + c * (card_w + gap_x)
        y = y0 + r * (card_h + gap_y)
        card_body.append(
            f'<rect x="{x}" y="{y}" width="{card_w}" height="{card_h}" rx="12" fill="#ffffff" stroke="#d7e2e9"/>'
            f'<text x="{x+18}" y="{y+35}" font-size="18" fill="#5d707d">{escape(label)}</text>'
            f'<text x="{x+18}" y="{y+72}" font-size="36" font-weight="700" fill="#1f2f39">{escape(value)}</text>'
        )

    bars = [
        ("Unique IDs / Review Rows", unique_ratio, f"{pct(unique_ratio)}"),
        ("Duplicate Rows / Review Rows", dup_ratio, f"{pct(dup_ratio)}"),
        ("Multi-Region Listings / Global Listings", multi_ratio, f"{pct(multi_ratio)}"),
    ]
    bar_body = []
    bar_x = 70
    bar_y_start = 330
    bar_w = 820
    bar_h = 22
    row_gap = 78
    for idx, (label, fraction, label_value) in enumerate(bars):
        y = bar_y_start + idx * row_gap
        fill_w = bar_w * max(0.0, min(1.0, fraction))
        bar_body.append(
            f'<text x="{bar_x}" y="{y-10}" font-size="20" fill="#22323d">{escape(label)}</text>'
            f'<text x="{bar_x+bar_w+20}" y="{y-10}" font-size="20" fill="#22323d">{escape(label_value)}</text>'
            f'<rect x="{bar_x}" y="{y}" width="{bar_w}" height="{bar_h}" rx="11" fill="#e6edf2" stroke="#cdd9e2"/>'
            f'<rect x="{bar_x}" y="{y}" width="{fill_w:.2f}" height="{bar_h}" rx="11" fill="#0f766e"/>'
        )

    table_rows = []
    table_x, table_y = 930, 330
    table_w = 230
    row_h = 36
    table_rows.append(
        f'<rect x="{table_x}" y="{table_y}" width="{table_w}" height="{row_h}" fill="#caece8" stroke="#b9ddd8"/>'
        f'<text x="{table_x+12}" y="{table_y+24}" font-size="16" fill="#14443f">Region</text>'
        f'<text x="{table_x+210}" y="{table_y+24}" text-anchor="end" font-size="16" fill="#14443f">Dup Rows</text>'
    )
    if duplicate_regions:
        for i, (region, count) in enumerate(sorted(duplicate_regions.items(), key=lambda kv: kv[1], reverse=True), start=1):
            y = table_y + i * row_h
            table_rows.append(
                f'<rect x="{table_x}" y="{y}" width="{table_w}" height="{row_h}" fill="#ffffff" stroke="#d7e2e9"/>'
                f'<text x="{table_x+12}" y="{y+24}" font-size="15" fill="#1f2f39">{escape(region)}</text>'
                f'<text x="{table_x+210}" y="{y+24}" text-anchor="end" font-size="15" fill="#1f2f39">{format_int(int(count))}</text>'
            )
    else:
        y = table_y + row_h
        table_rows.append(
            f'<rect x="{table_x}" y="{y}" width="{table_w}" height="{row_h}" fill="#ffffff" stroke="#d7e2e9"/>'
            f'<text x="{table_x+12}" y="{y+24}" font-size="15" fill="#1f2f39">None</text>'
        )

    body = (
        '<text x="40" y="46" font-size="34" font-weight="700" fill="#1f2f39">Overall Reviews Summary</text>'
        '<text x="40" y="70" font-size="16" fill="#637784">Source: reviews_overall_summary.json</text>'
        + "".join(card_body)
        + "".join(bar_body)
        + "".join(table_rows)
    )
    write_svg(output_path, width, height, body)


def chart_properties_by_region(rows: list[dict[str, str]], output_path: Path) -> None:
    width, height = 1300, 720
    sorted_rows = sorted(rows, key=lambda x: int(float(x["properties"])), reverse=True)
    max_value = max(int(float(r["properties"])) for r in sorted_rows) if sorted_rows else 1

    left = 240
    top = 90
    bar_h = 48
    gap = 20
    chart_w = 980

    parts = [
        '<text x="40" y="44" font-size="34" font-weight="700" fill="#1f2f39">Properties By Region</text>',
        '<text x="40" y="68" font-size="16" fill="#637784">Source: reviews_region_summary.csv</text>',
    ]

    for i, row in enumerate(sorted_rows):
        region = row["region"]
        value = int(float(row["properties"]))
        share = float(row["property_share_pct"])
        y = top + i * (bar_h + gap)
        w = (value / max_value) * chart_w if max_value else 0
        parts.append(
            f'<text x="30" y="{y+32}" font-size="21" fill="#1f2f39">{escape(region)}</text>'
            f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="9" fill="#e8eff4"/>'
            f'<rect x="{left}" y="{y}" width="{w:.2f}" height="{bar_h}" rx="9" fill="#0f766e"/>'
            f'<text x="{left+10}" y="{y+32}" font-size="19" fill="#ffffff">{format_int(value)}</text>'
            f'<text x="{left+chart_w+15}" y="{y+32}" font-size="19" fill="#344955">{share:.2f}%</text>'
        )
    write_svg(output_path, width, height, "".join(parts))


def chart_avg_reviews(rows: list[dict[str, str]], output_path: Path) -> None:
    width, height = 1300, 720
    sorted_rows = sorted(rows, key=lambda x: float(x["avg_reviews_per_property"]), reverse=True)
    max_value = max(float(r["avg_reviews_per_property"]) for r in sorted_rows) if sorted_rows else 1.0

    left = 240
    top = 90
    bar_h = 48
    gap = 20
    chart_w = 980

    parts = [
        '<text x="40" y="44" font-size="34" font-weight="700" fill="#1f2f39">Avg Reviews Per Property By Region</text>',
        '<text x="40" y="68" font-size="16" fill="#637784">Source: reviews_region_summary.csv</text>',
    ]

    for i, row in enumerate(sorted_rows):
        region = row["region"]
        value = float(row["avg_reviews_per_property"])
        y = top + i * (bar_h + gap)
        w = (value / max_value) * chart_w if max_value else 0
        parts.append(
            f'<text x="30" y="{y+32}" font-size="21" fill="#1f2f39">{escape(region)}</text>'
            f'<rect x="{left}" y="{y}" width="{chart_w}" height="{bar_h}" rx="9" fill="#e8eff4"/>'
            f'<rect x="{left}" y="{y}" width="{w:.2f}" height="{bar_h}" rx="9" fill="#144f84"/>'
            f'<text x="{left+10}" y="{y+32}" font-size="19" fill="#ffffff">{value:.2f}</text>'
        )
    write_svg(output_path, width, height, "".join(parts))


def chart_bin_heatmap(rows: list[dict[str, str]], output_path: Path) -> None:
    bins = ["1", "2-5", "6-20", "21-50", "51-100", "101-250", "251+"]
    width, height = 1280, 760
    left, top = 250, 140
    cell_w, cell_h = 130, 56

    vmax = 0.0
    for row in rows:
        for b in bins:
            vmax = max(vmax, float(row[b]))

    parts = [
        '<text x="40" y="44" font-size="34" font-weight="700" fill="#1f2f39">Property Review Count Distribution Heatmap (%)</text>',
        '<text x="40" y="68" font-size="16" fill="#637784">Source: reviews_property_distribution_bins_pct.csv</text>',
    ]

    for i, b in enumerate(bins):
        x = left + i * cell_w
        parts.append(f'<text x="{x+cell_w/2:.1f}" y="{top-20}" text-anchor="middle" font-size="18" fill="#22323d">{escape(b)}</text>')

    for r_i, row in enumerate(rows):
        y = top + r_i * cell_h
        parts.append(f'<text x="30" y="{y+36}" font-size="20" fill="#1f2f39">{escape(row["region"])}</text>')
        for c_i, b in enumerate(bins):
            x = left + c_i * cell_w
            val = float(row[b])
            fill = color_scale(val, vmax)
            text_color = "#ffffff" if val > (0.55 * vmax) else "#1f2f39"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w-4}" height="{cell_h-4}" rx="8" fill="{fill}" stroke="#d4e0e8"/>'
                f'<text x="{x+(cell_w-4)/2:.1f}" y="{y+34}" text-anchor="middle" font-size="17" fill="{text_color}">{val:.2f}</text>'
            )

    parts.append(
        '<text x="40" y="730" font-size="14" fill="#637784">'
        "Values are % of properties in each review-count bucket within a region."
        "</text>"
    )
    write_svg(output_path, width, height, "".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate SVG charts from EDA outputs.")
    parser.add_argument("--eda-dir", default="eda_output", help="Directory containing EDA CSV/JSON outputs")
    parser.add_argument("--out-dir", default="outputs", help="Directory to write SVG files")
    args = parser.parse_args()

    eda_dir = Path(args.eda_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_path = eda_dir / "reviews_overall_summary.json"
    region_path = eda_dir / "reviews_region_summary.csv"
    bins_path = eda_dir / "reviews_property_distribution_bins_pct.csv"

    overall = json.loads(overall_path.read_text(encoding="utf-8"))
    region_rows = load_region_summary(region_path)
    bins_rows = load_bins_pct(bins_path)

    chart_overall(overall, out_dir / "overall_summary.svg")
    chart_properties_by_region(region_rows, out_dir / "properties_by_region.svg")
    chart_avg_reviews(region_rows, out_dir / "avg_reviews_per_property_by_region.svg")
    chart_bin_heatmap(bins_rows, out_dir / "distribution_heatmap_by_region.svg")

    print(f"Wrote: {out_dir / 'overall_summary.svg'}")
    print(f"Wrote: {out_dir / 'properties_by_region.svg'}")
    print(f"Wrote: {out_dir / 'avg_reviews_per_property_by_region.svg'}")
    print(f"Wrote: {out_dir / 'distribution_heatmap_by_region.svg'}")


if __name__ == "__main__":
    main()
