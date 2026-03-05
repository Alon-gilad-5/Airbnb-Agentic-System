#!/usr/bin/env python3
"""Export flat CSV and markdown tables from results_summary.json."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.results_eval.common import load_json
from scripts.results_eval.reporting import build_results_markdown, write_per_agent_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export per-agent CSV and markdown results tables.")
    parser.add_argument("--summary-path", default="outputs/eval/results_summary.json")
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary_path = (ROOT / args.summary_path).resolve()
    output_dir = (ROOT / args.output_dir).resolve()
    summary = load_json(summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_per_agent_csv(summary=summary, output_path=output_dir / "per_agent_metrics.csv")
    markdown = build_results_markdown(repo_root=ROOT, summary=summary)
    (output_dir / "results_tables.md").write_text(markdown, encoding="utf-8")
    print(
        "[done] exported results tables: "
        f"summary={summary_path} output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()

