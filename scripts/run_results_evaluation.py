#!/usr/bin/env python3
"""Run reproducible offline evaluation suite for Results-section reporting."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.results_eval.analyst_eval import evaluate_analyst
from scripts.results_eval.common import (
    parse_cases,
    summarize_rubric_scores,
    utc_now_iso,
    write_json,
)
from scripts.results_eval.mail_eval import evaluate_mail
from scripts.results_eval.market_watch_eval import evaluate_market_watch
from scripts.results_eval.pricing_eval import evaluate_pricing
from scripts.results_eval.reporting import build_results_markdown, write_per_agent_csv
from scripts.results_eval.reviews_eval import evaluate_reviews
from scripts.results_eval.rubric import ensure_rubric_template, load_rubric_rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run offline results evaluation suite.")
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--output-dir", default="outputs/eval")
    return parser


def _collect_case_rows_for_rubric(repo_root: Path, case_file: str) -> list[dict[str, Any]]:
    path = repo_root / "eval" / "cases" / case_file
    if not path.exists():
        return []
    return [
        {"case_id": case.case_id, "split": case.split}
        for case in parse_cases(path)
    ]


def main() -> None:
    args = _build_parser().parse_args()
    repo_root = ROOT
    output_dir = (repo_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    reviews = evaluate_reviews(repo_root=repo_root, split=args.split)
    analyst = evaluate_analyst(repo_root=repo_root, split=args.split)
    pricing = evaluate_pricing(repo_root=repo_root, split=args.split)
    market_watch = evaluate_market_watch(repo_root=repo_root, split=args.split)
    mail = evaluate_mail(repo_root=repo_root, split=args.split)

    review_rubric_path = repo_root / "eval" / "manual" / "reviews_rubric_scores.csv"
    mail_rubric_path = repo_root / "eval" / "manual" / "mail_rubric_scores.csv"
    ensure_rubric_template(
        path=review_rubric_path,
        rows=_collect_case_rows_for_rubric(repo_root, "reviews_cases.jsonl"),
    )
    ensure_rubric_template(
        path=mail_rubric_path,
        rows=_collect_case_rows_for_rubric(repo_root, "mail_cases.jsonl"),
    )

    rubric = {
        "reviews": summarize_rubric_scores(
            rubric_rows=load_rubric_rows(review_rubric_path),
            split=args.split,
        ),
        "mail": summarize_rubric_scores(
            rubric_rows=load_rubric_rows(mail_rubric_path),
            split=args.split,
        ),
    }

    agents = {
        "reviews": reviews,
        "analyst": analyst,
        "pricing": pricing,
        "market_watch": market_watch,
        "mail": mail,
    }

    overall = {
        "mean_task_success_rate": round(mean(float(item.get("task_success_rate", 0.0)) for item in agents.values()), 4),
        "agents_evaluated": len(agents),
    }
    summary = {
        "generated_at_utc": utc_now_iso(),
        "split": args.split,
        "agents": agents,
        "rubric": rubric,
        "overall": overall,
    }

    summary_path = output_dir / "results_summary.json"
    write_json(summary_path, summary)
    write_per_agent_csv(summary=summary, output_path=output_dir / "per_agent_metrics.csv")
    md = build_results_markdown(repo_root=repo_root, summary=summary)
    (output_dir / "results_tables.md").write_text(md, encoding="utf-8")

    print(
        "[done] results evaluation completed: "
        f"summary={summary_path} "
        f"agents={len(agents)} split={args.split}"
    )


if __name__ == "__main__":
    main()
