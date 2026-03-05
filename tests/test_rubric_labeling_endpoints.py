from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import app.main as main_module
from app.schemas import RubricLabelSaveRequest
from fastapi import HTTPException


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["case_id", "split", "grounding", "actionability", "tone_policy_safety", "notes"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_rubric_labeling_data_and_save(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / "outputs" / "tmp_test_rubric_labeling"
    reviews_cases = tmp_dir / "reviews_cases.jsonl"
    mail_cases = tmp_dir / "mail_cases.jsonl"
    reviews_csv = tmp_dir / "reviews_rubric.csv"
    mail_csv = tmp_dir / "mail_rubric.csv"
    results_summary = tmp_dir / "results_summary.json"

    _write_jsonl(
        reviews_cases,
        [
            {
                "case_id": "case_r_1",
                "split": "test",
                "prompt": "What do guests say about wifi?",
                "context": {"property_id": "p1"},
                "expected": {"should_answer": True},
                "tags": ["reviews", "wifi"],
            }
        ],
    )
    _write_jsonl(
        mail_cases,
        [
            {
                "case_id": "case_m_1",
                "split": "test",
                "prompt": "Process inbox",
                "context": {},
                "expected": {"category": "guest_message"},
                "tags": ["mail"],
            }
        ],
    )
    _write_csv(
        reviews_csv,
        [
            {
                "case_id": "case_r_1",
                "split": "test",
                "grounding": "",
                "actionability": "",
                "tone_policy_safety": "",
                "notes": "",
            }
        ],
    )
    _write_csv(
        mail_csv,
        [
            {
                "case_id": "case_m_1",
                "split": "test",
                "grounding": "1",
                "actionability": "1",
                "tone_policy_safety": "2",
                "notes": "pre-scored",
            }
        ],
    )
    results_summary.parent.mkdir(parents=True, exist_ok=True)
    results_summary.write_text(
        json.dumps(
            {
                "agents": {
                    "reviews": {
                        "case_results": [
                            {
                                "case_id": "case_r_1",
                                "split": "test",
                                "pass": True,
                                "failure_reason": None,
                                "metadata": {
                                    "predicted_answer": True,
                                    "selected_total": 3,
                                    "relevant_selected": 2,
                                },
                            }
                        ]
                    },
                    "mail": {
                        "case_results": [
                            {
                                "case_id": "case_m_1",
                                "split": "test",
                                "pass": True,
                                "failure_reason": None,
                                "metadata": {
                                    "response_text": "Drafted reply.",
                                    "draft_text": "Thanks for your message.",
                                },
                            }
                        ]
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(main_module, "RUBRIC_REVIEWS_CASES_PATH", reviews_cases)
    monkeypatch.setattr(main_module, "RUBRIC_MAIL_CASES_PATH", mail_cases)
    monkeypatch.setattr(main_module, "RUBRIC_REVIEWS_CSV_PATH", reviews_csv)
    monkeypatch.setattr(main_module, "RUBRIC_MAIL_CSV_PATH", mail_csv)
    monkeypatch.setattr(main_module, "EVAL_RESULTS_SUMMARY_PATH", results_summary)

    data = main_module.rubric_labeling_data(split="test", source="all")
    assert data.status == "ok"
    assert data.total_cases == 2
    reviews_case = next(case for case in data.cases if case.source == "reviews")
    mail_case = next(case for case in data.cases if case.source == "mail")
    assert reviews_case.result_preview is not None
    assert "Host-facing recommendation (offline draft):" in reviews_case.result_preview
    assert reviews_case.result_metadata is not None
    assert "generated_host_recommendation" in reviews_case.result_metadata
    assert mail_case.result_preview == "Thanks for your message."
    assert mail_case.scored is True

    save_response = main_module.rubric_labeling_save(
        RubricLabelSaveRequest(
            source="reviews",
            case_id="case_r_1",
            split="test",
            grounding=2,
            actionability=1,
            tone_policy_safety=2,
            notes="clear recommendation",
        )
    )
    assert save_response.status == "ok"
    assert save_response.scored is True

    with reviews_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["grounding"] == "2"
    assert rows[0]["actionability"] == "1"
    assert rows[0]["tone_policy_safety"] == "2"
    assert rows[0]["notes"] == "clear recommendation"


def test_rubric_labeling_save_returns_json_error_on_locked_csv(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tmp_dir = repo_root / "outputs" / "tmp_test_rubric_labeling_locked"
    reviews_cases = tmp_dir / "reviews_cases.jsonl"
    mail_cases = tmp_dir / "mail_cases.jsonl"
    reviews_csv = tmp_dir / "reviews_rubric.csv"
    mail_csv = tmp_dir / "mail_rubric.csv"
    results_summary = tmp_dir / "results_summary.json"

    _write_jsonl(
        reviews_cases,
        [
            {
                "case_id": "case_r_locked",
                "split": "test",
                "prompt": "Any noise complaints?",
                "context": {"property_id": "p1"},
                "expected": {"should_answer": True},
                "tags": ["reviews", "noise"],
            }
        ],
    )
    _write_jsonl(mail_cases, [])
    _write_csv(
        reviews_csv,
        [
            {
                "case_id": "case_r_locked",
                "split": "test",
                "grounding": "",
                "actionability": "",
                "tone_policy_safety": "",
                "notes": "",
            }
        ],
    )
    _write_csv(mail_csv, [])
    results_summary.parent.mkdir(parents=True, exist_ok=True)
    results_summary.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(main_module, "RUBRIC_REVIEWS_CASES_PATH", reviews_cases)
    monkeypatch.setattr(main_module, "RUBRIC_MAIL_CASES_PATH", mail_cases)
    monkeypatch.setattr(main_module, "RUBRIC_REVIEWS_CSV_PATH", reviews_csv)
    monkeypatch.setattr(main_module, "RUBRIC_MAIL_CSV_PATH", mail_csv)
    monkeypatch.setattr(main_module, "EVAL_RESULTS_SUMMARY_PATH", results_summary)

    def _raise_locked(*_args, **_kwargs):
        raise PermissionError("The process cannot access the file because it is being used by another process.")

    monkeypatch.setattr(main_module, "_write_rubric_rows", _raise_locked)

    with pytest.raises(HTTPException) as exc:
        main_module.rubric_labeling_save(
            RubricLabelSaveRequest(
                source="reviews",
                case_id="case_r_locked",
                split="test",
                grounding=1,
                actionability=1,
                tone_policy_safety=1,
                notes="",
            )
        )

    assert exc.value.status_code == 500
    assert "locked or not writable" in str(exc.value.detail)
