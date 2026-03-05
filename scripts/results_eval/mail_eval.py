"""Offline evaluator for mail-agent classification and policy behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from app.agents.mail_agent import (
    CATEGORY_NON_AIRBNB,
    MailAgent,
    MailAgentConfig,
    _classify_email,
)
from app.schemas import StepLog
from app.services.gmail_service import EmailMessage, GmailService
from app.services.mail_mock_emails import get_mock_for_category
from scripts.results_eval.common import compute_reliability, compute_task_success_rate, parse_cases


MAIL_MODULE_NAMES = {
    "mail_agent.fetch_inbox",
    "mail_agent.airbnb_filter",
    "mail_agent.classify",
    "mail_agent.policy",
    "mail_agent.guest_message_policy",
    "mail_agent.leave_review_policy",
    "mail_agent.property_review_policy",
    "mail_agent.answer_generation",
    "mail_agent.guest_reply_generation",
    "mail_agent.leave_review_generation",
    "mail_agent.property_review_response_generation",
    "mail_agent.review_reply_option_generation",
    "mail_agent.push_fetch",
}


@dataclass
class _DummyChatService:
    is_available: bool = True
    model: str = "offline-eval"

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        return "Offline-eval drafted response."


class _FixedInboxGmailService(GmailService):
    def __init__(self, messages: list[EmailMessage]) -> None:
        super().__init__(
            enabled=True,
            gauth_path="__eval_gauth__.json",
            accounts_path="__eval_accounts__.json",
            gmail_client_id=None,
            gmail_client_secret=None,
            gmail_refresh_token=None,
        )
        self._messages = messages

    def list_unread_messages(self, max_results: int = 20) -> list[EmailMessage]:
        return list(self._messages[:max_results])


def _make_gmail() -> GmailService:
    return GmailService(
        enabled=True,
        gauth_path="__eval_gauth__.json",
        accounts_path="__eval_accounts__.json",
        gmail_client_id=None,
        gmail_client_secret=None,
        gmail_refresh_token=None,
    )


def _is_step_log(step: Any) -> bool:
    return isinstance(step, StepLog) and isinstance(step.module, str) and isinstance(step.prompt, dict) and isinstance(step.response, dict)


def _message_from_case(case: dict[str, Any]) -> EmailMessage:
    category = str(case.get("mock_category", "guest_message"))
    variant = case.get("mock_variant")
    raw = get_mock_for_category(category, variant)
    return GmailService._raw_to_message(raw)


def evaluate_mail(
    *,
    repo_root: Path,
    split: str,
) -> dict[str, Any]:
    """Evaluate mail-agent classification + policy actions on mock messages."""

    cases = parse_cases(repo_root / "eval" / "cases" / "mail_cases.jsonl", split=split)
    case_results: list[dict[str, Any]] = []
    classification_hits = 0
    classification_total = 0
    policy_hits = 0
    policy_total = 0

    for case in cases:
        started = time.perf_counter()
        exception = None
        contract_ok = False
        trace_ok = False
        passed = False
        failure_reason = "uninitialized"
        try:
            message = _message_from_case(case.raw)
            expected_category = str(case.expected.get("category", "")).strip()
            expected_action = case.expected.get("action")
            expected_requires_owner = case.expected.get("requires_owner")
            action_record: dict[str, Any] | None = None

            classifier_gmail = _make_gmail()
            predicted_class = _classify_email(message, classifier_gmail)
            classification_total += 1
            classification_ok = predicted_class.category == expected_category
            classification_hits += 1 if classification_ok else 0

            mode = str(case.raw.get("mode", "run_on_messages")).strip().lower()
            if mode == "run_with_action":
                agent = MailAgent(
                    gmail_service=_FixedInboxGmailService([message]),
                    chat_service=_DummyChatService(),
                    config=MailAgentConfig(bad_review_threshold=3, auto_send_enabled=False),
                )
                owner_action = case.raw.get("owner_action") if isinstance(case.raw.get("owner_action"), dict) else {}
                result = agent.run_with_action(case.prompt, owner_action=owner_action, context=case.context)
            else:
                agent = MailAgent(
                    gmail_service=_make_gmail(),
                    chat_service=_DummyChatService(),
                    config=MailAgentConfig(bad_review_threshold=3, auto_send_enabled=False),
                )
                result = agent.run_on_messages([message], prompt=case.prompt, context=case.context)

            contract_ok = (
                isinstance(result.response, str)
                and isinstance(result.steps, list)
                and all(_is_step_log(step) for step in result.steps)
            )
            modules = [step.module for step in result.steps]
            required_modules = set(case.expected.get("required_modules", ["mail_agent.answer_generation"]))
            trace_ok = set(modules).issubset(MAIL_MODULE_NAMES) and required_modules.issubset(set(modules))

            action_ok = True
            requires_owner_ok = True
            if expected_action is not None:
                policy_total += 1
                for item in result.mail_actions or []:
                    if str(item.get("email_id")) == str(message.id):
                        action_record = item
                        break
                if expected_action == "excluded":
                    action_ok = action_record is None
                else:
                    action_ok = action_record is not None and str(action_record.get("action")) == str(expected_action)
                    if action_record is not None and expected_requires_owner is not None:
                        requires_owner_ok = bool(action_record.get("requires_owner")) == bool(expected_requires_owner)
                policy_hits += 1 if (action_ok and requires_owner_ok) else 0

            passed = classification_ok and action_ok and requires_owner_ok
            failure_reason = None if passed else "classification_or_policy_mismatch"
        except Exception as exc:  # pragma: no cover - defensive guard
            exception = f"{type(exc).__name__}: {exc}"
            failure_reason = "exception"

        latency_ms = (time.perf_counter() - started) * 1000.0
        case_results.append(
            {
                "agent": "mail",
                "split": split,
                "case_id": case.case_id,
                "pass": passed,
                "failure_reason": failure_reason,
                "latency_ms": round(latency_ms, 3),
                "metadata": {
                    "contract_ok": contract_ok,
                    "step_trace_ok": trace_ok,
                    "exception": exception,
                    "response_text": (result.response if 'result' in locals() else None),
                    "mail_action": action_record,
                    "draft_text": (action_record.get("draft") if isinstance(action_record, dict) else None),
                    "predicted_category": (predicted_class.category if 'predicted_class' in locals() else None),
                },
            }
        )

    reliability = compute_reliability(case_results)
    classification_accuracy = (classification_hits / classification_total) if classification_total else 0.0
    policy_accuracy = (policy_hits / policy_total) if policy_total else 0.0
    task_success_rate = compute_task_success_rate(case_results)
    return {
        "agent": "mail",
        "split": split,
        "primary_metric_name": "classification_accuracy",
        "primary_metric": round(classification_accuracy, 4),
        "metrics": {
            "case_count": len(case_results),
            "classification_accuracy": round(classification_accuracy, 4),
            "primary_metric_hits": classification_hits,
            "primary_metric_total": classification_total,
            "policy_action_accuracy": round(policy_accuracy, 4),
            "task_success_rate": task_success_rate,
            "task_success_passes": sum(1 for row in case_results if bool(row.get("pass"))),
            "task_success_total": len(case_results),
        },
        "reliability": reliability,
        "task_success_rate": task_success_rate,
        "case_results": case_results,
    }
