"""Competitive analysis agent over structured listing data."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from statistics import mean
from typing import Any, Literal

from app.agents.base import Agent, AgentResult
from app.schemas import (
    AnalysisCategoryBucket,
    AnalysisCategoricalItem,
    AnalysisNeighborPoint,
    AnalysisNumericItem,
    StepLog,
)
from app.services.chat_service import ChatService
from app.services.listing_store import (
    PROPERTY_SPEC_CATEGORICAL_COLUMNS,
    PROPERTY_SPEC_COLUMNS,
    PROPERTY_SPEC_NUMERIC_COLUMNS,
    REVIEW_SCORE_COLUMNS,
)

ANALYSIS_CATEGORIES = {"review_scores", "property_specs"}
COLUMN_LABELS = {
    "review_scores_rating": "Overall Rating",
    "review_scores_accuracy": "Accuracy",
    "review_scores_cleanliness": "Cleanliness",
    "review_scores_checkin": "Check-in",
    "review_scores_communication": "Communication",
    "review_scores_location": "Location",
    "review_scores_value": "Value",
    "accommodates": "Accommodates",
    "bathrooms": "Bathrooms",
    "bedrooms": "Bedrooms",
    "beds": "Beds",
    "price": "Nightly Price",
    "property_type": "Property Type",
    "room_type": "Room Type",
    "host_is_superhost": "Host Superhost",
}


@dataclass
class AnalystRunOutcome:
    """Detailed analyst-agent output used by the analysis endpoint."""

    analysis_category: Literal["review_scores", "property_specs"]
    narrative: str
    error: str | None
    numeric_comparison: list[AnalysisNumericItem]
    categorical_comparison: list[AnalysisCategoricalItem]
    steps: list[StepLog]


def _context_str(context: dict[str, object], key: str) -> str | None:
    value = context.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if math.isclose(value, round(value), rel_tol=0.0, abs_tol=1e-9):
        return str(int(round(value)))
    return f"{value:.2f}"


def _column_label(column: str) -> str:
    return COLUMN_LABELS.get(column, column.replace("_", " ").title())


def _normalize_point_value(value: float | int) -> float:
    return float(value)


class AnalystAgent(Agent):
    """Benchmarks one property against its configured neighbor set."""

    name = "analyst_agent"

    _analyst_keywords = {
        "benchmark",
        "competitive analysis",
        "analyze my scores",
        "analyse my scores",
        "analyze my property specs",
        "analyse my property specs",
        "compare my property specs",
        "compare my review scores",
        "how do i compare to neighbors",
        "how do i benchmark",
    }

    def __init__(
        self,
        *,
        listing_store: Any | None,
        neighbor_store: Any | None,
        chat_service: ChatService,
    ) -> None:
        self._listing_store = listing_store
        self._neighbor_store = neighbor_store
        self._chat = chat_service

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Execute analysis and return the narrative via the base agent contract."""

        outcome = self.analyze(prompt, context=context)
        return AgentResult(response=outcome.narrative, steps=outcome.steps)

    def explain_selection(
        self,
        *,
        prompt: str,
        property_id: str,
        category: Literal["review_scores", "property_specs"],
        selection_type: Literal["numeric_point", "numeric_extreme", "categorical_bucket"],
        metric_column: str,
        selection_payload: dict[str, Any],
    ) -> AgentResult:
        """Explain one selected visualization item without recomputing the full analysis."""

        label = _column_label(metric_column)
        system_prompt = (
            "You are a hospitality benchmarking analyst. "
            "Explain one selected item from an interactive competitor benchmarking visualization. "
            "Use only the supplied facts. Do not invent data. "
            "Focus on what the selection means and why it matters."
        )
        selection_summary = self._selection_payload_to_text(
            selection_type=selection_type,
            metric_column=metric_column,
            selection_payload=selection_payload,
        )
        user_prompt = (
            f"Original user request: {prompt}\n"
            f"Property ID: {property_id}\n"
            f"Analysis category: {category}\n"
            f"Selected metric: {label} ({metric_column})\n"
            f"Selection type: {selection_type}\n"
            f"Selection data:\n{selection_summary}\n\n"
            "Return at most 2 short paragraphs. Explain what the selection shows and what action it suggests."
        )
        try:
            if not self._chat.is_available:
                raise RuntimeError("Chat service unavailable")
            text = self._chat.generate(system_prompt=system_prompt, user_prompt=user_prompt).strip()
            if not text:
                raise RuntimeError("Empty explanation")
            response_payload: dict[str, Any] = {"text": text}
        except Exception as exc:
            text = self._deterministic_selection_explanation(
                selection_type=selection_type,
                metric_column=metric_column,
                selection_payload=selection_payload,
            )
            response_payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "fallback_used": True,
                "text": text,
            }

        steps = [
            StepLog(
                module="analyst_agent.selection_explanation",
                prompt={
                    "model": (self._chat.model if self._chat.is_available else None),
                    "selection_type": selection_type,
                    "metric_column": metric_column,
                    "selection_payload": selection_payload,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                },
                response=response_payload,
            )
        ]
        return AgentResult(response=text, steps=steps)

    def analyze(self, prompt: str, context: dict[str, object] | None = None) -> AnalystRunOutcome:
        """Execute structured analysis and return full comparison artifacts."""

        ctx = context or {}
        property_id = _context_str(ctx, "property_id")
        category = self._resolve_category(prompt, ctx)
        steps: list[StepLog] = []

        if not property_id:
            return self._error_outcome(
                "Property ID is required for competitive analysis.",
                steps,
                category=category,
            )
        if self._neighbor_store is None:
            return self._error_outcome(
                "Neighbor store is unavailable. Check DATABASE_URL and retry.",
                steps,
                category=category,
            )
        if self._listing_store is None:
            return self._error_outcome(
                "Listing store is unavailable. Check DATABASE_URL and retry.",
                steps,
                category=category,
            )

        try:
            neighbor_ids = self._neighbor_store.get_neighbors(property_id) or []
        except Exception as exc:
            return self._error_outcome(
                f"Neighbor lookup failed: {type(exc).__name__}: {exc}",
                steps,
                category=category,
            )

        neighbor_ids = [neighbor_id for neighbor_id in neighbor_ids if neighbor_id and neighbor_id != property_id]
        steps.append(
            StepLog(
                module="analyst_agent.neighbor_lookup",
                prompt={"property_id": property_id},
                response={
                    "neighbor_count": len(neighbor_ids),
                    "neighbor_ids": neighbor_ids[:10],
                },
            )
        )
        if not neighbor_ids:
            return self._error_outcome(
                f"No neighbors found for property_id={property_id}.",
                steps,
                category=category,
            )

        selected_columns = self._columns_for_category(category)
        listing_ids = [property_id] + neighbor_ids
        try:
            rows = self._listing_store.get_listings_by_ids(listing_ids, selected_columns)
        except Exception as exc:
            return self._error_outcome(
                f"Listing fetch failed: {type(exc).__name__}: {exc}",
                steps,
                category=category,
            )

        rows_by_id = {str(row.get("id", "")).strip(): row for row in rows if str(row.get("id", "")).strip()}
        owner_row = rows_by_id.get(property_id)
        neighbor_rows = [rows_by_id[neighbor_id] for neighbor_id in neighbor_ids if neighbor_id in rows_by_id]

        steps.append(
            StepLog(
                module="analyst_agent.data_fetch",
                prompt={
                    "category": category,
                    "property_id": property_id,
                    "requested_columns": selected_columns,
                    "requested_listing_count": len(listing_ids),
                },
                response={
                    "rows_returned": len(rows),
                    "owner_found": owner_row is not None,
                    "neighbor_rows_found": len(neighbor_rows),
                    "missing_neighbor_count": len(neighbor_ids) - len(neighbor_rows),
                },
            )
        )

        if owner_row is None:
            return self._error_outcome(
                f"Owner property {property_id} is missing from large_dataset_table.",
                steps,
                category=category,
            )
        if not neighbor_rows:
            return self._error_outcome(
                f"No neighbor listing rows were found in large_dataset_table for property_id={property_id}.",
                steps,
                category=category,
            )

        numeric_comparison, categorical_comparison = self._build_comparison(
            owner_row=owner_row,
            neighbor_rows=neighbor_rows,
            category=category,
        )
        steps.append(
            StepLog(
                module="analyst_agent.comparison_compute",
                prompt={
                    "category": category,
                    "numeric_columns": self._numeric_columns_for_category(category),
                    "categorical_columns": self._categorical_columns_for_category(category),
                },
                response={
                    "numeric_items": len(numeric_comparison),
                    "categorical_items": len(categorical_comparison),
                    "neighbor_rows_used": len(neighbor_rows),
                },
            )
        )

        narrative = self._build_narrative(
            prompt=prompt,
            category=category,
            property_id=property_id,
            property_name=str(owner_row.get("name", "")).strip() or None,
            numeric_comparison=numeric_comparison,
            categorical_comparison=categorical_comparison,
        )
        steps.append(self._last_answer_step)

        return AnalystRunOutcome(
            analysis_category=category,
            narrative=narrative,
            error=None,
            numeric_comparison=numeric_comparison,
            categorical_comparison=categorical_comparison,
            steps=steps,
        )

    @property
    def _last_answer_step(self) -> StepLog:
        return self.__last_answer_step

    @_last_answer_step.setter
    def _last_answer_step(self, step: StepLog) -> None:
        self.__last_answer_step = step

    def _error_outcome(
        self,
        message: str,
        steps: list[StepLog],
        *,
        category: Literal["review_scores", "property_specs"],
    ) -> AnalystRunOutcome:
        self._last_answer_step = StepLog(
            module="analyst_agent.answer_generation",
            prompt={"status": "skipped"},
            response={"status": "skipped", "reason": message},
        )
        return AnalystRunOutcome(
            analysis_category=category,
            narrative=message,
            error=message,
            numeric_comparison=[],
            categorical_comparison=[],
            steps=steps + [self._last_answer_step],
        )

    def _resolve_category(
        self,
        prompt: str,
        context: dict[str, object],
    ) -> Literal["review_scores", "property_specs"]:
        explicit = _context_str(context, "analysis_category")
        if explicit in ANALYSIS_CATEGORIES:
            return explicit  # type: ignore[return-value]

        lowered = prompt.lower()
        review_patterns = [
            r"\breview(?:s)?\s+score(?:s)?\b",
            r"\breview(?:s)?\s+rating(?:s)?\b",
            r"\bguest\s+review(?:s)?\b",
            r"\brating(?:s)?\b",
            r"\bcleanliness\b",
            r"\bcheck[\s-]?in\b",
            r"\bcommunication\b",
            r"\baccuracy\b",
            r"\blocation\s+(?:score|rating)\b",
            r"\bvalue\s+(?:score|rating)\b",
        ]
        if any(re.search(pattern, lowered) for pattern in review_patterns):
            return "review_scores"
        return "property_specs"

    def _columns_for_category(self, category: Literal["review_scores", "property_specs"]) -> list[str]:
        if category == "review_scores":
            return list(REVIEW_SCORE_COLUMNS)
        return list(PROPERTY_SPEC_COLUMNS)

    def _numeric_columns_for_category(self, category: Literal["review_scores", "property_specs"]) -> list[str]:
        if category == "review_scores":
            return list(REVIEW_SCORE_COLUMNS)
        return list(PROPERTY_SPEC_NUMERIC_COLUMNS)

    def _categorical_columns_for_category(self, category: Literal["review_scores", "property_specs"]) -> list[str]:
        if category == "review_scores":
            return []
        return list(PROPERTY_SPEC_CATEGORICAL_COLUMNS)

    def _build_comparison(
        self,
        *,
        owner_row: dict[str, Any],
        neighbor_rows: list[dict[str, Any]],
        category: Literal["review_scores", "property_specs"],
    ) -> tuple[list[AnalysisNumericItem], list[AnalysisCategoricalItem]]:
        numeric_items: list[AnalysisNumericItem] = []
        categorical_items: list[AnalysisCategoricalItem] = []

        for column in self._numeric_columns_for_category(category):
            owner_value = self._normalize_numeric(column, owner_row.get(column))
            neighbor_points: list[AnalysisNeighborPoint] = []
            for row in neighbor_rows:
                normalized = self._normalize_numeric(column, row.get(column))
                if normalized is None:
                    continue
                neighbor_points.append(
                    AnalysisNeighborPoint(
                        listing_id=str(row.get("id", "")).strip(),
                        listing_name=(str(row.get("name", "")).strip() or None),
                        value=_normalize_point_value(normalized),
                    )
                )
            neighbor_points.sort(
                key=lambda point: (point.value, (point.listing_name or "").lower(), point.listing_id)
            )
            neighbor_values = [point.value for point in neighbor_points]
            neighbor_min = min(neighbor_values) if neighbor_values else None
            neighbor_max = max(neighbor_values) if neighbor_values else None
            neighbor_min_points = (
                [point for point in neighbor_points if neighbor_min is not None and math.isclose(point.value, neighbor_min, abs_tol=1e-9)]
                if neighbor_values
                else []
            )
            neighbor_max_points = (
                [point for point in neighbor_points if neighbor_max is not None and math.isclose(point.value, neighbor_max, abs_tol=1e-9)]
                if neighbor_values
                else []
            )
            numeric_items.append(
                AnalysisNumericItem(
                    label=_column_label(column),
                    column=column,
                    owner_value=owner_value,
                    neighbor_avg=(mean(neighbor_values) if neighbor_values else None),
                    neighbor_min=neighbor_min,
                    neighbor_max=neighbor_max,
                    neighbor_count=len(neighbor_values),
                    neighbor_points=neighbor_points,
                    neighbor_min_points=neighbor_min_points,
                    neighbor_max_points=neighbor_max_points,
                )
            )

        for column in self._categorical_columns_for_category(category):
            owner_value = self._normalize_category(column, owner_row.get(column))
            bucket_members: dict[str, list[tuple[str, str | None]]] = {}
            neighbor_values: list[str] = []
            for row in neighbor_rows:
                normalized = self._normalize_category(column, row.get(column))
                if normalized is None:
                    continue
                listing_id = str(row.get("id", "")).strip()
                listing_name = str(row.get("name", "")).strip() or None
                neighbor_values.append(normalized)
                bucket_members.setdefault(normalized, []).append((listing_id, listing_name))

            counts = Counter(neighbor_values)
            total = sum(counts.values())
            buckets = [
                AnalysisCategoryBucket(
                    value=value,
                    count=count,
                    pct=(round((count / total) * 100.0, 1) if total else 0.0),
                    listing_ids=[
                        listing_id
                        for listing_id, _ in sorted(
                            bucket_members.get(value, []),
                            key=lambda item: (((item[1] or "").lower()), item[0]),
                        )
                    ],
                    listing_names=[
                        listing_name or listing_id
                        for listing_id, listing_name in sorted(
                            bucket_members.get(value, []),
                            key=lambda item: (((item[1] or "").lower()), item[0]),
                        )
                    ],
                )
                for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
            ]
            categorical_items.append(
                AnalysisCategoricalItem(
                    label=_column_label(column),
                    column=column,
                    owner_value=owner_value,
                    neighbor_count=total,
                    buckets=buckets,
                )
            )

        return numeric_items, categorical_items

    def _normalize_numeric(self, column: str, value: Any) -> float | int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            if isinstance(value, float) and math.isnan(value):
                return None
            return float(value)
        raw = str(value).strip()
        if not raw:
            return None
        raw = raw.replace(",", "")
        if column == "price":
            raw = raw.replace("$", "")
        if raw.endswith("%"):
            raw = raw[:-1]
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        number = float(match.group(0))
        return number

    def _normalize_category(self, column: str, value: Any) -> str | None:
        if value is None:
            return None
        raw = str(value).strip()
        if not raw:
            return None
        lowered = raw.lower()
        if column == "host_is_superhost":
            if lowered in {"t", "true", "1", "yes", "y"}:
                return "true"
            if lowered in {"f", "false", "0", "no", "n"}:
                return "false"
        return raw

    def _build_narrative(
        self,
        *,
        prompt: str,
        category: Literal["review_scores", "property_specs"],
        property_id: str,
        property_name: str | None,
        numeric_comparison: list[AnalysisNumericItem],
        categorical_comparison: list[AnalysisCategoricalItem],
    ) -> str:
        compact_summary = self._build_compact_summary(
            category=category,
            property_id=property_id,
            property_name=property_name,
            numeric_comparison=numeric_comparison,
            categorical_comparison=categorical_comparison,
        )

        if not self._chat.is_available:
            text = self._deterministic_narrative(compact_summary)
            self._last_answer_step = StepLog(
                module="analyst_agent.answer_generation",
                prompt={"model": None, "fallback": True},
                response={"text": text, "fallback_used": True},
            )
            return text

        system_prompt = (
            "You are a hospitality benchmarking analyst. "
            "Write a concise, business-oriented comparison using only the supplied computed stats. "
            "Do not invent data. Mention strengths, weaknesses, and one clear recommendation."
        )
        user_prompt = (
            f"User request: {prompt}\n\n"
            f"Analysis category: {category}\n"
            f"Computed comparison summary:\n{compact_summary}\n\n"
            "Return 3 short paragraphs maximum. Avoid bullet lists."
        )
        try:
            text = self._chat.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            fallback = self._deterministic_narrative(compact_summary)
            self._last_answer_step = StepLog(
                module="analyst_agent.answer_generation",
                prompt={"model": self._chat.model, "system_prompt": system_prompt, "user_prompt": user_prompt},
                response={"error": f"{type(exc).__name__}: {exc}", "fallback_used": True, "text": fallback},
            )
            return fallback

        text = text.strip() or self._deterministic_narrative(compact_summary)
        self._last_answer_step = StepLog(
            module="analyst_agent.answer_generation",
            prompt={"model": self._chat.model, "system_prompt": system_prompt, "user_prompt": user_prompt},
            response={"text": text},
        )
        return text

    def _build_compact_summary(
        self,
        *,
        category: str,
        property_id: str,
        property_name: str | None,
        numeric_comparison: list[AnalysisNumericItem],
        categorical_comparison: list[AnalysisCategoricalItem],
    ) -> str:
        lines = [
            f"Property: {property_name or property_id}",
            f"Category: {category}",
        ]
        for item in numeric_comparison:
            lines.append(
                (
                    f"NUM {item.label} [{item.column}]: owner={_format_number(item.owner_value)}; "
                    f"neighbor_avg={_format_number(item.neighbor_avg)}; "
                    f"neighbor_min={_format_number(item.neighbor_min)}; "
                    f"neighbor_max={_format_number(item.neighbor_max)}; "
                    f"neighbor_count={item.neighbor_count}"
                )
            )
        for item in categorical_comparison:
            bucket_text = ", ".join(f"{bucket.value}={bucket.count} ({bucket.pct}%)" for bucket in item.buckets[:5])
            lines.append(
                f"CAT {item.label} [{item.column}]: owner={item.owner_value or 'n/a'}; neighbor_count={item.neighbor_count}; buckets={bucket_text or 'n/a'}"
            )
        return "\n".join(lines)

    def _deterministic_narrative(self, compact_summary: str) -> str:
        lines = compact_summary.splitlines()
        numeric_lines = [line for line in lines if line.startswith("NUM ")]
        category_lines = [line for line in lines if line.startswith("CAT ")]
        summary = []
        if numeric_lines:
            summary.append("Competitive analysis completed using structured listing comparisons.")
            summary.extend(numeric_lines[:3])
        if category_lines:
            summary.append("Categorical positioning against neighbors:")
            summary.extend(category_lines[:2])
        if not summary:
            summary.append("Competitive analysis completed, but no comparable values were available.")
        return "\n".join(summary)

    def _selection_payload_to_text(
        self,
        *,
        selection_type: Literal["numeric_point", "numeric_extreme", "categorical_bucket"],
        metric_column: str,
        selection_payload: dict[str, Any],
    ) -> str:
        lines = [f"Metric label: {selection_payload.get('metric_label') or _column_label(metric_column)}"]
        for key, value in selection_payload.items():
            if isinstance(value, list):
                joined = ", ".join(str(item) for item in value[:20])
                if len(value) > 20:
                    joined += ", ..."
                lines.append(f"{key}: {joined}")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines)

    def _deterministic_selection_explanation(
        self,
        *,
        selection_type: Literal["numeric_point", "numeric_extreme", "categorical_bucket"],
        metric_column: str,
        selection_payload: dict[str, Any],
    ) -> str:
        metric_label = str(selection_payload.get("metric_label") or _column_label(metric_column))
        owner_value = selection_payload.get("owner_value")
        if selection_type == "numeric_point":
            point_role = selection_payload.get("point_role")
            if point_role == "owner":
                selected_value = selection_payload.get("selected_value")
                neighbor_avg = selection_payload.get("neighbor_avg")
                higher_count = selection_payload.get("higher_count")
                tied_count = selection_payload.get("tied_count")
                lower_count = selection_payload.get("lower_count")
                count = selection_payload.get("neighbor_count")
                return (
                    f"Your property is at {selected_value} for {metric_label}, versus neighbor average {neighbor_avg}. "
                    f"Among {count} comparable neighbors, {higher_count} are above you, {tied_count} are tied, and {lower_count} are below you."
                )
            listing_name = selection_payload.get("listing_name") or selection_payload.get("listing_id") or "Selected neighbor"
            selected_value = selection_payload.get("selected_value")
            neighbor_avg = selection_payload.get("neighbor_avg")
            rank = selection_payload.get("rank")
            count = selection_payload.get("neighbor_count")
            return (
                f"{listing_name} is one neighbor reference point for {metric_label} at {selected_value}. "
                f"Your property is at {owner_value} versus neighbor average {neighbor_avg}. "
                f"This selected listing ranks {rank} out of {count} among the comparable neighbors."
            )
        if selection_type == "numeric_extreme":
            extreme_type = selection_payload.get("extreme_type") or "extreme"
            selected_value = selection_payload.get("selected_value")
            listings = selection_payload.get("listing_names") or []
            listing_text = ", ".join(str(name) for name in listings[:5]) or "matching neighbors"
            return (
                f"The selected {extreme_type} for {metric_label} is {selected_value}. "
                f"Neighbors at that level include {listing_text}. "
                f"Your property is at {owner_value}, which shows how far you are from the edge of the local distribution."
            )
        bucket_value = selection_payload.get("bucket_value") or "selected bucket"
        pct = selection_payload.get("bucket_pct")
        count = selection_payload.get("bucket_count")
        listing_names = selection_payload.get("listing_names") or []
        listing_text = ", ".join(str(name) for name in listing_names[:5]) or "matching neighbors"
        return (
            f"The selected bucket for {metric_label} is '{bucket_value}', representing {count} neighbors ({pct}%). "
            f"Examples in this bucket include {listing_text}. "
            f"Your property is classified as {owner_value}, so this shows how aligned you are with the local market mix."
        )
