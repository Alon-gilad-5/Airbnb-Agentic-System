"""Domain agent that answers questions from Airbnb guest reviews.

Uses a LangChain-first architecture with composable RunnableLambda stages
for the reviews RAG pipeline.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableLambda

from app.agents.base import Agent, AgentResult
from app.schemas import StepLog
from app.services.chat_service import ChatService
from app.services.embeddings import EmbeddingService
from app.services.pinecone_retriever import PineconeRetriever, RetrievedReview
from app.services.region_utils import canonicalize_region
from app.services.web_review_ingest import WebReviewIngestService
from app.services.web_review_scraper import PlaywrightReviewScraper, ScrapedReview

NO_EVIDENCE_RESPONSE = "I couldn't find enough data to answer your question."
LOW_EVIDENCE_PREFIX = "Evidence is limited (based on only 1-2 relevant reviews), so confidence is low."


@dataclass
class ReviewsAgentConfig:
    """Configuration knobs for retrieval depth and output synthesis."""

    top_k: int = 8
    max_context_reviews: int = 6
    relevance_score_threshold: float = 0.40
    thin_evidence_min: int = 1
    thin_evidence_max: int = 2
    max_answer_words: int = 140
    max_citations: int = 4
    min_lexical_relevance_for_upsert: float = 0.15


# ---------------------------------------------------------------------------
# Module-level helper functions (shared between pipeline and backward-compat)
# ---------------------------------------------------------------------------


def _context_str(ctx: dict[str, Any], key: str) -> str | None:
    value = ctx.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    similarity = dot_product / (norm_a * norm_b)
    return (similarity + 1.0) / 2.0


def _score_scraped_relevance(embedding_service: Any, prompt: str, review_text: str) -> float:
    if not embedding_service.is_available:
        return 0.5
    try:
        query_embedding = embedding_service.embed_query(prompt)
        review_embedding = embedding_service.embed_query(review_text)
        return _cosine_similarity(query_embedding, review_embedding)
    except Exception:
        return 0.5


def _build_evidence_context(matches: list[RetrievedReview], max_reviews: int) -> str:
    snippets: list[str] = []
    for i, match in enumerate(matches[:max_reviews], start=1):
        md = match.metadata
        region = md.get("location", md.get("region", "unknown"))
        property_id = md.get("property_id", "unknown")
        review_date = md.get("review_date", "unknown")
        text = str(md.get("review_text", "")).strip()
        snippets.append(
            f"[{i}] region={region}; property_id={property_id}; date={review_date}; "
            f"score={match.score:.4f}; text={text}"
        )
    return "\n".join(snippets)


def _deterministic_summary(matches: list[RetrievedReview]) -> str:
    count = len(matches[:3])
    if count == 0:
        return "No matching reviews were found."
    return (
        f"Found {count} matching review{'s' if count != 1 else ''}. "
        "The LLM synthesis service is currently unavailable, so only raw evidence is shown below.\n\n"
        "Confidence: low"
    )


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " ..."


def _detect_hallucination_risk(answer_text: str) -> tuple[bool, list[str]]:
    phrases = [
        "many guests", "a lot of reviews", "most guests",
        "guests generally", "everyone",
    ]
    lowered = answer_text.lower()
    matched = [phrase for phrase in phrases if re.search(rf"\b{re.escape(phrase)}\b", lowered)]
    return bool(matched), matched


def _build_evidence_snippets(evidence: list[RetrievedReview], max_reviews: int) -> list[str]:
    """Build short human-readable evidence snippets for the UI evidence panel."""
    snippets: list[str] = []
    for match in evidence[:max_reviews]:
        md = match.metadata
        text = str(md.get("review_text", "")).strip()
        short = text[:200] + ("..." if len(text) > 200 else "")
        date = md.get("review_date", "unknown")
        reviewer = md.get("reviewer_name", "Guest")
        score = f"{match.score:.2f}"
        snippets.append(f"[{score}] \"{short}\" — {reviewer}, {date}")
    return snippets


def _build_citations(evidence: list[RetrievedReview], max_citations: int) -> list[str]:
    citations: list[str] = []
    for match in evidence:
        md = match.metadata
        source_type = str(md.get("source_type", "")).strip().lower()
        source = str(md.get("source", "")).strip()
        source_url = str(md.get("source_url", "")).strip()
        if source_type == "web_scrape" or source:
            source_label = source if source else "External Review Source"
            url_label = source_url if source_url else "url_unavailable"
            citation = f"Source ({source_label}) + {url_label}"
        else:
            region = str(md.get("region", "unknown"))
            property_id = str(md.get("property_id", "unknown"))
            review_id = str(md.get("review_id", match.vector_id))
            citation = f"Internal VDB ({region}/{property_id}/{review_id})"
        if citation not in citations:
            citations.append(citation)
        if len(citations) >= max_citations:
            break
    return citations


def _append_citations(answer_text: str, citations: list[str]) -> str:
    if not citations:
        return answer_text
    lines = [answer_text, "", "Citations:"]
    lines.extend([f"- {citation}" for citation in citations])
    return "\n".join(lines)


def _upsert_scraped(ingest_service: Any, scraped_reviews: list[Any], ctx: dict[str, Any]) -> StepLog:
    if not scraped_reviews:
        return StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": 0},
            response={"status": "skipped", "reason": "No scraped reviews collected"},
        )
    if not ingest_service.is_available:
        return StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": len(scraped_reviews)},
            response={"status": "skipped", "reason": "Web quarantine ingest service unavailable"},
        )
    try:
        result = ingest_service.upsert_scraped_reviews(reviews=scraped_reviews, context=ctx)
        return StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": result.attempted},
            response={
                "status": "ok",
                "upserted": result.upserted,
                "namespace": result.namespace,
                "vector_ids": result.vector_ids[:5],
            },
        )
    except Exception as exc:
        return StepLog(
            module="reviews_agent.web_quarantine_upsert",
            prompt={"attempted": len(scraped_reviews)},
            response={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
        )


# ---------------------------------------------------------------------------
# LangChain-first reviews RAG pipeline
# ---------------------------------------------------------------------------


class ReviewsPipeline:
    """LangChain-first reviews RAG pipeline composed of RunnableLambda stages.

    Each stage is a named runnable that accepts pipeline state and returns
    incremental updates.  The orchestrator merges updates and handles
    conditional branching between stages.
    """

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        retriever: PineconeRetriever,
        chat_service: ChatService,
        web_scraper: PlaywrightReviewScraper,
        web_ingest_service: WebReviewIngestService,
        config: ReviewsAgentConfig,
    ) -> None:
        self._embedding = embedding_service
        self._retriever = retriever
        self._chat = chat_service
        self._scraper = web_scraper
        self._ingest = web_ingest_service
        self._config = config

        self.build_filter = RunnableLambda(self._build_filter_stage).with_config(
            run_name="BuildMetadataFilter",
        )
        self.retrieve = RunnableLambda(self._retrieve_stage).with_config(
            run_name="Retrieve",
        )
        self.fallback_decision = RunnableLambda(self._fallback_decision_stage).with_config(
            run_name="FallbackDecision",
        )
        self.scrape = RunnableLambda(self._scrape_stage).with_config(
            run_name="WebScrape",
        )
        self.merge_evidence = RunnableLambda(self._merge_evidence_stage).with_config(
            run_name="MergeEvidence",
        )
        self.evidence_guard = RunnableLambda(self._evidence_guard_stage).with_config(
            run_name="EvidenceGuard",
        )
        self.generate_answer = RunnableLambda(self._generate_answer_stage).with_config(
            run_name="GenerateAnswer",
        )
        self.finalize = RunnableLambda(self._finalize_stage).with_config(
            run_name="Finalize",
        )

    # -- state merge helper --

    @staticmethod
    def _apply(state: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
        """Merge stage updates into pipeline state, accumulating steps via addition."""
        merged = dict(state)
        new_steps = updates.pop("steps", [])
        merged.update(updates)
        merged["steps"] = merged.get("steps", []) + new_steps
        return merged

    # -- main orchestration --

    def invoke(self, state: dict[str, Any]) -> dict[str, Any]:
        """Execute the full reviews pipeline with conditional branching."""

        state = self._apply(state, self.build_filter.invoke(state))
        state = self._apply(state, self.retrieve.invoke(state))

        if state.get("final_answer") and not state.get("should_answer", True):
            return state

        state = self._apply(state, self.fallback_decision.invoke(state))

        if state.get("fallback_triggered"):
            state = self._apply(state, self.scrape.invoke(state))

        state = self._apply(state, self.merge_evidence.invoke(state))
        state = self._apply(state, self.evidence_guard.invoke(state))

        if not state.get("should_answer"):
            return state

        state = self._apply(state, self.generate_answer.invoke(state))

        if state.get("final_answer") and not state.get("llm_answer"):
            return state

        state = self._apply(state, self.finalize.invoke(state))
        return state

    # -- stage implementations --

    def _build_filter_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = state["prompt"]
        ctx = state.get("context") or {}
        property_id = _context_str(ctx, "property_id")
        region = canonicalize_region(_context_str(ctx, "region"))

        if property_id and region:
            mf: dict[str, Any] | None = {
                "$and": [
                    {"property_id": {"$eq": property_id}},
                    {"region": {"$eq": region}},
                ]
            }
        elif property_id:
            mf = {"property_id": {"$eq": property_id}}
        elif region:
            mf = {"region": {"$eq": region}}
        else:
            mf = None
            lowered = prompt.lower()
            known_regions = [
                "los angeles", "oakland", "pacific grove", "san diego",
                "san francisco", "san mateo", "santa clara", "santa cruz county",
            ]
            for r in known_regions:
                if r in lowered:
                    mf = {"region": {"$eq": r}}
                    break

        return {"metadata_filter": mf}

    def _retrieve_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = state["prompt"]
        metadata_filter = state.get("metadata_filter")
        cfg = self._config

        if not self._embedding.is_available:
            return {
                "matches": [],
                "final_answer": (
                    "Embedding service is not configured. Set LLMOD_API_KEY and BASE_URL "
                    "to enable vector search."
                ),
                "should_answer": False,
                "steps": [],
            }
        if not self._retriever.is_available:
            return {
                "matches": [],
                "final_answer": "Pinecone is not configured. Set PINECONE_API_KEY to enable retrieval.",
                "should_answer": False,
                "steps": [],
            }

        try:
            query_embedding = self._embedding.embed_query(prompt)
            matches = self._retriever.query(
                embedding=query_embedding,
                top_k=cfg.top_k,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            return {
                "matches": [],
                "final_answer": (
                    "I could not access retrieval services right now. "
                    "Please verify LLMOD/Pinecone connectivity and try again."
                ),
                "should_answer": False,
                "steps": [
                    StepLog(
                        module="reviews_agent.retrieval",
                        prompt={
                            "user_prompt": prompt,
                            "top_k": cfg.top_k,
                            "metadata_filter": metadata_filter,
                        },
                        response={"error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
            }

        return {
            "matches": matches,
            "steps": [
                StepLog(
                    module="reviews_agent.retrieval",
                    prompt={
                        "user_prompt": prompt,
                        "top_k": cfg.top_k,
                        "metadata_filter": metadata_filter,
                    },
                    response={
                        "match_count": len(matches),
                        "top_match_ids": [m.vector_id for m in matches[:5]],
                    },
                )
            ],
        }

    def _fallback_decision_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        matches = state.get("matches") or []
        top_score = matches[0].score if matches else 0.0
        fallback_triggered = (not matches) or (top_score < self._config.relevance_score_threshold)
        return {"fallback_triggered": fallback_triggered}

    def _scrape_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        prompt = state["prompt"]
        ctx = state.get("context") or {}
        matches = state.get("matches") or []
        top_score = matches[0].score if matches else 0.0
        cfg = self._config

        if not self._scraper.is_available:
            return {
                "web_matches": [],
                "steps": [
                    StepLog(
                        module="reviews_agent.web_scrape",
                        prompt={"triggered": True, "reason": "low_internal_relevance", "top_score": top_score},
                        response={"status": "skipped", "reason": "Playwright scraper is disabled/unavailable"},
                    ),
                    StepLog(
                        module="reviews_agent.web_quarantine_upsert",
                        prompt={"attempted": 0},
                        response={"status": "skipped", "reason": "No scraped reviews to upsert"},
                    ),
                ],
            }

        source_urls_obj = ctx.get("source_urls")
        source_urls = source_urls_obj if isinstance(source_urls_obj, dict) else None
        max_scrape_raw = ctx.get("max_scrape_reviews")
        max_scrape: int | None = None
        if isinstance(max_scrape_raw, int):
            max_scrape = max_scrape_raw
        elif isinstance(max_scrape_raw, str) and max_scrape_raw.isdigit():
            max_scrape = int(max_scrape_raw)

        scraped, scraper_meta = self._scraper.scrape_reviews(
            prompt=prompt,
            property_name=_context_str(ctx, "property_name"),
            city=_context_str(ctx, "city"),
            region=_context_str(ctx, "region"),
            source_urls=source_urls,
            max_reviews=max_scrape,
        )

        property_id = _context_str(ctx, "property_id")
        region_norm = canonicalize_region(_context_str(ctx, "region"))
        converted: list[RetrievedReview] = []
        for idx, review in enumerate(scraped, start=1):
            score = _score_scraped_relevance(self._embedding, prompt, review.review_text)
            metadata: dict[str, Any] = {
                "source_type": "web_scrape",
                "source": review.source,
                "source_url": review.source_url,
                "review_text": review.review_text,
                "review_date": review.review_date or "unknown",
                "reviewer_name": review.reviewer_name or "unknown",
                "rating": review.rating,
            }
            if property_id:
                metadata["property_id"] = property_id
            if region_norm:
                metadata["region"] = region_norm
            converted.append(RetrievedReview(vector_id=f"web:{review.source}:{idx}", score=score, metadata=metadata))

        relevant_converted = [m for m in converted if m.score >= cfg.relevance_score_threshold]
        gated_scraped = [
            review for review, match in zip(scraped, converted)
            if match.score >= cfg.min_lexical_relevance_for_upsert
        ]
        rejected_by_relevance_gate = max(0, len(scraped) - len(gated_scraped))

        scrape_step = StepLog(
            module="reviews_agent.web_scrape",
            prompt={
                "prompt": prompt,
                "property_name": _context_str(ctx, "property_name"),
                "city": _context_str(ctx, "city"),
                "region": _context_str(ctx, "region"),
                "max_scrape_reviews": max_scrape,
                "min_lexical_relevance_for_upsert": cfg.min_lexical_relevance_for_upsert,
            },
            response={
                "scraper_status": scraper_meta.get("status"),
                "scraped_raw_count": scraper_meta.get("raw_count", 0),
                "scraped_deduped_count": scraper_meta.get("deduped_count", 0),
                "converted_count": len(converted),
                "relevant_converted_count": len(relevant_converted),
                "upsert_candidate_count": len(gated_scraped),
                "rejected_by_relevance_gate": rejected_by_relevance_gate,
                "attempted_targets": scraper_meta.get("attempted_targets", []),
                "noise_rejection_stats": scraper_meta.get("noise_rejection_stats", {}),
                "errors": scraper_meta.get("errors", []),
            },
        )

        upsert_step = _upsert_scraped(self._ingest, gated_scraped, ctx)
        return {"web_matches": converted, "steps": [scrape_step, upsert_step]}

    def _merge_evidence_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        matches = state.get("matches") or []
        web_matches = state.get("web_matches") or []
        cfg = self._config
        relevant = [m for m in matches if m.score >= cfg.relevance_score_threshold]
        relevant.extend([m for m in web_matches if m.score >= cfg.relevance_score_threshold])
        return {"relevant_matches": relevant, "evidence_count": len(relevant)}

    def _evidence_guard_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        count = state.get("evidence_count", 0)
        cfg = self._config

        if count == 0:
            decision = "fallback_no_evidence"
            should_answer = False
            disclaimer: str | None = None
        elif cfg.thin_evidence_min <= count <= cfg.thin_evidence_max:
            decision = "answer_with_low_evidence_disclaimer"
            should_answer = True
            disclaimer = LOW_EVIDENCE_PREFIX
        else:
            decision = "answer_normal"
            should_answer = True
            disclaimer = None

        update: dict[str, Any] = {
            "should_answer": should_answer,
            "disclaimer_prefix": disclaimer,
            "steps": [
                StepLog(
                    module="reviews_agent.evidence_guard",
                    prompt={
                        "relevance_score_threshold": cfg.relevance_score_threshold,
                        "thin_evidence_min": cfg.thin_evidence_min,
                        "thin_evidence_max": cfg.thin_evidence_max,
                    },
                    response={
                        "decision": decision,
                        "relevant_evidence_count": count,
                    },
                )
            ],
        }
        if not should_answer:
            update["final_answer"] = NO_EVIDENCE_RESPONSE
        return update

    def _generate_answer_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        relevant_matches = state.get("relevant_matches") or []
        cfg = self._config
        evidence_for_context = relevant_matches[:cfg.max_context_reviews]
        evidence_count = state.get("evidence_count", 0)
        prompt_text = state["prompt"]

        if not self._chat.is_available:
            fallback = _deterministic_summary(evidence_for_context)
            final = self._finalize_answer(fallback, evidence_for_context, state.get("disclaimer_prefix"), evidence_count)
            return {"final_answer": final["answer"], "steps": final["steps"]}

        context_text = _build_evidence_context(evidence_for_context, cfg.max_context_reviews)
        system_prompt = (
            "You are a hospitality insights assistant. "
            "Answer only from provided review evidence. "
            "If evidence is weak, say so explicitly. "
            "Do not use unsupported broad quantifiers (e.g., many guests, most guests, a lot of reviews) "
            "unless that is clearly supported by the provided evidence."
        )
        user_prompt = (
            f"Question:\n{prompt_text}\n\n"
            f"Review Evidence:\n{context_text}\n\n"
            f"Relevant evidence count: {evidence_count}\n"
            "Confidence policy:\n"
            "- 1-2 relevant reviews => confidence must be low.\n"
            "- 3+ relevant reviews => confidence can be medium/high only if supported.\n\n"
            "Return ONLY a concise 2-4 sentence business-friendly answer that directly "
            "addresses the question, followed by a confidence level.\n"
            "Format:\n"
            "<answer text>\n\n"
            "Confidence: <high/medium/low>\n\n"
            "Do NOT include evidence bullets, citations, or review numbers in your answer. "
            "Just provide a clear, natural-language summary."
        )

        try:
            llm_answer = self._chat.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            return {
                "final_answer": (
                    "I retrieved relevant reviews but could not generate the final summary right now. "
                    "Please retry in a moment."
                ),
                "steps": [
                    StepLog(
                        module="reviews_agent.answer_generation",
                        prompt={
                            "model": self._chat.model,
                            "system_prompt": system_prompt,
                            "user_prompt": user_prompt,
                        },
                        response={"error": f"{type(exc).__name__}: {exc}"},
                    )
                ],
            }

        answer_step = StepLog(
            module="reviews_agent.answer_generation",
            prompt={
                "model": self._chat.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            response={"text": llm_answer},
        )
        if not llm_answer.strip():
            llm_answer = _deterministic_summary(evidence_for_context)
            answer_step.response["fallback_used"] = True
            answer_step.response["fallback_reason"] = "empty_model_output"
            answer_step.response["text"] = llm_answer

        return {"llm_answer": llm_answer, "steps": [answer_step]}

    def _finalize_stage(self, state: dict[str, Any]) -> dict[str, Any]:
        relevant_matches = state.get("relevant_matches") or []
        cfg = self._config
        evidence_for_context = relevant_matches[:cfg.max_context_reviews]
        disclaimer_prefix = state.get("disclaimer_prefix")
        evidence_count = state.get("evidence_count", 0)
        llm_answer = state.get("llm_answer", "")

        result = self._finalize_answer(llm_answer, evidence_for_context, disclaimer_prefix, evidence_count)
        return {"final_answer": result["answer"], "steps": result["steps"]}

    def _finalize_answer(
        self,
        answer_text: str,
        evidence: list[RetrievedReview],
        disclaimer_prefix: str | None,
        evidence_count: int,
    ) -> dict[str, Any]:
        """Shared finalization: word cap, disclaimer, structured evidence, hallucination guard."""
        cfg = self._config
        final_answer = answer_text.strip()
        final_answer = _cap_words(final_answer, cfg.max_answer_words)
        if disclaimer_prefix:
            final_answer = f"{disclaimer_prefix}\n\n{final_answer}"

        evidence_snippets = _build_evidence_snippets(evidence, cfg.max_context_reviews)
        citations = _build_citations(evidence, cfg.max_citations)

        sections = [final_answer, "---EVIDENCE---"]
        for snippet in evidence_snippets:
            sections.append(snippet)
        sections.append("---CITATIONS---")
        for citation in citations:
            sections.append(citation)

        structured_answer = "\n".join(sections)

        risk_flag, matched_phrases = _detect_hallucination_risk(final_answer)
        hall_step = StepLog(
            module="reviews_agent.hallucination_guard",
            prompt={
                "checked_phrases": [
                    "many guests", "a lot of reviews", "most guests",
                    "guests generally", "everyone",
                ],
            },
            response={
                "risk_flag": risk_flag,
                "matched_phrases": matched_phrases,
                "evidence_count": evidence_count,
                "action": "flag_only",
            },
        )
        return {"answer": structured_answer, "steps": [hall_step]}


# ---------------------------------------------------------------------------
# ReviewsAgent – thin wrapper delegating to ReviewsPipeline
# ---------------------------------------------------------------------------


class ReviewsAgent(Agent):
    """Retrieval-augmented QA agent over review vectors + metadata.

    Thin wrapper that delegates execution to a LangChain-first ReviewsPipeline.
    """

    name = "reviews_agent"

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService,
        retriever: PineconeRetriever,
        chat_service: ChatService,
        web_scraper: PlaywrightReviewScraper,
        web_ingest_service: WebReviewIngestService,
        config: ReviewsAgentConfig | None = None,
    ) -> None:
        self.embedding_service = embedding_service
        self.retriever = retriever
        self.chat_service = chat_service
        self.web_scraper = web_scraper
        self.web_ingest_service = web_ingest_service
        self.config = config or ReviewsAgentConfig()
        self._pipeline = ReviewsPipeline(
            embedding_service=embedding_service,
            retriever=retriever,
            chat_service=chat_service,
            web_scraper=web_scraper,
            web_ingest_service=web_ingest_service,
            config=self.config,
        )

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Execute retrieval and response synthesis with full trace steps."""

        result = self._pipeline.invoke({
            "prompt": prompt,
            "context": context or {},
            "steps": [],
        })
        return AgentResult(
            response=result.get("final_answer", NO_EVIDENCE_RESPONSE),
            steps=result.get("steps", []),
        )

    # -- Backward-compatible helpers exposed for existing tests --

    def _build_metadata_filter(self, prompt: str, context: dict[str, object]) -> dict[str, Any] | None:
        """Optional heuristic filters to reduce retrieval scope when prompt names a region."""

        property_id = self._context_str(context, "property_id")
        region = canonicalize_region(self._context_str(context, "region"))
        if property_id and region:
            return {
                "$and": [
                    {"property_id": {"$eq": property_id}},
                    {"region": {"$eq": region}},
                ]
            }
        if property_id:
            return {"property_id": {"$eq": property_id}}
        if region:
            return {"region": {"$eq": region}}

        lowered = prompt.lower()
        known_regions = [
            "los angeles", "oakland", "pacific grove", "san diego",
            "san francisco", "san mateo", "santa clara", "santa cruz county",
        ]
        for region in known_regions:
            if region in lowered:
                return {"region": {"$eq": region}}
        return None

    def _scrape_fallback(
        self,
        *,
        prompt: str,
        context: dict[str, object],
    ) -> tuple[list[RetrievedReview], StepLog, StepLog]:
        """Scrape additional web reviews and convert them into retrieval-compatible evidence."""

        source_urls_obj = context.get("source_urls")
        source_urls = source_urls_obj if isinstance(source_urls_obj, dict) else None
        max_scrape_raw = context.get("max_scrape_reviews")
        max_scrape: int | None = None
        if isinstance(max_scrape_raw, int):
            max_scrape = max_scrape_raw
        elif isinstance(max_scrape_raw, str) and max_scrape_raw.isdigit():
            max_scrape = int(max_scrape_raw)

        scraped, scraper_meta = self.web_scraper.scrape_reviews(
            prompt=prompt,
            property_name=self._context_str(context, "property_name"),
            city=self._context_str(context, "city"),
            region=self._context_str(context, "region"),
            source_urls=source_urls,
            max_reviews=max_scrape,
        )
        converted = self._convert_scraped_to_matches(scraped_reviews=scraped, prompt=prompt, context=context)
        relevant_converted = [m for m in converted if m.score >= self.config.relevance_score_threshold]
        gated_scraped = [
            review
            for review, match in zip(scraped, converted)
            if match.score >= self.config.min_lexical_relevance_for_upsert
        ]
        rejected_by_relevance_gate = max(0, len(scraped) - len(gated_scraped))
        step = StepLog(
            module="reviews_agent.web_scrape",
            prompt={
                "prompt": prompt,
                "property_name": self._context_str(context, "property_name"),
                "city": self._context_str(context, "city"),
                "region": self._context_str(context, "region"),
                "max_scrape_reviews": max_scrape,
                "min_lexical_relevance_for_upsert": self.config.min_lexical_relevance_for_upsert,
            },
            response={
                "scraper_status": scraper_meta.get("status"),
                "scraped_raw_count": scraper_meta.get("raw_count", 0),
                "scraped_deduped_count": scraper_meta.get("deduped_count", 0),
                "converted_count": len(converted),
                "relevant_converted_count": len(relevant_converted),
                "upsert_candidate_count": len(gated_scraped),
                "rejected_by_relevance_gate": rejected_by_relevance_gate,
                "attempted_targets": scraper_meta.get("attempted_targets", []),
                "noise_rejection_stats": scraper_meta.get("noise_rejection_stats", {}),
                "errors": scraper_meta.get("errors", []),
            },
        )
        upsert_step = self._upsert_scraped_reviews(scraped_reviews=gated_scraped, context=context)
        return converted, step, upsert_step

    def _upsert_scraped_reviews(self, *, scraped_reviews: list[ScrapedReview], context: dict[str, object]) -> StepLog:
        """Persist scraped reviews in quarantine namespace without blocking final answer on failure."""

        if not scraped_reviews:
            return StepLog(
                module="reviews_agent.web_quarantine_upsert",
                prompt={"attempted": 0},
                response={"status": "skipped", "reason": "No scraped reviews collected"},
            )
        if not self.web_ingest_service.is_available:
            return StepLog(
                module="reviews_agent.web_quarantine_upsert",
                prompt={"attempted": len(scraped_reviews)},
                response={"status": "skipped", "reason": "Web quarantine ingest service unavailable"},
            )
        try:
            result = self.web_ingest_service.upsert_scraped_reviews(
                reviews=scraped_reviews,
                context=context,
            )
            return StepLog(
                module="reviews_agent.web_quarantine_upsert",
                prompt={"attempted": result.attempted},
                response={
                    "status": "ok",
                    "upserted": result.upserted,
                    "namespace": result.namespace,
                    "vector_ids": result.vector_ids[:5],
                },
            )
        except Exception as exc:
            return StepLog(
                module="reviews_agent.web_quarantine_upsert",
                prompt={"attempted": len(scraped_reviews)},
                response={"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            )

    def _convert_scraped_to_matches(
        self,
        *,
        scraped_reviews: list[ScrapedReview],
        prompt: str,
        context: dict[str, object],
    ) -> list[RetrievedReview]:
        """Convert scraped reviews into RetrievedReview objects with semantic relevance scores."""

        converted: list[RetrievedReview] = []
        property_id = self._context_str(context, "property_id")
        region = canonicalize_region(self._context_str(context, "region"))
        for idx, review in enumerate(scraped_reviews, start=1):
            score = self._score_scraped_relevance(prompt=prompt, review_text=review.review_text)
            metadata: dict[str, Any] = {
                "source_type": "web_scrape",
                "source": review.source,
                "source_url": review.source_url,
                "review_text": review.review_text,
                "review_date": review.review_date or "unknown",
                "reviewer_name": review.reviewer_name or "unknown",
                "rating": review.rating,
            }
            if property_id:
                metadata["property_id"] = property_id
            if region:
                metadata["region"] = region
            converted.append(
                RetrievedReview(vector_id=f"web:{review.source}:{idx}", score=score, metadata=metadata)
            )
        return converted

    def _score_scraped_relevance(self, *, prompt: str, review_text: str) -> float:
        """Semantic relevance scoring using embeddings (0..1)."""

        if not self.embedding_service.is_available:
            return 0.5

        try:
            query_embedding = self.embedding_service.embed_query(prompt)
            review_embedding = self.embedding_service.embed_query(review_text)
            return _cosine_similarity(query_embedding, review_embedding)
        except Exception:
            return 0.4

    def _context_str(self, context: dict[str, object], key: str) -> str | None:
        """Safely read optional string values from request context."""

        value = context.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None
