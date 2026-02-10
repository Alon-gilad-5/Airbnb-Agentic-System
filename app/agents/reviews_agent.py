"""Domain agent that answers questions from Airbnb guest reviews."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from app.agents.base import Agent, AgentResult
from app.schemas import StepLog
from app.services.chat_service import ChatService
from app.services.embeddings import EmbeddingService
from app.services.pinecone_retriever import PineconeRetriever, RetrievedReview
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


class ReviewsAgent(Agent):
    """Retrieval-augmented QA agent over review vectors + metadata."""

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

    def run(self, prompt: str, context: dict[str, object] | None = None) -> AgentResult:
        """Execute retrieval and response synthesis with full trace steps."""

        context = context or {}
        steps: list[StepLog] = []
        metadata_filter = self._build_metadata_filter(prompt, context)

        if not self.embedding_service.is_available:
            return AgentResult(
                response=(
                    "Embedding service is not configured. Set LLMOD_API_KEY and BASE_URL "
                    "to enable vector search."
                ),
                steps=steps,
            )
        if not self.retriever.is_available:
            return AgentResult(
                response="Pinecone is not configured. Set PINECONE_API_KEY to enable retrieval.",
                steps=steps,
            )

        try:
            query_embedding = self.embedding_service.embed_query(prompt)
            matches = self.retriever.query(
                embedding=query_embedding,
                top_k=self.config.top_k,
                metadata_filter=metadata_filter,
            )
        except Exception as exc:
            # Graceful failure keeps API schema stable and avoids top-level 500-like behavior.
            steps.append(
                StepLog(
                    module="reviews_agent.retrieval",
                    prompt={
                        "user_prompt": prompt,
                        "top_k": self.config.top_k,
                        "metadata_filter": metadata_filter,
                    },
                    response={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            return AgentResult(
                response=(
                    "I could not access retrieval services right now. "
                    "Please verify LLMOD/Pinecone connectivity and try again."
                ),
                steps=steps,
            )

        retrieval_step = StepLog(
            module="reviews_agent.retrieval",
            prompt={
                "user_prompt": prompt,
                "top_k": self.config.top_k,
                "metadata_filter": metadata_filter,
            },
            response={
                "match_count": len(matches),
                "top_match_ids": [m.vector_id for m in matches[:5]],
            },
        )
        steps.append(retrieval_step)

        # Decide if we should activate web fallback: no matches or weak top score.
        top_score = matches[0].score if matches else 0.0
        fallback_triggered = (not matches) or (top_score < self.config.relevance_score_threshold)
        web_matches: list[RetrievedReview] = []
        if fallback_triggered and self.web_scraper.is_available:
            web_matches, scrape_step, upsert_step = self._scrape_fallback(prompt=prompt, context=context)
            steps.append(scrape_step)
            steps.append(upsert_step)
        elif fallback_triggered:
            steps.append(
                StepLog(
                    module="reviews_agent.web_scrape",
                    prompt={"triggered": True, "reason": "low_internal_relevance", "top_score": top_score},
                    response={"status": "skipped", "reason": "Playwright scraper is disabled/unavailable"},
                )
            )
            steps.append(
                StepLog(
                    module="reviews_agent.web_quarantine_upsert",
                    prompt={"attempted": 0},
                    response={"status": "skipped", "reason": "No scraped reviews to upsert"},
                )
            )

        # Merge internal + scraped evidence and apply relevance threshold consistently.
        # In this version, scraped reviews are assigned pseudo-scores where 1.0 is strong lexical relevance.
        relevant_matches = [m for m in matches if m.score >= self.config.relevance_score_threshold]
        relevant_matches.extend([m for m in web_matches if m.score >= self.config.relevance_score_threshold])
        relevant_evidence_count = self._count_relevant_evidence(relevant_matches)
        evidence_step, should_answer, disclaimer_prefix = self._apply_evidence_guard(
            relevant_evidence_count=relevant_evidence_count
        )
        steps.append(evidence_step)
        if not should_answer:
            return AgentResult(response=NO_EVIDENCE_RESPONSE, steps=steps)

        evidence_for_context = relevant_matches[: self.config.max_context_reviews]
        context = self._build_context(evidence_for_context)
        if not self.chat_service.is_available:
            # Fallback keeps the endpoint usable even if chat model config is missing.
            final_answer = self._deterministic_summary(evidence_for_context)
            final_answer = self._finalize_answer(
                answer_text=final_answer,
                evidence=evidence_for_context,
                disclaimer_prefix=disclaimer_prefix,
                evidence_count=relevant_evidence_count,
                steps=steps,
            )
            return AgentResult(response=final_answer, steps=steps)

        system_prompt = (
            "You are a hospitality insights assistant. "
            "Answer only from provided review evidence. "
            "If evidence is weak, say so explicitly. "
            "Do not use unsupported broad quantifiers (e.g., many guests, most guests, a lot of reviews) "
            "unless that is clearly supported by the provided evidence."
        )
        user_prompt = (
            f"Question:\n{prompt}\n\n"
            f"Review Evidence:\n{context}\n\n"
            f"Relevant evidence count: {relevant_evidence_count}\n"
            "Confidence policy:\n"
            "- 1-2 relevant reviews => confidence must be low.\n"
            "- 3+ relevant reviews => confidence can be medium/high only if supported.\n\n"
            "Return a concise business-friendly answer with:\n"
            "1) direct answer\n2) key evidence bullets\n3) confidence (high/medium/low)\n"
            "4) explicit source-based citations"
        )
        try:
            llm_answer = self.chat_service.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        except Exception as exc:
            steps.append(
                StepLog(
                    module="reviews_agent.answer_generation",
                    prompt={
                        "model": self.chat_service.model,
                        "system_prompt": system_prompt,
                        "user_prompt": user_prompt,
                    },
                    response={"error": f"{type(exc).__name__}: {exc}"},
                )
            )
            return AgentResult(
                response=(
                    "I retrieved relevant reviews but could not generate the final summary right now. "
                    "Please retry in a moment."
                ),
                steps=steps,
            )

        answer_step = StepLog(
            module="reviews_agent.answer_generation",
            prompt={
                "model": self.chat_service.model,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            },
            response={"text": llm_answer},
        )
        if not llm_answer.strip():
            llm_answer = self._deterministic_summary(evidence_for_context)
            answer_step.response["fallback_used"] = True
            answer_step.response["fallback_reason"] = "empty_model_output"
            answer_step.response["text"] = llm_answer
        steps.append(answer_step)

        final_answer = self._finalize_answer(
            answer_text=llm_answer,
            evidence=evidence_for_context,
            disclaimer_prefix=disclaimer_prefix,
            evidence_count=relevant_evidence_count,
            steps=steps,
        )
        return AgentResult(response=final_answer, steps=steps)

    def _build_metadata_filter(self, prompt: str, context: dict[str, object]) -> dict[str, Any] | None:
        """Optional heuristic filters to reduce retrieval scope when prompt names a region."""

        if context.get("region"):
            return {"region": {"$eq": str(context["region"]).lower()}}

        lowered = prompt.lower()
        known_regions = [
            "los angeles",
            "oakland",
            "pacific grove",
            "san diego",
            "san francisco",
            "san mateo",
            "santa clara",
            "santa cruz county",
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
        step = StepLog(
            module="reviews_agent.web_scrape",
            prompt={
                "prompt": prompt,
                "property_name": self._context_str(context, "property_name"),
                "city": self._context_str(context, "city"),
                "region": self._context_str(context, "region"),
                "max_scrape_reviews": max_scrape,
            },
            response={
                "scraper_status": scraper_meta.get("status"),
                "scraped_raw_count": scraper_meta.get("raw_count", 0),
                "scraped_deduped_count": scraper_meta.get("deduped_count", 0),
                "converted_count": len(converted),
                "relevant_converted_count": len(relevant_converted),
                "errors": scraper_meta.get("errors", []),
            },
        )
        upsert_step = self._upsert_scraped_reviews(scraped_reviews=scraped, context=context)
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
        """Convert scraped reviews into RetrievedReview objects with lexical relevance scores."""

        converted: list[RetrievedReview] = []
        property_id = self._context_str(context, "property_id") or "unknown"
        region = (self._context_str(context, "region") or "unknown").lower()
        for idx, review in enumerate(scraped_reviews, start=1):
            score = self._score_scraped_relevance(prompt=prompt, review_text=review.review_text)
            metadata = {
                "source_type": "web_scrape",
                "source": review.source,
                "source_url": review.source_url,
                "review_text": review.review_text,
                "review_date": review.review_date or "unknown",
                "reviewer_name": review.reviewer_name or "unknown",
                "rating": review.rating,
                "property_id": property_id,
                "region": region,
            }
            converted.append(
                RetrievedReview(
                    vector_id=f"web:{review.source}:{idx}",
                    score=score,
                    metadata=metadata,
                )
            )
        return converted

    def _score_scraped_relevance(self, *, prompt: str, review_text: str) -> float:
        """Simple lexical relevance scoring for scraped text (0..1)."""

        query_tokens = self._tokenize(prompt)
        if not query_tokens:
            return 0.0
        review_tokens = self._tokenize(review_text)
        if not review_tokens:
            return 0.0
        overlap = len(query_tokens & review_tokens)
        ratio = overlap / len(query_tokens)
        # Keep score bounded and conservative; evidence guard still applies.
        return min(1.0, ratio)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize text for lightweight lexical matching."""

        tokens = re.findall(r"[a-zA-Z]{3,}", text.lower())
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "what",
            "your",
            "about",
            "have",
            "has",
            "are",
            "was",
            "were",
            "hotel",
            "guest",
            "guests",
            "review",
            "reviews",
        }
        return {t for t in tokens if t not in stopwords}

    def _context_str(self, context: dict[str, object], key: str) -> str | None:
        """Safely read optional string values from request context."""

        value = context.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    def _build_context(self, matches: list[RetrievedReview]) -> str:
        """Convert top vector matches into compact evidence snippets for the LLM."""

        snippets: list[str] = []
        for i, match in enumerate(matches[: self.config.max_context_reviews], start=1):
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

    def _count_relevant_evidence(self, relevant_matches: list[RetrievedReview]) -> int:
        """Count relevant evidence after score-threshold filtering."""

        return len(relevant_matches)

    def _apply_evidence_guard(self, relevant_evidence_count: int) -> tuple[StepLog, bool, str | None]:
        """Apply no-evidence and thin-evidence policies before answer generation."""

        if relevant_evidence_count == 0:
            decision = "fallback_no_evidence"
            should_answer = False
            disclaimer = None
        elif self.config.thin_evidence_min <= relevant_evidence_count <= self.config.thin_evidence_max:
            decision = "answer_with_low_evidence_disclaimer"
            should_answer = True
            disclaimer = LOW_EVIDENCE_PREFIX
        else:
            decision = "answer_normal"
            should_answer = True
            disclaimer = None

        step = StepLog(
            module="reviews_agent.evidence_guard",
            prompt={
                "relevance_score_threshold": self.config.relevance_score_threshold,
                "thin_evidence_min": self.config.thin_evidence_min,
                "thin_evidence_max": self.config.thin_evidence_max,
            },
            response={
                "decision": decision,
                "relevant_evidence_count": relevant_evidence_count,
            },
        )
        return step, should_answer, disclaimer

    def _detect_hallucination_risk(self, answer_text: str) -> tuple[bool, list[str]]:
        """Flag broad claims that are often unsupported by thin evidence."""

        phrases = [
            "many guests",
            "a lot of reviews",
            "most guests",
            "guests generally",
            "everyone",
        ]
        lowered = answer_text.lower()
        matched = [phrase for phrase in phrases if re.search(rf"\b{re.escape(phrase)}\b", lowered)]
        return bool(matched), matched

    def _build_citations(self, evidence: list[RetrievedReview]) -> list[str]:
        """Generate citations for both internal and scraped evidence sources."""

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
            if len(citations) >= self.config.max_citations:
                break
        return citations

    def _append_citations(self, answer_text: str, citations: list[str]) -> str:
        """Attach standardized citation block to the final answer."""

        if not citations:
            return answer_text
        lines = [answer_text, "", "Citations:"]
        lines.extend([f"- {citation}" for citation in citations])
        return "\n".join(lines)

    def _finalize_answer(
        self,
        *,
        answer_text: str,
        evidence: list[RetrievedReview],
        disclaimer_prefix: str | None,
        evidence_count: int,
        steps: list[StepLog],
    ) -> str:
        """Finalize answer with disclaimer, citations, and hallucination guard step."""

        final_answer = answer_text.strip()
        final_answer = self._cap_words(final_answer, self.config.max_answer_words)
        if disclaimer_prefix:
            final_answer = f"{disclaimer_prefix}\n\n{final_answer}"

        citations = self._build_citations(evidence)
        final_answer = self._append_citations(final_answer, citations)

        risk_flag, matched_phrases = self._detect_hallucination_risk(final_answer)
        steps.append(
            StepLog(
                module="reviews_agent.hallucination_guard",
                prompt={
                    "checked_phrases": [
                        "many guests",
                        "a lot of reviews",
                        "most guests",
                        "guests generally",
                        "everyone",
                    ],
                },
                response={
                    "risk_flag": risk_flag,
                    "matched_phrases": matched_phrases,
                    "evidence_count": evidence_count,
                    "action": "flag_only",
                },
            )
        )
        return final_answer

    def _deterministic_summary(self, matches: list[RetrievedReview]) -> str:
        """Non-LLM fallback summary used when model output is unavailable/empty."""

        lines = ["Based on the strongest matching reviews, here is a concise evidence summary:"]
        for i, match in enumerate(matches[:3], start=1):
            md = match.metadata
            text = str(md.get("review_text", "")).strip()
            region = str(md.get("location", md.get("region", "unknown")))
            property_id = str(md.get("property_id", "unknown"))
            short = text[:140] + ("..." if len(text) > 140 else "")
            lines.append(f"{i}. (score={match.score:.4f}, region={region}, property_id={property_id}) {short}")
        return "\n".join(lines)

    def _cap_words(self, text: str, max_words: int) -> str:
        """Cap generated text length by word count to control token usage."""

        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words]) + " ..."
