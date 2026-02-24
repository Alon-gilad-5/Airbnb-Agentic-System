#!/usr/bin/env python3
"""Export candidate evidence pool for manual threshold-gold labeling."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load UTF-8 JSONL rows."""

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write UTF-8 JSONL rows."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def retry_call(fn: Any, label: str, max_attempts: int = 5, base_sleep_sec: float = 1.0) -> Any:
    """Retry wrapper for embedding/query network calls."""

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception:
            if attempt >= max_attempts:
                raise
            time.sleep(min(base_sleep_sec * (2 ** (attempt - 1)), 20.0))
    raise RuntimeError(f"Retry failed unexpectedly for {label}")


def metadata_filter_for_case(case: dict[str, Any]) -> dict[str, Any] | None:
    """Build property-aware metadata filter matching ReviewsAgent behavior."""

    property_id = str(case.get("property_id", "")).strip()
    region = str(case.get("region", "")).strip().lower()
    if property_id and region:
        return {"$and": [{"property_id": {"$eq": property_id}}, {"region": {"$eq": region}}]}
    if property_id:
        return {"property_id": {"$eq": property_id}}
    if region:
        return {"region": {"$eq": region}}
    return None


def parse_query_matches(response: Any) -> list[dict[str, Any]]:
    """Normalize Pinecone query response across dict/object SDK shapes."""

    raw_matches = getattr(response, "matches", None)
    if raw_matches is None and isinstance(response, dict):
        raw_matches = response.get("matches", [])
    raw_matches = raw_matches or []

    out: list[dict[str, Any]] = []
    for raw in raw_matches:
        if isinstance(raw, dict):
            vector_id = str(raw.get("id", ""))
            score = float(raw.get("score", 0.0))
            metadata = dict(raw.get("metadata", {}) or {})
        else:
            vector_id = str(getattr(raw, "id", ""))
            score = float(getattr(raw, "score", 0.0))
            metadata = dict(getattr(raw, "metadata", {}) or {})
        out.append({"vector_id": vector_id, "score": score, "metadata": metadata})
    return out


def build_case_candidates(
    *,
    case: dict[str, Any],
    embedding: list[float],
    index: Any,
    namespace: str,
    top_k: int,
    max_review_text_chars: int,
) -> dict[str, Any]:
    """Query Pinecone for one case and emit candidate rows for manual labeling."""

    response = retry_call(
        lambda: index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace,
            filter=metadata_filter_for_case(case),
        ),
        label="pinecone_query",
    )
    normalized = parse_query_matches(response)
    candidates: list[dict[str, Any]] = []
    for m in normalized[:top_k]:
        md = m["metadata"]
        review_text = str(md.get("review_text", "")).strip()
        candidates.append(
            {
                "vector_id": m["vector_id"],
                "score": m["score"],
                "review_text": review_text[:max_review_text_chars],
                "review_id": md.get("review_id"),
                "property_id": md.get("property_id"),
                "region": md.get("region"),
                "review_date": md.get("review_date"),
            }
        )
    return {
        "case_id": case["case_id"],
        "property_id": case.get("property_id"),
        "region": case.get("region"),
        "tier": case.get("tier"),
        "topic": case.get("topic"),
        "prompt": case.get("prompt"),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }


def build_parser() -> argparse.ArgumentParser:
    """CLI for exporting manual-label candidate pool."""

    parser = argparse.ArgumentParser(description="Export retrieval candidates for manual gold labeling.")
    parser.add_argument("--cases-path", default="outputs/reviews_threshold_cases.jsonl")
    parser.add_argument("--output-path", default="outputs/reviews_threshold_label_pool.jsonl")
    parser.add_argument("--namespace", default="airbnb-reviews-test")
    parser.add_argument("--index-name", default=os.getenv("PINECONE_INDEX_NAME", "airbnb-reviews"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-review-text-chars", type=int, default=700)
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small"),
    )
    parser.add_argument(
        "--embedding-deployment",
        default=os.getenv("EMBEDDING_DEPLOYMENT", os.getenv("EMBEDDING_MODEL", "RPRTHPB-text-embedding-3-small")),
    )
    parser.add_argument("--azure-endpoint", default=os.getenv("BASE_URL"))
    return parser


def main() -> None:
    """Run candidate export for manual labeling."""

    load_dotenv()
    args = build_parser().parse_args()
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    embedding_api_key = os.getenv("LLMOD_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not pinecone_api_key:
        raise SystemExit("Missing PINECONE_API_KEY")
    if not embedding_api_key:
        raise SystemExit("Missing LLMOD_API_KEY (or OPENAI_API_KEY)")
    if not args.azure_endpoint:
        raise SystemExit("Missing BASE_URL (or --azure-endpoint)")

    cases = load_jsonl(Path(args.cases_path))
    prompts = [str(case.get("prompt", "")) for case in cases]
    embeddings_client = AzureOpenAIEmbeddings(
        model=args.embedding_model,
        azure_deployment=args.embedding_deployment,
        azure_endpoint=args.azure_endpoint,
        api_key=embedding_api_key,
        check_embedding_ctx_length=False,
    )
    vectors = retry_call(lambda: embeddings_client.embed_documents(prompts), label="embed_documents")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(args.index_name)
    out_rows: list[dict[str, Any]] = []
    for case, embedding in zip(cases, vectors):
        out_rows.append(
            build_case_candidates(
                case=case,
                embedding=embedding,
                index=index,
                namespace=args.namespace,
                top_k=args.top_k,
                max_review_text_chars=args.max_review_text_chars,
            )
        )
    write_jsonl(Path(args.output_path), out_rows)
    print(f"[done] wrote label pool: path={args.output_path} cases={len(out_rows)} top_k={args.top_k}")


if __name__ == "__main__":
    main()

