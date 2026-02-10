#!/usr/bin/env python3
"""Create a Pinecone index and upsert Airbnb reviews with metadata.

Design notes:
- One vector per review row for high recall on guest-feedback questions.
- Deterministic vector IDs (`region_slug:review_id`) make re-runs idempotent.
- Metadata keeps fields needed for filtering (region/property/date/reviewer).
- Checkpoint cursor (`file + row`) allows safe resume after partial failures.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


REGION_FIXES = {
    "los angels": "los angeles",
    "okland": "oakland",
    "san fransisco": "san francisco",
}


@dataclass
class ReviewRecord:
    """Single review prepared for embedding/upsert with resume cursor fields."""

    vector_id: str
    text_for_embedding: str
    metadata: dict[str, Any]
    source_file: str
    source_row_number: int


@dataclass
class IngestStats:
    """Counters used for progress logs and final ingestion summary."""

    rows_seen: int = 0
    rows_embedded: int = 0
    rows_upserted: int = 0
    rows_skipped_empty: int = 0


def canonical_region(raw_region: str) -> str:
    """Normalize known misspellings to stable region names."""
    lower = raw_region.strip().lower()
    return REGION_FIXES.get(lower, lower)


def title_region(region: str) -> str:
    """Convert canonical lowercase region text into title-cased display text."""
    return " ".join(part.capitalize() for part in region.split())


def slugify(text: str) -> str:
    """Create a stable, URL-safe slug used in deterministic vector IDs."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug


def normalize_review_text(text: str) -> str:
    """Collapse all whitespace so embedding input stays compact and consistent."""
    return " ".join(text.split())


def retry_call(fn: Any, label: str, max_attempts: int = 6, base_sleep_sec: float = 1.5) -> Any:
    """Retry external API calls with bounded exponential backoff."""
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as exc:  # pragma: no cover - external API errors
            if attempt == max_attempts:
                raise
            delay = min(base_sleep_sec * (2 ** (attempt - 1)), 30.0)
            print(f"[warn] {label} failed (attempt {attempt}/{max_attempts}): {exc}")
            print(f"[warn] retrying in {delay:.1f}s")
            time.sleep(delay)
    raise RuntimeError("retry_call exhausted unexpectedly")


def list_index_names(pc: Pinecone) -> set[str]:
    """Handle shape differences across Pinecone SDK versions."""
    raw = pc.list_indexes()
    if hasattr(raw, "names"):
        return set(raw.names())
    if isinstance(raw, list):
        names: set[str] = set()
        for item in raw:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict) and "name" in item:
                names.add(str(item["name"]))
        return names
    if hasattr(raw, "indexes"):
        names = set()
        for item in raw.indexes:
            name = getattr(item, "name", None)
            if name:
                names.add(str(name))
        return names
    raise RuntimeError(f"Unexpected list_indexes() response type: {type(raw)}")


def ensure_index(
    pc: Pinecone,
    index_name: str,
    dimension: int,
    metric: str,
    cloud: str,
    region: str,
) -> None:
    """Create index if missing and block until the index reports ready."""
    existing = list_index_names(pc)
    if index_name not in existing:
        print(
            f"[info] creating index '{index_name}' (dim={dimension}, metric={metric}, cloud={cloud}, region={region})"
        )
        retry_call(
            lambda: pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            ),
            label="create_index",
        )
    else:
        print(f"[info] index '{index_name}' already exists")

    for _ in range(60):
        desc = retry_call(lambda: pc.describe_index(index_name), label="describe_index")
        status = getattr(desc, "status", None)
        ready = False
        if isinstance(status, dict):
            ready = bool(status.get("ready"))
        elif status is not None:
            ready = bool(getattr(status, "ready", False))
        if ready:
            print(f"[info] index '{index_name}' is ready")
            return
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for index '{index_name}' readiness")


def record_from_row(
    row: dict[str, str],
    raw_region: str,
    source_file: str,
    source_row_number: int,
    max_metadata_chars: int,
) -> ReviewRecord | None:
    """Convert a CSV row into a review record for embedding + upsert."""
    listing_id = row["listing_id"].strip()
    review_id = row["id"].strip()
    review_date = row["date"].strip()
    reviewer_id = row["reviewer_id"].strip()
    reviewer_name = row["reviewer_name"].strip()
    comments_raw = row["comments"].strip()

    if not comments_raw:
        return None

    comments = normalize_review_text(comments_raw)
    region = canonical_region(raw_region)
    region_slug = slugify(region)
    location = title_region(region)
    # Region prefix prevents accidental collisions if review IDs overlap across files.
    vector_id = f"{region_slug}:{review_id}"

    year = None
    if len(review_date) >= 4 and review_date[:4].isdigit():
        year = int(review_date[:4])

    metadata: dict[str, Any] = {
        "region_raw": raw_region,
        "region": region,
        "location": location,
        "property_id": listing_id,
        "review_id": review_id,
        "review_date": review_date,
        "reviewer_id": reviewer_id,
        "reviewer_name": reviewer_name,
        "source_file": source_file,
        "review_char_count": len(comments),
        "review_word_count": len(comments.split()),
        "contains_html_break": "<br" in comments.lower(),
        # Keep metadata lightweight while preserving enough source text for inspection/debugging.
        "review_text": comments[:max_metadata_chars],
    }
    if year is not None:
        metadata["year"] = year

    # Keep embeddings focused on review semantics; use metadata for filters/sorting.
    text_for_embedding = comments
    return ReviewRecord(
        vector_id=vector_id,
        text_for_embedding=text_for_embedding,
        metadata=metadata,
        source_file=source_file,
        source_row_number=source_row_number,
    )


def embed_texts(embeddings_client: AzureOpenAIEmbeddings, texts: list[str]) -> list[list[float]]:
    """Embed a batch of review texts using the configured LLMOD/Azure endpoint."""
    # Uses LLMOD's OpenAI-compatible Azure endpoint.
    return retry_call(
        lambda: embeddings_client.embed_documents(texts),
        label="AzureOpenAIEmbeddings.embed_documents",
    )


def upsert_batch(index: Any, namespace: str, vectors: list[dict[str, Any]]) -> None:
    """Upsert vectors to Pinecone with retry logic for transient failures."""
    retry_call(lambda: index.upsert(vectors=vectors, namespace=namespace), label="pinecone.upsert")


def iter_review_files(reviews_dir: Path) -> list[Path]:
    """Return sorted regional review archives to keep ingestion order deterministic."""
    files = sorted(reviews_dir.glob("*reviews.gz"))
    if not files:
        raise FileNotFoundError(f"No '*reviews.gz' files found in: {reviews_dir}")
    return files


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load persisted resume cursor; return None when no checkpoint exists."""
    if not checkpoint_path.exists():
        return None
    return json.loads(checkpoint_path.read_text(encoding="utf-8"))


def save_checkpoint(
    checkpoint_path: Path,
    *,
    index_name: str,
    namespace: str,
    embedding_model: str,
    file_name: str,
    row_number: int,
    rows_upserted: int,
) -> None:
    """Atomically persist latest committed cursor after successful upsert chunks."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "index_name": index_name,
        "namespace": namespace,
        "embedding_model": embedding_model,
        "cursor_file": file_name,
        "cursor_row": row_number,
        "rows_upserted": rows_upserted,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # Atomic replace prevents partial checkpoint files on interruption.
    tmp_path.replace(checkpoint_path)


def should_skip_row(
    *,
    checkpoint: dict[str, Any] | None,
    file_name: str,
    row_number: int,
    ordered_files: list[str],
) -> bool:
    """Return True for rows already committed in the checkpoint cursor."""
    if checkpoint is None:
        return False
    cursor_file = checkpoint.get("cursor_file")
    cursor_row = int(checkpoint.get("cursor_row", 0))
    if cursor_file not in ordered_files:
        return False

    cursor_idx = ordered_files.index(cursor_file)
    file_idx = ordered_files.index(file_name)
    if file_idx < cursor_idx:
        return True
    if file_idx == cursor_idx and row_number <= cursor_row:
        return True
    return False


def ingest_reviews(
    reviews_dir: Path,
    pinecone_index: Any,
    index_name: str,
    namespace: str,
    embeddings_client: AzureOpenAIEmbeddings,
    embedding_model: str,
    embed_batch_size: int,
    upsert_batch_size: int,
    max_metadata_chars: int,
    max_reviews: int | None,
    progress_every: int,
    checkpoint_path: Path,
    resume: bool,
    reset_checkpoint: bool,
) -> IngestStats:
    """Stream files, embed reviews, upsert vectors, and persist resume checkpoints."""
    stats = IngestStats()
    pending: list[ReviewRecord] = []
    review_files = iter_review_files(reviews_dir)
    ordered_file_names = [p.name for p in review_files]

    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"[info] deleted checkpoint: {checkpoint_path}")

    checkpoint = load_checkpoint(checkpoint_path) if resume else None
    if checkpoint:
        # Guard against applying a cursor from a different target/index setup.
        mismatch = (
            checkpoint.get("index_name") != index_name
            or checkpoint.get("namespace") != namespace
            or checkpoint.get("embedding_model") != embedding_model
        )
        if mismatch:
            print(
                "[warn] checkpoint context mismatch (index/namespace/model). "
                "Ignoring checkpoint for this run."
            )
            checkpoint = None
        else:
            print(
                f"[info] resume enabled from checkpoint: file={checkpoint.get('cursor_file')} "
                f"row={checkpoint.get('cursor_row')}"
            )

    def flush_pending() -> None:
        """Flush in-memory records: embed batch, chunked upsert, then checkpoint."""
        nonlocal pending
        if not pending:
            return
        # Embed first, then upsert in smaller chunks to stay within request limits.
        embeddings = embed_texts(
            embeddings_client=embeddings_client,
            texts=[r.text_for_embedding for r in pending],
        )
        stats.rows_embedded += len(embeddings)
        vectors: list[dict[str, Any]] = []
        for record, embedding in zip(pending, embeddings):
            vectors.append(
                {
                    "id": record.vector_id,
                    "values": embedding,
                    "metadata": record.metadata,
                }
            )
        for i in range(0, len(vectors), upsert_batch_size):
            chunk_vectors = vectors[i : i + upsert_batch_size]
            chunk_records = pending[i : i + upsert_batch_size]
            upsert_batch(index=pinecone_index, namespace=namespace, vectors=chunk_vectors)
            stats.rows_upserted += len(chunk_vectors)
            # Cursor is updated only after successful upsert of this chunk.
            last_record = chunk_records[-1]
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                index_name=index_name,
                namespace=namespace,
                embedding_model=embedding_model,
                file_name=last_record.source_file,
                row_number=last_record.source_row_number,
                rows_upserted=stats.rows_upserted,
            )
        pending = []

    for file_path in review_files:
        raw_region = file_path.name.replace(" reviews.gz", "")
        print(f"[info] processing {file_path.name}")
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"}
            if not reader.fieldnames or set(reader.fieldnames) != expected:
                raise ValueError(f"Unexpected schema in {file_path}: {reader.fieldnames}")

            for row_number, row in enumerate(reader, start=1):
                # Skip rows already committed in prior runs.
                if should_skip_row(
                    checkpoint=checkpoint,
                    file_name=file_path.name,
                    row_number=row_number,
                    ordered_files=ordered_file_names,
                ):
                    continue

                stats.rows_seen += 1

                if max_reviews is not None and max_reviews > 0:
                    # Hard budget cap: stop after committing currently buffered rows.
                    if stats.rows_embedded + len(pending) >= max_reviews:
                        flush_pending()
                        return stats

                record = record_from_row(
                    row=row,
                    raw_region=raw_region,
                    source_file=file_path.name,
                    source_row_number=row_number,
                    max_metadata_chars=max_metadata_chars,
                )
                if record is None:
                    stats.rows_skipped_empty += 1
                    continue
                pending.append(record)

                if len(pending) >= embed_batch_size:
                    flush_pending()

                if progress_every > 0 and stats.rows_seen % progress_every == 0:
                    print(
                        f"[progress] seen={stats.rows_seen:,} embedded={stats.rows_embedded:,} "
                        f"upserted={stats.rows_upserted:,} skipped_empty={stats.rows_skipped_empty:,}"
                    )
        # Flush at file boundaries so resume cursors always map to committed rows.
        flush_pending()

    flush_pending()
    return stats


def build_parser() -> argparse.ArgumentParser:
    """Build CLI for index creation, budgeted upserting, and resume control."""
    parser = argparse.ArgumentParser(
        description="Create Pinecone index and upsert Airbnb review vectors with metadata."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "create-index", "upsert"],
        default="all",
        help="Run index creation, upsert, or both.",
    )
    parser.add_argument("--reviews-dir", default="airbnb_reviews", help="Directory with '*reviews.gz' files.")
    parser.add_argument("--index-name", default="airbnb-reviews", help="Pinecone index name.")
    parser.add_argument(
        "--namespace",
        default="airbnb-reviews",
        help="Pinecone namespace (useful for versioned re-indexing).",
    )
    parser.add_argument("--dimension", type=int, default=1536, help="Embedding dimension.")
    parser.add_argument("--metric", default="cosine", help="Pinecone metric (cosine, dotproduct, euclidean).")
    parser.add_argument("--cloud", default="aws", help="Pinecone serverless cloud.")
    parser.add_argument("--region", default="us-east-1", help="Pinecone serverless region.")
    parser.add_argument(
        "--embedding-model",
        default="RPRTHPB-text-embedding-3-small",
        help="Embedding model name passed to AzureOpenAIEmbeddings.",
    )
    parser.add_argument(
        "--azure-endpoint",
        default=None,
        help="Optional Azure endpoint/base URL (defaults to BASE_URL env var).",
    )
    parser.add_argument(
        "--embedding-deployment",
        default=None,
        help="Azure deployment name (defaults to --embedding-model).",
    )
    parser.add_argument("--embed-batch-size", type=int, default=100, help="Batch size for embedding requests.")
    parser.add_argument("--upsert-batch-size", type=int, default=100, help="Batch size for Pinecone upserts.")
    parser.add_argument(
        "--max-metadata-chars",
        type=int,
        default=1200,
        help="Max review text chars stored in metadata under review_text.",
    )
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=100,
        help="Budget-safe cap for rows to embed/upsert in this run. Use -1 for full run.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N rows seen (0 to disable).",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="ingest_state/reviews_ingest_checkpoint.json",
        help="Checkpoint file used to resume after partial failures.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and start from the first row.",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete checkpoint before running.",
    )
    return parser


def main() -> None:
    """Resolve config from env/CLI, then run index creation and/or ingestion."""
    load_dotenv()
    args = build_parser().parse_args()

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # LLMOD key is used for embedding calls routed via AzureOpenAIEmbeddings.
    embedding_api_key = os.getenv("LLMOD_API_KEY") or os.getenv("OPENAI_API_KEY")
    azure_endpoint = (
        args.azure_endpoint
        or os.getenv("BASE_URL")
        or os.getenv("LLMOD_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    embedding_deployment = args.embedding_deployment or args.embedding_model
    max_reviews = None if args.max_reviews == -1 else args.max_reviews
    resume = not args.no_resume

    if not pinecone_api_key:
        raise SystemExit("Missing PINECONE_API_KEY in environment/.env")
    if args.mode in {"all", "upsert"} and not embedding_api_key:
        raise SystemExit("Missing LLMOD_API_KEY (or OPENAI_API_KEY) in environment/.env")
    if args.mode in {"all", "upsert"} and not azure_endpoint:
        raise SystemExit("Missing BASE_URL (or --azure-endpoint) for AzureOpenAIEmbeddings")

    pc = Pinecone(api_key=pinecone_api_key)

    if args.mode in {"all", "create-index"}:
        ensure_index(
            pc=pc,
            index_name=args.index_name,
            dimension=args.dimension,
            metric=args.metric,
            cloud=args.cloud,
            region=args.region,
        )

    if args.mode in {"all", "upsert"}:
        index = pc.Index(args.index_name)
        print(f"[info] azure endpoint: {azure_endpoint}")
        print(f"[info] embedding deployment: {embedding_deployment}")
        # Same configuration style used in your previous working LLMOD setup.
        embeddings_client = AzureOpenAIEmbeddings(
            model=args.embedding_model,
            azure_deployment=embedding_deployment,
            azure_endpoint=azure_endpoint,
            api_key=embedding_api_key,
            check_embedding_ctx_length=False,
        )
        stats = ingest_reviews(
            reviews_dir=Path(args.reviews_dir),
            pinecone_index=index,
            index_name=args.index_name,
            namespace=args.namespace,
            embeddings_client=embeddings_client,
            embedding_model=args.embedding_model,
            embed_batch_size=args.embed_batch_size,
            upsert_batch_size=args.upsert_batch_size,
            max_metadata_chars=args.max_metadata_chars,
            max_reviews=max_reviews,
            progress_every=args.progress_every,
            checkpoint_path=Path(args.checkpoint_path),
            resume=resume,
            reset_checkpoint=args.reset_checkpoint,
        )
        print("[done] ingestion completed")
        print(
            f"[done] seen={stats.rows_seen:,} embedded={stats.rows_embedded:,} "
            f"upserted={stats.rows_upserted:,} skipped_empty={stats.rows_skipped_empty:,}"
        )


if __name__ == "__main__":
    main()
