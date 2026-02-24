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
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class FeatureSpec:
    """Declarative mapping from listing columns to Pinecone metadata keys."""

    source_column: str
    parser: Callable[[str], Any | None]


@dataclass
class IngestStats:
    """Counters used for progress logs and final ingestion summary."""

    rows_seen: int = 0
    rows_embedded: int = 0
    rows_upserted: int = 0
    rows_skipped_empty: int = 0
    rows_skipped_property_filter: int = 0
    rows_enriched_with_listing_fields: int = 0
    rows_missing_listing_enrichment: int = 0
    selected_properties_targeted: int = 0
    ingested_property_ids: set[str] = field(default_factory=set)


@dataclass
class ListingLookupLoadStats:
    """Operational counters from loading listing lookup rows."""

    files_seen: int = 0
    rows_seen: int = 0
    rows_loaded: int = 0
    rows_skipped_missing_id: int = 0
    duplicate_keys: int = 0


@dataclass
class PropertyUniverse:
    """Property-level corpus summary used for deterministic test-slice selection."""

    review_count_by_property: dict[str, int]
    first_seen_order: dict[str, int]
    region_by_property: dict[str, str]


def parse_non_empty_text(raw_value: str) -> str | None:
    """Normalize optional text values and drop empty strings."""

    value = raw_value.strip()
    return value or None


def parse_float_or_none(raw_value: str) -> float | None:
    """Parse optional numeric text into float; return None for invalid/empty values."""

    value = raw_value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


# Source-of-truth mapping for listing metadata enrichment.
# To add fields later, append another entry here (target key -> source column + parser).
LISTING_METADATA_MAP: dict[str, FeatureSpec] = {
    "listing_url": FeatureSpec(source_column="listing_url", parser=parse_non_empty_text),
    "property_name": FeatureSpec(source_column="name", parser=parse_non_empty_text),
    "listing_name": FeatureSpec(source_column="name", parser=parse_non_empty_text),
    "latitude": FeatureSpec(source_column="latitude", parser=parse_float_or_none),
    "longitude": FeatureSpec(source_column="longitude", parser=parse_float_or_none),
    "host_name": FeatureSpec(source_column="host_name", parser=parse_non_empty_text),
}


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


def metadata_scalar_or_none(value: Any) -> str | int | float | bool | None:
    """Return Pinecone-safe scalar values and drop unsupported/empty values."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def iter_listing_files(listings_dir: Path) -> list[Path]:
    """Return sorted regional listing archives for deterministic lookup loading."""

    files = sorted(listings_dir.glob("*data.gz"))
    if not files:
        raise FileNotFoundError(f"No '*data.gz' files found in: {listings_dir}")
    return files


def extract_listing_metadata(
    *,
    listing_row: dict[str, str],
    metadata_map: Mapping[str, FeatureSpec],
) -> dict[str, str | int | float | bool]:
    """Extract configured listing metadata fields using declarative feature specs."""

    metadata: dict[str, str | int | float | bool] = {}
    for target_key, spec in metadata_map.items():
        raw_value = listing_row.get(spec.source_column, "")
        parsed = spec.parser(raw_value)
        clean_value = metadata_scalar_or_none(parsed)
        if clean_value is not None:
            metadata[target_key] = clean_value
    return metadata


def load_listing_lookup(
    *,
    listings_dir: Path,
    metadata_map: Mapping[str, FeatureSpec],
) -> tuple[dict[tuple[str, str], dict[str, str]], ListingLookupLoadStats]:
    """Load listing rows keyed by (region, property_id) for review metadata enrichment."""

    lookup: dict[tuple[str, str], dict[str, str]] = {}
    stats = ListingLookupLoadStats()
    files = iter_listing_files(listings_dir)

    required_columns = {"id"} | {spec.source_column for spec in metadata_map.values()}
    for file_path in files:
        stats.files_seen += 1
        raw_region = file_path.name.replace(" data.gz", "")
        region = canonical_region(raw_region)

        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            missing = sorted(required_columns - fieldnames)
            if missing:
                raise ValueError(f"Missing required listing columns in {file_path.name}: {missing}")

            for row in reader:
                stats.rows_seen += 1
                listing_id = (row.get("id") or "").strip()
                if not listing_id:
                    stats.rows_skipped_missing_id += 1
                    continue
                key = (region, listing_id)
                if key in lookup:
                    stats.duplicate_keys += 1
                # Store the full row so future metadata-map additions do not require lookup shape changes.
                lookup[key] = row
                stats.rows_loaded += 1

    return lookup, stats


def parse_property_ids_file(property_ids_file: Path) -> list[str]:
    """Read property IDs (one-per-line) and preserve deterministic order."""

    if not property_ids_file.exists():
        raise FileNotFoundError(f"Property IDs file not found: {property_ids_file}")

    ids: list[str] = []
    seen: set[str] = set()
    for raw_line in property_ids_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen:
            continue
        seen.add(line)
        ids.append(line)
    return ids


def load_selection_state(selection_state_path: Path) -> dict[str, Any] | None:
    """Load persisted property-selection state; return None when missing."""

    if not selection_state_path.exists():
        return None
    return json.loads(selection_state_path.read_text(encoding="utf-8"))


def save_selection_state(
    selection_state_path: Path,
    *,
    namespace: str,
    selection_order: str,
    selected_property_ids: set[str],
    added_in_run: set[str],
) -> None:
    """Atomically persist selected property IDs for incremental test ingestion."""

    selection_state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "namespace": namespace,
        "selection_order": selection_order,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "cumulative_property_count": len(selected_property_ids),
        "added_in_last_run_count": len(added_in_run),
        "selected_property_ids": sorted(selected_property_ids),
    }
    tmp_path = selection_state_path.with_suffix(selection_state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp_path.replace(selection_state_path)


def build_property_universe(reviews_dir: Path) -> PropertyUniverse:
    """Scan review archives to compute property-level counts and deterministic ordering metadata."""

    review_count_by_property: dict[str, int] = {}
    first_seen_order: dict[str, int] = {}
    region_by_property: dict[str, str] = {}
    seen_counter = 0

    review_files = iter_review_files(reviews_dir)
    for file_path in review_files:
        raw_region = file_path.name.replace(" reviews.gz", "")
        region = canonical_region(raw_region)
        with gzip.open(file_path, "rt", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            expected = {"listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments"}
            if not reader.fieldnames or set(reader.fieldnames) != expected:
                raise ValueError(f"Unexpected schema in {file_path}: {reader.fieldnames}")

            for row in reader:
                property_id = (row.get("listing_id") or "").strip()
                if not property_id:
                    continue
                review_count_by_property[property_id] = review_count_by_property.get(property_id, 0) + 1
                if property_id not in first_seen_order:
                    first_seen_order[property_id] = seen_counter
                    seen_counter += 1
                if property_id not in region_by_property:
                    region_by_property[property_id] = region

    return PropertyUniverse(
        review_count_by_property=review_count_by_property,
        first_seen_order=first_seen_order,
        region_by_property=region_by_property,
    )


def ordered_properties(universe: PropertyUniverse, selection_order: str) -> list[str]:
    """Return deterministic property order for auto/mixed selection strategies."""

    properties = list(universe.review_count_by_property.keys())
    if selection_order == "review_count_desc":
        return sorted(
            properties,
            key=lambda p: (-universe.review_count_by_property[p], p),
        )
    if selection_order == "dataset_order":
        return sorted(
            properties,
            key=lambda p: (universe.first_seen_order[p], p),
        )
    if selection_order == "region_balanced":
        by_region: dict[str, list[str]] = {}
        for property_id in properties:
            region = universe.region_by_property.get(property_id, "unknown")
            by_region.setdefault(region, []).append(property_id)

        for region in by_region:
            by_region[region].sort(key=lambda p: (-universe.review_count_by_property[p], p))

        out: list[str] = []
        region_order = sorted(by_region.keys())
        index_by_region = {region: 0 for region in region_order}
        while True:
            appended_any = False
            for region in region_order:
                idx = index_by_region[region]
                pool = by_region[region]
                if idx >= len(pool):
                    continue
                out.append(pool[idx])
                index_by_region[region] = idx + 1
                appended_any = True
            if not appended_any:
                break
        return out
    raise ValueError(f"Unsupported selection order: {selection_order}")


def resolve_selection_mode(selection_mode: str, property_ids_file: Path | None) -> str:
    """Resolve mixed mode into concrete manual/auto behavior."""

    if selection_mode == "mixed":
        return "manual" if property_ids_file is not None else "auto"
    return selection_mode


def resolve_target_properties(
    *,
    reviews_dir: Path,
    selection_mode: str,
    selection_order: str,
    property_ids_file: Path | None,
    max_properties: int | None,
    selection_state_path: Path,
    namespace: str,
    allow_reingest: bool,
    use_selection_state: bool,
) -> tuple[list[str], set[str], dict[str, Any]]:
    """Resolve final property target list and prior state for incremental ingestion."""

    universe = build_property_universe(reviews_dir)
    state = load_selection_state(selection_state_path) if use_selection_state else None
    previously_selected_ids: set[str] = set()
    if state:
        raw_ns = str(state.get("namespace", "")).strip()
        if raw_ns and raw_ns != namespace:
            raise ValueError(
                f"Selection state namespace mismatch: file has '{raw_ns}', run uses '{namespace}'. "
                "Use a different --selection-state-path or matching --namespace."
            )
        raw_ids = state.get("selected_property_ids", [])
        if isinstance(raw_ids, list):
            previously_selected_ids = {str(item).strip() for item in raw_ids if str(item).strip()}

    mode = resolve_selection_mode(selection_mode, property_ids_file)
    manual_unknown_count = 0
    if mode == "manual":
        if property_ids_file is None:
            raise ValueError("selection_mode=manual requires --property-ids-file")
        base = parse_property_ids_file(property_ids_file)
        manual_unknown_count = len([p for p in base if p not in universe.review_count_by_property])
        base_ids = [p for p in base if p in universe.review_count_by_property]
    elif mode == "auto":
        base_ids = ordered_properties(universe, selection_order)
    else:
        raise ValueError(f"Unsupported resolved selection mode: {mode}")

    if allow_reingest:
        filtered = base_ids
    else:
        filtered = [p for p in base_ids if p not in previously_selected_ids]

    if max_properties is not None and max_properties > 0:
        target = filtered[:max_properties]
    else:
        target = filtered

    debug = {
        "mode": mode,
        "use_selection_state": use_selection_state,
        "universe_property_count": len(universe.review_count_by_property),
        "previously_selected_count": len(previously_selected_ids),
        "candidate_count_after_filter": len(filtered),
        "target_count": len(target),
        "manual_unknown_count": manual_unknown_count,
    }
    return target, previously_selected_ids, debug


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
    listing_lookup: Mapping[tuple[str, str], dict[str, str]],
    listing_metadata_map: Mapping[str, FeatureSpec],
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
    listing_row = listing_lookup.get((region, listing_id))
    if listing_row is not None:
        metadata.update(
            extract_listing_metadata(
                listing_row=listing_row,
                metadata_map=listing_metadata_map,
            )
        )

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
    listings_dir: Path,
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
    selected_property_ids: set[str] | None = None,
) -> IngestStats:
    """Stream files, embed reviews, upsert vectors, and persist resume checkpoints."""
    stats = IngestStats()
    stats.selected_properties_targeted = len(selected_property_ids or set())
    listing_lookup, listing_lookup_stats = load_listing_lookup(
        listings_dir=listings_dir,
        metadata_map=LISTING_METADATA_MAP,
    )
    print(
        "[info] listing lookup loaded: "
        f"files={listing_lookup_stats.files_seen} rows_seen={listing_lookup_stats.rows_seen:,} "
        f"rows_loaded={listing_lookup_stats.rows_loaded:,} "
        f"skipped_missing_id={listing_lookup_stats.rows_skipped_missing_id:,} "
        f"duplicate_keys={listing_lookup_stats.duplicate_keys:,}"
    )
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

                listing_id = (row.get("listing_id") or "").strip()
                if selected_property_ids is not None and listing_id not in selected_property_ids:
                    stats.rows_skipped_property_filter += 1
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
                    listing_lookup=listing_lookup,
                    listing_metadata_map=LISTING_METADATA_MAP,
                )
                if record is None:
                    stats.rows_skipped_empty += 1
                    continue
                stats.ingested_property_ids.add(str(record.metadata["property_id"]))
                if any(key in record.metadata for key in LISTING_METADATA_MAP):
                    stats.rows_enriched_with_listing_fields += 1
                else:
                    listing_key = (
                        str(record.metadata["region"]),
                        str(record.metadata["property_id"]),
                    )
                    if listing_key not in listing_lookup:
                        stats.rows_missing_listing_enrichment += 1
                pending.append(record)

                if len(pending) >= embed_batch_size:
                    flush_pending()

                if progress_every > 0 and stats.rows_seen % progress_every == 0:
                    print(
                        f"[progress] seen={stats.rows_seen:,} embedded={stats.rows_embedded:,} "
                        f"upserted={stats.rows_upserted:,} skipped_empty={stats.rows_skipped_empty:,} "
                        f"skipped_property_filter={stats.rows_skipped_property_filter:,} "
                        f"listing_enriched={stats.rows_enriched_with_listing_fields:,} "
                        f"listing_missing={stats.rows_missing_listing_enrichment:,} "
                        f"unique_properties_ingested={len(stats.ingested_property_ids):,}"
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
    parser.add_argument(
        "--listings-dir",
        default="airbnb_listing_data",
        help="Directory with '*data.gz' listing files used for metadata enrichment.",
    )
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
    parser.add_argument(
        "--property-ids-file",
        default=None,
        help="Optional file with one property_id per line for manual/mixed selection.",
    )
    parser.add_argument(
        "--max-properties",
        type=int,
        default=None,
        help="Optional cap for unique properties selected for ingestion.",
    )
    parser.add_argument(
        "--selection-state-path",
        default="ingest_state/test_property_selection.json",
        help="Path to persisted selected property IDs used for incremental auto-selection.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["auto", "manual", "mixed"],
        default="auto",
        help="Property selection behavior: auto from corpus, manual from file, or mixed fallback.",
    )
    parser.add_argument(
        "--selection-order",
        choices=["review_count_desc", "dataset_order", "region_balanced"],
        default="review_count_desc",
        help="Ordering strategy used by auto/mixed property selection.",
    )
    parser.add_argument(
        "--update-selection-state",
        action="store_true",
        help="Persist selected property IDs after successful ingestion for incremental future runs.",
    )
    parser.add_argument(
        "--allow-reingest",
        action="store_true",
        help="Allow selecting properties already present in selection state.",
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
    property_ids_file = Path(args.property_ids_file) if args.property_ids_file else None
    selection_state_path = Path(args.selection_state_path)

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
        selection_needed = bool(
            args.update_selection_state
            or args.max_properties is not None
            or property_ids_file is not None
            or args.selection_mode != "auto"
            or args.allow_reingest
        )
        selected_property_ids: set[str] | None = None
        previously_selected_ids: set[str] = set()
        if selection_needed:
            selected_property_ids_list, previously_selected_ids, selection_debug = resolve_target_properties(
                reviews_dir=Path(args.reviews_dir),
                selection_mode=args.selection_mode,
                selection_order=args.selection_order,
                property_ids_file=property_ids_file,
                max_properties=args.max_properties,
                selection_state_path=selection_state_path,
                namespace=args.namespace,
                allow_reingest=bool(args.allow_reingest),
                use_selection_state=True,
            )
            selected_property_ids = set(selected_property_ids_list)
            print(
                "[info] property selection: "
                f"mode={selection_debug['mode']} order={args.selection_order} "
                f"use_selection_state={selection_debug['use_selection_state']} "
                f"universe={selection_debug['universe_property_count']:,} "
                f"previously_selected={selection_debug['previously_selected_count']:,} "
                f"candidate_after_filter={selection_debug['candidate_count_after_filter']:,} "
                f"target={selection_debug['target_count']:,} "
                f"manual_unknown={selection_debug['manual_unknown_count']:,}"
            )
        else:
            print("[info] property selection: disabled (full corpus mode)")

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
            listings_dir=Path(args.listings_dir),
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
            selected_property_ids=selected_property_ids,
        )
        if args.update_selection_state:
            merged_selected = set(previously_selected_ids) | set(stats.ingested_property_ids)
            added_in_run = set(stats.ingested_property_ids) - set(previously_selected_ids)
            save_selection_state(
                selection_state_path,
                namespace=args.namespace,
                selection_order=args.selection_order,
                selected_property_ids=merged_selected,
                added_in_run=added_in_run,
            )
            print(
                "[info] selection state updated: "
                f"path={selection_state_path} "
                f"added_in_run={len(added_in_run):,} "
                f"cumulative={len(merged_selected):,}"
            )
        print("[done] ingestion completed")
        print(
            f"[done] seen={stats.rows_seen:,} embedded={stats.rows_embedded:,} "
            f"upserted={stats.rows_upserted:,} skipped_empty={stats.rows_skipped_empty:,} "
            f"skipped_property_filter={stats.rows_skipped_property_filter:,} "
            f"listing_enriched={stats.rows_enriched_with_listing_fields:,} "
            f"listing_missing={stats.rows_missing_listing_enrichment:,} "
            f"selected_properties_targeted={stats.selected_properties_targeted:,} "
            f"unique_properties_ingested={len(stats.ingested_property_ids):,}"
        )


if __name__ == "__main__":
    main()
