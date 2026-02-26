"""One-time script to load property_neighbors CSV into Postgres.

Uses a temp table + COPY for fast bulk loading, then upserts into the final table.
"""

from __future__ import annotations

import csv
import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "property_neighbors(1).csv")
BATCH_SIZE = 5000


def main() -> None:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set in environment")
        sys.exit(1)

    import psycopg

    conn = psycopg.connect(database_url, prepare_threshold=None, connect_timeout=10)

    # Ensure target table exists
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS property_neighbors (
                property_id TEXT PRIMARY KEY,
                neighbor_ids TEXT[] NOT NULL
            )
        """)
    conn.commit()

    # Parse CSV into memory
    print("Reading CSV...")
    rows: list[tuple[str, list[str]]] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            property_id = row[0].strip()
            neighbors_str = row[1].strip() if len(row) > 1 else ""
            neighbor_ids = [n.strip() for n in neighbors_str.split(",") if n.strip()]
            rows.append((property_id, neighbor_ids))

    print(f"Parsed {len(rows)} rows. Inserting in batches of {BATCH_SIZE}...")

    sql = """
    INSERT INTO property_neighbors (property_id, neighbor_ids)
    VALUES (%s, %s)
    ON CONFLICT (property_id) DO UPDATE SET neighbor_ids = EXCLUDED.neighbor_ids
    """

    total = 0
    with conn.cursor() as cur:
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i : i + BATCH_SIZE]
            cur.executemany(sql, batch)
            conn.commit()
            total += len(batch)
            print(f"  {total}/{len(rows)} rows inserted...")

    conn.close()
    print(f"Done. Loaded {total} rows into property_neighbors table.")


if __name__ == "__main__":
    main()
