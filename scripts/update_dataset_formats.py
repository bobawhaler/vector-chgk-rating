"""Script to reclassify tournament_type for team participations played at online venues."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chgk_rating.data_loader import is_online_venue

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = Path("data/chgk_cache.db")
PARQUET_PATH = Path("data/dataset.parquet")


def main() -> None:
    if not DB_PATH.exists() or not PARQUET_PATH.exists():
        logger.error("Database or dataset parquet not found.")
        return

    logger.info("Opening cache database %s...", DB_PATH)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    logger.info("Querying tournament results from api_cache...")
    c.execute("SELECT url, response_json FROM api_cache WHERE url LIKE '%/results%'")
    rows = c.fetchall()
    logger.info("Fetched %d tournament result records from cache.", len(rows))

    online_pairs = set()
    for url, rjson in rows:
        if not rjson:
            continue
        try:
            tid_part = url.split("/tournaments/")[1].split("/results")[0]
            tid = int(tid_part)
            data = json.loads(rjson)
            if isinstance(data, list):
                for t in data:
                    team_obj = t.get("team", {})
                    team_id = team_obj.get("id") or t.get("idTeam")
                    if not team_id:
                        continue
                    team_id = int(team_id)
                    sr = t.get("synchRequest") or {}
                    vname = (sr.get("venue") or {}).get("name")
                    if is_online_venue(vname):
                        online_pairs.add((tid, team_id))
        except Exception:
            pass

    logger.info("Identified %d unique (tournament_id, team_id) online participations.", len(online_pairs))

    logger.info("Reading parquet dataset from %s...", PARQUET_PATH)
    parquet_file = pq.ParquetFile(PARQUET_PATH)
    schema = parquet_file.schema_arrow

    temp_output_path = PARQUET_PATH.with_suffix(".temp.parquet")
    writer = pq.ParquetWriter(temp_output_path, schema)

    total_rows = 0
    updated_rows = 0

    for i in range(parquet_file.num_row_groups):
        df_chunk = parquet_file.read_row_group(i).to_pandas()
        
        chunk_pairs = list(zip(df_chunk["tournament_id"], df_chunk["team_id"]))
        mask_to_online = [
            (pair in online_pairs and cur_type != "online")
            for pair, cur_type in zip(chunk_pairs, df_chunk["tournament_type"])
        ]
        
        num_changed = sum(mask_to_online)
        if num_changed > 0:
            df_chunk.loc[mask_to_online, "tournament_type"] = "online"
            updated_rows += num_changed

        total_rows += len(df_chunk)
        table = pa.Table.from_pandas(df_chunk, schema=schema, preserve_index=False)
        writer.write_table(table)

        if (i + 1) % 5 == 0 or (i + 1) == parquet_file.num_row_groups:
            logger.info("Processed row group %d/%d (total rows: %d, updated: %d)",
                        i + 1, parquet_file.num_row_groups, total_rows, updated_rows)

    writer.close()
    logger.info("Finished writing. Replacing %s with updated dataset...", PARQUET_PATH)
    temp_output_path.replace(PARQUET_PATH)
    logger.info("Dataset updated successfully! Total rows: %d, Updated to online: %d", total_rows, updated_rows)


if __name__ == "__main__":
    main()
