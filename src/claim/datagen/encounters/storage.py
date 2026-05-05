from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import sqlite3
from typing import Any

from claim.datagen import io as datagen_io
from claim.datagen.encounters import constants, dataset, matching, parsers
from claim.datagen.encounters.contracts import DatasetFileSpec
from claim.cli_logging import CliLogger


def initialize_build_database(path: Path) -> sqlite3.Connection:
    if path.exists():
        path.unlink()

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute(
        """
        CREATE TABLE beneficiaries (
            desynpuf_id TEXT NOT NULL,
            year INTEGER NOT NULL,
            payload TEXT NOT NULL,
            PRIMARY KEY (desynpuf_id, year)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE anchors (
            cluster_id TEXT PRIMARY KEY,
            desynpuf_id TEXT NOT NULL,
            anchor_year INTEGER,
            from_date TEXT,
            thru_date TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX anchors_desynpuf_idx ON anchors (desynpuf_id)")
    connection.execute(
        """
        CREATE TABLE carriers (
            row_id INTEGER PRIMARY KEY AUTOINCREMENT,
            desynpuf_id TEXT NOT NULL,
            from_date TEXT,
            thru_date TEXT,
            payload TEXT NOT NULL
        )
        """
    )
    connection.execute("CREATE INDEX carriers_desynpuf_idx ON carriers (desynpuf_id)")
    return connection


def flush_batch(
    connection: sqlite3.Connection,
    sql: str,
    batch: list[tuple[Any, ...]],
) -> None:
    if not batch:
        return
    connection.executemany(sql, batch)
    batch.clear()


def load_beneficiaries_into_database(
    connection: sqlite3.Connection,
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]],
    logger: CliLogger,
    *,
    show_progress: bool,
) -> int:
    inserted = 0
    sql = "INSERT OR REPLACE INTO beneficiaries (desynpuf_id, year, payload) VALUES (?, ?, ?)"

    for spec, path in grouped_paths.get("beneficiary", []):
        logger.info("Loading beneficiary summary", file=path.name, year=spec.year)
        batch: list[tuple[Any, ...]] = []
        with connection:
            for row in dataset.iter_csv_rows_from_csv(
                path,
                description=f"beneficiary {path.name}",
                show_progress=show_progress,
            ):
                desynpuf_id = parsers.clean_text(
                    row.get("DESYNPUF_ID"),
                    uppercase=True,
                    allow_all_zero=False,
                )
                if desynpuf_id is None or spec.year is None:
                    continue
                payload = parsers.simplify_beneficiary_row(row, spec.year)
                batch.append((desynpuf_id, spec.year, json.dumps(payload, sort_keys=True)))
                inserted += 1
                if len(batch) >= constants.DEFAULT_SQLITE_BATCH_SIZE:
                    flush_batch(connection, sql, batch)
            flush_batch(connection, sql, batch)

    return inserted


def load_facility_anchors_into_database(
    connection: sqlite3.Connection,
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]],
    logger: CliLogger,
    *,
    show_progress: bool,
) -> int:
    processed = 0
    insert_sql = (
        "INSERT INTO anchors (cluster_id, desynpuf_id, anchor_year, from_date, thru_date, payload) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    select_sql = "SELECT payload FROM anchors WHERE cluster_id = ?"
    update_sql = (
        "UPDATE anchors SET anchor_year = ?, from_date = ?, thru_date = ?, payload = ? "
        "WHERE cluster_id = ?"
    )

    for claim_type in ("inpatient", "outpatient"):
        for _, path in grouped_paths.get(claim_type, []):
            logger.info("Loading facility claims", claim_type=claim_type, file=path.name)
            cursor = connection.cursor()
            with connection:
                for row in dataset.iter_csv_rows_from_csv(
                    path,
                    description=f"{claim_type} {path.name}",
                    show_progress=show_progress,
                ):
                    anchor = parsers.build_facility_anchor(row, claim_type)
                    if anchor is None:
                        continue
                    payload = parsers.serialize_anchor_for_storage(anchor)
                    serialized_payload = json.dumps(payload, sort_keys=True)
                    try:
                        cursor.execute(
                            insert_sql,
                            (
                                anchor["cluster_id"],
                                anchor["desynpuf_id"],
                                anchor.get("anchor_year"),
                                payload.get("anchor_from_date"),
                                payload.get("anchor_thru_date"),
                                serialized_payload,
                            ),
                        )
                    except sqlite3.IntegrityError:
                        existing_payload_row = cursor.execute(
                            select_sql,
                            (anchor["cluster_id"],),
                        ).fetchone()
                        if existing_payload_row is None:
                            raise
                        existing_anchor = parsers.deserialize_anchor_from_storage(
                            json.loads(existing_payload_row[0])
                        )
                        merged_anchor = matching.merge_facility_anchors(existing_anchor, anchor)
                        merged_payload = parsers.serialize_anchor_for_storage(merged_anchor)
                        cursor.execute(
                            update_sql,
                            (
                                merged_anchor.get("anchor_year"),
                                merged_payload.get("anchor_from_date"),
                                merged_payload.get("anchor_thru_date"),
                                json.dumps(merged_payload, sort_keys=True),
                                merged_anchor["cluster_id"],
                            ),
                        )
                    processed += 1

    return processed


def load_carrier_claims_into_database(
    connection: sqlite3.Connection,
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]],
    logger: CliLogger,
    *,
    show_progress: bool,
) -> int:
    inserted = 0
    sql = "INSERT INTO carriers (desynpuf_id, from_date, thru_date, payload) VALUES (?, ?, ?, ?)"

    for _, path in grouped_paths.get("carrier", []):
        logger.info("Loading carrier claims", file=path.name)
        batch: list[tuple[Any, ...]] = []
        with connection:
            for row in dataset.iter_csv_rows_from_csv(
                path,
                description=f"carrier {path.name}",
                show_progress=show_progress,
            ):
                carrier_claim = parsers.build_carrier_claim(row)
                if carrier_claim is None:
                    continue
                desynpuf_id, claim_from, claim_thru, claim = carrier_claim
                payload = parsers.serialize_carrier_record_for_storage(
                    desynpuf_id,
                    claim_from,
                    claim_thru,
                    claim,
                )
                batch.append(
                    (
                        desynpuf_id,
                        payload.get("claim_from"),
                        payload.get("claim_thru"),
                        json.dumps(payload, sort_keys=True),
                    )
                )
                inserted += 1
                if len(batch) >= constants.DEFAULT_SQLITE_BATCH_SIZE:
                    flush_batch(connection, sql, batch)
            flush_batch(connection, sql, batch)

    return inserted


def beneficiary_context_for_anchor(
    anchor_year: int | None,
    contexts_for_id: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if not contexts_for_id:
        return None
    if anchor_year in contexts_for_id:
        return contexts_for_id[anchor_year]
    fallback_year = sorted(contexts_for_id)[0]
    return contexts_for_id[fallback_year]


def load_beneficiary_contexts_for_id(
    cursor: sqlite3.Cursor,
    desynpuf_id: str,
) -> dict[int, dict[str, Any]]:
    contexts: dict[int, dict[str, Any]] = {}
    for year, payload in cursor.execute(
        "SELECT year, payload FROM beneficiaries WHERE desynpuf_id = ? ORDER BY year",
        (desynpuf_id,),
    ):
        contexts[int(year)] = json.loads(payload)
    return contexts


def load_carrier_records_for_id(
    cursor: sqlite3.Cursor,
    desynpuf_id: str,
) -> list[tuple[date | None, date | None, dict[str, Any]]]:
    records: list[tuple[date | None, date | None, dict[str, Any]]] = []
    for (payload,) in cursor.execute(
        "SELECT payload FROM carriers WHERE desynpuf_id = ? ORDER BY from_date, row_id",
        (desynpuf_id,),
    ):
        records.append(parsers.deserialize_carrier_record_from_storage(json.loads(payload)))
    return records


def materialize_encounter_cache(
    connection: sqlite3.Connection,
    encounter_cache_path: Path,
    sample_number: int,
    carrier_match_window_days: int,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> dict[str, int]:
    anchor_total = int(connection.execute("SELECT COUNT(*) FROM anchors").fetchone()[0])
    temp_path = encounter_cache_path.with_suffix(f"{encounter_cache_path.suffix}.part")
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Materializing encounter cache",
        encounter_cache_path=encounter_cache_path,
        anchors=anchor_total,
    )

    anchor_cursor = connection.cursor()
    beneficiary_cursor = connection.cursor()
    carrier_cursor = connection.cursor()
    progress = datagen_io.create_progress_bar(
        description="materialize encounter cache",
        total=anchor_total,
        unit="row",
        show_progress=show_progress,
    )

    total_records = 0
    clusters_with_carrier = 0
    current_id: str | None = None
    current_beneficiary_contexts: dict[int, dict[str, Any]] = {}
    current_carrier_records: list[tuple[date | None, date | None, dict[str, Any]]] = []

    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            for desynpuf_id, payload_text in anchor_cursor.execute(
                "SELECT desynpuf_id, payload FROM anchors ORDER BY desynpuf_id, from_date, cluster_id"
            ):
                if desynpuf_id != current_id:
                    current_id = desynpuf_id
                    current_beneficiary_contexts = load_beneficiary_contexts_for_id(
                        beneficiary_cursor,
                        desynpuf_id,
                    )
                    current_carrier_records = load_carrier_records_for_id(
                        carrier_cursor,
                        desynpuf_id,
                    )

                anchor = parsers.deserialize_anchor_from_storage(json.loads(payload_text))
                beneficiary = beneficiary_context_for_anchor(
                    anchor.get("anchor_year"),
                    current_beneficiary_contexts,
                )
                if beneficiary is not None:
                    anchor["beneficiary"] = beneficiary

                for claim_from, claim_thru, claim in current_carrier_records:
                    if not matching.carrier_claim_matches_anchor(
                        anchor,
                        claim_from,
                        claim_thru,
                        carrier_match_window_days,
                    ):
                        continue
                    anchor["claims"].append(claim)
                    anchor["diagnoses"].extend(claim["diagnosis_codes"])
                    anchor["procedure_codes"].extend(claim["line_items"])

                record = matching.finalize_cluster_record(
                    anchor,
                    sample_number,
                    carrier_match_window_days,
                )
                json.dump(record, handle, sort_keys=True)
                handle.write("\n")
                total_records += 1
                if bool(record["summary"]["has_carrier_claim"]):
                    clusters_with_carrier += 1
                if progress is not None:
                    progress.update(1)
    finally:
        datagen_io.close_progress_bar(progress)

    temp_path.replace(encounter_cache_path)
    logger.success(
        "Built encounter cache",
        encounter_cache_path=encounter_cache_path,
        total_records=total_records,
        clusters_with_carrier=clusters_with_carrier,
    )
    return {
        "total_records": total_records,
        "clusters_with_carrier": clusters_with_carrier,
    }
