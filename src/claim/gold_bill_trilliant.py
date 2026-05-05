from __future__ import annotations

import json
import logging
import re
import statistics
import threading
import time
from dataclasses import asdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from typing import Sequence
from urllib.parse import urljoin

import requests

from claim.cli_logging import CliLogger
from claim import gold_bill_support
from claim.inpatient_anchor import inpatient_anchor_lookup
from claim.normalization import normalize_code_system

try:
    import duckdb
except ImportError:  # pragma: no cover - exercised only when dependency is missing
    duckdb = None

DEFAULT_REQUEST_TIMEOUT_SECONDS = 120
TRILLIANT_DB_URL = "https://oria-data.trillianthealth.com/trilliant_mrf_data.duckdb"
TRILLIANT_PUBLIC_BASE_URL = "https://oria-data.trillianthealth.com"
TRILLIANT_PUBLIC_SEARCH_INDEX_URL = f"{TRILLIANT_PUBLIC_BASE_URL}/search-index.json"
TRILLIANT_SCHEMA_CACHE_VERSION = 1
RATE_LOOKUP_CACHE_VERSION = 4
TRILLIANT_PRIOR_CACHE_VERSION = 2
HOSPITAL_CANDIDATE_CACHE_VERSION = 2
TRILLIANT_ALIAS = "trilliant"
TRILLIANT_USER_AGENT = "medical-billing-eval/0.1 (gold-bill generator)"
LOGGER = logging.getLogger(__name__)
PUBLIC_DIRECTORY_STATE_PRIOR_HOSPITAL_CAP = 50
PUBLIC_DIRECTORY_NATIONAL_PRIOR_HOSPITAL_CAP = 100
TRILLIANT_STATE_PRIOR_MIN_HOSPITALS = 3
TRILLIANT_NATIONAL_PRIOR_MIN_HOSPITALS = 8
BOUNDED_STATE_PRIOR_HOSPITAL_CAP = 8
TRILLIANT_PRIOR_CACHE_MISS_ENTRY_TYPE = "trilliant_prior_miss"
TRILLIANT_PRIOR_CACHE_MISS_REASON_NO_RESULT = "no_result"
TRILLIANT_PRIOR_CACHE_MISS_REASON_INSUFFICIENT_SUPPORT = "insufficient_support"
TRILLIANT_PRIOR_PROGRESS_LOG_DELAY_SECONDS = 5.0
TRILLIANT_PRIOR_PROGRESS_LOG_INTERVAL_SECONDS = 10.0

DEFAULT_NORMAL_PAYERS = [
    "UnitedHealthcare",
    "Blue Cross Blue Shield",
    "Aetna",
    "Humana",
    "Medicare Advantage",
]
INPATIENT_DRG_FAMILY_CODE_SYSTEMS = frozenset({"MS-DRG", "APR-DRG", "DRG"})
INPATIENT_OTHER_ANCHOR_CODE_SYSTEMS = frozenset({"REVENUE_CODE", "ICD10_PCS"})
INPATIENT_SHORT_STAY_ALLOWED_FLOOR = 2_000.0
INPATIENT_LONG_STAY_ALLOWED_FLOOR = 3_000.0
SPECIALTY_FAMILY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "behavioral": ("behavioral", "psychiatric", "psych"),
    "rehab": ("rehab", "rehabilitation"),
    "ltac": ("ltac", "long term acute"),
    "swing_bed": ("swing bed",),
}

DUCKDB_PROGRESS_DISABLE_STATEMENTS = (
    "SET enable_progress_bar = false",
    "SET enable_print_progress_bar = false",
    "PRAGMA disable_progress_bar",
)


@dataclass(slots=True)
class TrilliantSchema:
    hospital_table: str
    rate_table: str
    hospital_name_column: str | None
    hospital_npi_column: str | None
    hospital_zip_column: str | None
    hospital_state_column: str | None
    hospital_county_name_column: str | None
    hospital_county_fips_column: str | None
    hospital_address_columns: list[str]
    hospital_identity_columns: list[str]
    rate_billing_code_column: str
    rate_payer_column: str | None
    rate_negotiated_rate_column: str | None
    rate_gross_charge_column: str | None
    rate_state_column: str | None
    rate_identity_columns: list[str]
    source_mode: str = "aggregate"
    rate_code_columns: list[str] | None = None
    rate_description_column: str | None = None
    rate_standard_dollar_column: str | None = None
    rate_percentage_column: str | None = None
    rate_estimated_amount_column: str | None = None
    rate_minimum_column: str | None = None
    rate_maximum_column: str | None = None
    rate_median_amount_column: str | None = None


def ensure_duckdb_available() -> None:
    if duckdb is None:
        raise RuntimeError(
            "The 'duckdb' package is not installed. Run `uv sync` to install project dependencies."
        )


def disable_duckdb_progress_bar(connection: Any) -> None:
    for statement in DUCKDB_PROGRESS_DISABLE_STATEMENTS:
        try:
            connection.execute(statement)
        except Exception:
            continue


def connect_trilliant_cache(
    cache_path: Path,
    db_url: str,
) -> Any:
    ensure_duckdb_available()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(cache_path))
    disable_duckdb_progress_bar(connection)
    escaped_url = db_url.replace("'", "''")
    try:
        connection.execute("INSTALL httpfs")
    except Exception as exc:
        LOGGER.warning(
            "DuckDB httpfs install failed; continuing with existing extension if available: %s",
            exc,
        )
    connection.execute("LOAD httpfs")
    ensure_cache_tables(connection)
    try:
        connection.execute(f"ATTACH '{escaped_url}' AS {TRILLIANT_ALIAS} (READ_ONLY)")
    except Exception as exc:
        # The public Trilliant directory no longer exposes a consolidated DuckDB
        # at the historical default URL. Keep the local cache connection alive and
        # let schema discovery fall back to the public directory/search-index flow.
        LOGGER.warning(
            "Failed to attach Trilliant DuckDB at %s; falling back to public-directory schema discovery: %s",
            db_url,
            exc,
        )
    return connection


def ensure_cache_tables(connection: Any) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_cache (
            cache_key TEXT PRIMARY KEY,
            schema_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS hospital_candidate_cache (
            geography_key TEXT PRIMARY KEY,
            candidates_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS rate_quote_cache (
            lookup_key TEXT PRIMARY KEY,
            quote_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS artifact_cache (
            cache_key TEXT PRIMARY KEY,
            payload_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )


def load_cached_artifact(
    connection: Any,
    cache_key: str,
    *,
    lock: threading.Lock | None = None,
) -> Any | None:
    def operation() -> list[dict[str, Any]]:
        return gold_bill_support.fetch_rows_as_dicts(
            connection.execute(
                "SELECT payload_json FROM artifact_cache WHERE cache_key = ?",
                [cache_key],
            )
        )

    rows = gold_bill_support.synchronized_call(lock, operation)
    if not rows:
        return None
    return json.loads(str(rows[0]["payload_json"]))


def save_cached_artifact(
    connection: Any,
    cache_key: str,
    payload: Any,
    *,
    lock: threading.Lock | None = None,
) -> None:
    gold_bill_support.synchronized_call(
        lock,
        lambda: connection.execute(
            """
            INSERT OR REPLACE INTO artifact_cache (cache_key, payload_json, updated_at)
            VALUES (?, ?, ?)
            """,
            [cache_key, json.dumps(payload, sort_keys=True), gold_bill_support.iso_timestamp()],
        ),
    )


def attach_alias_table_description(connection: Any, table_name: str) -> list[str]:
    result = connection.execute(
        f"DESCRIBE SELECT * FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(table_name)}"
    )
    return [row[0] for row in result.fetchall()]


def public_directory_trilliant_schema() -> TrilliantSchema:
    return TrilliantSchema(
        hospital_table="public_search_index",
        rate_table="standard_charge_details",
        hospital_name_column="hospital_name",
        hospital_npi_column=None,
        hospital_zip_column="hospital_zip",
        hospital_state_column="hospital_state",
        hospital_county_name_column="hospital_county_name",
        hospital_county_fips_column="hospital_county_fips",
        hospital_address_columns=["hospital_address"],
        hospital_identity_columns=["hospital_directory_id"],
        rate_billing_code_column="billing_code",
        rate_payer_column="payer_name",
        rate_negotiated_rate_column="estimated_amount",
        rate_gross_charge_column="gross_charge",
        rate_state_column=None,
        rate_identity_columns=["hospital_directory_id"],
        source_mode="public_directory",
        rate_code_columns=["cpt", "hcpcs"],
        rate_description_column="description",
        rate_standard_dollar_column="standard_charge_dollar",
        rate_percentage_column="standard_charge_percentage",
        rate_estimated_amount_column="estimated_amount",
        rate_minimum_column="minimum",
        rate_maximum_column="maximum",
        rate_median_amount_column="median_amount",
    )


def discover_trilliant_schema(
    connection: Any,
    *,
    hospital_table_override: str | None,
    rate_table_override: str | None,
) -> TrilliantSchema:
    try:
        table_rows = connection.execute(f"SHOW TABLES FROM {TRILLIANT_ALIAS}").fetchall()
    except Exception as exc:
        LOGGER.warning(
            "SHOW TABLES failed for attached Trilliant database; falling back to public-directory schema discovery: %s",
            exc,
        )
        return public_directory_trilliant_schema()
    table_names = [str(row[0]) for row in table_rows]
    if not table_names:
        return public_directory_trilliant_schema()

    cache_key = json.dumps(
        {
            "version": TRILLIANT_SCHEMA_CACHE_VERSION,
            "tables": sorted(table_names),
            "hospital_table_override": hospital_table_override,
            "rate_table_override": rate_table_override,
        },
        sort_keys=True,
    )
    cached_rows = gold_bill_support.fetch_rows_as_dicts(
        connection.execute(
            "SELECT schema_json FROM schema_cache WHERE cache_key = ?",
            [cache_key],
        )
    )
    if cached_rows:
        payload = json.loads(cached_rows[0]["schema_json"])
        return TrilliantSchema(**payload)

    described_tables = {
        table_name: attach_alias_table_description(connection, table_name)
        for table_name in table_names
    }

    def choose_table(
        override: str | None,
        words: Sequence[str],
        column_aliases: Sequence[str],
    ) -> str:
        if override:
            if override not in described_tables:
                raise RuntimeError(f"Configured Trilliant table {override!r} was not found.")
            return override
        best_table: str | None = None
        best_score = -1
        for table_name, columns in described_tables.items():
            score = gold_bill_support.score_table_name(table_name, words)
            for alias in column_aliases:
                if gold_bill_support.pick_best_column(columns, [alias]):
                    score += 3
            if score > best_score:
                best_table = table_name
                best_score = score
        if best_table is None:
            raise RuntimeError(f"Could not discover a Trilliant table for {words!r}")
        return best_table

    hospital_table = choose_table(
        hospital_table_override,
        ["hospital", "facility", "provider"],
        ["zip", "npi", "name"],
    )
    rate_table = choose_table(
        rate_table_override,
        ["rate", "negotiated", "charge", "billing"],
        ["billing_code", "payer", "negotiated_rate", "gross_charge"],
    )

    hospital_columns = described_tables[hospital_table]
    rate_columns = described_tables[rate_table]

    schema = TrilliantSchema(
        hospital_table=hospital_table,
        rate_table=rate_table,
        hospital_name_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["hospital_name", "facility_name", "provider_name", "name"],
        ),
        hospital_npi_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["hospital_npi", "facility_npi", "provider_npi", "npi"],
        ),
        hospital_zip_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["zip", "zip_code", "zipcode", "postal_code"],
        ),
        hospital_state_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["state", "state_code", "state_abbrev", "state_abrv"],
        ),
        hospital_county_name_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["county_name", "county", "county_title"],
        ),
        hospital_county_fips_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["county_fips", "county_geoid", "countyfp", "fips_county"],
        ),
        hospital_address_columns=[
            column
            for column in (
                gold_bill_support.pick_best_column(hospital_columns, ["address", "street_address", "full_address"]),
                gold_bill_support.pick_best_column(hospital_columns, ["address_1", "address1", "addr1"]),
                gold_bill_support.pick_best_column(hospital_columns, ["city"]),
            )
            if column is not None
        ],
        hospital_identity_columns=gold_bill_support.unique_preserving_order(
            [
                column
                for column in (
                    gold_bill_support.pick_best_column(hospital_columns, ["npi"]),
                    gold_bill_support.pick_best_column(hospital_columns, ["provider_num", "provider_number", "ccn"]),
                    gold_bill_support.pick_best_column(hospital_columns, ["facility_id", "hospital_id", "provider_id"]),
                    gold_bill_support.pick_best_column(hospital_columns, ["name", "facility_name", "hospital_name"]),
                )
                if column is not None
            ]
        ),
        rate_billing_code_column=detect_csv_column(
            rate_columns,
            ["billing_code", "code", "hcpcs_code", "cpt_code", "procedure_code"],
        ),
        rate_payer_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["payer_name", "payer", "plan_name", "insurance_name"],
        ),
        rate_negotiated_rate_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["negotiated_rate", "allowed_amount", "rate", "negotiated_amount"],
        ),
        rate_gross_charge_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["gross_charge", "standard_charge_gross", "chargemaster_rate", "billed_charge"],
        ),
        rate_state_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["state", "state_code", "state_abbrev"],
        ),
        rate_identity_columns=gold_bill_support.unique_preserving_order(
            [
                column
                for column in (
                    gold_bill_support.pick_best_column(rate_columns, ["npi"]),
                    gold_bill_support.pick_best_column(rate_columns, ["provider_num", "provider_number", "ccn"]),
                    gold_bill_support.pick_best_column(rate_columns, ["facility_id", "hospital_id", "provider_id"]),
                    gold_bill_support.pick_best_column(rate_columns, ["hospital_name", "facility_name", "provider_name"]),
                    gold_bill_support.pick_best_column(rate_columns, ["zip", "zip_code", "zipcode"]),
                )
                if column is not None
            ]
        ),
    )

    connection.execute(
        """
        INSERT OR REPLACE INTO schema_cache (cache_key, schema_json, updated_at)
        VALUES (?, ?, ?)
        """,
        [cache_key, json.dumps(asdict(schema), sort_keys=True), gold_bill_support.iso_timestamp()],
    )
    return schema


def detect_csv_column(headers: Sequence[str], aliases: Sequence[str]) -> str:
    column = gold_bill_support.pick_best_column(headers, aliases)
    if column is None:
        raise ValueError(
            f"Could not locate a matching CSV column for aliases {aliases!r} in {list(headers)!r}"
        )
    return column


def extract_last_zip_code(value: str | None) -> str | None:
    text = gold_bill_support.normalize_text(value)
    if text is None:
        return None
    matches = gold_bill_support.ZIP_RE.findall(text)
    if not matches:
        return None
    return matches[-1]


def fetch_public_trilliant_search_index(
    connection: Any,
    session: requests.Session,
    *,
    cache_lock: threading.Lock | None = None,
) -> list[dict[str, Any]]:
    cache_key = "trilliant_public_search_index_v1"
    cached = load_cached_artifact(connection, cache_key, lock=cache_lock)
    if isinstance(cached, list):
        return cached

    response = session.get(
        TRILLIANT_PUBLIC_SEARCH_INDEX_URL,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": TRILLIANT_USER_AGENT},
    )
    try:
        response.raise_for_status()
        payload = response.json()
    finally:
        response.close()
    if not isinstance(payload, list):
        raise RuntimeError(
            f"Expected a JSON array from {TRILLIANT_PUBLIC_SEARCH_INDEX_URL}, got {type(payload)!r}"
        )
    save_cached_artifact(connection, cache_key, payload, lock=cache_lock)
    return payload


def normalize_public_directory_hospital_candidate(item: dict[str, Any]) -> dict[str, Any]:
    hospital_id = gold_bill_support.normalize_text(item.get("id"))
    hospital_name = gold_bill_support.normalize_text(item.get("hospitalName")) or gold_bill_support.normalize_text(
        item.get("locationName")
    )
    address = gold_bill_support.normalize_text(item.get("address"))
    return {
        "hospital_name": hospital_name,
        "hospital_npi": None,
        "hospital_zip": extract_last_zip_code(address),
        "hospital_state": gold_bill_support.normalize_state_code(item.get("state")),
        "hospital_county_name": None,
        "hospital_county_fips": None,
        "hospital_address": address,
        "hospital_city": gold_bill_support.normalize_text(item.get("city")),
        "hospital_directory_id": hospital_id,
        "hospital_page_url": (
            urljoin(TRILLIANT_PUBLIC_BASE_URL, f"/hospital/{hospital_id}")
            if hospital_id
            else None
        ),
        "identity_fields": {"hospital_directory_id": hospital_id} if hospital_id else {},
    }


def reverse_zip_to_county_mapping(
    local_hud_mapping: dict[str, list[str]] | None,
) -> dict[str, list[str]]:
    reverse: dict[str, list[str]] = {}
    if not local_hud_mapping:
        return reverse
    for county_fips, zip_codes in local_hud_mapping.items():
        normalized_county = gold_bill_support.normalize_text(county_fips)
        if normalized_county is None:
            continue
        for zip_code in zip_codes:
            normalized_zip = gold_bill_support.normalize_zip(zip_code)
            if normalized_zip is None:
                continue
            reverse.setdefault(normalized_zip, []).append(normalized_county)
    return {
        zip_code: gold_bill_support.unique_preserving_order(counties)
        for zip_code, counties in reverse.items()
    }


def public_directory_specialty_families(text: str | None) -> set[str]:
    normalized = gold_bill_support.normalize_text(text)
    if normalized is None:
        return set()
    lowered = normalized.lower()
    return {
        family
        for family, keywords in SPECIALTY_FAMILY_KEYWORDS.items()
        if any(keyword in lowered for keyword in keywords)
    }


def specialty_filter_matches_direction(
    candidate: dict[str, Any],
    *,
    primary_clinical_direction: str | None,
) -> bool:
    candidate_text = " ".join(
        value
        for value in (
            gold_bill_support.normalize_text(candidate.get("hospital_name")),
            gold_bill_support.normalize_text(candidate.get("hospital_address")),
        )
        if value
    )
    candidate_families = public_directory_specialty_families(candidate_text)
    if not candidate_families:
        return True
    direction_families = public_directory_specialty_families(primary_clinical_direction)
    return bool(candidate_families & direction_families)


def filter_public_directory_candidates(
    directory_rows: list[dict[str, Any]],
    *,
    zip_codes: Sequence[str] | None = None,
    state_value: str | None,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], str]:
    normalized_candidates = [
        normalize_public_directory_hospital_candidate(row)
        for row in directory_rows
    ]
    normalized_candidates = [
        candidate
        for candidate in normalized_candidates
        if candidate.get("hospital_name") and candidate.get("hospital_directory_id")
    ]
    state_abbrev = (
        gold_bill_support.normalize_state_code(state_value)
        if state_value and len(str(state_value).strip()) == 2
        else None
    )
    zip_set = {
        gold_bill_support.normalize_zip(code)
        for code in (zip_codes or [])
        if gold_bill_support.normalize_zip(code)
    }

    if zip_set:
        zip_matched = [
            candidate
            for candidate in normalized_candidates
            if candidate.get("hospital_zip") in zip_set
        ]
        if zip_matched:
            zip_matched.sort(
                key=lambda candidate: (
                    str(candidate.get("hospital_zip") or ""),
                    str(candidate.get("hospital_name") or ""),
                    str(candidate.get("hospital_directory_id") or ""),
                )
            )
            if state_abbrev:
                in_state_non_zip = [
                    candidate
                    for candidate in normalized_candidates
                    if (
                        candidate.get("hospital_zip") not in zip_set
                        and gold_bill_support.normalize_state_code(candidate.get("hospital_state"))
                        == state_abbrev
                    )
                ]
                in_state_non_zip.sort(
                    key=lambda candidate: (
                        str(candidate.get("hospital_city") or ""),
                        str(candidate.get("hospital_name") or ""),
                        str(candidate.get("hospital_directory_id") or ""),
                    )
                )
                return (zip_matched + in_state_non_zip)[:max_candidates], "directory_zip"
            return zip_matched[:max_candidates], "directory_zip"

    if state_abbrev:
        matched = [
            candidate
            for candidate in normalized_candidates
            if gold_bill_support.normalize_state_code(candidate.get("hospital_state")) == state_abbrev
        ]
        if matched:
            matched.sort(
                key=lambda candidate: (
                    str(candidate.get("hospital_city") or ""),
                    str(candidate.get("hospital_name") or ""),
                    str(candidate.get("hospital_directory_id") or ""),
                )
            )
            return matched[:max_candidates], "directory_state"

    normalized_candidates.sort(
        key=lambda candidate: (
            str(candidate.get("hospital_state") or ""),
            str(candidate.get("hospital_city") or ""),
            str(candidate.get("hospital_name") or ""),
            str(candidate.get("hospital_directory_id") or ""),
        )
    )
    return normalized_candidates[:max_candidates], "directory_all"


def public_directory_candidate_pool(
    candidates: Sequence[dict[str, Any]],
    *,
    stage: str,
    zip_codes: Sequence[str],
    county_fips: str | None,
    state_value: str | None,
    reverse_zip_to_counties: dict[str, list[str]] | None,
    primary_clinical_direction: str | None,
    specialty_filter_active: bool,
) -> list[dict[str, Any]]:
    zip_set = {
        gold_bill_support.normalize_zip(code)
        for code in zip_codes
        if gold_bill_support.normalize_zip(code)
    }
    state_abbrev = (
        gold_bill_support.normalize_state_code(state_value)
        if state_value and len(str(state_value).strip()) == 2
        else None
    )

    def matches_stage(candidate: dict[str, Any]) -> bool:
        candidate_zip = gold_bill_support.normalize_zip(candidate.get("hospital_zip"))
        candidate_state = gold_bill_support.normalize_state_code(candidate.get("hospital_state"))
        county_match = (
            county_fips is not None
            and candidate_zip is not None
            and county_fips in (reverse_zip_to_counties or {}).get(candidate_zip, [])
        )
        if stage == "zip":
            return candidate_zip in zip_set
        if stage == "county":
            return county_match and candidate_zip not in zip_set
        if stage == "same_state":
            return (
                (state_abbrev is None or candidate_state is None or candidate_state == state_abbrev)
                and candidate_zip not in zip_set
                and not county_match
            )
        raise ValueError(f"Unsupported public-directory stage: {stage}")

    filtered = [candidate for candidate in candidates if matches_stage(candidate)]
    if specialty_filter_active:
        filtered = [
            candidate
            for candidate in filtered
            if specialty_filter_matches_direction(
                candidate,
                primary_clinical_direction=primary_clinical_direction,
            )
        ]
    return filtered


def fetch_public_hospital_duckdb_url(
    connection: Any,
    session: requests.Session,
    hospital: dict[str, Any],
    *,
    cache_lock: threading.Lock | None = None,
    force_refresh: bool = False,
) -> str:
    hospital_id = gold_bill_support.normalize_text(hospital.get("hospital_directory_id"))
    page_url = gold_bill_support.normalize_text(hospital.get("hospital_page_url"))
    if hospital_id is None or page_url is None:
        raise RuntimeError("Missing Trilliant public-directory identifiers for hospital selection.")

    cache_key = f"trilliant_public_hospital_duckdb_url_v2:{hospital_id}"
    if not force_refresh:
        cached = load_cached_artifact(connection, cache_key, lock=cache_lock)
        if isinstance(cached, str) and cached:
            return cached

    response = session.get(
        page_url,
        timeout=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": TRILLIANT_USER_AGENT},
    )
    try:
        response.raise_for_status()
        matches = re.findall(r"/data/[^\"']+?_parsed\.duckdb", response.text)
    finally:
        response.close()
    if not matches:
        raise RuntimeError(f"Could not locate a parsed DuckDB link on {page_url}")
    duckdb_url = urljoin(TRILLIANT_PUBLIC_BASE_URL, matches[0])
    save_cached_artifact(connection, cache_key, duckdb_url, lock=cache_lock)
    return duckdb_url


def open_remote_trilliant_database(db_url: str) -> Any:
    ensure_duckdb_available()
    connection = duckdb.connect(":memory:")
    disable_duckdb_progress_bar(connection)
    escaped_url = db_url.replace("'", "''")
    try:
        connection.execute("INSTALL httpfs")
    except Exception as exc:
        LOGGER.warning(
            "DuckDB httpfs install failed for remote hospital database; continuing with existing extension if available: %s",
            exc,
        )
    connection.execute("LOAD httpfs")
    try:
        connection.execute(f"ATTACH '{escaped_url}' AS {TRILLIANT_ALIAS} (READ_ONLY)")
    except Exception:
        connection.close()
        raise
    return connection


def open_public_directory_hospital_database(
    connection: Any,
    session: requests.Session,
    hospital: dict[str, Any],
    *,
    cache_lock: threading.Lock | None = None,
) -> tuple[Any, TrilliantSchema, str]:
    attempted_urls: set[str] = set()
    last_error: Exception | None = None

    def try_open(db_url: str | None) -> tuple[Any, TrilliantSchema, str] | None:
        nonlocal last_error
        normalized_url = gold_bill_support.normalize_text(db_url)
        if normalized_url is None or normalized_url in attempted_urls:
            return None
        attempted_urls.add(normalized_url)
        remote_connection: Any | None = None
        try:
            remote_connection = open_remote_trilliant_database(normalized_url)
            remote_schema = discover_public_hospital_rate_schema(remote_connection)
            hospital["duckdb_url"] = normalized_url
            return remote_connection, remote_schema, normalized_url
        except Exception as exc:
            last_error = exc
            if remote_connection is not None:
                remote_connection.close()
            return None

    opened = try_open(hospital.get("duckdb_url"))
    if opened is not None:
        return opened

    opened = try_open(
        fetch_public_hospital_duckdb_url(
            connection,
            session,
            hospital,
            cache_lock=cache_lock,
            force_refresh=False,
        )
    )
    if opened is not None:
        return opened

    opened = try_open(
        fetch_public_hospital_duckdb_url(
            connection,
            session,
            hospital,
            cache_lock=cache_lock,
            force_refresh=True,
        )
    )
    if opened is not None:
        return opened

    if last_error is not None:
        raise last_error
    raise RuntimeError("Could not open a public-directory hospital DuckDB.")


def discover_public_hospital_rate_schema(connection: Any) -> TrilliantSchema:
    table_rows = connection.execute(f"SHOW TABLES FROM {TRILLIANT_ALIAS}").fetchall()
    table_names = {str(row[0]) for row in table_rows}
    if "standard_charge_details" in table_names:
        rate_table = "standard_charge_details"
    elif "standard_charges" in table_names:
        rate_table = "standard_charges"
    else:
        raise RuntimeError(
            "Could not locate a standard charge table in the Trilliant hospital DuckDB."
        )

    hospital_table = "hospitals" if "hospitals" in table_names else "cms_hpt_metadata"
    hospital_columns = attach_alias_table_description(connection, hospital_table)
    rate_columns = attach_alias_table_description(connection, rate_table)
    rate_code_columns = [column for column in ("cpt", "hcpcs") if column in rate_columns]
    if not rate_code_columns and "code" in rate_columns:
        rate_code_columns = ["code"]
    if not rate_code_columns:
        raise RuntimeError(
            f"Could not discover CPT/HCPCS columns in Trilliant table {rate_table!r}."
        )

    return TrilliantSchema(
        hospital_table=hospital_table,
        rate_table=rate_table,
        hospital_name_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["hospital_name", "hpt_hospital_name", "location_name", "mrf_hospital_name"],
        ),
        hospital_npi_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["npi", "type_2_npi"],
        ),
        hospital_zip_column=None,
        hospital_state_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["hospital_state", "state"],
        ),
        hospital_county_name_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["county_name", "county"],
        ),
        hospital_county_fips_column=gold_bill_support.pick_best_column(
            hospital_columns,
            ["county_fips", "countyfp", "fips_county"],
        ),
        hospital_address_columns=[
            column
            for column in (
                gold_bill_support.pick_best_column(hospital_columns, ["hospital_address", "address"]),
            )
            if column is not None
        ],
        hospital_identity_columns=["hospital_directory_id"],
        rate_billing_code_column=rate_code_columns[0],
        rate_payer_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["payer_name", "payer", "plan_name"],
        ),
        rate_negotiated_rate_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["estimated_amount", "median_amount"],
        ),
        rate_gross_charge_column=gold_bill_support.pick_best_column(
            rate_columns,
            ["gross_charge"],
        ),
        rate_state_column=None,
        rate_identity_columns=["hospital_directory_id"],
        source_mode="public_directory",
        rate_code_columns=rate_code_columns,
        rate_description_column=gold_bill_support.pick_best_column(rate_columns, ["description"]),
        rate_standard_dollar_column=gold_bill_support.pick_best_column(rate_columns, ["standard_charge_dollar"]),
        rate_percentage_column=gold_bill_support.pick_best_column(rate_columns, ["standard_charge_percentage"]),
        rate_estimated_amount_column=gold_bill_support.pick_best_column(rate_columns, ["estimated_amount"]),
        rate_minimum_column=gold_bill_support.pick_best_column(rate_columns, ["minimum"]),
        rate_maximum_column=gold_bill_support.pick_best_column(rate_columns, ["maximum"]),
        rate_median_amount_column=gold_bill_support.pick_best_column(rate_columns, ["median_amount"]),
    )


def fetch_hospital_candidates_from_remote(
    connection: Any,
    schema: TrilliantSchema,
    *,
    zip_codes: list[str],
    county_fips: str | None,
    county_name: str | None,
    state_value: str | None,
    max_candidates: int,
) -> tuple[list[dict[str, Any]], str]:
    selected_columns = gold_bill_support.unique_preserving_order(
        [
            column
            for column in (
                schema.hospital_name_column,
                schema.hospital_npi_column,
                schema.hospital_zip_column,
                schema.hospital_state_column,
                schema.hospital_county_name_column,
                schema.hospital_county_fips_column,
                *schema.hospital_address_columns,
                *schema.hospital_identity_columns,
            )
            if column is not None
        ]
    )
    select_clause = ", ".join(
        gold_bill_support.quote_ident(column) for column in selected_columns
    ) or "*"
    params: list[Any] = []
    mode = "state"
    if zip_codes and schema.hospital_zip_column:
        placeholders = ", ".join("?" for _ in zip_codes)
        sql = (
            f"SELECT {select_clause} FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.hospital_table)} "
            f"WHERE regexp_replace(CAST({gold_bill_support.quote_ident(schema.hospital_zip_column)} AS VARCHAR), '[^0-9]', '', 'g') "
            f"IN ({placeholders}) LIMIT ?"
        )
        params.extend(zip_codes)
        params.append(max_candidates)
        mode = "zip"
    elif county_fips and schema.hospital_county_fips_column:
        sql = (
            f"SELECT {select_clause} FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.hospital_table)} "
            f"WHERE regexp_replace(CAST({gold_bill_support.quote_ident(schema.hospital_county_fips_column)} AS VARCHAR), '[^0-9]', '', 'g') = ? "
            f"LIMIT ?"
        )
        params.extend([county_fips, max_candidates])
        mode = "county_fips"
    elif county_name and state_value and schema.hospital_county_name_column and schema.hospital_state_column:
        sql = (
            f"SELECT {select_clause} FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.hospital_table)} "
            f"WHERE lower(CAST({gold_bill_support.quote_ident(schema.hospital_county_name_column)} AS VARCHAR)) = lower(?) "
            f"AND upper(CAST({gold_bill_support.quote_ident(schema.hospital_state_column)} AS VARCHAR)) = upper(?) "
            f"LIMIT ?"
        )
        params.extend([county_name, state_value, max_candidates])
        mode = "county_name"
    elif state_value and schema.hospital_state_column:
        sql = (
            f"SELECT {select_clause} FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.hospital_table)} "
            f"WHERE upper(CAST({gold_bill_support.quote_ident(schema.hospital_state_column)} AS VARCHAR)) = upper(?) "
            f"LIMIT ?"
        )
        params.extend([state_value, max_candidates])
        mode = "state"
    else:
        raise RuntimeError("Could not construct a hospital lookup query for the discovered Trilliant schema.")

    rows = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
    return rows, mode


def build_hospital_geography_key(
    zip_codes: list[str],
    county_fips: str | None,
    county_name: str | None,
    state_value: str | None,
    *,
    source_mode: str,
) -> str:
    return json.dumps(
        {
            "cache_version": HOSPITAL_CANDIDATE_CACHE_VERSION,
            "source_mode": source_mode,
            "zip_codes": sorted(zip_codes),
            "county_fips": county_fips,
            "county_name": county_name,
            "state": state_value,
        },
        sort_keys=True,
    )


def normalize_hospital_candidate(
    row: dict[str, Any],
    schema: TrilliantSchema,
) -> dict[str, Any]:
    address_parts: list[str] = []
    for column in schema.hospital_address_columns:
        value = gold_bill_support.normalize_text(
            str(row.get(column)) if row.get(column) is not None else None
        )
        if value:
            address_parts.append(value)
    identity_fields = {
        column: gold_bill_support.normalize_text(
            str(row.get(column)) if row.get(column) is not None else None
        )
        for column in schema.hospital_identity_columns
        if row.get(column) is not None
    }
    return {
        "hospital_name": gold_bill_support.normalize_text(
            str(row.get(schema.hospital_name_column))
            if schema.hospital_name_column and row.get(schema.hospital_name_column) is not None
            else None
        ),
        "hospital_npi": gold_bill_support.normalize_text(
            str(row.get(schema.hospital_npi_column))
            if schema.hospital_npi_column and row.get(schema.hospital_npi_column) is not None
            else None
        ),
        "hospital_zip": gold_bill_support.normalize_zip(
            str(row.get(schema.hospital_zip_column))
            if schema.hospital_zip_column and row.get(schema.hospital_zip_column) is not None
            else None
        ),
        "hospital_state": gold_bill_support.normalize_text(
            str(row.get(schema.hospital_state_column))
            if schema.hospital_state_column and row.get(schema.hospital_state_column) is not None
            else None
        ),
        "hospital_county_name": gold_bill_support.normalize_text(
            str(row.get(schema.hospital_county_name_column))
            if schema.hospital_county_name_column and row.get(schema.hospital_county_name_column) is not None
            else None
        ),
        "hospital_county_fips": gold_bill_support.normalize_text(
            str(row.get(schema.hospital_county_fips_column))
            if schema.hospital_county_fips_column and row.get(schema.hospital_county_fips_column) is not None
            else None
        ),
        "hospital_address": ", ".join(address_parts) if address_parts else None,
        "identity_fields": {
            key: value
            for key, value in identity_fields.items()
            if value is not None
        },
    }


def get_hospital_candidates(
    connection: Any,
    schema: TrilliantSchema,
    *,
    session: requests.Session,
    zip_codes: list[str],
    county_fips: str | None,
    county_name: str | None,
    state_value: str | None,
    max_candidates: int,
    cache_lock: threading.Lock | None = None,
) -> tuple[list[dict[str, Any]], str, bool]:
    geography_key = build_hospital_geography_key(
        zip_codes,
        county_fips,
        county_name,
        state_value,
        source_mode=schema.source_mode,
    )
    cached = gold_bill_support.synchronized_call(
        cache_lock,
        lambda: gold_bill_support.fetch_rows_as_dicts(
            connection.execute(
                "SELECT candidates_json FROM hospital_candidate_cache WHERE geography_key = ?",
                [geography_key],
            )
        ),
    )
    if cached:
        return json.loads(cached[0]["candidates_json"]), "cached", True

    if schema.source_mode == "public_directory":
        rows = fetch_public_trilliant_search_index(
            connection,
            session,
            cache_lock=cache_lock,
        )
        candidates, query_mode = filter_public_directory_candidates(
            rows,
            zip_codes=zip_codes,
            state_value=state_value,
            max_candidates=max_candidates,
        )
    else:
        rows, query_mode = fetch_hospital_candidates_from_remote(
            connection,
            schema,
            zip_codes=zip_codes,
            county_fips=county_fips,
            county_name=county_name,
            state_value=state_value,
            max_candidates=max_candidates,
        )
        candidates = [normalize_hospital_candidate(row, schema) for row in rows]
    gold_bill_support.synchronized_call(
        cache_lock,
        lambda: connection.execute(
            """
            INSERT OR REPLACE INTO hospital_candidate_cache (geography_key, candidates_json, updated_at)
            VALUES (?, ?, ?)
            """,
            [
                geography_key,
                json.dumps(candidates, sort_keys=True),
                gold_bill_support.iso_timestamp(),
            ],
        ),
    )
    return candidates, query_mode, False


def select_hospital_candidate(
    case_id: str,
    seed: int,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not candidates:
        raise RuntimeError("No hospital candidates were available for this encounter.")
    return gold_bill_support.stable_rng(seed, case_id, "hospital").choice(candidates)


def ordered_hospital_candidates_for_case(
    case_id: str,
    seed: int,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: gold_bill_support.stable_seed(
            seed,
            case_id,
            candidate.get("hospital_directory_id") or candidate.get("hospital_name") or "",
        ),
    )


def select_normal_payer(case_id: str, seed: int, payers: list[str]) -> str:
    if not payers:
        payers = list(DEFAULT_NORMAL_PAYERS)
    return gold_bill_support.stable_rng(seed, case_id, "payer").choice(payers)


def age_adjusted_normal_payers(
    payers: list[str],
    *,
    age_at_encounter: int | None,
) -> list[str]:
    if age_at_encounter is None or age_at_encounter >= 65:
        return list(payers)
    filtered = [
        payer
        for payer in payers
        if "medicare" not in gold_bill_support.normalize_text(payer or "").lower()
    ]
    fallback = [
        payer
        for payer in DEFAULT_NORMAL_PAYERS
        if "medicare" not in gold_bill_support.normalize_text(payer or "").lower()
    ]
    return filtered or fallback or list(payers)


def select_financial_scenario(
    case_id: str,
    seed: int,
    extreme_case_probability: float,
    payers: list[str],
    *,
    age_at_encounter: int | None = None,
) -> dict[str, Any]:
    rng = gold_bill_support.stable_rng(seed, case_id, "scenario")
    is_extreme = rng.random() < max(0.0, min(1.0, extreme_case_probability))
    if is_extreme:
        return {
            "scenario": rng.choice(["out_of_network", "self_pay"]),
            "payer_name": None,
        }
    candidate_payers = age_adjusted_normal_payers(
        payers or list(DEFAULT_NORMAL_PAYERS),
        age_at_encounter=age_at_encounter,
    )
    return {
        "scenario": "in_network",
        "payer_name": select_normal_payer(case_id, seed, candidate_payers),
    }


def choose_rate_identity_filter(
    hospital: dict[str, Any],
    schema: TrilliantSchema,
) -> tuple[str, str] | None:
    if schema.source_mode == "public_directory":
        hospital_id = gold_bill_support.normalize_text(
            hospital.get("hospital_directory_id")
        )
        if hospital_id:
            return ("hospital_directory_id", hospital_id)
        return None
    identity_fields = hospital.get("identity_fields") or {}
    for column in schema.rate_identity_columns:
        value = identity_fields.get(column)
        if value:
            return column, value
    if schema.hospital_zip_column and schema.hospital_zip_column in schema.rate_identity_columns:
        zip_value = hospital.get("hospital_zip")
        if zip_value:
            return schema.hospital_zip_column, zip_value
    return None


def build_rate_lookup_key(
    *,
    billing_code: str,
    lookup_level: str,
    payer_name: str | None,
    identity_filter: tuple[str, str] | None,
    state_value: str | None,
    max_quote_amount: float,
) -> str:
    return json.dumps(
        {
            "cache_version": RATE_LOOKUP_CACHE_VERSION,
            "billing_code": billing_code,
            "lookup_level": lookup_level,
            "payer_name": payer_name,
            "identity_filter": identity_filter,
            "state_value": state_value,
            "max_quote_amount": max_quote_amount,
        },
        sort_keys=True,
    )


def sanitized_quote_amount(
    value: Any,
    *,
    max_quote_amount: float,
) -> float | None:
    amount = gold_bill_support.parse_float(value)
    if amount is None or amount <= 0 or amount > max_quote_amount:
        return None
    return gold_bill_support.round_currency(amount)


def sanitize_rate_quote(
    quote: dict[str, Any],
    *,
    max_quote_amount: float,
) -> dict[str, Any]:
    sanitized = dict(quote)
    sanitized["negotiated_rate"] = sanitized_quote_amount(
        quote.get("negotiated_rate"),
        max_quote_amount=max_quote_amount,
    )
    sanitized["gross_charge"] = sanitized_quote_amount(
        quote.get("gross_charge"),
        max_quote_amount=max_quote_amount,
    )
    sanitized["standard_charge_dollar"] = sanitized_quote_amount(
        quote.get("standard_charge_dollar"),
        max_quote_amount=max_quote_amount,
    )
    sanitized["percentage_charge_amount"] = sanitized_quote_amount(
        quote.get("percentage_charge_amount"),
        max_quote_amount=max_quote_amount,
    )
    sanitized["minimum"] = sanitized_quote_amount(
        quote.get("minimum"),
        max_quote_amount=max_quote_amount,
    )
    sanitized["maximum"] = sanitized_quote_amount(
        quote.get("maximum"),
        max_quote_amount=max_quote_amount,
    )
    return sanitized


def quote_has_any_amount(quote: dict[str, Any]) -> bool:
    return gold_bill_support.parse_float(quote.get("negotiated_rate")) is not None or gold_bill_support.parse_float(
        quote.get("gross_charge")
    ) is not None


def quote_charge_anchor(
    quote: dict[str, Any],
) -> tuple[float | None, str | None]:
    for field in (
        "standard_charge_dollar",
        "percentage_charge_amount",
        "minimum",
        "maximum",
        "gross_charge",
    ):
        amount = gold_bill_support.parse_float(quote.get(field))
        if amount is not None and amount > 0:
            return amount, field
    return None, None


def quote_has_charge_anchor(quote: dict[str, Any]) -> bool:
    amount, _field = quote_charge_anchor(quote)
    return amount is not None


def quote_supports_scenario(
    quote: dict[str, Any],
    *,
    scenario_name: str,
) -> bool:
    if scenario_name == "in_network":
        return gold_bill_support.parse_float(quote.get("negotiated_rate")) is not None
    return quote_has_any_amount(quote)


def bounded_positive_amount_expression(
    expression: str,
    *,
    max_quote_amount: float,
) -> str:
    threshold_literal = format(float(max_quote_amount), ".15g")
    return (
        "CASE "
        f"WHEN try_cast(({expression}) AS DOUBLE) > 0 "
        f"AND try_cast(({expression}) AS DOUBLE) <= {threshold_literal} "
        f"THEN try_cast(({expression}) AS DOUBLE) "
        "ELSE NULL END"
    )


def directory_rate_code_expression(schema: TrilliantSchema) -> str:
    code_columns = schema.rate_code_columns or [schema.rate_billing_code_column]
    coalesce_arguments = [
        f"nullif(trim(CAST({gold_bill_support.quote_ident(column)} AS VARCHAR)), '')"
        for column in code_columns
    ]
    return "upper(trim(coalesce(" + ", ".join(coalesce_arguments) + ")))"


def directory_allowed_amount_expression(schema: TrilliantSchema) -> str:
    candidates: list[str] = []
    if schema.rate_estimated_amount_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_estimated_amount_column)} AS DOUBLE)"
        )
    if schema.rate_median_amount_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_median_amount_column)} AS DOUBLE)"
        )
    if not candidates:
        return "NULL"
    return "coalesce(" + ", ".join(candidates) + ")"


def directory_charge_fallback_expression(schema: TrilliantSchema) -> str:
    candidates: list[str] = []
    if schema.rate_standard_dollar_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_standard_dollar_column)} AS DOUBLE)"
        )
    if schema.rate_percentage_column and schema.rate_gross_charge_column:
        candidates.append(
            "("
            f"try_cast({gold_bill_support.quote_ident(schema.rate_gross_charge_column)} AS DOUBLE) * "
            f"try_cast({gold_bill_support.quote_ident(schema.rate_percentage_column)} AS DOUBLE) / 100.0"
            ")"
        )
    if schema.rate_minimum_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_minimum_column)} AS DOUBLE)"
        )
    if schema.rate_maximum_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_maximum_column)} AS DOUBLE)"
        )
    if schema.rate_gross_charge_column:
        candidates.append(
            f"try_cast({gold_bill_support.quote_ident(schema.rate_gross_charge_column)} AS DOUBLE)"
        )
    if not candidates:
        return "NULL"
    return "coalesce(" + ", ".join(candidates) + ")"


def directory_percentage_charge_expression(schema: TrilliantSchema) -> str:
    if schema.rate_percentage_column and schema.rate_gross_charge_column:
        return (
            "("
            f"try_cast({gold_bill_support.quote_ident(schema.rate_gross_charge_column)} AS DOUBLE) * "
            f"try_cast({gold_bill_support.quote_ident(schema.rate_percentage_column)} AS DOUBLE) / 100.0"
            ")"
        )
    return "NULL"


def rate_percentage_charge_expression(schema: TrilliantSchema) -> str:
    if schema.rate_percentage_column and schema.rate_gross_charge_column:
        return (
            "("
            f"try_cast({gold_bill_support.quote_ident(schema.rate_gross_charge_column)} AS DOUBLE) * "
            f"try_cast({gold_bill_support.quote_ident(schema.rate_percentage_column)} AS DOUBLE) / 100.0"
            ")"
        )
    return "NULL"


def _median_bounded_amount_expression(
    expression: str,
    *,
    max_quote_amount: float,
) -> str:
    return (
        "median("
        + bounded_positive_amount_expression(
            expression,
            max_quote_amount=max_quote_amount,
        )
        + ")"
    )


def directory_gross_charge_expression(schema: TrilliantSchema) -> str:
    if schema.rate_gross_charge_column:
        return f"try_cast({gold_bill_support.quote_ident(schema.rate_gross_charge_column)} AS DOUBLE)"
    return "NULL"


def query_rate_level_from_public_directory_hospital(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: list[str],
    lookup_level: str,
    payer_name: str | None,
    encounter_regime: str,
    scenario_name: str = "in_network",
    max_quote_amount: float,
) -> dict[str, dict[str, Any]]:
    if not billing_codes:
        return {}

    supported_columns = attach_alias_table_description(connection, schema.rate_table)
    code_expression = directory_rate_code_expression(schema)
    raw_allowed_expression = directory_allowed_amount_expression(schema)
    raw_charge_fallback_expression = directory_charge_fallback_expression(schema)
    allowed_expression = bounded_positive_amount_expression(
        raw_allowed_expression,
        max_quote_amount=max_quote_amount,
    )
    fallback_charge_expression = bounded_positive_amount_expression(
        raw_charge_fallback_expression,
        max_quote_amount=max_quote_amount,
    )
    if scenario_name == "in_network" and raw_allowed_expression == "NULL":
        return {}
    if (
        scenario_name != "in_network"
        and raw_allowed_expression == "NULL"
        and raw_charge_fallback_expression == "NULL"
    ):
        return {}
    where_clauses = [
        f"{code_expression} IN ({', '.join('?' for _ in billing_codes)})",
    ]
    if scenario_name == "in_network":
        where_clauses.append(f"{allowed_expression} IS NOT NULL")
    else:
        where_clauses.append(
            f"coalesce({allowed_expression}, {fallback_charge_expression}) IS NOT NULL"
        )
    params: list[Any] = [code.upper() for code in billing_codes]
    if payer_name and schema.rate_payer_column and lookup_level == "hospital_payer":
        where_clauses.append(
            f"lower(CAST({gold_bill_support.quote_ident(schema.rate_payer_column)} AS VARCHAR)) LIKE lower(?)"
        )
        params.append(f"%{payer_name}%")
    if "setting" in supported_columns:
        expected_setting = "INPATIENT" if encounter_regime == "inpatient" else "OUTPATIENT"
        where_clauses.append(
            f"(upper(trim(CAST(setting AS VARCHAR))) = '{expected_setting}' OR setting IS NULL)"
        )

    description_expression = (
        f"max(CAST({gold_bill_support.quote_ident(schema.rate_description_column)} AS VARCHAR))"
        if schema.rate_description_column
        else "NULL"
    )
    gross_expression = bounded_positive_amount_expression(
        directory_gross_charge_expression(schema),
        max_quote_amount=max_quote_amount,
    )
    standard_dollar_expression = _median_bounded_amount_expression(
        (
            f"try_cast({gold_bill_support.quote_ident(schema.rate_standard_dollar_column)} AS DOUBLE)"
            if schema.rate_standard_dollar_column
            else "NULL"
        ),
        max_quote_amount=max_quote_amount,
    )
    percentage_charge_expression = _median_bounded_amount_expression(
        directory_percentage_charge_expression(schema),
        max_quote_amount=max_quote_amount,
    )
    minimum_expression = _median_bounded_amount_expression(
        (
            f"try_cast({gold_bill_support.quote_ident(schema.rate_minimum_column)} AS DOUBLE)"
            if schema.rate_minimum_column
            else "NULL"
        ),
        max_quote_amount=max_quote_amount,
    )
    maximum_expression = _median_bounded_amount_expression(
        (
            f"try_cast({gold_bill_support.quote_ident(schema.rate_maximum_column)} AS DOUBLE)"
            if schema.rate_maximum_column
            else "NULL"
        ),
        max_quote_amount=max_quote_amount,
    )
    selected_gross_expression = (
        gross_expression
        if scenario_name == "in_network"
        else f"coalesce({gross_expression}, {fallback_charge_expression})"
    )
    sql = (
        f"SELECT {code_expression} AS billing_code, "
        f"median({allowed_expression}) AS negotiated_rate, "
        f"median({selected_gross_expression}) AS gross_charge, "
        f"{standard_dollar_expression} AS standard_charge_dollar, "
        f"{percentage_charge_expression} AS percentage_charge_amount, "
        f"{minimum_expression} AS minimum, "
        f"{maximum_expression} AS maximum, "
        f"{description_expression} AS description, "
        "count(*) AS row_count "
        f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
        f"WHERE {' AND '.join(where_clauses)} "
        "GROUP BY 1"
    )
    rows = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
    quotes: dict[str, dict[str, Any]] = {}
    for row in rows:
        billing_code = gold_bill_support.normalize_code(row.get("billing_code"))
        if billing_code is None:
            continue
        quote = sanitize_rate_quote(
            {
                "billing_code": billing_code,
                "negotiated_rate": row.get("negotiated_rate"),
                "gross_charge": row.get("gross_charge"),
                "standard_charge_dollar": row.get("standard_charge_dollar"),
                "percentage_charge_amount": row.get("percentage_charge_amount"),
                "minimum": row.get("minimum"),
                "maximum": row.get("maximum"),
                "description": gold_bill_support.normalize_text(row.get("description")),
                "row_count": int(row.get("row_count") or 0),
                "lookup_level": lookup_level,
                "payer_name": payer_name,
                "identity_filter": None,
                "state_value": None,
            },
            max_quote_amount=max_quote_amount,
        )
        if quote_has_any_amount(quote):
            quotes[billing_code] = quote
    return quotes


def query_rate_level(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: list[str],
    lookup_level: str,
    payer_name: str | None,
    identity_filter: tuple[str, str] | None,
    state_value: str | None,
    encounter_regime: str = "outpatient",
    scenario_name: str = "in_network",
    max_quote_amount: float,
) -> dict[str, dict[str, Any]]:
    if schema.source_mode == "public_directory":
        return query_rate_level_from_public_directory_hospital(
            connection,
            schema,
            billing_codes=billing_codes,
            lookup_level=lookup_level,
            payer_name=payer_name,
            encounter_regime=encounter_regime,
            scenario_name=scenario_name,
            max_quote_amount=max_quote_amount,
        )
    if not billing_codes:
        return {}
    where_clauses = [
        f"upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_billing_code_column)} AS VARCHAR))) "
        f"IN ({', '.join('?' for _ in billing_codes)})"
    ]
    params: list[Any] = [code.upper() for code in billing_codes]

    if identity_filter is not None and lookup_level in {"hospital_payer", "hospital_average"}:
        column, value = identity_filter
        where_clauses.append(
            f"upper(trim(CAST({gold_bill_support.quote_ident(column)} AS VARCHAR))) = upper(?)"
        )
        params.append(value)

    if payer_name and schema.rate_payer_column and lookup_level == "hospital_payer":
        where_clauses.append(
            f"lower(CAST({gold_bill_support.quote_ident(schema.rate_payer_column)} AS VARCHAR)) LIKE lower(?)"
        )
        params.append(f"%{payer_name}%")

    negotiated_expr = (
        "median("
        + bounded_positive_amount_expression(
            gold_bill_support.quote_ident(schema.rate_negotiated_rate_column),
            max_quote_amount=max_quote_amount,
        )
        + ")"
        if schema.rate_negotiated_rate_column
        else "NULL"
    )
    gross_expr = (
        _median_bounded_amount_expression(
            gold_bill_support.quote_ident(schema.rate_gross_charge_column),
            max_quote_amount=max_quote_amount,
        )
        if schema.rate_gross_charge_column
        else "NULL"
    )
    standard_dollar_expr = (
        _median_bounded_amount_expression(
            gold_bill_support.quote_ident(schema.rate_standard_dollar_column),
            max_quote_amount=max_quote_amount,
        )
        if schema.rate_standard_dollar_column
        else "NULL"
    )
    percentage_charge_expr = _median_bounded_amount_expression(
        rate_percentage_charge_expression(schema),
        max_quote_amount=max_quote_amount,
    )
    minimum_expr = (
        _median_bounded_amount_expression(
            gold_bill_support.quote_ident(schema.rate_minimum_column),
            max_quote_amount=max_quote_amount,
        )
        if schema.rate_minimum_column
        else "NULL"
    )
    maximum_expr = (
        _median_bounded_amount_expression(
            gold_bill_support.quote_ident(schema.rate_maximum_column),
            max_quote_amount=max_quote_amount,
        )
        if schema.rate_maximum_column
        else "NULL"
    )
    sql = (
        f"SELECT upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_billing_code_column)} AS VARCHAR))) AS billing_code, "
        f"{negotiated_expr} AS negotiated_rate, "
        f"{gross_expr} AS gross_charge, "
        f"{standard_dollar_expr} AS standard_charge_dollar, "
        f"{percentage_charge_expr} AS percentage_charge_amount, "
        f"{minimum_expr} AS minimum, "
        f"{maximum_expr} AS maximum, "
        "count(*) AS row_count "
        f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
        f"WHERE {' AND '.join(where_clauses)} "
        "GROUP BY 1"
    )
    rows = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
    quotes: dict[str, dict[str, Any]] = {}
    for row in rows:
        billing_code = gold_bill_support.normalize_code(row.get("billing_code"))
        if billing_code is None:
            continue
        quote = sanitize_rate_quote(
            {
                "billing_code": billing_code,
                "negotiated_rate": row.get("negotiated_rate"),
                "gross_charge": row.get("gross_charge"),
                "standard_charge_dollar": row.get("standard_charge_dollar"),
                "percentage_charge_amount": row.get("percentage_charge_amount"),
                "minimum": row.get("minimum"),
                "maximum": row.get("maximum"),
                "row_count": int(row.get("row_count") or 0),
                "lookup_level": lookup_level,
                "payer_name": payer_name,
                "identity_filter": identity_filter,
                "state_value": state_value,
            },
            max_quote_amount=max_quote_amount,
        )
        if quote_has_any_amount(quote):
            quotes[billing_code] = quote
    return quotes


def cached_rate_quotes(
    connection: Any,
    lookup_keys: list[str],
    *,
    cache_lock: threading.Lock | None = None,
) -> dict[str, dict[str, Any]]:
    if not lookup_keys:
        return {}
    placeholders = ", ".join("?" for _ in lookup_keys)
    rows = gold_bill_support.synchronized_call(
        cache_lock,
        lambda: gold_bill_support.fetch_rows_as_dicts(
            connection.execute(
                f"SELECT lookup_key, quote_json FROM rate_quote_cache WHERE lookup_key IN ({placeholders})",
                lookup_keys,
            )
        ),
    )
    return {
        str(row["lookup_key"]): json.loads(str(row["quote_json"]))
        for row in rows
    }


def query_selected_hospital_exact_professional_rows(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: Sequence[str],
    encounter_regime: str,
    identity_filter: tuple[str, str] | None,
    max_quote_amount: float,
) -> dict[str, dict[str, Any]]:
    requested_codes = gold_bill_support.unique_preserving_order(
        code
        for code in (
            gold_bill_support.normalize_code(billing_code)
            for billing_code in billing_codes
        )
        if code is not None
    )
    if not requested_codes:
        return {}
    if schema.source_mode == "public_directory":
        supported_columns = attach_alias_table_description(connection, schema.rate_table)
        code_expression = directory_rate_code_expression(schema)
        description_expression = (
            f"max(CAST({gold_bill_support.quote_ident(schema.rate_description_column)} AS VARCHAR))"
            if schema.rate_description_column
            else "NULL"
        )
        where_clauses = [
            f"{code_expression} IN ({', '.join('?' for _ in requested_codes)})",
        ]
        params: list[Any] = [code.upper() for code in requested_codes]
        if "setting" in supported_columns:
            expected_setting = "INPATIENT" if encounter_regime == "inpatient" else "OUTPATIENT"
            where_clauses.append(
                f"(upper(trim(CAST(setting AS VARCHAR))) = '{expected_setting}' OR setting IS NULL)"
            )
        sql = (
            f"SELECT {code_expression} AS billing_code, "
            f"median({bounded_positive_amount_expression(directory_allowed_amount_expression(schema), max_quote_amount=max_quote_amount)}) AS negotiated_rate, "
            f"{_median_bounded_amount_expression(directory_gross_charge_expression(schema), max_quote_amount=max_quote_amount)} AS gross_charge, "
            f"{_median_bounded_amount_expression((f'try_cast({gold_bill_support.quote_ident(schema.rate_standard_dollar_column)} AS DOUBLE)' if schema.rate_standard_dollar_column else 'NULL'), max_quote_amount=max_quote_amount)} AS standard_charge_dollar, "
            f"{_median_bounded_amount_expression(directory_percentage_charge_expression(schema), max_quote_amount=max_quote_amount)} AS percentage_charge_amount, "
            f"{_median_bounded_amount_expression((f'try_cast({gold_bill_support.quote_ident(schema.rate_minimum_column)} AS DOUBLE)' if schema.rate_minimum_column else 'NULL'), max_quote_amount=max_quote_amount)} AS minimum, "
            f"{_median_bounded_amount_expression((f'try_cast({gold_bill_support.quote_ident(schema.rate_maximum_column)} AS DOUBLE)' if schema.rate_maximum_column else 'NULL'), max_quote_amount=max_quote_amount)} AS maximum, "
            f"{description_expression} AS description, "
            "count(*) AS row_count "
            f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
            f"WHERE {' AND '.join(where_clauses)} "
            "GROUP BY 1"
        )
    else:
        if identity_filter is None:
            return {}
        column, value = identity_filter
        sql = (
            f"SELECT upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_billing_code_column)} AS VARCHAR))) AS billing_code, "
            f"{('median(' + bounded_positive_amount_expression(gold_bill_support.quote_ident(schema.rate_negotiated_rate_column), max_quote_amount=max_quote_amount) + ')' if schema.rate_negotiated_rate_column else 'NULL')} AS negotiated_rate, "
            f"{(_median_bounded_amount_expression(gold_bill_support.quote_ident(schema.rate_gross_charge_column), max_quote_amount=max_quote_amount) if schema.rate_gross_charge_column else 'NULL')} AS gross_charge, "
            f"{(_median_bounded_amount_expression(gold_bill_support.quote_ident(schema.rate_standard_dollar_column), max_quote_amount=max_quote_amount) if schema.rate_standard_dollar_column else 'NULL')} AS standard_charge_dollar, "
            f"{_median_bounded_amount_expression(rate_percentage_charge_expression(schema), max_quote_amount=max_quote_amount)} AS percentage_charge_amount, "
            f"{(_median_bounded_amount_expression(gold_bill_support.quote_ident(schema.rate_minimum_column), max_quote_amount=max_quote_amount) if schema.rate_minimum_column else 'NULL')} AS minimum, "
            f"{(_median_bounded_amount_expression(gold_bill_support.quote_ident(schema.rate_maximum_column), max_quote_amount=max_quote_amount) if schema.rate_maximum_column else 'NULL')} AS maximum, "
            "count(*) AS row_count "
            f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
            f"WHERE upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_billing_code_column)} AS VARCHAR))) IN ({', '.join('?' for _ in requested_codes)}) "
            f"AND upper(trim(CAST({gold_bill_support.quote_ident(column)} AS VARCHAR))) = upper(?) "
            "GROUP BY 1"
        )
        params = [code.upper() for code in requested_codes] + [value]
    rows = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
    quotes: dict[str, dict[str, Any]] = {}
    for row in rows:
        billing_code = gold_bill_support.normalize_code(row.get("billing_code"))
        if billing_code is None:
            continue
        quote = sanitize_rate_quote(
            {
                "billing_code": billing_code,
                "negotiated_rate": row.get("negotiated_rate"),
                "gross_charge": row.get("gross_charge"),
                "standard_charge_dollar": row.get("standard_charge_dollar"),
                "percentage_charge_amount": row.get("percentage_charge_amount"),
                "minimum": row.get("minimum"),
                "maximum": row.get("maximum"),
                "description": gold_bill_support.normalize_text(row.get("description")),
                "row_count": int(row.get("row_count") or 0),
                "lookup_level": "hospital_exact",
                "payer_name": None,
                "identity_filter": identity_filter,
                "state_value": None,
            },
            max_quote_amount=max_quote_amount,
        )
        if quote_has_any_amount(quote) or quote_has_charge_anchor(quote):
            quotes[billing_code] = quote
    return quotes


def resolve_selected_hospital_exact_professional_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    session: requests.Session,
    billing_codes: Sequence[str],
    hospital: dict[str, Any],
    encounter_regime: str,
    max_quote_amount: float,
    cache_lock: threading.Lock | None = None,
) -> dict[str, dict[str, Any]]:
    requested_codes = gold_bill_support.unique_preserving_order(
        code
        for code in (
            gold_bill_support.normalize_code(billing_code)
            for billing_code in billing_codes
        )
        if code is not None
    )
    if not requested_codes:
        return {}
    cached_exact_quotes = hospital.get("_professional_exact_quote_by_code")
    if isinstance(cached_exact_quotes, dict) and all(
        code in cached_exact_quotes for code in requested_codes
    ):
        return {
            code: dict(cached_exact_quotes[code])
            for code in requested_codes
            if code in cached_exact_quotes
        }
    identity_filter = choose_rate_identity_filter(hospital, schema)
    if schema.source_mode != "public_directory":
        exact_quotes = query_selected_hospital_exact_professional_rows(
            connection,
            schema,
            billing_codes=requested_codes,
            encounter_regime=encounter_regime,
            identity_filter=identity_filter,
            max_quote_amount=max_quote_amount,
        )
        hospital["_professional_exact_quote_by_code"] = {
            **(cached_exact_quotes or {}),
            **exact_quotes,
        }
        return exact_quotes

    remote_connection: Any | None = None
    try:
        remote_connection, remote_schema, hospital_duckdb_url = (
            open_public_directory_hospital_database(
                connection,
                session,
                hospital,
                cache_lock=cache_lock,
            )
        )
        exact_quotes = query_selected_hospital_exact_professional_rows(
            remote_connection,
            remote_schema,
            billing_codes=requested_codes,
            encounter_regime=encounter_regime,
            identity_filter=None,
            max_quote_amount=max_quote_amount,
        )
        hospital["_professional_exact_quote_by_code"] = {
            **(cached_exact_quotes or {}),
            **exact_quotes,
        }
        return exact_quotes
    finally:
        if remote_connection is not None:
            remote_connection.close()


def professional_billing_code_family(
    billing_code: str | None,
) -> str | None:
    normalized_code = gold_bill_support.normalize_code(billing_code)
    if normalized_code is None:
        return None
    if normalized_code.isdigit():
        return normalized_code[:3] if len(normalized_code) >= 3 else None
    if (
        len(normalized_code) >= 3
        and normalized_code[0].isalpha()
        and normalized_code[1:].isalnum()
    ):
        return normalized_code[:3]
    return None


def build_hospital_local_professional_support(
    priced_professional_quotes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    support_entries: list[dict[str, Any]] = []
    ratios_by_family: dict[str, list[float]] = {}
    hospital_ratios: list[float] = []
    for quote in priced_professional_quotes:
        billing_code = gold_bill_support.normalize_code(quote.get("billing_code"))
        if billing_code is None:
            continue
        allowed_amount = gold_bill_support.parse_float(quote.get("negotiated_rate"))
        if allowed_amount is None or allowed_amount <= 0:
            continue
        charge_anchor, charge_field = quote_charge_anchor(quote)
        if charge_anchor is None or charge_field is None or charge_anchor <= 0:
            continue
        ratio = allowed_amount / charge_anchor
        if ratio <= 0:
            continue
        support_entries.append(
            {
                "billing_code": billing_code,
                "family": professional_billing_code_family(billing_code),
                "ratio": ratio,
                "charge_field": charge_field,
            }
        )
        hospital_ratios.append(ratio)
        family = professional_billing_code_family(billing_code)
        if family is not None:
            ratios_by_family.setdefault(family, []).append(ratio)
    return {
        "support_entries": support_entries,
        "ratios_by_family": ratios_by_family,
        "hospital_ratios": hospital_ratios,
    }


def build_hospital_local_estimate_quote(
    exact_quote: dict[str, Any],
    support_data: dict[str, Any],
    *,
    scenario_name: str,
    max_quote_amount: float,
) -> dict[str, Any] | None:
    billing_code = gold_bill_support.normalize_code(exact_quote.get("billing_code"))
    if billing_code is None:
        return None
    charge_anchor, charge_field = quote_charge_anchor(exact_quote)
    if charge_anchor is None or charge_field is None or charge_anchor <= 0:
        return None
    family = professional_billing_code_family(billing_code)
    family_ratios = (
        list((support_data.get("ratios_by_family") or {}).get(family) or [])
        if family is not None
        else []
    )
    if family_ratios:
        estimate_basis = "family"
        support_ratios = family_ratios
    else:
        estimate_basis = "hospital"
        support_ratios = list(support_data.get("hospital_ratios") or [])
    if not support_ratios:
        return None
    estimate_ratio = statistics.median(support_ratios)
    estimated_allowed_raw = estimate_ratio * charge_anchor
    if estimated_allowed_raw <= 0:
        return None
    sanitized_anchor = sanitized_quote_amount(
        charge_anchor,
        max_quote_amount=max_quote_amount,
    )
    if sanitized_anchor is None:
        return None
    if scenario_name == "in_network":
        estimated_allowed_raw = min(estimated_allowed_raw, sanitized_anchor, max_quote_amount)
    estimated_allowed = sanitized_quote_amount(
        estimated_allowed_raw,
        max_quote_amount=max_quote_amount,
    )
    if estimated_allowed is None:
        return None
    return {
        "billing_code": billing_code,
        "negotiated_rate": estimated_allowed,
        "gross_charge": sanitized_anchor,
        "standard_charge_dollar": exact_quote.get("standard_charge_dollar"),
        "percentage_charge_amount": exact_quote.get("percentage_charge_amount"),
        "minimum": exact_quote.get("minimum"),
        "maximum": exact_quote.get("maximum"),
        "description": gold_bill_support.normalize_text(exact_quote.get("description")),
        "row_count": int(exact_quote.get("row_count") or 0),
        "lookup_level": "hospital_local_estimate",
        "payer_name": None,
        "identity_filter": exact_quote.get("identity_filter"),
        "state_value": exact_quote.get("state_value"),
        "baseline_amount": estimated_allowed,
        "estimate_basis": estimate_basis,
        "estimate_support_line_count": len(support_ratios),
        "estimate_charge_field": charge_field,
        "estimate_ratio": round(float(estimate_ratio), 6),
    }


def cached_professional_quote_corpus(
    connection: Any,
    *,
    billing_code: str,
    scenario_name: str,
    payer_name: str | None,
    max_quote_amount: float,
    hospital_ids: set[str] | None = None,
    match_payer: bool = True,
    cache_lock: threading.Lock | None = None,
) -> list[dict[str, Any]]:
    normalized_code = gold_bill_support.normalize_code(billing_code)
    if normalized_code is None:
        return []
    rows = gold_bill_support.synchronized_call(
        cache_lock,
        lambda: gold_bill_support.fetch_rows_as_dicts(
            connection.execute(
                """
                SELECT lookup_key, quote_json
                FROM rate_quote_cache
                WHERE json_extract_string(lookup_key, '$.billing_code') = ?
                """,
                [normalized_code],
            )
        ),
    )
    if scenario_name == "in_network":
        allowed_levels = (
            {"hospital_payer", "hospital_average"}
            if match_payer
            else {"hospital_average"}
        )
    else:
        allowed_levels = {"hospital_average"}
    normalized_payer = gold_bill_support.normalize_text(payer_name)
    corpus: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for row in rows:
        lookup_key = json.loads(str(row["lookup_key"]))
        quote = sanitize_rate_quote(
            json.loads(str(row["quote_json"])),
            max_quote_amount=max_quote_amount,
        )
        lookup_level = gold_bill_support.normalize_text(lookup_key.get("lookup_level"))
        if lookup_level not in allowed_levels:
            continue
        lookup_payer = gold_bill_support.normalize_text(lookup_key.get("payer_name"))
        if match_payer and scenario_name == "in_network" and lookup_payer != normalized_payer:
            continue
        identity_filter = lookup_key.get("identity_filter") or []
        hospital_id = (
            gold_bill_support.normalize_text(identity_filter[1])
            if len(identity_filter) >= 2
            else None
        )
        if hospital_ids is not None and hospital_id not in hospital_ids:
            continue
        dedupe_key = (hospital_id or "", lookup_level or "")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        if not quote_has_any_amount(quote):
            continue
        corpus.append(
            {
                **quote,
                "lookup_level": lookup_level,
                "payer_name": lookup_payer,
                "identity_filter": identity_filter,
                "estimate_source_hospital_id": hospital_id,
            }
        )
    return corpus


def cached_professional_family_quote_corpus(
    connection: Any,
    *,
    billing_family: str,
    scenario_name: str,
    payer_name: str | None,
    max_quote_amount: float,
    hospital_ids: set[str] | None = None,
    match_payer: bool = True,
    cache_lock: threading.Lock | None = None,
) -> list[dict[str, Any]]:
    normalized_family = gold_bill_support.normalize_text(billing_family)
    if normalized_family is None:
        return []
    rows = gold_bill_support.synchronized_call(
        cache_lock,
        lambda: gold_bill_support.fetch_rows_as_dicts(
            connection.execute(
                """
                SELECT lookup_key, quote_json
                FROM rate_quote_cache
                WHERE substr(json_extract_string(lookup_key, '$.billing_code'), 1, 3) = ?
                """,
                [normalized_family],
            )
        ),
    )
    if scenario_name == "in_network":
        allowed_levels = (
            {"hospital_payer", "hospital_average"}
            if match_payer
            else {"hospital_average"}
        )
    else:
        allowed_levels = {"hospital_average"}
    normalized_payer = gold_bill_support.normalize_text(payer_name)
    corpus: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for row in rows:
        lookup_key = json.loads(str(row["lookup_key"]))
        quote = sanitize_rate_quote(
            json.loads(str(row["quote_json"])),
            max_quote_amount=max_quote_amount,
        )
        quote_code = gold_bill_support.normalize_code(lookup_key.get("billing_code"))
        if professional_billing_code_family(quote_code) != normalized_family:
            continue
        lookup_level = gold_bill_support.normalize_text(lookup_key.get("lookup_level"))
        if lookup_level not in allowed_levels:
            continue
        lookup_payer = gold_bill_support.normalize_text(lookup_key.get("payer_name"))
        if match_payer and scenario_name == "in_network" and lookup_payer != normalized_payer:
            continue
        identity_filter = lookup_key.get("identity_filter") or []
        hospital_id = (
            gold_bill_support.normalize_text(identity_filter[1])
            if len(identity_filter) >= 2
            else None
        )
        if hospital_ids is not None and hospital_id not in hospital_ids:
            continue
        dedupe_key = (quote_code or "", hospital_id or "", lookup_level or "")
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        if not quote_has_any_amount(quote):
            continue
        corpus.append(
            {
                **quote,
                "lookup_level": lookup_level,
                "payer_name": lookup_payer,
                "identity_filter": identity_filter,
                "estimate_source_hospital_id": hospital_id,
            }
        )
    return corpus


def build_cached_corpus_ratio_estimate_quote(
    exact_quote: dict[str, Any],
    cached_quotes: Sequence[dict[str, Any]],
    *,
    scenario_name: str,
    max_quote_amount: float,
    estimate_scope: str,
) -> dict[str, Any] | None:
    support_ratios: list[float] = []
    for quote in cached_quotes:
        allowed_amount = gold_bill_support.parse_float(quote.get("negotiated_rate"))
        charge_anchor, _charge_field = quote_charge_anchor(quote)
        if (
            allowed_amount is None
            or allowed_amount <= 0
            or charge_anchor is None
            or charge_anchor <= 0
        ):
            continue
        ratio = allowed_amount / charge_anchor
        if ratio > 0:
            support_ratios.append(ratio)
    if not support_ratios:
        return None
    estimate_quote = build_hospital_local_estimate_quote(
        exact_quote,
        {
            "ratios_by_family": {},
            "hospital_ratios": support_ratios,
        },
        scenario_name=scenario_name,
        max_quote_amount=max_quote_amount,
    )
    if estimate_quote is None:
        return None
    estimate_quote["lookup_level"] = "cached_corpus_estimate"
    estimate_quote["estimate_basis"] = "cached_code_ratio"
    estimate_quote["estimate_scope"] = estimate_scope
    estimate_quote["estimate_support_line_count"] = len(support_ratios)
    estimate_quote["estimate_source_lookup_level"] = "cached_corpus"
    return estimate_quote


def build_cached_corpus_direct_quote(
    billing_code: str,
    cached_quotes: Sequence[dict[str, Any]],
    *,
    max_quote_amount: float,
    estimate_scope: str,
) -> dict[str, Any] | None:
    normalized_code = gold_bill_support.normalize_code(billing_code)
    if normalized_code is None:
        return None
    allowed_amounts = [
        amount
        for amount in (
            gold_bill_support.parse_float(quote.get("negotiated_rate"))
            for quote in cached_quotes
        )
        if amount is not None and amount > 0
    ]
    if not allowed_amounts:
        return None
    allowed_amount = sanitized_quote_amount(
        statistics.median(allowed_amounts),
        max_quote_amount=max_quote_amount,
    )
    if allowed_amount is None:
        return None
    charge_anchors = [
        anchor
        for anchor, _field in (quote_charge_anchor(quote) for quote in cached_quotes)
        if anchor is not None and anchor > 0
    ]
    gross_charge = sanitized_quote_amount(
        statistics.median(charge_anchors) if charge_anchors else allowed_amount,
        max_quote_amount=max_quote_amount,
    )
    if gross_charge is None:
        gross_charge = allowed_amount
    description = next(
        (
            gold_bill_support.normalize_text(quote.get("description"))
            for quote in cached_quotes
            if gold_bill_support.normalize_text(quote.get("description")) is not None
        ),
        None,
    )
    return {
        "billing_code": normalized_code,
        "negotiated_rate": allowed_amount,
        "gross_charge": gross_charge,
        "description": description,
        "row_count": len(allowed_amounts),
        "lookup_level": "cached_corpus_quote",
        "payer_name": None,
        "identity_filter": None,
        "state_value": None,
        "baseline_amount": allowed_amount,
        "estimate_basis": "cached_code_median",
        "estimate_support_line_count": len(allowed_amounts),
        "estimate_scope": estimate_scope,
        "estimate_source_lookup_level": "cached_corpus",
    }


def cache_rate_quote(
    connection: Any,
    lookup_key: str,
    quote: dict[str, Any],
    *,
    cache_lock: threading.Lock | None = None,
) -> None:
    gold_bill_support.synchronized_call(
        cache_lock,
        lambda: connection.execute(
            """
            INSERT OR REPLACE INTO rate_quote_cache (lookup_key, quote_json, updated_at)
            VALUES (?, ?, ?)
            """,
            [lookup_key, json.dumps(quote, sort_keys=True), gold_bill_support.iso_timestamp()],
        ),
    )


def build_trilliant_prior_cache_miss(
    *,
    reason: str,
    state_value: str | None,
    hospital_count: int = 0,
    supporting_row_count: int = 0,
) -> dict[str, Any]:
    return {
        "cache_entry_type": TRILLIANT_PRIOR_CACHE_MISS_ENTRY_TYPE,
        "cache_miss_reason": reason,
        "lookup_level": "trilliant_prior",
        "state_value": state_value,
        "hospital_count": hospital_count,
        "supporting_row_count": supporting_row_count,
    }


def cached_trilliant_prior_quote(
    connection: Any,
    lookup_key: str,
    *,
    max_quote_amount: float,
    cache_lock: threading.Lock | None = None,
) -> tuple[str, dict[str, Any] | None]:
    cached = cached_rate_quotes(connection, [lookup_key], cache_lock=cache_lock)
    entry = cached.get(lookup_key)
    if entry is None:
        return "missing", None
    if (
        gold_bill_support.normalize_text(entry.get("cache_entry_type"))
        == TRILLIANT_PRIOR_CACHE_MISS_ENTRY_TYPE
    ):
        return "miss", None
    cached_quote = sanitize_rate_quote(
        entry,
        max_quote_amount=max_quote_amount,
    )
    if quote_has_any_amount(cached_quote):
        return "quote", cached_quote
    return "miss", None


def cached_trilliant_prior_quotes(
    connection: Any,
    lookup_keys: dict[str, str],
    *,
    max_quote_amount: float,
    cache_lock: threading.Lock | None = None,
) -> dict[str, tuple[str, dict[str, Any] | None]]:
    if not lookup_keys:
        return {}
    cached = cached_rate_quotes(
        connection,
        list(lookup_keys.values()),
        cache_lock=cache_lock,
    )
    statuses: dict[str, tuple[str, dict[str, Any] | None]] = {}
    for billing_code, lookup_key in lookup_keys.items():
        entry = cached.get(lookup_key)
        if entry is None:
            statuses[billing_code] = ("missing", None)
            continue
        if (
            gold_bill_support.normalize_text(entry.get("cache_entry_type"))
            == TRILLIANT_PRIOR_CACHE_MISS_ENTRY_TYPE
        ):
            statuses[billing_code] = ("miss", None)
            continue
        cached_quote = sanitize_rate_quote(
            entry,
            max_quote_amount=max_quote_amount,
        )
        if quote_has_any_amount(cached_quote):
            statuses[billing_code] = ("quote", cached_quote)
            continue
        statuses[billing_code] = ("miss", None)
    return statuses


def build_trilliant_prior_lookup_key(
    *,
    billing_code: str,
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    source_mode: str,
    max_quote_amount: float,
) -> str:
    return json.dumps(
        {
            "cache_version": TRILLIANT_PRIOR_CACHE_VERSION,
            "lookup_type": "trilliant_prior",
            "billing_code": billing_code,
            "encounter_regime": encounter_regime,
            "claim_class": claim_class,
            "state_value": state_value,
            "source_mode": source_mode,
            "max_quote_amount": max_quote_amount,
        },
        sort_keys=True,
    )


def _normalize_prior_billing_codes(billing_codes: Sequence[str]) -> list[str]:
    return gold_bill_support.unique_preserving_order(
        billing_code
        for billing_code in (
            gold_bill_support.normalize_code(code)
            for code in billing_codes
        )
        if billing_code is not None
    )


def _trilliant_prior_scope_name(state_value: str | None) -> str:
    return "state" if state_value else "national"


def _trilliant_prior_scope_hospital_cap(state_value: str | None) -> int:
    return (
        PUBLIC_DIRECTORY_STATE_PRIOR_HOSPITAL_CAP
        if state_value
        else PUBLIC_DIRECTORY_NATIONAL_PRIOR_HOSPITAL_CAP
    )


def _trilliant_prior_scope_min_hospitals(scope: str) -> int:
    if scope == "state":
        return TRILLIANT_STATE_PRIOR_MIN_HOSPITALS
    return TRILLIANT_NATIONAL_PRIOR_MIN_HOSPITALS


def _log_trilliant_prior_progress(
    *,
    logger: CliLogger | None,
    case_id: str | None,
    scope: str,
    codes_remaining: int,
    hospitals_scanned: int,
    hospital_scan_cap: int,
    started_at: float,
    last_progress_log_at: float | None,
) -> float | None:
    if logger is None:
        return last_progress_log_at
    now = time.monotonic()
    elapsed = now - started_at
    if elapsed < TRILLIANT_PRIOR_PROGRESS_LOG_DELAY_SECONDS:
        return last_progress_log_at
    if (
        last_progress_log_at is not None
        and now - last_progress_log_at < TRILLIANT_PRIOR_PROGRESS_LOG_INTERVAL_SECONDS
    ):
        return last_progress_log_at
    logger.info(
        "Trilliant prior progress",
        case_id=case_id,
        scope=scope,
        codes_remaining=codes_remaining,
        hospitals_scanned=hospitals_scanned,
        hospital_scan_cap=hospital_scan_cap,
        elapsed=elapsed,
    )
    return now


def _build_trilliant_prior_quote(
    *,
    billing_code: str,
    negotiated_rate: float,
    supporting_row_count: int,
    hospital_count: int,
    encounter_regime: str,
    state_value: str | None,
    hospitals_scanned: int | None = None,
    hospital_cap: int | None = None,
) -> dict[str, Any]:
    quote = {
        "billing_code": billing_code,
        "negotiated_rate": negotiated_rate,
        "gross_charge": None,
        "row_count": supporting_row_count,
        "lookup_level": "trilliant_prior",
        "payer_name": None,
        "identity_filter": None,
        "state_value": state_value,
        "baseline_amount": negotiated_rate,
        "hospital_count": hospital_count,
        "supporting_row_count": supporting_row_count,
        "encounter_regime": encounter_regime,
        "prior_lookup_scope": _trilliant_prior_scope_name(state_value),
    }
    if hospitals_scanned is not None:
        quote["prior_hospital_scan_count"] = hospitals_scanned
    if hospital_cap is not None:
        quote["prior_hospital_scan_cap"] = hospital_cap
    return quote


def query_trilliant_prior_quotes_with_miss_cacheability(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: Sequence[str],
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    public_directory_candidates: Sequence[dict[str, Any]] | None = None,
    hospital_cap_override: int | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
    started_at: float | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, bool], int]:
    del claim_class
    requested_codes = _normalize_prior_billing_codes(billing_codes)
    if not requested_codes:
        return {}, {}, 0

    if schema.source_mode == "public_directory" and not relation_exists(
        connection,
        table_name=schema.rate_table,
    ):
        if session is None:
            return {}, {billing_code: False for billing_code in requested_codes}, 0
        candidates = list(public_directory_candidates or [])
        if not candidates:
            search_rows = fetch_public_trilliant_search_index(
                connection,
                session,
                cache_lock=cache_lock,
            )
            candidates, _query_mode = filter_public_directory_candidates(
                search_rows,
                state_value=state_value,
                max_candidates=max(1, len(search_rows)),
            )
        scope = _trilliant_prior_scope_name(state_value)
        hospital_cap = (
            max(1, int(hospital_cap_override))
            if hospital_cap_override is not None
            else _trilliant_prior_scope_hospital_cap(state_value)
        )
        if logger is not None:
            logger.info(
                "Trilliant prior scope start",
                case_id=case_id,
                scope=scope,
                code_count=len(requested_codes),
                hospital_scan_cap=hospital_cap,
            )
        aggregate_by_code = {
            billing_code: {
                "hospital_medians": [],
                "supporting_row_count": 0,
            }
            for billing_code in requested_codes
        }
        unresolved_codes = set(requested_codes)
        scan_complete = True
        hospitals_scanned = 0
        last_progress_log_at: float | None = None
        effective_started_at = started_at if started_at is not None else time.monotonic()
        for candidate in candidates[:hospital_cap]:
            if not unresolved_codes:
                break
            hospitals_scanned += 1
            remote_connection: Any | None = None
            try:
                remote_connection, remote_schema, _hospital_duckdb_url = (
                    open_public_directory_hospital_database(
                        connection,
                        session,
                        candidate,
                        cache_lock=cache_lock,
                    )
                )
                hospital_quotes = query_rate_level_from_public_directory_hospital(
                    remote_connection,
                    remote_schema,
                    billing_codes=sorted(unresolved_codes),
                    lookup_level="hospital_average",
                    payer_name=None,
                    encounter_regime=encounter_regime,
                    scenario_name="in_network",
                    max_quote_amount=max_quote_amount,
                )
                for billing_code in tuple(unresolved_codes):
                    hospital_quote = hospital_quotes.get(billing_code)
                    negotiated_rate = (
                        sanitized_quote_amount(
                            hospital_quote.get("negotiated_rate"),
                            max_quote_amount=max_quote_amount,
                        )
                        if hospital_quote is not None
                        else None
                    )
                    hospital_row_count = (
                        int(hospital_quote.get("row_count") or 0)
                        if hospital_quote is not None
                        else 0
                    )
                    if negotiated_rate is None or hospital_row_count <= 0:
                        continue
                    aggregate_by_code[billing_code]["hospital_medians"].append(negotiated_rate)
                    aggregate_by_code[billing_code]["supporting_row_count"] += hospital_row_count
            except Exception:
                scan_complete = False
            finally:
                if remote_connection is not None:
                    remote_connection.close()
            last_progress_log_at = _log_trilliant_prior_progress(
                logger=logger,
                case_id=case_id,
                scope=scope,
                codes_remaining=len(unresolved_codes),
                hospitals_scanned=hospitals_scanned,
                hospital_scan_cap=hospital_cap,
                started_at=effective_started_at,
                last_progress_log_at=last_progress_log_at,
            )
        quotes: dict[str, dict[str, Any]] = {}
        miss_cacheability = {billing_code: scan_complete for billing_code in requested_codes}
        for billing_code in requested_codes:
            hospital_medians = aggregate_by_code[billing_code]["hospital_medians"]
            supporting_row_count = int(aggregate_by_code[billing_code]["supporting_row_count"] or 0)
            if not hospital_medians or supporting_row_count <= 0:
                continue
            negotiated_rate = gold_bill_support.round_currency(
                float(statistics.median(hospital_medians))
            )
            quotes[billing_code] = _build_trilliant_prior_quote(
                billing_code=billing_code,
                negotiated_rate=negotiated_rate,
                supporting_row_count=supporting_row_count,
                hospital_count=len(hospital_medians),
                encounter_regime=encounter_regime,
                state_value=state_value,
                hospitals_scanned=hospitals_scanned,
                hospital_cap=hospital_cap,
            )
        return quotes, miss_cacheability, hospitals_scanned

    if schema.source_mode == "public_directory":
        supported_columns = attach_alias_table_description(connection, schema.rate_table)
        raw_allowed_expression = directory_allowed_amount_expression(schema)
        if raw_allowed_expression == "NULL":
            return {}, {billing_code: True for billing_code in requested_codes}, 0
        allowed_expression = bounded_positive_amount_expression(
            raw_allowed_expression,
            max_quote_amount=max_quote_amount,
        )
        code_expression = directory_rate_code_expression(schema)
        where_clauses = [
            f"{code_expression} IN ({', '.join('?' for _ in requested_codes)})",
            f"{allowed_expression} IS NOT NULL",
        ]
        params: list[Any] = list(requested_codes)
    else:
        if not relation_exists(connection, table_name=schema.rate_table):
            return {}, {billing_code: True for billing_code in requested_codes}, 0
        supported_columns = attach_alias_table_description(connection, schema.rate_table)
        if schema.rate_negotiated_rate_column is None:
            return {}, {billing_code: True for billing_code in requested_codes}, 0
        allowed_expression = bounded_positive_amount_expression(
            gold_bill_support.quote_ident(schema.rate_negotiated_rate_column),
            max_quote_amount=max_quote_amount,
        )
        code_expression = (
            f"upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_billing_code_column)} AS VARCHAR)))"
        )
        where_clauses = [
            f"{code_expression} IN ({', '.join('?' for _ in requested_codes)})",
            f"{allowed_expression} IS NOT NULL",
        ]
        params = list(requested_codes)

    if "setting" in supported_columns:
        expected_setting = "INPATIENT" if encounter_regime == "inpatient" else "OUTPATIENT"
        where_clauses.append(
            f"(upper(trim(CAST(setting AS VARCHAR))) = '{expected_setting}' OR setting IS NULL)"
        )
    if state_value and schema.rate_state_column and schema.rate_state_column in supported_columns:
        where_clauses.append(
            f"upper(trim(CAST({gold_bill_support.quote_ident(schema.rate_state_column)} AS VARCHAR))) = upper(?)"
        )
        params.append(state_value)
    identity_columns = [
        column
        for column in schema.rate_identity_columns
        if column in supported_columns
    ]
    identity_expression = coalesced_identity_expression(identity_columns)
    sql = (
        "WITH filtered AS ("
        f"SELECT {code_expression} AS billing_code, "
        f"{identity_expression} AS hospital_identity, "
        f"{allowed_expression} AS allowed_amount "
        f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
        f"WHERE {' AND '.join(where_clauses)}"
        "), hospital_medians AS ("
        "SELECT billing_code, hospital_identity, median(allowed_amount) AS hospital_median, "
        "count(*) AS supporting_rows "
        "FROM filtered "
        "WHERE hospital_identity IS NOT NULL AND allowed_amount IS NOT NULL "
        "GROUP BY 1, 2"
        ") "
        "SELECT billing_code, "
        "median(hospital_median) AS negotiated_rate, "
        "count(*) AS hospital_count, "
        "coalesce(sum(supporting_rows), 0) AS supporting_row_count "
        "FROM hospital_medians "
        "GROUP BY 1"
    )
    rows = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
    quotes: dict[str, dict[str, Any]] = {}
    for row in rows:
        billing_code = gold_bill_support.normalize_code(row.get("billing_code"))
        if billing_code is None:
            continue
        negotiated_rate = sanitized_quote_amount(
            row.get("negotiated_rate"),
            max_quote_amount=max_quote_amount,
        )
        hospital_count = int(row.get("hospital_count") or 0)
        supporting_row_count = int(row.get("supporting_row_count") or 0)
        if negotiated_rate is None or hospital_count <= 0 or supporting_row_count <= 0:
            continue
        quotes[billing_code] = _build_trilliant_prior_quote(
            billing_code=billing_code,
            negotiated_rate=negotiated_rate,
            supporting_row_count=supporting_row_count,
            hospital_count=hospital_count,
            encounter_regime=encounter_regime,
            state_value=state_value,
        )
    return quotes, {billing_code: True for billing_code in requested_codes}, 0


def relation_exists(
    connection: Any,
    *,
    table_name: str,
) -> bool:
    try:
        attach_alias_table_description(connection, table_name)
    except Exception:
        return False
    return True


def coalesced_identity_expression(columns: Sequence[str]) -> str:
    if not columns:
        return "'__missing_identity__'"
    pieces = [
        f"nullif(trim(CAST({gold_bill_support.quote_ident(column)} AS VARCHAR)), '')"
        for column in columns
    ]
    return "coalesce(" + ", ".join(pieces) + ", '__missing_identity__')"


def source_claim_index(encounter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(claim["claim_id"]): claim
        for claim in encounter.get("claims", [])
        if claim.get("claim_id")
    }


def source_baseline_amount(
    occurrence: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    source_medicare_multiplier: float,
) -> float | None:
    for field in (
        "allowed_charge_amount",
        "medicare_payment_amount",
        "payment_amount",
    ):
        parsed = gold_bill_support.parse_float(occurrence.get(field))
        if parsed is not None and parsed > 0:
            return parsed

    claim = claim_index.get(str(occurrence.get("source_claim_id", "")))
    if claim:
        amounts = claim.get("amounts") or {}
        for field in ("allowed_charge_amount", "claim_payment_amount"):
            parsed = gold_bill_support.parse_float(amounts.get(field))
            if parsed is not None and parsed > 0:
                return parsed

    for field in ("submitted_charge_amount", "total_charge_amount"):
        parsed = gold_bill_support.parse_float(occurrence.get(field))
        if parsed is not None and parsed > 0:
            multiplier = max(1.0, source_medicare_multiplier)
            return parsed / multiplier
    return None


def observed_occurrence_allowed_amount(
    occurrence: dict[str, Any],
) -> tuple[float | None, str | None]:
    for field in (
        "allowed_charge_amount",
        "medicare_payment_amount",
        "payment_amount",
    ):
        parsed = gold_bill_support.parse_float(occurrence.get(field))
        if parsed is not None and parsed > 0:
            return parsed, field
    return None, None


def inpatient_length_of_stay(encounter: dict[str, Any]) -> int | None:
    encounter_window = encounter.get("encounter") or {}
    admission_date = gold_bill_support.normalize_text(
        encounter_window.get("admission_date")
    ) or gold_bill_support.normalize_text(encounter_window.get("from_date"))
    discharge_date = gold_bill_support.normalize_text(
        encounter_window.get("discharge_date")
    ) or gold_bill_support.normalize_text(encounter_window.get("thru_date"))
    try:
        parsed_admission = date.fromisoformat(admission_date) if admission_date else None
    except ValueError:
        parsed_admission = None
    try:
        parsed_discharge = date.fromisoformat(discharge_date) if discharge_date else None
    except ValueError:
        parsed_discharge = None
    if parsed_admission is None or parsed_discharge is None:
        return None
    return max(1, (parsed_discharge - parsed_admission).days)


def inpatient_anchor_floor_amount(length_of_stay: int | None) -> float:
    if length_of_stay is not None and length_of_stay >= 3:
        return INPATIENT_LONG_STAY_ALLOWED_FLOOR
    return INPATIENT_SHORT_STAY_ALLOWED_FLOOR


def source_claim_for_occurrence(
    occurrence: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    return claim_index.get(str(occurrence.get("source_claim_id", "")))


def positive_amount(value: Any) -> float | None:
    parsed = gold_bill_support.parse_float(value)
    if parsed is None or parsed <= 0:
        return None
    return parsed


def claim_charge_amount(claim: dict[str, Any] | None) -> float | None:
    amounts = (claim or {}).get("amounts") or {}
    for field in ("submitted_charge_amount", "total_charge_amount"):
        parsed = positive_amount(amounts.get(field))
        if parsed is not None:
            return parsed
    return None


def occurrence_or_claim_charge_amount(
    occurrence: dict[str, Any],
    claim: dict[str, Any] | None,
) -> float | None:
    for field in ("submitted_charge_amount", "total_charge_amount"):
        parsed = positive_amount(occurrence.get(field))
        if parsed is not None:
            return parsed
    return claim_charge_amount(claim)


def drg_family_source_allowed_amount(
    occurrence: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    *,
    source_medicare_multiplier: float,
) -> float | None:
    claim = source_claim_for_occurrence(occurrence, claim_index)
    for field in (
        "allowed_charge_amount",
        "medicare_payment_amount",
        "payment_amount",
    ):
        parsed = positive_amount(occurrence.get(field))
        if parsed is not None:
            return parsed
    amounts = (claim or {}).get("amounts") or {}
    for field in ("allowed_charge_amount", "claim_payment_amount"):
        parsed = positive_amount(amounts.get(field))
        if parsed is not None:
            return parsed
    charge_amount = occurrence_or_claim_charge_amount(occurrence, claim)
    if charge_amount is None:
        return None
    multiplier = max(1.0, source_medicare_multiplier)
    return charge_amount / multiplier


def build_imputed_rate_quote(
    occurrence: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    *,
    source_medicare_multiplier: float,
) -> dict[str, Any]:
    baseline = source_baseline_amount(occurrence, claim_index, source_medicare_multiplier)
    if baseline is None:
        baseline = 100.0
    negotiated_rate = gold_bill_support.round_currency(
        baseline * max(1.0, source_medicare_multiplier)
    )
    return {
        "billing_code": gold_bill_support.normalize_code(occurrence.get("code")),
        "negotiated_rate": negotiated_rate,
        "gross_charge": None,
        "row_count": 0,
        "lookup_level": "source_imputed",
        "payer_name": None,
        "identity_filter": None,
        "state_value": None,
        "baseline_amount": gold_bill_support.round_currency(baseline),
    }


def build_component_imputed_rate_quote(
    occurrence: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    *,
    encounter: dict[str, Any],
    source_medicare_multiplier: float,
) -> dict[str, Any]:
    code_system = normalize_code_system(occurrence.get("code_system"))
    encounter_regime = gold_bill_support.normalize_text(
        ((encounter.get("encounter") or {}).get("anchor_claim_type"))
    )
    if (
        code_system not in INPATIENT_DRG_FAMILY_CODE_SYSTEMS
        or encounter_regime != "inpatient"
    ):
        return build_imputed_rate_quote(
            occurrence,
            claim_index,
            source_medicare_multiplier=source_medicare_multiplier,
        )

    source_allowed_amount = drg_family_source_allowed_amount(
        occurrence,
        claim_index,
        source_medicare_multiplier=source_medicare_multiplier,
    )
    floor_amount = inpatient_anchor_floor_amount(inpatient_length_of_stay(encounter))
    negotiated_rate = max(source_allowed_amount or 0.0, floor_amount)
    gross_charge = occurrence_or_claim_charge_amount(
        occurrence,
        source_claim_for_occurrence(occurrence, claim_index),
    )
    return {
        "billing_code": gold_bill_support.normalize_code(occurrence.get("code")),
        "negotiated_rate": gold_bill_support.round_currency(negotiated_rate),
        "gross_charge": (
            gold_bill_support.round_currency(gross_charge)
            if gross_charge is not None
            else None
        ),
        "row_count": 0,
        "lookup_level": "inpatient_anchor_imputed",
        "payer_name": None,
        "identity_filter": None,
        "state_value": None,
        "baseline_amount": gold_bill_support.round_currency(negotiated_rate),
        "source_allowed_amount": (
            gold_bill_support.round_currency(source_allowed_amount)
            if source_allowed_amount is not None
            else None
        ),
        "floor_amount": gold_bill_support.round_currency(floor_amount),
        "floor_applied": source_allowed_amount is None or source_allowed_amount < floor_amount,
    }


def build_source_observed_quote(
    occurrence: dict[str, Any],
) -> dict[str, Any] | None:
    observed_amount, observed_field = observed_occurrence_allowed_amount(occurrence)
    if observed_amount is None:
        return None
    billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
    if billing_code is None:
        return None
    return {
        "billing_code": billing_code,
        "negotiated_rate": gold_bill_support.round_currency(observed_amount),
        "gross_charge": None,
        "row_count": 1,
        "lookup_level": "source_observed",
        "payer_name": None,
        "identity_filter": None,
        "state_value": None,
        "baseline_amount": gold_bill_support.round_currency(observed_amount),
        "source_observed_field": observed_field,
    }


def _component_quote_key(code_system: str, billing_code: str) -> str:
    lookup = inpatient_anchor_lookup(code_system, billing_code)
    normalized_code_system = lookup.code_system
    normalized_billing_code = lookup.billing_code
    if normalized_code_system is None or normalized_billing_code is None:
        raise ValueError(
            f"Unsupported component quote key: code_system={code_system!r}, billing_code={billing_code!r}"
        )
    return f"{normalized_code_system}:{normalized_billing_code}"


def query_public_directory_component_rate_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    occurrences: Sequence[dict[str, Any]],
    lookup_level: str,
    payer_name: str | None,
    scenario_name: str = "in_network",
    max_quote_amount: float,
) -> dict[str, dict[str, Any]]:
    if not occurrences:
        return {}
    supported_columns = attach_alias_table_description(connection, schema.rate_table)
    rows_out: dict[str, dict[str, Any]] = {}
    type_mappings = {
        "MS-DRG": [("ms_drg", None)],
        "APR-DRG": [("other_code1", "APR-DRG"), ("other_code2", "APR-DRG")],
        "DRG": [("other_code1", "DRG"), ("other_code2", "DRG")],
        "REVENUE_CODE": [("rc", None)],
        "ICD10_PCS": [("icd", None)],
    }
    raw_allowed_expression = directory_allowed_amount_expression(schema)
    raw_charge_fallback_expression = directory_charge_fallback_expression(schema)
    allowed_expression = bounded_positive_amount_expression(
        raw_allowed_expression,
        max_quote_amount=max_quote_amount,
    )
    fallback_charge_expression = bounded_positive_amount_expression(
        raw_charge_fallback_expression,
        max_quote_amount=max_quote_amount,
    )
    if scenario_name == "in_network" and raw_allowed_expression == "NULL":
        return {}
    if (
        scenario_name != "in_network"
        and raw_allowed_expression == "NULL"
        and raw_charge_fallback_expression == "NULL"
    ):
        return {}
    gross_expression = bounded_positive_amount_expression(
        directory_gross_charge_expression(schema),
        max_quote_amount=max_quote_amount,
    )
    selected_gross_expression = (
        gross_expression
        if scenario_name == "in_network"
        else f"coalesce({gross_expression}, {fallback_charge_expression})"
    )
    for occurrence in occurrences:
        lookup = inpatient_anchor_lookup(
            str(occurrence.get("code_system")),
            str(occurrence.get("code")),
        )
        code = lookup.billing_code
        code_system = lookup.code_system
        if code is None or code_system not in type_mappings:
            continue
        candidate_rows = type_mappings[code_system]
        found_quote = False
        for candidate_code in lookup.candidates:
            for code_column, type_value in candidate_rows:
                if code_column not in supported_columns:
                    continue
                where_clauses = [
                    f"upper(trim(CAST({gold_bill_support.quote_ident(code_column)} AS VARCHAR))) = ?",
                ]
                if scenario_name == "in_network":
                    where_clauses.append(f"{allowed_expression} IS NOT NULL")
                else:
                    where_clauses.append(
                        f"coalesce({allowed_expression}, {fallback_charge_expression}) IS NOT NULL"
                    )
                params: list[Any] = [candidate_code]
                if type_value is not None:
                    type_column = (
                        "other_code1_type" if code_column == "other_code1" else "other_code2_type"
                    )
                    if type_column not in supported_columns:
                        continue
                    where_clauses.append(
                        f"upper(trim(CAST({gold_bill_support.quote_ident(type_column)} AS VARCHAR))) = ?"
                    )
                    params.append(type_value)
                if "setting" in supported_columns:
                    where_clauses.append(
                        "(upper(trim(CAST(setting AS VARCHAR))) = 'INPATIENT' OR setting IS NULL)"
                    )
                if payer_name and schema.rate_payer_column and lookup_level == "hospital_payer":
                    where_clauses.append(
                        f"lower(CAST({gold_bill_support.quote_ident(schema.rate_payer_column)} AS VARCHAR)) LIKE lower(?)"
                    )
                    params.append(f"%{payer_name}%")
                description_expression = (
                    f"max(CAST({gold_bill_support.quote_ident(schema.rate_description_column)} AS VARCHAR))"
                    if schema.rate_description_column
                    else "NULL"
                )
                sql = (
                    f"SELECT median({allowed_expression}) AS negotiated_rate, "
                    f"median({selected_gross_expression}) AS gross_charge, "
                    f"{description_expression} AS description, "
                    "count(*) AS row_count "
                    f"FROM {TRILLIANT_ALIAS}.{gold_bill_support.quote_ident(schema.rate_table)} "
                    f"WHERE {' AND '.join(where_clauses)}"
                )
                row = gold_bill_support.fetch_rows_as_dicts(connection.execute(sql, params))
                if not row:
                    continue
                quote = sanitize_rate_quote(
                    {
                        "billing_code": code,
                        "negotiated_rate": row[0].get("negotiated_rate"),
                        "gross_charge": row[0].get("gross_charge"),
                        "description": gold_bill_support.normalize_text(row[0].get("description")),
                        "row_count": int(row[0].get("row_count") or 0),
                        "lookup_level": lookup_level,
                        "payer_name": payer_name,
                        "identity_filter": None,
                        "state_value": None,
                        "baseline_amount": None,
                    },
                    max_quote_amount=max_quote_amount,
                )
                if quote_has_any_amount(quote):
                    rows_out[_component_quote_key(code_system, code)] = quote
                    found_quote = True
                    break
            if found_quote:
                break
    return rows_out


def resolve_component_rate_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    session: requests.Session,
    occurrences: Sequence[dict[str, Any]],
    scenario: dict[str, Any],
    hospital: dict[str, Any],
    source_state_value: str | None,
    max_quote_amount: float,
    cache_lock: threading.Lock | None = None,
) -> dict[str, dict[str, Any]]:
    del source_state_value
    if not occurrences:
        return {}
    level_order = (
        ["hospital_payer", "hospital_average"]
        if scenario["scenario"] == "in_network"
        else ["hospital_average"]
    )
    remote_connection: Any | None = None
    remote_schema: TrilliantSchema = schema
    try:
        if schema.source_mode == "public_directory":
            remote_connection, remote_schema, _hospital_duckdb_url = (
                open_public_directory_hospital_database(
                    connection,
                    session,
                    hospital,
                    cache_lock=cache_lock,
                )
            )
        query_connection = remote_connection or connection
        quote_by_code: dict[str, dict[str, Any]] = {}
        for level in level_order:
            missing = [
                {
                    "code": gold_bill_support.normalize_code(occurrence.get("code")),
                    "code_system": normalize_code_system(occurrence.get("code_system")),
                }
                for occurrence in occurrences
                if gold_bill_support.normalize_code(occurrence.get("code")) is not None
                and normalize_code_system(occurrence.get("code_system")) is not None
                and _component_quote_key(
                    str(occurrence.get("code_system")),
                    str(occurrence.get("code")),
                )
                not in quote_by_code
            ]
            if not missing:
                break
            quote_by_code.update(
                query_public_directory_component_rate_quotes(
                    query_connection,
                    remote_schema,
                    occurrences=missing,
                    lookup_level=level,
                    payer_name=scenario.get("payer_name"),
                    scenario_name=scenario["scenario"],
                    max_quote_amount=max_quote_amount,
                )
            )
        return quote_by_code
    finally:
        if remote_connection is not None:
            remote_connection.close()


def query_trilliant_prior_quote(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_code: str,
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
) -> dict[str, Any] | None:
    quotes, _miss_cacheability, _hospitals_scanned = query_trilliant_prior_quotes_with_miss_cacheability(
        connection,
        schema,
        billing_codes=[billing_code],
        encounter_regime=encounter_regime,
        claim_class=claim_class,
        state_value=state_value,
        max_quote_amount=max_quote_amount,
        session=session,
        cache_lock=cache_lock,
        logger=logger,
        case_id=case_id,
    )
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    if normalized_billing_code is None:
        return None
    return quotes.get(normalized_billing_code)


def query_trilliant_prior_quote_with_miss_cacheability(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_code: str,
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    quotes, miss_cacheability, _hospitals_scanned = query_trilliant_prior_quotes_with_miss_cacheability(
        connection,
        schema,
        billing_codes=[billing_code],
        encounter_regime=encounter_regime,
        claim_class=claim_class,
        state_value=state_value,
        max_quote_amount=max_quote_amount,
        session=session,
        cache_lock=cache_lock,
        logger=logger,
        case_id=case_id,
    )
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    if normalized_billing_code is None:
        return None, False
    return (
        quotes.get(normalized_billing_code),
        miss_cacheability.get(normalized_billing_code, False),
    )


def resolve_trilliant_prior_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: Sequence[str],
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    public_directory_candidates: Sequence[dict[str, Any]] | None = None,
    allow_national_fallback: bool = True,
    hospital_cap_override: int | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
    scenario_name: str | None = None,
) -> dict[str, dict[str, Any]]:
    requested_codes = _normalize_prior_billing_codes(billing_codes)
    if not requested_codes:
        return {}

    started_at = time.monotonic()
    if logger is not None:
        logger.info(
            "Starting trilliant prior resolution",
            case_id=case_id,
            scenario=scenario_name,
            encounter_regime=encounter_regime,
            missing_professional_codes=requested_codes,
        )

    resolved_quotes: dict[str, dict[str, Any]] = {}
    unresolved_codes = list(requested_codes)
    cache_hits = 0
    accepted_state = 0
    accepted_national = 0
    hospitals_scanned_state = 0
    hospitals_scanned_national = 0
    public_candidates_by_scope: dict[str, list[dict[str, Any]]] = {}
    rate_table_name = gold_bill_support.normalize_text(getattr(schema, "rate_table", None))

    public_directory_missing_rate_table = (
        schema.source_mode == "public_directory"
        and bool(rate_table_name)
        and not relation_exists(connection, table_name=rate_table_name)
    )
    if public_directory_missing_rate_table and session is not None:
        if public_directory_candidates is not None:
            candidate_list = list(public_directory_candidates)
            if state_value:
                normalized_state_value = gold_bill_support.normalize_state_code(state_value)
                public_candidates_by_scope["state"] = [
                    candidate
                    for candidate in candidate_list
                    if gold_bill_support.normalize_state_code(candidate.get("hospital_state"))
                    == normalized_state_value
                ]
            public_candidates_by_scope["national"] = candidate_list
        else:
            search_rows = fetch_public_trilliant_search_index(
                connection,
                session,
                cache_lock=cache_lock,
            )
            max_candidates = max(1, len(search_rows))
            if state_value:
                public_candidates_by_scope["state"], _state_query_mode = filter_public_directory_candidates(
                    search_rows,
                    state_value=state_value,
                    max_candidates=max_candidates,
                )
            public_candidates_by_scope["national"], _national_query_mode = filter_public_directory_candidates(
                search_rows,
                state_value=None,
                max_candidates=max_candidates,
            )
    if (
        public_directory_missing_rate_table
        and allow_national_fallback
        and state_value
        and public_candidates_by_scope.get("state")
        and public_candidates_by_scope.get("national")
    ):
        state_ids = {
            gold_bill_support.normalize_text(candidate.get("hospital_directory_id"))
            or gold_bill_support.normalize_text(candidate.get("hospital_name"))
            for candidate in public_candidates_by_scope["state"]
        }
        national_extra_candidates = [
            candidate
            for candidate in public_candidates_by_scope["national"]
            if (
                gold_bill_support.normalize_text(candidate.get("hospital_directory_id"))
                or gold_bill_support.normalize_text(candidate.get("hospital_name"))
            )
            not in state_ids
        ]
        if not national_extra_candidates:
            allow_national_fallback = False

    def resolve_scope(
        scope: str,
        scope_state_value: str | None,
        candidate_codes: list[str],
    ) -> list[str]:
        nonlocal cache_hits
        nonlocal accepted_state
        nonlocal accepted_national
        nonlocal hospitals_scanned_state
        nonlocal hospitals_scanned_national
        if not candidate_codes:
            return []
        lookup_keys = {
            billing_code: build_trilliant_prior_lookup_key(
                billing_code=billing_code,
                encounter_regime=encounter_regime,
                claim_class=claim_class,
                state_value=scope_state_value,
                source_mode=schema.source_mode,
                max_quote_amount=max_quote_amount,
            )
            for billing_code in candidate_codes
        }
        cache_statuses = cached_trilliant_prior_quotes(
            connection,
            lookup_keys,
            max_quote_amount=max_quote_amount,
            cache_lock=cache_lock,
        )
        remaining_codes: list[str] = []
        remote_codes: list[str] = []
        for billing_code in candidate_codes:
            cache_status, cached_quote = cache_statuses.get(billing_code, ("missing", None))
            if cache_status == "quote" and cached_quote is not None:
                resolved_quotes[billing_code] = {**cached_quote, "prior_cache_status": "hit"}
                cache_hits += 1
                continue
            remaining_codes.append(billing_code)
            if cache_status != "miss":
                remote_codes.append(billing_code)
        if not remote_codes:
            return remaining_codes

        remote_quotes, miss_cacheability, hospitals_scanned = query_trilliant_prior_quotes_with_miss_cacheability(
            connection,
            schema,
            billing_codes=remote_codes,
            encounter_regime=encounter_regime,
            claim_class=claim_class,
            state_value=scope_state_value,
            max_quote_amount=max_quote_amount,
            session=session,
            cache_lock=cache_lock,
            public_directory_candidates=public_candidates_by_scope.get(scope),
            hospital_cap_override=hospital_cap_override,
            logger=logger,
            case_id=case_id,
            started_at=started_at,
        )
        if scope == "state":
            hospitals_scanned_state = hospitals_scanned
        else:
            hospitals_scanned_national = hospitals_scanned
        accepted_count = 0
        threshold = _trilliant_prior_scope_min_hospitals(scope)
        for billing_code in remote_codes:
            lookup_key = lookup_keys[billing_code]
            remote_quote = remote_quotes.get(billing_code)
            is_cacheable = miss_cacheability.get(billing_code, False)
            if remote_quote is not None:
                hospital_count = int(remote_quote.get("hospital_count") or 0)
                if hospital_count >= threshold:
                    cache_rate_quote(connection, lookup_key, remote_quote, cache_lock=cache_lock)
                    resolved_quotes[billing_code] = {**remote_quote, "prior_cache_status": "miss"}
                    accepted_count += 1
                    continue
                if is_cacheable:
                    cache_rate_quote(
                        connection,
                        lookup_key,
                        build_trilliant_prior_cache_miss(
                            reason=TRILLIANT_PRIOR_CACHE_MISS_REASON_INSUFFICIENT_SUPPORT,
                            state_value=scope_state_value,
                            hospital_count=hospital_count,
                            supporting_row_count=int(remote_quote.get("supporting_row_count") or 0),
                        ),
                        cache_lock=cache_lock,
                    )
                continue
            if is_cacheable:
                cache_rate_quote(
                    connection,
                    lookup_key,
                    build_trilliant_prior_cache_miss(
                        reason=TRILLIANT_PRIOR_CACHE_MISS_REASON_NO_RESULT,
                        state_value=scope_state_value,
                    ),
                    cache_lock=cache_lock,
                )
        if scope == "state":
            accepted_state += accepted_count
        else:
            accepted_national += accepted_count
        return [billing_code for billing_code in remaining_codes if billing_code not in resolved_quotes]

    can_filter_prior_by_state = bool(
        state_value and (schema.rate_state_column or public_directory_missing_rate_table)
    )
    if can_filter_prior_by_state:
        unresolved_codes = resolve_scope("state", state_value, unresolved_codes)
    if allow_national_fallback:
        unresolved_codes = resolve_scope("national", None, unresolved_codes)

    if logger is not None:
        logger.info(
            "Completed trilliant prior resolution",
            case_id=case_id,
            requested_codes=requested_codes,
            resolved_codes=[billing_code for billing_code in requested_codes if billing_code in resolved_quotes],
            cache_hits=cache_hits,
            accepted_state=accepted_state,
            accepted_national=accepted_national,
            misses=len([billing_code for billing_code in requested_codes if billing_code not in resolved_quotes]),
            hospitals_scanned_state=hospitals_scanned_state,
            hospitals_scanned_national=hospitals_scanned_national,
            elapsed=time.monotonic() - started_at,
        )
    return resolved_quotes


def resolve_bounded_state_trilliant_prior_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_codes: Sequence[str],
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    public_directory_candidates: Sequence[dict[str, Any]] | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
    scenario_name: str | None = None,
    hospital_cap: int = BOUNDED_STATE_PRIOR_HOSPITAL_CAP,
) -> dict[str, dict[str, Any]]:
    if state_value is None:
        return {}
    return resolve_trilliant_prior_quotes(
        connection,
        schema,
        billing_codes=billing_codes,
        encounter_regime=encounter_regime,
        claim_class=claim_class,
        state_value=state_value,
        max_quote_amount=max_quote_amount,
        session=session,
        cache_lock=cache_lock,
        public_directory_candidates=public_directory_candidates,
        allow_national_fallback=False,
        hospital_cap_override=hospital_cap,
        logger=logger,
        case_id=case_id,
        scenario_name=scenario_name,
    )


def resolve_trilliant_prior_quote(
    connection: Any,
    schema: TrilliantSchema,
    *,
    billing_code: str,
    encounter_regime: str,
    claim_class: str,
    state_value: str | None,
    max_quote_amount: float,
    session: requests.Session | None = None,
    cache_lock: threading.Lock | None = None,
    logger: CliLogger | None = None,
    case_id: str | None = None,
    scenario_name: str | None = None,
) -> dict[str, Any] | None:
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    if normalized_billing_code is None:
        return None
    quotes = resolve_trilliant_prior_quotes(
        connection,
        schema,
        billing_codes=[normalized_billing_code],
        encounter_regime=encounter_regime,
        claim_class=claim_class,
        state_value=state_value,
        max_quote_amount=max_quote_amount,
        session=session,
        cache_lock=cache_lock,
        logger=logger,
        case_id=case_id,
        scenario_name=scenario_name,
    )
    return quotes.get(normalized_billing_code)


def resolve_rate_quotes(
    connection: Any,
    schema: TrilliantSchema,
    *,
    session: requests.Session,
    billing_codes: list[str],
    scenario: dict[str, Any],
    hospital: dict[str, Any],
    encounter_regime: str,
    source_state_value: str | None,
    max_quote_amount: float,
    cache_lock: threading.Lock | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    identity_filter = choose_rate_identity_filter(hospital, schema)
    quote_by_code: dict[str, dict[str, Any]] = {}
    telemetry = {"cache_hits": 0, "cache_misses": 0}

    del source_state_value
    level_order = (
        ["hospital_payer", "hospital_average"]
        if scenario["scenario"] == "in_network"
        else ["hospital_average"]
    )

    remote_connection: Any | None = None
    remote_schema: TrilliantSchema = schema
    try:
        if schema.source_mode == "public_directory":
            remote_connection, remote_schema, _hospital_duckdb_url = (
                open_public_directory_hospital_database(
                    connection,
                    session,
                    hospital,
                    cache_lock=cache_lock,
                )
            )

        query_connection = remote_connection or connection
        for level in level_order:
            missing_codes = [code for code in billing_codes if code not in quote_by_code]
            if not missing_codes:
                break
            lookup_keys = [
                build_rate_lookup_key(
                    billing_code=code,
                    lookup_level=level,
                    payer_name=scenario.get("payer_name"),
                    identity_filter=identity_filter,
                    state_value=None,
                    max_quote_amount=max_quote_amount,
                )
                for code in missing_codes
            ]
            cached = cached_rate_quotes(
                connection,
                lookup_keys,
                cache_lock=cache_lock,
            )
            for code, lookup_key in zip(missing_codes, lookup_keys, strict=False):
                if lookup_key in cached:
                    quote = sanitize_rate_quote(
                        cached[lookup_key],
                        max_quote_amount=max_quote_amount,
                    )
                    if quote_supports_scenario(quote, scenario_name=scenario["scenario"]):
                        quote_by_code[code] = quote
                        telemetry["cache_hits"] += 1
            remaining_codes = [code for code in missing_codes if code not in quote_by_code]
            if not remaining_codes:
                continue
            remote_results = query_rate_level(
                query_connection,
                remote_schema,
                billing_codes=remaining_codes,
                lookup_level=level,
                payer_name=scenario.get("payer_name"),
                identity_filter=identity_filter,
                state_value=None,
                encounter_regime=encounter_regime,
                scenario_name=scenario["scenario"],
                max_quote_amount=max_quote_amount,
            )
            for code in remaining_codes:
                lookup_key = build_rate_lookup_key(
                    billing_code=code,
                    lookup_level=level,
                    payer_name=scenario.get("payer_name"),
                    identity_filter=identity_filter,
                    state_value=None,
                    max_quote_amount=max_quote_amount,
                )
                if code in remote_results:
                    quote = sanitize_rate_quote(
                        remote_results[code],
                        max_quote_amount=max_quote_amount,
                    )
                    if quote_has_any_amount(quote):
                        cache_rate_quote(
                            connection,
                            lookup_key,
                            quote,
                            cache_lock=cache_lock,
                        )
                        if quote_supports_scenario(quote, scenario_name=scenario["scenario"]):
                            quote_by_code[code] = quote
                telemetry["cache_misses"] += 1
    finally:
        if remote_connection is not None:
            remote_connection.close()

    return quote_by_code, telemetry


def select_public_directory_hospital_with_rate_coverage(
    connection: Any,
    schema: TrilliantSchema,
    *,
    session: requests.Session,
    case_id: str,
    seed: int,
    candidates: list[dict[str, Any]],
    billing_codes: list[str],
    component_occurrences: Sequence[dict[str, Any]],
    scenario: dict[str, Any],
    encounter_regime: str = "outpatient",
    zip_codes: Sequence[str] = (),
    county_fips: str | None = None,
    local_hud_mapping: dict[str, list[str]] | None = None,
    primary_clinical_direction: str | None = None,
    source_state_value: str | None = None,
    max_hospital_rate_probes: int = 1,
    max_quote_amount: float = 1_000_000.0,
    cache_lock: threading.Lock | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    if not candidates:
        raise RuntimeError("No hospital candidates were available for this encounter.")
    reverse_zip_to_counties = reverse_zip_to_county_mapping(local_hud_mapping)
    pool_specs: list[tuple[str, bool]] = [
        ("zip", True),
        ("county", True),
        ("same_state", True),
    ]
    same_state_unfiltered_fallback_invoked = False
    target_drg_quote_count = sum(
        1
        for occurrence in component_occurrences
        if normalize_code_system(occurrence.get("code_system"))
        in INPATIENT_DRG_FAMILY_CODE_SYSTEMS
    )
    target_other_anchor_quote_count = sum(
        1
        for occurrence in component_occurrences
        if normalize_code_system(occurrence.get("code_system"))
        in INPATIENT_OTHER_ANCHOR_CODE_SYSTEMS
    )
    target_service_quote_count = len(billing_codes)

    def quote_score(
        service_quotes: dict[str, dict[str, Any]],
        component_quotes: dict[str, dict[str, Any]],
    ) -> tuple[int, int, int, int]:
        drg_quote_count = sum(
            1
            for occurrence in component_occurrences
            if normalize_code_system(occurrence.get("code_system"))
            in INPATIENT_DRG_FAMILY_CODE_SYSTEMS
            and _component_quote_key(
                str(occurrence.get("code_system")),
                str(occurrence.get("code")),
            )
            in component_quotes
        )
        other_anchor_quote_count = sum(
            1
            for occurrence in component_occurrences
            if normalize_code_system(occurrence.get("code_system"))
            in INPATIENT_OTHER_ANCHOR_CODE_SYSTEMS
            and _component_quote_key(
                str(occurrence.get("code_system")),
                str(occurrence.get("code")),
            )
            in component_quotes
        )
        service_quote_count = sum(1 for code in billing_codes if code in service_quotes)
        total_quote_count = drg_quote_count + other_anchor_quote_count + service_quote_count
        return (
            drg_quote_count,
            other_anchor_quote_count,
            service_quote_count,
            total_quote_count,
        )

    def has_full_quote_coverage(score: tuple[int, int, int, int]) -> bool:
        return (
            score[0] >= target_drg_quote_count
            and score[1] >= target_other_anchor_quote_count
            and score[2] >= target_service_quote_count
        )

    def probe_pool(
        pool_candidates: Sequence[dict[str, Any]],
    ) -> tuple[
        dict[str, Any] | None,
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, int],
        tuple[int, int, int, int],
        int,
        int,
        int,
    ]:
        ordered_candidates = ordered_hospital_candidates_for_case(
            case_id,
            seed,
            list(pool_candidates),
        )
        if not ordered_candidates:
            return (
                None,
                {},
                {},
                {"cache_hits": 0, "cache_misses": 0},
                (-1, -1, -1, -1),
                0,
                0,
                0,
            )
        probe_limit = max(1, min(len(ordered_candidates), max_hospital_rate_probes))
        total_probe_limit = probe_limit
        best_candidate: dict[str, Any] | None = None
        best_service_quotes: dict[str, dict[str, Any]] = {}
        best_component_quotes: dict[str, dict[str, Any]] = {}
        best_telemetry = {"cache_hits": 0, "cache_misses": 0}
        best_score = (-1, -1, -1, -1)
        best_probe_index = 0
        actual_probe_count = 0

        def probe_candidates(start_index: int, stop_index: int) -> bool:
            nonlocal actual_probe_count
            nonlocal best_candidate
            nonlocal best_service_quotes
            nonlocal best_component_quotes
            nonlocal best_telemetry
            nonlocal best_score
            nonlocal best_probe_index
            for probe_index, candidate in enumerate(
                ordered_candidates[start_index:stop_index],
                start=start_index + 1,
            ):
                actual_probe_count = probe_index
                service_quotes, telemetry = resolve_rate_quotes(
                    connection,
                    schema,
                    session=session,
                    billing_codes=billing_codes,
                    scenario=scenario,
                    hospital=candidate,
                    encounter_regime=encounter_regime,
                    source_state_value=source_state_value,
                    max_quote_amount=max_quote_amount,
                    cache_lock=cache_lock,
                )
                component_quotes = resolve_component_rate_quotes(
                    connection,
                    schema,
                    session=session,
                    occurrences=component_occurrences,
                    scenario=scenario,
                    hospital=candidate,
                    source_state_value=source_state_value,
                    max_quote_amount=max_quote_amount,
                    cache_lock=cache_lock,
                )
                score = quote_score(service_quotes, component_quotes)
                if score > best_score:
                    best_candidate = candidate
                    best_service_quotes = service_quotes
                    best_component_quotes = component_quotes
                    best_telemetry = telemetry
                    best_score = score
                    best_probe_index = probe_index
                if has_full_quote_coverage(score):
                    return True
            return False

        phase_one_complete = probe_candidates(0, probe_limit)
        if (
            not phase_one_complete
            and component_occurrences
            and best_score[0] + best_score[1] == 0
            and probe_limit < len(ordered_candidates)
        ):
            total_probe_limit = max(probe_limit, min(len(ordered_candidates), 12))
            probe_candidates(probe_limit, total_probe_limit)
        return (
            best_candidate,
            best_service_quotes,
            best_component_quotes,
            best_telemetry,
            best_score,
            best_probe_index,
            actual_probe_count,
            total_probe_limit if total_probe_limit else probe_limit,
        )

    fallback_result: tuple[
        dict[str, Any] | None,
        dict[str, dict[str, Any]],
        dict[str, dict[str, Any]],
        dict[str, int],
        tuple[int, int, int, int],
        int,
        int,
        int,
        str | None,
        bool,
    ] | None = None

    for stage, specialty_filter_active in pool_specs:
        pool_candidates = public_directory_candidate_pool(
            candidates,
            stage=stage,
            zip_codes=zip_codes,
            county_fips=county_fips,
            state_value=source_state_value,
            reverse_zip_to_counties=reverse_zip_to_counties,
            primary_clinical_direction=primary_clinical_direction,
            specialty_filter_active=specialty_filter_active,
        )
        if not pool_candidates and specialty_filter_active:
            if stage == "same_state":
                same_state_unfiltered_fallback_invoked = True
            pool_candidates = public_directory_candidate_pool(
                candidates,
                stage=stage,
                zip_codes=zip_codes,
                county_fips=county_fips,
                state_value=source_state_value,
                reverse_zip_to_counties=reverse_zip_to_counties,
                primary_clinical_direction=primary_clinical_direction,
                specialty_filter_active=False,
            )
            specialty_filter_active = False
        if not pool_candidates:
            continue
        probed = probe_pool(pool_candidates)
        (
            best_candidate,
            best_service_quotes,
            best_component_quotes,
            best_telemetry,
            best_score,
            best_probe_index,
            actual_probe_count,
            total_probe_limit,
        ) = probed
        if best_candidate is None:
            continue
        result = (
            best_candidate,
            best_service_quotes,
            best_component_quotes,
            best_telemetry,
            best_score,
            best_probe_index,
            actual_probe_count,
            total_probe_limit,
            stage,
            specialty_filter_active,
        )
        if fallback_result is None or best_score > fallback_result[4]:
            fallback_result = result
        if has_full_quote_coverage(best_score) or stage == "same_state":
            break

    if fallback_result is None:
        raise RuntimeError("No viable hospital remained after selection filtering.")

    (
        best_candidate,
        best_service_quotes,
        best_component_quotes,
        best_telemetry,
        best_score,
        best_probe_index,
        actual_probe_count,
        total_probe_limit,
        geographic_stage,
        specialty_filter_active,
    ) = fallback_result
    selected_candidate = dict(best_candidate)
    selected_candidate["selection_strategy"] = "rate_coverage"
    selected_candidate["quoted_code_coverage"] = max(best_score[3], 0)
    selected_candidate["selection_probe_count"] = actual_probe_count
    selected_candidate["selection_probe_limit"] = total_probe_limit
    selected_candidate["selection_probe_rank"] = best_probe_index or 1
    selected_candidate["selection_stage_candidate_count"] = len(
        public_directory_candidate_pool(
            candidates,
            stage=geographic_stage or "same_state",
            zip_codes=zip_codes,
            county_fips=county_fips,
            state_value=source_state_value,
            reverse_zip_to_counties=reverse_zip_to_counties,
            primary_clinical_direction=primary_clinical_direction,
            specialty_filter_active=bool(specialty_filter_active),
        )
    )
    selected_candidate["drg_quote_count"] = max(best_score[0], 0)
    selected_candidate["other_anchor_quote_count"] = max(best_score[1], 0)
    selected_candidate["service_quote_count"] = max(best_score[2], 0)
    selected_candidate["total_quote_count"] = max(best_score[3], 0)
    selected_candidate["selection_score"] = list(best_score)
    selected_candidate["geographic_stage"] = geographic_stage
    selected_candidate["specialty_filter_active"] = specialty_filter_active
    selected_candidate["same_state_unfiltered_fallback_invoked"] = (
        same_state_unfiltered_fallback_invoked
    )
    can_preload_exact_quotes = bool(billing_codes)
    if schema.source_mode == "public_directory":
        can_preload_exact_quotes = can_preload_exact_quotes and bool(
            gold_bill_support.normalize_text(selected_candidate.get("duckdb_url"))
            or (
                gold_bill_support.normalize_text(selected_candidate.get("hospital_directory_id"))
                and gold_bill_support.normalize_text(selected_candidate.get("hospital_page_url"))
            )
        )
    selected_candidate["_professional_exact_quote_by_code"] = (
        resolve_selected_hospital_exact_professional_quotes(
            connection,
            schema,
            session=session,
            billing_codes=billing_codes,
            hospital=selected_candidate,
            encounter_regime=encounter_regime,
            max_quote_amount=max_quote_amount,
            cache_lock=cache_lock,
        )
        if can_preload_exact_quotes
        else {}
    )
    return selected_candidate, best_service_quotes, best_component_quotes, best_telemetry
