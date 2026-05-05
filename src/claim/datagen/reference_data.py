from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence
import zipfile
from xml.etree import ElementTree as ET

import requests

from claim.datagen import constants
from claim.datagen.contracts import ReferenceDataPaths, ReferenceFileSpec
from claim.datagen.io import (
    build_session,
    close_progress_bar,
    create_progress_bar,
    load_json_file,
    write_json_atomic,
)
from claim.datagen.utils import normalize_county_code
from claim.cli_logging import CliLogger
from claim.gold_bill_support import (
    ZIP_RE,
    normalize_state_code,
    normalize_text,
    normalize_zip,
    pick_best_column,
    unique_preserving_order,
)

WARNED_HUD_MISSING_TOKEN_COUNTIES: set[str] = set()


def download_file_if_missing(
    session: requests.Session,
    spec: ReferenceFileSpec,
    target_dir: Path,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / spec.filename
    if path.exists():
        logger.info("Skipping existing reference file", file=path.name)
        return path

    temp_path = path.with_suffix(f"{path.suffix}.part")
    logger.info("Downloading reference file", file=path.name, url=spec.url)
    response = session.get(
        spec.url,
        timeout=constants.DEFAULT_REQUEST_TIMEOUT_SECONDS,
        stream=True,
    )
    try:
        response.raise_for_status()
        total_bytes = response.headers.get("Content-Length")
        progress = create_progress_bar(
            description=f"download {path.name}",
            total=int(total_bytes) if total_bytes and total_bytes.isdigit() else None,
            unit="B",
            show_progress=show_progress,
        )
        try:
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
        finally:
            close_progress_bar(progress)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        response.close()

    temp_path.replace(path)
    logger.success("Downloaded reference file", file=path.name)
    return path


def ensure_reference_data(
    reference_data_dir: Path,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> ReferenceDataPaths:
    reference_data_dir.mkdir(parents=True, exist_ok=True)
    with build_session() as session:
        nber_path = download_file_if_missing(
            session,
            constants.REFERENCE_NBER_SSA_FIPS_SPEC,
            reference_data_dir / "nber",
            logger,
            show_progress=show_progress,
        )
    return ReferenceDataPaths(ssa_fips_csv=nber_path)


def detect_csv_column(headers: Sequence[str], aliases: Sequence[str]) -> str:
    column = pick_best_column(headers, aliases)
    if column is None:
        raise ValueError(
            f"Could not locate a matching CSV column for aliases {aliases!r} in {list(headers)!r}"
        )
    return column


def load_ssa_fips_crosswalk(csv_path: Path) -> dict[tuple[str, str], dict[str, str | None]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {csv_path}")
        headers = list(reader.fieldnames)
        normalized_headers = {
            header.lower().replace("_", ""): header
            for header in headers
        }
        combined_ssa_column = normalized_headers.get("ssacode")
        combined_fips_county_column = (
            normalized_headers.get("fipscounty")
            or normalized_headers.get("countyfips")
            or normalized_headers.get("countyfp")
        )
        state_abbrev_column = pick_best_column(
            headers,
            ["state", "state_abbrev", "stabbrev", "stusps"],
        )
        county_name_rate_column = pick_best_column(
            headers,
            ["countyname_rate", "county_rate", "county_name_rate"],
        )
        county_name_fips_column = pick_best_column(
            headers,
            ["countyname_fips", "county_name_fips", "county_name", "county"],
        )
        state_name_column = pick_best_column(
            headers,
            ["state_name", "state_full_name"],
        )

        if combined_ssa_column and combined_fips_county_column:
            grouped_rows: dict[tuple[str, str], list[dict[str, str | None]]] = {}
            for row in reader:
                ssa_code = normalize_county_code(row.get(combined_ssa_column))
                county_fips = normalize_county_code(row.get(combined_fips_county_column))
                if ssa_code is None or county_fips is None or len(ssa_code) < 5:
                    continue
                key = (ssa_code[:2], ssa_code[-3:])
                grouped_rows.setdefault(key, []).append(
                    {
                        "state_fips": county_fips[:2],
                        "county_fips": county_fips[:5],
                        "county_name": normalize_text(row.get(county_name_fips_column))
                        if county_name_fips_column
                        else None,
                        "county_rate_name": normalize_text(row.get(county_name_rate_column))
                        if county_name_rate_column
                        else None,
                        "state_name": normalize_text(row.get(state_name_column))
                        if state_name_column
                        else None,
                        "state_abbrev": normalize_text(row.get(state_abbrev_column))
                        if state_abbrev_column
                        else None,
                    }
                )

            mapping: dict[tuple[str, str], dict[str, Any]] = {}
            for key, candidates in grouped_rows.items():
                preferred = next(
                    (
                        candidate
                        for candidate in candidates
                        if candidate.get("county_name")
                        and candidate.get("county_rate_name")
                        and candidate["county_name"] == candidate["county_rate_name"]
                    ),
                    candidates[0],
                )
                county_fips_candidates = unique_preserving_order(
                    [
                        str(candidate["county_fips"])
                        for candidate in candidates
                        if candidate.get("county_fips")
                    ]
                )
                mapping[key] = {
                    "state_fips": preferred.get("state_fips"),
                    "county_fips": preferred.get("county_fips"),
                    "county_name": preferred.get("county_rate_name")
                    or preferred.get("county_name"),
                    "state_name": preferred.get("state_name"),
                    "state_abbrev": preferred.get("state_abbrev"),
                    "county_fips_candidates": county_fips_candidates,
                }
            return mapping

        ssa_state_column = detect_csv_column(
            headers,
            ["ssa_state", "state_ssa", "ssastate", "ssa_st", "state"],
        )
        ssa_county_column = detect_csv_column(
            headers,
            ["ssa_county", "county_ssa", "ssacounty", "ssa_cnty", "county"],
        )
        fips_state_column = detect_csv_column(
            headers,
            ["fips_state", "state_fips", "fipstate", "statefp"],
        )
        fips_county_column = detect_csv_column(
            headers,
            ["fips_county", "county_fips", "fipscty", "countyfp"],
        )
        county_name_column = pick_best_column(
            headers,
            ["county_name", "county", "countytitle", "county_title"],
        )
        state_name_column = pick_best_column(
            headers,
            ["state_name", "state_abbrev", "state", "stabbrev", "stusps"],
        )

        mapping: dict[tuple[str, str], dict[str, str | None]] = {}
        for row in reader:
            ssa_state = normalize_state_code(row.get(ssa_state_column))
            ssa_county = normalize_county_code(row.get(ssa_county_column))
            fips_state = normalize_state_code(row.get(fips_state_column))
            county_fips_component = normalize_county_code(row.get(fips_county_column))
            if (
                ssa_state is None
                or ssa_county is None
                or fips_state is None
                or county_fips_component is None
            ):
                continue
            county_fips = f"{fips_state}{county_fips_component[-3:]}"
            mapping[(ssa_state, ssa_county[-3:])] = {
                "state_fips": fips_state,
                "county_fips": county_fips,
                "county_name": normalize_text(row.get(county_name_column))
                if county_name_column
                else None,
                "state_name": normalize_text(row.get(state_name_column))
                if state_name_column
                else None,
                "state_abbrev": normalize_text(row.get(state_name_column))
                if state_name_column
                else None,
            }
    return mapping


def load_hud_crosswalk_csv(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header in {path}")
        headers = list(reader.fieldnames)
        county_column = detect_csv_column(
            headers,
            ["county", "county_fips", "geoid", "county_geoid"],
        )
        zip_column = detect_csv_column(
            headers,
            ["zip", "zip_code", "zipcode", "postal_code"],
        )

        mapping: dict[str, list[str]] = {}
        for row in reader:
            county_value = normalize_text(row.get(county_column))
            zip_value = normalize_zip(row.get(zip_column))
            if county_value is None or zip_value is None:
                continue
            county_digits = "".join(character for character in county_value if character.isdigit())
            if len(county_digits) == 5:
                county_key = county_digits
            elif len(county_digits) == 3:
                county_key = county_digits.zfill(5)
            else:
                continue
            mapping.setdefault(county_key, []).append(zip_value)
        return {key: unique_preserving_order(values) for key, values in mapping.items()}


def excel_column_index(cell_reference: str) -> int:
    letters = []
    for character in cell_reference:
        if character.isalpha():
            letters.append(character.upper())
        else:
            break
    if not letters:
        return 0
    index = 0
    for character in letters:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return max(0, index - 1)


def read_xlsx_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in archive.namelist():
        return []
    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    shared_strings: list[str] = []
    for item in root.findall(f"{{{constants.EXCEL_MAIN_NS}}}si"):
        text_fragments = [
            node.text or ""
            for node in item.iterfind(f".//{{{constants.EXCEL_MAIN_NS}}}t")
        ]
        shared_strings.append("".join(text_fragments))
    return shared_strings


def read_xlsx_cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.iterfind(f".//{{{constants.EXCEL_MAIN_NS}}}t")
        )
    value_node = cell.find(f"{{{constants.EXCEL_MAIN_NS}}}v")
    if value_node is None:
        return ""
    raw_value = value_node.text or ""
    if cell_type == "s":
        try:
            return shared_strings[int(raw_value)]
        except (IndexError, ValueError):
            return raw_value
    return raw_value


def read_xlsx_sheet_rows(
    archive: zipfile.ZipFile,
    sheet_path: str,
    shared_strings: list[str],
) -> list[list[str]]:
    root = ET.fromstring(archive.read(sheet_path))
    rows: list[list[str]] = []
    for row_node in root.iterfind(f".//{{{constants.EXCEL_MAIN_NS}}}row"):
        values: list[str] = []
        for cell in row_node.findall(f"{{{constants.EXCEL_MAIN_NS}}}c"):
            cell_reference = cell.attrib.get("r", "")
            column_index = excel_column_index(cell_reference)
            while len(values) <= column_index:
                values.append("")
            values[column_index] = read_xlsx_cell_value(cell, shared_strings)
        while values and values[-1] == "":
            values.pop()
        rows.append(values)
    return rows


def candidate_sheet_paths_from_workbook(archive: zipfile.ZipFile) -> list[tuple[str, str]]:
    workbook_root = ET.fromstring(archive.read("xl/workbook.xml"))
    rels_root = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    rel_map = {
        rel.attrib["Id"]: rel.attrib["Target"]
        for rel in rels_root.findall(f"{{{constants.EXCEL_PACKAGE_REL_NS}}}Relationship")
    }
    sheet_paths: list[tuple[str, str]] = []
    for sheet in workbook_root.iterfind(f".//{{{constants.EXCEL_MAIN_NS}}}sheet"):
        name = sheet.attrib.get("name", "")
        rel_id = sheet.attrib.get(f"{{{constants.EXCEL_REL_NS}}}id")
        if not rel_id or rel_id not in rel_map:
            continue
        target = rel_map[rel_id]
        normalized_target = target if target.startswith("xl/") else f"xl/{target.lstrip('/')}"
        sheet_paths.append((name, normalized_target))
    return sheet_paths


def load_hud_crosswalk_xlsx(path: Path) -> dict[str, list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_strings = read_xlsx_shared_strings(archive)
        selected_rows: list[list[str]] | None = None
        for _, sheet_path in candidate_sheet_paths_from_workbook(archive):
            rows = read_xlsx_sheet_rows(archive, sheet_path, shared_strings)
            if not rows:
                continue
            headers = [normalize_text(value) or "" for value in rows[0]]
            if pick_best_column(
                headers,
                ["county", "county_fips", "geoid", "county_geoid"],
            ) and pick_best_column(
                headers,
                ["zip", "zip_code", "zipcode", "postal_code"],
            ):
                selected_rows = rows
                break
        if selected_rows is None:
            raise ValueError(f"Could not find a worksheet with ZIP and COUNTY columns in {path}")

    headers = [normalize_text(value) or "" for value in selected_rows[0]]
    county_column = detect_csv_column(
        headers,
        ["county", "county_fips", "geoid", "county_geoid"],
    )
    zip_column = detect_csv_column(
        headers,
        ["zip", "zip_code", "zipcode", "postal_code"],
    )
    county_index = headers.index(county_column)
    zip_index = headers.index(zip_column)

    mapping: dict[str, list[str]] = {}
    for row in selected_rows[1:]:
        county_value = normalize_text(row[county_index] if county_index < len(row) else None)
        zip_value = normalize_zip(row[zip_index] if zip_index < len(row) else None)
        if county_value is None or zip_value is None:
            continue
        county_digits = "".join(character for character in county_value if character.isdigit())
        if len(county_digits) == 5:
            county_key = county_digits
        elif len(county_digits) == 3:
            county_key = county_digits.zfill(5)
        else:
            continue
        mapping.setdefault(county_key, []).append(zip_value)
    return {key: unique_preserving_order(values) for key, values in mapping.items()}


def load_hud_crosswalk(path: Path) -> dict[str, list[str]]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_hud_crosswalk_csv(path)
    if suffix == ".xlsx":
        return load_hud_crosswalk_xlsx(path)
    raise ValueError(
        f"Unsupported HUD crosswalk format for {path}. Expected .csv or .xlsx."
    )


def collect_zip_codes(value: Any, sink: set[str], *, allow_string_scan: bool = False) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if "zip" in key.lower():
                if isinstance(child, str):
                    for match in ZIP_RE.findall(child):
                        sink.add(match)
                elif isinstance(child, list):
                    for item in child:
                        collect_zip_codes(item, sink, allow_string_scan=True)
                elif isinstance(child, dict):
                    collect_zip_codes(child, sink, allow_string_scan=True)
            else:
                collect_zip_codes(child, sink, allow_string_scan=False)
        return
    if isinstance(value, list):
        for item in value:
            collect_zip_codes(item, sink, allow_string_scan=allow_string_scan)
        return
    if allow_string_scan and isinstance(value, str):
        for match in ZIP_RE.findall(value):
            sink.add(match)


def extract_zip_codes_from_hud_response(payload: Any) -> list[str]:
    zip_codes: set[str] = set()
    collect_zip_codes(payload, zip_codes)
    return sorted(zip_codes)


def cached_hud_response_path(reference_data_dir: Path, county_fips: str) -> Path:
    return reference_data_dir / "hud-usps" / f"{county_fips}.json"


def fetch_hud_zip_codes_for_county(
    session: requests.Session,
    reference_data_dir: Path,
    county_fips: str,
    token: str | None,
    logger: CliLogger,
) -> list[str]:
    cache_path = cached_hud_response_path(reference_data_dir, county_fips)
    if cache_path.exists():
        payload = load_json_file(cache_path)
        return extract_zip_codes_from_hud_response(payload)

    if not token:
        if county_fips not in WARNED_HUD_MISSING_TOKEN_COUNTIES:
            logger.warning(
                "HUD ZIP lookup unavailable; no local crosswalk path or HUD API token was provided",
                county_fips=county_fips,
            )
            WARNED_HUD_MISSING_TOKEN_COUNTIES.add(county_fips)
        return []

    logger.info("Fetching HUD county ZIP mapping", county_fips=county_fips)
    response = session.get(
        constants.HUD_USPS_API_URL,
        params={"type": 7, "query": county_fips},
        headers={"Authorization": f"Bearer {token}"},
        timeout=constants.DEFAULT_REQUEST_TIMEOUT_SECONDS,
    )
    try:
        response.raise_for_status()
        payload = response.json()
    finally:
        response.close()

    write_json_atomic(cache_path, payload)
    zip_codes = extract_zip_codes_from_hud_response(payload)
    logger.success(
        "Fetched HUD county ZIP mapping",
        county_fips=county_fips,
        zip_codes=len(zip_codes),
    )
    return zip_codes


def build_geography_context(
    encounter: dict[str, Any],
    ssa_fips_mapping: dict[tuple[str, str], dict[str, str | None]],
    *,
    local_hud_mapping: dict[str, list[str]] | None,
    reference_data_dir: Path,
    session: requests.Session,
    hud_user_token: str | None,
    logger: CliLogger,
) -> dict[str, Any]:
    beneficiary = encounter.get("beneficiary") or {}
    ssa_state_code = normalize_state_code(beneficiary.get("state_code"))
    ssa_county_code = normalize_county_code(beneficiary.get("county_code"))
    geography = {
        "ssa_state_code": ssa_state_code,
        "ssa_county_code": ssa_county_code[-3:] if ssa_county_code else None,
        "state_fips": None,
        "county_fips": None,
        "county_name": None,
        "state_abbrev": None,
        "state_name": None,
        "zip_codes": [],
    }
    if ssa_state_code and ssa_county_code:
        match = ssa_fips_mapping.get((ssa_state_code, ssa_county_code[-3:]))
        if match:
            geography.update(match)

    county_fips = geography.get("county_fips")
    if county_fips and local_hud_mapping and county_fips in local_hud_mapping:
        geography["zip_codes"] = local_hud_mapping[county_fips]
    elif county_fips:
        geography["zip_codes"] = fetch_hud_zip_codes_for_county(
            session,
            reference_data_dir,
            county_fips,
            hud_user_token,
            logger,
        )

    return geography
