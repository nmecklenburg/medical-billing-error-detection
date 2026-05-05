from __future__ import annotations

from datetime import date, timedelta
import json
from typing import Any, Iterable

from claim.datagen.encounters.parsers import format_date


def normalize_date_range(
    start: date | None,
    end: date | None,
) -> tuple[date | None, date | None]:
    if start is None and end is not None:
        return end, end
    if start is not None and end is None:
        return start, start
    return start, end


def date_ranges_equal_with_slack(
    left_from: date | None,
    left_thru: date | None,
    right_from: date | None,
    right_thru: date | None,
    slack_days: int,
) -> bool:
    left_start, left_end = normalize_date_range(left_from, left_thru)
    right_start, right_end = normalize_date_range(right_from, right_thru)
    if left_start is None or right_start is None or left_end is None or right_end is None:
        return False
    slack = max(0, slack_days)
    return (
        abs((left_start - right_start).days) <= slack
        and abs((left_end - right_end).days) <= slack
    )


def date_range_contains_with_slack(
    outer_from: date | None,
    outer_thru: date | None,
    inner_from: date | None,
    inner_thru: date | None,
    slack_days: int,
) -> bool:
    outer_start, outer_end = normalize_date_range(outer_from, outer_thru)
    inner_start, inner_end = normalize_date_range(inner_from, inner_thru)
    if outer_start is None or inner_start is None or outer_end is None or inner_end is None:
        return False
    delta = timedelta(days=max(0, slack_days))
    return outer_start - delta <= inner_start and inner_end <= outer_end + delta


def carrier_claim_matches_anchor(
    anchor: dict[str, Any],
    claim_from: date | None,
    claim_thru: date | None,
    slack_days: int,
) -> bool:
    match_from = anchor.get("carrier_match_from_date")
    match_thru = anchor.get("carrier_match_thru_date")
    if anchor.get("anchor_claim_type") == "outpatient":
        return date_ranges_equal_with_slack(
            match_from,
            match_thru,
            claim_from,
            claim_thru,
            slack_days,
        )
    return date_range_contains_with_slack(
        match_from,
        match_thru,
        claim_from,
        claim_thru,
        slack_days,
    )


def min_non_null_date(*values: date | None) -> date | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return min(candidates)


def max_non_null_date(*values: date | None) -> date | None:
    candidates = [value for value in values if value is not None]
    if not candidates:
        return None
    return max(candidates)


def append_unique_dicts(
    target: list[dict[str, Any]],
    additions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {json.dumps(item, sort_keys=True) for item in target}
    for item in additions:
        serialized = json.dumps(item, sort_keys=True)
        if serialized in seen:
            continue
        seen.add(serialized)
        target.append(item)
    return target


def merge_facility_anchors(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    if existing["cluster_id"] != incoming["cluster_id"]:
        raise ValueError("Cannot merge facility anchors with different cluster IDs")

    merged = dict(existing)
    merged["anchor_from_date"] = min_non_null_date(
        existing.get("anchor_from_date"),
        incoming.get("anchor_from_date"),
    )
    merged["anchor_thru_date"] = max_non_null_date(
        existing.get("anchor_thru_date"),
        incoming.get("anchor_thru_date"),
    )
    merged["carrier_match_from_date"] = min_non_null_date(
        existing.get("carrier_match_from_date"),
        incoming.get("carrier_match_from_date"),
    )
    merged["carrier_match_thru_date"] = max_non_null_date(
        existing.get("carrier_match_thru_date"),
        incoming.get("carrier_match_thru_date"),
    )
    anchor_date = (
        merged.get("carrier_match_from_date")
        or merged.get("anchor_from_date")
        or merged.get("anchor_thru_date")
    )
    merged["anchor_year"] = anchor_date.year if anchor_date is not None else merged.get(
        "anchor_year"
    )
    merged["claims"] = append_unique_dicts(list(existing["claims"]), incoming["claims"])
    merged["diagnoses"] = append_unique_dicts(list(existing["diagnoses"]), incoming["diagnoses"])
    merged["procedure_codes"] = append_unique_dicts(
        list(existing["procedure_codes"]),
        incoming["procedure_codes"],
    )
    return merged


def unique_code_cluster(
    occurrences: Iterable[dict[str, Any]],
    *,
    allowed_systems: set[str],
) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for occurrence in occurrences:
        code_system = str(occurrence.get("code_system", ""))
        code = str(occurrence.get("code", ""))
        if code_system not in allowed_systems or not code:
            continue
        key = (code_system, code)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            {
                "code_system": code_system,
                "code": code,
                "description": None,
            }
        )
    return unique


def finalize_cluster_record(
    anchor: dict[str, Any],
    sample_number: int,
    carrier_match_window_days: int,
) -> dict[str, Any]:
    diagnosis_cluster = unique_code_cluster(anchor["diagnoses"], allowed_systems={"ICD9"})
    hcpcs_cluster = unique_code_cluster(
        anchor["procedure_codes"],
        allowed_systems={"HCPCS/CPT"},
    )
    icd9_procedure_cluster = unique_code_cluster(
        anchor["procedure_codes"],
        allowed_systems={"ICD9_PROCEDURE"},
    )
    claim_types_present = sorted(
        {
            str(claim.get("claim_type"))
            for claim in anchor["claims"]
            if claim.get("claim_type") is not None
        }
    )
    carrier_line_item_count = sum(
        len(claim.get("line_items", []))
        for claim in anchor["claims"]
        if claim.get("claim_type") == "carrier"
    )

    return {
        "cluster_id": anchor["cluster_id"],
        "sample_number": sample_number,
        "desynpuf_id": anchor["desynpuf_id"],
        "encounter": {
            "anchor_claim_type": anchor["anchor_claim_type"],
            "anchor_claim_id": anchor["anchor_claim_id"],
            "from_date": format_date(anchor["anchor_from_date"]),
            "thru_date": format_date(anchor["anchor_thru_date"]),
            "carrier_match_window_days": carrier_match_window_days,
        },
        "beneficiary": anchor.get("beneficiary"),
        "code_cluster": {
            "diagnosis_codes": diagnosis_cluster,
            "hcpcs_cpt_codes": hcpcs_cluster,
            "icd9_procedure_codes": icd9_procedure_cluster,
        },
        "diagnoses": anchor["diagnoses"],
        "procedure_codes": anchor["procedure_codes"],
        "claims": anchor["claims"],
        "summary": {
            "claim_types_present": claim_types_present,
            "diagnosis_code_count": len(anchor["diagnoses"]),
            "unique_diagnosis_code_count": len(diagnosis_cluster),
            "procedure_code_count": len(anchor["procedure_codes"]),
            "unique_hcpcs_cpt_code_count": len(hcpcs_cluster),
            "unique_icd9_procedure_code_count": len(icd9_procedure_cluster),
            "carrier_line_item_count": carrier_line_item_count,
            "has_facility_claim": True,
            "has_carrier_claim": carrier_line_item_count > 0,
        },
    }
