from __future__ import annotations

from datetime import date
from typing import Any


def clean_text(
    value: str | None,
    *,
    uppercase: bool = False,
    allow_all_zero: bool = True,
) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.upper() in {"NULL", "NAN", "NONE"}:
        return None
    if not allow_all_zero and set(normalized) == {"0"}:
        return None
    if uppercase:
        normalized = normalized.upper()
    return normalized


def parse_money(value: str | None) -> float | None:
    normalized = clean_text(value, allow_all_zero=True)
    if normalized is None:
        return None
    try:
        return float(normalized)
    except ValueError:
        return None


def parse_int(value: str | None) -> int | None:
    normalized = clean_text(value, allow_all_zero=True)
    if normalized is None:
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


def parse_cms_date(value: str | None) -> date | None:
    normalized = clean_text(value, allow_all_zero=False)
    if normalized is None or len(normalized) != 8 or not normalized.isdigit():
        return None
    try:
        return date(
            year=int(normalized[0:4]),
            month=int(normalized[4:6]),
            day=int(normalized[6:8]),
        )
    except ValueError:
        return None


def parse_iso_date(value: str | None) -> date | None:
    normalized = clean_text(value, allow_all_zero=False)
    if normalized is None:
        return None
    try:
        return date.fromisoformat(normalized)
    except ValueError:
        return None


def format_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def add_code_occurrence(
    occurrences: list[dict[str, Any]],
    *,
    code_system: str,
    code: str | None,
    source_claim_type: str,
    source_claim_id: str,
    source_field: str,
    position: int | None = None,
    line_number: int | None = None,
) -> None:
    normalized_code = clean_text(code, uppercase=True, allow_all_zero=False)
    if normalized_code is None:
        return

    occurrence = {
        "code_system": code_system,
        "code": normalized_code,
        "source_claim_type": source_claim_type,
        "source_claim_id": source_claim_id,
        "source_field": source_field,
    }
    if position is not None:
        occurrence["position"] = position
    if line_number is not None:
        occurrence["line_number"] = line_number
    occurrences.append(occurrence)


def extract_diagnosis_occurrences(
    row: dict[str, str],
    *,
    source_claim_type: str,
    source_claim_id: str,
    numbered_limit: int,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []

    for field in ("PRNCPAL_DGNS_CD", "ADMTNG_ICD9_DGNS_CD"):
        add_code_occurrence(
            occurrences,
            code_system="ICD9",
            code=row.get(field),
            source_claim_type=source_claim_type,
            source_claim_id=source_claim_id,
            source_field=field,
        )

    for index in range(1, numbered_limit + 1):
        for prefix in ("ICD9_DGNS_CD_", "LINE_ICD9_DGNS_CD_"):
            field = f"{prefix}{index}"
            add_code_occurrence(
                occurrences,
                code_system="ICD9",
                code=row.get(field),
                source_claim_type=source_claim_type,
                source_claim_id=source_claim_id,
                source_field=field,
                position=index,
            )

    return occurrences


def extract_icd9_procedure_occurrences(
    row: dict[str, str],
    *,
    source_claim_type: str,
    source_claim_id: str,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for index in range(1, 7):
        field = f"ICD9_PRCDR_CD_{index}"
        add_code_occurrence(
            occurrences,
            code_system="ICD9_PROCEDURE",
            code=row.get(field),
            source_claim_type=source_claim_type,
            source_claim_id=source_claim_id,
            source_field=field,
            position=index,
        )
    return occurrences


def extract_outpatient_hcpcs_occurrences(
    row: dict[str, str],
    *,
    source_claim_id: str,
) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for index in range(1, 46):
        field = f"HCPCS_CD_{index}"
        code = clean_text(row.get(field), uppercase=True, allow_all_zero=False)
        if code is None:
            continue

        occurrence: dict[str, Any] = {
            "code_system": "HCPCS/CPT",
            "code": code,
            "source_claim_type": "outpatient",
            "source_claim_id": source_claim_id,
            "source_field": field,
            "line_number": index,
        }

        revenue_center = clean_text(row.get(f"REV_CNTR_{index}"), allow_all_zero=False)
        service_date = parse_cms_date(row.get(f"REV_CNTR_DT_{index}"))
        payment_amount = parse_money(row.get(f"REV_CNTR_PMT_AMT_{index}"))
        total_charge_amount = parse_money(row.get(f"REV_CNTR_TOT_CHRG_AMT_{index}"))

        if revenue_center is not None:
            occurrence["revenue_center"] = revenue_center
        if service_date is not None:
            occurrence["service_date"] = format_date(service_date)
        if payment_amount is not None:
            occurrence["payment_amount"] = payment_amount
        if total_charge_amount is not None:
            occurrence["total_charge_amount"] = total_charge_amount

        occurrences.append(occurrence)

    return occurrences


def extract_carrier_line_items(
    row: dict[str, str],
    source_claim_id: str,
) -> list[dict[str, Any]]:
    line_items: list[dict[str, Any]] = []
    for index in range(1, 14):
        field = f"HCPCS_CD_{index}"
        code = clean_text(row.get(field), uppercase=True, allow_all_zero=False)
        if code is None:
            continue

        line_item: dict[str, Any] = {
            "code_system": "HCPCS/CPT",
            "code": code,
            "source_claim_type": "carrier",
            "source_claim_id": source_claim_id,
            "source_field": field,
            "line_number": index,
        }

        provider_npi = clean_text(row.get(f"PRF_PHYSN_NPI_{index}"), allow_all_zero=False)
        line_diagnosis_code = clean_text(
            row.get(f"LINE_ICD9_DGNS_CD_{index}"),
            uppercase=True,
            allow_all_zero=False,
        )
        amount_fields = {
            "medicare_payment_amount": row.get(f"LINE_NCH_PMT_AMT_{index}"),
            "beneficiary_deductible_amount": row.get(f"LINE_BENE_PTB_DDCTBL_AMT_{index}"),
            "beneficiary_coinsurance_amount": row.get(f"LINE_COINSRNC_AMT_{index}"),
            "primary_payer_paid_amount": row.get(f"LINE_BENE_PRMRY_PYR_PD_AMT_{index}"),
            "allowed_charge_amount": row.get(f"LINE_ALOWD_CHRG_AMT_{index}"),
            "submitted_charge_amount": row.get(f"LINE_SBMTD_CHRG_AMT_{index}"),
        }

        if provider_npi is not None:
            line_item["performing_physician_npi"] = provider_npi
        if line_diagnosis_code is not None:
            line_item["line_diagnosis_code"] = line_diagnosis_code

        for key, raw_value in amount_fields.items():
            parsed = parse_money(raw_value)
            if parsed is not None:
                line_item[key] = parsed

        line_items.append(line_item)

    return line_items


def facility_claim_amounts(row: dict[str, str], claim_type: str) -> dict[str, float]:
    raw_fields = {
        "claim_payment_amount": row.get("CLM_PMT_AMT"),
    }
    if claim_type == "inpatient":
        raw_fields["beneficiary_deductible_amount"] = row.get("NCH_BENE_IP_DDCTBL_AMT")
    if claim_type == "outpatient":
        raw_fields["beneficiary_deductible_amount"] = row.get("NCH_BENE_PTB_DDCTBL_AMT")
        raw_fields["beneficiary_coinsurance_amount"] = row.get("NCH_BENE_PTB_COINSRNC_AMT")
        raw_fields["blood_deductible_liability_amount"] = row.get(
            "NCH_BENE_BLOOD_DDCTBL_LBLTY_AMT"
        )

    amounts: dict[str, float] = {}
    for key, raw_value in raw_fields.items():
        parsed = parse_money(raw_value)
        if parsed is not None:
            amounts[key] = parsed
    return amounts


def carrier_claim_amounts(row: dict[str, str]) -> dict[str, float]:
    raw_fields = {
        "claim_payment_amount": row.get("CLM_PMT_AMT"),
        "submitted_charge_amount": row.get("NCH_CARR_CLM_SBMTD_CHRG_AMT"),
        "allowed_charge_amount": row.get("NCH_CARR_CLM_ALOWD_AMT"),
        "beneficiary_cash_deductible_amount": row.get("CARR_CLM_CASH_DDCTBL_APLD_AMT"),
        "primary_payer_paid_amount": row.get("CARR_CLM_PRMRY_PYR_PD_AMT"),
    }
    amounts: dict[str, float] = {}
    for key, raw_value in raw_fields.items():
        parsed = parse_money(raw_value)
        if parsed is not None:
            amounts[key] = parsed
    return amounts


def physician_npis(row: dict[str, str]) -> dict[str, str]:
    npis = {
        "attending_physician_npi": clean_text(row.get("AT_PHYSN_NPI"), allow_all_zero=False),
        "operating_physician_npi": clean_text(row.get("OP_PHYSN_NPI"), allow_all_zero=False),
        "other_physician_npi": clean_text(row.get("OT_PHYSN_NPI"), allow_all_zero=False),
    }
    return {key: value for key, value in npis.items() if value is not None}


def build_facility_anchor(row: dict[str, str], claim_type: str) -> dict[str, Any] | None:
    desynpuf_id = clean_text(row.get("DESYNPUF_ID"), uppercase=True, allow_all_zero=False)
    claim_id = clean_text(row.get("CLM_ID"), allow_all_zero=False)
    if desynpuf_id is None or claim_id is None:
        return None

    claim_from = parse_cms_date(row.get("CLM_FROM_DT"))
    claim_thru = parse_cms_date(row.get("CLM_THRU_DT")) or claim_from
    anchor_date = claim_from or claim_thru
    admission_date = None
    discharge_date = None
    carrier_match_from = claim_from
    carrier_match_thru = claim_thru
    if claim_type == "inpatient":
        admission_date = parse_cms_date(row.get("CLM_ADMSN_DT")) or claim_from
        discharge_date = (
            parse_cms_date(row.get("NCH_BENE_DSCHRG_DT"))
            or claim_thru
            or admission_date
        )
        carrier_match_from = admission_date
        carrier_match_thru = discharge_date

    diagnosis_occurrences = extract_diagnosis_occurrences(
        row,
        source_claim_type=claim_type,
        source_claim_id=claim_id,
        numbered_limit=10,
    )
    procedure_occurrences = extract_icd9_procedure_occurrences(
        row,
        source_claim_type=claim_type,
        source_claim_id=claim_id,
    )
    if claim_type == "outpatient":
        procedure_occurrences.extend(
            extract_outpatient_hcpcs_occurrences(row, source_claim_id=claim_id)
        )

    claim: dict[str, Any] = {
        "claim_type": claim_type,
        "claim_id": claim_id,
        "from_date": format_date(claim_from),
        "thru_date": format_date(claim_thru),
        "diagnosis_codes": diagnosis_occurrences,
        "procedure_codes": procedure_occurrences,
    }
    if admission_date is not None:
        claim["admission_date"] = format_date(admission_date)
    if discharge_date is not None:
        claim["discharge_date"] = format_date(discharge_date)

    provider_num = clean_text(row.get("PRVDR_NUM"), allow_all_zero=False)
    if provider_num is not None:
        claim["provider_num"] = provider_num
    claim_npis = physician_npis(row)
    if claim_npis:
        claim["physician_npis"] = claim_npis

    claim_amounts = facility_claim_amounts(row, claim_type)
    if claim_amounts:
        claim["amounts"] = claim_amounts

    utilization_days = parse_int(row.get("CLM_UTLZTN_DAY_CNT"))
    if utilization_days is not None:
        claim["utilization_day_count"] = utilization_days

    cluster_id = f"{desynpuf_id}-{claim_type}-{claim_id}"
    return {
        "cluster_id": cluster_id,
        "desynpuf_id": desynpuf_id,
        "anchor_claim_type": claim_type,
        "anchor_claim_id": claim_id,
        "anchor_from_date": claim_from,
        "anchor_thru_date": claim_thru,
        "carrier_match_from_date": carrier_match_from,
        "carrier_match_thru_date": carrier_match_thru,
        "anchor_year": anchor_date.year if anchor_date is not None else None,
        "claims": [claim],
        "diagnoses": list(diagnosis_occurrences),
        "procedure_codes": list(procedure_occurrences),
    }


def build_carrier_claim(
    row: dict[str, str],
) -> tuple[str, date | None, date | None, dict[str, Any]] | None:
    desynpuf_id = clean_text(row.get("DESYNPUF_ID"), uppercase=True, allow_all_zero=False)
    claim_id = clean_text(row.get("CLM_ID"), allow_all_zero=False)
    if desynpuf_id is None or claim_id is None:
        return None

    line_items = extract_carrier_line_items(row, claim_id)
    if not line_items:
        return None

    claim_from = parse_cms_date(row.get("CLM_FROM_DT"))
    claim_thru = parse_cms_date(row.get("CLM_THRU_DT")) or claim_from
    diagnosis_occurrences = extract_diagnosis_occurrences(
        row,
        source_claim_type="carrier",
        source_claim_id=claim_id,
        numbered_limit=13,
    )
    claim: dict[str, Any] = {
        "claim_type": "carrier",
        "claim_id": claim_id,
        "from_date": format_date(claim_from),
        "thru_date": format_date(claim_thru),
        "diagnosis_codes": diagnosis_occurrences,
        "line_items": line_items,
    }

    claim_amounts = carrier_claim_amounts(row)
    if claim_amounts:
        claim["amounts"] = claim_amounts

    return desynpuf_id, claim_from, claim_thru, claim


def simplify_beneficiary_row(row: dict[str, str], year: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "year": year,
    }

    field_mapping = {
        "sex_code": row.get("BENE_SEX_IDENT_CD"),
        "race_code": row.get("BENE_RACE_CD"),
        "state_code": row.get("SP_STATE_CODE"),
        "county_code": row.get("BENE_COUNTY_CD"),
    }
    for key, raw_value in field_mapping.items():
        normalized = clean_text(raw_value, allow_all_zero=False)
        if normalized is not None:
            context[key] = normalized

    birth_date = parse_cms_date(row.get("BENE_BIRTH_DT"))
    death_date = parse_cms_date(row.get("BENE_DEATH_DT"))
    if birth_date is not None:
        context["birth_date"] = format_date(birth_date)
    if death_date is not None:
        context["death_date"] = format_date(death_date)

    return context


def serialize_anchor_for_storage(anchor: dict[str, Any]) -> dict[str, Any]:
    payload = dict(anchor)
    payload["anchor_from_date"] = format_date(anchor.get("anchor_from_date"))
    payload["anchor_thru_date"] = format_date(anchor.get("anchor_thru_date"))
    payload["carrier_match_from_date"] = format_date(anchor.get("carrier_match_from_date"))
    payload["carrier_match_thru_date"] = format_date(anchor.get("carrier_match_thru_date"))
    return payload


def deserialize_anchor_from_storage(payload: dict[str, Any]) -> dict[str, Any]:
    anchor = dict(payload)
    anchor["anchor_from_date"] = parse_iso_date(payload.get("anchor_from_date"))
    anchor["anchor_thru_date"] = parse_iso_date(payload.get("anchor_thru_date"))
    anchor["carrier_match_from_date"] = parse_iso_date(payload.get("carrier_match_from_date"))
    anchor["carrier_match_thru_date"] = parse_iso_date(payload.get("carrier_match_thru_date"))
    return anchor


def serialize_carrier_record_for_storage(
    desynpuf_id: str,
    claim_from: date | None,
    claim_thru: date | None,
    claim: dict[str, Any],
) -> dict[str, Any]:
    return {
        "desynpuf_id": desynpuf_id,
        "claim_from": format_date(claim_from),
        "claim_thru": format_date(claim_thru),
        "claim": claim,
    }


def deserialize_carrier_record_from_storage(
    payload: dict[str, Any],
) -> tuple[date | None, date | None, dict[str, Any]]:
    return (
        parse_iso_date(payload.get("claim_from")),
        parse_iso_date(payload.get("claim_thru")),
        dict(payload["claim"]),
    )
