from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

from claim.datagen import cluster, constants, utils
from claim.datagen.contracts import ClusterGenerationContext, PricingBuildContext
from claim.datagen.llm import enrich_bill_descriptions, request_aftercare_summary
from claim import gold_bill_support, gold_bill_trilliant
from claim.gold_bill_trilliant import TrilliantSchema
from claim.inpatient_anchor import canonicalize_inpatient_anchor_code_pair

ACCOUNTING_FIELDS = (
    "billed_charge",
    "contractual_adjustment",
    "allowed_amount",
    "insurance_paid",
    "patient_responsibility",
)


def compute_line_item_accounting(
    quote: dict[str, Any],
    *,
    scenario: str,
    insurance_coverage_ratio: float,
    billed_charge_multiplier: float,
) -> dict[str, float]:
    allowed_candidate = gold_bill_support.parse_float(quote.get("negotiated_rate"))
    gross_candidate = gold_bill_support.parse_float(quote.get("gross_charge"))

    if scenario == "in_network":
        if allowed_candidate is None or allowed_candidate <= 0:
            raise RuntimeError("Missing allowed amount for in-network pricing.")
        allowed_amount = gold_bill_support.round_currency(allowed_candidate)
        billed_charge_candidate = (
            gross_candidate
            if gross_candidate and gross_candidate > 0
            else allowed_amount * max(1.0, billed_charge_multiplier)
        )
        billed_charge = gold_bill_support.round_currency(
            max(allowed_amount, float(billed_charge_candidate))
        )
        contractual_adjustment = gold_bill_support.round_currency(
            max(0.0, billed_charge - allowed_amount)
        )
        insurance_paid = gold_bill_support.round_currency(
            allowed_amount * max(0.0, min(1.0, insurance_coverage_ratio))
        )
        patient_responsibility = gold_bill_support.round_currency(
            max(0.0, allowed_amount - insurance_paid)
        )
        return {
            "billed_charge": billed_charge,
            "contractual_adjustment": contractual_adjustment,
            "allowed_amount": allowed_amount,
            "insurance_paid": insurance_paid,
            "patient_responsibility": patient_responsibility,
        }

    billed_charge_source = gross_candidate if gross_candidate and gross_candidate > 0 else allowed_candidate
    if billed_charge_source is None or billed_charge_source <= 0:
        raise RuntimeError("Missing billed charge for extreme pricing.")
    billed_charge = gold_bill_support.round_currency(billed_charge_source)
    return {
        "billed_charge": billed_charge,
        "contractual_adjustment": 0.0,
        "allowed_amount": billed_charge,
        "insurance_paid": 0.0,
        "patient_responsibility": billed_charge,
    }


def bill_totals(
    line_items: list[dict[str, Any]],
    case_components: Sequence[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    components = list(case_components or [])
    totals = {
        "line_item_count": len(line_items),
        "billed_charge": 0.0,
        "contractual_adjustment": 0.0,
        "allowed_amount": 0.0,
        "insurance_paid": 0.0,
        "patient_responsibility": 0.0,
    }
    for priced_item in list(line_items) + components:
        accounting = priced_item["accounting"]
        for field in ACCOUNTING_FIELDS:
            totals[field] = gold_bill_support.round_currency(
                float(totals[field]) + float(accounting[field])
            )
    return totals


def claim_accounting_summary(
    line_items: Sequence[dict[str, Any]],
    case_components: Sequence[dict[str, Any]] | None = None,
) -> dict[str, float | int]:
    components = list(case_components or [])
    totals = bill_totals(list(line_items), components)
    totals["component_count"] = len(components)
    return totals


def line_item_claim_class(line_item: dict[str, Any]) -> str:
    claim_class = gold_bill_support.normalize_text(line_item.get("claim_class"))
    if claim_class == "carrier":
        return "professional"
    if claim_class == "facility":
        return "facility"
    if claim_class == "unknown":
        return "unknown"
    source_claim_type = gold_bill_support.normalize_text(line_item.get("source_claim_type"))
    return "professional" if source_claim_type == "carrier" else "facility"


def occurrence_support_status(occurrence: dict[str, Any]) -> str:
    return gold_bill_support.normalize_text(occurrence.get("support_status")) or "llm_augmented"


def occurrence_claim_class(occurrence: dict[str, Any]) -> str:
    claim_class = gold_bill_support.normalize_text(occurrence.get("claim_class"))
    if claim_class in {"carrier", "facility", "unknown"}:
        return claim_class
    source_claim_type = gold_bill_support.normalize_text(occurrence.get("source_claim_type"))
    if source_claim_type == "carrier":
        return "carrier"
    if source_claim_type in {"outpatient", "inpatient"}:
        return "facility"
    return "unknown"


def pricing_provenance(quote: dict[str, Any]) -> dict[str, Any]:
    return {
        "lookup_level": quote.get("lookup_level"),
        "payer_name": quote.get("payer_name"),
        "row_count": quote.get("row_count"),
        "identity_filter": quote.get("identity_filter"),
        "state_value": quote.get("state_value"),
        "baseline_amount": quote.get("baseline_amount"),
        "source_allowed_amount": quote.get("source_allowed_amount"),
        "floor_amount": quote.get("floor_amount"),
        "floor_applied": quote.get("floor_applied"),
        "source_observed_field": quote.get("source_observed_field"),
        "hospital_count": quote.get("hospital_count"),
        "supporting_row_count": quote.get("supporting_row_count"),
        "prior_lookup_scope": quote.get("prior_lookup_scope"),
        "prior_hospital_scan_count": quote.get("prior_hospital_scan_count"),
        "prior_hospital_scan_cap": quote.get("prior_hospital_scan_cap"),
        "prior_cache_status": quote.get("prior_cache_status"),
        "estimate_basis": quote.get("estimate_basis"),
        "estimate_support_line_count": quote.get("estimate_support_line_count"),
        "estimate_charge_field": quote.get("estimate_charge_field"),
        "estimate_ratio": quote.get("estimate_ratio"),
        "estimate_scope": quote.get("estimate_scope"),
        "estimate_source_hospital_id": quote.get("estimate_source_hospital_id"),
        "estimate_source_hospital_name": quote.get("estimate_source_hospital_name"),
        "estimate_source_lookup_level": quote.get("estimate_source_lookup_level"),
    }


def has_outpatient_facility_pricing_path(
    occurrence: dict[str, Any],
    quote_by_code: dict[str, dict[str, Any]],
) -> bool:
    billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
    if billing_code is None:
        return False
    if billing_code in quote_by_code:
        return True
    return gold_bill_trilliant.build_source_observed_quote(occurrence) is not None


def select_payable_occurrences(
    procedure_occurrences: Sequence[dict[str, Any]],
    quote_by_code: dict[str, dict[str, Any]],
    component_quote_by_code: dict[str, dict[str, Any]],
    *,
    encounter_regime: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    hcpcs_occurrences = [
        occurrence
        for occurrence in procedure_occurrences
        if utils.normalize_generated_code_system(occurrence.get("code_system")) == "HCPCS/CPT"
    ]
    component_occurrences = [
        occurrence
        for occurrence in procedure_occurrences
        if utils.normalize_generated_code_system(occurrence.get("code_system")) != "HCPCS/CPT"
    ]
    if encounter_regime != "inpatient":
        payable_lines = [
            occurrence
            for occurrence in hcpcs_occurrences
            if occurrence_claim_class(occurrence) == "carrier"
            or (
                occurrence_claim_class(occurrence) == "facility"
                and has_outpatient_facility_pricing_path(occurrence, quote_by_code)
            )
        ]
        return payable_lines, [], False

    payable_drg_components = [
        occurrence
        for occurrence in component_occurrences
        if utils.normalize_generated_code_system(occurrence.get("code_system"))
        in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS
    ]
    has_payable_drg_component = bool(payable_drg_components)
    if has_payable_drg_component:
        payable_lines = [
            occurrence
            for occurrence in hcpcs_occurrences
            if occurrence_claim_class(occurrence) == "carrier"
        ]
        return payable_lines, payable_drg_components, True

    payable_lines = [
        occurrence
        for occurrence in hcpcs_occurrences
        if occurrence_claim_class(occurrence) == "carrier"
        or (
            occurrence_claim_class(occurrence) == "facility"
            and gold_bill_support.normalize_code(occurrence.get("code")) in quote_by_code
        )
    ]
    payable_components: list[dict[str, Any]] = []
    for occurrence in component_occurrences:
        raw_billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
        code_system = utils.normalize_generated_code_system(occurrence.get("code_system"))
        canonical_code_system, canonical_billing_code = canonicalize_inpatient_anchor_code_pair(
            code_system,
            raw_billing_code,
        )
        billing_code = canonical_billing_code or raw_billing_code
        normalized_code_system = canonical_code_system or code_system
        if billing_code is None or normalized_code_system is None:
            continue
        if f"{normalized_code_system}:{billing_code}" in component_quote_by_code:
            payable_components.append(occurrence)
    return payable_lines, payable_components, False


def resolve_clean_line_quote(
    occurrence: dict[str, Any],
    quote_by_code: dict[str, dict[str, Any]],
    *,
    session: Any,
    connection: Any,
    schema: TrilliantSchema,
    scenario: dict[str, Any],
    encounter_regime: str,
    source_state_value: str | None,
    max_quote_amount: float,
    cache_lock: Any,
    local_estimate_context: dict[str, Any] | None = None,
    logger: Any | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    del session, connection, schema, encounter_regime, source_state_value, max_quote_amount, cache_lock, logger, case_id
    billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
    if billing_code is None:
        raise RuntimeError("Missing billing code for payable HCPCS/CPT occurrence.")
    quote = quote_by_code.get(billing_code)
    if quote is not None:
        return quote
    if occurrence_support_status(occurrence) == "raw_supported":
        observed_quote = gold_bill_trilliant.build_source_observed_quote(occurrence)
        if observed_quote is not None:
            return observed_quote
    if (
        scenario["scenario"] in {"in_network", "self_pay", "out_of_network"}
        and occurrence_claim_class(occurrence) == "carrier"
    ):
        if local_estimate_context is not None:
            local_estimate_context.setdefault("quotes", {})
            local_estimate_context.setdefault("misses", set())
            local_estimate_quote = local_estimate_context["quotes"].get(billing_code)
            if local_estimate_quote is not None:
                return local_estimate_quote
    raise RuntimeError(f"Missing price for payable HCPCS/CPT {billing_code}.")


def resolve_case_hospital_local_estimate_quotes(
    procedure_occurrences: Sequence[dict[str, Any]],
    quote_by_code: dict[str, dict[str, Any]],
    *,
    hospital: dict[str, Any],
    local_estimate_context: dict[str, Any],
    session: Any,
    connection: Any,
    schema: TrilliantSchema,
    encounter: dict[str, Any],
    scenario: dict[str, Any],
    encounter_regime: str,
    max_quote_amount: float,
    cache_lock: Any,
    logger: Any | None,
    hospital_candidates: Sequence[dict[str, Any]] | None = None,
    geography: dict[str, Any] | None = None,
    local_hud_mapping: dict[str, list[str]] | None = None,
    primary_clinical_direction: str | None = None,
    seed: int = 0,
    max_hospital_rate_probes: int = 1,
    max_nearby_fallback_probes: int = constants.DEFAULT_MAX_NEARBY_HOSPITAL_FALLBACK_PROBES,
) -> None:
    local_estimate_context.setdefault("quotes", {})
    local_estimate_context.setdefault("misses", set())
    if scenario["scenario"] not in {"in_network", "self_pay", "out_of_network"}:
        return
    min_scoped_cache_support = 3

    def uses_source_observed(occurrence: dict[str, Any]) -> bool:
        return (
            occurrence_support_status(occurrence) == "raw_supported"
            and gold_bill_trilliant.build_source_observed_quote(occurrence) is not None
        )

    missing_codes = gold_bill_support.unique_preserving_order(
        billing_code
        for occurrence in procedure_occurrences
        for billing_code in [gold_bill_support.normalize_code(occurrence.get("code"))]
        if (
            billing_code is not None
            and utils.normalize_generated_code_system(occurrence.get("code_system")) == "HCPCS/CPT"
            and occurrence_claim_class(occurrence) == "carrier"
            and not uses_source_observed(occurrence)
            and billing_code not in quote_by_code
            and billing_code not in local_estimate_context["quotes"]
            and billing_code not in local_estimate_context["misses"]
        )
    )
    if not missing_codes:
        return

    case_id = gold_bill_support.normalize_text(encounter.get("cluster_id"))
    professional_codes = gold_bill_support.unique_preserving_order(
        billing_code
        for occurrence in procedure_occurrences
        for billing_code in [gold_bill_support.normalize_code(occurrence.get("code"))]
        if (
            billing_code is not None
            and utils.normalize_generated_code_system(occurrence.get("code_system")) == "HCPCS/CPT"
            and occurrence_claim_class(occurrence) == "carrier"
            and not uses_source_observed(occurrence)
        )
    )
    support_quotes = [
        {
            **quote_by_code[billing_code],
            "billing_code": billing_code,
        }
        for occurrence in procedure_occurrences
        for billing_code in [gold_bill_support.normalize_code(occurrence.get("code"))]
        if (
            billing_code is not None
            and utils.normalize_generated_code_system(occurrence.get("code_system")) == "HCPCS/CPT"
            and occurrence_claim_class(occurrence) == "carrier"
            and not uses_source_observed(occurrence)
            and billing_code in quote_by_code
            and gold_bill_support.normalize_text(quote_by_code[billing_code].get("lookup_level"))
            in {"hospital_payer", "hospital_average"}
        )
    ]
    support_data = gold_bill_trilliant.build_hospital_local_professional_support(
        support_quotes
    )

    exact_quotes = gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes(
        connection,
        schema,
        session=session,
        billing_codes=missing_codes,
        hospital=hospital,
        encounter_regime=encounter_regime,
        max_quote_amount=max_quote_amount,
        cache_lock=cache_lock,
    )
    unresolved_codes = list(missing_codes)
    if support_data.get("hospital_ratios"):
        for billing_code in list(unresolved_codes):
            exact_quote = exact_quotes.get(billing_code)
            if exact_quote is None:
                continue
            estimate_quote = gold_bill_trilliant.build_hospital_local_estimate_quote(
                exact_quote,
                support_data,
                scenario_name=scenario["scenario"],
                max_quote_amount=max_quote_amount,
            )
            if estimate_quote is None:
                continue
            local_estimate_context["quotes"][billing_code] = estimate_quote
            unresolved_codes.remove(billing_code)
            if logger is not None:
                logger.info(
                    "Resolved selected-hospital local estimate",
                    case_id=case_id,
                    billing_code=billing_code,
                    scenario=scenario["scenario"],
                    basis=estimate_quote.get("estimate_basis"),
                    support_count=estimate_quote.get("estimate_support_line_count"),
                    charge_field=estimate_quote.get("estimate_charge_field"),
                )

    def clone_with_nearby_metadata(
        quote: dict[str, Any],
        *,
        lookup_level: str,
        scope: str,
        source_hospital: dict[str, Any],
        source_lookup_level: str | None,
    ) -> dict[str, Any]:
        cloned = dict(quote)
        cloned["lookup_level"] = lookup_level
        cloned["estimate_scope"] = scope
        cloned["estimate_source_hospital_id"] = source_hospital.get("hospital_directory_id")
        cloned["estimate_source_hospital_name"] = source_hospital.get("hospital_name")
        cloned["estimate_source_lookup_level"] = source_lookup_level
        return cloned

    def is_anesthesia_professional_code(billing_code: str) -> bool:
        return billing_code.isdigit() and len(billing_code) == 5 and 100 <= int(billing_code) <= 1999

    if unresolved_codes and schema.source_mode != "public_directory":
        warning_message = (
            "Nearby-hospital fallback is deprecated and unsupported for non-public-directory "
            "Trilliant schemas."
        )
        if logger is not None:
            logger.warning(
                warning_message,
                case_id=case_id,
                source_mode=schema.source_mode,
            )
        else:
            gold_bill_trilliant.LOGGER.warning(
                "%s case_id=%s source_mode=%s",
                warning_message,
                case_id,
                schema.source_mode,
            )
        raise RuntimeError(warning_message)

    def nearby_candidate_groups() -> list[tuple[str, list[dict[str, Any]]]]:
        all_candidates = list(hospital_candidates or [])
        selected_hospital_id = gold_bill_support.normalize_text(
            hospital.get("hospital_directory_id")
        )
        filtered_candidates = [
            candidate
            for candidate in all_candidates
            if gold_bill_support.normalize_text(candidate.get("hospital_directory_id"))
            != selected_hospital_id
        ]
        if not filtered_candidates:
            return []
        reverse_zip_to_counties = gold_bill_trilliant.reverse_zip_to_county_mapping(
            local_hud_mapping
        )
        zip_codes = list((geography or {}).get("zip_codes") or [])
        county_fips = gold_bill_support.normalize_text((geography or {}).get("county_fips"))
        state_value = gold_bill_support.normalize_text((geography or {}).get("state_abbrev"))
        pools: list[tuple[str, list[dict[str, Any]]]] = []
        for stage in ("zip", "county", "same_state"):
            pool = gold_bill_trilliant.public_directory_candidate_pool(
                filtered_candidates,
                stage=stage,
                zip_codes=zip_codes,
                county_fips=county_fips,
                state_value=state_value,
                reverse_zip_to_counties=reverse_zip_to_counties,
                primary_clinical_direction=primary_clinical_direction,
                specialty_filter_active=True,
            )
            if not pool:
                pool = gold_bill_trilliant.public_directory_candidate_pool(
                    filtered_candidates,
                    stage=stage,
                    zip_codes=zip_codes,
                    county_fips=county_fips,
                    state_value=state_value,
                    reverse_zip_to_counties=reverse_zip_to_counties,
                    primary_clinical_direction=primary_clinical_direction,
                    specialty_filter_active=False,
                )
            if not pool:
                continue
            pools.append(
                (
                    stage,
                    gold_bill_trilliant.ordered_hospital_candidates_for_case(
                        case_id or "",
                        seed,
                        pool,
                    ),
                )
            )
        return pools

    candidate_groups = nearby_candidate_groups()
    remaining_nearby_probes = max(0, int(max_nearby_fallback_probes))
    for scope, candidates in candidate_groups:
        if not unresolved_codes:
            break
        if remaining_nearby_probes <= 0:
            break
        for candidate in candidates[:remaining_nearby_probes]:
            if not unresolved_codes:
                break
            candidate_quote_by_code, _telemetry = gold_bill_trilliant.resolve_rate_quotes(
                connection,
                schema,
                session=session,
                billing_codes=professional_codes,
                scenario=scenario,
                hospital=dict(candidate),
                encounter_regime=encounter_regime,
                source_state_value=gold_bill_support.normalize_text(
                    candidate.get("hospital_state")
                ),
                max_quote_amount=max_quote_amount,
                cache_lock=cache_lock,
            )
            remaining_nearby_probes -= 1
            candidate_support_data = gold_bill_trilliant.build_hospital_local_professional_support(
                [
                    {
                        **candidate_quote_by_code[billing_code],
                        "billing_code": billing_code,
                    }
                    for billing_code in professional_codes
                    if billing_code in candidate_quote_by_code
                ]
            )
            for billing_code in list(unresolved_codes):
                candidate_direct_quote = candidate_quote_by_code.get(billing_code)
                selected_exact_quote = exact_quotes.get(billing_code)
                if selected_exact_quote is not None and candidate_direct_quote is not None:
                    charge_anchor, _charge_field = gold_bill_trilliant.quote_charge_anchor(
                        candidate_direct_quote
                    )
                    allowed_amount = gold_bill_support.parse_float(
                        candidate_direct_quote.get("negotiated_rate")
                    )
                    if (
                        charge_anchor is not None
                        and charge_anchor > 0
                        and allowed_amount is not None
                        and allowed_amount > 0
                    ):
                        exact_support = {
                            "ratios_by_family": {},
                            "hospital_ratios": [allowed_amount / charge_anchor],
                        }
                        estimate_quote = gold_bill_trilliant.build_hospital_local_estimate_quote(
                            selected_exact_quote,
                            exact_support,
                            scenario_name=scenario["scenario"],
                            max_quote_amount=max_quote_amount,
                        )
                        if estimate_quote is not None:
                            estimate_quote = clone_with_nearby_metadata(
                                estimate_quote,
                                lookup_level="nearby_hospital_estimate",
                                scope=scope,
                                source_hospital=candidate,
                                source_lookup_level=gold_bill_support.normalize_text(
                                    candidate_direct_quote.get("lookup_level")
                                ),
                            )
                            estimate_quote["estimate_basis"] = "nearby_exact"
                            local_estimate_context["quotes"][billing_code] = estimate_quote
                            unresolved_codes.remove(billing_code)
                            if logger is not None:
                                logger.info(
                                    "Resolved nearby-hospital estimate",
                                    case_id=case_id,
                                    billing_code=billing_code,
                                    scenario=scenario["scenario"],
                                    scope=scope,
                                    basis=estimate_quote.get("estimate_basis"),
                                    source_hospital_id=candidate.get("hospital_directory_id"),
                                )
                            continue
                if (
                    selected_exact_quote is not None
                    and candidate_support_data.get("hospital_ratios")
                ):
                    estimate_quote = gold_bill_trilliant.build_hospital_local_estimate_quote(
                        selected_exact_quote,
                        candidate_support_data,
                        scenario_name=scenario["scenario"],
                        max_quote_amount=max_quote_amount,
                    )
                    if estimate_quote is not None:
                        estimate_quote = clone_with_nearby_metadata(
                            estimate_quote,
                            lookup_level="nearby_hospital_estimate",
                            scope=scope,
                            source_hospital=candidate,
                            source_lookup_level=gold_bill_support.normalize_text(
                                (candidate_direct_quote or {}).get("lookup_level")
                            ),
                        )
                        estimate_quote["estimate_basis"] = (
                            f"nearby_{estimate_quote.get('estimate_basis')}"
                        )
                        local_estimate_context["quotes"][billing_code] = estimate_quote
                        unresolved_codes.remove(billing_code)
                        if logger is not None:
                            logger.info(
                                "Resolved nearby-hospital estimate",
                                case_id=case_id,
                                billing_code=billing_code,
                                scenario=scenario["scenario"],
                                scope=scope,
                                basis=estimate_quote.get("estimate_basis"),
                                source_hospital_id=candidate.get("hospital_directory_id"),
                            )
                        continue
                if candidate_direct_quote is not None:
                    direct_quote = clone_with_nearby_metadata(
                        candidate_direct_quote,
                        lookup_level="nearby_hospital_quote",
                        scope=scope,
                        source_hospital=candidate,
                        source_lookup_level=gold_bill_support.normalize_text(
                            candidate_direct_quote.get("lookup_level")
                        ),
                    )
                    direct_quote["baseline_amount"] = direct_quote.get("negotiated_rate")
                    direct_quote["estimate_basis"] = "nearby_direct"
                    local_estimate_context["quotes"][billing_code] = direct_quote
                    unresolved_codes.remove(billing_code)
                    if logger is not None:
                        logger.info(
                            "Resolved nearby-hospital direct quote",
                            case_id=case_id,
                            billing_code=billing_code,
                            scenario=scenario["scenario"],
                            scope=scope,
                            source_hospital_id=candidate.get("hospital_directory_id"),
                        )

    scoped_candidate_ids = {
        scope: {
            hospital_id
            for hospital_id in (
                gold_bill_support.normalize_text(candidate.get("hospital_directory_id"))
                for candidate in candidates
            )
            if hospital_id is not None
        }
        for scope, candidates in candidate_groups
    }
    for billing_code in list(unresolved_codes):
        selected_exact_quote = exact_quotes.get(billing_code)
        cached_quote = None
        for estimate_scope in ("zip", "county", "same_state", "national"):
            hospital_ids = None
            if estimate_scope != "national":
                hospital_ids = scoped_candidate_ids.get(estimate_scope)
                if not hospital_ids:
                    continue
            cached_quotes = gold_bill_trilliant.cached_professional_quote_corpus(
                connection,
                billing_code=billing_code,
                scenario_name=scenario["scenario"],
                payer_name=scenario.get("payer_name"),
                max_quote_amount=max_quote_amount,
                hospital_ids=hospital_ids,
                cache_lock=cache_lock,
            )
            if (
                not cached_quotes
                and scenario["scenario"] == "in_network"
                and estimate_scope == "national"
            ):
                cached_quotes = gold_bill_trilliant.cached_professional_quote_corpus(
                    connection,
                    billing_code=billing_code,
                    scenario_name=scenario["scenario"],
                    payer_name=scenario.get("payer_name"),
                    max_quote_amount=max_quote_amount,
                    hospital_ids=None,
                    match_payer=False,
                    cache_lock=cache_lock,
                )
            if not cached_quotes:
                continue
            if selected_exact_quote is not None:
                cached_quote = gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote(
                    selected_exact_quote,
                    cached_quotes,
                    scenario_name=scenario["scenario"],
                    max_quote_amount=max_quote_amount,
                    estimate_scope=estimate_scope,
                )
                if cached_quote is not None:
                    if (
                        estimate_scope != "national"
                        and int(cached_quote.get("estimate_support_line_count") or 0)
                        < min_scoped_cache_support
                    ):
                        cached_quote = None
                        continue
                    if logger is not None:
                        logger.info(
                            "Resolved cached-corpus estimate",
                            case_id=case_id,
                            billing_code=billing_code,
                            scenario=scenario["scenario"],
                            scope=estimate_scope,
                            support_count=cached_quote.get("estimate_support_line_count"),
                        )
                    break
            cached_quote = gold_bill_trilliant.build_cached_corpus_direct_quote(
                billing_code,
                cached_quotes,
                max_quote_amount=max_quote_amount,
                estimate_scope=estimate_scope,
            )
            if cached_quote is not None:
                if (
                    estimate_scope != "national"
                    and int(cached_quote.get("estimate_support_line_count") or 0)
                    < min_scoped_cache_support
                ):
                    cached_quote = None
                    continue
                if logger is not None:
                    logger.info(
                        "Resolved cached-corpus direct quote",
                        case_id=case_id,
                        billing_code=billing_code,
                        scenario=scenario["scenario"],
                        scope=estimate_scope,
                        support_count=cached_quote.get("estimate_support_line_count"),
                    )
                break
        if cached_quote is None:
            continue
        local_estimate_context["quotes"][billing_code] = cached_quote
        unresolved_codes.remove(billing_code)

    if unresolved_codes:
        augmented_support_quotes = [
            {
                **quote_by_code[billing_code],
                "billing_code": billing_code,
            }
            for billing_code in professional_codes
            if (
                billing_code in quote_by_code
                and gold_bill_support.normalize_text(quote_by_code[billing_code].get("lookup_level"))
                in {"hospital_payer", "hospital_average"}
            )
        ]
        augmented_support_quotes.extend(
            {
                **quote,
                "billing_code": billing_code,
            }
            for billing_code, quote in local_estimate_context["quotes"].items()
            if billing_code in professional_codes
        )
        augmented_support_data = gold_bill_trilliant.build_hospital_local_professional_support(
            augmented_support_quotes
        )
        if augmented_support_data.get("hospital_ratios"):
            for billing_code in list(unresolved_codes):
                selected_exact_quote = exact_quotes.get(billing_code)
                if selected_exact_quote is None:
                    continue
                estimate_quote = gold_bill_trilliant.build_hospital_local_estimate_quote(
                    selected_exact_quote,
                    augmented_support_data,
                    scenario_name=scenario["scenario"],
                    max_quote_amount=max_quote_amount,
                )
                if estimate_quote is None:
                    continue
                estimate_quote["lookup_level"] = "cached_corpus_estimate"
                estimate_quote["estimate_basis"] = "resolved_support"
                estimate_quote["estimate_scope"] = "national"
                estimate_quote["estimate_source_lookup_level"] = "resolved_support"
                local_estimate_context["quotes"][billing_code] = estimate_quote
                unresolved_codes.remove(billing_code)
                if logger is not None:
                    logger.info(
                        "Resolved support-seeded estimate",
                        case_id=case_id,
                        billing_code=billing_code,
                        scenario=scenario["scenario"],
                        support_count=estimate_quote.get("estimate_support_line_count"),
                    )

    for billing_code in list(unresolved_codes):
        billing_family = gold_bill_trilliant.professional_billing_code_family(billing_code)
        if billing_family is None:
            continue
        family_quote = None
        for estimate_scope in ("zip", "county", "same_state", "national"):
            hospital_ids = None
            if estimate_scope != "national":
                hospital_ids = scoped_candidate_ids.get(estimate_scope)
                if not hospital_ids:
                    continue
            family_quotes = gold_bill_trilliant.cached_professional_family_quote_corpus(
                connection,
                billing_family=billing_family,
                scenario_name=scenario["scenario"],
                payer_name=scenario.get("payer_name"),
                max_quote_amount=max_quote_amount,
                hospital_ids=hospital_ids,
                cache_lock=cache_lock,
            )
            if (
                not family_quotes
                and scenario["scenario"] == "in_network"
                and estimate_scope == "national"
            ):
                family_quotes = gold_bill_trilliant.cached_professional_family_quote_corpus(
                    connection,
                    billing_family=billing_family,
                    scenario_name=scenario["scenario"],
                    payer_name=scenario.get("payer_name"),
                    max_quote_amount=max_quote_amount,
                    hospital_ids=None,
                    match_payer=False,
                    cache_lock=cache_lock,
                )
            if not family_quotes:
                continue
            family_quote = gold_bill_trilliant.build_cached_corpus_direct_quote(
                billing_code,
                family_quotes,
                max_quote_amount=max_quote_amount,
                estimate_scope=estimate_scope,
            )
            if family_quote is None:
                continue
            if (
                estimate_scope != "national"
                and int(family_quote.get("estimate_support_line_count") or 0)
                < min_scoped_cache_support
            ):
                family_quote = None
                continue
            family_quote["estimate_basis"] = "cached_family_median"
            if logger is not None:
                logger.info(
                    "Resolved cached-family direct quote",
                    case_id=case_id,
                    billing_code=billing_code,
                    scenario=scenario["scenario"],
                    scope=estimate_scope,
                    support_count=family_quote.get("estimate_support_line_count"),
                )
            break
        if family_quote is None:
            continue
        local_estimate_context["quotes"][billing_code] = family_quote
        unresolved_codes.remove(billing_code)

    if unresolved_codes:
        resolved_allowed_amounts = [
            amount
            for amount in (
                gold_bill_support.parse_float(quote.get("negotiated_rate"))
                for quote in [
                    *[
                        quote_by_code[billing_code]
                        for billing_code in professional_codes
                        if billing_code in quote_by_code
                    ],
                    *list(local_estimate_context["quotes"].values()),
                ]
            )
            if amount is not None and amount > 0
        ]
        max_resolved_allowed = max(resolved_allowed_amounts, default=None)
        if max_resolved_allowed is not None and max_resolved_allowed > 0:
            for billing_code in list(unresolved_codes):
                if not is_anesthesia_professional_code(billing_code):
                    continue
                estimated_allowed = gold_bill_trilliant.sanitized_quote_amount(
                    max_resolved_allowed * 0.1,
                    max_quote_amount=max_quote_amount,
                )
                if estimated_allowed is None or estimated_allowed <= 0:
                    continue
                local_estimate_context["quotes"][billing_code] = {
                    "billing_code": billing_code,
                    "negotiated_rate": estimated_allowed,
                    "gross_charge": estimated_allowed,
                    "lookup_level": "cached_corpus_quote",
                    "payer_name": None,
                    "identity_filter": None,
                    "state_value": None,
                    "baseline_amount": estimated_allowed,
                    "estimate_basis": "anesthesia_share",
                    "estimate_support_line_count": 1,
                    "estimate_scope": "bill_support",
                    "estimate_source_lookup_level": "resolved_support",
                }
                unresolved_codes.remove(billing_code)
                if logger is not None:
                    logger.info(
                        "Resolved anesthesia-share estimate",
                        case_id=case_id,
                        billing_code=billing_code,
                        scenario=scenario["scenario"],
                        support_count=1,
                    )

    local_estimate_context["misses"].update(unresolved_codes)


def inpatient_pricing_validation_errors(bill_record: dict[str, Any]) -> list[str]:
    if (utils.encounter_regime(bill_record) or "outpatient") != "inpatient":
        return []
    components = utils.bill_case_components(bill_record)
    if not any(
        component.get("code_system") in constants.INPATIENT_FACILITY_ANCHOR_CODE_SYSTEMS
        for component in components
    ):
        return ["Inpatient bill is missing a facility anchor component."]
    errors: list[str] = []
    length_of_stay = gold_bill_trilliant.inpatient_length_of_stay(
        (bill_record.get("source_encounter") or {})
    ) or utils.parse_int(
        (bill_record.get("admission") or {}).get("length_of_stay")
    )
    floor_amount = gold_bill_trilliant.inpatient_anchor_floor_amount(length_of_stay)
    for component in components:
        if component.get("code_system") not in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS:
            continue
        provenance = component.get("pricing_provenance") or {}
        lookup_level = gold_bill_support.normalize_text(provenance.get("lookup_level"))
        if lookup_level == "source_imputed":
            errors.append(
                "DRG-family component used generic source_imputed pricing: "
                f"{component.get('code_system')} {component.get('billing_code')}"
            )
            continue
        if lookup_level != "inpatient_anchor_imputed":
            continue
        allowed_amount = gold_bill_support.parse_float(
            (component.get("accounting") or {}).get("allowed_amount")
        )
        if allowed_amount is None or allowed_amount < floor_amount:
            errors.append(
                "Inpatient anchor-imputed DRG-family component fell below the LOS floor: "
                f"{component.get('code_system')} {component.get('billing_code')} "
                f"(allowed={allowed_amount}, floor={floor_amount})"
            )
    return errors


def build_inpatient_admission(
    source_encounter: dict[str, Any],
    facility: dict[str, Any],
    case_components: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    encounter = (source_encounter.get("encounter") or {})
    admission_date = gold_bill_support.normalize_text(
        encounter.get("admission_date")
    ) or gold_bill_support.normalize_text(encounter.get("from_date"))
    discharge_date = gold_bill_support.normalize_text(
        encounter.get("discharge_date")
    ) or gold_bill_support.normalize_text(encounter.get("thru_date"))
    length_of_stay = None
    parsed_admission = utils.parse_iso_date(admission_date)
    parsed_discharge = utils.parse_iso_date(discharge_date)
    if parsed_admission is not None and parsed_discharge is not None:
        length_of_stay = max(1, (parsed_discharge - parsed_admission).days)
    drg_component = next(
        (
            component
            for component in case_components
            if component.get("code_system") in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS
        ),
        None,
    )
    admission = {
        "hospital": facility.get("hospital_name"),
        "admission_date": admission_date,
        "discharge_date": discharge_date,
        "length_of_stay": length_of_stay,
    }
    if drg_component is not None:
        admission["drg"] = {
            "code": drg_component.get("billing_code"),
            "description": drg_component.get("description"),
            "code_system": drg_component.get("code_system"),
        }
    return admission


def build_facility_claim(
    *,
    case_id: str,
    facility: dict[str, Any],
    line_items: Sequence[dict[str, Any]],
    case_components: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    facility_lines = [
        line_item for line_item in line_items if line_item_claim_class(line_item) == "facility"
    ]
    payment_basis = "line_item"
    if case_components:
        payment_basis = "drg" if any(
            component.get("code_system") in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS
            for component in case_components
        ) else "mixed"
    primary_drg = next(
        (
            component
            for component in case_components
            if component.get("code_system") in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS
        ),
        None,
    )
    claim_id = gold_bill_support.normalize_text(
        (primary_drg or {}).get("source_claim_id")
    ) or gold_bill_support.normalize_text(
        (facility_lines[0] if facility_lines else {}).get("source_claim_id")
    ) or f"{case_id}-facility"
    facility_claim = {
        "claim_id": claim_id,
        "claim_class": "facility",
        "payment_basis": payment_basis,
        "billing_provider": {
            "hospital_name": facility.get("hospital_name"),
            "npi": facility.get("npi"),
        },
        "charge_lines": facility_lines,
        "payment_components": list(case_components),
        "claim_totals": claim_accounting_summary(facility_lines, case_components),
    }
    if primary_drg is not None:
        facility_claim["drg"] = {
            "code": primary_drg.get("billing_code"),
            "description": primary_drg.get("description"),
            "code_system": primary_drg.get("code_system"),
        }
    return facility_claim


def build_professional_claims(
    *,
    case_id: str,
    line_items: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    professional_lines = [
        line_item for line_item in line_items if line_item_claim_class(line_item) == "professional"
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for index, line_item in enumerate(professional_lines, start=1):
        claim_id = gold_bill_support.normalize_text(line_item.get("source_claim_id")) or (
            f"{case_id}-professional-{index}"
        )
        if claim_id not in grouped:
            grouped[claim_id] = []
            order.append(claim_id)
        grouped[claim_id].append(line_item)
    return [
        {
            "claim_id": claim_id,
            "claim_class": "professional",
            "claim_type": gold_bill_support.normalize_text(
                grouped[claim_id][0].get("source_claim_type")
            ),
            "lines": grouped[claim_id],
            "claim_totals": claim_accounting_summary(grouped[claim_id]),
        }
        for claim_id in order
    ]


def build_line_items(
    procedure_occurrences: Sequence[dict[str, Any]],
    quote_by_code: dict[str, dict[str, Any]],
    *,
    session: Any,
    connection: Any,
    schema: TrilliantSchema,
    encounter: dict[str, Any],
    scenario: dict[str, Any],
    insurance_coverage_ratio: float,
    billed_charge_multiplier: float,
    source_state_value: str | None,
    max_quote_amount: float,
    cache_lock: Any,
    local_estimate_context: dict[str, Any] | None = None,
    logger: Any | None = None,
) -> list[dict[str, Any]]:
    line_items: list[dict[str, Any]] = []
    encounter_window = encounter.get("encounter") or {}
    encounter_regime = utils.encounter_regime(encounter) or "outpatient"
    for index, occurrence in enumerate(procedure_occurrences, start=1):
        billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
        code_system = utils.normalize_generated_code_system(occurrence.get("code_system"))
        if billing_code is None or code_system != "HCPCS/CPT":
            continue
        quote = resolve_clean_line_quote(
            occurrence,
            quote_by_code,
            session=session,
            connection=connection,
            schema=schema,
            scenario=scenario,
            encounter_regime=encounter_regime,
            source_state_value=source_state_value,
            max_quote_amount=max_quote_amount,
            cache_lock=cache_lock,
            local_estimate_context=local_estimate_context,
            logger=logger,
            case_id=gold_bill_support.normalize_text(encounter.get("cluster_id")),
        )
        accounting = compute_line_item_accounting(
            quote,
            scenario=scenario["scenario"],
            insurance_coverage_ratio=insurance_coverage_ratio,
            billed_charge_multiplier=billed_charge_multiplier,
        )
        service_date = (
            gold_bill_support.normalize_text(occurrence.get("service_date"))
            or gold_bill_support.normalize_text(encounter_window.get("from_date"))
        )
        line_item = {
            "line_id": f"{encounter['cluster_id']}-line-{index}",
            "billing_code": billing_code,
            "code_system": code_system,
            "description": gold_bill_support.normalize_text(quote.get("description")),
            "service_date": service_date,
            "quantity": 1,
            "support_status": occurrence_support_status(occurrence),
            "billing_side_hint": gold_bill_support.normalize_text(
                occurrence.get("billing_side_hint")
            ),
            "claim_class": occurrence_claim_class(occurrence),
            "source_claim_type": occurrence.get("source_claim_type"),
            "raw_source_claim_type": occurrence.get("raw_source_claim_type"),
            "source_claim_id": occurrence.get("source_claim_id"),
            "source_field": occurrence.get("source_field"),
            "source_line_number": occurrence.get("line_number"),
            "pricing_provenance": pricing_provenance(quote),
            "accounting": accounting,
        }
        if encounter_regime == "inpatient":
            line_item["impact_group_id"] = f"{encounter['cluster_id']}-line-{index}"
        line_items.append(line_item)
    return line_items


def build_case_components(
    procedure_occurrences: Sequence[dict[str, Any]],
    quote_by_code: dict[str, dict[str, Any]],
    *,
    encounter: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    scenario: dict[str, Any],
    insurance_coverage_ratio: float,
    billed_charge_multiplier: float,
    source_medicare_multiplier: float,
) -> list[dict[str, Any]]:
    components: list[dict[str, Any]] = []
    for index, occurrence in enumerate(procedure_occurrences, start=1):
        raw_billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
        code_system = utils.normalize_generated_code_system(occurrence.get("code_system"))
        canonical_code_system, canonical_billing_code = (
            canonicalize_inpatient_anchor_code_pair(code_system, raw_billing_code)
        )
        billing_code = canonical_billing_code or raw_billing_code
        code_system = canonical_code_system or code_system
        if billing_code is None or code_system == "HCPCS/CPT":
            continue
        quote = resolve_clean_component_quote(
            occurrence,
            quote_by_code,
            encounter=encounter,
            claim_index=claim_index,
            source_medicare_multiplier=source_medicare_multiplier,
        )
        accounting = compute_line_item_accounting(
            quote,
            scenario=scenario["scenario"],
            insurance_coverage_ratio=insurance_coverage_ratio,
            billed_charge_multiplier=billed_charge_multiplier,
        )
        component_id = f"{encounter['cluster_id']}-component-{index}"
        components.append(
            {
                "component_id": component_id,
                "component_type": code_system.lower().replace("-", "_"),
                "billing_code": billing_code,
                "code_system": code_system,
                "description": gold_bill_support.normalize_text(quote.get("description")),
                "linked_line_ids": [],
                "service_date": gold_bill_support.normalize_text(occurrence.get("service_date")),
                "impact_group_id": component_id,
                "source_claim_type": occurrence.get("source_claim_type"),
                "source_claim_id": occurrence.get("source_claim_id"),
                "source_field": occurrence.get("source_field"),
                "source_line_number": occurrence.get("line_number"),
                "pricing_provenance": pricing_provenance(quote),
                "accounting": accounting,
            }
        )
    return components


def resolve_clean_component_quote(
    occurrence: dict[str, Any],
    quote_by_code: dict[str, dict[str, Any]],
    *,
    encounter: dict[str, Any],
    claim_index: dict[str, dict[str, Any]],
    source_medicare_multiplier: float,
) -> dict[str, Any]:
    raw_billing_code = gold_bill_support.normalize_code(occurrence.get("code"))
    code_system = utils.normalize_generated_code_system(occurrence.get("code_system"))
    canonical_code_system, canonical_billing_code = canonicalize_inpatient_anchor_code_pair(
        code_system,
        raw_billing_code,
    )
    billing_code = canonical_billing_code or raw_billing_code
    code_system = canonical_code_system or code_system
    if billing_code is None or code_system is None or code_system == "HCPCS/CPT":
        raise RuntimeError("Missing pricing identity for inpatient component occurrence.")
    quote = quote_by_code.get(f"{code_system}:{billing_code}")
    if quote is not None:
        return quote
    if code_system not in constants.INPATIENT_DRG_FAMILY_CODE_SYSTEMS:
        raise RuntimeError(
            f"Missing price for payable inpatient component {code_system} {billing_code}."
        )
    return gold_bill_trilliant.build_component_imputed_rate_quote(
        occurrence,
        claim_index,
        encounter=encounter,
        source_medicare_multiplier=source_medicare_multiplier,
    )


def build_bill_from_generated_cluster_record(
    encounter: dict[str, Any],
    generated_cluster_record: dict[str, Any],
    *,
    context: PricingBuildContext,
) -> tuple[dict[str, Any], dict[str, int]]:
    if (
        context.client is None
        or context.session is None
        or context.trilliant_connection is None
        or context.trilliant_schema is None
    ):
        raise RuntimeError(
            "Pricing build context is missing a client, session, Trilliant connection, or schema."
        )

    usage = utils.usage_accumulator()
    encounter = utils.shift_encounter_dates(
        encounter,
        years=constants.DEFAULT_DATE_SHIFT_YEARS,
    )
    source_encounter = generated_cluster_record.get(
        "source_encounter"
    ) or cluster.build_source_encounter_payload(encounter)
    geography = generated_cluster_record.get("geography") or {}
    patient = generated_cluster_record.get("patient") or cluster.build_patient_payload(
        encounter,
        geography,
    )
    date_shift = generated_cluster_record.get("date_shift") or {
        "applied": bool(encounter.get("date_shift_years_applied")),
        "years": (
            constants.DEFAULT_DATE_SHIFT_YEARS
            if encounter.get("date_shift_years_applied")
            else 0
        ),
    }
    code_cluster_payload = generated_cluster_record.get("code_cluster") or {}
    if not code_cluster_payload:
        raise RuntimeError("Generated cluster record is missing code_cluster payload.")
    code_cluster = cluster.normalize_code_cluster_payload(code_cluster_payload)
    code_cluster = cluster.canonicalize_inpatient_anchor_codes(
        code_cluster,
        encounter=encounter,
    )

    source_state_value = utils.source_state_value_for_encounter(encounter, geography)
    diagnoses = cluster.build_generated_diagnoses(code_cluster, encounter)
    generated_procedure_occurrences = cluster.build_generated_procedure_occurrences(
        code_cluster,
        encounter,
    )
    encounter_regime = utils.encounter_regime(encounter) or "outpatient"
    claim_index = gold_bill_trilliant.source_claim_index(encounter)
    billing_codes = gold_bill_support.unique_preserving_order(
        occurrence["code"]
        for occurrence in generated_procedure_occurrences
        if (
            occurrence.get("code") is not None
            and utils.normalize_generated_code_system(occurrence.get("code_system"))
            == "HCPCS/CPT"
        )
    )
    component_codes = [
        occurrence
        for occurrence in generated_procedure_occurrences
        if utils.normalize_generated_code_system(occurrence.get("code_system")) != "HCPCS/CPT"
    ]

    scenario = gold_bill_trilliant.select_financial_scenario(
        encounter["cluster_id"],
        context.seed,
        context.extreme_case_probability,
        context.normal_payers,
        age_at_encounter=patient.get("age_at_encounter"),
    )

    with utils.acquire_pricing_gate(
        context.pricing_gate,
        logger=context.logger,
        throttle_metrics=context.throttle_metrics,
        throttle_metrics_lock=context.throttle_metrics_lock,
        case_id=encounter.get("cluster_id"),
        pricing_workers=context.pricing_workers,
    ):
        hospital_candidates, hospital_query_mode, hospital_cache_hit = (
            gold_bill_trilliant.get_hospital_candidates(
            context.trilliant_connection,
            context.trilliant_schema,
            session=context.session,
            zip_codes=list(geography.get("zip_codes") or []),
            county_fips=geography.get("county_fips"),
            county_name=geography.get("county_name"),
            state_value=source_state_value,
            max_candidates=context.max_hospital_candidates,
            cache_lock=context.trilliant_cache_lock,
            )
        )
        if context.trilliant_schema.source_mode == "public_directory":
            hospital, quote_by_code, component_quote_by_code, rate_telemetry = (
                gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage(
                    context.trilliant_connection,
                    context.trilliant_schema,
                    session=context.session,
                    case_id=encounter["cluster_id"],
                    seed=context.seed,
                    candidates=hospital_candidates,
                    billing_codes=billing_codes,
                    component_occurrences=component_codes,
                    scenario=scenario,
                    encounter_regime=encounter_regime,
                    zip_codes=list(geography.get("zip_codes") or []),
                    county_fips=geography.get("county_fips"),
                    local_hud_mapping=context.local_hud_mapping,
                    primary_clinical_direction=code_cluster.primary_clinical_direction,
                    source_state_value=source_state_value,
                    max_hospital_rate_probes=context.max_hospital_rate_probes,
                    max_quote_amount=context.max_quote_amount,
                    cache_lock=context.trilliant_cache_lock,
                )
            )
        else:
            hospital = gold_bill_trilliant.select_hospital_candidate(
                encounter["cluster_id"],
                context.seed,
                hospital_candidates,
            )
            quote_by_code, rate_telemetry = gold_bill_trilliant.resolve_rate_quotes(
                context.trilliant_connection,
                context.trilliant_schema,
                session=context.session,
                billing_codes=billing_codes,
                scenario=scenario,
                hospital=hospital,
                encounter_regime=encounter_regime,
                source_state_value=hospital.get("hospital_state") or source_state_value,
                max_quote_amount=context.max_quote_amount,
                cache_lock=context.trilliant_cache_lock,
            )
            component_quote_by_code = gold_bill_trilliant.resolve_component_rate_quotes(
                context.trilliant_connection,
                context.trilliant_schema,
                session=context.session,
                occurrences=component_codes,
                scenario=scenario,
                hospital=hospital,
                source_state_value=hospital.get("hospital_state") or source_state_value,
                max_quote_amount=context.max_quote_amount,
                cache_lock=context.trilliant_cache_lock,
            )

    payable_line_occurrences, payable_component_occurrences, has_payable_drg_component = (
        select_payable_occurrences(
            generated_procedure_occurrences,
            quote_by_code,
            component_quote_by_code,
            encounter_regime=encounter_regime,
        )
    )
    if encounter_regime == "inpatient" and not component_codes:
        raise RuntimeError("Inpatient bill is missing a facility anchor component.")
    if not payable_line_occurrences and not payable_component_occurrences:
        raise RuntimeError("No payable bill items remained after support/bundling filtering.")
    if (
        encounter_regime == "inpatient"
        and not has_payable_drg_component
        and component_codes
        and not payable_component_occurrences
    ):
        raise RuntimeError(
            "Missing selected-hospital quote for surviving inpatient facility anchors."
        )

    local_estimate_context: dict[str, Any] = {"quotes": {}, "misses": set()}
    resolve_case_hospital_local_estimate_quotes(
        payable_line_occurrences,
        quote_by_code,
        hospital=hospital,
        local_estimate_context=local_estimate_context,
        session=context.session,
        connection=context.trilliant_connection,
        schema=context.trilliant_schema,
        encounter=encounter,
        scenario=scenario,
        encounter_regime=encounter_regime,
        max_quote_amount=context.max_quote_amount,
        cache_lock=context.trilliant_cache_lock,
        logger=context.logger,
        hospital_candidates=hospital_candidates,
        geography=geography,
        local_hud_mapping=context.local_hud_mapping,
        primary_clinical_direction=code_cluster.primary_clinical_direction,
        seed=context.seed,
        max_hospital_rate_probes=context.max_hospital_rate_probes,
        max_nearby_fallback_probes=context.max_nearby_fallback_probes,
    )

    line_items = build_line_items(
        payable_line_occurrences,
        quote_by_code,
        session=context.session,
        connection=context.trilliant_connection,
        schema=context.trilliant_schema,
        encounter=encounter,
        scenario=scenario,
        insurance_coverage_ratio=context.insurance_coverage_ratio,
        billed_charge_multiplier=context.billed_charge_multiplier,
        source_state_value=source_state_value,
        max_quote_amount=context.max_quote_amount,
        cache_lock=context.trilliant_cache_lock,
        local_estimate_context=local_estimate_context,
        logger=context.logger,
    )
    case_components = build_case_components(
        payable_component_occurrences,
        component_quote_by_code,
        encounter=encounter,
        claim_index=claim_index,
        scenario=scenario,
        insurance_coverage_ratio=context.insurance_coverage_ratio,
        billed_charge_multiplier=context.billed_charge_multiplier,
        source_medicare_multiplier=context.source_medicare_multiplier,
    )
    if not line_items and not case_components:
        raise RuntimeError("Encounter did not produce any billable line items or case components.")

    bill_record = {
        "case_id": encounter["cluster_id"],
        "encounter_regime": encounter_regime,
        "schema_version": constants.GOLD_BILL_SCHEMA_VERSION,
        "generated_at": gold_bill_support.iso_timestamp(),
        "source_encounter": source_encounter,
        "patient": patient,
        "facility": {
            "hospital_name": hospital.get("hospital_name"),
            "npi": hospital.get("hospital_npi"),
            "address": hospital.get("hospital_address"),
            "zip_code": hospital.get("hospital_zip"),
            "state": hospital.get("hospital_state"),
            "city": hospital.get("hospital_city"),
            "county_name": hospital.get("hospital_county_name"),
            "county_fips": hospital.get("hospital_county_fips"),
            "selection_provenance": {
                "candidate_count": len(hospital_candidates),
                "query_mode": hospital_query_mode,
                "cache_hit": hospital_cache_hit,
                "trilliant_source_mode": context.trilliant_schema.source_mode,
                "hospital_directory_id": hospital.get("hospital_directory_id"),
                "hospital_page_url": hospital.get("hospital_page_url"),
                "duckdb_url": hospital.get("duckdb_url"),
                "selection_strategy": hospital.get("selection_strategy"),
                "quoted_code_coverage": hospital.get("quoted_code_coverage"),
                "selection_probe_count": hospital.get("selection_probe_count"),
                "selection_probe_limit": hospital.get("selection_probe_limit"),
                "selection_probe_rank": hospital.get("selection_probe_rank"),
                "selection_stage_candidate_count": hospital.get("selection_stage_candidate_count"),
                "drg_quote_count": hospital.get("drg_quote_count"),
                "other_anchor_quote_count": hospital.get("other_anchor_quote_count"),
                "service_quote_count": hospital.get("service_quote_count"),
                "total_quote_count": hospital.get("total_quote_count"),
                "selection_score": hospital.get("selection_score"),
                "geographic_stage": hospital.get("geographic_stage"),
                "specialty_filter_active": hospital.get("specialty_filter_active"),
                "same_state_unfiltered_fallback_invoked": hospital.get(
                    "same_state_unfiltered_fallback_invoked"
                ),
            },
        },
        "payer_scenario": {
            "scenario": scenario["scenario"],
            "payer_name": scenario.get("payer_name"),
            "insurance_coverage_ratio": context.insurance_coverage_ratio
            if scenario["scenario"] == "in_network"
            else 0.0,
        },
        "diagnoses": diagnoses,
        "line_items": line_items,
        "case_components": case_components,
        "bill_totals": bill_totals(line_items, case_components),
        "provenance": {
            "reference_sources": {
                "ssa_fips": constants.REFERENCE_NBER_SSA_FIPS_URL,
            },
            "date_shift": {
                "applied": bool(date_shift.get("applied")),
                "years": int(date_shift.get("years") or 0),
            },
            "cluster_generation": cluster.build_cluster_generation_provenance(
                encounter,
                code_cluster,
                diagnoses=diagnoses,
                generated_procedures=generated_procedure_occurrences,
                model=gold_bill_support.normalize_text(generated_cluster_record.get("model"))
                or context.model,
            ),
            "trilliant_schema": asdict(context.trilliant_schema),
            "pricing": {
                "billed_charge_multiplier": context.billed_charge_multiplier,
                "source_medicare_multiplier": context.source_medicare_multiplier,
                "max_quote_amount": context.max_quote_amount,
                "rate_cache_hits": rate_telemetry["cache_hits"],
                "rate_cache_misses": rate_telemetry["cache_misses"],
                "trilliant_source_mode": context.trilliant_schema.source_mode,
            },
        },
    }
    if encounter_regime == "inpatient":
        bill_record["admission"] = build_inpatient_admission(
            source_encounter,
            bill_record["facility"],
            case_components,
        )
        bill_record["facility_claim"] = build_facility_claim(
            case_id=encounter["cluster_id"],
            facility=bill_record["facility"],
            line_items=line_items,
            case_components=case_components,
        )
        bill_record["professional_claims"] = build_professional_claims(
            case_id=encounter["cluster_id"],
            line_items=line_items,
        )
        validation_errors = inpatient_pricing_validation_errors(bill_record)
        if validation_errors:
            raise RuntimeError(
                "Inpatient pricing QC failed: " + "; ".join(validation_errors)
            )

    description_usage = enrich_bill_descriptions(
        bill_record,
        client=context.client,
        model=context.model,
        reasoning_effort=context.reasoning_effort,
        request_delay_seconds=context.request_delay_seconds,
        description_cache=context.description_cache,
    )
    utils.add_usage(usage, description_usage)
    if encounter_regime == "inpatient":
        bill_record.pop("line_items", None)
        bill_record.pop("case_components", None)
    return bill_record, usage


def build_bill_for_encounter(
    encounter: dict[str, Any],
    *,
    cluster_context: ClusterGenerationContext,
    pricing_context: PricingBuildContext,
) -> tuple[dict[str, Any], dict[str, int]]:
    usage = utils.usage_accumulator()
    cluster_record, cluster_usage = cluster.build_generated_cluster_record_for_encounter(
        encounter,
        context=cluster_context,
    )
    utils.add_usage(usage, cluster_usage)
    bill_record, pricing_usage = build_bill_from_generated_cluster_record(
        encounter,
        cluster_record,
        context=pricing_context,
    )
    utils.add_usage(usage, pricing_usage)
    return bill_record, usage


def generate_aftercare_record_for_bill(
    client: Any,
    *,
    model: str,
    reasoning_effort: str,
    request_delay_seconds: float,
    bill_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    aftercare_payload, aftercare_usage = request_aftercare_summary(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        bill_record=bill_record,
    )
    utils.apply_request_delay(request_delay_seconds)
    aftercare_record = {
        "case_id": bill_record["case_id"],
        "generated_at": gold_bill_support.iso_timestamp(),
        "model": model,
        "summary": aftercare_payload,
    }
    return aftercare_record, utils.usage_totals(aftercare_usage)


def generate_bill_for_encounter(
    encounter: dict[str, Any],
    *,
    cluster_context: ClusterGenerationContext,
    pricing_context: PricingBuildContext,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, int]]:
    if cluster_context.client is None:
        raise RuntimeError("Cluster generation context is missing an OpenAI client.")
    bill_record, usage = build_bill_for_encounter(
        encounter,
        cluster_context=cluster_context,
        pricing_context=pricing_context,
    )
    aftercare_record, aftercare_usage = generate_aftercare_record_for_bill(
        cluster_context.client,
        model=cluster_context.model,
        reasoning_effort=cluster_context.reasoning_effort,
        request_delay_seconds=cluster_context.request_delay_seconds,
        bill_record=bill_record,
    )
    utils.add_usage(usage, aftercare_usage)
    return bill_record, aftercare_record, usage
