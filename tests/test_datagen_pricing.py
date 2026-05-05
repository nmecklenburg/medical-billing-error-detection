import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import duckdb

from claim.cli_logging import CliLogger
from claim.datagen import cluster as cluster_module
from claim.datagen.contracts import ClusterGenerationContext, DescriptionCacheStore, PricingBuildContext
from claim.datagen.llm import description_cache_key
from claim.datagen.pricing import (
    build_bill_from_generated_cluster_record,
    generate_bill_for_encounter,
    resolve_case_hospital_local_estimate_quotes,
    resolve_clean_line_quote,
    select_payable_occurrences,
)
from claim.gold_bill_trilliant import TrilliantSchema, ensure_cache_tables


class FakeResponses:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)

    def create(self, **_: object) -> object:
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            output_text=json.dumps(payload),
            usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            output=[],
        )


class FakeOpenAIClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.responses = FakeResponses(payloads)


class DatagenPricingTests(unittest.TestCase):
    def test_generate_bill_for_encounter_uses_current_package_pipeline_shape(self) -> None:
        encounter = {
            "cluster_id": "case-1",
            "desynpuf_id": "ABC123",
            "beneficiary": {
                "birth_date": "1980-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                    "amounts": {"claim_payment_amount": 50.0},
                }
            ],
            "diagnoses": [
                {
                    "code": "4019",
                    "code_system": "ICD9",
                    "source_claim_type": "outpatient",
                    "source_claim_id": "claim-1",
                    "source_field": "ICD9_DGNS_CD_1",
                }
            ],
            "procedure_codes": [
                {
                    "code": "99213",
                    "code_system": "HCPCS/CPT",
                    "source_claim_type": "outpatient",
                    "source_claim_id": "claim-1",
                    "source_field": "HCPCS_CD_1",
                    "line_number": 1,
                    "service_date": "2008-06-01",
                    "payment_amount": 40.0,
                }
            ],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "outpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-01",
            },
            "summary": {
                "unique_hcpcs_cpt_code_count": 1,
                "claim_types_present": ["outpatient"],
            },
            "sample_number": 1,
        }
        generated_cluster_record = {
            "case_id": "case-1",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "date_shift": {"applied": True, "years": 11},
            "code_cluster": {
                "primary_clinical_direction": "Outpatient hypertension follow-up",
                "clinical_direction_rationale": "Routine office visit.",
                "retained_input_codes": [
                    {
                        "code_system": "ICD9",
                        "code": "4019",
                        "reason": "Source diagnosis retained.",
                    },
                    {
                        "code_system": "HCPCS/CPT",
                        "code": "99213",
                        "reason": "Source office visit retained.",
                    },
                ],
                "filtered_input_codes": [],
                "generated_diagnoses": [
                    {
                        "selected_icd10_code": "I10",
                        "supporting_input_code": "4019",
                        "supporting_input_code_system": "ICD9",
                    }
                ],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "supporting_input_code": "99213",
                        "supporting_input_code_system": "HCPCS/CPT",
                        "service_date": "2008-06-01",
                        "billing_side_hint": "professional",
                    }
                ],
            },
        }
        client = FakeOpenAIClient(
            [
                {
                    "document_title": "After-Visit Summary",
                    "encounter_overview": "You were seen for hypertension follow-up.",
                    "diagnoses": ["Essential hypertension"],
                    "services_received": ["Established patient office visit"],
                    "home_care_instructions": ["Take medications as directed."],
                    "follow_up_actions": ["Follow up with your primary care clinician."],
                    "warning_signs": ["Seek care for worsening symptoms."],
                }
            ]
        )
        description_cache = DescriptionCacheStore(
            cache={
                description_cache_key("gpt-5-mini", "ICD10", "I10"): "Essential hypertension",
                description_cache_key("gpt-5-mini", "HCPCS/CPT", "99213"): "Established patient office visit",
            },
            model="gpt-5-mini",
        )
        schema = TrilliantSchema(
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
        )
        cluster_context = ClusterGenerationContext(
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            ssa_fips_mapping={},
            local_hud_mapping=None,
            reference_data_dir=Path("."),
            hud_user_token=None,
            logger=CliLogger("test", use_color=False),
            client=client,
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1_000_000.0,
            description_cache=description_cache,
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.cluster.build_generated_cluster_record_for_encounter",
                return_value=(
                    generated_cluster_record,
                    {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [
                        {
                            "hospital_name": "Example Hospital",
                            "hospital_state": "AL",
                            "hospital_zip": "36003",
                            "hospital_directory_id": "dir-1",
                        }
                    ],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_npi": "1234567890",
                        "hospital_address": "1 Main St",
                        "hospital_zip": "36003",
                        "hospital_state": "AL",
                        "hospital_city": "Prattville",
                        "hospital_county_name": "Autauga County",
                        "hospital_county_fips": "01001",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "coverage",
                        "quoted_code_coverage": 1,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 1,
                        "total_quote_count": 1,
                        "selection_score": [0, 0, 1, 1],
                    },
                    {
                        "99213": {
                            "description": "Established patient office visit",
                            "negotiated_rate": 100.0,
                            "gross_charge": 140.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        }
                    },
                    {},
                    {"cache_hits": 0, "cache_misses": 1},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_component_rate_quotes",
                return_value={},
            ),
        ):
            bill_record, aftercare_record, usage = generate_bill_for_encounter(
                encounter,
                cluster_context=cluster_context,
                pricing_context=pricing_context,
            )

        self.assertEqual(bill_record["case_id"], "case-1")
        self.assertEqual(bill_record["facility"]["hospital_name"], "Example Hospital")
        self.assertEqual(
            bill_record["source_encounter"]["encounter"]["from_date"],
            "2019-06-01",
        )
        self.assertEqual(bill_record["payer_scenario"]["scenario"], "in_network")
        self.assertEqual(bill_record["line_items"][0]["service_date"], "2019-06-01")
        self.assertEqual(bill_record["line_items"][0]["claim_class"], "carrier")
        self.assertEqual(bill_record["line_items"][0]["source_claim_type"], "carrier")
        self.assertEqual(bill_record["line_items"][0]["raw_source_claim_type"], "outpatient")
        self.assertEqual(bill_record["line_items"][0]["billing_side_hint"], "professional")
        self.assertEqual(
            bill_record["bill_totals"],
            {
                "line_item_count": 1,
                "billed_charge": 140.0,
                "contractual_adjustment": 40.0,
                "allowed_amount": 100.0,
                "insurance_paid": 80.0,
                "patient_responsibility": 20.0,
            },
        )
        self.assertEqual(aftercare_record["case_id"], "case-1")
        self.assertEqual(aftercare_record["summary"]["document_title"], "After-Visit Summary")
        self.assertEqual(usage, {"input_tokens": 13, "output_tokens": 8, "total_tokens": 21})

    def test_outpatient_raw_supported_facility_line_uses_source_observed_when_hospital_quote_missing(
        self,
    ) -> None:
        encounter = {
            "cluster_id": "case-source-observed",
            "beneficiary": {
                "birth_date": "1980-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                }
            ],
            "diagnoses": [{"code": "4019", "code_system": "ICD9"}],
            "procedure_codes": [
                {
                    "code": "99213",
                    "code_system": "HCPCS/CPT",
                    "source_claim_type": "outpatient",
                    "source_claim_id": "claim-1",
                    "source_field": "HCPCS_CD_1",
                    "line_number": 1,
                    "service_date": "2008-06-01",
                    "payment_amount": 40.0,
                }
            ],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "outpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-01",
            },
            "summary": {"claim_types_present": ["outpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-source-observed",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Outpatient follow-up",
                "clinical_direction_rationale": "Routine visit.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "I10"}],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "billing_side_hint": "facility",
                    }
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_state",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 0,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 1,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 0,
                        "total_quote_count": 0,
                        "selection_score": [0, 0, 0, 0],
                    },
                    {},
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        self.assertEqual(len(bill_record["line_items"]), 1)
        self.assertEqual(
            bill_record["line_items"][0]["pricing_provenance"]["lookup_level"],
            "source_observed",
        )
        self.assertEqual(bill_record["line_items"][0]["source_claim_type"], "outpatient")
        self.assertEqual(bill_record["line_items"][0]["billing_side_hint"], "facility")
        self.assertEqual(bill_record["line_items"][0]["accounting"]["allowed_amount"], 40.0)

    def test_outpatient_llm_augmented_hcpcs_can_bill_when_side_and_quotes_are_defensible(
        self,
    ) -> None:
        encounter = {
            "cluster_id": "case-outpatient-filtering",
            "beneficiary": {
                "birth_date": "1980-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                }
            ],
            "diagnoses": [{"code": "78900", "code_system": "ICD9"}],
            "procedure_codes": [
                {
                    "code": "99213",
                    "code_system": "HCPCS/CPT",
                    "source_claim_type": "outpatient",
                    "source_claim_id": "claim-1",
                    "source_field": "HCPCS_CD_1",
                    "line_number": 1,
                    "service_date": "2008-06-01",
                    "payment_amount": 80.0,
                }
            ],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "outpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-01",
            },
            "summary": {"claim_types_present": ["outpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-outpatient-filtering",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Outpatient abdominal pain evaluation",
                "clinical_direction_rationale": "Evaluation and discharge.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "R109"}],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "supporting_input_code": "99213",
                        "supporting_input_code_system": "HCPCS/CPT",
                        "billing_side_hint": "facility",
                    },
                    {
                        "billing_code": "80053",
                        "code_system": "HCPCS/CPT",
                        "billing_side_hint": "professional",
                    },
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_state",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 2,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 1,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 2,
                        "total_quote_count": 2,
                        "selection_score": [0, 0, 2, 2],
                    },
                    {
                        "99213": {
                            "description": "Established patient office visit",
                            "negotiated_rate": 100.0,
                            "gross_charge": 140.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        },
                        "80053": {
                            "description": "Comprehensive metabolic panel",
                            "negotiated_rate": 35.0,
                            "gross_charge": 50.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        },
                    },
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        self.assertEqual(
            [line["billing_code"] for line in bill_record["line_items"]],
            ["99213", "80053"],
        )
        line_by_code = {line["billing_code"]: line for line in bill_record["line_items"]}
        self.assertEqual(line_by_code["99213"]["support_status"], "raw_supported")
        self.assertEqual(line_by_code["99213"]["claim_class"], "facility")
        self.assertEqual(line_by_code["99213"]["source_claim_type"], "outpatient")
        self.assertEqual(line_by_code["80053"]["support_status"], "llm_augmented")
        self.assertEqual(line_by_code["80053"]["claim_class"], "carrier")
        self.assertEqual(line_by_code["80053"]["source_claim_type"], "carrier")

    def test_outpatient_raw_supported_carrier_provenance_can_bill_as_facility(self) -> None:
        encounter = {
            "cluster_id": "case-outpatient-side-flip-facility",
            "beneficiary": {
                "birth_date": "1980-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-outpatient",
                    "claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                },
                {
                    "claim_id": "claim-carrier",
                    "claim_type": "carrier",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                },
            ],
            "diagnoses": [{"code": "4019", "code_system": "ICD9"}],
            "procedure_codes": [
                {
                    "code": "99213",
                    "code_system": "HCPCS/CPT",
                    "source_claim_type": "carrier",
                    "source_claim_id": "claim-carrier",
                    "source_field": "HCPCS_CD_1",
                    "line_number": 1,
                    "service_date": "2008-06-01",
                }
            ],
            "encounter": {
                "anchor_claim_id": "claim-outpatient",
                "anchor_claim_type": "outpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-01",
            },
            "summary": {"claim_types_present": ["outpatient", "carrier"]},
        }
        generated_cluster_record = {
            "case_id": "case-outpatient-side-flip-facility",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Outpatient follow-up",
                "clinical_direction_rationale": "Facility-side test case.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "I10"}],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "supporting_input_code": "99213",
                        "supporting_input_code_system": "HCPCS/CPT",
                        "billing_side_hint": "facility",
                    }
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_state",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 1,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 1,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 1,
                        "total_quote_count": 1,
                        "selection_score": [0, 0, 1, 1],
                    },
                    {
                        "99213": {
                            "description": "Established patient office visit",
                            "negotiated_rate": 100.0,
                            "gross_charge": 140.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        }
                    },
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        self.assertEqual(len(bill_record["line_items"]), 1)
        self.assertEqual(bill_record["line_items"][0]["claim_class"], "facility")
        self.assertEqual(bill_record["line_items"][0]["source_claim_type"], "outpatient")
        self.assertEqual(bill_record["line_items"][0]["raw_source_claim_type"], "carrier")
        self.assertEqual(bill_record["line_items"][0]["billing_side_hint"], "facility")

    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quotes")
    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quote")
    def test_resolve_clean_line_quote_uses_hospital_local_estimate_for_in_network_professional_side(
        self,
        mock_resolve_trilliant_prior_quote: mock.Mock,
        mock_resolve_trilliant_prior_quotes: mock.Mock,
    ) -> None:
        local_estimate_context = {"quotes": {"99213": {
            "billing_code": "99213",
            "negotiated_rate": 95.0,
            "gross_charge": 150.0,
            "row_count": 1,
            "lookup_level": "hospital_local_estimate",
            "payer_name": None,
            "identity_filter": None,
            "state_value": None,
            "baseline_amount": 95.0,
            "estimate_basis": "family",
            "estimate_support_line_count": 2,
            "estimate_charge_field": "standard_charge_dollar",
            "estimate_ratio": 0.5,
        }}, "misses": set()}
        occurrence = {
            "code": "99213",
            "code_system": "HCPCS/CPT",
            "support_status": "llm_augmented",
            "claim_class": "carrier",
        }

        quote = resolve_clean_line_quote(
            occurrence,
            {},
            session=object(),
            connection=object(),
            schema=SimpleNamespace(source_mode="public_directory"),
            scenario={"scenario": "in_network", "payer_name": "Aetna"},
            encounter_regime="outpatient",
            source_state_value="AL",
            max_quote_amount=1_000_000.0,
            cache_lock=None,
            local_estimate_context=local_estimate_context,
        )

        self.assertEqual(quote["lookup_level"], "hospital_local_estimate")
        self.assertEqual(quote["negotiated_rate"], 95.0)
        mock_resolve_trilliant_prior_quote.assert_not_called()
        mock_resolve_trilliant_prior_quotes.assert_not_called()

    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quotes")
    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quote")
    def test_resolve_clean_line_quote_skips_local_estimate_for_facility_side_hcpcs(
        self,
        mock_resolve_trilliant_prior_quote: mock.Mock,
        mock_resolve_trilliant_prior_quotes: mock.Mock,
    ) -> None:
        occurrence = {
            "code": "99213",
            "code_system": "HCPCS/CPT",
            "support_status": "llm_augmented",
            "claim_class": "facility",
        }

        with self.assertRaisesRegex(RuntimeError, "Missing price for payable HCPCS/CPT 99213"):
            resolve_clean_line_quote(
                occurrence,
                {},
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                source_state_value="AL",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                local_estimate_context={
                    "quotes": {
                        "99213": {
                            "billing_code": "99213",
                            "negotiated_rate": 95.0,
                            "gross_charge": 150.0,
                            "lookup_level": "hospital_local_estimate",
                        }
                    },
                    "misses": set(),
                },
            )

        mock_resolve_trilliant_prior_quote.assert_not_called()
        mock_resolve_trilliant_prior_quotes.assert_not_called()

    def test_resolve_clean_line_quote_uses_hospital_local_estimate_for_self_pay_and_out_of_network_professional_side(
        self,
    ) -> None:
        occurrence = {
            "code": "99213",
            "code_system": "HCPCS/CPT",
            "support_status": "llm_augmented",
            "claim_class": "carrier",
        }
        with mock.patch(
            "claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quote",
        ) as mock_resolve_trilliant_prior_quote:
            for scenario_name in ("self_pay", "out_of_network"):
                quote = resolve_clean_line_quote(
                    occurrence,
                    {},
                    session=object(),
                    connection=object(),
                    schema=SimpleNamespace(source_mode="public_directory"),
                    scenario={"scenario": scenario_name, "payer_name": None},
                    encounter_regime="outpatient",
                    source_state_value="AL",
                    max_quote_amount=1_000_000.0,
                    cache_lock=None,
                    local_estimate_context={
                        "quotes": {
                            "99213": {
                                "billing_code": "99213",
                                "negotiated_rate": 95.0,
                                "gross_charge": 150.0,
                                "row_count": 1,
                                "lookup_level": "hospital_local_estimate",
                                "baseline_amount": 95.0,
                                "estimate_basis": "hospital",
                                "estimate_support_line_count": 1,
                                "estimate_charge_field": "gross_charge",
                                "estimate_ratio": 0.633333,
                            }
                        },
                        "misses": set(),
                    },
                )
                self.assertEqual(quote["lookup_level"], "hospital_local_estimate")
                self.assertEqual(quote["negotiated_rate"], 95.0)

        mock_resolve_trilliant_prior_quote.assert_not_called()

    def test_resolve_case_hospital_local_estimate_quotes_builds_same_family_estimate(self) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99213",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "99214",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "99283",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        with mock.patch(
            "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
            return_value={
                "99283": {
                    "billing_code": "99283",
                    "standard_charge_dollar": 500.0,
                    "row_count": 1,
                }
            },
        ) as mock_exact_rows, mock.patch(
            "claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quotes",
        ) as mock_resolve_trilliant_prior_quotes:
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {
                    "99213": {
                        "billing_code": "99213",
                        "negotiated_rate": 100.0,
                        "standard_charge_dollar": 200.0,
                        "lookup_level": "hospital_payer",
                    },
                    "99214": {
                        "billing_code": "99214",
                        "negotiated_rate": 150.0,
                        "standard_charge_dollar": 300.0,
                        "lookup_level": "hospital_average",
                    },
                },
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
            )

        quote = local_estimate_context["quotes"]["99283"]
        self.assertEqual(quote["lookup_level"], "hospital_local_estimate")
        self.assertEqual(quote["negotiated_rate"], 250.0)
        self.assertEqual(quote["gross_charge"], 500.0)
        self.assertEqual(quote["estimate_basis"], "family")
        self.assertEqual(quote["estimate_support_line_count"], 2)
        mock_exact_rows.assert_called_once()
        mock_resolve_trilliant_prior_quotes.assert_not_called()

    def test_resolve_case_hospital_local_estimate_quotes_falls_back_to_hospital_support(self) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "80053",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "99283",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        with mock.patch(
            "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
            return_value={
                "99283": {
                    "billing_code": "99283",
                    "minimum": 300.0,
                    "row_count": 1,
                }
            },
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {
                    "80053": {
                        "billing_code": "80053",
                        "negotiated_rate": 40.0,
                        "gross_charge": 100.0,
                        "lookup_level": "hospital_average",
                    }
                },
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
            )

        quote = local_estimate_context["quotes"]["99283"]
        self.assertEqual(quote["negotiated_rate"], 120.0)
        self.assertEqual(quote["gross_charge"], 300.0)
        self.assertEqual(quote["estimate_basis"], "hospital")

    def test_resolve_case_hospital_local_estimate_quotes_uses_nearby_direct_quote_when_selected_exact_missing(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "97112",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "97012",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        selected_hospital = {
            "hospital_directory_id": "selected-hospital",
            "hospital_zip": "54601",
            "hospital_state": "WI",
        }
        nearby_hospital = {
            "hospital_directory_id": "nearby-hospital",
            "hospital_name": "Nearby Medical Center",
            "hospital_zip": "54601",
            "hospital_state": "WI",
        }
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_rate_quotes",
                return_value=(
                    {
                        "97112": {
                            "billing_code": "97112",
                            "negotiated_rate": 110.0,
                            "gross_charge": 150.0,
                            "lookup_level": "hospital_average",
                        },
                        "97012": {
                            "billing_code": "97012",
                            "negotiated_rate": 67.0,
                            "gross_charge": 80.0,
                            "lookup_level": "hospital_average",
                        },
                    },
                    {"cache_hits": 0, "cache_misses": 2},
                ),
            ),
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital=selected_hospital,
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[selected_hospital, nearby_hospital],
                geography={"zip_codes": ["54601"], "state_abbrev": "WI"},
            )

        quote = local_estimate_context["quotes"]["97112"]
        self.assertEqual(quote["lookup_level"], "nearby_hospital_quote")
        self.assertEqual(quote["negotiated_rate"], 110.0)
        self.assertEqual(quote["estimate_basis"], "nearby_direct")
        self.assertEqual(quote["estimate_source_hospital_id"], "nearby-hospital")
        self.assertEqual(quote["estimate_scope"], "zip")

    def test_resolve_case_hospital_local_estimate_quotes_uses_nearby_exact_ratio_when_selected_exact_exists(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "27447",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        selected_hospital = {
            "hospital_directory_id": "selected-hospital",
            "hospital_zip": "60123",
            "hospital_state": "IL",
        }
        nearby_hospital = {
            "hospital_directory_id": "nearby-hospital",
            "hospital_name": "Nearby Medical Center",
            "hospital_zip": "60123",
            "hospital_state": "IL",
        }
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={
                    "27447": {
                        "billing_code": "27447",
                        "standard_charge_dollar": 1000.0,
                        "row_count": 1,
                    }
                },
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_rate_quotes",
                return_value=(
                    {
                        "27447": {
                            "billing_code": "27447",
                            "negotiated_rate": 300.0,
                            "gross_charge": 600.0,
                            "lookup_level": "hospital_average",
                        },
                        "99223": {
                            "billing_code": "99223",
                            "negotiated_rate": 120.0,
                            "gross_charge": 200.0,
                            "lookup_level": "hospital_average",
                        },
                    },
                    {"cache_hits": 0, "cache_misses": 2},
                ),
            ),
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital=selected_hospital,
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[selected_hospital, nearby_hospital],
                geography={"zip_codes": ["60123"], "state_abbrev": "IL"},
            )

        quote = local_estimate_context["quotes"]["27447"]
        self.assertEqual(quote["lookup_level"], "nearby_hospital_estimate")
        self.assertEqual(quote["negotiated_rate"], 500.0)
        self.assertEqual(quote["gross_charge"], 1000.0)
        self.assertEqual(quote["estimate_basis"], "nearby_exact")
        self.assertEqual(quote["estimate_source_hospital_id"], "nearby-hospital")
        self.assertEqual(quote["estimate_scope"], "zip")

    def test_resolve_case_hospital_local_estimate_quotes_can_reach_same_state_nearby_hospital(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        selected_hospital = {
            "hospital_directory_id": "selected-hospital",
            "hospital_zip": "30060",
            "hospital_state": "GA",
        }
        zip_nearby_hospital = {
            "hospital_directory_id": "zip-nearby",
            "hospital_name": "Zip Nearby",
            "hospital_zip": "30060",
            "hospital_state": "GA",
        }
        state_nearby_hospital = {
            "hospital_directory_id": "state-nearby",
            "hospital_name": "State Nearby",
            "hospital_zip": "30342",
            "hospital_state": "GA",
        }

        def resolve_rate_quotes_side_effect(
            _connection: object,
            _schema: object,
            *,
            hospital: dict[str, object],
            **_kwargs: object,
        ) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
            if hospital.get("hospital_directory_id") == "zip-nearby":
                return {}, {"cache_hits": 0, "cache_misses": 1}
            if hospital.get("hospital_directory_id") == "state-nearby":
                return (
                    {
                        "99223": {
                            "billing_code": "99223",
                            "negotiated_rate": 215.0,
                            "gross_charge": 300.0,
                            "lookup_level": "hospital_average",
                        }
                    },
                    {"cache_hits": 0, "cache_misses": 1},
                )
            raise AssertionError(f"Unexpected hospital probe: {hospital}")

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_rate_quotes",
                side_effect=resolve_rate_quotes_side_effect,
            ) as mock_resolve_rate_quotes,
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital=selected_hospital,
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[
                    selected_hospital,
                    zip_nearby_hospital,
                    state_nearby_hospital,
                ],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_nearby_fallback_probes=6,
            )

        quote = local_estimate_context["quotes"]["99223"]
        self.assertEqual(quote["lookup_level"], "nearby_hospital_quote")
        self.assertEqual(quote["negotiated_rate"], 215.0)
        self.assertEqual(quote["estimate_basis"], "nearby_direct")
        self.assertEqual(quote["estimate_source_hospital_id"], "state-nearby")
        self.assertEqual(quote["estimate_scope"], "same_state")
        self.assertEqual(mock_resolve_rate_quotes.call_count, 2)

    def test_resolve_case_hospital_local_estimate_quotes_rejects_non_public_directory_nearby_fallback(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        selected_hospital = {"hospital_directory_id": "selected-hospital"}
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.LOGGER.warning",
            ) as mock_warning,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Nearby-hospital fallback is deprecated and unsupported for non-public-directory Trilliant schemas.",
            ):
                resolve_case_hospital_local_estimate_quotes(
                    occurrences,
                    {},
                    hospital=selected_hospital,
                    local_estimate_context=local_estimate_context,
                    session=object(),
                    connection=object(),
                    schema=SimpleNamespace(source_mode="aggregate"),
                    encounter={"cluster_id": "case-1"},
                    scenario={"scenario": "in_network", "payer_name": "Aetna"},
                    encounter_regime="outpatient",
                    max_quote_amount=1_000_000.0,
                    cache_lock=None,
                    logger=None,
                    hospital_candidates=[selected_hospital, {"identity_fields": {"hospital_id": "nearby-1"}}],
                    max_hospital_rate_probes=2,
                )

        mock_warning.assert_called_once()

    def test_resolve_case_hospital_local_estimate_quotes_limits_nearby_probe_count_with_independent_budget(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        selected_hospital = {
            "hospital_directory_id": "selected-hospital",
            "hospital_zip": "30060",
            "hospital_state": "GA",
            "selection_probe_count": 1,
        }
        nearby_candidates = [
            {
                "hospital_directory_id": "nearby-1",
                "hospital_name": "Nearby 1",
                "hospital_zip": "30060",
                "hospital_state": "GA",
            },
            {
                "hospital_directory_id": "nearby-2",
                "hospital_name": "Nearby 2",
                "hospital_zip": "30060",
                "hospital_state": "GA",
            },
            {
                "hospital_directory_id": "nearby-3",
                "hospital_name": "Nearby 3",
                "hospital_zip": "30060",
                "hospital_state": "GA",
            },
        ]
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                return_value=[],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_family_quote_corpus",
                return_value=[],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_rate_quotes",
                return_value=({}, {"cache_hits": 0, "cache_misses": 1}),
            ) as mock_resolve_rate_quotes,
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital=selected_hospital,
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[selected_hospital, *nearby_candidates],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(mock_resolve_rate_quotes.call_count, 2)
        self.assertEqual(local_estimate_context["quotes"], {})
        self.assertEqual(local_estimate_context["misses"], {"99223"})

    def test_resolve_case_hospital_local_estimate_quotes_uses_cached_corpus_ratio_estimate(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        exact_quote = {
            "billing_code": "99223",
            "gross_charge": 500.0,
            "standard_charge_dollar": 500.0,
            "lookup_level": "hospital_exact",
        }
        cached_estimate = {
            "billing_code": "99223",
            "lookup_level": "cached_corpus_estimate",
            "negotiated_rate": 240.0,
            "gross_charge": 500.0,
            "estimate_basis": "cached_code_ratio",
            "estimate_scope": "national",
            "estimate_support_line_count": 6,
        }
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={"99223": exact_quote},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                return_value=[{"billing_code": "99223", "negotiated_rate": 220.0, "gross_charge": 440.0}],
            ) as mock_cached_corpus,
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote",
                return_value=cached_estimate,
            ) as mock_cached_estimate,
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_direct_quote",
            ) as mock_cached_direct,
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="inpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(local_estimate_context["quotes"]["99223"]["lookup_level"], "cached_corpus_estimate")
        self.assertEqual(local_estimate_context["misses"], set())
        self.assertEqual(mock_cached_corpus.call_count, 1)
        mock_cached_estimate.assert_called_once()
        mock_cached_direct.assert_not_called()

    def test_resolve_case_hospital_local_estimate_quotes_uses_cached_corpus_direct_quote(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        cached_direct_quote = {
            "billing_code": "99223",
            "lookup_level": "cached_corpus_quote",
            "negotiated_rate": 260.0,
            "gross_charge": 390.0,
            "estimate_basis": "cached_code_median",
            "estimate_scope": "national",
            "estimate_support_line_count": 9,
        }
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                return_value=[{"billing_code": "99223", "negotiated_rate": 260.0, "gross_charge": 390.0}],
            ) as mock_cached_corpus,
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote",
            ) as mock_cached_estimate,
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_direct_quote",
                return_value=cached_direct_quote,
            ) as mock_cached_direct,
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "out_of_network", "payer_name": None},
                encounter_regime="inpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(local_estimate_context["quotes"]["99223"]["lookup_level"], "cached_corpus_quote")
        self.assertEqual(local_estimate_context["misses"], set())
        self.assertEqual(mock_cached_corpus.call_count, 1)
        mock_cached_estimate.assert_not_called()
        mock_cached_direct.assert_called_once()

    def test_resolve_case_hospital_local_estimate_quotes_uses_resolved_support_for_remaining_exact_code(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "11042",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "11043",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        exact_quotes = {
            "11042": {
                "billing_code": "11042",
                "gross_charge": 400.0,
                "standard_charge_dollar": 400.0,
                "lookup_level": "hospital_exact",
            },
            "11043": {
                "billing_code": "11043",
                "gross_charge": 600.0,
                "standard_charge_dollar": 600.0,
                "lookup_level": "hospital_exact",
            },
        }
        cached_quote = {
            "billing_code": "11042",
            "lookup_level": "cached_corpus_quote",
            "negotiated_rate": 240.0,
            "gross_charge": 400.0,
            "estimate_basis": "cached_code_median",
            "estimate_scope": "national",
            "estimate_support_line_count": 4,
        }
        seeded_estimate = {
            "billing_code": "11043",
            "lookup_level": "hospital_local_estimate",
            "negotiated_rate": 360.0,
            "gross_charge": 600.0,
            "estimate_basis": "hospital",
            "estimate_support_line_count": 1,
        }
        def cached_corpus_side_effect(
            _connection: object,
            *,
            billing_code: str,
            **_kwargs: object,
        ) -> list[dict[str, object]]:
            if billing_code == "11042":
                return [{"billing_code": "11042", "negotiated_rate": 240.0, "gross_charge": 400.0}]
            return []

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value=exact_quotes,
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                side_effect=cached_corpus_side_effect,
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote",
                return_value=None,
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_direct_quote",
                side_effect=[cached_quote, None],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_hospital_local_estimate_quote",
                return_value=seeded_estimate,
            ) as mock_seeded_estimate,
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="inpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(local_estimate_context["quotes"]["11042"]["lookup_level"], "cached_corpus_quote")
        self.assertEqual(
            local_estimate_context["quotes"]["11043"]["estimate_basis"],
            "resolved_support",
        )
        self.assertEqual(local_estimate_context["misses"], set())
        mock_seeded_estimate.assert_called_once()

    def test_resolve_case_hospital_local_estimate_quotes_uses_cached_family_direct_quote(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "72020",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            }
        ]
        family_quote = {
            "billing_code": "72020",
            "lookup_level": "cached_corpus_quote",
            "negotiated_rate": 180.0,
            "gross_charge": 250.0,
            "estimate_basis": "cached_family_median",
            "estimate_scope": "national",
            "estimate_support_line_count": 5,
        }
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                return_value=[],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_direct_quote",
                return_value=family_quote,
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_family_quote_corpus",
                return_value=[{"billing_code": "72050", "negotiated_rate": 180.0, "gross_charge": 250.0}],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote",
                return_value=None,
            ),
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "self_pay", "payer_name": None},
                encounter_regime="inpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(local_estimate_context["quotes"]["72020"]["estimate_basis"], "cached_family_median")
        self.assertEqual(local_estimate_context["misses"], set())

    def test_resolve_case_hospital_local_estimate_quotes_uses_anesthesia_share_estimate(
        self,
    ) -> None:
        local_estimate_context: dict[str, object] = {"quotes": {}, "misses": set()}
        occurrences = [
            {
                "code": "27447",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "01402",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
        ]
        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_selected_hospital_exact_professional_quotes",
                return_value={},
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_quote_corpus",
                side_effect=lambda *_args, **kwargs: (
                    [{"billing_code": "27447", "negotiated_rate": 5000.0, "gross_charge": 7000.0}]
                    if kwargs.get("billing_code") == "27447"
                    else []
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_ratio_estimate_quote",
                return_value=None,
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.build_cached_corpus_direct_quote",
                side_effect=[
                    {
                        "billing_code": "27447",
                        "lookup_level": "cached_corpus_quote",
                        "negotiated_rate": 5000.0,
                        "gross_charge": 7000.0,
                        "estimate_basis": "cached_code_median",
                        "estimate_scope": "national",
                        "estimate_support_line_count": 3,
                    },
                    None,
                ],
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.cached_professional_family_quote_corpus",
                return_value=[],
            ),
        ):
            resolve_case_hospital_local_estimate_quotes(
                occurrences,
                {},
                hospital={"hospital_directory_id": "selected-hospital"},
                local_estimate_context=local_estimate_context,
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                encounter={"cluster_id": "case-1"},
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="inpatient",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                logger=None,
                hospital_candidates=[],
                geography={"zip_codes": ["30060"], "state_abbrev": "GA"},
                max_hospital_rate_probes=2,
                max_nearby_fallback_probes=2,
            )

        self.assertEqual(local_estimate_context["quotes"]["01402"]["estimate_basis"], "anesthesia_share")
        self.assertEqual(local_estimate_context["quotes"]["01402"]["negotiated_rate"], 500.0)
        self.assertEqual(local_estimate_context["misses"], set())


    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quotes")
    @mock.patch("claim.datagen.pricing.gold_bill_trilliant.resolve_trilliant_prior_quote")
    def test_resolve_clean_line_quote_does_not_call_trilliant_prior_when_local_estimate_is_missing(
        self,
        mock_resolve_trilliant_prior_quote: mock.Mock,
        mock_resolve_trilliant_prior_quotes: mock.Mock,
    ) -> None:
        occurrence = {
            "code": "99213",
            "code_system": "HCPCS/CPT",
            "support_status": "llm_augmented",
            "claim_class": "carrier",
        }

        with self.assertRaisesRegex(RuntimeError, "Missing price for payable HCPCS/CPT 99213"):
            resolve_clean_line_quote(
                occurrence,
                {},
                session=object(),
                connection=object(),
                schema=SimpleNamespace(source_mode="public_directory"),
                scenario={"scenario": "in_network", "payer_name": "Aetna"},
                encounter_regime="outpatient",
                source_state_value="AL",
                max_quote_amount=1_000_000.0,
                cache_lock=None,
                local_estimate_context={"quotes": {}, "misses": {"99213"}},
            )

        mock_resolve_trilliant_prior_quote.assert_not_called()
        mock_resolve_trilliant_prior_quotes.assert_not_called()

    def test_select_payable_occurrences_outpatient_requires_pricing_path_for_facility_side(
        self,
    ) -> None:
        payable_lines, payable_components, has_payable_drg_component = select_payable_occurrences(
            [
                {
                    "code": "99213",
                    "code_system": "HCPCS/CPT",
                    "support_status": "raw_supported",
                    "claim_class": "carrier",
                },
                {
                    "code": "80053",
                    "code_system": "HCPCS/CPT",
                    "support_status": "llm_augmented",
                    "claim_class": "facility",
                },
                {
                    "code": "71045",
                    "code_system": "HCPCS/CPT",
                    "support_status": "raw_supported",
                    "claim_class": "facility",
                    "payment_amount": 22.0,
                },
                {
                    "code": "36415",
                    "code_system": "HCPCS/CPT",
                    "support_status": "llm_augmented",
                    "claim_class": "unknown",
                },
            ],
            {"80053": {"lookup_level": "hospital_payer"}},
            {},
            encounter_regime="outpatient",
        )

        self.assertEqual([occurrence["code"] for occurrence in payable_lines], ["99213", "80053", "71045"])
        self.assertEqual(payable_components, [])
        self.assertFalse(has_payable_drg_component)

    def test_outpatient_llm_augmented_facility_hcpcs_without_quote_is_dropped(self) -> None:
        payable_lines, payable_components, has_payable_drg_component = select_payable_occurrences(
            [
                {
                    "code": "80053",
                    "code_system": "HCPCS/CPT",
                    "support_status": "llm_augmented",
                    "claim_class": "facility",
                }
            ],
            {},
            {},
            encounter_regime="outpatient",
        )

        self.assertEqual(payable_lines, [])
        self.assertEqual(payable_components, [])
        self.assertFalse(has_payable_drg_component)

    def test_select_payable_occurrences_inpatient_drg_suppresses_facility_side_hcpcs(self) -> None:
        payable_lines, payable_components, has_payable_drg_component = select_payable_occurrences(
            [
                {
                    "code": "99223",
                    "code_system": "HCPCS/CPT",
                    "support_status": "raw_supported",
                    "claim_class": "carrier",
                },
                {
                    "code": "71045",
                    "code_system": "HCPCS/CPT",
                    "support_status": "llm_augmented",
                    "claim_class": "facility",
                },
                {
                    "code": "331",
                    "code_system": "MS-DRG",
                },
            ],
            {},
            {},
            encounter_regime="inpatient",
        )

        self.assertEqual([occurrence["code"] for occurrence in payable_lines], ["99223"])
        self.assertEqual([occurrence["code"] for occurrence in payable_components], ["331"])
        self.assertTrue(has_payable_drg_component)

    def test_select_payable_occurrences_inpatient_non_drg_facility_side_requires_direct_quote(
        self,
    ) -> None:
        occurrences = [
            {
                "code": "99223",
                "code_system": "HCPCS/CPT",
                "support_status": "llm_augmented",
                "claim_class": "carrier",
            },
            {
                "code": "71045",
                "code_system": "HCPCS/CPT",
                "support_status": "raw_supported",
                "claim_class": "facility",
            },
            {
                "code": "0250",
                "code_system": "REVENUE_CODE",
            },
        ]

        payable_lines, payable_components, has_payable_drg_component = select_payable_occurrences(
            occurrences,
            {"71045": {"lookup_level": "hospital_payer"}},
            {"REVENUE_CODE:0250": {"lookup_level": "hospital_payer"}},
            encounter_regime="inpatient",
        )

        self.assertEqual([occurrence["code"] for occurrence in payable_lines], ["99223", "71045"])
        self.assertEqual([occurrence["code"] for occurrence in payable_components], ["0250"])
        self.assertFalse(has_payable_drg_component)

        payable_lines_without_quote, _, _ = select_payable_occurrences(
            occurrences,
            {},
            {"REVENUE_CODE:0250": {"lookup_level": "hospital_payer"}},
            encounter_regime="inpatient",
        )
        self.assertEqual(
            [occurrence["code"] for occurrence in payable_lines_without_quote],
            ["99223"],
        )

    def test_generated_outpatient_occurrence_uses_llm_billing_side_and_flags_conflict(self) -> None:
        code_cluster = cluster_module.normalize_code_cluster_payload(
            {
                "primary_clinical_direction": "Outpatient hypertension follow-up",
                "clinical_direction_rationale": "Routine visit.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "I10"}],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "supporting_input_code": "99213",
                        "supporting_input_code_system": "HCPCS/CPT",
                        "billing_side_hint": "professional",
                    }
                ],
            }
        )

        occurrences = cluster_module.build_generated_procedure_occurrences(
            code_cluster,
            {
                "encounter": {
                    "anchor_claim_id": "claim-1",
                    "anchor_claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                },
                "procedure_codes": [
                    {
                        "code": "99213",
                        "code_system": "HCPCS/CPT",
                        "source_claim_type": "outpatient",
                        "source_claim_id": "claim-1",
                        "source_field": "HCPCS_CD_1",
                        "line_number": 1,
                        "service_date": "2008-06-01",
                    }
                ],
            },
        )

        self.assertEqual(occurrences[0]["support_status"], "raw_supported")
        self.assertEqual(occurrences[0]["claim_class"], "carrier")
        self.assertEqual(occurrences[0]["billing_side_hint"], "professional")
        self.assertEqual(occurrences[0]["source_claim_type"], "carrier")
        self.assertEqual(occurrences[0]["raw_source_claim_type"], "outpatient")

    def test_generated_outpatient_occurrence_uses_facility_source_claim_type_when_side_flipped(
        self,
    ) -> None:
        code_cluster = cluster_module.normalize_code_cluster_payload(
            {
                "primary_clinical_direction": "Outpatient follow-up",
                "clinical_direction_rationale": "Facility test.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "I10"}],
                "generated_line_items": [
                    {
                        "billing_code": "99213",
                        "code_system": "HCPCS/CPT",
                        "supporting_input_code": "99213",
                        "supporting_input_code_system": "HCPCS/CPT",
                        "billing_side_hint": "facility",
                    }
                ],
            }
        )

        occurrences = cluster_module.build_generated_procedure_occurrences(
            code_cluster,
            {
                "encounter": {
                    "anchor_claim_id": "claim-1",
                    "anchor_claim_type": "outpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-01",
                },
                "procedure_codes": [
                    {
                        "code": "99213",
                        "code_system": "HCPCS/CPT",
                        "source_claim_type": "carrier",
                        "source_claim_id": "claim-1",
                        "source_field": "HCPCS_CD_1",
                        "line_number": 1,
                        "service_date": "2008-06-01",
                    }
                ],
            },
        )

        self.assertEqual(occurrences[0]["claim_class"], "facility")
        self.assertEqual(occurrences[0]["source_claim_type"], "outpatient")
        self.assertEqual(occurrences[0]["raw_source_claim_type"], "carrier")

    def test_generated_inpatient_anchor_without_raw_match_inherits_anchor_claim_linkage(self) -> None:
        code_cluster = cluster_module.normalize_code_cluster_payload(
            {
                "primary_clinical_direction": "Inpatient bowel surgery",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "K565"}],
                "generated_line_items": [
                    {"billing_code": "331", "code_system": "MS-DRG"}
                ],
            }
        )

        occurrences = cluster_module.build_generated_procedure_occurrences(
            code_cluster,
            {
                "encounter": {
                    "anchor_claim_id": "claim-1",
                    "anchor_claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-03",
                },
                "procedure_codes": [],
            },
        )

        self.assertEqual(occurrences[0]["source_claim_type"], "inpatient")
        self.assertEqual(occurrences[0]["source_claim_id"], "claim-1")
        self.assertEqual(occurrences[0]["source_field"], "ENCOUNTER_ANCHOR_CLAIM")
        self.assertIsNone(occurrences[0]["line_number"])

    def test_only_primary_synthetic_inpatient_anchor_inherits_anchor_claim_linkage(self) -> None:
        code_cluster = cluster_module.normalize_code_cluster_payload(
            {
                "primary_clinical_direction": "Inpatient dehydration admission",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "E860"}],
                "generated_line_items": [
                    {"billing_code": "0250", "code_system": "REVENUE_CODE"},
                    {"billing_code": "331", "code_system": "MS-DRG"},
                ],
            }
        )

        occurrences = cluster_module.build_generated_procedure_occurrences(
            code_cluster,
            {
                "encounter": {
                    "anchor_claim_id": "claim-1",
                    "anchor_claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-03",
                },
                "procedure_codes": [],
            },
        )

        revenue_occurrence = next(
            occurrence for occurrence in occurrences if occurrence["code_system"] == "REVENUE_CODE"
        )
        drg_occurrence = next(
            occurrence for occurrence in occurrences if occurrence["code_system"] == "MS-DRG"
        )
        self.assertIsNone(revenue_occurrence["source_claim_id"])
        self.assertIsNone(revenue_occurrence["source_field"])
        self.assertEqual(drg_occurrence["source_claim_id"], "claim-1")
        self.assertEqual(drg_occurrence["source_field"], "ENCOUNTER_ANCHOR_CLAIM")

    def test_inpatient_public_directory_reuses_selected_component_quotes(self) -> None:
        encounter = {
            "cluster_id": "case-inpatient-reuse",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-03",
                    "amounts": {"claim_payment_amount": 4000.0},
                }
            ],
            "diagnoses": [{"code": "5609", "code_system": "ICD9"}],
            "procedure_codes": [],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "inpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-03",
            },
            "summary": {"claim_types_present": ["inpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-inpatient-reuse",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Inpatient bowel surgery",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "K565"}],
                "generated_line_items": [
                    {"billing_code": "331", "code_system": "MS-DRG"}
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_npi": "1234567890",
                        "hospital_address": "1 Main St",
                        "hospital_zip": "36003",
                        "hospital_state": "AL",
                        "hospital_city": "Prattville",
                        "hospital_county_name": "Autauga County",
                        "hospital_county_fips": "01001",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 1,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 1,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 0,
                        "total_quote_count": 1,
                        "selection_score": [1, 0, 0, 1],
                    },
                    {},
                    {
                        "MS-DRG:331": {
                            "billing_code": "331",
                            "description": "Major bowel procedure",
                            "negotiated_rate": 8000.0,
                            "gross_charge": 12000.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        }
                    },
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.resolve_component_rate_quotes",
            ) as mock_component_quotes,
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        mock_component_quotes.assert_not_called()
        component = bill_record["facility_claim"]["payment_components"][0]
        self.assertEqual(component["accounting"]["allowed_amount"], 8000.0)
        self.assertEqual(component["pricing_provenance"]["lookup_level"], "hospital_payer")
        self.assertEqual(
            bill_record["facility"]["selection_provenance"]["selection_score"],
            [1, 0, 0, 1],
        )
        self.assertEqual(usage, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})

    def test_pricing_heals_prefixed_inpatient_cluster_anchor_before_lookup_and_write(self) -> None:
        encounter = {
            "cluster_id": "case-inpatient-prefixed-heal",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-03",
                    "amounts": {"claim_payment_amount": 4000.0},
                }
            ],
            "diagnoses": [{"code": "5609", "code_system": "ICD9"}],
            "procedure_codes": [],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "inpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-03",
            },
            "summary": {"claim_types_present": ["inpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-inpatient-prefixed-heal",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Inpatient bowel surgery",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "K565"}],
                "generated_line_items": [
                    {"billing_code": "MSDRG331", "code_system": "MS-DRG"}
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 1,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 1,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 0,
                        "total_quote_count": 1,
                        "selection_score": [1, 0, 0, 1],
                    },
                    {},
                    {
                        "MS-DRG:331": {
                            "billing_code": "331",
                            "description": "Major bowel procedure",
                            "negotiated_rate": 8000.0,
                            "gross_charge": 12000.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        }
                    },
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ) as mock_select_hospital,
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        self.assertEqual(
            mock_select_hospital.call_args.kwargs["component_occurrences"][0]["code"],
            "331",
        )
        component = bill_record["facility_claim"]["payment_components"][0]
        self.assertEqual(component["billing_code"], "331")
        self.assertEqual(component["pricing_provenance"]["lookup_level"], "hospital_payer")

    def test_inpatient_anchorless_bill_fails_before_write(self) -> None:
        encounter = {
            "cluster_id": "case-inpatient-anchorless",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-02",
                    "amounts": {"claim_payment_amount": 4000.0},
                }
            ],
            "diagnoses": [{"code": "486", "code_system": "ICD9"}],
            "procedure_codes": [],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "inpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-02",
            },
            "summary": {"claim_types_present": ["inpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-inpatient-anchorless",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Inpatient pneumonia admission",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "J189"}],
                "generated_line_items": [
                    {
                        "billing_code": "99223",
                        "code_system": "HCPCS/CPT",
                        "billing_side_hint": "professional",
                    }
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 1,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 1,
                        "total_quote_count": 1,
                        "selection_score": [0, 0, 1, 1],
                    },
                    {
                        "99223": {
                            "description": "Initial hospital care",
                            "negotiated_rate": 100.0,
                            "gross_charge": 135.0,
                            "lookup_level": "hospital_payer",
                            "payer_name": "Aetna",
                            "row_count": 1,
                            "identity_filter": None,
                            "state_value": "AL",
                            "baseline_amount": None,
                        }
                    },
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "missing a facility anchor component"):
                build_bill_from_generated_cluster_record(
                    encounter,
                    generated_cluster_record,
                    context=pricing_context,
                )

    def test_inpatient_anchor_fallback_uses_source_claim_payment_or_floor(self) -> None:
        encounter = {
            "cluster_id": "case-inpatient-fallback",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-04",
                    "amounts": {"claim_payment_amount": 2500.0},
                }
            ],
            "diagnoses": [{"code": "5609", "code_system": "ICD9"}],
            "procedure_codes": [],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "inpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-04",
            },
            "summary": {"claim_types_present": ["inpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-inpatient-fallback",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Inpatient bowel surgery",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "K565"}],
                "generated_line_items": [
                    {"billing_code": "MSDRG331", "code_system": "MS-DRG"}
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 0,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 0,
                        "total_quote_count": 0,
                        "selection_score": [0, 0, 0, 0],
                    },
                    {},
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        component = bill_record["facility_claim"]["payment_components"][0]
        self.assertEqual(component["billing_code"], "331")
        self.assertEqual(component["pricing_provenance"]["lookup_level"], "inpatient_anchor_imputed")
        self.assertEqual(component["pricing_provenance"]["source_allowed_amount"], 2500.0)
        self.assertEqual(component["pricing_provenance"]["floor_amount"], 3000.0)
        self.assertTrue(component["pricing_provenance"]["floor_applied"])
        self.assertEqual(component["accounting"]["allowed_amount"], 3000.0)

    def test_inpatient_multiple_synthetic_anchors_do_not_double_charge_anchor_claim_payment(self) -> None:
        encounter = {
            "cluster_id": "case-inpatient-multi-anchor",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "sex_code": "2",
                "race_code": "1",
                "state_code": "01",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim_type": "inpatient",
                    "from_date": "2008-06-01",
                    "thru_date": "2008-06-04",
                    "amounts": {"claim_payment_amount": 2500.0},
                }
            ],
            "diagnoses": [{"code": "27651", "code_system": "ICD9"}],
            "procedure_codes": [],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "inpatient",
                "from_date": "2008-06-01",
                "thru_date": "2008-06-04",
            },
            "summary": {"claim_types_present": ["inpatient"]},
        }
        generated_cluster_record = {
            "case_id": "case-inpatient-multi-anchor",
            "model": "gpt-5-mini",
            "geography": {
                "zip_codes": ["36003"],
                "county_fips": "01001",
                "county_name": "Autauga County",
                "state_abbrev": "AL",
                "state_name": "Alabama",
            },
            "code_cluster": {
                "primary_clinical_direction": "Inpatient dehydration admission",
                "clinical_direction_rationale": "Facility stay.",
                "retained_input_codes": [],
                "filtered_input_codes": [],
                "generated_diagnoses": [{"selected_icd10_code": "E860"}],
                "generated_line_items": [
                    {"billing_code": "0250", "code_system": "REVENUE_CODE"},
                    {"billing_code": "331", "code_system": "MS-DRG"},
                ],
            },
        }
        schema = TrilliantSchema(
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
        )
        pricing_context = PricingBuildContext(
            seed=0,
            model="openai:gpt-5-mini",
            reasoning_effort="low",
            request_delay_seconds=0.0,
            normal_payers=["Aetna"],
            extreme_case_probability=0.0,
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
            source_medicare_multiplier=2.5,
            max_hospital_candidates=10,
            max_hospital_rate_probes=2,
            max_quote_amount=1000000.0,
            description_cache=DescriptionCacheStore(cache={}, model="gpt-5-mini"),
            logger=CliLogger("test", use_color=False),
            client=SimpleNamespace(),
            session=SimpleNamespace(),
            trilliant_connection=SimpleNamespace(),
            trilliant_schema=schema,
        )

        with (
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.get_hospital_candidates",
                return_value=(
                    [{"hospital_name": "Example Hospital", "hospital_state": "AL", "hospital_directory_id": "dir-1"}],
                    "directory_zip",
                    False,
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.gold_bill_trilliant.select_public_directory_hospital_with_rate_coverage",
                return_value=(
                    {
                        "hospital_name": "Example Hospital",
                        "hospital_state": "AL",
                        "hospital_directory_id": "dir-1",
                        "selection_strategy": "rate_coverage",
                        "quoted_code_coverage": 0,
                        "selection_probe_count": 1,
                        "selection_probe_limit": 2,
                        "selection_probe_rank": 1,
                        "drg_quote_count": 0,
                        "other_anchor_quote_count": 0,
                        "service_quote_count": 0,
                        "total_quote_count": 0,
                        "selection_score": [0, 0, 0, 0],
                    },
                    {},
                    {},
                    {"cache_hits": 0, "cache_misses": 0},
                ),
            ),
            mock.patch(
                "claim.datagen.pricing.enrich_bill_descriptions",
                return_value={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            ),
        ):
            bill_record, _usage = build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=pricing_context,
            )

        component_by_system = {
            component["code_system"]: component
            for component in bill_record["facility_claim"]["payment_components"]
        }
        self.assertEqual(component_by_system["MS-DRG"]["source_claim_id"], "claim-1")
        self.assertEqual(component_by_system["MS-DRG"]["accounting"]["allowed_amount"], 3000.0)
        self.assertNotIn("REVENUE_CODE", component_by_system)
        self.assertEqual(bill_record["bill_totals"]["allowed_amount"], 3000.0)


if __name__ == "__main__":
    unittest.main()
