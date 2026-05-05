import unittest

from claim.datagen.pipeline import pricing_fallback_log_fields


class DatagenPipelineTests(unittest.TestCase):
    def test_pricing_fallback_log_fields_counts_hospital_local_estimate_usage(self) -> None:
        bill_record = {
            "case_id": "case-1",
            "facility": {
                "selection_provenance": {
                    "geographic_stage": "zip",
                    "candidate_count": 5,
                    "selection_stage_candidate_count": 2,
                    "selection_probe_count": 1,
                    "selection_probe_limit": 1,
                    "selection_probe_rank": 1,
                    "quoted_code_coverage": 1,
                    "same_state_unfiltered_fallback_invoked": False,
                }
            },
            "provenance": {
                "pricing": {
                    "rate_cache_hits": 3,
                    "rate_cache_misses": 4,
                }
            },
            "line_items": [
                {
                    "billing_code": "99213",
                    "pricing_provenance": {
                        "lookup_level": "hospital_local_estimate",
                    },
                }
            ],
            "case_components": [],
        }

        fallback_fields = pricing_fallback_log_fields(bill_record)

        self.assertIsNotNone(fallback_fields)
        assert fallback_fields is not None
        self.assertEqual(fallback_fields["hospital_local_estimate_lines"], 1)
        self.assertEqual(fallback_fields["nearby_hospital_estimate_lines"], 0)
        self.assertEqual(fallback_fields["nearby_hospital_quote_lines"], 0)
        self.assertEqual(fallback_fields["source_observed_lines"], 0)
        self.assertEqual(fallback_fields["trilliant_prior_lines"], 0)

    def test_pricing_fallback_log_fields_counts_nearby_hospital_usage(self) -> None:
        bill_record = {
            "case_id": "case-nearby",
            "facility": {
                "selection_provenance": {
                    "geographic_stage": "zip",
                    "selection_probe_count": 1,
                    "same_state_unfiltered_fallback_invoked": False,
                }
            },
            "provenance": {"pricing": {}},
            "line_items": [
                {
                    "billing_code": "27447",
                    "pricing_provenance": {
                        "lookup_level": "nearby_hospital_estimate",
                    },
                },
                {
                    "billing_code": "97112",
                    "pricing_provenance": {
                        "lookup_level": "nearby_hospital_quote",
                    },
                },
            ],
            "case_components": [],
        }

        fallback_fields = pricing_fallback_log_fields(bill_record)

        self.assertIsNotNone(fallback_fields)
        assert fallback_fields is not None
        self.assertEqual(fallback_fields["nearby_hospital_estimate_lines"], 1)
        self.assertEqual(fallback_fields["nearby_hospital_quote_lines"], 1)

    def test_pricing_fallback_log_fields_counts_cached_corpus_usage(self) -> None:
        bill_record = {
            "case_id": "case-cached",
            "facility": {
                "selection_provenance": {
                    "geographic_stage": "zip",
                    "selection_probe_count": 1,
                    "same_state_unfiltered_fallback_invoked": False,
                }
            },
            "provenance": {"pricing": {}},
            "line_items": [
                {
                    "billing_code": "27447",
                    "pricing_provenance": {
                        "lookup_level": "cached_corpus_estimate",
                    },
                },
                {
                    "billing_code": "99223",
                    "pricing_provenance": {
                        "lookup_level": "cached_corpus_quote",
                    },
                },
            ],
            "case_components": [],
        }

        fallback_fields = pricing_fallback_log_fields(bill_record)

        self.assertIsNotNone(fallback_fields)
        assert fallback_fields is not None
        self.assertEqual(fallback_fields["cached_corpus_estimate_lines"], 1)
        self.assertEqual(fallback_fields["cached_corpus_quote_lines"], 1)

    def test_pricing_fallback_log_fields_counts_inpatient_professional_claim_usage(self) -> None:
        bill_record = {
            "case_id": "case-inpatient",
            "facility": {
                "selection_provenance": {
                    "geographic_stage": "zip",
                    "selection_probe_count": 1,
                    "same_state_unfiltered_fallback_invoked": False,
                }
            },
            "provenance": {"pricing": {}},
            "facility_claim": {
                "charge_lines": [],
                "payment_components": [],
            },
            "professional_claims": [
                {
                    "lines": [
                        {
                            "billing_code": "99223",
                            "pricing_provenance": {
                                "lookup_level": "cached_corpus_quote",
                            },
                        }
                    ]
                }
            ],
        }

        fallback_fields = pricing_fallback_log_fields(bill_record)

        self.assertIsNotNone(fallback_fields)
        assert fallback_fields is not None
        self.assertEqual(fallback_fields["cached_corpus_quote_lines"], 1)

    def test_pricing_fallback_log_fields_returns_none_without_any_fallback_signal(self) -> None:
        bill_record = {
            "case_id": "case-2",
            "facility": {
                "selection_provenance": {
                    "geographic_stage": "zip",
                    "selection_probe_count": 1,
                    "same_state_unfiltered_fallback_invoked": False,
                }
            },
            "provenance": {"pricing": {}},
            "line_items": [
                {
                    "billing_code": "99213",
                    "pricing_provenance": {
                        "lookup_level": "hospital_payer",
                    },
                }
            ],
            "case_components": [],
        }

        self.assertIsNone(pricing_fallback_log_fields(bill_record))
