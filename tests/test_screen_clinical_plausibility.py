import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import screen_inpatient_mismatch_plausibility as plausibility


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def make_outpatient_bill(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "encounter_regime": "outpatient",
        "facility": {"hospital_name": "Example Hospital"},
        "payer_scenario": {"scenario": "in_network"},
        "diagnoses": [
            {"selected_icd10_code": "R079", "description": "Chest pain"},
        ],
        "line_items": [
            {
                "billing_code": "99283",
                "code_system": "HCPCS/CPT",
                "description": "Emergency department visit level 3",
                "pricing_provenance": {"lookup_level": "hospital_payer"},
                "accounting": {"allowed_amount": 100.0},
            }
        ],
        "bill_totals": {
            "billed_charge": 140.0,
            "allowed_amount": 100.0,
            "patient_responsibility": 20.0,
        },
        "provenance": {
            "cluster_generation": {
                "primary_clinical_direction": "Outpatient chest pain evaluation"
            }
        },
    }


def make_inpatient_bill(case_id: str) -> dict[str, object]:
    return {
        "case_id": case_id,
        "encounter_regime": "inpatient",
        "facility": {"hospital_name": "Example Hospital"},
        "payer_scenario": {"scenario": "in_network"},
        "diagnoses": [
            {"selected_icd10_code": "K802", "description": "Calculus of gallbladder"},
        ],
        "facility_claim": {
            "payment_components": [
                {
                    "billing_code": "331",
                    "code_system": "MS-DRG",
                    "description": "Major bowel procedure without CC",
                    "pricing_provenance": {"lookup_level": "hospital_payer"},
                    "accounting": {"allowed_amount": 8000.0},
                }
            ],
            "charge_lines": [],
        },
        "professional_claims": [
            {
                "lines": [
                    {
                        "billing_code": "47562",
                        "code_system": "HCPCS/CPT",
                        "description": "Laparoscopic cholecystectomy",
                        "pricing_provenance": {"lookup_level": "hospital_payer"},
                        "accounting": {"allowed_amount": 300.0},
                    }
                ]
            }
        ],
        "bill_totals": {
            "billed_charge": 10000.0,
            "allowed_amount": 8300.0,
            "patient_responsibility": 1660.0,
        },
        "provenance": {
            "cluster_generation": {
                "primary_clinical_direction": "Inpatient gallbladder surgery"
            }
        },
    }


class ScreenClinicalPlausibilityTests(unittest.TestCase):
    def test_collect_dataset_bills_reads_inpatient_and_outpatient_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            write_jsonl(dataset_dir / "inpatient_bills.jsonl", [make_inpatient_bill("in-1")])
            write_jsonl(dataset_dir / "outpatient_bills.jsonl", [make_outpatient_bill("out-1")])

            bills = plausibility.collect_dataset_bills(dataset_dir)

        self.assertEqual([bill["case_id"] for bill in bills], ["in-1", "out-1"])

    def test_collect_dataset_bills_rejects_legacy_only_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir)
            write_jsonl(dataset_dir / "bills.jsonl", [make_outpatient_bill("legacy-1")])

            with self.assertRaises(FileNotFoundError) as exc:
                plausibility.collect_dataset_bills(dataset_dir)

        self.assertIn("Missing split bill files", str(exc.exception))

    def test_build_case_prompt_handles_outpatient_bill(self) -> None:
        prompt = plausibility.build_case_prompt(make_outpatient_bill("out-1"))

        self.assertIn("Encounter regime: outpatient", prompt)
        self.assertIn("Emergency department visit level 3", prompt)
        self.assertIn("Payment / anchor components:\n- None", prompt)

    def test_main_scans_full_dataset_and_writes_top_level_output(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            output_path = root / "clinical_plausibility_flags.minimax-m2.7.json"
            write_jsonl(dataset_dir / "inpatient_bills.jsonl", [make_inpatient_bill("in-1")])
            write_jsonl(dataset_dir / "outpatient_bills.jsonl", [make_outpatient_bill("out-1")])
            args = SimpleNamespace(
                dataset_dir=dataset_dir,
                output_path=output_path,
                model="openrouter:minimax/minimax-m2.7",
                reasoning_effort="low",
                timeout_seconds=120.0,
                max_cases=None,
                max_workers=2,
                progress_every=1,
                max_retries=0,
                dotenv_path=None,
            )

            def screen_side_effect(*, bill: dict[str, object], **_: object) -> dict[str, object]:
                if bill["case_id"] == "out-1":
                    return {
                        "is_problem": True,
                        "confidence": 0.92,
                        "severity": "high",
                        "summary": "Outpatient bill conflicts with documented direction.",
                        "contradictions": ["Different encounter type implied."],
                        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                    }
                return {
                    "is_problem": False,
                    "confidence": 0.2,
                    "severity": "low",
                    "summary": "Looks coherent.",
                    "contradictions": [],
                    "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
                }

            with (
                mock.patch.object(plausibility, "parse_args", return_value=args),
                mock.patch.object(plausibility, "ensure_model_client", return_value=object()),
                mock.patch.object(plausibility, "screen_case", side_effect=screen_side_effect),
            ):
                plausibility.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["screened_case_count"], 2)
        self.assertEqual(payload["screened_candidate_count"], 2)
        self.assertEqual(payload["max_workers"], 2)
        self.assertEqual(payload["max_retries"], 0)
        self.assertEqual(payload["failed_case_count"], 0)
        self.assertEqual(payload["flagged_problem_count"], 1)
        self.assertEqual(payload["flagged_problems"][0]["case_id"], "out-1")
        self.assertEqual(payload["flagged_problems"][0]["encounter_regime"], "outpatient")

    def test_main_continues_when_one_case_screen_fails(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_dir = root / "dataset"
            output_path = root / "clinical_plausibility_flags.minimax-m2.7.json"
            write_jsonl(dataset_dir / "inpatient_bills.jsonl", [make_inpatient_bill("in-1")])
            write_jsonl(dataset_dir / "outpatient_bills.jsonl", [make_outpatient_bill("out-1")])
            args = SimpleNamespace(
                dataset_dir=dataset_dir,
                output_path=output_path,
                model="openrouter:minimax/minimax-m2.7",
                reasoning_effort="low",
                timeout_seconds=120.0,
                max_cases=None,
                max_workers=2,
                progress_every=1,
                max_retries=1,
                dotenv_path=None,
            )

            def screen_side_effect(*, bill: dict[str, object], **_: object) -> dict[str, object]:
                if bill["case_id"] == "in-1":
                    raise RuntimeError("The model returned no structured output.")
                return {
                    "is_problem": True,
                    "confidence": 0.9,
                    "severity": "medium",
                    "summary": "Conflicting outpatient episode.",
                    "contradictions": ["Clinical direction mismatch."],
                    "usage": {"input_tokens": 9, "output_tokens": 5, "total_tokens": 14},
                }

            with (
                mock.patch.object(plausibility, "parse_args", return_value=args),
                mock.patch.object(plausibility, "ensure_model_client", return_value=object()),
                mock.patch.object(plausibility, "screen_case", side_effect=screen_side_effect),
            ):
                plausibility.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(payload["screened_case_count"], 2)
        self.assertEqual(payload["failed_case_count"], 1)
        self.assertEqual(payload["failed_cases"][0]["case_id"], "in-1")
        self.assertEqual(payload["failed_cases"][0]["attempt_count"], 2)
        self.assertEqual(payload["flagged_problem_count"], 1)
        self.assertEqual(payload["flagged_problems"][0]["case_id"], "out-1")


if __name__ == "__main__":
    unittest.main()
