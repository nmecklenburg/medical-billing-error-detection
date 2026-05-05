import io
import json
import threading
import time
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import requests

import duckdb

from claim.cli_logging import CliLogger
from scripts._deprecated.generate_gold_bills import TRILLIANT_ALIAS
from scripts._deprecated.generate_gold_bills import attach_descriptions_to_bill
from scripts._deprecated.generate_gold_bills import build_aftercare_prompt_payload
from scripts._deprecated.generate_gold_bills import build_generated_cluster_record_for_encounter
from scripts._deprecated.generate_gold_bills import build_imputed_rate_quote
from scripts._deprecated.generate_gold_bills import ClusterGenerationContext
from scripts._deprecated.generate_gold_bills import cleanup_stale_partial_files
from scripts._deprecated.generate_gold_bills import compute_line_item_accounting
from scripts._deprecated.generate_gold_bills import description_cache_key
from scripts._deprecated.generate_gold_bills import detect_csv_column
from scripts._deprecated.generate_gold_bills import DescriptionCacheStore
from scripts._deprecated.generate_gold_bills import filter_public_directory_candidates
from scripts._deprecated.generate_gold_bills import discover_public_hospital_rate_schema
from scripts._deprecated.generate_gold_bills import discover_trilliant_schema
from scripts._deprecated.generate_gold_bills import ensure_cache_tables
from scripts._deprecated.generate_gold_bills import extract_zip_codes_from_hud_response
from scripts._deprecated.generate_gold_bills import generate_bill_for_encounter
from scripts._deprecated.generate_gold_bills import JsonlStreamWriter
from scripts._deprecated.generate_gold_bills import load_hud_crosswalk
from scripts._deprecated.generate_gold_bills import load_ssa_fips_crosswalk
from scripts._deprecated.generate_gold_bills import load_existing_bills_state
from scripts._deprecated.generate_gold_bills import load_existing_case_ids
from scripts._deprecated.generate_gold_bills import PricingBuildContext
from scripts._deprecated.generate_gold_bills import query_rate_level
from scripts._deprecated.generate_gold_bills import quote_supports_scenario
from scripts._deprecated.generate_gold_bills import repair_jsonl_trailing_partial_line
from scripts._deprecated.generate_gold_bills import run_stage_tasks
from scripts._deprecated.generate_gold_bills import sample_encounters_from_cache
from scripts._deprecated.generate_gold_bills import sanitize_rate_quote
from scripts._deprecated.generate_gold_bills import select_financial_scenario
from scripts._deprecated.generate_gold_bills import shutdown_executor
from scripts._deprecated.generate_gold_bills import shift_encounter_dates
from scripts._deprecated.generate_gold_bills import usage_totals
from scripts._deprecated.generate_gold_bills import validate_resumable_case_ids


def excel_column_name(index: int) -> str:
    column = index + 1
    characters: list[str] = []
    while column > 0:
        column, remainder = divmod(column - 1, 26)
        characters.append(chr(ord("A") + remainder))
    return "".join(reversed(characters))


def build_inline_sheet_xml(rows: list[list[str]]) -> str:
    row_xml: list[str] = []
    for row_index, row in enumerate(rows, start=1):
        cells: list[str] = []
        for column_index, value in enumerate(row):
            cell_reference = f"{excel_column_name(column_index)}{row_index}"
            safe_value = (
                value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            cells.append(
                f'<c r="{cell_reference}" t="inlineStr"><is><t>{safe_value}</t></is></c>'
            )
        row_xml.append(f'<row r="{row_index}">{"".join(cells)}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(row_xml)}</sheetData>'
        "</worksheet>"
    )


def write_minimal_xlsx(path: Path, sheets: list[tuple[str, list[list[str]]]]) -> None:
    workbook_sheets = []
    workbook_rels = []
    for index, (name, rows) in enumerate(sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{name}" sheetId="{index}" r:id="rId{index}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{index}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{index}.xml"/>'
        )
    workbook_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f'<sheets>{"".join(workbook_sheets)}</sheets>'
        "</workbook>"
    )
    rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'{"".join(workbook_rels)}'
        "</Relationships>"
    )
    content_types = [
        '<Override PartName="/xl/workbook.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
    ]
    for index in range(1, len(sheets) + 1):
        content_types.append(
            f'<Override PartName="/xl/worksheets/sheet{index}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f'{"".join(content_types)}'
        "</Types>"
    )
    root_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("_rels/.rels", root_rels_xml)
        archive.writestr("xl/workbook.xml", workbook_xml)
        archive.writestr("xl/_rels/workbook.xml.rels", rels_xml)
        for index, (_, rows) in enumerate(sheets, start=1):
            archive.writestr(
                f"xl/worksheets/sheet{index}.xml",
                build_inline_sheet_xml(rows),
            )


class FakeResponses:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = list(payloads)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        return SimpleNamespace(
            output_text=json.dumps(payload),
            usage={"input_tokens": 11, "output_tokens": 7, "total_tokens": 18},
            output=[],
        )


class FakeOpenAIClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.responses = FakeResponses(payloads)


class GenerateGoldBillsTests(unittest.TestCase):
    def test_shutdown_executor_uses_non_blocking_mode_when_interrupted(self) -> None:
        calls: list[tuple[bool, bool]] = []

        class FakeExecutor:
            def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
                calls.append((wait, cancel_futures))

        shutdown_executor(FakeExecutor(), interrupted=True)
        shutdown_executor(FakeExecutor(), interrupted=False)

        self.assertEqual(calls, [(False, True), (True, False)])

    def test_run_stage_tasks_starts_timer_when_worker_begins_processing(self) -> None:
        release_event = threading.Event()
        release_time = {"value": 0.0}
        started_at: dict[int, float | None] = {}
        failures: list[str] = []

        def release_workers() -> None:
            time.sleep(0.05)
            release_time["value"] = time.perf_counter()
            release_event.set()

        def process_item(item: int) -> int:
            if item in {1, 2} and not release_event.wait(timeout=1.0):
                raise RuntimeError("timed out waiting to release queued workers")
            return item

        releaser = threading.Thread(target=release_workers)
        releaser.start()
        try:
            interrupted = run_stage_tasks(
                stage="timing",
                items=[1, 2, 3, 4],
                worker_count=2,
                progress_description="timing test",
                show_progress=False,
                logger=CliLogger("test", use_color=False),
                case_id_for_item=lambda item: str(item),
                process_item=process_item,
                on_success=lambda payload, context: started_at.__setitem__(payload, context.case_start),
                on_failure=lambda _stage, _case_id, error: failures.append(str(error)),
                interrupt_details=lambda: {},
            )
        finally:
            releaser.join(timeout=1.0)

        self.assertFalse(interrupted)
        self.assertEqual(failures, [])
        self.assertIsNotNone(started_at[3])
        self.assertIsNotNone(started_at[4])
        self.assertGreaterEqual(started_at[3], release_time["value"])
        self.assertGreaterEqual(started_at[4], release_time["value"])

    def test_build_generated_cluster_record_reviews_demographically_implausible_cluster(self) -> None:
        client = FakeOpenAIClient(
            [
                {
                    "primary_clinical_direction": "Outpatient hypertension follow-up",
                    "clinical_direction_rationale": "Routine office follow-up with one bad generated screening code.",
                    "retained_input_codes": [
                        {
                            "code_system": "ICD9",
                            "code": "4019",
                            "reason": "Essential hypertension diagnosis.",
                        },
                        {
                            "code_system": "HCPCS/CPT",
                            "code": "99213",
                            "reason": "Office visit present in source.",
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
                            "service_date": "2020-06-01",
                        },
                        {
                            "billing_code": "G0123",
                            "code_system": "HCPCS/CPT",
                            "supporting_input_code": None,
                            "supporting_input_code_system": None,
                            "service_date": "2020-06-01",
                        },
                    ],
                },
                {
                    "primary_clinical_direction": "Outpatient hypertension follow-up",
                    "clinical_direction_rationale": "Routine office follow-up with hypertension-compatible services only.",
                    "retained_input_codes": [
                        {
                            "code_system": "ICD9",
                            "code": "4019",
                            "reason": "Essential hypertension diagnosis.",
                        },
                        {
                            "code_system": "HCPCS/CPT",
                            "code": "99213",
                            "reason": "Office visit present in source.",
                        },
                    ],
                    "filtered_input_codes": [
                        {
                            "code_system": "HCPCS/CPT",
                            "code": "G0123",
                            "reason": "Cervical cytology screening is incompatible with a male patient and unrelated to hypertension follow-up.",
                        },
                    ],
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
                            "service_date": "2020-06-01",
                        },
                        {
                            "billing_code": "36415",
                            "code_system": "HCPCS/CPT",
                            "supporting_input_code": None,
                            "supporting_input_code_system": None,
                            "service_date": "2020-06-01",
                        },
                    ],
                },
            ]
        )
        encounter = {
            "cluster_id": "review-case",
            "desynpuf_id": "ABC123",
            "beneficiary": {
                "birth_date": "1950-01-01",
                "state_code": "01",
                "county_code": "001",
                "sex_code": "1",
                "race_code": "1",
            },
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
                }
            ],
            "encounter": {
                "anchor_claim_id": "claim-1",
                "anchor_claim_type": "outpatient",
                "from_date": "2009-06-01",
                "thru_date": "2009-06-01",
            },
            "summary": {
                "unique_hcpcs_cpt_code_count": 1,
                "claim_types_present": ["outpatient"],
            },
            "sample_number": 1,
        }

        with TemporaryDirectory() as tmpdir, requests.Session() as session:
            context = ClusterGenerationContext(
                openai_model="gpt-5-mini",
                reasoning_effort="low",
                request_delay_seconds=0.0,
                ssa_fips_mapping={
                    ("01", "001"): {
                        "state_fips": "01",
                        "county_fips": "01001",
                        "county_name": "Autauga County",
                        "state_name": "AL",
                    }
                },
                local_hud_mapping={"01001": ["36003"]},
                reference_data_dir=Path(tmpdir),
                hud_user_token=None,
                logger=CliLogger("test", use_color=False),
                client=client,
                session=session,
            )

            cluster_record, usage = build_generated_cluster_record_for_encounter(
                encounter,
                context=context,
            )

        generated_codes = [
            item["billing_code"]
            for item in cluster_record["code_cluster"]["generated_line_items"]
        ]
        filtered_codes = [
            item["code"]
            for item in cluster_record["code_cluster"]["filtered_input_codes"]
        ]
        self.assertEqual(generated_codes, ["99213", "36415"])
        self.assertIn("G0123", filtered_codes)
        self.assertEqual(usage, {"input_tokens": 22, "output_tokens": 14, "total_tokens": 36})

    def test_load_ssa_fips_crosswalk_detects_columns(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ssa.csv"
            path.write_text(
                "\n".join(
                    [
                        "ssa_state,ssa_county,fips_state,fips_county,county_name,state_name",
                        "01,001,01,001,Autauga County,AL",
                    ]
                ),
                encoding="utf-8",
            )

            mapping = load_ssa_fips_crosswalk(path)

            self.assertEqual(
                mapping[("01", "001")],
                {
                    "state_fips": "01",
                    "county_fips": "01001",
                    "county_name": "Autauga County",
                    "state_name": "AL",
                    "state_abbrev": "AL",
                },
            )

    def test_load_ssa_fips_crosswalk_handles_combined_ssa_code_format(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ssa.csv"
            path.write_text(
                "\n".join(
                    [
                        "countyname_fips,state,fipscounty,cbsa_code,cbsa_name,ssa_code,state_name,countyname_rate",
                        'ST. LOUIS,MO,29189,41180,"St. Louis, MO-IL",26950,Missouri,ST. LOUIS CITY',
                        'ST. LOUIS CITY,MO,29510,41180,"St. Louis, MO-IL",26950,Missouri,ST. LOUIS CITY',
                    ]
                ),
                encoding="utf-8",
            )

            mapping = load_ssa_fips_crosswalk(path)

            self.assertEqual(
                mapping[("26", "950")],
                {
                    "state_fips": "29",
                    "county_fips": "29510",
                    "county_name": "ST. LOUIS CITY",
                    "state_name": "Missouri",
                    "state_abbrev": "MO",
                    "county_fips_candidates": ["29189", "29510"],
                },
            )

    def test_extract_zip_codes_from_hud_response_collects_nested_values(self) -> None:
        payload = {
            "data": {
                "results": [
                    {"zip_code": "36003", "county": "01001"},
                    {"zip_codes": ["36004", "36003"]},
                ]
            }
        }

        self.assertEqual(
            extract_zip_codes_from_hud_response(payload),
            ["36003", "36004"],
        )

    def test_load_hud_crosswalk_accepts_xlsx_export_sheet(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "zip_county_crosswalk.xlsx"
            write_minimal_xlsx(
                path,
                [
                    ("Export Worksheet", [["ZIP", "COUNTY"], ["00501", "36103"], ["00601", "72081"]]),
                    ("SQL", [["select * from ZIP_COUNTY_122025 order by zip"]]),
                ],
            )

            mapping = load_hud_crosswalk(path)

            self.assertEqual(mapping["36103"], ["00501"])
            self.assertEqual(mapping["72081"], ["00601"])

    def test_sample_encounters_from_cache_filters_for_hcpcs(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "encounters.jsonl"
            rows = [
                {
                    "cluster_id": "no-codes",
                    "summary": {"unique_hcpcs_cpt_code_count": 0},
                },
                {
                    "cluster_id": "with-codes-a",
                    "summary": {"unique_hcpcs_cpt_code_count": 1},
                },
                {
                    "cluster_id": "with-codes-b",
                    "summary": {"unique_hcpcs_cpt_code_count": 2},
                },
            ]
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row))
                    handle.write("\n")

            sampled, stats = sample_encounters_from_cache(
                path,
                sample_size=2,
                seed=0,
                logger=CliLogger("test", use_color=False),
                show_progress=False,
            )

            self.assertEqual(stats["eligible_records"], 2)
            self.assertEqual(
                [row["cluster_id"] for row in sampled],
                ["with-codes-a", "with-codes-b"],
            )

    def test_shift_encounter_dates_moves_legacy_dates_once(self) -> None:
        encounter = {
            "beneficiary": {
                "birth_date": "1950-01-01",
                "death_date": "2009-12-31",
            },
            "claims": [
                {
                    "claim_id": "claim-1",
                    "from_date": "2008-09-04",
                    "thru_date": "2008-09-05",
                    "admission_date": "2008-09-04",
                    "discharge_date": "2008-09-05",
                }
            ],
            "procedure_codes": [
                {
                    "code": "99283",
                    "code_system": "HCPCS/CPT",
                    "service_date": "2008-09-04",
                }
            ],
            "encounter": {
                "from_date": "2008-09-04",
                "thru_date": "2008-09-05",
            },
        }

        shifted = shift_encounter_dates(encounter)

        self.assertEqual(shifted["beneficiary"]["birth_date"], "1961-01-01")
        self.assertEqual(shifted["beneficiary"]["death_date"], "2020-12-31")
        self.assertEqual(shifted["claims"][0]["from_date"], "2019-09-04")
        self.assertEqual(shifted["claims"][0]["thru_date"], "2019-09-05")
        self.assertEqual(shifted["claims"][0]["admission_date"], "2019-09-04")
        self.assertEqual(shifted["claims"][0]["discharge_date"], "2019-09-05")
        self.assertEqual(shifted["procedure_codes"][0]["service_date"], "2019-09-04")
        self.assertEqual(shifted["encounter"]["from_date"], "2019-09-04")
        self.assertEqual(shift_encounter_dates(shifted), shifted)

    def test_build_aftercare_prompt_payload_omits_billing_references(self) -> None:
        payload = build_aftercare_prompt_payload(
            {
                "case_id": "case-1",
                "source_encounter": {
                    "encounter": {
                        "anchor_claim_type": "outpatient",
                        "from_date": "2008-09-04",
                        "thru_date": "2008-09-04",
                    }
                },
                "facility": {
                    "hospital_name": "Autauga Medical Center",
                    "city": "Prattville",
                    "state": "AL",
                },
                "patient": {
                    "age_at_encounter": 58,
                    "sex_code": "1",
                },
                "payer_scenario": {
                    "scenario": "in_network",
                    "payer_name": "Aetna",
                },
                "diagnoses": [
                    {
                        "source_icd9_code": "4019",
                        "selected_icd10_code": "I10",
                        "description": "High blood pressure",
                    }
                ],
                "line_items": [
                    {
                        "billing_code": "99283",
                        "description": "Emergency department evaluation",
                        "service_date": "2008-09-04",
                        "source_claim_type": "outpatient",
                    }
                ],
                "bill_totals": {
                    "billed_charge": 650.0,
                    "patient_responsibility": 80.0,
                },
            }
        )

        self.assertEqual(
            payload,
            {
                "case_id": "case-1",
                "visit_context": {
                    "visit_type": "outpatient",
                    "from_date": "2008-09-04",
                    "thru_date": "2008-09-04",
                    "facility_name": "Autauga Medical Center",
                    "facility_city": "Prattville",
                    "facility_state": "AL",
                },
                "patient": {
                    "age_at_encounter": 58,
                    "sex": "male",
                },
                "diagnoses": ["High blood pressure"],
                "line_items": [
                    {
                        "description": "Emergency department evaluation",
                        "service_date": "2008-09-04",
                        "care_setting": "outpatient",
                    }
                ],
            },
        )

    def test_filter_public_directory_candidates_prefers_zip_matches(self) -> None:
        rows = [
            {
                "id": "hospital-b",
                "hospitalName": "Bravo Regional",
                "address": "2 Main St, Somewhere, AL 36004",
                "city": "Somewhere",
                "state": "AL",
            },
            {
                "id": "hospital-a",
                "hospitalName": "Alpha Medical Center",
                "address": "1 Main St, Somewhere, AL 36003",
                "city": "Somewhere",
                "state": "AL",
            },
            {
                "id": "hospital-c",
                "hospitalName": "Charlie General",
                "address": "3 Main St, Elsewhere, GA 30000",
                "city": "Elsewhere",
                "state": "GA",
            },
        ]

        candidates, query_mode = filter_public_directory_candidates(
            rows,
            zip_codes=["36003"],
            state_value="AL",
            max_candidates=10,
        )

        self.assertEqual(query_mode, "directory_zip")
        self.assertEqual(
            [candidate["hospital_directory_id"] for candidate in candidates],
            ["hospital-a"],
        )

    def test_jsonl_stream_writer_appends_lines_incrementally(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            with JsonlStreamWriter(path) as writer:
                writer.write({"case_id": "case-1"})
                writer.write({"case_id": "case-2"})
            with JsonlStreamWriter(path, append=True) as writer:
                writer.write({"case_id": "case-3"})

            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                [
                    '{"case_id": "case-1"}',
                    '{"case_id": "case-2"}',
                    '{"case_id": "case-3"}',
                ],
            )

    def test_repair_jsonl_trailing_partial_line_truncates_invalid_tail(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(
                '{"case_id": "case-1"}\n{"case_id": "case-2"}\n{"case_id": "case',
                encoding="utf-8",
            )

            trimmed = repair_jsonl_trailing_partial_line(
                path,
                logger=CliLogger("test"),
                label="test rows",
            )

            self.assertTrue(trimmed)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ['{"case_id": "case-1"}', '{"case_id": "case-2"}'],
            )

    def test_load_existing_state_and_validate_resume_subset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            bills_path = Path(tmpdir) / "bills.jsonl"
            bills_path.write_text(
                "\n".join(
                    [
                        json.dumps({"case_id": "case-1", "payer_scenario": {"scenario": "in_network"}}),
                        json.dumps({"case_id": "case-2", "payer_scenario": {"scenario": "self_pay"}}),
                        json.dumps({"case_id": "case-1", "payer_scenario": {"scenario": "in_network"}}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            logger = CliLogger("test")

            case_ids, scenario_counts = load_existing_bills_state(bills_path, logger=logger)
            self.assertEqual(case_ids, {"case-1", "case-2"})
            self.assertEqual(scenario_counts, {"in_network": 1, "self_pay": 1})

            aftercare_path = Path(tmpdir) / "aftercare.jsonl"
            aftercare_path.write_text(
                json.dumps({"case_id": "case-1"}) + "\n",
                encoding="utf-8",
            )
            aftercare_ids = load_existing_case_ids(
                aftercare_path,
                logger=logger,
                label="aftercare",
            )
            self.assertEqual(aftercare_ids, {"case-1"})

            validate_resumable_case_ids(
                {"case-1", "case-2", "case-3"},
                case_ids,
                path=bills_path,
                label="bill",
            )
            with self.assertRaises(RuntimeError):
                validate_resumable_case_ids(
                    {"case-1"},
                    case_ids,
                    path=bills_path,
                    label="bill",
                )

    def test_discover_trilliant_schema_finds_hospital_and_rate_tables(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "remote.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals_table (
                        hospital_name TEXT,
                        npi TEXT,
                        zip_code TEXT,
                        state TEXT,
                        county_fips TEXT,
                        address TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE rates_table (
                        billing_code TEXT,
                        payer_name TEXT,
                        negotiated_rate DOUBLE,
                        gross_charge DOUBLE,
                        npi TEXT,
                        state TEXT
                    )
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_trilliant_schema(
                    cache,
                    hospital_table_override=None,
                    rate_table_override=None,
                )
            finally:
                cache.close()

            self.assertEqual(schema.hospital_table, "hospitals_table")
            self.assertEqual(schema.rate_table, "rates_table")
            self.assertEqual(schema.hospital_name_column, "hospital_name")
            self.assertEqual(schema.rate_billing_code_column, "billing_code")

    def test_compute_line_item_accounting_balances_network_and_extreme_cases(self) -> None:
        in_network = compute_line_item_accounting(
            {"negotiated_rate": 400.0, "gross_charge": 650.0},
            scenario="in_network",
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
        )
        extreme = compute_line_item_accounting(
            {"negotiated_rate": 400.0, "gross_charge": 650.0},
            scenario="out_of_network",
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
        )

        self.assertEqual(
            in_network,
            {
                "billed_charge": 650.0,
                "contractual_adjustment": 250.0,
                "allowed_amount": 400.0,
                "insurance_paid": 320.0,
                "patient_responsibility": 80.0,
            },
        )
        self.assertEqual(
            extreme,
            {
                "billed_charge": 650.0,
                "contractual_adjustment": 0.0,
                "allowed_amount": 650.0,
                "insurance_paid": 0.0,
                "patient_responsibility": 650.0,
            },
        )

    def test_compute_line_item_accounting_clamps_in_network_billed_charge_to_allowed_amount(self) -> None:
        accounting = compute_line_item_accounting(
            {"negotiated_rate": 2200.0, "gross_charge": 150.0},
            scenario="in_network",
            insurance_coverage_ratio=0.8,
            billed_charge_multiplier=1.35,
        )

        self.assertEqual(
            accounting,
            {
                "billed_charge": 2200.0,
                "contractual_adjustment": 0.0,
                "allowed_amount": 2200.0,
                "insurance_paid": 1760.0,
                "patient_responsibility": 440.0,
            },
        )

    def test_select_financial_scenario_avoids_medicare_advantage_for_under_65_default_cases(self) -> None:
        payer_names = {
            select_financial_scenario(
                "case-under-65",
                seed,
                0.0,
                ["Medicare Advantage", "Aetna", "UnitedHealthcare"],
                age_at_encounter=61,
            )["payer_name"]
            for seed in range(12)
        }

        self.assertNotIn("Medicare Advantage", payer_names)
        self.assertTrue(payer_names.issubset({"Aetna", "UnitedHealthcare"}))

    def test_query_rate_level_supports_public_directory_standard_charge_details(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Aetna', 400.0, NULL, NULL, NULL, NULL, NULL),
                    (2, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Other Payer', 420.0, NULL, NULL, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    identity_filter=None,
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(
            quotes["99283"],
            {
                "billing_code": "99283",
                "negotiated_rate": 400.0,
                "gross_charge": 650.0,
                "description": "Emergency department visit",
                "row_count": 1,
                "lookup_level": "hospital_payer",
                "payer_name": "Aetna",
                "identity_filter": None,
                "state_value": None,
            },
        )

    def test_query_rate_level_filters_implausible_public_directory_amounts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "hospital.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals (
                        hospital_id BIGINT,
                        hospital_name TEXT,
                        hospital_address TEXT,
                        hospital_state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals VALUES
                    (1, 'Alpha Medical Center', '1 Main St, Somewhere, AL 36003', 'AL')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE standard_charge_details (
                        charge_id BIGINT,
                        hospital_id BIGINT,
                        description TEXT,
                        gross_charge DOUBLE,
                        cpt TEXT,
                        hcpcs TEXT,
                        payer_name TEXT,
                        standard_charge_dollar DOUBLE,
                        standard_charge_percentage DOUBLE,
                        estimated_amount DOUBLE,
                        minimum DOUBLE,
                        maximum DOUBLE,
                        median_amount DOUBLE
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO standard_charge_details VALUES
                    (1, 1, 'Emergency department visit', 650.0, '99283', NULL, 'Aetna', 400.0, NULL, NULL, NULL, NULL, NULL),
                    (2, 1, 'Emergency department visit', 999999999.0, '99283', NULL, 'Aetna', 999999999.0, NULL, NULL, NULL, NULL, NULL)
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_public_hospital_rate_schema(cache)
                quotes = query_rate_level(
                    cache,
                    schema,
                    billing_codes=["99283"],
                    lookup_level="hospital_payer",
                    payer_name="Aetna",
                    identity_filter=None,
                    state_value=None,
                    max_quote_amount=10_000_000.0,
                )
            finally:
                cache.close()

        self.assertEqual(quotes["99283"]["negotiated_rate"], 400.0)
        self.assertEqual(quotes["99283"]["gross_charge"], 650.0)

    def test_sanitize_rate_quote_supports_scenario_specific_fallback(self) -> None:
        quote = sanitize_rate_quote(
            {
                "billing_code": "99283",
                "negotiated_rate": 999999999.0,
                "gross_charge": 650.0,
            },
            max_quote_amount=10_000_000.0,
        )

        self.assertIsNone(quote["negotiated_rate"])
        self.assertEqual(quote["gross_charge"], 650.0)
        self.assertFalse(quote_supports_scenario(quote, scenario_name="in_network"))
        self.assertTrue(quote_supports_scenario(quote, scenario_name="out_of_network"))

    def test_generate_bill_for_encounter_end_to_end(self) -> None:
        with TemporaryDirectory() as tmpdir:
            remote_path = Path(tmpdir) / "remote.duckdb"
            remote = duckdb.connect(str(remote_path))
            try:
                remote.execute(
                    """
                    CREATE TABLE hospitals_table (
                        hospital_name TEXT,
                        npi TEXT,
                        zip_code TEXT,
                        state TEXT,
                        county_fips TEXT,
                        address TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO hospitals_table VALUES
                    ('Autauga Medical Center', '1234567890', '36003', 'AL', '01001', '123 Main St')
                    """
                )
                remote.execute(
                    """
                    CREATE TABLE rates_table (
                        billing_code TEXT,
                        payer_name TEXT,
                        negotiated_rate DOUBLE,
                        gross_charge DOUBLE,
                        npi TEXT,
                        state TEXT
                    )
                    """
                )
                remote.execute(
                    """
                    INSERT INTO rates_table VALUES
                    ('99283', 'Aetna', 400.0, 650.0, '1234567890', 'AL')
                    """
                )
            finally:
                remote.close()

            cache = duckdb.connect(":memory:")
            try:
                ensure_cache_tables(cache)
                escaped_remote_path = str(remote_path).replace("'", "''")
                cache.execute(
                    f"ATTACH '{escaped_remote_path}' AS {TRILLIANT_ALIAS} (READ_ONLY)"
                )
                schema = discover_trilliant_schema(
                    cache,
                    hospital_table_override=None,
                    rate_table_override=None,
                )
                client = FakeOpenAIClient(
                    [
                        {
                            "primary_clinical_direction": "Outpatient hypertension evaluation",
                            "clinical_direction_rationale": "The cluster is most consistent with evaluating hypertension in an outpatient emergency setting, while the diabetes diagnosis is unrelated noise.",
                            "retained_input_codes": [
                                {
                                    "code_system": "ICD9",
                                    "code": "4019",
                                    "reason": "Supports the hypertension-focused direction.",
                                },
                                {
                                    "code_system": "HCPCS/CPT",
                                    "code": "99283",
                                    "reason": "Fits an outpatient emergency evaluation.",
                                },
                            ],
                            "filtered_input_codes": [
                                {
                                    "code_system": "ICD9",
                                    "code": "25000",
                                    "reason": "Not needed for the selected clinical direction.",
                                }
                            ],
                            "generated_diagnoses": [
                                {
                                    "selected_icd10_code": "I10",
                                    "supporting_input_code": "4019",
                                    "supporting_input_code_system": "ICD9",
                                },
                                {
                                    "selected_icd10_code": "R079",
                                    "supporting_input_code": None,
                                    "supporting_input_code_system": None,
                                },
                            ],
                            "generated_line_items": [
                                {
                                    "billing_code": "99283",
                                    "code_system": "HCPCS/CPT",
                                    "supporting_input_code": "99283",
                                    "supporting_input_code_system": "HCPCS/CPT",
                                    "service_date": "2008-09-04",
                                },
                                {
                                    "billing_code": "93000",
                                    "code_system": "HCPCS/CPT",
                                    "supporting_input_code": None,
                                    "supporting_input_code_system": None,
                                    "service_date": "2008-09-04",
                                },
                            ],
                        },
                        {
                            "primary_clinical_direction": "Outpatient hypertension evaluation",
                            "clinical_direction_rationale": "The cluster is most consistent with evaluating hypertension in an outpatient emergency setting, while the diabetes diagnosis is unrelated noise.",
                            "retained_input_codes": [
                                {
                                    "code_system": "ICD9",
                                    "code": "4019",
                                    "reason": "Supports the hypertension-focused direction.",
                                },
                                {
                                    "code_system": "HCPCS/CPT",
                                    "code": "99283",
                                    "reason": "Fits an outpatient emergency evaluation.",
                                },
                            ],
                            "filtered_input_codes": [
                                {
                                    "code_system": "ICD9",
                                    "code": "25000",
                                    "reason": "Not needed for the selected clinical direction.",
                                }
                            ],
                            "generated_diagnoses": [
                                {
                                    "selected_icd10_code": "I10",
                                    "supporting_input_code": "4019",
                                    "supporting_input_code_system": "ICD9",
                                },
                                {
                                    "selected_icd10_code": "R079",
                                    "supporting_input_code": None,
                                    "supporting_input_code_system": None,
                                },
                            ],
                            "generated_line_items": [
                                {
                                    "billing_code": "99283",
                                    "code_system": "HCPCS/CPT",
                                    "supporting_input_code": "99283",
                                    "supporting_input_code_system": "HCPCS/CPT",
                                    "service_date": "2008-09-04",
                                },
                                {
                                    "billing_code": "93000",
                                    "code_system": "HCPCS/CPT",
                                    "supporting_input_code": None,
                                    "supporting_input_code_system": None,
                                    "service_date": "2008-09-04",
                                },
                            ],
                        },
                        {
                            "descriptions": [
                                {
                                    "code_system": "ICD10",
                                    "code": "I10",
                                    "description": "High blood pressure",
                                },
                                {
                                    "code_system": "ICD10",
                                    "code": "R079",
                                    "description": "Chest pain",
                                },
                                {
                                    "code_system": "HCPCS/CPT",
                                    "code": "99283",
                                    "description": "Emergency department evaluation",
                                },
                                {
                                    "code_system": "HCPCS/CPT",
                                    "code": "93000",
                                    "description": "Electrocardiogram",
                                },
                            ]
                        },
                        {
                            "document_title": "After-Visit Summary",
                            "encounter_overview": "You were seen and evaluated in the emergency department.",
                            "diagnoses": ["Hypertension", "Chest pain"],
                            "services_received": ["Emergency evaluation", "Electrocardiogram"],
                            "home_care_instructions": ["Continue routine care at home."],
                            "follow_up_actions": ["Follow up with your primary care doctor."],
                            "warning_signs": ["Return for worsening symptoms."],
                        },
                    ]
                )
                encounter = {
                    "cluster_id": "case-1",
                    "desynpuf_id": "ABC123",
                    "beneficiary": {
                        "birth_date": "1950-01-01",
                        "state_code": "01",
                        "county_code": "001",
                        "sex_code": "1",
                        "race_code": "1",
                    },
                    "claims": [
                        {
                            "claim_id": "claim-1",
                            "claim_type": "outpatient",
                            "from_date": "2008-09-04",
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
                        },
                        {
                            "code": "25000",
                            "code_system": "ICD9",
                            "source_claim_type": "outpatient",
                            "source_claim_id": "claim-1",
                            "source_field": "ICD9_DGNS_CD_2",
                        }
                    ],
                    "procedure_codes": [
                        {
                            "code": "99283",
                            "code_system": "HCPCS/CPT",
                            "source_claim_type": "outpatient",
                            "source_claim_id": "claim-1",
                            "source_field": "HCPCS_CD_1",
                            "line_number": 1,
                        }
                    ],
                    "encounter": {
                        "anchor_claim_id": "claim-1",
                        "anchor_claim_type": "outpatient",
                        "from_date": "2008-09-04",
                        "thru_date": "2008-09-04",
                    },
                    "summary": {
                        "unique_hcpcs_cpt_code_count": 1,
                        "claim_types_present": ["outpatient"],
                    },
                    "sample_number": 1,
                }
                with requests.Session() as session:
                    cluster_context = ClusterGenerationContext(
                        openai_model="gpt-5-mini",
                        reasoning_effort="low",
                        request_delay_seconds=0.0,
                        ssa_fips_mapping={
                            ("01", "001"): {
                                "state_fips": "01",
                                "county_fips": "01001",
                                "county_name": "Autauga County",
                                "state_name": "AL",
                            }
                        },
                        local_hud_mapping={"01001": ["36003"]},
                        reference_data_dir=Path(tmpdir),
                        hud_user_token=None,
                        logger=CliLogger("test", use_color=False),
                        client=client,
                        session=session,
                    )
                    pricing_context = PricingBuildContext(
                        seed=0,
                        openai_model="gpt-5-mini",
                        reasoning_effort="low",
                        request_delay_seconds=0.0,
                        normal_payers=["Aetna"],
                        extreme_case_probability=0.0,
                        insurance_coverage_ratio=0.8,
                        billed_charge_multiplier=1.35,
                        source_medicare_multiplier=2.5,
                        max_hospital_candidates=10,
                        max_hospital_rate_probes=6,
                        max_quote_amount=10_000_000.0,
                        description_cache=DescriptionCacheStore(
                            cache={},
                            model="gpt-5-mini",
                        ),
                        logger=CliLogger("test", use_color=False),
                        client=client,
                        session=session,
                        trilliant_connection=cache,
                        trilliant_schema=schema,
                    )
                    bill_record, aftercare_record, usage = generate_bill_for_encounter(
                        encounter,
                        cluster_context=cluster_context,
                        pricing_context=pricing_context,
                    )
            finally:
                cache.close()

            self.assertEqual(bill_record["facility"]["hospital_name"], "Autauga Medical Center")
            self.assertEqual(bill_record["payer_scenario"]["scenario"], "in_network")
            self.assertEqual(bill_record["patient"]["birth_date"], "1961-01-01")
            self.assertEqual(
                bill_record["source_encounter"]["encounter"]["from_date"],
                "2019-09-04",
            )
            self.assertEqual(
                bill_record["source_encounter"]["encounter"]["thru_date"],
                "2019-09-04",
            )
            self.assertEqual(bill_record["diagnoses"][0]["selected_icd10_code"], "I10")
            self.assertIsNone(bill_record["diagnoses"][1]["source_icd9_code"])
            self.assertEqual(bill_record["line_items"][0]["service_date"], "2019-09-04")
            self.assertEqual(bill_record["line_items"][1]["service_date"], "2019-09-04")
            self.assertEqual(
                bill_record["line_items"][0]["accounting"],
                {
                    "billed_charge": 650.0,
                    "contractual_adjustment": 250.0,
                    "allowed_amount": 400.0,
                    "insurance_paid": 320.0,
                    "patient_responsibility": 80.0,
                },
            )
            self.assertEqual(
                bill_record["line_items"][1]["accounting"],
                {
                    "billed_charge": 337.5,
                    "contractual_adjustment": 87.5,
                    "allowed_amount": 250.0,
                    "insurance_paid": 200.0,
                    "patient_responsibility": 50.0,
                },
            )
            self.assertEqual(
                bill_record["line_items"][1]["pricing_provenance"]["lookup_level"],
                "source_imputed",
            )
            self.assertEqual(bill_record["line_items"][0]["source_claim_id"], "claim-1")
            self.assertIsNone(bill_record["line_items"][1]["source_claim_id"])
            self.assertEqual(
                bill_record["provenance"]["cluster_generation"]["primary_clinical_direction"],
                "Outpatient hypertension evaluation",
            )
            self.assertTrue(bill_record["provenance"]["date_shift"]["applied"])
            self.assertEqual(bill_record["provenance"]["date_shift"]["years"], 11)
            self.assertEqual(
                bill_record["provenance"]["cluster_generation"]["filtered_input_codes"][0]["code"],
                "25000",
            )
            self.assertEqual(aftercare_record["summary"]["document_title"], "After-Visit Summary")
            self.assertEqual(usage, {"input_tokens": 44, "output_tokens": 28, "total_tokens": 72})


if __name__ == "__main__":
    unittest.main()
