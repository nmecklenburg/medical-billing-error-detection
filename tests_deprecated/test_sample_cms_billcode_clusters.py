import csv
import io
import json
import shutil
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock
import zipfile

from claim.cli_logging import CliLogger
from scripts._deprecated.sample_cms_billcode_clusters import default_encounter_cache_path
from scripts._deprecated.sample_cms_billcode_clusters import default_extracted_root
from scripts._deprecated.sample_cms_billcode_clusters import dataset_file_specs
from scripts._deprecated.sample_cms_billcode_clusters import download_file_if_missing
from scripts._deprecated.sample_cms_billcode_clusters import encounter_cache_metadata_path
from scripts._deprecated.sample_cms_billcode_clusters import effective_sample_size
from scripts._deprecated.sample_cms_billcode_clusters import carrier_claim_matches_anchor
from scripts._deprecated.sample_cms_billcode_clusters import ENCOUNTER_CACHE_SCHEMA_VERSION
from scripts._deprecated.sample_cms_billcode_clusters import extract_carrier_line_items
from scripts._deprecated.sample_cms_billcode_clusters import find_csv_member_name
from scripts._deprecated.sample_cms_billcode_clusters import iter_jsonl_records
from scripts._deprecated.sample_cms_billcode_clusters import merge_facility_anchors
from scripts._deprecated.sample_cms_billcode_clusters import sample_encounter_clusters
from scripts._deprecated.sample_cms_billcode_clusters import select_final_anchors


def write_zip_csv(
    path: Path,
    member_name: str,
    fieldnames: list[str],
    rows: list[dict[str, str]],
) -> None:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, buffer.getvalue())


class CmsBillcodeClusterTests(unittest.TestCase):
    def test_effective_sample_size_test_one_overrides_sample_size(self) -> None:
        self.assertEqual(effective_sample_size(25, test_one=True), 1)
        self.assertEqual(effective_sample_size(25, test_one=False), 25)

    def test_carrier_claim_matches_outpatient_requires_exact_dates_by_default(self) -> None:
        outpatient_anchor = {
            "anchor_claim_type": "outpatient",
            "carrier_match_from_date": date(2008, 3, 14),
            "carrier_match_thru_date": date(2008, 3, 14),
        }

        self.assertTrue(
            carrier_claim_matches_anchor(
                outpatient_anchor,
                date(2008, 3, 14),
                date(2008, 3, 14),
                0,
            )
        )
        self.assertFalse(
            carrier_claim_matches_anchor(
                outpatient_anchor,
                date(2008, 3, 15),
                date(2008, 3, 15),
                0,
            )
        )
        self.assertTrue(
            carrier_claim_matches_anchor(
                outpatient_anchor,
                date(2008, 3, 15),
                date(2008, 3, 15),
                1,
            )
        )

    def test_carrier_claim_matches_inpatient_uses_admission_discharge_window(self) -> None:
        inpatient_anchor = {
            "anchor_claim_type": "inpatient",
            "carrier_match_from_date": date(2008, 3, 10),
            "carrier_match_thru_date": date(2008, 3, 20),
        }

        self.assertTrue(
            carrier_claim_matches_anchor(
                inpatient_anchor,
                date(2008, 3, 12),
                date(2008, 3, 12),
                0,
            )
        )
        self.assertFalse(
            carrier_claim_matches_anchor(
                inpatient_anchor,
                date(2008, 3, 16),
                date(2008, 3, 21),
                0,
            )
        )
        self.assertFalse(
            carrier_claim_matches_anchor(
                inpatient_anchor,
                date(2008, 3, 9),
                date(2008, 3, 12),
                0,
            )
        )

    def test_find_csv_member_name_discovers_single_csv_member(self) -> None:
        with TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "sample.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("notes.txt", "ignore")
                archive.writestr("claims.csv", "DESYNPUF_ID,CLM_ID\nP1,1\n")

            self.assertEqual(find_csv_member_name(zip_path), "claims.csv")

    def test_iter_jsonl_records_supports_progress_mode(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            path.write_text('{"a": 1}\n{"b": 2}\n', encoding="utf-8")

            records = list(
                iter_jsonl_records(
                    path,
                    description="records",
                    show_progress=True,
                )
            )

            self.assertEqual(records, [{"a": 1}, {"b": 2}])

    def test_extract_carrier_line_items_pivots_numbered_hcpcs_columns(self) -> None:
        row = {
            "HCPCS_CD_1": "99283",
            "PRF_PHYSN_NPI_1": "9812739123",
            "LINE_NCH_PMT_AMT_1": "120.00",
            "LINE_BENE_PTB_DDCTBL_AMT_1": "0.00",
            "HCPCS_CD_2": "93000",
            "LINE_NCH_PMT_AMT_2": "25.00",
            "LINE_BENE_PTB_DDCTBL_AMT_2": "0.00",
        }

        line_items = extract_carrier_line_items(row, "196661176988405")

        self.assertEqual(len(line_items), 2)
        self.assertEqual(line_items[0]["code"], "99283")
        self.assertEqual(line_items[0]["performing_physician_npi"], "9812739123")
        self.assertEqual(line_items[0]["medicare_payment_amount"], 120.0)
        self.assertEqual(line_items[1]["code"], "93000")
        self.assertEqual(line_items[1]["line_number"], 2)

    def test_merge_facility_anchors_combines_duplicate_claim_rows(self) -> None:
        existing = {
            "cluster_id": "abc-outpatient-1",
            "desynpuf_id": "ABC",
            "anchor_claim_type": "outpatient",
            "anchor_claim_id": "1",
            "anchor_from_date": None,
            "anchor_thru_date": None,
            "carrier_match_from_date": None,
            "carrier_match_thru_date": None,
            "anchor_year": None,
            "claims": [{"claim_type": "outpatient", "claim_id": "1", "procedure_codes": []}],
            "diagnoses": [],
            "procedure_codes": [{"code_system": "HCPCS/CPT", "code": "99283"}],
        }
        incoming = {
            "cluster_id": "abc-outpatient-1",
            "desynpuf_id": "ABC",
            "anchor_claim_type": "outpatient",
            "anchor_claim_id": "1",
            "anchor_from_date": date(2008, 3, 14),
            "anchor_thru_date": date(2008, 3, 14),
            "carrier_match_from_date": date(2008, 3, 14),
            "carrier_match_thru_date": date(2008, 3, 14),
            "anchor_year": 2008,
            "claims": [{"claim_type": "outpatient", "claim_id": "1", "from_date": "2008-03-14"}],
            "diagnoses": [{"code_system": "ICD9", "code": "78650"}],
            "procedure_codes": [{"code_system": "HCPCS/CPT", "code": "93000"}],
        }

        merged = merge_facility_anchors(existing, incoming)

        self.assertEqual(merged["anchor_from_date"], date(2008, 3, 14))
        self.assertEqual(merged["carrier_match_from_date"], date(2008, 3, 14))
        self.assertEqual(merged["anchor_year"], 2008)
        self.assertEqual(len(merged["claims"]), 2)
        self.assertEqual(
            {item["code"] for item in merged["procedure_codes"]},
            {"99283", "93000"},
        )

    def test_download_file_if_missing_skips_existing_zip(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            spec = dataset_file_specs(1)[0]
            path = dataset_root / spec.filename
            path.write_bytes(b"already here")
            session = MagicMock()
            logger = CliLogger("test", use_color=False)

            result = download_file_if_missing(session, spec, dataset_root, logger)

            self.assertEqual(result, path)
            session.get.assert_not_called()

    def test_select_final_anchors_prefers_clusters_with_carrier_claims(self) -> None:
        anchors = [
            {
                "cluster_id": "without-carrier",
                "claims": [{"claim_type": "outpatient"}],
            },
            {
                "cluster_id": "with-carrier",
                "claims": [{"claim_type": "outpatient"}, {"claim_type": "carrier"}],
            },
        ]

        selected = select_final_anchors(anchors, 1)

        self.assertEqual([anchor["cluster_id"] for anchor in selected], ["with-carrier"])

    def test_sample_encounter_clusters_builds_holistic_bundle_from_fixture_zips(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "DE-SynPUF"
            self._write_fixture_dataset(dataset_root)

            records, stats = sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            self.assertEqual(stats["final_clusters"], 1)
            self.assertEqual(stats["clusters_with_carrier"], 1)
            self.assertEqual(len(records), 1)

            record = records[0]
            self.assertEqual(record["desynpuf_id"], "00013D2EFD8E45D1")
            self.assertEqual(record["encounter"]["anchor_claim_type"], "outpatient")
            self.assertEqual(record["beneficiary"]["sex_code"], "1")
            self.assertEqual(record["summary"]["has_carrier_claim"], True)
            self.assertEqual(record["summary"]["carrier_line_item_count"], 1)
            self.assertEqual(
                {entry["code"] for entry in record["code_cluster"]["diagnosis_codes"]},
                {"78650"},
            )
            self.assertEqual(
                {entry["code"] for entry in record["code_cluster"]["hcpcs_cpt_codes"]},
                {"93000", "99283"},
            )
            self.assertEqual(len(record["claims"]), 2)
            cache_path = default_encounter_cache_path(dataset_root, 1, 0)
            metadata_path = encounter_cache_metadata_path(cache_path)
            self.assertTrue(cache_path.exists())
            self.assertTrue(metadata_path.exists())

    def test_sample_encounter_clusters_reuses_existing_encounter_cache(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "DE-SynPUF"
            self._write_fixture_dataset(dataset_root)

            first_records, _ = sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            for spec in dataset_file_specs(1):
                raw_zip = dataset_root / spec.filename
                if raw_zip.exists():
                    raw_zip.unlink()
            extracted_root = default_extracted_root(dataset_root, 1)
            if extracted_root.exists():
                shutil.rmtree(extracted_root)

            second_records, second_stats = sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            self.assertEqual(first_records, second_records)
            self.assertEqual(second_stats["total_cache_records"], 1)

    def test_sample_encounter_clusters_rebuilds_stale_cache_metadata(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "DE-SynPUF"
            self._write_fixture_dataset(dataset_root)

            sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            cache_path = default_encounter_cache_path(dataset_root, 1, 0)
            metadata_path = encounter_cache_metadata_path(cache_path)
            with metadata_path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "schema_version": ENCOUNTER_CACHE_SCHEMA_VERSION - 1,
                        "sample_number": 1,
                        "carrier_match_window_days": 0,
                    },
                    handle,
                )

            sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata["schema_version"], ENCOUNTER_CACHE_SCHEMA_VERSION)

    def test_sample_encounter_clusters_merges_duplicate_outpatient_rows(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "DE-SynPUF"
            self._write_fixture_dataset(
                dataset_root,
                extra_outpatient_rows=[
                    {
                        "DESYNPUF_ID": "00013D2EFD8E45D1",
                        "CLM_ID": "196661176988405",
                        "CLM_FROM_DT": "",
                        "CLM_THRU_DT": "",
                        "PRVDR_NUM": "010001",
                        "CLM_PMT_AMT": "50.00",
                        "NCH_BENE_PTB_DDCTBL_AMT": "",
                        "NCH_BENE_PTB_COINSRNC_AMT": "",
                        "ICD9_DGNS_CD_1": "",
                        "HCPCS_CD_1": "99283",
                        "REV_CNTR_1": "",
                        "REV_CNTR_DT_1": "",
                        "REV_CNTR_PMT_AMT_1": "",
                        "REV_CNTR_TOT_CHRG_AMT_1": "",
                    }
                ],
            )

            records, stats = sample_encounter_clusters(
                dataset_root,
                sample_number=1,
                sample_size=1,
                seed=0,
                carrier_match_window_days=0,
            )

            self.assertEqual(stats["final_clusters"], 1)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["encounter"]["from_date"], "2008-03-14")
            self.assertEqual(record["summary"]["has_carrier_claim"], True)
            self.assertEqual(len(record["claims"]), 3)

    def _write_fixture_dataset(
        self,
        dataset_root: Path,
        *,
        extra_outpatient_rows: list[dict[str, str]] | None = None,
        extra_inpatient_rows: list[dict[str, str]] | None = None,
    ) -> None:
        specs = {spec.filename: spec for spec in dataset_file_specs(1)}

        beneficiary_fields = [
            "DESYNPUF_ID",
            "BENE_BIRTH_DT",
            "BENE_SEX_IDENT_CD",
            "BENE_RACE_CD",
            "SP_STATE_CODE",
            "BENE_COUNTY_CD",
        ]
        outpatient_fields = [
            "DESYNPUF_ID",
            "CLM_ID",
            "CLM_FROM_DT",
            "CLM_THRU_DT",
            "PRVDR_NUM",
            "CLM_PMT_AMT",
            "NCH_BENE_PTB_DDCTBL_AMT",
            "NCH_BENE_PTB_COINSRNC_AMT",
            "ICD9_DGNS_CD_1",
            "HCPCS_CD_1",
            "REV_CNTR_1",
            "REV_CNTR_DT_1",
            "REV_CNTR_PMT_AMT_1",
            "REV_CNTR_TOT_CHRG_AMT_1",
        ]
        inpatient_fields = [
            "DESYNPUF_ID",
            "CLM_ID",
            "CLM_FROM_DT",
            "CLM_THRU_DT",
            "CLM_ADMSN_DT",
            "NCH_BENE_DSCHRG_DT",
            "PRVDR_NUM",
            "CLM_PMT_AMT",
            "NCH_BENE_IP_DDCTBL_AMT",
            "ICD9_DGNS_CD_1",
            "ICD9_PRCDR_CD_1",
        ]
        carrier_fields = [
            "DESYNPUF_ID",
            "CLM_ID",
            "CLM_FROM_DT",
            "CLM_THRU_DT",
            "ICD9_DGNS_CD_1",
            "HCPCS_CD_1",
            "PRF_PHYSN_NPI_1",
            "LINE_NCH_PMT_AMT_1",
            "LINE_BENE_PTB_DDCTBL_AMT_1",
        ]

        write_zip_csv(
            dataset_root / specs["DE1_0_2008_Beneficiary_Summary_File_Sample_1.zip"].filename,
            "beneficiary_2008.csv",
            beneficiary_fields,
            [
                {
                    "DESYNPUF_ID": "00013D2EFD8E45D1",
                    "BENE_BIRTH_DT": "19400101",
                    "BENE_SEX_IDENT_CD": "1",
                    "BENE_RACE_CD": "1",
                    "SP_STATE_CODE": "05",
                    "BENE_COUNTY_CD": "001",
                }
            ],
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2009_Beneficiary_Summary_File_Sample_1.zip"].filename,
            "beneficiary_2009.csv",
            beneficiary_fields,
            [],
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2010_Beneficiary_Summary_File_Sample_1.zip"].filename,
            "beneficiary_2010.csv",
            beneficiary_fields,
            [],
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2008_to_2010_Outpatient_Claims_Sample_1.zip"].filename,
            "outpatient.csv",
            outpatient_fields,
            [
                {
                    "DESYNPUF_ID": "00013D2EFD8E45D1",
                    "CLM_ID": "196661176988405",
                    "CLM_FROM_DT": "20080314",
                    "CLM_THRU_DT": "20080314",
                    "PRVDR_NUM": "010001",
                    "CLM_PMT_AMT": "250.00",
                    "NCH_BENE_PTB_DDCTBL_AMT": "50.00",
                    "NCH_BENE_PTB_COINSRNC_AMT": "20.00",
                    "ICD9_DGNS_CD_1": "78650",
                    "HCPCS_CD_1": "99283",
                    "REV_CNTR_1": "0450",
                    "REV_CNTR_DT_1": "20080314",
                    "REV_CNTR_PMT_AMT_1": "120.00",
                    "REV_CNTR_TOT_CHRG_AMT_1": "300.00",
                }
            ]
            + list(extra_outpatient_rows or []),
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2008_to_2010_Inpatient_Claims_Sample_1.zip"].filename,
            "inpatient.csv",
            inpatient_fields,
            list(extra_inpatient_rows or []),
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2008_to_2010_Carrier_Claims_Sample_1A.zip"].filename,
            "carrier_1a.csv",
            carrier_fields,
            [
                {
                    "DESYNPUF_ID": "00013D2EFD8E45D1",
                    "CLM_ID": "296661176988405",
                    "CLM_FROM_DT": "20080314",
                    "CLM_THRU_DT": "20080314",
                    "ICD9_DGNS_CD_1": "78650",
                    "HCPCS_CD_1": "93000",
                    "PRF_PHYSN_NPI_1": "9812739123",
                    "LINE_NCH_PMT_AMT_1": "25.00",
                    "LINE_BENE_PTB_DDCTBL_AMT_1": "0.00",
                }
            ],
        )
        write_zip_csv(
            dataset_root / specs["DE1_0_2008_to_2010_Carrier_Claims_Sample_1B.zip"].filename,
            "carrier_1b.csv",
            carrier_fields,
            [],
        )


if __name__ == "__main__":
    unittest.main()
