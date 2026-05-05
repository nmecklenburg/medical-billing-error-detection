import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from claim.datagen import io as datagen_io
from claim.cli_logging import CliLogger


class DatagenIoTests(unittest.TestCase):
    def test_cleanup_stale_partial_files_removes_nested_partials(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            nested = root / "nested"
            nested.mkdir()
            keep = nested / "rows.jsonl"
            keep.write_text("{}", encoding="utf-8")
            stale_files = [
                root / "bills.jsonl.part",
                nested / "aftercare.jsonl.part",
            ]
            for path in stale_files:
                path.write_text("partial", encoding="utf-8")

            removed = datagen_io.cleanup_stale_partial_files([root])

            self.assertEqual(set(removed), set(stale_files))
            self.assertTrue(keep.exists())
            for path in stale_files:
                self.assertFalse(path.exists())

    def test_repair_jsonl_trailing_partial_line_truncates_invalid_tail(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "rows.jsonl"
            path.write_text(
                '{"case_id": "case-1"}\n{"case_id": "case-2"}\n{"case_id": "case',
                encoding="utf-8",
            )

            trimmed = datagen_io.repair_jsonl_trailing_partial_line(
                path,
                logger=CliLogger("test", use_color=False),
                label="test rows",
            )

            self.assertTrue(trimmed)
            self.assertEqual(
                path.read_text(encoding="utf-8").splitlines(),
                ['{"case_id": "case-1"}', '{"case_id": "case-2"}'],
            )

    def test_load_existing_state_and_validate_resume_subset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            logger = CliLogger("test", use_color=False)
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

            case_ids, scenario_counts = datagen_io.load_existing_bills_state(
                bills_path,
                logger=logger,
            )

            self.assertEqual(case_ids, {"case-1", "case-2"})
            self.assertEqual(scenario_counts, {"in_network": 1, "self_pay": 1})

            aftercare_path = Path(tmpdir) / "aftercare.jsonl"
            aftercare_path.write_text(json.dumps({"case_id": "case-1"}) + "\n", encoding="utf-8")

            aftercare_ids = datagen_io.load_existing_case_ids(
                aftercare_path,
                logger=logger,
                label="aftercare",
            )

            self.assertEqual(aftercare_ids, {"case-1"})

            datagen_io.validate_resumable_case_ids(
                {"case-1", "case-2", "case-3"},
                case_ids,
                path=bills_path,
                label="bill",
            )
            with self.assertRaises(RuntimeError):
                datagen_io.validate_resumable_case_ids(
                    {"case-1"},
                    case_ids,
                    path=bills_path,
                    label="bill",
                )


if __name__ == "__main__":
    unittest.main()
