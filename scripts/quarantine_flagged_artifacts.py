from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any


DEFAULT_DATASET_DIR = Path("outputs/arxiv-ready-clean-bill-gen-1k-v3")
DEFAULT_FLAGS_PATH = Path("clinical_plausibility_flags.minimax-m2.7.json")
DEFAULT_QUARANTINE_ROOT = Path("quarantine")
DEFAULT_LABEL = "clinical-plausibility"

TARGET_FILES = (
    "generated-code-clusters.jsonl",
    "inpatient_bills.jsonl",
    "outpatient_bills.jsonl",
    "aftercare-summaries.jsonl",
    "failures.jsonl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quarantine flagged case IDs out of a generated artifact directory by "
            "backing up top-level JSONL files, writing removed records into a "
            "timestamped quarantine bundle, and rewriting the live files with survivors."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Artifact directory to rewrite in place.",
    )
    parser.add_argument(
        "--flags-path",
        type=Path,
        default=DEFAULT_FLAGS_PATH,
        help="JSON file containing flagged_problems with case_id values.",
    )
    parser.add_argument(
        "--quarantine-root",
        type=Path,
        default=DEFAULT_QUARANTINE_ROOT,
        help="Root directory under which the quarantine bundle will be created.",
    )
    parser.add_argument(
        "--label",
        default=DEFAULT_LABEL,
        help="Short label to include in the quarantine directory name.",
    )
    parser.add_argument(
        "--timestamp",
        required=True,
        help="UTC timestamp suffix, for example 20260504T215759Z.",
    )
    return parser.parse_args()


def load_flagged_case_ids(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    flagged_problems = payload.get("flagged_problems") or []
    seen: set[str] = set()
    ordered: list[str] = []
    for problem in flagged_problems:
        case_id = str(problem.get("case_id") or "").strip()
        if not case_id or case_id in seen:
            continue
        seen.add(case_id)
        ordered.append(case_id)
    return ordered


def partition_jsonl_records(
    path: Path,
    *,
    flagged_case_ids: set[str],
) -> tuple[list[str], list[str], int]:
    kept_lines: list[str] = []
    removed_lines: list[str] = []
    total_count = 0
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.rstrip("\n")
            if not stripped:
                continue
            total_count += 1
            record = json.loads(stripped)
            if str(record.get("case_id") or "").strip() in flagged_case_ids:
                removed_lines.append(stripped)
            else:
                kept_lines.append(stripped)
    return kept_lines, removed_lines, total_count


def write_jsonl_lines(path: Path, lines: list[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line)
            handle.write("\n")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    flags_path = args.flags_path.resolve()
    quarantine_root = args.quarantine_root.resolve()
    flagged_case_ids = load_flagged_case_ids(flags_path)
    flagged_case_id_set = set(flagged_case_ids)

    quarantine_dir = quarantine_root / f"{dataset_dir.name}-{args.label}-{args.timestamp}"
    quarantine_dir.mkdir(parents=True, exist_ok=False)

    summary: dict[str, Any] = {
        "source_output_dir": str(dataset_dir),
        "created_at": args.timestamp,
        "flag_source": str(flags_path),
        "flagged_case_ids": flagged_case_ids,
        "files": {},
    }

    for filename in TARGET_FILES:
        source_path = dataset_dir / filename
        if not source_path.exists():
            continue
        kept_lines, removed_lines, total_count = partition_jsonl_records(
            source_path,
            flagged_case_ids=flagged_case_id_set,
        )
        backup_path = quarantine_dir / f"{filename}.backup"
        removed_path = quarantine_dir / f"{filename}.removed.jsonl"
        shutil.copy2(source_path, backup_path)
        write_jsonl_lines(removed_path, removed_lines)
        write_jsonl_lines(source_path, kept_lines)
        summary["files"][filename] = {
            "original_count": total_count,
            "removed_count": len(removed_lines),
            "remaining_count": len(kept_lines),
            "backup_path": str(backup_path),
            "removed_records_path": str(removed_path),
        }

    summary_path = quarantine_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "quarantine_dir": str(quarantine_dir),
                "flagged_case_count": len(flagged_case_ids),
                "files": summary["files"],
            }
        )
    )


if __name__ == "__main__":
    main()
