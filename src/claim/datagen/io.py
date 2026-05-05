from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from tqdm.auto import tqdm
from urllib3.util.retry import Retry

from claim.datagen import constants
from claim.datagen.contracts import OutputPaths
from claim.datagen.utils import shift_encounter_dates, usage_accumulator, usage_totals
from claim.cli_logging import CliLogger
from claim.gold_bill_support import normalize_text
from claim.gold_bill_trilliant import TRILLIANT_USER_AGENT


def effective_sample_size(sample_size: int, test_one: bool) -> int:
    return 1 if test_one else sample_size


def sampled_encounters_path(
    encounter_cache_path: Path,
    sample_size: int,
    seed: int,
    test_one: bool,
) -> Path:
    actual_size = effective_sample_size(sample_size, test_one)
    output_dir = encounter_cache_path.parent / "gold-bill-samples"
    suffix = "test-one" if test_one else f"sample-{actual_size}"
    return output_dir / f"{encounter_cache_path.stem}-{suffix}-seed-{seed}.jsonl"


def default_inpatient_bills_path(output_dir: Path) -> Path:
    return output_dir / "inpatient_bills.jsonl"


def default_outpatient_bills_path(output_dir: Path) -> Path:
    return output_dir / "outpatient_bills.jsonl"


def bill_paths(output_dir: Path) -> tuple[Path, Path]:
    return (
        default_inpatient_bills_path(output_dir),
        default_outpatient_bills_path(output_dir),
    )


def default_generated_clusters_path(output_dir: Path) -> Path:
    return output_dir / "generated-code-clusters.jsonl"


def default_aftercare_path(output_dir: Path) -> Path:
    return output_dir / "aftercare-summaries.jsonl"


def default_failures_path(output_dir: Path) -> Path:
    return output_dir / "failures.jsonl"


def default_run_metadata_path(output_dir: Path) -> Path:
    return output_dir / "run-metadata.json"


def default_description_cache_path(reference_data_dir: Path) -> Path:
    return reference_data_dir / "code-description-cache.json"


def default_trilliant_cache_path(trilliant_cache_dir: Path) -> Path:
    return trilliant_cache_dir / "trilliant-cache.duckdb"


def build_output_paths(output_dir: Path) -> OutputPaths:
    return OutputPaths(
        clusters=default_generated_clusters_path(output_dir),
        inpatient_bills=default_inpatient_bills_path(output_dir),
        outpatient_bills=default_outpatient_bills_path(output_dir),
        aftercare=default_aftercare_path(output_dir),
        failures=default_failures_path(output_dir),
        metadata=default_run_metadata_path(output_dir),
    )


def ensure_output_paths(output_dir: Path) -> OutputPaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return build_output_paths(output_dir)


def load_env_file(dotenv_path: Path) -> bool:
    return bool(load_dotenv(dotenv_path=dotenv_path, override=False))


def should_use_progress() -> bool:
    if os.environ.get("TQDM_DISABLE"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def tqdm_line_writer(line: str, stream: Any) -> None:
    tqdm.write(line, file=stream)


def create_progress_bar(
    *,
    description: str,
    total: int | None,
    unit: str,
    show_progress: bool,
) -> tqdm[Any] | None:
    if not show_progress:
        return None
    return tqdm(
        total=total,
        desc=description,
        unit=unit,
        unit_scale=unit == "B",
        dynamic_ncols=True,
        leave=False,
        position=0,
    )


def close_progress_bar(progress: tqdm[Any] | None) -> None:
    if progress is not None:
        progress.close()


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers.update({"User-Agent": TRILLIANT_USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def cleanup_stale_partial_files(paths: Iterable[Path]) -> list[Path]:
    removed: list[Path] = []
    for directory in paths:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*.part")):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.part")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")
    temp_path.replace(path)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.part")
    with temp_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=False))
            handle.write("\n")
    temp_path.replace(path)


def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class JsonlStreamWriter:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a" if append else "w", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        self.handle.write(json.dumps(row, sort_keys=False))
        self.handle.write("\n")
        self.handle.flush()

    def close(self) -> None:
        if not self.handle.closed:
            self.handle.close()

    def __enter__(self) -> JsonlStreamWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def iter_jsonl_records(
    path: Path,
    *,
    description: str,
    show_progress: bool,
) -> Iterator[dict[str, Any]]:
    if not show_progress:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    total_bytes = path.stat().st_size
    progress = create_progress_bar(
        description=description,
        total=total_bytes,
        unit="B",
        show_progress=show_progress,
    )
    try:
        with path.open("rb") as raw_handle:
            for raw_line in raw_handle:
                if progress is not None:
                    progress.update(len(raw_line))
                line = raw_line.decode("utf-8").strip()
                if line:
                    yield json.loads(line)
    finally:
        close_progress_bar(progress)


def repair_jsonl_trailing_partial_line(
    path: Path,
    *,
    logger: CliLogger | None = None,
    label: str,
) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    last_good_offset = 0
    truncated = False
    bad_line_number = 0
    with path.open("rb") as handle:
        line_number = 0
        while True:
            line_start = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                break
            line_number += 1
            stripped = raw_line.strip()
            if not stripped:
                last_good_offset = handle.tell()
                continue
            try:
                json.loads(raw_line.decode("utf-8"))
            except Exception:
                truncated = True
                bad_line_number = line_number
                last_good_offset = line_start
                break
            last_good_offset = handle.tell()
    if truncated:
        with path.open("rb+") as handle:
            handle.truncate(last_good_offset)
        if logger is not None:
            logger.warning(
                "Trimmed trailing partial JSONL content",
                file=path.name,
                label=label,
                line=bad_line_number,
            )
    return truncated


def load_existing_case_ids(
    path: Path,
    *,
    logger: CliLogger,
    label: str,
) -> set[str]:
    if not path.exists():
        return set()
    repair_jsonl_trailing_partial_line(path, logger=logger, label=label)
    case_ids: set[str] = set()
    for record in iter_jsonl_records(path, description=f"load {label}", show_progress=False):
        case_id = normalize_text(record.get("case_id"))
        if case_id:
            case_ids.add(case_id)
    return case_ids


def load_existing_bills_state(
    path: Path,
    *,
    logger: CliLogger,
) -> tuple[set[str], dict[str, int]]:
    if not path.exists():
        return set(), {}
    repair_jsonl_trailing_partial_line(path, logger=logger, label="existing bills")
    case_ids: set[str] = set()
    scenario_counts: dict[str, int] = {}
    for record in iter_jsonl_records(path, description="load existing bills", show_progress=False):
        case_id = normalize_text(record.get("case_id"))
        if not case_id or case_id in case_ids:
            continue
        case_ids.add(case_id)
        scenario = str((record.get("payer_scenario") or {}).get("scenario") or "unknown")
        scenario_counts[scenario] = scenario_counts.get(scenario, 0) + 1
    return case_ids, scenario_counts


def load_existing_split_bills_state(
    inpatient_path: Path,
    outpatient_path: Path,
    *,
    logger: CliLogger,
) -> tuple[set[str], dict[str, int]]:
    case_ids: set[str] = set()
    scenario_counts: dict[str, int] = {}
    for path, label in (
        (inpatient_path, "existing inpatient bills"),
        (outpatient_path, "existing outpatient bills"),
    ):
        path_case_ids, path_scenario_counts = load_existing_bills_state(
            path,
            logger=logger,
        )
        case_ids.update(path_case_ids)
        for scenario, count in path_scenario_counts.items():
            scenario_counts[scenario] = scenario_counts.get(scenario, 0) + count
    return case_ids, scenario_counts


def iter_bill_records(
    inpatient_path: Path | None,
    outpatient_path: Path | None,
) -> Iterator[dict[str, Any]]:
    seen_paths: set[Path] = set()
    for path, description in (
        (inpatient_path, "load inpatient bills"),
        (outpatient_path, "load outpatient bills"),
    ):
        if path is None or path in seen_paths or not path.exists():
            continue
        seen_paths.add(path)
        yield from iter_jsonl_records(path, description=description, show_progress=False)


def load_generated_cluster_records(
    path: Path,
    *,
    logger: CliLogger,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    repair_jsonl_trailing_partial_line(path, logger=logger, label="existing generated clusters")
    records: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl_records(
        path,
        description="load generated clusters",
        show_progress=False,
    ):
        case_id = normalize_text(record.get("case_id"))
        if case_id:
            records[case_id] = record
    return records


def validate_resumable_case_ids(
    sampled_case_ids: set[str],
    existing_case_ids: set[str],
    *,
    path: Path,
    label: str,
) -> None:
    unexpected = sorted(existing_case_ids - sampled_case_ids)
    if not unexpected:
        return
    preview = ", ".join(unexpected[:3])
    raise RuntimeError(
        f"Cannot resume from {path}: found {label} case IDs outside the current sampled subset "
        f"({preview}). Use --overwrite or choose a different --output-dir."
    )


def load_resumable_usage(
    metadata_path: Path,
    *,
    sampled_path: Path,
    encounter_cache_path: Path,
) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
    if not metadata_path.exists():
        return usage_accumulator(), usage_accumulator(), usage_accumulator()
    try:
        payload = load_json_file(metadata_path)
    except Exception:
        return usage_accumulator(), usage_accumulator(), usage_accumulator()
    if not isinstance(payload, dict):
        return usage_accumulator(), usage_accumulator(), usage_accumulator()
    if payload.get("sampled_encounters_path") != str(sampled_path):
        return usage_accumulator(), usage_accumulator(), usage_accumulator()
    if payload.get("encounter_cache_path") != str(encounter_cache_path):
        return usage_accumulator(), usage_accumulator(), usage_accumulator()
    summary = payload.get("summary") or {}
    cluster_usage = usage_totals((summary.get("cluster_stage") or {}).get("openai_usage"))
    pricing_usage = usage_totals((summary.get("pricing_stage") or {}).get("openai_usage"))
    if pricing_usage == usage_accumulator() and summary.get("bill_stage"):
        pricing_usage = usage_totals((summary.get("bill_stage") or {}).get("openai_usage"))
    aftercare_usage = usage_totals((summary.get("aftercare_stage") or {}).get("openai_usage"))
    return cluster_usage, pricing_usage, aftercare_usage


def upgrade_shifted_sampled_records(
    sampled_path: Path,
    records: Sequence[dict[str, Any]],
    *,
    logger: CliLogger,
) -> list[dict[str, Any]]:
    shifted_records = [shift_encounter_dates(record) for record in records]
    if shifted_records != list(records):
        write_jsonl_atomic(sampled_path, shifted_records)
        logger.info(
            "Upgraded sampled encounter subset with shifted clinical dates",
            path=sampled_path,
            sampled=len(shifted_records),
            date_shift_years=constants.DEFAULT_DATE_SHIFT_YEARS,
        )
    return shifted_records
