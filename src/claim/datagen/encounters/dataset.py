from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Iterator
import zipfile

import requests
from requests.adapters import HTTPAdapter
from tqdm.utils import CallbackIOWrapper
from urllib3.util.retry import Retry

from claim.datagen import io as datagen_io
from claim.datagen.encounters import constants
from claim.datagen.encounters.contracts import DatasetFileSpec
from claim.datagen.encounters.paths import default_extracted_root
from claim.cli_logging import CliLogger


def dataset_file_specs(sample_number: int) -> tuple[DatasetFileSpec, ...]:
    if sample_number != constants.DEFAULT_SAMPLE_NUMBER:
        raise ValueError("Only DE-SynPUF Sample 1 is supported right now.")
    return constants.SAMPLE_1_FILE_SPECS


def extracted_csv_path(extracted_root: Path, spec: DatasetFileSpec) -> Path:
    return extracted_root / f"{Path(spec.filename).stem}.csv"


def build_download_session() -> requests.Session:
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
    session.headers.update({"User-Agent": constants.ENCOUNTER_DOWNLOAD_USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def ensure_dataset_files(
    dataset_root: Path,
    sample_number: int,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> dict[str, list[tuple[DatasetFileSpec, Path]]]:
    dataset_root.mkdir(parents=True, exist_ok=True)
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]] = {}

    with build_download_session() as session:
        for spec in dataset_file_specs(sample_number):
            path = download_file_if_missing(
                session,
                spec,
                dataset_root,
                logger,
                show_progress=show_progress,
            )
            grouped_paths.setdefault(spec.category, []).append((spec, path))

    return grouped_paths


def download_file_if_missing(
    session: requests.Session,
    spec: DatasetFileSpec,
    dataset_root: Path,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> Path:
    output_path = dataset_root / spec.filename
    if output_path.exists():
        logger.info("Skipping existing dataset ZIP", file=spec.filename)
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.part")
    logger.info("Downloading dataset ZIP", file=spec.filename, url=spec.url)

    response = session.get(
        spec.url,
        timeout=constants.DEFAULT_TIMEOUT_SECONDS,
        stream=True,
    )
    try:
        response.raise_for_status()
        total_bytes = parse_int(response.headers.get("Content-Length"))
        progress = datagen_io.create_progress_bar(
            description=f"download {spec.filename}",
            total=total_bytes,
            unit="B",
            show_progress=show_progress,
        )
        try:
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=constants.DEFAULT_STREAM_CHUNK_SIZE
                ):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    if progress is not None:
                        progress.update(len(chunk))
        finally:
            datagen_io.close_progress_bar(progress)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        response.close()

    temp_path.replace(output_path)
    logger.success("Downloaded dataset ZIP", file=spec.filename)
    return output_path


def resolve_extracted_paths(
    dataset_root: Path,
    sample_number: int,
) -> dict[str, list[tuple[DatasetFileSpec, Path]]]:
    extracted_root = default_extracted_root(dataset_root, sample_number)
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]] = {}
    for spec in dataset_file_specs(sample_number):
        path = extracted_csv_path(extracted_root, spec)
        if not path.exists():
            raise FileNotFoundError(f"Missing extracted dataset file: {path}")
        grouped_paths.setdefault(spec.category, []).append((spec, path))
    return grouped_paths


def ensure_extracted_csv_files(
    dataset_root: Path,
    sample_number: int,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> dict[str, list[tuple[DatasetFileSpec, Path]]]:
    extracted_root = default_extracted_root(dataset_root, sample_number)
    grouped_paths: dict[str, list[tuple[DatasetFileSpec, Path]]] = {}
    missing_specs: list[DatasetFileSpec] = []

    for spec in dataset_file_specs(sample_number):
        path = extracted_csv_path(extracted_root, spec)
        if path.exists():
            grouped_paths.setdefault(spec.category, []).append((spec, path))
        else:
            missing_specs.append(spec)

    if not missing_specs:
        return grouped_paths

    ensure_dataset_files(
        dataset_root,
        sample_number,
        logger,
        show_progress=show_progress,
    )
    extracted_root.mkdir(parents=True, exist_ok=True)

    for spec in missing_specs:
        extract_csv_from_zip_if_missing(
            dataset_root / spec.filename,
            spec,
            extracted_root,
            logger,
            show_progress=show_progress,
        )

    return resolve_extracted_paths(dataset_root, sample_number)


def extract_csv_from_zip_if_missing(
    zip_path: Path,
    spec: DatasetFileSpec,
    extracted_root: Path,
    logger: CliLogger,
    *,
    show_progress: bool,
) -> Path:
    output_path = extracted_csv_path(extracted_root, spec)
    if output_path.exists():
        logger.info("Skipping existing extracted CSV", file=output_path.name)
        return output_path

    extracted_root.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.part")
    member_name = find_csv_member_name(zip_path)
    logger.info("Extracting dataset CSV", file=output_path.name, source=zip_path.name)

    with zipfile.ZipFile(zip_path) as archive:
        info = archive.getinfo(member_name)
        progress = datagen_io.create_progress_bar(
            description=f"extract {output_path.name}",
            total=info.file_size,
            unit="B",
            show_progress=show_progress,
        )
        try:
            with archive.open(member_name) as source_handle:
                with temp_path.open("wb") as output_handle:
                    while True:
                        chunk = source_handle.read(constants.DEFAULT_STREAM_CHUNK_SIZE)
                        if not chunk:
                            break
                        output_handle.write(chunk)
                        if progress is not None:
                            progress.update(len(chunk))
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise
        finally:
            datagen_io.close_progress_bar(progress)

    temp_path.replace(output_path)
    logger.success("Extracted dataset CSV", file=output_path.name)
    return output_path


def find_csv_member_name(zip_path: Path) -> str:
    with zipfile.ZipFile(zip_path) as archive:
        csv_members = [
            name
            for name in archive.namelist()
            if not name.endswith("/") and name.lower().endswith(".csv")
        ]

    if not csv_members:
        raise ValueError(f"No CSV member found inside {zip_path}")
    if len(csv_members) != 1:
        raise ValueError(
            f"Expected exactly one CSV member inside {zip_path}, found {len(csv_members)}"
        )
    return csv_members[0]


def normalize_csv_row(row: dict[str, str | None]) -> dict[str, str]:
    return {key: "" if value is None else value for key, value in row.items()}


def iter_csv_rows_from_zip(zip_path: Path) -> Iterator[dict[str, str]]:
    member_name = find_csv_member_name(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open(member_name) as raw_handle:
            text_handle = io.TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
            reader = csv.DictReader(text_handle)
            for row in reader:
                yield normalize_csv_row(row)


def iter_csv_rows_from_csv(
    path: Path,
    *,
    description: str,
    show_progress: bool,
) -> Iterator[dict[str, str]]:
    if not show_progress:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield normalize_csv_row(row)
        return

    total_bytes = path.stat().st_size
    progress = datagen_io.create_progress_bar(
        description=description,
        total=total_bytes,
        unit="B",
        show_progress=show_progress,
    )
    try:
        with path.open("rb") as raw_handle:
            wrapped_handle = (
                CallbackIOWrapper(progress.update, raw_handle, "read")
                if progress is not None
                else raw_handle
            )
            text_handle = io.TextIOWrapper(
                wrapped_handle,
                encoding="utf-8-sig",
                newline="",
            )
            reader = csv.DictReader(text_handle)
            for row in reader:
                yield normalize_csv_row(row)
    finally:
        datagen_io.close_progress_bar(progress)


def parse_int(value: str | None) -> int | None:
    normalized = clean_text(value, allow_all_zero=True)
    if normalized is None:
        return None
    try:
        return int(float(normalized))
    except ValueError:
        return None


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
