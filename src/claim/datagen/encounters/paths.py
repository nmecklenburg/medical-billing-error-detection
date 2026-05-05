from __future__ import annotations

from pathlib import Path

from claim.datagen.encounters import constants
from claim.datagen.encounters.contracts import EncounterPaths


def default_extracted_root(dataset_root: Path, sample_number: int) -> Path:
    return dataset_root / constants.DEFAULT_EXTRACTED_DIRNAME / f"sample-{sample_number}"


def default_encounter_cache_path(
    dataset_root: Path,
    sample_number: int,
    carrier_match_window_days: int,
) -> Path:
    filename = (
        f"sample-{sample_number}-encounters-window-{carrier_match_window_days}.jsonl"
    )
    return dataset_root / filename


def resolve_encounter_cache_path(
    dataset_root: Path,
    sample_number: int,
    carrier_match_window_days: int,
    encounter_cache_path: Path | None = None,
) -> Path:
    if encounter_cache_path is not None:
        return encounter_cache_path
    return default_encounter_cache_path(
        dataset_root,
        sample_number,
        carrier_match_window_days,
    )


def encounter_cache_metadata_path(encounter_cache_path: Path) -> Path:
    return encounter_cache_path.with_suffix(".metadata.json")


def build_database_path(encounter_cache_path: Path) -> Path:
    return encounter_cache_path.with_suffix(".build.sqlite.part")


def build_paths(
    dataset_root: Path,
    encounter_cache_path: Path,
    sample_number: int,
) -> EncounterPaths:
    return EncounterPaths(
        dataset_root=dataset_root,
        extracted_root=default_extracted_root(dataset_root, sample_number),
        encounter_cache_path=encounter_cache_path,
        metadata_path=encounter_cache_metadata_path(encounter_cache_path),
        build_database_path=build_database_path(encounter_cache_path),
    )
