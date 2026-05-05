from __future__ import annotations

from pathlib import Path

from claim.datagen import io as datagen_io
from claim.datagen.encounters import constants, dataset, paths, storage
from claim.datagen.encounters.contracts import (
    EncounterBuildConfig,
    EncounterCacheMetadata,
)
from claim.cli_logging import CliLogger


def read_encounter_cache_metadata(
    encounter_cache_path: Path,
) -> EncounterCacheMetadata | None:
    metadata_path = paths.encounter_cache_metadata_path(encounter_cache_path)
    if not metadata_path.exists():
        return None
    payload = datagen_io.load_json_file(metadata_path)
    if not isinstance(payload, dict):
        return None
    return EncounterCacheMetadata.from_payload(payload)


def build_encounter_cache(
    config: EncounterBuildConfig,
    logger: CliLogger,
) -> EncounterCacheMetadata:
    grouped_paths = dataset.ensure_extracted_csv_files(
        config.dataset_root,
        config.sample_number,
        logger,
        show_progress=config.show_progress,
    )
    encounter_paths = paths.build_paths(
        config.dataset_root,
        config.encounter_cache_path,
        config.sample_number,
    )
    if encounter_paths.build_database_path.exists():
        encounter_paths.build_database_path.unlink()

    logger.info(
        "Building encounter cache index",
        build_db_path=encounter_paths.build_database_path,
    )
    connection = storage.initialize_build_database(encounter_paths.build_database_path)
    try:
        beneficiary_rows = storage.load_beneficiaries_into_database(
            connection,
            grouped_paths,
            logger,
            show_progress=config.show_progress,
        )
        facility_claims = storage.load_facility_anchors_into_database(
            connection,
            grouped_paths,
            logger,
            show_progress=config.show_progress,
        )
        carrier_claims = storage.load_carrier_claims_into_database(
            connection,
            grouped_paths,
            logger,
            show_progress=config.show_progress,
        )
        materialized_stats = storage.materialize_encounter_cache(
            connection,
            config.encounter_cache_path,
            config.sample_number,
            config.carrier_match_window_days,
            logger,
            show_progress=config.show_progress,
        )
    finally:
        connection.close()
        if encounter_paths.build_database_path.exists():
            encounter_paths.build_database_path.unlink()

    metadata = EncounterCacheMetadata(
        schema_version=constants.ENCOUNTER_CACHE_SCHEMA_VERSION,
        sample_number=config.sample_number,
        carrier_match_window_days=config.carrier_match_window_days,
        beneficiary_rows_indexed=beneficiary_rows,
        facility_claims_indexed=facility_claims,
        carrier_claims_indexed=carrier_claims,
        total_records=materialized_stats["total_records"],
        clusters_with_carrier=materialized_stats["clusters_with_carrier"],
        encounter_cache_path=str(config.encounter_cache_path),
        extracted_root=str(encounter_paths.extracted_root),
    )
    datagen_io.write_json_atomic(encounter_paths.metadata_path, metadata.as_payload())
    logger.success(
        "Wrote encounter cache metadata",
        metadata_path=encounter_paths.metadata_path,
    )
    return metadata


def ensure_encounter_cache(
    config: EncounterBuildConfig,
    logger: CliLogger,
) -> EncounterCacheMetadata:
    metadata = read_encounter_cache_metadata(config.encounter_cache_path)
    if config.encounter_cache_path.exists() and not config.rebuild:
        if metadata is not None and metadata.matches(
            schema_version=constants.ENCOUNTER_CACHE_SCHEMA_VERSION,
            sample_number=config.sample_number,
            carrier_match_window_days=config.carrier_match_window_days,
        ):
            logger.info(
                "Reusing existing encounter cache",
                encounter_cache_path=config.encounter_cache_path,
            )
            return metadata
        logger.warning(
            "Encounter cache metadata is stale; rebuilding cache",
            encounter_cache_path=config.encounter_cache_path,
            metadata_path=paths.encounter_cache_metadata_path(config.encounter_cache_path),
            expected_schema_version=constants.ENCOUNTER_CACHE_SCHEMA_VERSION,
        )

    if config.encounter_cache_path.exists():
        logger.warning(
            "Rebuilding encounter cache",
            encounter_cache_path=config.encounter_cache_path,
        )

    return build_encounter_cache(config, logger)
