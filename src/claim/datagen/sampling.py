from __future__ import annotations

from pathlib import Path
import random
from typing import Any

from claim.datagen import constants
from claim.datagen.io import (
    iter_jsonl_records,
    upgrade_shifted_sampled_records,
    write_jsonl_atomic,
)
from claim.datagen.utils import encounter_regime, shift_encounter_dates
from claim.cli_logging import CliLogger


def encounter_has_hcpcs_codes(record: dict[str, Any]) -> bool:
    summary = record.get("summary") or {}
    return int(summary.get("unique_hcpcs_cpt_code_count") or 0) > 0


def encounter_is_eligible(record: dict[str, Any]) -> bool:
    regime = encounter_regime(record)
    if regime is None:
        return encounter_has_hcpcs_codes(record)
    if regime == "outpatient":
        return encounter_has_hcpcs_codes(record)
    if regime == "inpatient":
        summary = record.get("summary") or {}
        return bool(summary.get("has_facility_claim"))
    return False


def sample_encounters_from_cache(
    encounter_cache_path: Path,
    *,
    sample_size: int,
    seed: int,
    logger: CliLogger,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    regime_candidates: dict[str, list[dict[str, Any]]] = {
        "inpatient": [],
        "outpatient": [],
    }
    scanned = 0
    eligible = 0

    for scanned, record in enumerate(
        iter_jsonl_records(
            encounter_cache_path,
            description="sample encounters",
            show_progress=show_progress,
        ),
        start=1,
    ):
        if not encounter_is_eligible(record):
            continue
        regime = encounter_regime(record) or "outpatient"
        if regime not in regime_candidates:
            continue
        eligible += 1
        regime_candidates[regime].append(record)

    rng = random.Random(seed)
    for records in regime_candidates.values():
        rng.shuffle(records)

    inpatient_target = (sample_size + 1) // 2
    outpatient_target = max(0, sample_size - inpatient_target)
    reservoir = (
        regime_candidates["inpatient"][:inpatient_target]
        + regime_candidates["outpatient"][:outpatient_target]
    )
    if len(reservoir) < sample_size:
        remaining = (
            regime_candidates["inpatient"][inpatient_target:]
            + regime_candidates["outpatient"][outpatient_target:]
        )
        reservoir.extend(remaining[: max(0, sample_size - len(reservoir))])

    reservoir.sort(key=lambda item: str(item.get("cluster_id", "")))
    logger.info(
        "Sampled eligible encounters",
        scanned=scanned,
        eligible=eligible,
        sampled=len(reservoir),
        inpatient_sampled=sum(1 for item in reservoir if encounter_regime(item) == "inpatient"),
        outpatient_sampled=sum(1 for item in reservoir if encounter_regime(item) == "outpatient"),
    )
    return reservoir, {
        "records_scanned": scanned,
        "eligible_records": eligible,
        "sampled_records": len(reservoir),
        "eligible_inpatient_records": len(regime_candidates["inpatient"]),
        "eligible_outpatient_records": len(regime_candidates["outpatient"]),
        "sampled_inpatient_records": sum(
            1 for item in reservoir if encounter_regime(item) == "inpatient"
        ),
        "sampled_outpatient_records": sum(
            1
            for item in reservoir
            if (encounter_regime(item) or "outpatient") == "outpatient"
        ),
    }


def ensure_sampled_encounters(
    sampled_path: Path,
    encounter_cache_path: Path,
    *,
    sample_size: int,
    seed: int,
    overwrite: bool,
    logger: CliLogger,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if sampled_path.exists() and not overwrite:
        records = list(
            iter_jsonl_records(
                sampled_path,
                description="load sampled encounters",
                show_progress=show_progress,
            )
        )
        shifted_records = upgrade_shifted_sampled_records(
            sampled_path,
            records,
            logger=logger,
        )
        logger.info(
            "Reusing sampled encounter subset",
            path=sampled_path,
            sampled=len(shifted_records),
        )
        return shifted_records, {
            "records_scanned": 0,
            "eligible_records": len(shifted_records),
            "sampled_records": len(shifted_records),
        }

    records, stats = sample_encounters_from_cache(
        encounter_cache_path,
        sample_size=sample_size,
        seed=seed,
        logger=logger,
        show_progress=show_progress,
    )
    shifted_records = [shift_encounter_dates(record) for record in records]
    write_jsonl_atomic(sampled_path, shifted_records)
    logger.success(
        "Wrote sampled encounter subset",
        path=sampled_path,
        sampled=len(shifted_records),
        date_shift_years=constants.DEFAULT_DATE_SHIFT_YEARS,
    )
    return shifted_records, stats
