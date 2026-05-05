from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetFileSpec:
    filename: str
    url: str
    category: str
    year: int | None = None


@dataclass(frozen=True, slots=True)
class EncounterPaths:
    dataset_root: Path
    extracted_root: Path
    encounter_cache_path: Path
    metadata_path: Path
    build_database_path: Path


@dataclass(frozen=True, slots=True)
class EncounterBuildConfig:
    dataset_root: Path
    encounter_cache_path: Path
    sample_number: int
    carrier_match_window_days: int
    rebuild: bool
    show_progress: bool


@dataclass(frozen=True, slots=True)
class EncounterCacheMetadata:
    schema_version: int
    sample_number: int
    carrier_match_window_days: int
    beneficiary_rows_indexed: int
    facility_claims_indexed: int
    carrier_claims_indexed: int
    total_records: int
    clusters_with_carrier: int
    encounter_cache_path: str
    extracted_root: str

    def matches(
        self,
        *,
        schema_version: int,
        sample_number: int,
        carrier_match_window_days: int,
    ) -> bool:
        return (
            self.schema_version == schema_version
            and self.sample_number == sample_number
            and self.carrier_match_window_days == carrier_match_window_days
        )

    def as_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sample_number": self.sample_number,
            "carrier_match_window_days": self.carrier_match_window_days,
            "beneficiary_rows_indexed": self.beneficiary_rows_indexed,
            "facility_claims_indexed": self.facility_claims_indexed,
            "carrier_claims_indexed": self.carrier_claims_indexed,
            "total_records": self.total_records,
            "clusters_with_carrier": self.clusters_with_carrier,
            "encounter_cache_path": self.encounter_cache_path,
            "extracted_root": self.extracted_root,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> EncounterCacheMetadata | None:
        try:
            return cls(
                schema_version=int(payload["schema_version"]),
                sample_number=int(payload["sample_number"]),
                carrier_match_window_days=int(payload["carrier_match_window_days"]),
                beneficiary_rows_indexed=int(payload.get("beneficiary_rows_indexed") or 0),
                facility_claims_indexed=int(payload.get("facility_claims_indexed") or 0),
                carrier_claims_indexed=int(payload.get("carrier_claims_indexed") or 0),
                total_records=int(payload.get("total_records") or 0),
                clusters_with_carrier=int(payload.get("clusters_with_carrier") or 0),
                encounter_cache_path=str(payload.get("encounter_cache_path") or ""),
                extracted_root=str(payload.get("extracted_root") or ""),
            )
        except (KeyError, TypeError, ValueError):
            return None
