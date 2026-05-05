from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, Generic, Sequence, TypeVar

import requests

from claim.cli_logging import CliLogger
from claim.gold_bill_trilliant import TrilliantSchema

TTask = TypeVar("TTask")
TResult = TypeVar("TResult")
TValue = TypeVar("TValue")


@dataclass(frozen=True, slots=True)
class ReferenceFileSpec:
    filename: str
    url: str


@dataclass(frozen=True, slots=True)
class OutputPaths:
    clusters: Path
    inpatient_bills: Path
    outpatient_bills: Path
    aftercare: Path
    failures: Path
    metadata: Path


@dataclass(frozen=True, slots=True)
class ReferenceDataPaths:
    ssa_fips_csv: Path


@dataclass(frozen=True, slots=True)
class WorkerConfiguration:
    cluster_workers: int
    pricing_workers: int
    aftercare_workers: int


@dataclass(frozen=True, slots=True)
class GenerationRunConfig:
    encounter_cache_path: Path
    sampled_encounters_path: Path
    output_dir: Path
    reference_data_dir: Path
    trilliant_cache_dir: Path
    output_paths: OutputPaths
    description_cache_path: Path
    trilliant_cache_path: Path
    sample_size: int
    seed: int
    test_one: bool
    overwrite: bool
    model: str
    reasoning_effort: str
    timeout_seconds: float
    request_delay_seconds: float
    extreme_case_probability: float
    insurance_coverage_ratio: float
    billed_charge_multiplier: float
    source_medicare_multiplier: float
    normal_payers: tuple[str, ...]
    trilliant_db_url: str
    trilliant_hospital_table: str | None
    trilliant_rate_table: str | None
    hud_crosswalk_path: Path | None
    hud_user_token: str | None
    max_hospital_candidates: int
    workers: WorkerConfiguration
    max_hospital_rate_probes: int
    max_quote_amount: float
    show_progress: bool
    max_nearby_fallback_probes: int = 6


@dataclass(frozen=True, slots=True)
class InputCodeReview:
    code_system: str
    code: str
    reason: str


@dataclass(frozen=True, slots=True)
class GeneratedDiagnosisCode:
    selected_icd10_code: str
    supporting_input_code: str | None
    supporting_input_code_system: str | None


@dataclass(frozen=True, slots=True)
class GeneratedProcedureCode:
    billing_code: str
    code_system: str
    supporting_input_code: str | None
    supporting_input_code_system: str | None
    service_date: str | None
    billing_side_hint: str | None


@dataclass(frozen=True, slots=True)
class NormalizedCodeCluster:
    primary_clinical_direction: str | None
    clinical_direction_rationale: str | None
    retained_input_codes: tuple[InputCodeReview, ...]
    filtered_input_codes: tuple[InputCodeReview, ...]
    generated_diagnoses: tuple[GeneratedDiagnosisCode, ...]
    generated_line_items: tuple[GeneratedProcedureCode, ...]

    def as_payload(self) -> dict[str, Any]:
        return {
            "primary_clinical_direction": self.primary_clinical_direction,
            "clinical_direction_rationale": self.clinical_direction_rationale,
            "retained_input_codes": [
                {
                    "code_system": review.code_system,
                    "code": review.code,
                    "reason": review.reason,
                }
                for review in self.retained_input_codes
            ],
            "filtered_input_codes": [
                {
                    "code_system": review.code_system,
                    "code": review.code,
                    "reason": review.reason,
                }
                for review in self.filtered_input_codes
            ],
            "generated_diagnoses": [
                {
                    "selected_icd10_code": diagnosis.selected_icd10_code,
                    "supporting_input_code": diagnosis.supporting_input_code,
                    "supporting_input_code_system": diagnosis.supporting_input_code_system,
                }
                for diagnosis in self.generated_diagnoses
            ],
            "generated_line_items": [
                {
                    "billing_code": procedure.billing_code,
                    "code_system": procedure.code_system,
                    "supporting_input_code": procedure.supporting_input_code,
                    "supporting_input_code_system": procedure.supporting_input_code_system,
                    "service_date": procedure.service_date,
                    "billing_side_hint": procedure.billing_side_hint,
                }
                for procedure in self.generated_line_items
            ],
        }


@dataclass(slots=True)
class DescriptionCacheStore:
    cache: dict[str, str]
    model: str
    lock: threading.Lock | None = None

    def _with_cache(self, operation: Callable[[dict[str, str]], TValue]) -> TValue:
        if self.lock is None:
            return operation(self.cache)
        with self.lock:
            return operation(self.cache)

    def snapshot(self) -> dict[str, str]:
        return self._with_cache(lambda cache: dict(cache))

    def missing_requests(self, bill_record: dict[str, Any]) -> list[dict[str, str]]:
        from claim.datagen.llm import missing_description_requests

        return self._with_cache(
            lambda cache: missing_description_requests(
                bill_record,
                cache,
                model=self.model,
            )
        )

    def update(self, descriptions: Sequence[dict[str, str]]) -> None:
        from claim.datagen.llm import description_cache_key

        def operation(cache: dict[str, str]) -> None:
            for item in descriptions:
                cache[
                    description_cache_key(self.model, item["code_system"], item["code"])
                ] = item["description"]

        self._with_cache(operation)

    def attach(self, bill_record: dict[str, Any]) -> None:
        from claim.datagen.llm import attach_descriptions_to_bill

        def operation(cache: dict[str, str]) -> None:
            attach_descriptions_to_bill(bill_record, cache, model=self.model)

        self._with_cache(operation)


@dataclass(frozen=True, slots=True)
class ClusterGenerationContext:
    model: str
    reasoning_effort: str
    request_delay_seconds: float
    ssa_fips_mapping: dict[tuple[str, str], dict[str, str | None]]
    local_hud_mapping: dict[str, list[str]] | None
    reference_data_dir: Path
    hud_user_token: str | None
    logger: CliLogger
    timeout_seconds: float | None = None
    client: Any | None = None
    session: requests.Session | None = None


@dataclass(frozen=True, slots=True)
class PricingBuildContext:
    seed: int
    model: str
    reasoning_effort: str
    request_delay_seconds: float
    normal_payers: list[str]
    extreme_case_probability: float
    insurance_coverage_ratio: float
    billed_charge_multiplier: float
    source_medicare_multiplier: float
    max_hospital_candidates: int
    max_hospital_rate_probes: int
    max_quote_amount: float
    description_cache: DescriptionCacheStore
    logger: CliLogger
    max_nearby_fallback_probes: int = 6
    timeout_seconds: float | None = None
    trilliant_cache_path: Path | None = None
    trilliant_db_url: str | None = None
    client: Any | None = None
    session: requests.Session | None = None
    trilliant_connection: Any | None = None
    trilliant_schema: TrilliantSchema | None = None
    local_hud_mapping: dict[str, list[str]] | None = None
    trilliant_cache_lock: threading.Lock | None = None
    pricing_gate: threading.Semaphore | None = None
    pricing_workers: int = 1
    throttle_metrics: dict[str, Any] | None = None
    throttle_metrics_lock: threading.Lock | None = None


@dataclass(slots=True)
class PipelineRunState:
    cluster_successful_cases: int
    cluster_failed_cases: int
    pricing_successful_cases: int
    pricing_failed_cases: int
    aftercare_successful_cases: int
    aftercare_failed_cases: int
    cluster_usage: dict[str, int]
    pricing_usage: dict[str, int]
    aftercare_usage: dict[str, int]
    scenario_counts: dict[str, int]
    built_since_cache_flush: int = 0


@dataclass(slots=True)
class StageTaskContext(Generic[TTask]):
    item: TTask
    case_id: str | None
    case_start: float | None = None
