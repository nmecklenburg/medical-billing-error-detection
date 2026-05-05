from __future__ import annotations

from calendar import monthrange
from contextlib import contextmanager
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import date
import threading
import time
from typing import Any, Iterable, Iterator

from claim.datagen import constants
from claim.cli_logging import CliLogger
from claim.gold_bill_support import normalize_state_code, normalize_text
from claim.normalization import normalize_code_system


def normalize_generated_code_system(value: str | None) -> str | None:
    return normalize_code_system(value)


def encounter_regime(record: dict[str, Any]) -> str | None:
    encounter = (record.get("encounter") or {}) if "encounter" in record else (
        ((record.get("source_encounter") or {}).get("encounter") or {})
    )
    return normalize_text(encounter.get("anchor_claim_type"))


def normalize_county_code(value: str | None) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return digits.zfill(3 if len(digits) <= 3 else 5)
    return text


def parse_iso_date(value: str | None) -> date | None:
    text = normalize_text(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = normalize_text(str(value))
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def format_date(value: date | None) -> str | None:
    return value.isoformat() if value is not None else None


def shift_date_by_years(value: date, *, years: int) -> date:
    target_year = value.year + years
    target_day = min(value.day, monthrange(target_year, value.month)[1])
    return date(target_year, value.month, target_day)


def shift_iso_date(value: str | None, *, years: int) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    parsed = parse_iso_date(text)
    if parsed is None:
        return text
    return format_date(shift_date_by_years(parsed, years=years))


def collect_encounter_date_years(
    value: Any,
    *,
    field_names: set[str] | frozenset[str],
    field_name: str | None = None,
) -> list[int]:
    if isinstance(value, dict):
        years: list[int] = []
        for child_key, child_value in value.items():
            years.extend(
                collect_encounter_date_years(
                    child_value,
                    field_names=field_names,
                    field_name=child_key,
                )
            )
        return years
    if isinstance(value, list):
        years: list[int] = []
        for item in value:
            years.extend(
                collect_encounter_date_years(
                    item,
                    field_names=field_names,
                    field_name=field_name,
                )
            )
        return years
    if isinstance(value, str) and field_name in field_names:
        parsed = parse_iso_date(value)
        if parsed is not None:
            return [parsed.year]
    return []


def encounter_requires_date_shift(
    encounter: dict[str, Any],
    *,
    legacy_max_year: int = constants.LEGACY_ENCOUNTER_MAX_YEAR,
) -> bool:
    if normalize_text(str(encounter.get(constants.DATE_SHIFT_METADATA_FIELD))) == str(
        constants.DEFAULT_DATE_SHIFT_YEARS
    ):
        return False
    years = collect_encounter_date_years(
        encounter,
        field_names=constants.SHIFT_TRIGGER_DATE_FIELDS,
    )
    return any(year <= legacy_max_year for year in years)


def shift_encounter_date_fields(
    value: Any,
    *,
    years: int,
    field_name: str | None = None,
) -> Any:
    if isinstance(value, dict):
        return {
            child_key: shift_encounter_date_fields(
                child_value,
                years=years,
                field_name=child_key,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [
            shift_encounter_date_fields(
                item,
                years=years,
                field_name=field_name,
            )
            for item in value
        ]
    if isinstance(value, str) and field_name in constants.SHIFTABLE_ENCOUNTER_DATE_FIELDS:
        return shift_iso_date(value, years=years)
    return value


def shift_encounter_dates(
    encounter: dict[str, Any],
    *,
    years: int = constants.DEFAULT_DATE_SHIFT_YEARS,
) -> dict[str, Any]:
    if normalize_text(str(encounter.get(constants.DATE_SHIFT_METADATA_FIELD))) == str(
        years
    ):
        return encounter
    if not encounter_requires_date_shift(encounter):
        return encounter
    shifted = shift_encounter_date_fields(encounter, years=years)
    if not isinstance(shifted, dict):
        raise TypeError("Shifted encounter payload must remain a mapping.")
    shifted[constants.DATE_SHIFT_METADATA_FIELD] = years
    return shifted


def cancel_pending_futures(futures: Iterable[Future[Any]]) -> None:
    for future in futures:
        future.cancel()


def shutdown_executor(
    executor: ThreadPoolExecutor | None,
    *,
    interrupted: bool,
) -> None:
    if executor is None:
        return
    executor.shutdown(wait=not interrupted, cancel_futures=interrupted)


def compute_age_on_date(birth_date: date | None, encounter_date: date | None) -> int | None:
    if birth_date is None or encounter_date is None:
        return None
    years = encounter_date.year - birth_date.year
    if (encounter_date.month, encounter_date.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def usage_accumulator() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def usage_totals(usage: dict[str, Any] | None) -> dict[str, int]:
    if not usage:
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def add_usage(target: dict[str, int], usage: dict[str, Any] | None) -> None:
    totals = usage_totals(usage)
    for key in target:
        target[key] += totals[key]


def apply_request_delay(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def patient_sex_label(value: str | None) -> str | None:
    code = normalize_text(value)
    if code == "1":
        return "male"
    if code == "2":
        return "female"
    return None


def throttle_metrics_accumulator() -> dict[str, Any]:
    return {
        "local_trilliant_wait_events": 0,
        "local_trilliant_wait_seconds": 0.0,
        "upstream_events": 0,
        "upstream_by_stage": {},
    }


def source_state_value_for_encounter(
    encounter: dict[str, Any],
    geography: dict[str, Any],
) -> str | None:
    return (
        normalize_state_code(geography.get("state_abbrev"))
        or normalize_text(geography.get("state_name"))
        or normalize_text((encounter.get("beneficiary") or {}).get("state_code"))
    )


def looks_like_throttle_error(error: Exception | str) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in ("429", "rate limit", "too many requests", "throttl")
    )


def record_upstream_throttle(
    throttle_metrics: dict[str, Any],
    throttle_metrics_lock: threading.Lock | None,
    *,
    logger: CliLogger,
    stage: str,
    case_id: str | None,
    error: Exception | str,
) -> None:
    if not looks_like_throttle_error(error):
        return
    if throttle_metrics_lock is None:
        throttle_metrics["upstream_events"] += 1
        by_stage = throttle_metrics.setdefault("upstream_by_stage", {})
        by_stage[stage] = int(by_stage.get(stage) or 0) + 1
    else:
        with throttle_metrics_lock:
            throttle_metrics["upstream_events"] += 1
            by_stage = throttle_metrics.setdefault("upstream_by_stage", {})
            by_stage[stage] = int(by_stage.get(stage) or 0) + 1
    logger.warning(
        "Observed upstream throttling",
        stage=stage,
        case_id=case_id,
        error=str(error),
    )


@contextmanager
def acquire_pricing_gate(
    pricing_gate: threading.Semaphore | None,
    *,
    logger: CliLogger,
    throttle_metrics: dict[str, Any] | None,
    throttle_metrics_lock: threading.Lock | None,
    case_id: str | None,
    pricing_workers: int,
    stage: str = "pricing",
) -> Iterator[None]:
    if pricing_gate is None:
        yield
        return

    wait_start = time.perf_counter()
    pricing_gate.acquire()
    waited_seconds = time.perf_counter() - wait_start
    if throttle_metrics is not None:
        if throttle_metrics_lock is None:
            throttle_metrics["local_trilliant_wait_seconds"] = float(
                throttle_metrics.get("local_trilliant_wait_seconds") or 0.0
            ) + waited_seconds
        else:
            with throttle_metrics_lock:
                throttle_metrics["local_trilliant_wait_seconds"] = float(
                    throttle_metrics.get("local_trilliant_wait_seconds") or 0.0
                ) + waited_seconds
    if waited_seconds >= constants.LOCAL_TRILLIANT_WAIT_LOG_THRESHOLD_SECONDS:
        if throttle_metrics is not None:
            if throttle_metrics_lock is None:
                throttle_metrics["local_trilliant_wait_events"] = int(
                    throttle_metrics.get("local_trilliant_wait_events") or 0
                ) + 1
            else:
                with throttle_metrics_lock:
                    throttle_metrics["local_trilliant_wait_events"] = int(
                        throttle_metrics.get("local_trilliant_wait_events") or 0
                    ) + 1
        logger.warning(
            "Pricing worker waited for Trilliant access",
            case_id=case_id,
            stage=stage,
            wait_seconds=round(waited_seconds, 3),
            pricing_workers=pricing_workers,
        )
    try:
        yield
    finally:
        pricing_gate.release()


def count_non_imputed_line_items(bill_record: dict[str, Any]) -> int:
    return sum(
        1
        for line_item in bill_line_items(bill_record)
        if (line_item.get("pricing_provenance") or {}).get("lookup_level")
        != "source_imputed"
    )


def bill_line_items(bill_record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = bill_record.get("line_items")
    if isinstance(explicit, list):
        return explicit
    seen_ids: set[str] = set()
    flattened: list[dict[str, Any]] = []
    facility_claim = bill_record.get("facility_claim") or {}
    for line_item in facility_claim.get("charge_lines") or []:
        line_id = normalize_text(str(line_item.get("line_id"))) if line_item.get("line_id") is not None else None
        if line_id is not None and line_id in seen_ids:
            continue
        if line_id is not None:
            seen_ids.add(line_id)
        flattened.append(line_item)
    for claim in bill_record.get("professional_claims") or []:
        for line_item in claim.get("lines") or []:
            line_id = normalize_text(str(line_item.get("line_id"))) if line_item.get("line_id") is not None else None
            if line_id is not None and line_id in seen_ids:
                continue
            if line_id is not None:
                seen_ids.add(line_id)
            flattened.append(line_item)
    return flattened


def bill_case_components(bill_record: dict[str, Any]) -> list[dict[str, Any]]:
    explicit = bill_record.get("case_components")
    if isinstance(explicit, list):
        return explicit
    facility_claim = bill_record.get("facility_claim") or {}
    return list(facility_claim.get("payment_components") or [])
