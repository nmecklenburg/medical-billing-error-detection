from __future__ import annotations

from collections import Counter
from concurrent.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import asdict, replace
from pathlib import Path
import threading
import time
from typing import Any, Callable, Sequence

from claim.datagen.cluster import build_generated_cluster_record_for_encounter
from claim.datagen import constants, io, utils
from claim.datagen.contracts import (
    ClusterGenerationContext,
    DescriptionCacheStore,
    GenerationRunConfig,
    PipelineRunState,
    PricingBuildContext,
    StageTaskContext,
    TTask,
    TResult,
)
from claim.datagen.llm import ensure_model_client, load_description_cache
from claim.datagen.pricing import (
    build_bill_from_generated_cluster_record,
    generate_aftercare_record_for_bill,
)
from claim.datagen.reference_data import (
    ensure_reference_data,
    load_hud_crosswalk,
    load_ssa_fips_crosswalk,
)
from claim.datagen.sampling import ensure_sampled_encounters
from claim.cli_logging import CliLogger, format_seconds
from claim.gold_bill_support import iso_timestamp, normalize_text
from claim.gold_bill_trilliant import connect_trilliant_cache, discover_trilliant_schema

STAGE_HEARTBEAT_INTERVAL_SECONDS = 10.0


def summarize_results(
    *,
    cluster_successful_cases: int,
    cluster_failed_cases: int,
    pricing_successful_cases: int,
    pricing_failed_cases: int,
    aftercare_successful_cases: int,
    aftercare_failed_cases: int,
    scenario_counts: dict[str, int],
    sampled_stats: dict[str, int],
    cluster_usage: dict[str, int],
    pricing_usage: dict[str, int],
    aftercare_usage: dict[str, int],
    throttle_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "sampled": sampled_stats,
        "cluster_stage": {
            "successful_cases": cluster_successful_cases,
            "failed_cases": cluster_failed_cases,
            "openai_usage": cluster_usage,
        },
        "pricing_stage": {
            "successful_cases": pricing_successful_cases,
            "failed_cases": pricing_failed_cases,
            "scenario_counts": scenario_counts,
            "openai_usage": pricing_usage,
        },
        "aftercare_stage": {
            "successful_cases": aftercare_successful_cases,
            "failed_cases": aftercare_failed_cases,
            "openai_usage": aftercare_usage,
        },
        "throttling": throttle_metrics,
        "completed_cases": aftercare_successful_cases,
        "openai_usage": {
            "input_tokens": (
                cluster_usage["input_tokens"]
                + pricing_usage["input_tokens"]
                + aftercare_usage["input_tokens"]
            ),
            "output_tokens": (
                cluster_usage["output_tokens"]
                + pricing_usage["output_tokens"]
                + aftercare_usage["output_tokens"]
            ),
            "total_tokens": (
                cluster_usage["total_tokens"]
                + pricing_usage["total_tokens"]
                + aftercare_usage["total_tokens"]
            ),
        },
    }


def snapshot_description_cache(
    description_cache: DescriptionCacheStore,
) -> dict[str, str]:
    return description_cache.snapshot()


def maybe_flush_description_cache(
    state: PipelineRunState,
    *,
    description_cache_path: Path,
    description_cache: DescriptionCacheStore,
    force: bool = False,
) -> None:
    if force or state.built_since_cache_flush >= constants.DESCRIPTION_CACHE_FLUSH_INTERVAL:
        io.write_json_atomic(
            description_cache_path,
            snapshot_description_cache(description_cache),
        )
        state.built_since_cache_flush = 0


def record_stage_failure(
    state: PipelineRunState,
    *,
    failures_writer: io.JsonlStreamWriter,
    throttle_metrics: dict[str, Any],
    throttle_metrics_lock: threading.Lock | None,
    logger: CliLogger,
    stage: str,
    case_id: str | None,
    error: Exception | str,
) -> None:
    message = str(error)
    utils.record_upstream_throttle(
        throttle_metrics,
        throttle_metrics_lock,
        logger=logger,
        stage=stage,
        case_id=case_id,
        error=error,
    )
    failures_writer.write(
        {
            "stage": stage,
            "case_id": case_id,
            "error": message,
            "generated_at": iso_timestamp(),
        }
    )
    if stage == "cluster":
        state.cluster_failed_cases += 1
    elif stage == "pricing":
        state.pricing_failed_cases += 1
    else:
        state.aftercare_failed_cases += 1
    logger.warning(
        "Failed to generate case",
        stage=stage,
        case_id=case_id,
        error=message,
    )


def finalize_cluster_success(
    state: PipelineRunState,
    *,
    clusters_writer: io.JsonlStreamWriter,
    logger: CliLogger,
    cluster_record: dict[str, Any],
    usage: dict[str, int],
) -> None:
    clusters_writer.write(cluster_record)
    utils.add_usage(state.cluster_usage, usage)
    state.cluster_successful_cases += 1
    code_cluster = cluster_record.get("code_cluster") or {}
    logger.info(
        "Generated code cluster",
        case_id=cluster_record.get("case_id"),
        diagnoses=len(code_cluster.get("generated_diagnoses") or []),
        line_items=len(code_cluster.get("generated_line_items") or []),
    )


def pricing_lookup_level_summary(
    bill_record: dict[str, Any],
) -> tuple[Counter[str], dict[str, int | str]]:
    priced_items = list(bill_record.get("line_items") or []) + list(
        bill_record.get("case_components") or []
    )
    facility_claim = bill_record.get("facility_claim") or {}
    priced_items.extend(list(facility_claim.get("charge_lines") or []))
    priced_items.extend(list(facility_claim.get("payment_components") or []))
    for professional_claim in list(bill_record.get("professional_claims") or []):
        priced_items.extend(list((professional_claim or {}).get("lines") or []))
    counts: Counter[str] = Counter()
    prior_scope_counts: Counter[str] = Counter()
    unique_prior_keys: set[tuple[str, str]] = set()
    prior_hospital_scans = 0
    prior_hospital_scan_caps = 0
    prior_cache_hits = 0
    for priced_item in priced_items:
        pricing = priced_item.get("pricing_provenance") or {}
        lookup_level = normalize_text(pricing.get("lookup_level")) or "unknown"
        counts[lookup_level] += 1
        if lookup_level != "trilliant_prior":
            continue
        scope = normalize_text(pricing.get("prior_lookup_scope")) or "unknown"
        prior_scope_counts[scope] += 1
        unique_key = (str(priced_item.get("billing_code") or ""), scope)
        if unique_key in unique_prior_keys:
            continue
        unique_prior_keys.add(unique_key)
        prior_hospital_scans += int(pricing.get("prior_hospital_scan_count") or 0)
        prior_hospital_scan_caps += int(pricing.get("prior_hospital_scan_cap") or 0)
        if normalize_text(pricing.get("prior_cache_status")) == "hit":
            prior_cache_hits += 1
    return counts, {
        "state_prior_lines": int(prior_scope_counts.get("state") or 0),
        "national_prior_lines": int(prior_scope_counts.get("national") or 0),
        "prior_hospital_scans": prior_hospital_scans,
        "prior_hospital_scan_caps": prior_hospital_scan_caps,
        "prior_cache_hits": prior_cache_hits,
    }


def pricing_fallback_log_fields(
    bill_record: dict[str, Any],
) -> dict[str, Any] | None:
    selection = ((bill_record.get("facility") or {}).get("selection_provenance") or {})
    pricing = ((bill_record.get("provenance") or {}).get("pricing") or {})
    lookup_counts, prior_summary = pricing_lookup_level_summary(bill_record)
    fallback_triggered = any(
        [
            normalize_text(selection.get("geographic_stage")) not in {None, "", "zip"},
            bool(selection.get("same_state_unfiltered_fallback_invoked")),
            int(selection.get("selection_probe_count") or 0) > 1,
            int(lookup_counts.get("hospital_local_estimate") or 0) > 0,
            int(lookup_counts.get("nearby_hospital_estimate") or 0) > 0,
            int(lookup_counts.get("nearby_hospital_quote") or 0) > 0,
            int(lookup_counts.get("cached_corpus_estimate") or 0) > 0,
            int(lookup_counts.get("cached_corpus_quote") or 0) > 0,
            int(lookup_counts.get("trilliant_prior") or 0) > 0,
            int(lookup_counts.get("source_observed") or 0) > 0,
            int(lookup_counts.get("source_imputed") or 0) > 0,
            int(lookup_counts.get("inpatient_anchor_imputed") or 0) > 0,
        ]
    )
    if not fallback_triggered:
        return None
    return {
        "case_id": bill_record.get("case_id"),
        "selection_stage": selection.get("geographic_stage"),
        "candidate_count": selection.get("candidate_count"),
        "stage_candidate_count": selection.get("selection_stage_candidate_count"),
        "hospital_probes": selection.get("selection_probe_count"),
        "probe_limit": selection.get("selection_probe_limit"),
        "probe_rank": selection.get("selection_probe_rank"),
        "quote_coverage": selection.get("quoted_code_coverage"),
        "same_state_unfiltered": selection.get("same_state_unfiltered_fallback_invoked"),
        "hospital_local_estimate_lines": int(lookup_counts.get("hospital_local_estimate") or 0),
        "nearby_hospital_estimate_lines": int(lookup_counts.get("nearby_hospital_estimate") or 0),
        "nearby_hospital_quote_lines": int(lookup_counts.get("nearby_hospital_quote") or 0),
        "cached_corpus_estimate_lines": int(lookup_counts.get("cached_corpus_estimate") or 0),
        "cached_corpus_quote_lines": int(lookup_counts.get("cached_corpus_quote") or 0),
        "source_observed_lines": int(lookup_counts.get("source_observed") or 0),
        "trilliant_prior_lines": int(lookup_counts.get("trilliant_prior") or 0),
        "state_prior_lines": prior_summary["state_prior_lines"],
        "national_prior_lines": prior_summary["national_prior_lines"],
        "prior_hospital_scans": prior_summary["prior_hospital_scans"],
        "prior_hospital_scan_caps": prior_summary["prior_hospital_scan_caps"],
        "prior_cache_hits": prior_summary["prior_cache_hits"],
        "source_imputed_lines": int(lookup_counts.get("source_imputed") or 0),
        "anchor_imputed_components": int(lookup_counts.get("inpatient_anchor_imputed") or 0),
        "rate_cache_hits": pricing.get("rate_cache_hits"),
        "rate_cache_misses": pricing.get("rate_cache_misses"),
    }


def finalize_pricing_success(
    state: PipelineRunState,
    *,
    inpatient_bills_writer: io.JsonlStreamWriter,
    outpatient_bills_writer: io.JsonlStreamWriter,
    description_cache_path: Path,
    description_cache: DescriptionCacheStore,
    logger: CliLogger,
    bill_record: dict[str, Any],
    usage: dict[str, int],
    case_start: float | None,
) -> None:
    regime = normalize_text(bill_record.get("encounter_regime")) or "outpatient"
    writer = inpatient_bills_writer if regime == "inpatient" else outpatient_bills_writer
    writer.write(bill_record)
    utils.add_usage(state.pricing_usage, usage)
    state.pricing_successful_cases += 1
    state.built_since_cache_flush += 1
    maybe_flush_description_cache(
        state,
        description_cache_path=description_cache_path,
        description_cache=description_cache,
    )
    scenario = str((bill_record.get("payer_scenario") or {}).get("scenario") or "unknown")
    state.scenario_counts[scenario] = state.scenario_counts.get(scenario, 0) + 1
    selection_provenance = (
        (bill_record.get("facility") or {}).get("selection_provenance") or {}
    )
    started_at = case_start if case_start is not None else time.perf_counter()
    log_fields = {
        "case_id": bill_record.get("case_id"),
        "line_items": len(bill_record.get("line_items") or []),
        "quoted_line_items": utils.count_non_imputed_line_items(bill_record),
        "hospital_probes": selection_provenance.get("selection_probe_count"),
        "quote_coverage": selection_provenance.get("quoted_code_coverage"),
        "seconds": format_seconds(time.perf_counter() - started_at),
    }
    # Emitting one log line per completed bill through tqdm.write() causes the
    # active pricing bar to be repeatedly cleared and redrawn, which can show up
    # as duplicated finished bars in terminal transcripts. Keep per-case logs
    # when no progress UI is active; otherwise emit a coarse periodic summary.
    fallback_fields = pricing_fallback_log_fields(bill_record)
    if logger.line_writer is None:
        logger.info("Generated bill", **log_fields)
    elif state.pricing_successful_cases % 25 == 0:
        logger.info(
            "Pricing progress",
            completed_bills=state.pricing_successful_cases,
            latest_case_id=log_fields["case_id"],
            latest_seconds=log_fields["seconds"],
        )
    if fallback_fields is not None:
        logger.info("Pricing fallback detail", **fallback_fields)


def finalize_aftercare_success(
    state: PipelineRunState,
    *,
    aftercare_writer: io.JsonlStreamWriter,
    aftercare_record: dict[str, Any],
    usage: dict[str, int],
) -> None:
    aftercare_writer.write(aftercare_record)
    utils.add_usage(state.aftercare_usage, usage)
    state.aftercare_successful_cases += 1


def drain_stage_futures(
    futures: dict[Future[TResult], StageTaskContext[TTask]],
    *,
    wait_for_all: bool,
    stage: str,
    progress: Any,
    on_success: Callable[[TResult, StageTaskContext[TTask]], None],
    on_failure: Callable[[str, str | None, Exception | str], None],
    logger: CliLogger,
    submitted_count: int,
    total_count: int,
    completed_count: Callable[[], int],
) -> None:
    if not futures:
        return
    while futures:
        done, _ = wait(
            futures.keys(),
            timeout=STAGE_HEARTBEAT_INTERVAL_SECONDS,
            return_when=ALL_COMPLETED if wait_for_all else FIRST_COMPLETED,
        )
        if not done:
            oldest_started_at = min(
                (
                    context.case_start
                    for context in futures.values()
                    if context.case_start is not None
                ),
                default=None,
            )
            logger.info(
                "Stage heartbeat",
                stage=stage,
                completed=completed_count(),
                submitted=submitted_count,
                total=total_count,
                in_flight=len(futures),
                oldest_in_flight=(
                    format_seconds(time.perf_counter() - oldest_started_at)
                    if oldest_started_at is not None
                    else None
                ),
            )
            continue
        for future in done:
            context = futures.pop(future)
            try:
                payload = future.result()
            except Exception as exc:
                on_failure(stage, context.case_id, exc)
            else:
                on_success(payload, context)
            if progress is not None:
                progress.update(1)
        if not wait_for_all:
            break


def run_timed_stage_item(
    process_item: Callable[[TTask], TResult],
    item: TTask,
    context: StageTaskContext[TTask],
) -> TResult:
    context.case_start = time.perf_counter()
    return process_item(item)


def run_stage_tasks(
    *,
    stage: str,
    items: Sequence[TTask],
    worker_count: int,
    progress_description: str,
    show_progress: bool,
    logger: CliLogger,
    case_id_for_item: Callable[[TTask], str | None],
    process_item: Callable[[TTask], TResult],
    on_success: Callable[[TResult, StageTaskContext[TTask]], None],
    on_failure: Callable[[str, str | None, Exception | str], None],
    interrupt_details: Callable[[], dict[str, Any]],
) -> bool:
    progress = io.create_progress_bar(
        description=progress_description,
        total=len(items),
        unit="case",
        show_progress=show_progress,
    )
    executor: ThreadPoolExecutor | None = None
    in_flight: dict[Future[TResult], StageTaskContext[TTask]] = {}
    interrupted = False
    submitted_count = 0
    completed_stage_items = 0

    def handle_success(payload: TResult, context: StageTaskContext[TTask]) -> None:
        nonlocal completed_stage_items
        completed_stage_items += 1
        on_success(payload, context)

    def handle_failure(stage_name: str, case_id: str | None, error: Exception | str) -> None:
        nonlocal completed_stage_items
        completed_stage_items += 1
        on_failure(stage_name, case_id, error)

    try:
        if worker_count > 1:
            executor = ThreadPoolExecutor(max_workers=worker_count)
        for item in items:
            context = StageTaskContext(
                item=item,
                case_id=case_id_for_item(item),
            )
            if executor is None:
                try:
                    payload = run_timed_stage_item(process_item, item, context)
                except Exception as exc:
                    handle_failure(stage, context.case_id, exc)
                else:
                    handle_success(payload, context)
                if progress is not None:
                    progress.update(1)
                submitted_count += 1
                continue

            future = executor.submit(run_timed_stage_item, process_item, item, context)
            in_flight[future] = context
            submitted_count += 1
            if len(in_flight) >= max(1, worker_count * 2):
                drain_stage_futures(
                    in_flight,
                    wait_for_all=False,
                    stage=stage,
                    progress=progress,
                    on_success=handle_success,
                    on_failure=handle_failure,
                    logger=logger,
                    submitted_count=submitted_count,
                    total_count=len(items),
                    completed_count=lambda: completed_stage_items,
                )

        if executor is not None and in_flight:
            logger.info(
                "Stage work queued",
                stage=stage,
                submitted=submitted_count,
                total=len(items),
                in_flight=len(in_flight),
                workers=worker_count,
            )

        drain_stage_futures(
            in_flight,
            wait_for_all=True,
            stage=stage,
            progress=progress,
            on_success=handle_success,
            on_failure=handle_failure,
            logger=logger,
            submitted_count=submitted_count,
            total_count=len(items),
            completed_count=lambda: completed_stage_items,
        )
    except KeyboardInterrupt:
        interrupted = True
        logger.warning(
            f"Interrupt received during {stage} stage; canceling pending work and preserving partial outputs",
            **interrupt_details(),
        )
        utils.cancel_pending_futures(in_flight.keys())
    finally:
        utils.shutdown_executor(executor, interrupted=interrupted)
        io.close_progress_bar(progress)
    return interrupted


def run_cluster_stage(
    *,
    encounters: Sequence[dict[str, Any]],
    cluster_context: ClusterGenerationContext,
    cluster_workers: int,
    sampled_path: Path,
    show_progress: bool,
    logger: CliLogger,
    state: PipelineRunState,
    clusters_writer: io.JsonlStreamWriter,
    failures_writer: io.JsonlStreamWriter,
    throttle_metrics: dict[str, Any],
    throttle_metrics_lock: threading.Lock | None,
) -> bool:
    logger.info(
        "Starting cluster generation stage",
        source_path=sampled_path,
        cases=len(encounters),
        cluster_workers=cluster_workers,
        llm_calls_per_case=2,
        llm_calls_upper_bound=len(encounters) * 2,
    )
    return run_stage_tasks(
        stage="cluster",
        items=list(encounters),
        worker_count=cluster_workers,
        progress_description="generate clusters",
        show_progress=show_progress,
        logger=logger,
        case_id_for_item=lambda encounter: normalize_text(encounter.get("cluster_id")),
        process_item=lambda encounter: build_generated_cluster_record_async(
            encounter=encounter,
            context=cluster_context,
        ),
        on_success=lambda payload, context: finalize_cluster_success(
            state,
            clusters_writer=clusters_writer,
            logger=logger,
            cluster_record=payload[0],
            usage=payload[1],
        ),
        on_failure=lambda stage_name, case_id, error: record_stage_failure(
            state,
            failures_writer=failures_writer,
            throttle_metrics=throttle_metrics,
            throttle_metrics_lock=throttle_metrics_lock,
            logger=logger,
            stage=stage_name,
            case_id=case_id,
            error=error,
        ),
        interrupt_details=lambda: {
            "case_failures": state.cluster_failed_cases,
            "completed_clusters": state.cluster_successful_cases,
        },
    )


def run_pricing_stage(
    *,
    pricing_inputs: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    pricing_context: PricingBuildContext,
    pricing_workers: int,
    clusters_path: Path,
    show_progress: bool,
    logger: CliLogger,
    state: PipelineRunState,
    inpatient_bills_writer: io.JsonlStreamWriter,
    outpatient_bills_writer: io.JsonlStreamWriter,
    failures_writer: io.JsonlStreamWriter,
    description_cache_path: Path,
    throttle_metrics: dict[str, Any],
    throttle_metrics_lock: threading.Lock | None,
) -> bool:
    logger.info(
        "Starting pricing stage",
        source_path=clusters_path,
        cases=len(pricing_inputs),
        pricing_workers=pricing_workers,
        trilliant_source_mode=pricing_context.trilliant_schema.source_mode,
        max_hospital_rate_probes=pricing_context.max_hospital_rate_probes,
        max_nearby_fallback_probes=pricing_context.max_nearby_fallback_probes,
    )
    interrupted = run_stage_tasks(
        stage="pricing",
        items=list(pricing_inputs),
        worker_count=pricing_workers,
        progress_description="price bills",
        show_progress=show_progress,
        logger=logger,
        case_id_for_item=lambda item: normalize_text(item[0].get("cluster_id")),
        process_item=lambda item: build_priced_bill_record_async(
            encounter=item[0],
            generated_cluster_record=item[1],
            context=pricing_context,
        ),
        on_success=lambda payload, context: finalize_pricing_success(
            state,
            inpatient_bills_writer=inpatient_bills_writer,
            outpatient_bills_writer=outpatient_bills_writer,
            description_cache_path=description_cache_path,
            description_cache=pricing_context.description_cache,
            logger=logger,
            bill_record=payload[0],
            usage=payload[1],
            case_start=context.case_start,
        ),
        on_failure=lambda stage_name, case_id, error: record_stage_failure(
            state,
            failures_writer=failures_writer,
            throttle_metrics=throttle_metrics,
            throttle_metrics_lock=throttle_metrics_lock,
            logger=logger,
            stage=stage_name,
            case_id=case_id,
            error=error,
        ),
        interrupt_details=lambda: {
            "case_failures": state.pricing_failed_cases,
            "completed_bills": state.pricing_successful_cases,
        },
    )
    maybe_flush_description_cache(
        state,
        description_cache_path=description_cache_path,
        description_cache=pricing_context.description_cache,
        force=True,
    )
    return interrupted


def run_aftercare_stage(
    *,
    bill_records: Sequence[dict[str, Any]],
    timeout_seconds: float,
    model: str,
    reasoning_effort: str,
    request_delay_seconds: float,
    aftercare_workers: int,
    bills_path: Path,
    show_progress: bool,
    logger: CliLogger,
    state: PipelineRunState,
    aftercare_writer: io.JsonlStreamWriter,
    failures_writer: io.JsonlStreamWriter,
    throttle_metrics: dict[str, Any],
    throttle_metrics_lock: threading.Lock | None,
) -> bool:
    logger.info(
        "Starting after-care generation stage",
        source_path=bills_path,
        cases=len(bill_records),
        aftercare_workers=aftercare_workers,
    )
    return run_stage_tasks(
        stage="aftercare",
        items=list(bill_records),
        worker_count=aftercare_workers,
        progress_description="generate aftercare",
        show_progress=show_progress,
        logger=logger,
        case_id_for_item=lambda bill_record: normalize_text(bill_record.get("case_id")),
        process_item=lambda bill_record: generate_aftercare_record_async(
            timeout_seconds=timeout_seconds,
            model=model,
            reasoning_effort=reasoning_effort,
            request_delay_seconds=request_delay_seconds,
            bill_record=bill_record,
        ),
        on_success=lambda payload, context: finalize_aftercare_success(
            state,
            aftercare_writer=aftercare_writer,
            aftercare_record=payload[0],
            usage=payload[1],
        ),
        on_failure=lambda stage_name, case_id, error: record_stage_failure(
            state,
            failures_writer=failures_writer,
            throttle_metrics=throttle_metrics,
            throttle_metrics_lock=throttle_metrics_lock,
            logger=logger,
            stage=stage_name,
            case_id=case_id,
            error=error,
        ),
        interrupt_details=lambda: {
            "completed_aftercare": state.aftercare_successful_cases,
            "completed_bills": state.pricing_successful_cases,
        },
    )


def build_generated_cluster_record_async(
    *,
    encounter: dict[str, Any],
    context: ClusterGenerationContext,
) -> tuple[dict[str, Any], dict[str, int]]:
    if context.timeout_seconds is None:
        raise RuntimeError("Cluster generation context is missing timeout_seconds.")
    client = ensure_model_client(context.model, context.timeout_seconds)
    with io.build_session() as session:
        runtime_context = replace(
            context,
            client=client,
            session=session,
        )
        return build_generated_cluster_record_for_encounter(
            encounter,
            context=runtime_context,
        )


def build_priced_bill_record_async(
    *,
    encounter: dict[str, Any],
    generated_cluster_record: dict[str, Any],
    context: PricingBuildContext,
) -> tuple[dict[str, Any], dict[str, int]]:
    if (
        context.timeout_seconds is None
        or context.trilliant_cache_path is None
        or context.trilliant_db_url is None
        or context.trilliant_connection is None
        or context.trilliant_schema is None
    ):
        raise RuntimeError("Pricing build context is missing worker configuration.")
    client = ensure_model_client(context.model, context.timeout_seconds)
    worker_connection = context.trilliant_connection
    worker_cache_lock = context.trilliant_cache_lock
    owned_connection = None
    if context.trilliant_schema.source_mode != "public_directory":
        owned_connection = connect_trilliant_cache(
            context.trilliant_cache_path,
            context.trilliant_db_url,
        )
        worker_connection = owned_connection
        worker_cache_lock = None
    try:
        with io.build_session() as session:
            runtime_context = replace(
                context,
                client=client,
                session=session,
                trilliant_connection=worker_connection,
                trilliant_cache_lock=worker_cache_lock,
            )
            return build_bill_from_generated_cluster_record(
                encounter,
                generated_cluster_record,
                context=runtime_context,
            )
    finally:
        if owned_connection is not None:
            owned_connection.close()


def generate_aftercare_record_async(
    *,
    timeout_seconds: float,
    model: str,
    reasoning_effort: str,
    request_delay_seconds: float,
    bill_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    client = ensure_model_client(model, timeout_seconds)
    return generate_aftercare_record_for_bill(
        client,
        model=model,
        reasoning_effort=reasoning_effort,
        request_delay_seconds=request_delay_seconds,
        bill_record=bill_record,
    )


def run_generation_pipeline(config: GenerationRunConfig) -> int:
    logger = CliLogger(
        "gold-bills",
        line_writer=io.tqdm_line_writer if config.show_progress else None,
    )

    removed = io.cleanup_stale_partial_files(
        [
            config.output_dir,
            config.sampled_encounters_path.parent,
            config.reference_data_dir,
            config.trilliant_cache_dir,
        ]
    )
    if removed:
        logger.info("Removed stale partial files", count=len(removed))

    start_time = time.perf_counter()
    logger.info(
        "Starting gold-bill generation run",
        encounter_cache=config.encounter_cache_path,
        sample_size=config.sample_size,
        test_one=config.test_one,
        output_dir=config.output_dir,
        model=config.model,
        cluster_workers=config.workers.cluster_workers,
        pricing_workers=config.workers.pricing_workers,
        aftercare_workers=config.workers.aftercare_workers,
        max_hospital_rate_probes=config.max_hospital_rate_probes,
        max_quote_amount=config.max_quote_amount,
    )

    if not config.encounter_cache_path.exists():
        raise FileNotFoundError(f"Encounter cache not found: {config.encounter_cache_path}")

    reference_paths = ensure_reference_data(
        config.reference_data_dir,
        logger,
        show_progress=config.show_progress,
    )
    ssa_fips_mapping = load_ssa_fips_crosswalk(reference_paths.ssa_fips_csv)
    local_hud_mapping = (
        load_hud_crosswalk(config.hud_crosswalk_path)
        if config.hud_crosswalk_path and config.hud_crosswalk_path.exists()
        else None
    )

    sampled_records, sampled_stats = ensure_sampled_encounters(
        config.sampled_encounters_path,
        config.encounter_cache_path,
        sample_size=config.sample_size,
        seed=config.seed,
        overwrite=config.overwrite,
        logger=logger,
        show_progress=config.show_progress,
    )

    description_cache = DescriptionCacheStore(
        cache=load_description_cache(config.description_cache_path),
        model=config.model,
        lock=threading.Lock(),
    )
    sampled_case_ids = {
        str(record.get("cluster_id"))
        for record in sampled_records
        if record.get("cluster_id") is not None
    }
    if config.overwrite:
        existing_cluster_ids: set[str] = set()
        existing_bill_ids: set[str] = set()
        existing_aftercare_ids: set[str] = set()
        existing_scenario_counts: dict[str, int] = {}
        resumed_cluster_usage = utils.usage_accumulator()
        resumed_pricing_usage = utils.usage_accumulator()
        resumed_aftercare_usage = utils.usage_accumulator()
    else:
        existing_cluster_ids = io.load_existing_case_ids(
            config.output_paths.clusters,
            logger=logger,
            label="existing generated clusters",
        )
        existing_bill_ids, existing_scenario_counts = io.load_existing_split_bills_state(
            config.output_paths.inpatient_bills,
            config.output_paths.outpatient_bills,
            logger=logger,
        )
        existing_aftercare_ids = io.load_existing_case_ids(
            config.output_paths.aftercare,
            logger=logger,
            label="existing aftercare",
        )
        if config.output_paths.failures.exists():
            io.repair_jsonl_trailing_partial_line(
                config.output_paths.failures,
                logger=logger,
                label="existing failures",
            )
        io.validate_resumable_case_ids(
            sampled_case_ids,
            existing_cluster_ids,
            path=config.output_paths.clusters,
            label="generated cluster",
        )
        io.validate_resumable_case_ids(
            sampled_case_ids,
            existing_bill_ids,
            path=config.output_paths.outpatient_bills,
            label="bill",
        )
        io.validate_resumable_case_ids(
            sampled_case_ids,
            existing_aftercare_ids,
            path=config.output_paths.aftercare,
            label="aftercare",
        )
        if not existing_aftercare_ids.issubset(existing_bill_ids):
            raise RuntimeError(
                f"Cannot resume from {config.output_paths.aftercare}: found aftercare case IDs without matching bills. "
                "Use --overwrite or choose a different --output-dir."
            )
        resumed_cluster_usage, resumed_pricing_usage, resumed_aftercare_usage = (
            io.load_resumable_usage(
                config.output_paths.metadata,
                sampled_path=config.sampled_encounters_path,
                encounter_cache_path=config.encounter_cache_path,
            )
        )
    effective_existing_cluster_ids = existing_cluster_ids | existing_bill_ids
    pending_cluster_records = [
        record
        for record in sampled_records
        if str(record.get("cluster_id")) not in effective_existing_cluster_ids
    ]
    resume_state = {
        "reused_clusters": len(effective_existing_cluster_ids),
        "reused_bills": len(existing_bill_ids),
        "reused_aftercare": len(existing_aftercare_ids),
        "pending_clusters_at_start": len(pending_cluster_records),
        "pending_pricing_at_start": 0,
        "pending_aftercare_at_start": 0,
    }
    trilliant_connection: Any | None = None
    trilliant_schema: Any | None = None
    trilliant_cache_lock: threading.Lock | None = None
    pricing_gate = threading.BoundedSemaphore(max(1, config.workers.pricing_workers))
    throttle_metrics = utils.throttle_metrics_accumulator()
    throttle_metrics_lock = threading.Lock()
    state = PipelineRunState(
        cluster_successful_cases=len(effective_existing_cluster_ids),
        cluster_failed_cases=0,
        pricing_successful_cases=len(existing_bill_ids),
        pricing_failed_cases=0,
        aftercare_successful_cases=len(existing_aftercare_ids),
        aftercare_failed_cases=0,
        cluster_usage=resumed_cluster_usage,
        pricing_usage=resumed_pricing_usage,
        aftercare_usage=resumed_aftercare_usage,
        scenario_counts=dict(existing_scenario_counts),
    )
    cluster_context = ClusterGenerationContext(
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        request_delay_seconds=config.request_delay_seconds,
        ssa_fips_mapping=ssa_fips_mapping,
        local_hud_mapping=local_hud_mapping,
        reference_data_dir=config.reference_data_dir,
        hud_user_token=config.hud_user_token,
        logger=logger,
        timeout_seconds=config.timeout_seconds,
    )
    pricing_context = PricingBuildContext(
        seed=config.seed,
        model=config.model,
        reasoning_effort=config.reasoning_effort,
        request_delay_seconds=config.request_delay_seconds,
        normal_payers=list(config.normal_payers),
        extreme_case_probability=config.extreme_case_probability,
        insurance_coverage_ratio=config.insurance_coverage_ratio,
        billed_charge_multiplier=config.billed_charge_multiplier,
        source_medicare_multiplier=config.source_medicare_multiplier,
        max_hospital_candidates=config.max_hospital_candidates,
        max_hospital_rate_probes=config.max_hospital_rate_probes,
        max_quote_amount=config.max_quote_amount,
        max_nearby_fallback_probes=config.max_nearby_fallback_probes,
        description_cache=description_cache,
        logger=logger,
        timeout_seconds=config.timeout_seconds,
        trilliant_cache_path=config.trilliant_cache_path,
        trilliant_db_url=config.trilliant_db_url,
        local_hud_mapping=local_hud_mapping,
        pricing_gate=pricing_gate,
        pricing_workers=config.workers.pricing_workers,
        throttle_metrics=throttle_metrics,
        throttle_metrics_lock=throttle_metrics_lock,
    )
    interrupted = False
    interrupted_stage: str | None = None
    logger.info(
        "Resume state",
        overwrite=config.overwrite,
        existing_clusters=len(effective_existing_cluster_ids),
        pending_clusters=len(pending_cluster_records),
        existing_bills=len(existing_bill_ids),
        existing_aftercare=len(existing_aftercare_ids),
    )
    try:
        append_outputs = not config.overwrite
        with (
            io.JsonlStreamWriter(config.output_paths.clusters, append=append_outputs) as clusters_writer,
            io.JsonlStreamWriter(
                config.output_paths.inpatient_bills,
                append=append_outputs,
            ) as inpatient_bills_writer,
            io.JsonlStreamWriter(
                config.output_paths.outpatient_bills,
                append=append_outputs,
            ) as outpatient_bills_writer,
            io.JsonlStreamWriter(config.output_paths.aftercare, append=append_outputs) as aftercare_writer,
            io.JsonlStreamWriter(config.output_paths.failures, append=append_outputs) as failures_writer,
        ):
            interrupted = run_cluster_stage(
                encounters=pending_cluster_records,
                cluster_context=cluster_context,
                cluster_workers=config.workers.cluster_workers,
                sampled_path=config.sampled_encounters_path,
                show_progress=config.show_progress,
                logger=logger,
                state=state,
                clusters_writer=clusters_writer,
                failures_writer=failures_writer,
                throttle_metrics=throttle_metrics,
                throttle_metrics_lock=throttle_metrics_lock,
            )
            if interrupted:
                interrupted_stage = "cluster"

            if not interrupted:
                generated_cluster_records = io.load_generated_cluster_records(
                    config.output_paths.clusters,
                    logger=logger,
                )
                pending_pricing_inputs = [
                    (encounter, generated_cluster_records[str(encounter.get("cluster_id"))])
                    for encounter in sampled_records
                    if str(encounter.get("cluster_id")) not in existing_bill_ids
                    and str(encounter.get("cluster_id")) in generated_cluster_records
                ]
                resume_state["pending_pricing_at_start"] = len(pending_pricing_inputs)
                if pending_pricing_inputs:
                    trilliant_connection = connect_trilliant_cache(
                        config.trilliant_cache_path,
                        config.trilliant_db_url,
                    )
                    trilliant_schema = discover_trilliant_schema(
                        trilliant_connection,
                        hospital_table_override=config.trilliant_hospital_table,
                        rate_table_override=config.trilliant_rate_table,
                    )
                    trilliant_cache_lock = (
                        threading.Lock()
                        if trilliant_schema.source_mode == "public_directory"
                        else None
                    )
                    pricing_context = replace(
                        pricing_context,
                        trilliant_connection=trilliant_connection,
                        trilliant_schema=trilliant_schema,
                        trilliant_cache_lock=trilliant_cache_lock,
                    )
                    logger.info(
                        "Configured Trilliant pricing source",
                        source_mode=trilliant_schema.source_mode,
                        db_url=config.trilliant_db_url,
                    )
                    interrupted = run_pricing_stage(
                        pricing_inputs=pending_pricing_inputs,
                        pricing_context=pricing_context,
                        pricing_workers=config.workers.pricing_workers,
                        clusters_path=config.output_paths.clusters,
                        show_progress=config.show_progress,
                        logger=logger,
                        state=state,
                        inpatient_bills_writer=inpatient_bills_writer,
                        outpatient_bills_writer=outpatient_bills_writer,
                        failures_writer=failures_writer,
                        description_cache_path=config.description_cache_path,
                        throttle_metrics=throttle_metrics,
                        throttle_metrics_lock=throttle_metrics_lock,
                    )
                    if interrupted:
                        interrupted_stage = "pricing"

            if not interrupted:
                pending_aftercare_bill_records = [
                    bill_record
                    for bill_record in io.iter_bill_records(
                        config.output_paths.inpatient_bills,
                        config.output_paths.outpatient_bills,
                    )
                    if normalize_text(bill_record.get("case_id")) not in existing_aftercare_ids
                ]
                resume_state["pending_aftercare_at_start"] = len(pending_aftercare_bill_records)
                interrupted = run_aftercare_stage(
                    bill_records=pending_aftercare_bill_records,
                    timeout_seconds=config.timeout_seconds,
                    model=config.model,
                    reasoning_effort=config.reasoning_effort,
                    request_delay_seconds=config.request_delay_seconds,
                    aftercare_workers=config.workers.aftercare_workers,
                    bills_path=config.output_paths.outpatient_bills,
                    show_progress=config.show_progress,
                    logger=logger,
                    state=state,
                    aftercare_writer=aftercare_writer,
                    failures_writer=failures_writer,
                    throttle_metrics=throttle_metrics,
                    throttle_metrics_lock=throttle_metrics_lock,
                )
                if interrupted:
                    interrupted_stage = "aftercare"

        io.write_json_atomic(
            config.description_cache_path,
            snapshot_description_cache(description_cache),
        )
        run_metadata = {
            "generated_at": iso_timestamp(),
            "schema_version": constants.GOLD_BILL_SCHEMA_VERSION,
            "model": config.model,
            "reasoning_effort": config.reasoning_effort,
            "date_shift_years": constants.DEFAULT_DATE_SHIFT_YEARS,
            "status": "interrupted" if interrupted else "completed",
            "interrupted_stage": interrupted_stage,
            "worker_configuration": {
                "cluster_workers": config.workers.cluster_workers,
                "pricing_workers": config.workers.pricing_workers,
                "aftercare_workers": config.workers.aftercare_workers,
                "max_hospital_rate_probes": config.max_hospital_rate_probes,
                "max_nearby_fallback_probes": config.max_nearby_fallback_probes,
                "max_quote_amount": config.max_quote_amount,
            },
            "resume": resume_state,
            "encounter_cache_path": str(config.encounter_cache_path),
            "sampled_encounters_path": str(config.sampled_encounters_path),
            "reference_paths": {
                "ssa_fips_csv": str(reference_paths.ssa_fips_csv),
                "hud_crosswalk_path": str(config.hud_crosswalk_path)
                if config.hud_crosswalk_path
                else None,
            },
            "outputs": {
                "clusters": str(config.output_paths.clusters),
                "inpatient_bills": str(config.output_paths.inpatient_bills),
                "outpatient_bills": str(config.output_paths.outpatient_bills),
                "aftercare": str(config.output_paths.aftercare),
                "failures": str(config.output_paths.failures),
                "metadata": str(config.output_paths.metadata),
            },
            "trilliant_cache_path": str(config.trilliant_cache_path),
            "trilliant_schema": asdict(trilliant_schema) if trilliant_schema is not None else None,
            "summary": summarize_results(
                cluster_successful_cases=state.cluster_successful_cases,
                cluster_failed_cases=state.cluster_failed_cases,
                pricing_successful_cases=state.pricing_successful_cases,
                pricing_failed_cases=state.pricing_failed_cases,
                aftercare_successful_cases=state.aftercare_successful_cases,
                aftercare_failed_cases=state.aftercare_failed_cases,
                scenario_counts=state.scenario_counts,
                sampled_stats=sampled_stats,
                cluster_usage=state.cluster_usage,
                pricing_usage=state.pricing_usage,
                aftercare_usage=state.aftercare_usage,
                throttle_metrics={
                    "local_trilliant_wait_events": int(
                        throttle_metrics.get("local_trilliant_wait_events") or 0
                    ),
                    "local_trilliant_wait_seconds": round(
                        float(throttle_metrics.get("local_trilliant_wait_seconds") or 0.0),
                        6,
                    ),
                    "upstream_events": int(throttle_metrics.get("upstream_events") or 0),
                    "upstream_by_stage": dict(throttle_metrics.get("upstream_by_stage") or {}),
                },
            ),
        }
        io.write_json_atomic(config.output_paths.metadata, run_metadata)
    finally:
        if trilliant_connection is not None:
            trilliant_connection.close()

    if interrupted:
        logger.warning(
            "Interrupted gold-bill generation run",
            stage=interrupted_stage,
            clusters=state.cluster_successful_cases,
            bills=state.pricing_successful_cases,
            aftercare=state.aftercare_successful_cases,
            failures=state.cluster_failed_cases
            + state.pricing_failed_cases
            + state.aftercare_failed_cases,
        )
        return 130
    logger.success(
        "Completed gold-bill generation run",
        clusters=state.cluster_successful_cases,
        bills=state.pricing_successful_cases,
        aftercare=state.aftercare_successful_cases,
        failures=state.cluster_failed_cases
        + state.pricing_failed_cases
        + state.aftercare_failed_cases,
        seconds=format_seconds(time.perf_counter() - start_time),
    )
    return 0
