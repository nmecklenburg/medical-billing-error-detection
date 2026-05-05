"""Main entrypoint for generating clean bills."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from claim.datagen import constants
from claim.datagen import io as datagen_io
from claim.datagen.contracts import GenerationRunConfig, WorkerConfiguration
from claim.datagen.encounters import ensure_encounter_cache, resolve_encounter_cache_path
from claim.datagen.encounters import constants as encounter_constants
from claim.datagen.encounters.contracts import EncounterBuildConfig
from claim.datagen.io import (
    default_description_cache_path,
    default_trilliant_cache_path,
    effective_sample_size,
    ensure_output_paths,
    load_env_file,
    sampled_encounters_path,
    should_use_progress,
)
from claim.datagen.pipeline import run_generation_pipeline
from claim.cli_logging import CliLogger
from claim import gold_bill_trilliant


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sample DE-SynPUF encounters, enrich them with geography and pricing, "
            "and generate gold-standard bill JSON plus after-care summaries."
        )
    )
    parser.add_argument(
        "--encounter-cache-path",
        type=Path,
        default=None,
        help=(
            "Optional path to the DE-SynPUF encounter cache JSONL. If omitted, "
            "the path is derived from --dataset-root and --carrier-match-window-days."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=constants.DEFAULT_DATASET_ROOT,
        help="Directory where DE-SynPUF source files, extracted CSVs, and default encounter caches live.",
    )
    parser.add_argument(
        "--carrier-match-window-days",
        type=int,
        default=encounter_constants.DEFAULT_CARRIER_MATCH_WINDOW_DAYS,
        help=(
            "Slack days used when building a missing encounter cache. Defaults to "
            f"{encounter_constants.DEFAULT_CARRIER_MATCH_WINDOW_DAYS}."
        ),
    )
    parser.add_argument(
        "--rebuild-encounter-cache",
        action="store_true",
        help="Rebuild the encounter cache before sampling, even if it already exists.",
    )
    parser.add_argument(
        "--sampled-encounters-path",
        type=Path,
        default=None,
        help="Optional override for the intermediate sampled-encounters JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=constants.DEFAULT_OUTPUT_SUBDIR,
        help=(
            "Directory where inpatient_bills.jsonl, outpatient_bills.jsonl, "
            "aftercare-summaries.jsonl, and run metadata are written."
        ),
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=constants.DEFAULT_REFERENCE_DATA_DIR,
        help="Directory for cached ICD and geography reference data.",
    )
    parser.add_argument(
        "--trilliant-cache-dir",
        type=Path,
        default=constants.DEFAULT_TRILLIANT_CACHE_DIR,
        help="Directory for the local DuckDB cache used for Trilliant schema discovery and rate lookups.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=constants.DEFAULT_SAMPLE_SIZE,
        help=f"Number of eligible encounters to sample. Defaults to {constants.DEFAULT_SAMPLE_SIZE}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=constants.DEFAULT_SEED,
        help=f"Random seed for deterministic sampling and scenario selection. Defaults to {constants.DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--test-one",
        action="store_true",
        help="Process exactly one deterministic sampled encounter. Overrides --sample-size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sampled or final output files.",
    )
    parser.add_argument(
        "--model",
        dest="model",
        default=constants.DEFAULT_MODEL,
        help=(
            "Model spec in provider:model format. `openai:<model>` uses the OpenAI "
            "client; any other provider uses OpenRouter. Defaults to "
            f"{constants.DEFAULT_MODEL}."
        ),
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default=constants.DEFAULT_REASONING_EFFORT,
        help=f"Reasoning effort to request from the model. Defaults to {constants.DEFAULT_REASONING_EFFORT}.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=constants.DEFAULT_TIMEOUT_SECONDS,
        help=f"Client timeout for LLM API requests. Defaults to {constants.DEFAULT_TIMEOUT_SECONDS}.",
    )
    parser.add_argument(
        "--request-delay-seconds",
        type=float,
        default=constants.DEFAULT_REQUEST_DELAY_SECONDS,
        help="Optional delay between LLM API requests.",
    )
    parser.add_argument(
        "--extreme-case-probability",
        type=float,
        default=constants.DEFAULT_EXTREME_CASE_PROBABILITY,
        help=(
            "Probability of generating an extreme out-of-network or self-pay scenario. "
            f"Defaults to {constants.DEFAULT_EXTREME_CASE_PROBABILITY}."
        ),
    )
    parser.add_argument(
        "--insurance-coverage-ratio",
        type=float,
        default=constants.DEFAULT_INSURANCE_COVERAGE_RATIO,
        help=(
            "Share of allowed amount paid by insurance for in-network cases. "
            f"Defaults to {constants.DEFAULT_INSURANCE_COVERAGE_RATIO}."
        ),
    )
    parser.add_argument(
        "--billed-charge-multiplier",
        type=float,
        default=constants.DEFAULT_BILLED_CHARGE_MULTIPLIER,
        help=(
            "Fallback multiplier for deriving billed charges from allowed amounts when "
            "gross-charge data is missing. "
            f"Defaults to {constants.DEFAULT_BILLED_CHARGE_MULTIPLIER}."
        ),
    )
    parser.add_argument(
        "--source-medicare-multiplier",
        type=float,
        default=constants.DEFAULT_SOURCE_MEDICARE_MULTIPLIER,
        help=(
            "Fallback multiplier applied to Medicare-like source amounts when Trilliant "
            "has no rate data for a billing code. "
            f"Defaults to {constants.DEFAULT_SOURCE_MEDICARE_MULTIPLIER}."
        ),
    )
    parser.add_argument(
        "--normal-payer",
        action="append",
        default=[],
        help=(
            "Allowed payer name fragment for normal cases. Repeat the flag to add "
            "multiple payers. If omitted, a default shortlist is used."
        ),
    )
    parser.add_argument(
        "--trilliant-db-url",
        default=gold_bill_trilliant.TRILLIANT_DB_URL,
        help="Remote DuckDB URL for the Trilliant MRF data lake.",
    )
    parser.add_argument(
        "--trilliant-hospital-table",
        default=None,
        help="Optional override for the Trilliant hospital table name.",
    )
    parser.add_argument(
        "--trilliant-rate-table",
        default=None,
        help="Optional override for the Trilliant rate table name.",
    )
    parser.add_argument(
        "--hud-crosswalk-path",
        type=Path,
        default=None,
        help="Optional local HUD ZIP-to-county crosswalk CSV or XLSX override.",
    )
    parser.add_argument(
        "--hud-user-token",
        default=os.environ.get("HUD_USER_API_TOKEN"),
        help=(
            "Optional HUD USER API token for on-demand county-to-ZIP lookups. "
            "Defaults to HUD_USER_API_TOKEN from the environment if set."
        ),
    )
    parser.add_argument(
        "--max-hospital-candidates",
        type=int,
        default=constants.DEFAULT_MAX_HOSPITAL_CANDIDATES,
        help=(
            "Limit on hospital candidates retrieved from Trilliant before deterministic "
            "selection. Defaults to "
            f"{constants.DEFAULT_MAX_HOSPITAL_CANDIDATES}."
        ),
    )
    parser.add_argument(
        "--cluster-workers",
        type=int,
        default=constants.DEFAULT_CLUSTER_WORKERS,
        help=(
            "Number of concurrent cluster-generation workers. This stage runs the structured "
            "LLM call that turns a raw sampled encounter cluster into a coherent synthetic "
            f"ICD-10 + HCPCS/CPT cluster. Defaults to {constants.DEFAULT_CLUSTER_WORKERS}."
        ),
    )
    parser.add_argument(
        "--pricing-workers",
        type=int,
        default=constants.DEFAULT_PRICING_WORKERS,
        help=(
            "Number of concurrent pricing/final bill assembly workers. This stage handles "
            "facility selection, Trilliant lookup, accounting, and final bill assembly. "
            f"Defaults to {constants.DEFAULT_PRICING_WORKERS}."
        ),
    )
    parser.add_argument(
        "--bill-workers",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--max-hospital-rate-probes",
        type=int,
        default=constants.DEFAULT_MAX_HOSPITAL_RATE_PROBES,
        help=(
            "Maximum number of candidate hospitals to probe for real rate coverage in the "
            "public-directory Trilliant flow before falling back to imputation for missing codes. "
            f"Defaults to {constants.DEFAULT_MAX_HOSPITAL_RATE_PROBES}."
        ),
    )
    parser.add_argument(
        "--max-nearby-fallback-probes",
        type=int,
        default=constants.DEFAULT_MAX_NEARBY_HOSPITAL_FALLBACK_PROBES,
        help=(
            "Maximum number of additional nearby hospitals to probe after selected-hospital "
            "pricing fails for professional HCPCS/CPT lines. Defaults to "
            f"{constants.DEFAULT_MAX_NEARBY_HOSPITAL_FALLBACK_PROBES}."
        ),
    )
    parser.add_argument(
        "--aftercare-workers",
        type=int,
        default=constants.DEFAULT_AFTERCARE_WORKERS,
        help=(
            "Number of concurrent after-care generation workers. After-care summaries run as "
            "a separate stage over the completed split bill outputs. "
            f"Defaults to {constants.DEFAULT_AFTERCARE_WORKERS}."
        ),
    )
    parser.add_argument(
        "--max-quote-amount",
        type=float,
        default=constants.DEFAULT_MAX_QUOTE_AMOUNT,
        help=(
            "Reject Trilliant quote amounts above this threshold as implausible and continue "
            "through the fallback waterfall. Defaults to "
            f"{constants.DEFAULT_MAX_QUOTE_AMOUNT:,.0f}."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_arg_parser().parse_args(argv)


def cluster_worker_count(args: argparse.Namespace) -> int:
    if args.bill_workers is not None:
        return max(1, int(args.bill_workers))
    return max(1, int(args.cluster_workers))


def pricing_worker_count(args: argparse.Namespace) -> int:
    if args.bill_workers is not None:
        return max(1, int(args.bill_workers))
    return max(1, int(args.pricing_workers))


def aftercare_worker_count(args: argparse.Namespace) -> int:
    return max(1, int(args.aftercare_workers))


def effective_encounter_cache_path(args: argparse.Namespace) -> Path:
    return resolve_encounter_cache_path(
        args.dataset_root,
        encounter_constants.DEFAULT_SAMPLE_NUMBER,
        int(args.carrier_match_window_days),
        args.encounter_cache_path,
    )


def build_encounter_build_config(
    args: argparse.Namespace,
    *,
    show_progress: bool,
) -> EncounterBuildConfig:
    return EncounterBuildConfig(
        dataset_root=args.dataset_root,
        encounter_cache_path=effective_encounter_cache_path(args),
        sample_number=encounter_constants.DEFAULT_SAMPLE_NUMBER,
        carrier_match_window_days=int(args.carrier_match_window_days),
        rebuild=bool(args.rebuild_encounter_cache),
        show_progress=show_progress,
    )


def build_run_config(
    args: argparse.Namespace,
    *,
    show_progress: bool | None = None,
) -> GenerationRunConfig:
    effective_size = effective_sample_size(args.sample_size, args.test_one)
    encounter_cache_path = effective_encounter_cache_path(args)
    sampled_path = args.sampled_encounters_path or sampled_encounters_path(
        encounter_cache_path,
        args.sample_size,
        args.seed,
        args.test_one,
    )
    output_paths = ensure_output_paths(args.output_dir)
    workers = WorkerConfiguration(
        cluster_workers=cluster_worker_count(args),
        pricing_workers=pricing_worker_count(args),
        aftercare_workers=aftercare_worker_count(args),
    )
    return GenerationRunConfig(
        encounter_cache_path=encounter_cache_path,
        sampled_encounters_path=sampled_path,
        output_dir=args.output_dir,
        reference_data_dir=args.reference_data_dir,
        trilliant_cache_dir=args.trilliant_cache_dir,
        output_paths=output_paths,
        description_cache_path=default_description_cache_path(args.reference_data_dir),
        trilliant_cache_path=default_trilliant_cache_path(args.trilliant_cache_dir),
        sample_size=effective_size,
        seed=int(args.seed),
        test_one=bool(args.test_one),
        overwrite=bool(args.overwrite),
        model=str(args.model),
        reasoning_effort=str(args.reasoning_effort),
        timeout_seconds=float(args.timeout_seconds),
        request_delay_seconds=float(args.request_delay_seconds),
        extreme_case_probability=float(args.extreme_case_probability),
        insurance_coverage_ratio=float(args.insurance_coverage_ratio),
        billed_charge_multiplier=float(args.billed_charge_multiplier),
        source_medicare_multiplier=float(args.source_medicare_multiplier),
        normal_payers=tuple(
            args.normal_payer or gold_bill_trilliant.DEFAULT_NORMAL_PAYERS
        ),
        trilliant_db_url=str(args.trilliant_db_url),
        trilliant_hospital_table=args.trilliant_hospital_table,
        trilliant_rate_table=args.trilliant_rate_table,
        hud_crosswalk_path=args.hud_crosswalk_path,
        hud_user_token=args.hud_user_token,
        max_hospital_candidates=int(args.max_hospital_candidates),
        workers=workers,
        max_hospital_rate_probes=int(args.max_hospital_rate_probes),
        max_quote_amount=float(args.max_quote_amount),
        show_progress=should_use_progress() if show_progress is None else show_progress,
        max_nearby_fallback_probes=int(args.max_nearby_fallback_probes),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    load_env_file(constants.DEFAULT_DOTENV_PATH)
    show_progress = should_use_progress()
    encounter_cache_path = effective_encounter_cache_path(args)
    if bool(args.rebuild_encounter_cache) or not encounter_cache_path.exists():
        logger = CliLogger(
            "gold-bills",
            line_writer=datagen_io.tqdm_line_writer if show_progress else None,
        )
        ensure_encounter_cache(
            build_encounter_build_config(args, show_progress=show_progress),
            logger,
        )
    config = build_run_config(args, show_progress=show_progress)
    return run_generation_pipeline(config)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
