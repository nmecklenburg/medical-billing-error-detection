from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

from claim.cli_logging import CliLogger, format_seconds


def positive_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def claim_payment_only(row: dict[str, Any]) -> float:
    total = 0.0
    for claim in row.get("claims", []):
        amounts = claim.get("amounts") or {}
        total += positive_float(amounts.get("claim_payment_amount"))
    return total


def claim_total_with_beneficiary(row: dict[str, Any]) -> float:
    total = 0.0
    for claim in row.get("claims", []):
        amounts = claim.get("amounts") or {}
        total += positive_float(amounts.get("claim_payment_amount"))
        total += positive_float(amounts.get("beneficiary_deductible_amount"))
        total += positive_float(amounts.get("beneficiary_coinsurance_amount"))
    return total


def carrier_line_paid_else_claim_payment(row: dict[str, Any]) -> float:
    total = 0.0
    for claim in row.get("claims", []):
        if claim.get("claim_type") == "carrier" and claim.get("line_items"):
            for line_item in claim["line_items"]:
                total += positive_float(
                    line_item.get("medicare_payment_amount")
                    or line_item.get("payment_amount")
                )
            continue
        amounts = claim.get("amounts") or {}
        total += positive_float(amounts.get("claim_payment_amount"))
    return total


def carrier_line_total_else_claim_total(row: dict[str, Any]) -> float:
    total = 0.0
    for claim in row.get("claims", []):
        if claim.get("claim_type") == "carrier" and claim.get("line_items"):
            for line_item in claim["line_items"]:
                allowed_amount = positive_float(line_item.get("allowed_charge_amount"))
                total += (
                    allowed_amount
                    if allowed_amount > 0
                    else positive_float(
                        line_item.get("medicare_payment_amount")
                        or line_item.get("payment_amount")
                    )
                )
                total += positive_float(line_item.get("beneficiary_deductible_amount"))
                total += positive_float(line_item.get("beneficiary_coinsurance_amount"))
            continue
        amounts = claim.get("amounts") or {}
        total += positive_float(amounts.get("claim_payment_amount"))
        total += positive_float(amounts.get("beneficiary_deductible_amount"))
        total += positive_float(amounts.get("beneficiary_coinsurance_amount"))
    return total


def allowed_or_paid_fallback_with_beneficiary(row: dict[str, Any]) -> float:
    total = 0.0
    for claim in row.get("claims", []):
        amounts = claim.get("amounts") or {}
        allowed_amount = positive_float(amounts.get("allowed_charge_amount"))
        total += (
            allowed_amount
            if allowed_amount > 0
            else positive_float(amounts.get("claim_payment_amount"))
        )
        total += positive_float(amounts.get("beneficiary_deductible_amount"))
        total += positive_float(amounts.get("beneficiary_coinsurance_amount"))
    return total


AGGREGATORS: dict[str, tuple[str, Callable[[dict[str, Any]], float]]] = {
    "claim-payment-only": (
        "Sum claim-header claim_payment_amount across claims in each encounter.",
        claim_payment_only,
    ),
    "claim-total-with-beneficiary": (
        "Sum claim-header claim_payment_amount plus deductible and coinsurance.",
        claim_total_with_beneficiary,
    ),
    "carrier-line-paid-else-claim-payment": (
        "Use carrier line-item medicare payment when available; otherwise use claim-header payment.",
        carrier_line_paid_else_claim_payment,
    ),
    "carrier-line-total-else-claim-total": (
        "Use carrier line-item allowed-or-paid amount plus beneficiary burden; otherwise use claim-header payment plus beneficiary burden.",
        carrier_line_total_else_claim_total,
    ),
    "allowed-or-paid-fallback-with-beneficiary": (
        "Use claim-header allowed amount when present, otherwise claim payment, then add beneficiary burden.",
        allowed_or_paid_fallback_with_beneficiary,
    ),
}


def quantile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute quantile of an empty series.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact encounter-level median and p99 totals by summing raw claim amounts "
            "within each encounter record of a sampled-encounter JSONL file."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Encounter JSONL path, e.g. outputs/datasets/DE-SynPUF/gold-bill-samples/sample-...jsonl",
    )
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=sorted(AGGREGATORS),
        help=(
            "Encounter-total definition to report. Repeat to report multiple modes. "
            "Defaults to claim-payment-only."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Report every supported encounter-total definition.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable JSON payload instead of plain text.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Emit a progress log every N encounters while scanning the input file.",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def selected_modes(args: argparse.Namespace) -> list[str]:
    if args.all:
        return list(AGGREGATORS)
    if args.modes:
        return args.modes
    return ["claim-payment-only"]


def format_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def render_header(logger: CliLogger, *, input_path: Path, modes: list[str]) -> None:
    logger.info("Encounter Total Quantiles")
    logger.info("Input", path=input_path)
    logger.info("Modes", selected=",".join(modes))


def main() -> int:
    args = parse_args()
    modes = selected_modes(args)
    totals_by_mode = {mode: [] for mode in modes}
    encounter_count = 0
    logger = CliLogger("encounter-quantiles")
    started_at = time.perf_counter()

    if not args.json:
        render_header(logger, input_path=args.input_path, modes=modes)

    with args.input_path.open() as handle:
        for line in handle:
            row = json.loads(line)
            encounter_count += 1
            for mode in modes:
                _, aggregator = AGGREGATORS[mode]
                totals_by_mode[mode].append(aggregator(row))
            if (
                not args.json
                and args.progress_every > 0
                and encounter_count % args.progress_every == 0
            ):
                elapsed = time.perf_counter() - started_at
                logger.info(
                    "Scanning encounters",
                    processed=encounter_count,
                    elapsed=format_seconds(elapsed),
                    rate=f"{encounter_count / max(elapsed, 1e-9):.0f}/s",
                )

    payload = {
        "input_path": str(args.input_path),
        "encounters": encounter_count,
        "results": [],
    }
    for mode in modes:
        totals = totals_by_mode[mode]
        description, _ = AGGREGATORS[mode]
        payload["results"].append(
            {
                "mode": mode,
                "description": description,
                "median": quantile(totals, 0.5),
                "p99": quantile(totals, 0.99),
                "min": min(totals),
                "max": max(totals),
            }
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    elapsed = time.perf_counter() - started_at
    logger.success(
        "Scan complete",
        encounters=encounter_count,
        elapsed=format_seconds(elapsed),
        rate=f"{encounter_count / max(elapsed, 1e-9):.0f}/s",
    )
    for result in payload["results"]:
        logger.info("Result", mode=result["mode"])
        logger.info("Definition", text=result["description"])
        logger.success(
            "Quantiles",
            median=format_currency(result["median"]),
            p99=format_currency(result["p99"]),
            min=format_currency(result["min"]),
            max=format_currency(result["max"]),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
