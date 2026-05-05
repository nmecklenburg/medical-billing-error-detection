from __future__ import annotations

from typing import NamedTuple

from claim import gold_bill_support
from claim.normalization import normalize_code_system

CANONICALIZABLE_INPATIENT_ANCHOR_CODE_SYSTEMS = frozenset(
    {"MS-DRG", "APR-DRG", "DRG", "REVENUE_CODE"}
)


def _strip_prefix(raw_code: str, prefix: str) -> str:
    if raw_code.startswith(prefix):
        return raw_code[len(prefix) :]
    return raw_code


def canonicalize_inpatient_anchor_billing_code(
    code_system: str | None,
    billing_code: str | None,
) -> str | None:
    normalized_code_system = normalize_code_system(code_system)
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    if (
        normalized_code_system not in CANONICALIZABLE_INPATIENT_ANCHOR_CODE_SYSTEMS
        or normalized_billing_code is None
    ):
        return None

    candidate = normalized_billing_code
    if normalized_code_system == "MS-DRG":
        candidate = _strip_prefix(candidate, "MSDRG")
        if not candidate.isdigit() or len(candidate) > 3:
            return None
        return candidate.zfill(3)
    if normalized_code_system == "REVENUE_CODE":
        for prefix in ("REVENUECODE", "IPRVREVENUE"):
            candidate = _strip_prefix(candidate, prefix)
        if not candidate.isdigit() or len(candidate) > 4:
            return None
        return candidate.zfill(4)
    if normalized_code_system == "APR-DRG":
        candidate = _strip_prefix(candidate, "APRDRG")
        if not candidate.isdigit():
            return None
        return candidate
    if normalized_code_system == "DRG":
        candidate = _strip_prefix(candidate, "DRG")
        if not candidate.isdigit():
            return None
        return candidate
    return None


def canonicalize_inpatient_anchor_code_pair(
    code_system: str | None,
    billing_code: str | None,
) -> tuple[str | None, str | None]:
    normalized_code_system = normalize_code_system(code_system)
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    canonical_billing_code = canonicalize_inpatient_anchor_billing_code(
        normalized_code_system,
        normalized_billing_code,
    )
    if canonical_billing_code is not None:
        return normalized_code_system, canonical_billing_code
    return normalized_code_system, normalized_billing_code


class InpatientAnchorLookup(NamedTuple):
    code_system: str | None
    billing_code: str | None
    candidates: tuple[str, ...]


def inpatient_anchor_lookup(
    code_system: str | None,
    billing_code: str | None,
) -> InpatientAnchorLookup:
    normalized_code_system = normalize_code_system(code_system)
    normalized_billing_code = gold_bill_support.normalize_code(billing_code)
    canonical_billing_code = canonicalize_inpatient_anchor_billing_code(
        normalized_code_system,
        normalized_billing_code,
    )
    if normalized_code_system is None:
        return InpatientAnchorLookup(None, None, ())
    if normalized_billing_code is None:
        return InpatientAnchorLookup(normalized_code_system, None, ())

    if canonical_billing_code is None:
        return InpatientAnchorLookup(
            normalized_code_system,
            normalized_billing_code,
            (normalized_billing_code,),
        )

    candidates: list[str] = [canonical_billing_code]
    if normalized_code_system in {"MS-DRG", "REVENUE_CODE"}:
        unpadded = str(int(canonical_billing_code))
        if unpadded != canonical_billing_code:
            candidates.append(unpadded)
    if normalized_billing_code not in candidates:
        candidates.append(normalized_billing_code)
    return InpatientAnchorLookup(
        normalized_code_system,
        canonical_billing_code,
        tuple(candidates),
    )
