from __future__ import annotations

import re

from claim.gold_bill_support import normalize_text

CANONICAL_CODE_SYSTEMS = (
    "HCPCS/CPT",
    "MS-DRG",
    "APR-DRG",
    "DRG",
    "REVENUE_CODE",
    "ICD10_PCS",
    "ICD10",
    "ICD9_PROCEDURE",
    "ICD9",
)

_NORMALIZE_CODE_SYSTEM_RE = re.compile(r"[^A-Z0-9]+")


def _code_system_key(value: str) -> str:
    return _NORMALIZE_CODE_SYSTEM_RE.sub("", value.upper())


_CODE_SYSTEM_BY_KEY = {
    _code_system_key(code_system): code_system
    for code_system in CANONICAL_CODE_SYSTEMS
}
_CODE_SYSTEM_BY_KEY.update(
    {
        "HCPCS": "HCPCS/CPT",
        "CPT": "HCPCS/CPT",
        "MSDRG": "MS-DRG",
        "REVENUECODE": "REVENUE_CODE",
    }
)


def normalize_code_system(value: str | None) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    return _CODE_SYSTEM_BY_KEY.get(_code_system_key(text))
