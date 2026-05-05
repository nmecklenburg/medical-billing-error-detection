from __future__ import annotations

import hashlib
import random
import re
import threading
from datetime import UTC
from datetime import datetime
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence

NORMALIZE_CODE_RE = re.compile(r"[^A-Z0-9]+")
ZIP_RE = re.compile(r"\b\d{5}\b")
SPLIT_WORD_RE = re.compile(r"[^a-z0-9]+")


def iso_timestamp() -> str:
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.upper() in {"NULL", "NONE", "NAN"}:
        return None
    return normalized


def normalize_code(value: str | None) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    return NORMALIZE_CODE_RE.sub("", text.upper()) or None


def normalize_zip(value: str | None) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if len(digits) < 5:
        return None
    return digits[:5]


def normalize_state_code(value: str | None) -> str | None:
    text = normalize_text(value)
    if text is None:
        return None
    digits = "".join(character for character in text if character.isdigit())
    if digits:
        return digits.zfill(2)
    if len(text) == 2:
        return text.upper()
    return text


def parse_float(value: Any) -> float | None:
    text = normalize_text(str(value) if value is not None else None)
    if text is None:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def quote_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def round_currency(value: float) -> float:
    return round(float(value) + 1e-9, 2)


def split_words(value: str) -> list[str]:
    return [token for token in SPLIT_WORD_RE.split(value.lower()) if token]


def score_table_name(table_name: str, preferred_words: Sequence[str]) -> int:
    tokens = split_words(table_name)
    score = 0
    for word in preferred_words:
        if word in tokens:
            score += 10
        elif word in table_name.lower():
            score += 4
    return score


def score_column_name(column_name: str, aliases: Sequence[str]) -> int:
    normalized = column_name.lower()
    score = 0
    for alias in aliases:
        alias_normalized = alias.lower()
        if normalized == alias_normalized:
            score = max(score, 100)
        elif normalized.endswith(alias_normalized):
            score = max(score, 80)
        elif alias_normalized in normalized:
            score = max(score, 60)
        elif normalized.replace("_", "") == alias_normalized.replace("_", ""):
            score = max(score, 90)
    return score


def pick_best_column(
    columns: Sequence[str],
    aliases: Sequence[str],
) -> str | None:
    best_column: str | None = None
    best_score = 0
    for column in columns:
        score = score_column_name(column, aliases)
        if score > best_score:
            best_column = column
            best_score = score
    return best_column


def unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def fetch_rows_as_dicts(result: Any) -> list[dict[str, Any]]:
    rows = result.fetchall()
    columns = [description[0] for description in result.description]
    return [dict(zip(columns, row, strict=False)) for row in rows]


def synchronized_call(
    lock: threading.Lock | None,
    operation: Callable[[], Any],
) -> Any:
    if lock is None:
        return operation()
    with lock:
        return operation()
