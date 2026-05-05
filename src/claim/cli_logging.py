from __future__ import annotations

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import TextIO

ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"
ANSI_DIM = "\033[2m"
ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_YELLOW = "\033[33m"
ANSI_BLUE = "\033[34m"
ANSI_MAGENTA = "\033[35m"
ANSI_CYAN = "\033[36m"

LEVEL_LABELS = {
    "debug": "DEBUG",
    "info": "INFO ",
    "success": "OK   ",
    "warning": "WARN ",
    "error": "ERROR",
}

LEVEL_COLORS = {
    "debug": ANSI_MAGENTA,
    "info": ANSI_CYAN,
    "success": ANSI_GREEN,
    "warning": ANSI_YELLOW,
    "error": ANSI_RED,
}


def should_use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def format_fields(fields: dict[str, Any]) -> str:
    rendered: list[str] = []
    for key, value in fields.items():
        if value is None:
            continue
        if isinstance(value, float):
            rendered.append(f"{key}={value:.3f}")
        else:
            rendered.append(f"{key}={value}")
    return " ".join(rendered)


def format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.2f}s"


@dataclass(slots=True)
class CliLogger:
    name: str
    stream: TextIO = sys.stdout
    use_color: bool | None = None
    line_writer: Callable[[str, TextIO], None] | None = None

    def __post_init__(self) -> None:
        if self.use_color is None:
            self.use_color = should_use_color(self.stream)

    def _colorize(self, text: str, color: str, *, bold: bool = False, dim: bool = False) -> str:
        if not self.use_color:
            return text

        prefix = color
        if bold:
            prefix += ANSI_BOLD
        if dim:
            prefix += ANSI_DIM
        return f"{prefix}{text}{ANSI_RESET}"

    def _log(self, level: str, message: str, **fields: Any) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        label_text = self._colorize(LEVEL_LABELS[level], LEVEL_COLORS[level], bold=True)
        name_text = self._colorize(self.name, ANSI_BLUE, dim=True)
        message_text = self._colorize(message, "", bold=True)
        rendered_fields = format_fields(fields)
        if rendered_fields:
            rendered_fields = self._colorize(rendered_fields, ANSI_DIM, dim=True)
            line = f"{timestamp} {label_text} {name_text} {message_text} {rendered_fields}"
        else:
            line = f"{timestamp} {label_text} {name_text} {message_text}"
        if self.line_writer is not None:
            self.line_writer(line, self.stream)
            return
        print(line, file=self.stream, flush=True)

    def debug(self, message: str, **fields: Any) -> None:
        self._log("debug", message, **fields)

    def info(self, message: str, **fields: Any) -> None:
        self._log("info", message, **fields)

    def success(self, message: str, **fields: Any) -> None:
        self._log("success", message, **fields)

    def warning(self, message: str, **fields: Any) -> None:
        self._log("warning", message, **fields)

    def error(self, message: str, **fields: Any) -> None:
        self._log("error", message, **fields)
