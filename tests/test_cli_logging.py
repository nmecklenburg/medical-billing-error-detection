import io
import os
import unittest
from unittest.mock import patch

from claim.cli_logging import ANSI_BLUE
from claim.cli_logging import ANSI_GREEN
from claim.cli_logging import ANSI_RESET
from claim.cli_logging import CliLogger
from claim.cli_logging import format_fields
from claim.cli_logging import should_use_color


class _TtyStringIO(io.StringIO):
    def isatty(self) -> bool:
        return True


class CliLoggingTests(unittest.TestCase):
    def test_should_use_color_honors_no_color(self) -> None:
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=True):
            self.assertFalse(should_use_color(_TtyStringIO()))

    def test_should_use_color_requires_tty(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(should_use_color(_TtyStringIO()))
            self.assertFalse(should_use_color(io.StringIO()))

    def test_format_fields_skips_none_and_formats_floats(self) -> None:
        rendered = format_fields({"count": 2, "duration": 1.23456, "skip": None})

        self.assertEqual(rendered, "count=2 duration=1.235")

    def test_logger_renders_plain_output_without_color(self) -> None:
        stream = io.StringIO()
        logger = CliLogger("fetch", stream=stream, use_color=False)

        logger.info("Started", articles=3, delay=1.5)

        output = stream.getvalue()
        self.assertIn("INFO ", output)
        self.assertIn("fetch", output)
        self.assertIn("Started", output)
        self.assertIn("articles=3", output)
        self.assertIn("delay=1.500", output)
        self.assertNotIn("\033[", output)

    def test_logger_renders_colored_output(self) -> None:
        stream = io.StringIO()
        logger = CliLogger("classify", stream=stream, use_color=True)

        logger.success("Completed", classified=5)

        output = stream.getvalue()
        self.assertIn(f"{ANSI_GREEN}", output)
        self.assertIn(f"{ANSI_BLUE}", output)
        self.assertIn(f"{ANSI_RESET}", output)
        self.assertIn("Completed", output)
        self.assertIn("classified=5", output)

    def test_logger_uses_custom_line_writer_when_provided(self) -> None:
        captured: list[str] = []

        def line_writer(line: str, stream: io.StringIO) -> None:
            captured.append(line)
            stream.write(f"custom:{line}\n")

        stream = io.StringIO()
        logger = CliLogger("sample", stream=stream, use_color=False, line_writer=line_writer)

        logger.info("Started", sample_size=1)

        self.assertEqual(len(captured), 1)
        self.assertIn("Started", captured[0])
        self.assertIn("custom:", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
