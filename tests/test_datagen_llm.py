from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from claim.datagen import constants, llm


class DatagenLlmClientSelectionTests(unittest.TestCase):
    def test_ensure_model_client_uses_openrouter_for_non_openai_provider(self) -> None:
        sentinel = object()
        with (
            patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=True),
            patch.object(llm, "OpenAI") as openai_mock,
        ):
            openai_mock.return_value = sentinel

            client = llm.ensure_model_client("qwen:qwen3.6-35b-a3b", 12.5)

        self.assertIs(client, sentinel)
        openai_mock.assert_called_once_with(
            api_key="test-key",
            base_url=constants.OPENROUTER_BASE_URL,
            timeout=12.5,
        )

    def test_ensure_model_client_uses_openai_for_openai_provider(self) -> None:
        sentinel = object()
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True),
            patch.object(llm, "OpenAI") as openai_mock,
        ):
            openai_mock.return_value = sentinel

            client = llm.ensure_model_client("openai:gpt-5-mini", 3.0)

        self.assertIs(client, sentinel)
        openai_mock.assert_called_once_with(timeout=3.0)

    def test_resolve_model_name_expands_non_openai_provider(self) -> None:
        self.assertEqual(
            llm.resolve_model_name("qwen:qwen3.6-35b-a3b"),
            "qwen/qwen3.6-35b-a3b",
        )

    def test_resolve_model_name_keeps_openai_model_name(self) -> None:
        self.assertEqual(llm.resolve_model_name("openai:gpt-5-mini"), "gpt-5-mini")


if __name__ == "__main__":
    unittest.main()
