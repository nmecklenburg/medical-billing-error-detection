import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock
from unittest.mock import patch

from claim.cli_logging import format_seconds
from scripts.classify_kff_docs import PRESET_TOPICS
from scripts.classify_kff_docs import aggregate_telemetry
from scripts.classify_kff_docs import build_manifest_payload
from scripts.classify_kff_docs import build_prompt_text
from scripts.classify_kff_docs import build_source_context_index
from scripts.classify_kff_docs import default_results_path
from scripts.classify_kff_docs import detect_cpt_hcpcs_codes
from scripts.classify_kff_docs import discover_pdf_paths
from scripts.classify_kff_docs import extract_pdf_text
from scripts.classify_kff_docs import load_env_file
from scripts.classify_kff_docs import normalize_classification_result
from scripts.classify_kff_docs import pick_test_one_pdf
from scripts.classify_kff_docs import source_contexts_for_pdf
from scripts.classify_kff_docs import truncate_text_for_prompt
from scripts.classify_kff_docs import write_json_atomic


class ClassifyKffDocsTests(unittest.TestCase):
    def test_discover_pdf_paths_filters_by_match(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a-eob.pdf").write_bytes(b"pdf")
            (root / "b-bill.pdf").write_bytes(b"pdf")
            (root / "notes.txt").write_text("ignore", encoding="utf-8")

            paths = discover_pdf_paths(root, match="bill")

            self.assertEqual(paths, [root / "b-bill.pdf"])

    def test_pick_test_one_pdf_is_deterministic(self) -> None:
        pdf_paths = [
            Path("/tmp/a.pdf"),
            Path("/tmp/b.pdf"),
            Path("/tmp/c.pdf"),
            Path("/tmp/d.pdf"),
        ]

        first_pick = pick_test_one_pdf(pdf_paths)
        second_pick = pick_test_one_pdf(pdf_paths)

        self.assertEqual(first_pick, second_pick)
        self.assertIn(first_pick, pdf_paths)

    def test_load_env_file_sets_openai_key_from_dotenv(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=from_dotenv\n", encoding="utf-8")

            with patch.dict(os.environ, {}, clear=True):
                loaded = load_env_file(dotenv_path)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "from_dotenv")

    def test_load_env_file_does_not_override_existing_env_var(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dotenv_path = Path(tmpdir) / ".env"
            dotenv_path.write_text("OPENAI_API_KEY=from_dotenv\n", encoding="utf-8")

            with patch.dict(os.environ, {"OPENAI_API_KEY": "already_set"}, clear=True):
                loaded = load_env_file(dotenv_path)

                self.assertTrue(loaded)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "already_set")

    def test_format_seconds_uses_ms_for_short_durations(self) -> None:
        self.assertEqual(format_seconds(0.1234), "123ms")
        self.assertEqual(format_seconds(1.23), "1.23s")

    def test_source_context_index_matches_by_absolute_path_and_filename(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            docs_dir = output_dir / "kff-bills"
            docs_dir.mkdir(parents=True)
            pdf_path = docs_dir / "4353248-Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.pdf"
            pdf_path.write_bytes(b"pdf")

            metadata_records = [
                {
                    "article_url": "https://kffhealthnews.org/news/example/",
                    "documents": [
                        {
                            "pdf_url": "https://s3.documentcloud.org/documents/4353248/Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.pdf",
                            "output_path": str(pdf_path),
                            "references": [
                                {
                                    "tag": "a",
                                    "url": "https://www.documentcloud.org/documents/4353248-Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.html",
                                }
                            ],
                        }
                    ],
                }
            ]

            source_index = build_source_context_index(metadata_records)
            contexts = source_contexts_for_pdf(pdf_path, source_index)

            self.assertEqual(
                contexts,
                [
                    {
                        "article_url": "https://kffhealthnews.org/news/example/",
                        "pdf_url": "https://s3.documentcloud.org/documents/4353248/Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.pdf",
                        "references": [
                            {
                                "tag": "a",
                                "url": "https://www.documentcloud.org/documents/4353248-Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.html",
                            }
                        ],
                    }
                ],
            )

    def test_normalize_classification_result_prefers_preset_and_slugifies_custom(self) -> None:
        preset_result = normalize_classification_result(
            {
                "topic": "Explanation Of Benefits",
                "topic_source": "custom",
                "is_medical_billing_related": True,
                "confidence": 1.2,
                "summary": "  EOB  ",
                "evidence": [" states member responsibility "],
                "secondary_topics": ["Insurance Claim", "insurance_claim"],
            }
        )
        custom_result = normalize_classification_result(
            {
                "topic": "White Paper",
                "topic_source": "preset",
                "is_medical_billing_related": False,
                "confidence": -0.2,
                "summary": "",
                "evidence": [],
                "secondary_topics": [],
            }
        )

        self.assertEqual(preset_result["topic"], "explanation_of_benefits")
        self.assertEqual(preset_result["topic_source"], "preset")
        self.assertEqual(preset_result["confidence"], 1.0)
        self.assertEqual(preset_result["secondary_topics"], ["insurance_claim"])

        self.assertEqual(custom_result["topic"], "white_paper")
        self.assertEqual(custom_result["topic_source"], "custom")
        self.assertEqual(custom_result["confidence"], 0.0)
        self.assertEqual(custom_result["evidence"], ["No supporting evidence provided."])

    def test_build_prompt_includes_preset_topics_and_source_context(self) -> None:
        prompt = build_prompt_text(
            Path("/tmp/4353248-example.pdf"),
            [
                {
                    "article_url": "https://kffhealthnews.org/news/example/",
                    "pdf_url": "https://s3.documentcloud.org/documents/4353248/example.pdf",
                    "references": [
                        {"tag": "a", "url": "https://www.documentcloud.org/documents/4353248-example.html"}
                    ],
                }
            ],
            {
                "method": "pypdf",
                "page_count": 2,
                "parser_char_count": 123,
                "parser_pages_with_text": 2,
                "pages_with_text": 2,
                "page_error_count": 0,
                "extracted_char_count": 123,
                "prompt_char_count": 123,
                "text_truncated": False,
                "text_found": True,
                "ocr_attempted": False,
                "ocr_used": False,
                "ocr_status": "not_needed",
                "ocr_min_parser_chars": 100,
                "ocr_engine": None,
                "ocr_pages_attempted": 0,
                "ocr_pages_with_text": 0,
                "ocr_error_count": 0,
                "ocr_unavailable_reason": None,
                "cpt_hcpcs_codes_present": True,
                "cpt_hcpcs_code_examples": ["A0429", "99213"],
                "prompt_text": "[Page 1]\nExample text",
            },
        )

        self.assertIn("Filename: 4353248-example.pdf", prompt)
        self.assertIn("Text extraction method: pypdf", prompt)
        self.assertIn("CPT/HCPCS codes detected: yes", prompt)
        self.assertIn("CPT/HCPCS code examples: A0429, 99213", prompt)
        self.assertIn("https://kffhealthnews.org/news/example/", prompt)
        self.assertIn("https://www.documentcloud.org/documents/4353248-example.html", prompt)
        self.assertIn("[Page 1]\nExample text", prompt)
        for topic in PRESET_TOPICS:
            self.assertIn(f"- {topic}", prompt)

    def test_truncate_text_for_prompt_keeps_head_and_tail(self) -> None:
        text = "A" * 80 + "B" * 80

        truncated, was_truncated = truncate_text_for_prompt(text, 100)

        self.assertTrue(was_truncated)
        self.assertIn("[... truncated ...]", truncated)
        self.assertTrue(truncated.startswith("A"))
        self.assertTrue(truncated.endswith("B"))

    def test_detect_cpt_hcpcs_codes_finds_hcpcs_and_contextual_cpt_codes(self) -> None:
        detection = detect_cpt_hcpcs_codes(
            "HCPCS A0429 ambulance transport. CPT code 99213 established patient visit."
        )

        self.assertTrue(detection["cpt_hcpcs_codes_present"])
        self.assertEqual(detection["cpt_hcpcs_code_examples"], ["A0429", "99213"])

    def test_detect_cpt_hcpcs_codes_ignores_plain_five_digit_numbers(self) -> None:
        detection = detect_cpt_hcpcs_codes(
            "Statement date 2024-03-01. Account 12345. ZIP 94105. Total charge 54321."
        )

        self.assertFalse(detection["cpt_hcpcs_codes_present"])
        self.assertEqual(detection["cpt_hcpcs_code_examples"], [])

    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_collects_page_text_and_stats(self, mock_pdf_reader: MagicMock) -> None:
        reader = MagicMock()
        page_1 = MagicMock()
        page_1.extract_text.return_value = " Hello   world \n\nAmount due "
        page_2 = MagicMock()
        page_2.extract_text.return_value = ""
        page_3 = MagicMock()
        page_3.extract_text.return_value = "Page three\nClaim number"
        reader.pages = [page_1, page_2, page_3]
        mock_pdf_reader.return_value = reader

        extraction = extract_pdf_text(
            Path("/tmp/example.pdf"),
            max_text_chars=10_000,
            ocr_min_parser_chars=0,
        )

        self.assertEqual(extraction["method"], "pypdf")
        self.assertEqual(extraction["page_count"], 3)
        self.assertGreater(extraction["parser_char_count"], 0)
        self.assertEqual(extraction["parser_pages_with_text"], 2)
        self.assertEqual(extraction["pages_with_text"], 2)
        self.assertEqual(extraction["page_error_count"], 0)
        self.assertTrue(extraction["text_found"])
        self.assertEqual(extraction["ocr_status"], "not_needed")
        self.assertFalse(extraction["ocr_used"])
        self.assertFalse(extraction["cpt_hcpcs_codes_present"])
        self.assertEqual(extraction["cpt_hcpcs_code_examples"], [])
        self.assertIn("[Page 1]\nHello world\nAmount due", extraction["prompt_text"])
        self.assertIn("[Page 3]\nPage three\nClaim number", extraction["prompt_text"])

    @patch("scripts.classify_kff_docs.ocr_pdf_document")
    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_uses_ocr_when_parser_text_is_below_threshold(
        self,
        mock_pdf_reader: MagicMock,
        mock_ocr_pdf_document: MagicMock,
    ) -> None:
        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "Bill summary"
        reader.pages = [page]
        mock_pdf_reader.return_value = reader
        mock_ocr_pdf_document.return_value = {
            "available": True,
            "reason": None,
            "engine": "tesseract",
            "pages_attempted": 1,
            "pages_with_text": 1,
            "error_count": 0,
            "full_text": "[Page 1]\nDetailed itemized bill with claim number and total amount due",
        }

        extraction = extract_pdf_text(
            Path("/tmp/sparse-parser.pdf"),
            max_text_chars=10_000,
            ocr_min_parser_chars=100,
        )

        self.assertEqual(extraction["method"], "pypdf+ocr")
        self.assertLess(extraction["parser_char_count"], 100)
        self.assertTrue(extraction["ocr_attempted"])
        self.assertTrue(extraction["ocr_used"])
        self.assertEqual(extraction["ocr_status"], "used")
        self.assertFalse(extraction["cpt_hcpcs_codes_present"])
        self.assertIn("Detailed itemized bill", extraction["prompt_text"])

    @patch("scripts.classify_kff_docs.ocr_pdf_document")
    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_skips_ocr_when_parser_text_meets_threshold(
        self,
        mock_pdf_reader: MagicMock,
        mock_ocr_pdf_document: MagicMock,
    ) -> None:
        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "A" * 200
        reader.pages = [page]
        mock_pdf_reader.return_value = reader

        extraction = extract_pdf_text(
            Path("/tmp/parser-sufficient.pdf"),
            max_text_chars=10_000,
            ocr_min_parser_chars=100,
        )

        mock_ocr_pdf_document.assert_not_called()
        self.assertEqual(extraction["method"], "pypdf")
        self.assertGreaterEqual(extraction["parser_char_count"], 100)
        self.assertFalse(extraction["ocr_attempted"])
        self.assertEqual(extraction["ocr_status"], "not_needed")

    @patch("scripts.classify_kff_docs.ocr_pdf_document")
    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_falls_back_to_ocr_for_image_only_pdf(
        self,
        mock_pdf_reader: MagicMock,
        mock_ocr_pdf_document: MagicMock,
    ) -> None:
        reader = MagicMock()
        page_1 = MagicMock()
        page_1.extract_text.return_value = ""
        page_2 = MagicMock()
        page_2.extract_text.return_value = ""
        reader.pages = [page_1, page_2]
        mock_pdf_reader.return_value = reader
        mock_ocr_pdf_document.return_value = {
            "available": True,
            "reason": None,
            "engine": "tesseract",
            "pages_attempted": 2,
            "pages_with_text": 2,
            "error_count": 0,
            "full_text": "[Page 1]\nTotal amount due\n\n[Page 2]\nClaim number",
        }

        extraction = extract_pdf_text(Path("/tmp/image-only.pdf"), max_text_chars=10_000)

        self.assertEqual(extraction["method"], "pypdf+ocr")
        self.assertEqual(extraction["parser_char_count"], 0)
        self.assertEqual(extraction["parser_pages_with_text"], 0)
        self.assertEqual(extraction["pages_with_text"], 2)
        self.assertTrue(extraction["ocr_attempted"])
        self.assertTrue(extraction["ocr_used"])
        self.assertEqual(extraction["ocr_status"], "used")
        self.assertEqual(extraction["ocr_engine"], "tesseract")
        self.assertEqual(extraction["ocr_pages_attempted"], 2)
        self.assertEqual(extraction["ocr_pages_with_text"], 2)
        self.assertTrue(extraction["text_found"])
        self.assertFalse(extraction["cpt_hcpcs_codes_present"])
        self.assertIn("Total amount due", extraction["prompt_text"])

    @patch("scripts.classify_kff_docs.ocr_pdf_document")
    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_records_unavailable_ocr_for_image_only_pdf(
        self,
        mock_pdf_reader: MagicMock,
        mock_ocr_pdf_document: MagicMock,
    ) -> None:
        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = ""
        reader.pages = [page]
        mock_pdf_reader.return_value = reader
        mock_ocr_pdf_document.return_value = {
            "available": False,
            "reason": "tesseract is not available on PATH",
            "engine": None,
            "pages_attempted": 0,
            "pages_with_text": 0,
            "error_count": 0,
            "full_text": "",
        }

        extraction = extract_pdf_text(Path("/tmp/unavailable-ocr.pdf"), max_text_chars=10_000)

        self.assertEqual(extraction["method"], "pypdf")
        self.assertEqual(extraction["parser_char_count"], 0)
        self.assertEqual(extraction["ocr_status"], "unavailable")
        self.assertFalse(extraction["ocr_attempted"])
        self.assertFalse(extraction["ocr_used"])
        self.assertEqual(
            extraction["ocr_unavailable_reason"],
            "tesseract is not available on PATH",
        )
        self.assertFalse(extraction["text_found"])

    @patch("scripts.classify_kff_docs.PdfReader")
    def test_extract_pdf_text_marks_detected_cpt_hcpcs_codes(self, mock_pdf_reader: MagicMock) -> None:
        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "HCPCS A0429\nCPT code 99213\nAmount due"
        reader.pages = [page]
        mock_pdf_reader.return_value = reader

        extraction = extract_pdf_text(
            Path("/tmp/codes.pdf"),
            max_text_chars=10_000,
            ocr_min_parser_chars=0,
        )

        self.assertTrue(extraction["cpt_hcpcs_codes_present"])
        self.assertEqual(extraction["cpt_hcpcs_code_examples"], ["A0429", "99213"])

    def test_manifest_payload_summarizes_topics(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "outputs"
            results_path = default_results_path(output_dir)
            documents = [
                {
                    "pdf_path": "/tmp/a.pdf",
                    "pdf_name": "a.pdf",
                    "status": "classified",
                    "text_extraction": {
                        "text_found": True,
                        "page_error_count": 0,
                        "extracted_char_count": 200,
                        "prompt_char_count": 150,
                        "ocr_attempted": False,
                        "ocr_used": False,
                        "ocr_status": "not_needed",
                        "ocr_pages_attempted": 0,
                        "ocr_pages_with_text": 0,
                        "cpt_hcpcs_codes_present": True,
                        "cpt_hcpcs_code_examples": ["99213"],
                    },
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 20,
                        "total_tokens": 120,
                    },
                    "telemetry": {
                        "processing_seconds": 1.25,
                        "extraction_seconds": 0.2,
                        "classification_seconds": 1.0,
                    },
                    "classification": {"topic": "bill"},
                },
                {
                    "pdf_path": "/tmp/b.pdf",
                    "pdf_name": "b.pdf",
                    "status": "classified",
                    "text_extraction": {
                        "text_found": False,
                        "page_error_count": 1,
                        "extracted_char_count": 0,
                        "prompt_char_count": 0,
                        "ocr_attempted": True,
                        "ocr_used": False,
                        "ocr_status": "failed",
                        "ocr_pages_attempted": 2,
                        "ocr_pages_with_text": 0,
                        "cpt_hcpcs_codes_present": False,
                        "cpt_hcpcs_code_examples": [],
                    },
                    "usage": {
                        "input_tokens": 50,
                        "output_tokens": 10,
                        "total_tokens": 60,
                    },
                    "telemetry": {
                        "processing_seconds": 2.5,
                        "extraction_seconds": 0.4,
                        "classification_seconds": 1.8,
                    },
                    "classification": {"topic": "bill"},
                    "last_run_action": "skipped_existing",
                },
                {
                    "pdf_path": "/tmp/c.pdf",
                    "pdf_name": "c.pdf",
                    "status": "failed",
                },
            ]

            payload = build_manifest_payload(
                model="gpt-5.4",
                reasoning_effort="low",
                results_path=results_path,
                metadata_path=None,
                documents=documents,
            )

            self.assertEqual(payload["summary"]["classified"], 2)
            self.assertEqual(payload["summary"]["failed"], 1)
            self.assertEqual(payload["summary"]["skipped_existing"], 1)
            self.assertEqual(payload["summary"]["topic_counts"], {"bill": 2})
            self.assertEqual(payload["telemetry"]["documents_with_text"], 1)
            self.assertEqual(payload["telemetry"]["documents_without_text"], 1)
            self.assertEqual(payload["telemetry"]["documents_with_extraction_errors"], 1)
            self.assertEqual(payload["telemetry"]["documents_with_ocr_attempted"], 1)
            self.assertEqual(payload["telemetry"]["documents_with_ocr_used"], 0)
            self.assertEqual(payload["telemetry"]["documents_with_cpt_hcpcs_codes"], 1)
            self.assertEqual(payload["telemetry"]["input_tokens"], 150)
            self.assertEqual(payload["telemetry"]["output_tokens"], 30)
            self.assertEqual(payload["telemetry"]["total_tokens"], 180)

    def test_aggregate_telemetry_sums_document_metrics(self) -> None:
        telemetry = aggregate_telemetry(
            [
                {
                    "text_extraction": {
                        "text_found": True,
                        "page_error_count": 0,
                        "extracted_char_count": 100,
                        "prompt_char_count": 80,
                        "ocr_attempted": False,
                        "ocr_used": False,
                        "ocr_status": "not_needed",
                        "ocr_pages_attempted": 0,
                        "ocr_pages_with_text": 0,
                        "cpt_hcpcs_codes_present": False,
                        "cpt_hcpcs_code_examples": [],
                    },
                    "usage": {
                        "input_tokens": 10,
                        "output_tokens": 3,
                        "total_tokens": 13,
                    },
                    "telemetry": {
                        "processing_seconds": 1.0,
                        "extraction_seconds": 0.2,
                        "classification_seconds": 0.7,
                    },
                },
                {
                    "text_extraction": {
                        "text_found": True,
                        "page_error_count": 2,
                        "extracted_char_count": 60,
                        "prompt_char_count": 55,
                        "ocr_attempted": True,
                        "ocr_used": True,
                        "ocr_status": "used",
                        "ocr_pages_attempted": 2,
                        "ocr_pages_with_text": 1,
                        "cpt_hcpcs_codes_present": True,
                        "cpt_hcpcs_code_examples": ["A0429"],
                    },
                    "usage": {
                        "input_tokens": 5,
                        "output_tokens": 1,
                        "total_tokens": 6,
                    },
                    "telemetry": {
                        "processing_seconds": 2.0,
                        "extraction_seconds": 0.5,
                        "classification_seconds": 1.2,
                    },
                },
            ]
        )

        self.assertEqual(telemetry["documents_with_text"], 2)
        self.assertEqual(telemetry["documents_without_text"], 0)
        self.assertEqual(telemetry["documents_with_extraction_errors"], 1)
        self.assertEqual(telemetry["documents_with_ocr_attempted"], 1)
        self.assertEqual(telemetry["documents_with_ocr_used"], 1)
        self.assertEqual(telemetry["documents_with_cpt_hcpcs_codes"], 1)
        self.assertEqual(telemetry["total_extracted_char_count"], 160)
        self.assertEqual(telemetry["total_prompt_char_count"], 135)
        self.assertEqual(telemetry["total_ocr_pages_attempted"], 2)
        self.assertEqual(telemetry["total_ocr_pages_with_text"], 1)
        self.assertEqual(telemetry["input_tokens"], 15)
        self.assertEqual(telemetry["output_tokens"], 4)
        self.assertEqual(telemetry["total_tokens"], 19)
        self.assertEqual(telemetry["total_processing_seconds"], 3.0)

    def test_write_json_atomic_writes_valid_json(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "document-classifications.json"
            payload = {"documents": [{"pdf_name": "example.pdf"}]}

            write_json_atomic(path, payload)

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
