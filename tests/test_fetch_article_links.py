import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.fetch_article_links import build_output_path
from scripts.fetch_article_links import cleanup_stale_partial_files
from scripts.fetch_article_links import documentcloud_reference_to_pdf_url
from scripts.fetch_article_links import extract_documentcloud_documents
from scripts.fetch_article_links import write_article_metadata


class DocumentCloudParsingTests(unittest.TestCase):
    def test_anchor_documentcloud_link_maps_to_pdf(self) -> None:
        html = """
        <article>
            <a
                href="https://www.documentcloud.org/documents/4353248-Blue-Cross-Blue-Shield-EOB-for-Urine-Drug-Screen.html"
                target="_blank"
                rel="noopener"
            >
                explanation of benefits
            </a>
        </article>
        """

        self.assertEqual(
            extract_documentcloud_documents(
                html,
                page_url="https://kffhealthnews.org/news/article/example/",
            ),
            [
                {
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

    def test_iframe_documentcloud_embed_maps_to_pdf(self) -> None:
        html = """
        <article>
            <iframe
                title="March 2024 BOTM (Hosted by DocumentCloud)"
                src="https://embed.documentcloud.org/documents/24475386-march-2024-botm/?embed=1"
                width="570"
                height="750"
            ></iframe>
        </article>
        """

        self.assertEqual(
            extract_documentcloud_documents(
                html,
                page_url="https://kffhealthnews.org/news/article/example/",
            ),
            [
                {
                    "pdf_url": "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf",
                    "references": [
                        {
                            "tag": "iframe",
                            "url": "https://embed.documentcloud.org/documents/24475386-march-2024-botm/?embed=1",
                        }
                    ],
                }
            ],
        )

    def test_duplicate_references_to_same_pdf_are_deduplicated(self) -> None:
        html = """
        <article>
            <a href="https://www.documentcloud.org/documents/24475386-march-2024-botm.html">
                link
            </a>
            <iframe src="https://embed.documentcloud.org/documents/24475386-march-2024-botm/?embed=1"></iframe>
        </article>
        """

        documents = extract_documentcloud_documents(
            html,
            page_url="https://kffhealthnews.org/news/article/example/",
        )

        self.assertEqual(len(documents), 1)
        self.assertEqual(
            documents[0]["pdf_url"],
            "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf",
        )
        self.assertEqual(
            documents[0]["references"],
            [
                {
                    "tag": "a",
                    "url": "https://www.documentcloud.org/documents/24475386-march-2024-botm.html",
                },
                {
                    "tag": "iframe",
                    "url": "https://embed.documentcloud.org/documents/24475386-march-2024-botm/?embed=1",
                },
            ],
        )

    def test_multiple_distinct_documentcloud_documents_are_all_returned(self) -> None:
        html = """
        <article>
            <a href="https://www.documentcloud.org/documents/111-first-document.html">first</a>
            <a href="https://www.documentcloud.org/documents/222-second-document.html">second</a>
        </article>
        """

        self.assertEqual(
            extract_documentcloud_documents(
                html,
                page_url="https://kffhealthnews.org/news/article/example/",
            ),
            [
                {
                    "pdf_url": "https://s3.documentcloud.org/documents/111/first-document.pdf",
                    "references": [
                        {
                            "tag": "a",
                            "url": "https://www.documentcloud.org/documents/111-first-document.html",
                        }
                    ],
                },
                {
                    "pdf_url": "https://s3.documentcloud.org/documents/222/second-document.pdf",
                    "references": [
                        {
                            "tag": "a",
                            "url": "https://www.documentcloud.org/documents/222-second-document.html",
                        }
                    ],
                },
            ],
        )

    def test_direct_s3_pdf_url_is_supported(self) -> None:
        self.assertEqual(
            documentcloud_reference_to_pdf_url(
                "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf?download=1"
            ),
            "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf",
        )

    def test_output_path_uses_shared_kff_bills_directory(self) -> None:
        output_path = build_output_path(
            Path("/tmp/outputs"),
            "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf",
        )

        self.assertEqual(
            output_path,
            Path("/tmp/outputs/kff-bills/24475386-march-2024-botm.pdf"),
        )

    def test_metadata_file_is_written_in_kff_bills_directory(self) -> None:
        metadata_records = [
            {
                "article_url": "https://kffhealthnews.org/news/article/example/",
                "status": "ok",
                "documents": [
                    {
                        "pdf_url": "https://s3.documentcloud.org/documents/24475386/march-2024-botm.pdf",
                        "output_path": "/tmp/outputs/kff-bills/24475386-march-2024-botm.pdf",
                        "references": [
                            {
                                "tag": "iframe",
                                "url": "https://embed.documentcloud.org/documents/24475386-march-2024-botm/?embed=1",
                            }
                        ],
                        "download_status": "downloaded",
                    }
                ],
            }
        ]

        with TemporaryDirectory() as tmpdir:
            metadata_path = write_article_metadata(Path(tmpdir), metadata_records)

            self.assertEqual(
                metadata_path,
                Path(tmpdir) / "kff-bills" / "article-document-metadata.json",
            )
            self.assertEqual(
                json.loads(metadata_path.read_text(encoding="utf-8")),
                metadata_records,
            )

    def test_cleanup_removes_stale_partials_only(self) -> None:
        with TemporaryDirectory() as tmpdir:
            documents_dir = Path(tmpdir) / "kff-bills"
            documents_dir.mkdir()
            stale_pdf = documents_dir / "24475386-march-2024-botm.pdf.part"
            stale_metadata = documents_dir / "article-document-metadata.json.part"
            completed_pdf = documents_dir / "24475386-march-2024-botm.pdf"
            stale_pdf.write_bytes(b"partial pdf")
            stale_metadata.write_text("partial metadata", encoding="utf-8")
            completed_pdf.write_bytes(b"complete pdf")

            removed_paths = cleanup_stale_partial_files(Path(tmpdir))

            self.assertEqual(sorted(removed_paths), [stale_pdf, stale_metadata])
            self.assertFalse(stale_pdf.exists())
            self.assertFalse(stale_metadata.exists())
            self.assertTrue(completed_pdf.exists())


if __name__ == "__main__":
    unittest.main()
