#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import count
from pathlib import Path
from urllib.parse import quote, unquote, urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from claim.cli_logging import CliLogger

ARCHIVE_URL = "https://kffhealthnews.org/news/tag/bill-of-the-month/"
ARCHIVE_PAGE_URL = "https://kffhealthnews.org/news/tag/bill-of-the-month/page/{page}/"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_KFF_DELAY_SECONDS = 1.5
DEFAULT_DOCUMENTCLOUD_DELAY_SECONDS = 0.5
DEFAULT_TIMEOUT_SECONDS = 30
DOCUMENT_OUTPUT_DIRNAME = "kff-bills"
METADATA_FILENAME = "article-document-metadata.json"
USER_AGENT = "medical-billing-eval/0.1 (polite sequential research script)"
DOCUMENTCLOUD_HOSTS = {
    "documentcloud.org",
    "www.documentcloud.org",
    "embed.documentcloud.org",
}


class RequestLimiter:
    def __init__(self, delay_seconds: float) -> None:
        self.delay_seconds = max(0.0, delay_seconds)
        self._next_allowed_at = 0.0

    def wait(self) -> None:
        if self.delay_seconds <= 0:
            return

        now = time.monotonic()
        if now < self._next_allowed_at:
            time.sleep(self._next_allowed_at - now)
        self._next_allowed_at = time.monotonic() + self.delay_seconds


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request(
    session: requests.Session,
    url: str,
    limiter: RequestLimiter,
    *,
    stream: bool = False,
) -> requests.Response:
    limiter.wait()
    return session.get(url, timeout=DEFAULT_TIMEOUT_SECONDS, stream=stream)


def fetch_optional_html(
    session: requests.Session,
    url: str,
    limiter: RequestLimiter,
) -> tuple[int, str]:
    response = request(session, url, limiter)
    try:
        return response.status_code, response.text
    finally:
        response.close()


def fetch_html(
    session: requests.Session,
    url: str,
    limiter: RequestLimiter,
) -> str:
    response = request(session, url, limiter)
    try:
        response.raise_for_status()
        return response.text
    finally:
        response.close()


def get_kff_bill_of_the_month_links(
    session: requests.Session,
    limiter: RequestLimiter,
    logger: CliLogger,
) -> list[str]:
    all_links: set[str] = set()

    for page in count(1):
        url = ARCHIVE_URL if page == 1 else ARCHIVE_PAGE_URL.format(page=page)
        logger.info("Fetching archive page", page=page, url=url)
        status_code, html = fetch_optional_html(session, url, limiter)

        if status_code == 404:
            break
        if status_code != 200:
            raise requests.HTTPError(
                f"Unexpected status {status_code} while fetching archive page {page}: {url}"
            )

        page_links = extract_archive_article_links(html, base_url=url)
        if not page_links:
            break

        all_links.update(page_links)

    return sorted(all_links)


def extract_archive_article_links(html: str, *, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: set[str] = set()

    for article in soup.find_all("article"):
        for tag in article.find_all("a", href=True):
            href = urljoin(base_url, tag["href"])
            parsed = urlparse(href)
            if parsed.netloc.endswith("kffhealthnews.org") and "/news/" in parsed.path:
                links.add(href)
                break

    return sorted(links)


def documentcloud_reference_to_pdf_url(url: str) -> str | None:
    parsed = urlparse(url)
    host = parsed.netloc.lower()

    if host == "s3.documentcloud.org":
        if parsed.path.startswith("/documents/") and parsed.path.lower().endswith(".pdf"):
            return parsed._replace(query="", fragment="").geturl()
        return None

    if host not in DOCUMENTCLOUD_HOSTS:
        return None

    path = parsed.path.rstrip("/")
    if not path.startswith("/documents/"):
        return None

    document_part = path.removeprefix("/documents/")
    if document_part.lower().endswith(".html"):
        document_part = document_part[:-5]

    if not document_part:
        return None

    if "-" in document_part:
        document_id, slug = document_part.split("-", 1)
    else:
        document_id, slug = document_part, document_part

    if not document_id.isdigit():
        return None

    normalized_slug = quote(unquote(slug), safe="-._~")
    return f"https://s3.documentcloud.org/documents/{document_id}/{normalized_slug}.pdf"


def extract_documentcloud_documents(html: str, *, page_url: str) -> list[dict[str, object]]:
    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("article") or soup
    documents: dict[str, dict[str, object]] = {}

    for tag in root.select("a[href], iframe[src]"):
        attribute = "href" if tag.name == "a" else "src"
        raw_url = tag.get(attribute)
        if not raw_url:
            continue

        absolute_url = urljoin(page_url, raw_url)
        pdf_url = documentcloud_reference_to_pdf_url(absolute_url)
        if pdf_url:
            document = documents.setdefault(
                pdf_url,
                {
                    "pdf_url": pdf_url,
                    "references": [],
                },
            )
            reference = {
                "tag": tag.name,
                "url": absolute_url,
            }
            references = document["references"]
            if reference not in references:
                references.append(reference)

    return list(documents.values())


def build_output_path(output_dir: Path, pdf_url: str) -> Path:
    parsed = urlparse(pdf_url)
    path_parts = [part for part in parsed.path.split("/") if part]

    if len(path_parts) >= 3 and path_parts[0] == "documents":
        document_id = path_parts[1]
        filename = path_parts[2]
        return output_dir / DOCUMENT_OUTPUT_DIRNAME / f"{document_id}-{filename}"

    return output_dir / DOCUMENT_OUTPUT_DIRNAME / Path(path_parts[-1]).name


def metadata_path(output_dir: Path) -> Path:
    return output_dir / DOCUMENT_OUTPUT_DIRNAME / METADATA_FILENAME


def cleanup_stale_partial_files(output_dir: Path) -> list[Path]:
    documents_dir = output_dir / DOCUMENT_OUTPUT_DIRNAME
    removed_paths: list[Path] = []

    if not documents_dir.exists():
        return removed_paths

    for path in sorted(documents_dir.glob("*.part")):
        if path.is_file():
            path.unlink()
            removed_paths.append(path)

    return removed_paths


def download_pdf(
    session: requests.Session,
    pdf_url: str,
    output_path: Path,
    limiter: RequestLimiter,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.part")

    response = request(session, pdf_url, limiter, stream=True)
    try:
        response.raise_for_status()
        with temp_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    handle.write(chunk)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise
    finally:
        response.close()

    temp_path.replace(output_path)


def write_article_metadata(output_dir: Path, metadata_records: list[dict[str, object]]) -> Path:
    output_path = metadata_path(output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(f"{output_path.suffix}.part")

    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata_records, handle, indent=2)
        handle.write("\n")

    temp_path.replace(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Bill of the Month PDFs linked from KFF article pages."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Base output directory. PDFs are written under "
            f"{DOCUMENT_OUTPUT_DIRNAME}/ inside this directory."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N article pages after discovery.",
    )
    parser.add_argument(
        "--kff-delay-seconds",
        type=float,
        default=DEFAULT_KFF_DELAY_SECONDS,
        help="Minimum delay between requests to KFF pages.",
    )
    parser.add_argument(
        "--documentcloud-delay-seconds",
        type=float,
        default=DEFAULT_DOCUMENTCLOUD_DELAY_SECONDS,
        help="Minimum delay between PDF downloads from DocumentCloud.",
    )
    return parser.parse_args()


def main() -> int:
    logger = CliLogger("fetch")
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.joinpath(DOCUMENT_OUTPUT_DIRNAME).mkdir(parents=True, exist_ok=True)
    removed_partials = cleanup_stale_partial_files(output_dir)
    kff_limiter = RequestLimiter(args.kff_delay_seconds)
    documentcloud_limiter = RequestLimiter(args.documentcloud_delay_seconds)

    downloaded = 0
    skipped = 0
    articles_without_documents = 0
    article_failures = 0
    document_failures = 0
    discovered_documents = 0
    unique_documents: set[str] = set()
    article_metadata: list[dict[str, object]] = []

    with build_session() as session:
        article_urls = get_kff_bill_of_the_month_links(session, kff_limiter, logger)
        if args.limit is not None:
            article_urls = article_urls[: args.limit]

        logger.info(
            "Starting fetch run",
            articles=len(article_urls),
            output_dir=output_dir / DOCUMENT_OUTPUT_DIRNAME,
            kff_delay=args.kff_delay_seconds,
            documentcloud_delay=args.documentcloud_delay_seconds,
        )
        if removed_partials:
            logger.warning("Removed stale partial files", count=len(removed_partials))

        for index, article_url in enumerate(article_urls, start=1):
            logger.info(
                "Fetching article",
                progress=f"{index}/{len(article_urls)}",
                article_url=article_url,
            )
            article_record: dict[str, object] = {
                "article_url": article_url,
                "status": "pending",
                "documents": [],
            }

            try:
                article_html = fetch_html(session, article_url, kff_limiter)
                documents = extract_documentcloud_documents(article_html, page_url=article_url)
                if not documents:
                    logger.warning("No DocumentCloud documents found", article_url=article_url)
                    article_record["status"] = "no_documents"
                    articles_without_documents += 1
                    article_metadata.append(article_record)
                    continue

                discovered_documents += len(documents)
                logger.info(
                    "Found DocumentCloud documents",
                    article_url=article_url,
                    documents=len(documents),
                )

                article_status = "ok"
                document_records: list[dict[str, object]] = []

                for document in documents:
                    pdf_url = str(document["pdf_url"])
                    unique_documents.add(pdf_url)
                    output_path = build_output_path(output_dir, pdf_url)
                    document_record = {
                        "pdf_url": pdf_url,
                        "output_path": str(output_path),
                        "references": document["references"],
                    }

                    if output_path.exists():
                        logger.info("Skipping existing PDF", pdf=output_path.name)
                        document_record["download_status"] = "skipped_existing"
                        skipped += 1
                        document_records.append(document_record)
                        continue

                    logger.info("Downloading PDF", pdf=output_path.name, pdf_url=pdf_url)
                    try:
                        download_pdf(session, pdf_url, output_path, documentcloud_limiter)
                        document_record["download_status"] = "downloaded"
                        downloaded += 1
                        logger.success("Downloaded PDF", pdf=output_path.name)
                    except Exception as exc:
                        logger.error("Failed to download PDF", pdf=output_path.name, error=exc)
                        document_record["download_status"] = "failed"
                        document_record["error"] = str(exc)
                        document_failures += 1
                        article_status = "partial_failure"

                    document_records.append(document_record)

                article_record["status"] = article_status
                article_record["documents"] = document_records
            except Exception as exc:
                logger.error("Failed to process article", article_url=article_url, error=exc)
                article_record["status"] = "failed"
                article_record["error"] = str(exc)
                article_failures += 1

            article_metadata.append(article_record)

    metadata_output_path = write_article_metadata(output_dir, article_metadata)
    logger.success(
        "Completed fetch run",
        metadata=metadata_output_path,
        articles=len(article_metadata),
        document_refs=discovered_documents,
        unique_documents=len(unique_documents),
        downloaded=downloaded,
        skipped=skipped,
        no_documents=articles_without_documents,
        article_failures=article_failures,
        document_failures=document_failures,
    )
    return 1 if article_failures or document_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
