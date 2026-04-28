#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "python-dotenv", "pyyaml", "tqdm", "beautifulsoup4", "pymupdf"]
# ///
"""Extract text from downloaded guideline files using Mistral OCR for PDFs.

This version replaces the brittle SDK signed-URL flow with direct HTTP calls:
    upload file -> POST /v1/files (purpose=ocr)
    OCR file    -> POST /v1/ocr with document.file_id

That removes an extra failure point and tends to behave better on awkward PDFs.

Usage:
    uv run scripts/extract_text_ocr.py --slug helmholtz-gemeinschaft
    uv run scripts/extract_text_ocr.py --all
    uv run scripts/extract_text_ocr.py --all --force

Reads MISTRAL_API_KEY from .env or environment.
Also accepts lowercase mistral_api_key for convenience.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT / "downloads"
TEXTS_DIR = ROOT / "texts"
GUIDELINES_FILE = ROOT / "guidelines.yaml"
SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".html", ".htm"}
MIN_TEXT_CHARS = 200
API_BASE = os.getenv("MISTRAL_API_BASE", "https://api.mistral.ai/v1").rstrip("/")
UPLOAD_TIMEOUT = (30, 600)  # connect, read
OCR_TIMEOUT = (30, 900)     # connect, read


class MistralAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def load_guidelines() -> list[dict]:
    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("guidelines.yaml must contain a YAML list")
    return data



def find_downloaded_file(slug: str) -> Path | None:
    candidates = sorted(
        p
        for p in DOWNLOADS_DIR.glob(f"{slug}.*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_INPUT_EXTENSIONS
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise RuntimeError(
            f"multiple downloaded files for '{slug}': {names}; clean up downloads/ first"
        )
    return candidates[0]


# -- Mistral OCR extraction -------------------------------------------------


def get_api_key() -> str:
    return (
        os.getenv("MISTRAL_API_KEY")
        or ""
    ).strip()



def make_session(api_key: str) -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "extract_text_ocr_fixed.py",
        }
    )
    return session



def is_transient_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 425, 429, 500, 502, 503, 504}



def should_retry_exception(exc: Exception) -> bool:
    if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
        return True
    status_code = getattr(exc, "status_code", None)
    return is_transient_status(status_code)



def backoff_seconds(attempt: int) -> float:
    base = min(5 * (2 ** attempt), 60)
    return base



def response_to_error(resp: requests.Response, context: str) -> MistralAPIError:
    text = (resp.text or "").strip()
    if len(text) > 600:
        text = text[:600] + " ..."
    message = f"{context} failed with HTTP {resp.status_code}"
    if text:
        message += f": {text}"
    return MistralAPIError(message, status_code=resp.status_code)



def upload_file_for_ocr(session: requests.Session, path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    with open(path, "rb") as fh:
        resp = session.post(
            f"{API_BASE}/files",
            data={"purpose": "ocr"},
            files={"file": (path.name, fh, mime_type)},
            timeout=UPLOAD_TIMEOUT,
        )
    if not resp.ok:
        raise response_to_error(resp, f"uploading {path.name}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise MistralAPIError(f"uploading {path.name} returned non-JSON") from exc

    file_id = payload.get("id")
    if not file_id:
        raise MistralAPIError(f"uploading {path.name} returned no file id: {payload}")
    return str(file_id)



def run_ocr_on_file_id(session: requests.Session, file_id: str) -> dict[str, Any]:
    resp = session.post(
        f"{API_BASE}/ocr",
        json={
            "model": "mistral-ocr-latest",
            "document": {"file_id": file_id},
        },
        timeout=OCR_TIMEOUT,
    )
    if not resp.ok:
        raise response_to_error(resp, f"OCR for file_id {file_id}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise MistralAPIError("OCR returned non-JSON") from exc
    if not isinstance(payload, dict):
        raise MistralAPIError(f"OCR returned unexpected payload type: {type(payload).__name__}")
    return payload



def pages_to_markdown(payload: dict[str, Any]) -> str:
    raw_pages = payload.get("pages")
    if not isinstance(raw_pages, list):
        raise MistralAPIError(f"OCR response missing pages list: {payload}")

    pages: list[str] = []
    for page in raw_pages:
        if not isinstance(page, dict):
            continue
        md = (page.get("markdown") or "").strip()
        if md:
            pages.append(md)

    if not pages:
        raise MistralAPIError("OCR returned no page markdown")

    return "\n\n---\n\n".join(pages)



def extract_pdf_ocr(path: Path, api_key: str, max_retries: int = 4) -> str:
    """Extract text from PDF using Mistral OCR API via direct HTTP.
    Retries transient HTTP/network failures with exponential backoff.
    """
    last_err: Exception | None = None

    with make_session(api_key) as session:
        for attempt in range(max_retries):
            try:
                file_id = upload_file_for_ocr(session, path)
                payload = run_ocr_on_file_id(session, file_id)
                return pages_to_markdown(payload)
            except Exception as exc:
                last_err = exc
                if should_retry_exception(exc) and attempt < max_retries - 1:
                    wait = backoff_seconds(attempt)
                    print(
                        f"    OCR retry {attempt + 1}/{max_retries} for {path.name} after {type(exc).__name__}: {wait:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(wait)
                    continue
                raise

    raise RuntimeError(f"OCR failed after {max_retries} retries: {last_err}")



def extract_pdf_pymupdf(path: Path) -> str:
    """Fallback: extract text from PDF using pymupdf (no API needed)."""
    import fitz

    doc = fitz.open(path)
    pages: list[str] = []
    try:
        for page in doc:
            text = page.get_text().strip()
            if text:
                pages.append(text)
    finally:
        doc.close()

    return "\n\n---\n\n".join(pages)


# -- HTML extraction (unchanged) -------------------------------------------


def extract_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    html = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    main = soup.find("main") or soup.find("article") or soup.find("body") or soup
    text = main.get_text(separator="\n", strip=True)

    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


# -- Dispatcher -------------------------------------------------------------


def extract_file(download_path: Path, api_key: str) -> tuple[str | None, bool]:
    """Returns (text, used_fallback)."""
    ext = download_path.suffix.lower()
    if ext == ".pdf":
        try:
            return extract_pdf_ocr(download_path, api_key), False
        except Exception as exc:
            print(f"    OCR failed ({exc}), falling back to pymupdf", file=sys.stderr)
            return extract_pdf_pymupdf(download_path), True
    if ext in {".html", ".htm"}:
        return extract_html(download_path), False
    return None, False



def select_guidelines(
    guidelines: list[dict],
    slug: str | None,
    all_entries: bool,
) -> list[dict]:
    if slug:
        selected = [e for e in guidelines if e.get("slug") == slug]
        if not selected:
            raise ValueError(f"no entry found for slug: {slug}")
        return selected
    if all_entries:
        return guidelines
    raise ValueError("pass --slug <slug> or --all")



def main() -> int:
    # Load from repo root explicitly so running from subdirs still picks up .env.
    load_dotenv(ROOT / ".env")
    load_dotenv()

    parser = argparse.ArgumentParser(description="Extract text using Mistral OCR")
    parser.add_argument("--slug", help="Extract only a specific slug")
    parser.add_argument("--all", action="store_true", help="Extract all entries")
    parser.add_argument("--force", action="store_true", help="Re-extract existing")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("MISTRAL_API_KEY not found (set in .env or env)", file=sys.stderr)
        return 1

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        guidelines = select_guidelines(load_guidelines(), args.slug, args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results = {"extracted": 0, "ocr_fallback": 0, "exists": 0, "missing": 0, "error": 0}
    fallback_slugs: list[str] = []
    sleep_s = float(os.getenv("REQUEST_SLEEP_SECONDS", "1.0"))

    for entry in tqdm(guidelines, desc="Extracting"):
        slug = entry["slug"]
        txt_path = TEXTS_DIR / f"{slug}.txt"

        if txt_path.exists() and not args.force:
            results["exists"] += 1
            continue

        try:
            download_path = find_downloaded_file(slug)
            if download_path is None:
                results["missing"] += 1
                continue

            text, used_fallback = extract_file(download_path, api_key)
            if text is None:
                raise RuntimeError(f"unsupported file type: {download_path.suffix}")

            if len(text.strip()) < MIN_TEXT_CHARS:
                raise RuntimeError("extracted text suspiciously short")

            header = (
                f"# {entry['institution']}\n"
                f"# Source: {entry['url']}\n"
                f"# Date: {entry.get('date', 'unknown')}\n"
                f"# Category: {entry['category']}\n"
                f"# Download file: {download_path.name}\n"
                f"# Extraction: {'pymupdf-fallback' if used_fallback else 'mistral-ocr'}\n\n"
            )
            txt_path.write_text(header + text + "\n", encoding="utf-8")
            results["extracted"] += 1
            if used_fallback:
                results["ocr_fallback"] += 1
                fallback_slugs.append(slug)

            if sleep_s > 0:
                time.sleep(sleep_s)

        except Exception as exc:
            print(f"  Error extracting {slug}: {exc}", file=sys.stderr)
            results["error"] += 1

    print(f"\nResults: {results}")
    if fallback_slugs:
        print(f"Fell back to pymupdf: {', '.join(fallback_slugs)}")
    return 1 if results["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
