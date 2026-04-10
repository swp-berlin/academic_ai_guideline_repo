#!/usr/bin/env python3
"""Download AI guidelines listed in guidelines.yaml.

This version is intentionally conservative:
- defaults to single-slug manual ingestion
- skips network access when a local file already exists
- rejects obviously unsafe URLs
- streams downloads to disk with a size cap
- uses atomic writes and basic retry logic
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT / "downloads"
GUIDELINES_FILE = ROOT / "guidelines.yaml"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ai-guidelines-collector/1.0; "
        "+https://github.com/YOURORG/ai-guidelines)"
    ),
}
TIMEOUT = 30
DELAY_BETWEEN_REQUESTS = 1.0
MAX_BYTES = 100 * 1024 * 1024  # 100 MB
ALLOWED_SCHEMES = {"http", "https"}
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".doc", ".docx"}


def load_guidelines() -> list[dict]:
    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError("guidelines.yaml must contain a YAML list")
    return data


def build_session() -> requests.Session:
    retry = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET", "HEAD"}),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)

    session = requests.Session()
    session.headers.update(HEADERS)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def find_existing_downloads(slug: str) -> list[Path]:
    return sorted(
        p for p in DOWNLOADS_DIR.glob(f"{slug}.*") if p.is_file()
    )


def validate_url(url: str) -> str | None:
    parsed = urlparse(url)

    if parsed.scheme not in ALLOWED_SCHEMES:
        return f"unsupported URL scheme: {parsed.scheme or '(missing)'}"

    if not parsed.netloc:
        return "URL is missing a host"

    if parsed.username or parsed.password:
        return "URL must not include embedded credentials"

    hostname = parsed.hostname
    if hostname is None:
        return "URL hostname could not be parsed"

    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        return "refusing localhost URL"

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        return None

    if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
        return "refusing private or non-public IP address"

    return None


def guess_extension(
    original_url: str,
    final_url: str,
    content_type: str,
    sniffed_prefix: bytes,
) -> str:
    lowered_content_type = content_type.lower()

    if "application/pdf" in lowered_content_type:
        return ".pdf"
    if "text/html" in lowered_content_type:
        return ".html"
    if "application/msword" in lowered_content_type:
        return ".doc"
    if (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        in lowered_content_type
    ):
        return ".docx"

    for candidate_url in (final_url, original_url):
        path = urlparse(candidate_url).path
        ext = Path(path).suffix.lower()
        if ext in SUPPORTED_EXTENSIONS:
            return ext

    if sniffed_prefix.startswith(b"%PDF-"):
        return ".pdf"

    return ".html"


def cleanup_other_candidates(slug: str, keep: Path) -> None:
    for candidate in find_existing_downloads(slug):
        if candidate != keep:
            candidate.unlink(missing_ok=True)


def download_entry(
    entry: dict,
    session: requests.Session,
    force: bool = False,
) -> dict:
    slug = entry["slug"]
    url = entry["url"]

    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return {"slug": slug, "status": "skipped", "reason": "not an HTTP(S) URL"}

    validation_error = validate_url(url)
    if validation_error:
        return {"slug": slug, "status": "error", "reason": validation_error}

    existing = find_existing_downloads(slug)
    if existing and not force:
        if len(existing) > 1:
            return {
                "slug": slug,
                "status": "error",
                "reason": (
                    "multiple existing download files found; "
                    "clean up downloads/ manually before retrying"
                ),
            }
        return {"slug": slug, "status": "exists", "path": str(existing[0])}

    tmp_path: Path | None = None
    try:
        with session.get(
            url,
            timeout=TIMEOUT,
            allow_redirects=True,
            stream=True,
        ) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            bytes_written = 0
            sniffed_prefix = bytearray()

            with tempfile.NamedTemporaryFile(
                delete=False,
                dir=DOWNLOADS_DIR,
                prefix=f".{slug}.",
                suffix=".part",
            ) as tmp:
                tmp_path = Path(tmp.name)
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue

                    if len(sniffed_prefix) < 16:
                        sniffed_prefix.extend(chunk[: 16 - len(sniffed_prefix)])

                    bytes_written += len(chunk)
                    if bytes_written > MAX_BYTES:
                        raise ValueError(
                            f"download exceeds size limit of {MAX_BYTES // (1024 * 1024)} MB"
                        )

                    tmp.write(chunk)

            ext = guess_extension(
                original_url=url,
                final_url=response.url,
                content_type=content_type,
                sniffed_prefix=bytes(sniffed_prefix),
            )
            destination = DOWNLOADS_DIR / f"{slug}{ext}"
            destination.parent.mkdir(parents=True, exist_ok=True)

            os.replace(tmp_path, destination)
            cleanup_other_candidates(slug, keep=destination)

            return {
                "slug": slug,
                "status": "downloaded",
                "path": str(destination),
                "size_kb": bytes_written // 1024,
                "extension": ext,
                "final_url": response.url,
                "content_type": content_type,
            }
    except (requests.RequestException, ValueError, OSError) as exc:
        if tmp_path and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        return {"slug": slug, "status": "error", "reason": str(exc)}


def select_guidelines(
    guidelines: list[dict],
    slug: str | None,
    all_entries: bool,
) -> list[dict]:
    if slug:
        selected = [entry for entry in guidelines if entry.get("slug") == slug]
        if not selected:
            raise ValueError(f"no entry found for slug: {slug}")
        return selected

    if all_entries:
        return guidelines

    raise ValueError("refusing bulk download by default; pass --slug <slug> or --all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download AI guidelines")
    parser.add_argument("--slug", help="Download only a specific slug")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all entries in guidelines.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if a local file already exists",
    )
    args = parser.parse_args()

    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        guidelines = select_guidelines(load_guidelines(), args.slug, args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    session = build_session()
    results = {"downloaded": 0, "exists": 0, "error": 0, "skipped": 0}
    errors: list[dict] = []

    for entry in tqdm(guidelines, desc="Downloading"):
        result = download_entry(entry, session=session, force=args.force)
        results[result["status"]] += 1

        if result["status"] == "error":
            errors.append(result)

        time.sleep(DELAY_BETWEEN_REQUESTS)

    print(f"\nResults: {results}")

    if errors:
        print("\nErrors:", file=sys.stderr)
        for error in errors:
            print(f"  {error['slug']}: {error['reason']}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())