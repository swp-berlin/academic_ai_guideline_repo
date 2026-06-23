#!/usr/bin/env python3
"""Validate guidelines.yaml schema and data integrity.

Run this in CI to catch malformed entries before merging PRs.
Warnings are used for brittle-but-sometimes-necessary URLs that should
usually be handled manually rather than via fully automatic CI.
"""

from __future__ import annotations

import ipaddress
import re
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import yaml

GUIDELINES_FILE = Path(__file__).resolve().parent.parent / "guidelines.yaml"

REQUIRED_FIELDS = {"institution", "url", "category", "slug"}
VALID_CATEGORIES = {"guideline", "template"}
SLUG_PATTERN = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
DATE_PATTERN = re.compile(r"^\d{4}(?:-\d{2}(?:-\d{2})?)?$")
LANGUAGE_PATTERN = re.compile(r"^[a-z]{2}$")
SUSPICIOUS_URL_MARKERS = {
    "token",
    "sig",
    "signature",
    "expires",
    "expiry",
    "exp",
    "hash",
    "jwt",
}


def validate_url(url: str, prefix: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        errors.append(f"{prefix}: url must start with http:// or https://")
        return errors, warnings

    if not parsed.netloc:
        errors.append(f"{prefix}: url is missing a hostname")
        return errors, warnings

    if parsed.username or parsed.password:
        errors.append(f"{prefix}: url must not contain embedded credentials")

    hostname = parsed.hostname
    if hostname is None:
        errors.append(f"{prefix}: could not parse hostname")
        return errors, warnings

    lowered = hostname.lower()
    if lowered in {"localhost", "127.0.0.1", "::1"}:
        errors.append(f"{prefix}: localhost URLs are not allowed")

    try:
        ip = ipaddress.ip_address(hostname)
    except ValueError:
        ip = None

    if ip and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved):
        errors.append(f"{prefix}: private or non-public IP addresses are not allowed")

    if len(url) > 300:
        warnings.append(
            f"{prefix}: very long URL; treat as brittle and prefer manual download"
        )

    if "/securedl/" in parsed.path or "/sdl-" in parsed.path:
        warnings.append(
            f"{prefix}: signed or expiring-looking download URL; prefer manual download"
        )

    if parsed.query:
        query_keys = {key.lower() for key in parse_qs(parsed.query).keys()}
        if query_keys & SUSPICIOUS_URL_MARKERS:
            warnings.append(
                f"{prefix}: query string looks tokenized/signed; prefer manual download"
            )

    return errors, warnings


def validate_entry(entry: dict, idx: int) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    prefix = f"Entry {idx} ({entry.get('institution', '???')})"

    for field in REQUIRED_FIELDS:
        if field not in entry or entry[field] in (None, ""):
            errors.append(f"{prefix}: missing required field '{field}'")

    slug = entry.get("slug", "")
    if slug and not SLUG_PATTERN.match(slug):
        errors.append(
            f"{prefix}: slug '{slug}' must be lowercase alphanumeric with hyphens"
        )

    category = entry.get("category", "")
    if category and category not in VALID_CATEGORIES:
        errors.append(
            f"{prefix}: category '{category}' must be one of {sorted(VALID_CATEGORIES)}"
        )

    date = entry.get("date")
    if date is not None:
        date_str = str(date)
        if not DATE_PATTERN.match(date_str):
            errors.append(
                f"{prefix}: date '{date_str}' should be YYYY, YYYY-MM, or YYYY-MM-DD"
            )

    language = entry.get("language")
    if language is not None and language != "" and not LANGUAGE_PATTERN.match(str(language)):
        errors.append(
            f"{prefix}: language '{language}' must be a 2-letter ISO 639-1 code"
        )

    url = entry.get("url", "")
    if url:
        url_errors, url_warnings = validate_url(url, prefix)
        errors.extend(url_errors)
        warnings.extend(url_warnings)

    return errors, warnings


def main() -> int:
    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        guidelines = yaml.safe_load(f)

    if not isinstance(guidelines, list):
        print("ERROR: guidelines.yaml must be a YAML list", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    all_warnings: list[str] = []
    slugs_seen: set[tuple[str, str]] = set()

    for idx, entry in enumerate(guidelines, 1):
        if not isinstance(entry, dict):
            all_errors.append(f"Entry {idx}: each list item must be a mapping/object")
            continue

        errors, warnings = validate_entry(entry, idx)
        all_errors.extend(errors)
        all_warnings.extend(warnings)

        slug = entry.get("slug", "")
        if slug:
            key = (slug, str(entry.get("version") or "1_0"))
            if key in slugs_seen:
                all_errors.append(
                    f"Entry {idx}: duplicate slug+version '{slug}' @ {key[1]}"
                )
            slugs_seen.add(key)

    if all_errors:
        print(f"Validation failed with {len(all_errors)} error(s):\n", file=sys.stderr)
        for error in all_errors:
            print(f"  ✗ {error}", file=sys.stderr)

        if all_warnings:
            print(f"\nWarnings ({len(all_warnings)}):\n", file=sys.stderr)
            for warning in all_warnings:
                print(f"  ! {warning}", file=sys.stderr)
        return 1

    print(f"✓ All {len(guidelines)} entries valid")
    if all_warnings:
        print(f"\nWarnings ({len(all_warnings)}):")
        for warning in all_warnings:
            print(f"  ! {warning}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())