#!/usr/bin/env python3
"""Extract text from downloaded guideline files.

This version is intentionally conservative:
- defaults to single-slug manual extraction
- refuses ambiguous multiple source files for one slug
- fails when extraction yields implausibly little text
- handles two-column PDF layouts via block-position analysis
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT / "downloads"
TEXTS_DIR = ROOT / "texts"
GUIDELINES_FILE = ROOT / "guidelines.yaml"
SUPPORTED_INPUT_EXTENSIONS = {".pdf", ".html", ".htm"}
MIN_TEXT_CHARS = 200


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
        names = ", ".join(candidate.name for candidate in candidates)
        raise RuntimeError(
            f"multiple downloaded files found for '{slug}': {names}; clean up downloads/ first"
        )

    return candidates[0]


# ---------------------------------------------------------------------------
# PDF extraction: column-aware
# ---------------------------------------------------------------------------

def _detect_column_split(text_blocks: list, page_width: float) -> float | None:
    """Detect whether the page has a two-column layout by clustering x0 positions.

    Returns the x-coordinate midpoint between columns, or None for single-column.
    Unlike a width-based heuristic, this correctly handles single-column pages
    where lines happen to vary in width (e.g. due to footnotes or short lines).
    """
    from collections import Counter

    if not text_blocks:
        return None

    # Bucket left-edge positions into ~30px bins
    bin_width = 30
    x0_bins: Counter[int] = Counter()
    for b in text_blocks:
        binned = round(b[0] / bin_width) * bin_width
        x0_bins[binned] += 1

    # Keep only bins with >= 2 blocks (ignore stray page numbers etc.)
    significant = {k: v for k, v in x0_bins.items() if v >= 2}
    if len(significant) <= 1:
        return None

    # Check if any two adjacent bin clusters are separated by > 25% of page width
    sorted_bins = sorted(significant.keys())
    for i in range(len(sorted_bins) - 1):
        gap = sorted_bins[i + 1] - sorted_bins[i]
        if gap > page_width * 0.25:
            return sorted_bins[i] + gap / 2

    return None  # bins are close together: single column with varying widths


def _extract_page_text(page) -> str:
    """Extract text from a single PDF page, handling multi-column layouts.

    Strategy:
    1. Get text blocks with bounding-box positions.
    2. Cluster left-edge (x0) positions to decide single vs. multi-column.
    3. Single-column: sort all blocks by y (natural reading order).
    4. Two-column: read left column top-to-bottom, then right column.
    """
    blocks = page.get_text("blocks")
    text_blocks = [b for b in blocks if b[6] == 0]  # type 0 = text

    if not text_blocks:
        return ""

    page_width = page.rect.width
    col_split = _detect_column_split(text_blocks, page_width)

    if col_split is None:
        # Single column: sort by vertical position
        text_blocks.sort(key=lambda b: b[1])
        return "\n".join(b[4].strip() for b in text_blocks)

    # Two-column: partition at the detected split point
    left = [(b[1], b[4]) for b in text_blocks if b[0] < col_split]
    right = [(b[1], b[4]) for b in text_blocks if b[0] >= col_split]

    left.sort(key=lambda x: x[0])
    right.sort(key=lambda x: x[0])

    parts = [t.strip() for _, t in left] + [t.strip() for _, t in right]
    return "\n".join(parts)


def extract_pdf(path: Path) -> str:
    """Extract text from a PDF using pymupdf with column-aware layout."""
    import fitz  # pymupdf

    doc = fitz.open(path)
    pages: list[str] = []
    try:
        for page in doc:
            text = _extract_page_text(page)
            if text.strip():
                pages.append(text.strip())
    finally:
        doc.close()

    return "\n\n---\n\n".join(pages)


# ---------------------------------------------------------------------------
# HTML extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def extract_file(download_path: Path) -> str | None:
    ext = download_path.suffix.lower()
    if ext == ".pdf":
        return extract_pdf(download_path)
    if ext in {".html", ".htm"}:
        return extract_html(download_path)
    return None


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

    raise ValueError("refusing bulk extraction by default; pass --slug <slug> or --all")


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from downloads")
    parser.add_argument("--slug", help="Extract only a specific slug")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Extract text for all entries in guidelines.yaml",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a text file already exists",
    )
    args = parser.parse_args()

    TEXTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        guidelines = select_guidelines(load_guidelines(), args.slug, args.all)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    results = {"extracted": 0, "exists": 0, "missing": 0, "error": 0}

    for entry in tqdm(guidelines, desc="Extracting text"):
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

            text = extract_file(download_path)
            if text is None:
                raise RuntimeError(f"unsupported file type: {download_path.suffix.lower()}")

            if len(text.strip()) < MIN_TEXT_CHARS:
                raise RuntimeError(
                    "extracted text is suspiciously short; review the downloaded file manually"
                )

            header = (
                f"# {entry['institution']}\n"
                f"# Source: {entry['url']}\n"
                f"# Date: {entry.get('date', 'unknown')}\n"
                f"# Category: {entry['category']}\n"
                f"# Download file: {download_path.name}\n\n"
            )
            txt_path.write_text(header + text + "\n", encoding="utf-8")
            results["extracted"] += 1
        except Exception as exc:
            print(f"  Error extracting {slug}: {exc}", file=sys.stderr)
            results["error"] += 1

    print(f"\nResults: {results}")
    return 1 if results["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())