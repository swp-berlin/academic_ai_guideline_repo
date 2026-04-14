#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv", "pyyaml", "tqdm"]
# ///
"""Clean extracted guideline texts using an LLM.

Sends each .txt file to the LLM for cleanup (removing PDF artifacts,
page numbers, footnote markers, navigation text, etc.) while preserving
every substantive word. Writes cleaned versions alongside originals.

Usage:
    uv run scripts/clean_texts.py --slug helmholtz-gemeinschaft
    uv run scripts/clean_texts.py --all
    uv run scripts/clean_texts.py --all --force   # re-clean everything

Reads MISTRAL_API_KEY from .env.
Outputs: texts_clean/{slug}.txt  (cleaned version)
         texts_clean/{slug}.diff.html  (side-by-side diff for review)
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXTS_DIR = PROJECT_ROOT / "texts"
CLEAN_DIR = PROJECT_ROOT / "texts_clean"

# ── cleanup prompt ────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are a careful text editor. Return only the cleaned text, no commentary."

CLEANUP_PROMPT = """
Clean up the following extracted document text. This text was extracted from a PDF
or HTML page and contains various artifacts that need to be removed or fixed.

RULES — follow these exactly:
1. REMOVE these artifacts:
   - Page numbers (standalone numbers like "3", "12", or "Page 3 of 10")
   - Page headers/footers that repeat across pages (e.g. "Stand 01.05.2025", institution name repeated as running header)
   - Table of contents with dotted lines and page numbers (e.g. "1. Introduction ....... 3")
   - Navigation text ("Return to table of contents", "opens a new window", "Download file:")
   - Footnote reference numbers in the body text (superscript numbers like ¹ ² ³ or [1] [2])
   - Section dividers (lines of dashes "---" or similar)
   - Boilerplate (cookie notices, reCAPTCHA text, newsletter signup forms)
   - Markdown image placeholders (![img-0.jpeg](img-0.jpeg))
   - URL-only lines (bare URLs on their own line that aren't part of a sentence)

2. FIX these formatting issues:
   - Broken hyphens from PDF line wrapping: "ver-\nwendet" → "verwendet"
   - Broken words across lines: merge them back together
   - Multiple blank lines → single blank line
   - Inconsistent bullet markers → normalize to "- "

3. KEEP the metadata header lines starting with # exactly as they are (# Institution, # Source, etc.)

4. PRESERVE footnote body text if it contains substantive definitions or explanations.
   Move it to a "FOOTNOTES" section at the end, clearly labeled.

5. DO NOT change, rephrase, summarize, or paraphrase ANY substantive content.
   Every sentence from the original that carries meaning must appear word-for-word
   in your output. You may only fix obvious OCR errors (e.g. "tbe" → "the") if
   you are very confident.

6. PRESERVE section headings and their numbering (1., 2., a), b), I., II., etc.)

7. Output ONLY the cleaned text. No commentary, no explanations, no markdown fences.

--- ORIGINAL TEXT ---
{text}
--- END ORIGINAL TEXT ---
""".strip()

# ── API call ──────────────────────────────────────────────────────────

def call_llm(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    max_retries: int = 2,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": CLEANUP_PROMPT.format(text=text)},
        ],
        "temperature": 0.0,
        "max_tokens": 16_384,
    }

    url = f"{base_url}/chat/completions"

    for attempt in range(max_retries + 1):
        try:
            r = client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            if isinstance(content, list):
                content = "\n".join(
                    item.get("text", "") for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            return content.strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                wait = 2 * (attempt + 1)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise

    raise RuntimeError("Exhausted retries")


# ── chunking for large documents ──────────────────────────────────────

MAX_CHUNK_CHARS = 64_000  # ~3k tokens, well within context window


def chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Split text into chunks at paragraph boundaries (double newlines).

    Each chunk stays under max_chars. If a single paragraph exceeds
    max_chars, it gets its own chunk (unavoidable).
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = re.split(r"(\n\n+)", text)
    # paragraphs is now alternating [content, separator, content, separator, ...]

    chunks: list[str] = []
    current = ""

    for part in paragraphs:
        if len(current) + len(part) > max_chars and current.strip():
            chunks.append(current)
            current = part
        else:
            current += part

    if current.strip():
        chunks.append(current)

    return chunks


def clean_document(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    text: str,
    sleep_s: float = 1.0,
) -> str:
    """Clean a document, chunking if necessary."""
    chunks = chunk_text(text)

    if len(chunks) == 1:
        return call_llm(client, base_url, api_key, model, text)

    cleaned_parts: list[str] = []
    for i, chunk in enumerate(chunks):
        cleaned = call_llm(client, base_url, api_key, model, chunk)
        # Strip markdown fences per chunk
        cleaned = re.sub(r"^```\w*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned_parts.append(cleaned)
        if sleep_s > 0 and i < len(chunks) - 1:
            time.sleep(sleep_s)

    return "\n\n".join(cleaned_parts)


# ── diff generation ───────────────────────────────────────────────────

_DIFF_CSS_OVERRIDE = """
<style>
  body { background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 12px; }
  table.diff { font-family: "JetBrains Mono", "Fira Code", monospace; font-size: 12px;
    border-collapse: collapse; width: 100%; background: #1e1e2e; }
  .diff_header { background-color: #2a2a3e; color: #8899aa; }
  td.diff_header { text-align: right; padding: 0 6px; }
  .diff_next { background-color: #2a2a3e; }
  .diff_add { background-color: rgba(80, 250, 123, 0.12); }
  .diff_chg { background-color: rgba(255, 184, 108, 0.12); }
  .diff_sub { background-color: rgba(255, 85, 85, 0.12); }
  td { padding: 1px 6px; white-space: pre-wrap; word-break: break-word;
    border-bottom: 1px solid #27273a; vertical-align: top; }
  span.diff_add { background: rgba(80, 250, 123, 0.3); }
  span.diff_chg { background: rgba(255, 184, 108, 0.3); }
  span.diff_sub { background: rgba(255, 85, 85, 0.3); }
  a { color: #0ea5e9; }
  .slug-header { font-size: 14px; font-weight: 500; color: #e94560;
    margin-bottom: 8px; padding: 8px 0; border-bottom: 1px solid #2a2a3e; }
</style>
"""


def generate_diff_html(slug: str, original: str, cleaned: str) -> str:
    d = difflib.HtmlDiff(wrapcolumn=90)
    html_str = d.make_file(
        original.splitlines(),
        cleaned.splitlines(),
        fromdesc="Original (extracted)",
        todesc="Cleaned (LLM)",
        context=True,
        numlines=3,
    )
    # Inject dark-mode CSS and slug header after <head>
    html_str = html_str.replace(
        "</head>",
        _DIFF_CSS_OVERRIDE + "</head>",
    )
    html_str = html_str.replace(
        "<body>",
        f'<body>\n<div class="slug-header">{slug}</div>',
    )
    return html_str


# ── main ──────────────────────────────────────────────────────────────

def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Clean extracted texts using LLM")
    parser.add_argument("--slug", help="Clean only a specific slug")
    parser.add_argument("--all", action="store_true", help="Clean all texts")
    parser.add_argument("--force", action="store_true", help="Re-clean existing")
    parser.add_argument("--diff-only", action="store_true",
                        help="Only regenerate diffs (no LLM calls)")
    args = parser.parse_args()

    if not args.slug and not args.all:
        parser.error("Specify --slug or --all")

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key and not args.diff_only:
        print("MISTRAL_API_KEY not found", file=sys.stderr)
        return 1

    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    sleep_s = float(os.getenv("REQUEST_SLEEP_SECONDS", "1.0"))

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    # Gather files
    if args.slug:
        txt_files = [TEXTS_DIR / f"{args.slug}.txt"]
    else:
        txt_files = sorted(TEXTS_DIR.glob("*.txt"))

    results = {"cleaned": 0, "exists": 0, "error": 0, "diff_only": 0}

    with httpx.Client(timeout=1200.0) as client:
        for txt_path in tqdm(txt_files, desc="Cleaning"):
            slug = txt_path.stem
            clean_path = CLEAN_DIR / f"{slug}.txt"
            diff_path = CLEAN_DIR / f"{slug}.diff.html"

            original = txt_path.read_text(encoding="utf-8")

            if args.diff_only:
                if clean_path.exists():
                    cleaned = clean_path.read_text(encoding="utf-8")
                    diff_html = generate_diff_html(slug, original, cleaned)
                    diff_path.write_text(diff_html, encoding="utf-8")
                    results["diff_only"] += 1
                continue

            if clean_path.exists() and not args.force:
                results["exists"] += 1
                continue

            try:
                n_chunks = len(chunk_text(original))
                if n_chunks > 1:
                    print(f"  {slug}: {len(original)} chars -> {n_chunks} chunks", file=sys.stderr)

                cleaned = clean_document(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    text=original,
                    sleep_s=sleep_s,
                )

                # Strip markdown fences if the model wrapped it
                cleaned = re.sub(r"^```\w*\s*", "", cleaned)
                cleaned = re.sub(r"\s*```$", "", cleaned)

                clean_path.write_text(cleaned + "\n", encoding="utf-8")

                # Generate diff
                diff_html = generate_diff_html(slug, original, cleaned)
                diff_path.write_text(diff_html, encoding="utf-8")

                results["cleaned"] += 1

                if sleep_s > 0:
                    time.sleep(sleep_s)

            except Exception as exc:
                print(f"  Error cleaning {slug}: {exc}", file=sys.stderr)
                results["error"] += 1

    print(f"\nResults: {results}")
    return 1 if results["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())