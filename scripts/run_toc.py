#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Extract or generate table-of-contents markdown files via Mistral.

For each input .txt document:
- If a Table of Contents exists, extract it.
- Otherwise, generate a ToC from headings found in the text.

Writes one markdown file per document to a dedicated output folder.

Usage:
    uv run scripts/run_toc.py --file texts/aspen-institute_1_0.txt
    uv run scripts/run_toc.py --dir texts
    uv run scripts/run_toc.py   # defaults to --dir texts

Reads MISTRAL_API_KEY (and optionally MISTRAL_MODEL, MISTRAL_BASE_URL,
MAX_RETRIES, REQUEST_SLEEP_SECONDS, TEMPERATURE) from .env.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SYSTEM_PROMPT = "You are an expert document-structure analyst. Return only valid JSON."

USER_PROMPT_TEMPLATE = """
Analyze the following document and produce a structured table of contents (ToC).

TASK:
1. Determine whether the document contains an explicit ToC section.
   - Indicators can include headings like "table of contents", "contents",
     "inhaltsverzeichnis", "inhalt", or similar in the document language.
2. If explicit ToC entries exist, extract them in order.
3. If no explicit ToC exists, generate a ToC from the document's heading
   structure (major and minor headings only).
4. Keep item titles concise and faithful to the source wording.
5. Preserve original language.

OUTPUT REQUIREMENTS:
- Return one JSON object with these fields:
  - "doc_id": string
  - "mode": "extracted_toc" or "generated_from_headings"
  - "items": array of objects with:
    - "id": integer starting at 1
    - "level": integer 1-6
    - "title": string
    - "numbering": string or null (for examples: "1", "2.3", "IV")
- Include only meaningful section items. Skip page numbers, URLs, nav labels,
  legal boilerplate footers, and duplicated items.

DOCUMENT ID: {doc_id}

--- DOCUMENT TEXT ---
{document_text}
--- END DOCUMENT TEXT ---

Return ONLY the JSON object. No commentary and no markdown fences.
""".strip()


def build_prompt(doc_id: str, document_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(doc_id=doc_id, document_text=document_text)


def clean_document_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([a-zA-ZäöüÄÖÜß])-\n([a-zA-ZäöüÄÖÜß])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def call_mistral(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
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
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 200_000,
        "response_format": {"type": "json_object"},
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
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            return content.strip()
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
                wait = 2 * (attempt + 1)
                print(f"  HTTP {e.response.status_code}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise

    raise RuntimeError("Exhausted retries")


def parse_response(raw: str) -> dict[str, Any]:
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
    return parsed


def validate_toc_response(data: dict[str, Any], doc_id: str) -> dict[str, Any]:
    mode = data.get("mode", "generated_from_headings")
    if mode not in {"extracted_toc", "generated_from_headings"}:
        mode = "generated_from_headings"

    raw_items = data.get("items", [])
    if not isinstance(raw_items, list):
        raw_items = []

    valid_items: list[dict[str, Any]] = []
    for raw in raw_items:
        if not isinstance(raw, dict):
            continue
        title = str(raw.get("title", "")).strip()
        if not title:
            continue

        level_raw = raw.get("level", 1)
        try:
            level = int(level_raw)
        except Exception:
            level = 1
        level = max(1, min(6, level))

        numbering_raw = raw.get("numbering", None)
        numbering: str | None
        if numbering_raw is None or str(numbering_raw).strip() == "":
            numbering = None
        else:
            numbering = str(numbering_raw).strip()

        valid_items.append(
            {
                "id": len(valid_items) + 1,
                "level": level,
                "title": title,
                "numbering": numbering,
            }
        )

    return {
        "doc_id": doc_id,
        "mode": mode,
        "items": valid_items,
    }


def render_markdown(doc_id: str, mode: str, items: list[dict[str, Any]]) -> str:
    lines = [f"# Table of Contents: {doc_id}", "", f"_Mode: {mode}_", ""]

    if not items:
        lines.append("_No table-of-contents entries could be extracted._")
        lines.append("")
        return "\n".join(lines)

    for item in items:
        level = int(item["level"])
        indent = "  " * (level - 1)
        numbering = item.get("numbering")
        title = item["title"]

        if numbering and not title.startswith(f"{numbering} "):
            label = f"{numbering} {title}"
        else:
            label = title

        lines.append(f"{indent}- {label}")

    lines.append("")
    return "\n".join(lines)


def write_markdown(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def process_file(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    text_path: Path,
    out_dir: Path,
    temperature: float,
    max_retries: int,
) -> None:
    doc_id = text_path.stem
    raw_text = text_path.read_text(encoding="utf-8")
    document_text = clean_document_text(raw_text)

    print(f"Processing {doc_id} ({len(document_text)} chars)...")

    prompt = build_prompt(doc_id, document_text)
    raw_response = call_mistral(
        client=client,
        base_url=base_url,
        api_key=api_key,
        model=model,
        prompt=prompt,
        temperature=temperature,
        max_retries=max_retries,
    )
    parsed = parse_response(raw_response)
    toc = validate_toc_response(parsed, doc_id)

    md = render_markdown(doc_id, toc["mode"], toc["items"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.md"
    write_markdown(out_path, md)

    print(f"  {doc_id}: {len(toc['items'])} items | mode={toc['mode']} -> {out_path}")


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract/generate table of contents markdown files via Mistral"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", type=Path, help="Process a single .txt file")
    group.add_argument("--dir", type=Path, help="Process all .txt files in directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("toc_markdown"),
        help="Output directory for markdown ToC files (default: toc_markdown)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if markdown output already exists",
    )
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("MISTRAL_API_KEY not found (set it in .env or environment)", file=sys.stderr)
        return 1

    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    max_retries = int(os.getenv("MAX_RETRIES", "2"))
    sleep_between = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.0"))

    if args.file:
        files = [args.file]
    else:
        source_dir = args.dir or Path("texts")
        files = sorted(source_dir.glob("*.txt"))

    for f in files:
        if not f.exists():
            print(f"File not found: {f}", file=sys.stderr)
            return 1

    print(f"Model: {model}")
    print(f"Files: {len(files)}")
    print(f"Output: {args.output}")

    with httpx.Client(timeout=6400.0) as client:
        for i, text_path in enumerate(files):
            out_path = args.output / f"{text_path.stem}.md"
            if out_path.exists() and not args.force:
                print(f"  exists, skipping {out_path.name} (use --force to regenerate)")
                continue

            try:
                process_file(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    text_path=text_path,
                    out_dir=args.output,
                    temperature=temperature,
                    max_retries=max_retries,
                )
            except Exception as e:
                print(f"  FAILED {text_path.name}: {e}", file=sys.stderr)
                continue

            if sleep_between > 0 and i < len(files) - 1:
                time.sleep(sleep_between)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
