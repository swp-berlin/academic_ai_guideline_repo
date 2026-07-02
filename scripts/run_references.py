#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Extract referenced external documents from each guideline via Mistral.

For each input .txt document, extract the list of *other documents* that the
guideline explicitly references as relevant/related to it -- for example laws
and regulations (GDPR/Datenschutz-Grundverordnung, EU AI Act), ethics
frameworks (EU Ethics Guidelines for Trustworthy AI), standards, other
institutional guidelines or codes of conduct, etc.

This is deliberately NOT a scientific-citation extractor: journal articles,
books and papers cited as academic evidence are ignored. Only documents that
are referenced as normative/related instruments are captured.

Anti-hallucination contract:
- Only documents explicitly named in the text are extracted.
- Names are copied verbatim, in the source language, exactly as written.
- Incomplete references are kept incomplete (no expanding abbreviations,
  no inventing official titles, dates or reference numbers).
- Each entry carries a verbatim "mention" snippet quoted from the source so
  every extraction is auditable against the text.

Writes one markdown file per document to a dedicated output folder.

Usage:
    uv run scripts/run_references.py --file texts_clean/eu-ai-act_1_0.txt
    uv run scripts/run_references.py --dir texts_clean
    uv run scripts/run_references.py   # defaults to --dir texts_clean

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


SYSTEM_PROMPT = (
    "You are a meticulous policy-document analyst. You extract only what is "
    "literally present in the text and never invent details. Return only valid JSON."
)

# Controlled vocabulary for the "type" field. Anything else is mapped to "other".
ALLOWED_TYPES = {
    "law_or_regulation",
    "framework_or_guideline",
    "standard",
    "code_of_conduct",
    "policy",
    "declaration_or_charter",
    "other",
}

USER_PROMPT_TEMPLATE = """
Analyze the following guideline / policy document and extract every OTHER
document that it references as being relevant or related to it.

WHAT TO EXTRACT (referenced instruments), for example:
- laws and regulations (e.g. "Datenschutz-Grundverordnung", "GDPR", "EU AI Act",
  a national copyright act, a university statute),
- ethics frameworks and guidelines (e.g. "EU Ethics Guidelines for Trustworthy AI"),
- technical standards (e.g. an ISO/IEC standard),
- codes of conduct, declarations, charters,
- other institutional policies or guidelines it points to.

WHAT TO IGNORE:
- Scientific / academic citations: journal articles, conference papers, books
  or reports cited as scholarly evidence (authors + year, DOIs, bibliographies).
- The document's own internal sections, headings, chapters or annexes.
- Generic mentions with no named document (e.g. "the law", "data protection",
  "applicable regulations") when NO specific instrument is named.
- Software products, tools, or AI systems (e.g. "ChatGPT") -- these are tools,
  not referenced documents.

STRICT ANTI-HALLUCINATION RULES:
- Extract ONLY documents that are explicitly named in the text below.
- Copy each document name VERBATIM, in its original language, exactly as it
  appears. Do NOT translate, do NOT expand abbreviations, do NOT add official
  full titles, dates, article numbers or reference numbers that are not in the
  text. If the text only says "GDPR", the name is "GDPR".
- If a reference is incomplete or informal, keep it incomplete. Reference it
  the same way the text does.
- For every entry, "mention" MUST be an exact, contiguous substring copied from
  the document text (the phrase/sentence where the document is referenced). If
  you cannot quote it verbatim, do not include the entry.
- Only fill "issuer" and "identifier" if that information is explicitly present
  in the text; otherwise use null. Never guess.
- If the document references no external instruments, return an empty list.

OUTPUT REQUIREMENTS:
- Return one JSON object with these fields:
  - "doc_id": string
  - "references": array of objects, each with:
    - "id": integer starting at 1
    - "name": string (verbatim referenced document name, original language)
    - "type": one of {allowed_types}
    - "issuer": string or null (organization/authority, only if stated in text)
    - "identifier": string or null (e.g. an explicit number/URL, only if in text)
    - "mention": string (exact verbatim substring from the document text)
- Merge obvious duplicates of the same instrument into one entry.

DOCUMENT ID: {doc_id}

--- DOCUMENT TEXT ---
{document_text}
--- END DOCUMENT TEXT ---

Return ONLY the JSON object. No commentary and no markdown fences.
""".strip()


def build_prompt(doc_id: str, document_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        document_text=document_text,
        allowed_types=", ".join(sorted(ALLOWED_TYPES)),
    )


def clean_document_text(raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([a-zA-ZäöüÄÖÜß])-\n([a-zA-ZäöüÄÖÜß])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_match(s: str) -> str:
    """Collapse whitespace so verbatim-substring checks tolerate our own
    reflowing of the source text."""
    return re.sub(r"\s+", " ", s).strip()


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


def validate_references_response(
    data: dict[str, Any], doc_id: str, document_text: str
) -> dict[str, Any]:
    """Keep only well-formed entries whose verbatim mention is actually present
    in the source text. This is the anti-hallucination gate: any entry the model
    cannot ground with a real substring quote is dropped."""
    haystack = normalize_for_match(document_text).casefold()

    raw_items = data.get("references", [])
    if not isinstance(raw_items, list):
        raw_items = []

    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped_ungrounded = 0

    for raw in raw_items:
        if not isinstance(raw, dict):
            continue

        name = str(raw.get("name", "")).strip()
        if not name:
            continue

        mention = str(raw.get("mention", "")).strip()
        # Anti-hallucination gate: the mention must be a real substring of the
        # source text (whitespace-normalized, case-insensitive).
        if not mention:
            dropped_ungrounded += 1
            continue
        needle = normalize_for_match(mention).casefold()
        if needle not in haystack:
            dropped_ungrounded += 1
            continue

        dedupe_key = normalize_for_match(name).casefold()
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        ref_type = str(raw.get("type", "other")).strip().lower()
        if ref_type not in ALLOWED_TYPES:
            ref_type = "other"

        def opt(field: str) -> str | None:
            v = raw.get(field, None)
            if v is None:
                return None
            v = str(v).strip()
            return v or None

        valid.append(
            {
                "id": len(valid) + 1,
                "name": name,
                "type": ref_type,
                "issuer": opt("issuer"),
                "identifier": opt("identifier"),
                "mention": normalize_for_match(mention),
            }
        )

    return {
        "doc_id": doc_id,
        "references": valid,
        "dropped_ungrounded": dropped_ungrounded,
    }


def render_markdown(doc_id: str, references: list[dict[str, Any]]) -> str:
    lines = [
        f"# Referenced Documents: {doc_id}",
        "",
        (
            "_Other documents (laws, regulations, frameworks, guidelines, codes of "
            "conduct, etc.) that this document references as relevant. Names are "
            "copied verbatim from the source; incomplete references are kept as-is. "
            "Scientific/academic citations are excluded._"
        ),
        "",
        f"_{len(references)} reference(s) extracted._",
        "",
    ]

    if not references:
        lines.append("_No external documents were referenced in this document._")
        lines.append("")
        return "\n".join(lines)

    for ref in references:
        lines.append(f"## {ref['id']}. {ref['name']}")
        lines.append("")
        lines.append(f"- **Type:** {ref['type']}")
        lines.append(f"- **Issuer:** {ref['issuer'] if ref['issuer'] else 'not stated'}")
        lines.append(
            f"- **Identifier:** {ref['identifier'] if ref['identifier'] else 'none'}"
        )
        lines.append(f"- **As referenced:** \"{ref['mention']}\"")
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
    result = validate_references_response(parsed, doc_id, document_text)

    md = render_markdown(doc_id, result["references"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{doc_id}.md"
    write_markdown(out_path, md)

    dropped = result["dropped_ungrounded"]
    note = f" | dropped {dropped} ungrounded" if dropped else ""
    print(
        f"  {doc_id}: {len(result['references'])} references{note} -> {out_path}"
    )


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Extract referenced external documents as markdown via Mistral"
    )
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--file", type=Path, help="Process a single .txt file")
    group.add_argument("--dir", type=Path, help="Process all .txt files in directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("references"),
        help="Output directory for markdown reference files (default: references)",
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
        source_dir = args.dir or Path("texts_clean")
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
