#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Single-pass AI guideline segmentation and B-coding via Mistral API.

Sends the full document in one prompt, gets back segmented JSON with B codes.
Validates segments against the source text and writes a single JSON per document
that the React explorer can consume.

Usage:
    uv run scripts/run_coding_v2.py --file texts/aspen-institute.txt
    uv run scripts/run_coding_v2.py --dir texts/

Reads MISTRAL_API_KEY (and optionally MISTRAL_MODEL, MISTRAL_BASE_URL,
OUTPUT_DIR, TEMPERATURE, MAX_RETRIES, REQUEST_SLEEP_SECONDS) from .env.

Output format (one {doc_id}.json per document):
    {
      "doc_id": "aspen-institute",
      "segments": [ { "id": 1, "B": "B3", "B_label": "...", "text": "..." }, ... ],
      "validation": {
        "total": 42, "matched": 41, "unmatched_count": 1,
        "unmatched": [ ... ],
        "source_coverage": 0.711,
        "uncovered_gaps": 9, "uncovered": [ ... ]
      }
    }
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


# ── codebook ────────────────────────────────────────────────────────────

B_CODEBOOK = {
    "B1": "definition of AI",
    "B2": "other definitions/terminology",
    "B3": "scope/application of the document",
    "B4": "purpose/rationale (why a guideline)/document status",
    "B5": "principles/values underlying document or AI use",
    "B6": "permitted or encouraged uses of AI",
    "B7": "restricted or prohibited uses of AI",
    "B8": "required safeguards/procedures for AI",
    "B9": "roles/accountability/oversight",
    "B10": "risks/limitations/concerns",
    "B11": "training/support/learning resources",
    "B12": "monitoring/revision/updating",
    "B13": "other/not coded/metadata",
}

B_CODES = list(B_CODEBOOK.keys())


# ── prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = "You are an expert qualitative coder for institutional AI-guideline documents. Return only valid JSON."

USER_PROMPT_TEMPLATE = """
Segment the following AI-guideline document into coherent topical units and assign a B code to each.

TASK:
1. Read the full document below.
2. Divide it into segments. A segment is one coherent codable unit: it may be a clause, a sentence, a paragraph, a list item, or a whole section. Segments must not overlap, and together must cover all substantive content. Skip metadata lines (headers starting with #, URLs, navigation text like "return to table of contents", "opens a new window", page numbers).
3. For each segment, assign exactly one B code (primary function).
4. Copy the segment text verbatim from the document. Do not paraphrase or summarize. If the segment is a list item, include only the item text, do not repeat the shared preamble for each item, because we need to string match to the original text. You need to stay true to the exact original text.

B CODES (primary function — what is the segment doing?):
{b_codebook}

CODING GUIDANCE:
- If a segment is about what the document IS, why it exists, who it is for, prefer B1/B2/B3/B4 over operational codes.
- Prefer B10 when describing a risk or concern; prefer B8 when prescribing a concrete safeguard in response.
- Use B5 for high-level values or principles that don't prescribe a concrete action.
- Use B9 when the segment assigns responsibility, accountability, or oversight roles.

Return a JSON array where each element has:
- "id": integer starting at 1
- "B": one of {b_codes}
- "text": the exact verbatim segment text from the document, do not modify the text in any way

DOCUMENT ID: {doc_id}

--- DOCUMENT TEXT ---
{document_text}
--- END DOCUMENT TEXT ---

Return ONLY the JSON array. No commentary, no markdown fences.
""".strip()


def build_prompt(doc_id: str, document_text: str) -> str:
    b_lines = "\n".join(f"  {k}: {v}" for k, v in B_CODEBOOK.items())
    return USER_PROMPT_TEMPLATE.format(
        b_codebook=b_lines,
        b_codes=B_CODES,
        doc_id=doc_id,
        document_text=document_text,
    )


REPAIR_PROMPT_TEMPLATE = """
You previously segmented and coded a document, but some passages were missed.
Below are the UNCOVERED PASSAGES that need to be segmented and coded.

TASK:
1. For each passage, decide: is it substantive guideline content, or is it
   metadata / navigation / boilerplate that should be skipped?
2. For substantive content, segment it into coherent units and assign a B code.
3. For non-substantive content (headings, page numbers, URLs, navigation text,
   footers), skip it entirely.
4. Copy the segment text verbatim from the passage. Do not paraphrase. If the segment is a list item, include only the item text, do not repeat the shared preamble for each item, because we need to string match to the original text. You need to stay true to the exact original text.

B CODES:
{b_codebook}

Return a JSON array (may be empty if all passages are non-substantive).
Each element has:
- "id": integer starting at 1
- "B": one of {b_codes}
- "text": the verbatim segment text

DOCUMENT ID: {doc_id}

--- UNCOVERED PASSAGES ---
{gaps_text}
--- END UNCOVERED PASSAGES ---

Return ONLY the JSON array. No commentary, no markdown fences.
""".strip()


def build_repair_prompt(doc_id: str, gaps: list[dict[str, Any]]) -> str:
    b_lines = "\n".join(f"  {k}: {v}" for k, v in B_CODEBOOK.items())
    parts = []
    for i, gap in enumerate(gaps):
        parts.append(f"[GAP {i}] ({gap['length']} chars)\n{gap['text']}")
    return REPAIR_PROMPT_TEMPLATE.format(
        b_codebook=b_lines,
        b_codes=B_CODES,
        doc_id=doc_id,
        gaps_text="\n\n".join(parts),
    )


# ── gap classification ────────────────────────────────────────────────

SKIP_PATTERNS = [
    re.compile(r"^#\s"),                                    # metadata headers
    re.compile(r"https?://\S+$"),                           # bare URLs
    re.compile(r"return to table of contents", re.I),
    re.compile(r"opens a new (window|document)", re.I),
    re.compile(r"^Stand\s+\d{2}\.\d{2}"),                   # page footers
    re.compile(r"^\d+$"),                                   # page numbers
    re.compile(r"^[-─]{3,}$"),                              # dividers
    re.compile(r"Sign up for", re.I),
    re.compile(r"reCAPTCHA", re.I),
    re.compile(r"Browse More Reports", re.I),
]


def is_substantive_gap(gap: dict[str, Any], min_chars: int = 40) -> bool:
    """Decide whether an uncovered gap contains substantive content
    that the model should have coded, vs. metadata/navigation/boilerplate."""
    text = gap["text"].strip()

    if len(text) < min_chars:
        return False

    for pat in SKIP_PATTERNS:
        if pat.search(text):
            return False

    # short headings without sentence-ending punctuation
    words = text.split()
    if len(words) <= 10 and not re.search(r"[.!?]", text):
        return False

    # numbered section headers: "I. Title", "2) Title", "a) Title", "A. Title"
    if re.match(r"^\.?\s*(?:[IVXLC]+\.|[A-Z]\.|[a-z]\)|\d+[.)]\s)", text.lstrip(". ")):
        if len(words) <= 12:
            return False

    return True


# ── text cleaning ──────────────────────────────────────────────────────

def clean_document_text(raw: str) -> str:
    """Light cleanup: normalize whitespace, dehyphenate PDF line breaks."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # PDF line-break dehyphenation: "verbun-\ndene" -> "verbundene"
    # Only join when a lowercase letter precedes the hyphen and a lowercase
    # letter follows the newline (avoids merging real compound words like
    # "Helmholtz-\nGemeinschaft" where the next line starts uppercase).
    text = re.sub(r"([a-zäöüß])-\n([a-zäöüß])", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── API call ───────────────────────────────────────────────────────────

def call_mistral(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
    max_retries: int = 2,
) -> str:
    """Call Mistral chat completion and return the raw content string."""
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
        "max_tokens": 128_768,
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
                    item.get("text", "") for item in content
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


# ── parsing & segment validation ─────────────────────────────────────

def parse_response(raw: str) -> list[dict[str, Any]]:
    """Extract JSON array from the model response."""
    text = raw.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    parsed = json.loads(text)

    if isinstance(parsed, dict):
        for key in ("segments", "results", "data", "items"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        else:
            raise ValueError(f"Expected JSON array, got object with keys: {list(parsed.keys())}")

    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")

    return parsed


def validate_segments(segments: list[dict[str, Any]], doc_id: str) -> list[dict[str, Any]]:
    """Validate B codes and build clean segment list."""
    validated = []
    for i, seg in enumerate(segments):
        b = seg.get("B", "")
        text = seg.get("text", "")

        if b not in B_CODES:
            print(f"  Warning: segment {i+1} has invalid B code '{b}', skipping", file=sys.stderr)
            continue
        if not text.strip():
            print(f"  Warning: segment {i+1} has empty text, skipping", file=sys.stderr)
            continue

        validated.append({
            "id": len(validated) + 1,
            "B": b,
            "B_label": B_CODEBOOK[b],
            "text": text.strip(),
        })

    return validated


# ── source validation ─────────────────────────────────────────────────

def _normalize_for_matching(text: str) -> str:
    """Normalize formatting artifacts so that model output can be matched
    against the source document even when either side has been cleaned up.

    Handles: unicode quotes/hyphens, footnote markers, PDF line-break
    hyphenation, bullet markers, page footers, punctuation spacing,
    inline footnote text, glued superscript numbers, and case differences.
    """
    t = text
    # ── unicode normalization ──
    t = t.replace("\u2018", "'").replace("\u2019", "'")     # curly single quotes
    t = t.replace("\u201C", '"').replace("\u201D", '"')     # curly double quotes
    t = t.replace("\u2013", "-").replace("\u2014", "-")     # en/em dash
    t = t.replace("\u2011", "-").replace("\u2010", "-")     # non-breaking / hyphen
    t = t.replace("\u00AD", "")                             # soft hyphen
    t = t.replace("\u00A0", " ")                            # non-breaking space
    # ── structural noise ──
    t = re.sub(r"\[\d+\]", "", t)                           # footnote markers [1]
    # inline superscript-style footnote numbers: "KI-5 Anwendungen" -> "KI- Anwendungen"
    t = re.sub(r"(?<=[a-zäöüßA-ZÄÖÜ])-(\d{1,2})\s+(?=[A-ZÄÖÜ])", "- ", t)
    # footnote numbers glued to word: "Policy6" -> "Policy"
    t = re.sub(r"([a-zäöüß])\d{1,2}(?=[\s.,;:!?)\]]|$)", r"\1", t)
    t = re.sub(r"\s\d{1,2}\s(?=Siehe\s)", " ", t)           # "1 Siehe z.B." footnote refs
    # inline footnote body: "Siehe z.B. ... Royal Society. 3"
    t = re.sub(r"Siehe\s+(?:z\.B\.\s+|u\.\s*a\.\s+|auch\s+).*?(?:\.\s*\d*\s*(?=\s[A-ZÄÖÜ])|\.\s*$)", " ", t)
    t = re.sub(r"Stand\s+\d{2}\.\d{2}\.\d{2,4}\s*\d*\s*(?:---?)?", "", t)  # page footers
    t = re.sub(r"\s*---\s*", " ", t)                        # section dividers
    # ── bullet / list markers ──
    t = re.sub(r"(?:^|\n)\s*[-•]\s+", " ", t)
    # ── FIRST collapse whitespace (newlines, tabs, multi-space) ──
    t = re.sub(r"\s+", " ", t).strip()
    # ── page numbers (must run AFTER whitespace collapse) ──
    # lone page numbers between sentences: ". 3 Probleme" -> ". Probleme"
    t = re.sub(r"(?<=\.)\s+\d{1,2}\s+(?=[A-ZÄÖÜa-zäöüß])", " ", t)
    # page numbers before section markers: "erlaubten 7 § 4" -> "erlaubten § 4"
    t = re.sub(r"\s+\d{1,2}\s+(?=§)", " ", t)
    # inline TOC debris: consecutive "§ N title" entries merged into body text
    t = re.sub(r"§\s*\d+\s+[a-zäöüßA-ZÄÖÜ][a-zäöüßA-ZÄÖÜ-]+(?:\s+[a-zäöüßA-ZÄÖÜ][a-zäöüßA-ZÄÖÜ-]+)*\s+(?=§)", "", t)
    # page numbers between text: "die 4 Grundsätze" or "Aufsicht) 14 b)"
    t = re.sub(r"(?<=[a-zäöüß)\]])\s+\d{1,2}\s+(?=[A-ZÄÖÜa-zäöüß§(])", " ", t)
    # ── PDF line-break hyphenation: "KI- Systemen" -> "KI-Systemen" ──
    t = re.sub(r"(\S)- (\S)", r"\1-\2", t)
    # ── word-internal dehyphenation: "verbun-dene" -> "verbundene" ──
    t = re.sub(r"([a-zäöüß])-([a-zäöüß])", r"\1\2", t)
    # ── punctuation spacing: "rights ." -> "rights." ──
    t = re.sub(r"\s+([.,;:!?)\]])", r"\1", t)
    # ── commas model may insert between list items / before URLs ──
    t = re.sub(r",\s*(?=https?://)", " ", t)
    # ── trailing punctuation that model may add/drop ──
    t = t.rstrip(".,;: ")
    # ── case-fold for matching ──
    t = t.lower()
    return t


def validate_against_source(
    segments: list[dict[str, Any]],
    source_text: str,
) -> dict[str, Any]:
    """Check every segment's text against the source document.

    Returns a dict with:
      - matched/unmatched segment counts and details
      - source_coverage: fraction of source chars covered by matched segments
      - uncovered: list of source text passages not covered by any segment
    """
    source_norm = _normalize_for_matching(source_text)

    matched_ids = []
    unmatched = []
    covered_ranges: list[tuple[int, int]] = []

    for seg in segments:
        seg_norm = _normalize_for_matching(seg["text"])
        idx = source_norm.find(seg_norm)

        if idx >= 0:
            matched_ids.append(seg["id"])
            covered_ranges.append((idx, idx + len(seg_norm)))
        else:
            words = seg_norm.split()
            prefix_match = 0
            for n in range(len(words), 0, -1):
                if " ".join(words[:n]) in source_norm:
                    prefix_match = n
                    break

            unmatched.append({
                "id": seg["id"],
                "B": seg["B"],
                "matched_words": f"{prefix_match}/{len(words)}",
                "text": seg["text"],
            })

    # find uncovered gaps
    covered_ranges.sort()
    uncovered: list[dict[str, Any]] = []
    prev_end = 0
    for start, end in covered_ranges:
        if start > prev_end:
            gap_text = source_norm[prev_end:start].strip()
            if len(gap_text) > 5:
                uncovered.append({
                    "char_start": prev_end,
                    "char_end": start,
                    "length": start - prev_end,
                    "text": gap_text,
                })
        prev_end = max(prev_end, end)
    if prev_end < len(source_norm):
        gap_text = source_norm[prev_end:].strip()
        if len(gap_text) > 5:
            uncovered.append({
                "char_start": prev_end,
                "char_end": len(source_norm),
                "length": len(source_norm) - prev_end,
                "text": gap_text,
            })

    covered_chars = sum(e - s for s, e in covered_ranges)
    coverage = covered_chars / len(source_norm) if source_norm else 0.0

    return {
        "total": len(segments),
        "matched": len(matched_ids),
        "unmatched_count": len(unmatched),
        "unmatched": unmatched,
        "source_coverage": round(min(coverage, 1.0), 3),
        "uncovered_chars": sum(g["length"] for g in uncovered),
        "uncovered_gaps": len(uncovered),
        "uncovered": uncovered,
    }


# ── source projection ────────────────────────────────────────────────

def project_to_source(
    segments: list[dict[str, Any]],
    source_text: str,
) -> list[dict[str, Any]]:
    """Project coded segments back onto the original source text.

    Returns an ordered list of spans covering 100% of the source text.
    Each span has:
      - char_start, char_end: offsets into source_text
      - text: the verbatim source text for this span
      - B: the B-code (or "skip" for uncoded regions)
      - B_label: human-readable label
      - segment_id: id of the matched segment (or None for skips)
      - match_type: "exact" or "gap"

    Strategy: for each segment, find its text in the original source
    using progressive relaxation (exact -> case-insensitive -> normalized).
    Then fill gaps between matched positions with "skip" spans.
    """
    source_lower = source_text.lower()

    matched_spans: list[dict[str, Any]] = []
    used_ranges: list[tuple[int, int]] = []  # prevent double-matching

    for seg in segments:
        seg_text = seg["text"].strip()
        pos = _find_in_source(seg_text, source_text, source_lower, used_ranges)
        if pos is not None:
            start, end = pos
            matched_spans.append({
                "char_start": start,
                "char_end": end,
                "B": seg["B"],
                "B_label": seg.get("B_label", B_CODEBOOK.get(seg["B"], "")),
                "segment_id": seg["id"],
                "match_type": "exact",
            })
            used_ranges.append((start, end))

    # Sort by position in source
    matched_spans.sort(key=lambda s: s["char_start"])

    # Build full coverage: fill gaps between matched spans
    result: list[dict[str, Any]] = []
    prev_end = 0

    for span in matched_spans:
        if span["char_start"] > prev_end:
            gap_text = source_text[prev_end:span["char_start"]]
            if gap_text.strip():
                result.append({
                    "char_start": prev_end,
                    "char_end": span["char_start"],
                    "text": gap_text,
                    "B": "skip",
                    "B_label": "structural/metadata",
                    "segment_id": None,
                    "match_type": "gap",
                })

        result.append({
            "char_start": span["char_start"],
            "char_end": span["char_end"],
            "text": source_text[span["char_start"]:span["char_end"]],
            "B": span["B"],
            "B_label": span["B_label"],
            "segment_id": span["segment_id"],
            "match_type": span["match_type"],
        })
        prev_end = max(prev_end, span["char_end"])

    # Trailing gap
    if prev_end < len(source_text):
        gap_text = source_text[prev_end:]
        if gap_text.strip():
            result.append({
                "char_start": prev_end,
                "char_end": len(source_text),
                "text": gap_text,
                "B": "skip",
                "B_label": "structural/metadata",
                "segment_id": None,
                "match_type": "gap",
            })

    return result


def _find_in_source(
    seg_text: str,
    source: str,
    source_lower: str,
    used_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find segment text in source using progressive relaxation.

    1. Exact substring match
    2. Case-insensitive match
    3. Normalized match (find position in normalized space, then
       map back to original using first/last word anchors)
    """
    # Strategy 1: exact match
    idx = source.find(seg_text)
    if idx >= 0 and not _overlaps(idx, idx + len(seg_text), used_ranges):
        return idx, idx + len(seg_text)

    # Strategy 2: case-insensitive match
    seg_lower = seg_text.lower()
    idx = source_lower.find(seg_lower)
    if idx >= 0 and not _overlaps(idx, idx + len(seg_lower), used_ranges):
        return idx, idx + len(seg_lower)

    # Strategy 3: use first and last ~30 chars as anchors
    # This handles cases where the model slightly modified the middle
    anchor_len = min(30, len(seg_lower) // 2)
    if anchor_len < 10:
        return None

    start_anchor = seg_lower[:anchor_len]
    end_anchor = seg_lower[-anchor_len:]

    start_idx = source_lower.find(start_anchor)
    if start_idx < 0:
        return None

    # search for end anchor after start
    end_search_start = start_idx + anchor_len
    end_idx = source_lower.find(end_anchor, end_search_start)
    if end_idx < 0:
        return None

    span_end = end_idx + len(end_anchor)
    if not _overlaps(start_idx, span_end, used_ranges):
        return start_idx, span_end

    return None


def _overlaps(start: int, end: int, used: list[tuple[int, int]]) -> bool:
    """Check if [start, end) overlaps any existing range."""
    for us, ue in used:
        if start < ue and end > us:
            return True
    return False


# ── output ────────────────────────────────────────────────────────────

def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ── main processing ───────────────────────────────────────────────────

def process_file(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    text_path: Path,
    out_dir: Path,
    temperature: float,
    max_retries: int,
) -> dict[str, Any]:
    doc_id = text_path.stem
    raw_text = text_path.read_text(encoding="utf-8")
    document_text = clean_document_text(raw_text)

    print(f"Processing {doc_id} ({len(document_text)} chars)...")

    # ── pass 1: full document coding ──
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

    segments_raw = parse_response(raw_response)
    segments = validate_segments(segments_raw, doc_id)
    report = validate_against_source(segments, document_text)

    print(
        f"  Pass 1: {len(segments)} segments | "
        f"match: {report['matched']}/{report['total']} | "
        f"coverage: {report['source_coverage']:.1%}"
    )

    # ── pass 2: repair substantive gaps ──
    substantive_gaps = [g for g in report["uncovered"] if is_substantive_gap(g)]

    if substantive_gaps:
        gap_chars = sum(g["length"] for g in substantive_gaps)
        print(
            f"  Repair: {len(substantive_gaps)} substantive gaps "
            f"({gap_chars} chars), sending repair pass..."
        )

        repair_prompt = build_repair_prompt(doc_id, substantive_gaps)
        try:
            repair_response = call_mistral(
                client=client,
                base_url=base_url,
                api_key=api_key,
                model=model,
                prompt=repair_prompt,
                temperature=temperature,
                max_retries=max_retries,
            )
            repair_raw = parse_response(repair_response)
            repair_segments = validate_segments(repair_raw, doc_id)

            if repair_segments:
                for seg in repair_segments:
                    seg["repaired"] = True
                for seg in segments:
                    seg.setdefault("repaired", False)

                segments = segments + repair_segments
                # renumber
                for i, seg in enumerate(segments):
                    seg["id"] = i + 1

                report = validate_against_source(segments, document_text)
                print(
                    f"  Pass 2: {len(segments)} segments (+{len(repair_segments)}) | "
                    f"match: {report['matched']}/{report['total']} | "
                    f"coverage: {report['source_coverage']:.1%}"
                )
        except Exception as e:
            print(f"  Repair pass failed: {e}", file=sys.stderr)
    else:
        print("  No substantive gaps, skipping repair.")

    # ── project to source for website ──
    projection = project_to_source(segments, document_text)
    coded_spans = [s for s in projection if s["B"] != "skip"]
    skip_spans = [s for s in projection if s["B"] == "skip"]
    total_chars = len(document_text)
    coded_chars = sum(s["char_end"] - s["char_start"] for s in coded_spans)
    print(
        f"  Projection: {len(projection)} spans "
        f"({len(coded_spans)} coded, {len(skip_spans)} skipped) | "
        f"char coverage: {coded_chars}/{total_chars} ({coded_chars/total_chars:.1%})"
    )

    # ── write outputs ──
    output = {
        "doc_id": doc_id,
        "segments": segments,
        "validation": report,
        "projection": projection,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / f"{doc_id}.json", output)

    # summary
    print(
        f"  {doc_id}: {len(segments)} segments | "
        f"match: {report['matched']}/{report['total']} | "
        f"coverage: {report['source_coverage']:.1%} | "
        f"unmatched: {report['unmatched_count']}"
    )

    if report["unmatched"]:
        for u in report["unmatched"]:
            print(
                f"    UNMATCHED id={u['id']} B={u['B']} ({u['matched_words']}): "
                f"{u['text'][:80]}",
                file=sys.stderr,
            )

    return output


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Segment and B-code AI guideline documents (single-pass via Mistral)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="Process a single .txt file")
    group.add_argument("--dir", type=Path, help="Process all .txt files in directory")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    args = parser.parse_args()

    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        print("MISTRAL_API_KEY not found (set it in .env or environment)", file=sys.stderr)
        return 1

    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
    model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
    temperature = float(os.getenv("TEMPERATURE", "0"))
    max_retries = int(os.getenv("MAX_RETRIES", "2"))
    out_dir = args.output or Path(os.getenv("OUTPUT_DIR", "outputs"))
    sleep_between = float(os.getenv("REQUEST_SLEEP_SECONDS", "0.0"))

    if args.file:
        files = [args.file]
    else:
        files = sorted(args.dir.glob("*.txt"))

    for f in files:
        if not f.exists():
            print(f"File not found: {f}", file=sys.stderr)
            return 1

    print(f"Model: {model}")
    print(f"Files: {len(files)}")

    with httpx.Client(timeout=6400.0) as client:
        for i, text_path in enumerate(files):
            try:
                process_file(
                    client=client,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    text_path=text_path,
                    out_dir=out_dir,
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