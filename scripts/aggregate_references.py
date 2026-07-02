#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Aggregate per-document reference files into one deduplicated master list.

Reads the per-document markdown produced by ``run_references.py`` (one file per
guideline in ``references/``) and combines every referenced instrument into a
single, deduplicated list.

Deduplication has two layers:
1. Exact/normalized string match (always applied, deterministic): identical
   spellings referenced by several documents collapse into one entry.
2. Semantic clustering via one Mistral call (default; disable with --no-llm):
   spelling/language variants of the SAME instrument -- e.g. "DSGVO", "GDPR",
   "Datenschutz-Grundverordnung" -- are merged into a single group.

Anti-hallucination contract for the clustering call:
- Only names that were actually extracted are passed to the model.
- The model may only GROUP the given names; it must not invent, translate or
  reword them. The canonical name of each group must be one of the provided
  variants. Any name the model drops or alters is re-added as its own group, so
  nothing is lost or fabricated.

Usage:
    uv run scripts/aggregate_references.py                 # references/ -> references/_all_references.md
    uv run scripts/aggregate_references.py --no-llm        # deterministic string dedup only
    uv run scripts/aggregate_references.py --input references --output references/_all_references.md

Reads MISTRAL_API_KEY (and optionally MISTRAL_MODEL, MISTRAL_BASE_URL,
MAX_RETRIES, TEMPERATURE) from .env.
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
    "You are a meticulous policy-document analyst. You only group the names you "
    "are given; you never invent, translate or reword them. Return only valid JSON."
)

CLUSTER_PROMPT_TEMPLATE = """
Below is a JSON list of names of referenced documents/instruments (laws,
regulations, frameworks, guidelines, codes of conduct, standards) that were
extracted from many different guideline documents. Different guidelines refer to
the same instrument with different spellings, abbreviations or languages.

TASK:
Group the names that refer to the SAME underlying instrument. For example
"DSGVO", "GDPR" and "Datenschutz-Grundverordnung" all denote the same regulation
and belong in one group; "EU AI Act", "AI Act" and "KI-Verordnung" belong in one
group.

STRICT RULES:
- Use ONLY the names given in the input list. Do NOT invent new names, do NOT
  translate, do NOT reword, do NOT expand abbreviations.
- Each group's "canonical_name" MUST be copied verbatim from one of the names in
  that group (pick the most complete/formal variant present in the group).
- Every input name must appear in exactly ONE group, under "variants".
- Do not drop any name. Do not merge instruments that are genuinely different
  (e.g. two different national laws) just because they look similar.
- If unsure whether two names are the same instrument, keep them separate.

OUTPUT:
Return one JSON object: {{"groups": [{{"canonical_name": string,
"variants": [string, ...]}}, ...]}}.

--- INPUT NAMES (JSON) ---
{names_json}
--- END INPUT NAMES ---

Return ONLY the JSON object. No commentary and no markdown fences.
""".strip()


HEADING_RE = re.compile(r"^##\s+\d+\.\s+(.*)$")
FIELD_RE = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.*)$")

# Values the per-document renderer uses to mean "absent".
ABSENT_VALUES = {"", "none", "not stated"}


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def norm_key(s: str) -> str:
    return normalize(s).casefold()


def strip_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] in "\"'" and s[-1] in "\"'":
        return s[1:-1].strip()
    return s


def parse_reference_file(path: Path) -> list[dict[str, Any]]:
    """Parse one per-document references markdown file into a list of entries."""
    doc_id = path.stem
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for line in path.read_text(encoding="utf-8").splitlines():
        heading = HEADING_RE.match(line)
        if heading:
            if current is not None:
                entries.append(current)
            current = {
                "name": normalize(heading.group(1)),
                "type": "other",
                "issuer": None,
                "mention": None,
                "doc_id": doc_id,
            }
            continue

        if current is None:
            continue

        field = FIELD_RE.match(line)
        if not field:
            continue
        key = field.group(1).strip().lower()
        value = field.group(2).strip()
        if key == "type":
            current["type"] = value or "other"
        elif key == "issuer":
            current["issuer"] = None if value.lower() in ABSENT_VALUES else value
        elif key == "as referenced":
            current["mention"] = strip_quotes(value) or None

    if current is not None:
        entries.append(current)

    return [e for e in entries if e["name"]]


def collect_entries(input_dir: Path, output_path: Path) -> list[dict[str, Any]]:
    """Read every per-document markdown file (skipping the aggregate output and
    any file whose name starts with '_')."""
    entries: list[dict[str, Any]] = []
    for path in sorted(input_dir.glob("*.md")):
        if path.name.startswith("_") or path.resolve() == output_path.resolve():
            continue
        entries.extend(parse_reference_file(path))
    return entries


def build_surface_forms(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Collapse exact/normalized-identical names into surface forms, tracking the
    documents that use each and a representative type/mention."""
    forms: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = norm_key(e["name"])
        form = forms.get(key)
        if form is None:
            form = {
                "name": e["name"],
                "type_counts": {},
                "doc_ids": set(),
                "example_mention": e.get("mention"),
            }
            forms[key] = form
        form["type_counts"][e["type"]] = form["type_counts"].get(e["type"], 0) + 1
        form["doc_ids"].add(e["doc_id"])
        if form["example_mention"] is None and e.get("mention"):
            form["example_mention"] = e["mention"]
    return forms


def dominant_type(type_counts: dict[str, int]) -> str:
    if not type_counts:
        return "other"
    return max(type_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def call_mistral(
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_retries: int,
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


def semantic_groups(
    surface_names: list[str],
    client: httpx.Client,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_retries: int,
) -> list[list[str]]:
    """Ask the model to cluster the surface names, then validate that the result
    only regroups the given names (nothing invented, nothing lost)."""
    prompt = CLUSTER_PROMPT_TEMPLATE.format(
        names_json=json.dumps(surface_names, ensure_ascii=False, indent=2)
    )
    raw = call_mistral(
        client, base_url, api_key, model, prompt, temperature, max_retries
    )
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    parsed = json.loads(raw)

    by_key = {norm_key(n): n for n in surface_names}
    assigned: set[str] = set()
    groups: list[list[str]] = []

    for grp in parsed.get("groups", []) if isinstance(parsed, dict) else []:
        if not isinstance(grp, dict):
            continue
        members: list[str] = []
        for variant in grp.get("variants", []):
            key = norm_key(str(variant))
            # Only accept names that were actually in the input and not yet used.
            if key in by_key and key not in assigned:
                assigned.add(key)
                members.append(by_key[key])
        if members:
            groups.append(members)

    # Any surface name the model dropped or altered survives as its own group.
    for key, name in by_key.items():
        if key not in assigned:
            groups.append([name])

    return groups


def build_master_list(
    forms: dict[str, dict[str, Any]], groups: list[list[str]]
) -> list[dict[str, Any]]:
    """Merge surface forms into final deduplicated entries per group."""
    by_key = {norm_key(f["name"]): f for f in forms.values()}
    master: list[dict[str, Any]] = []

    for member_names in groups:
        member_forms = [by_key[norm_key(n)] for n in member_names if norm_key(n) in by_key]
        if not member_forms:
            continue

        doc_ids: set[str] = set()
        type_counts: dict[str, int] = {}
        variants: list[str] = []
        example_mention: str | None = None
        for f in member_forms:
            doc_ids |= f["doc_ids"]
            for t, c in f["type_counts"].items():
                type_counts[t] = type_counts.get(t, 0) + c
            variants.append(f["name"])
            if example_mention is None and f.get("example_mention"):
                example_mention = f["example_mention"]

        # Canonical = variant referenced by the most documents, then the longest.
        canonical = max(
            member_forms,
            key=lambda f: (len(f["doc_ids"]), len(f["name"])),
        )["name"]
        # Order variants: canonical first, rest by descending document usage.
        variants = sorted(
            {v for v in variants},
            key=lambda v: (v != canonical, -len(by_key[norm_key(v)]["doc_ids"]), v.casefold()),
        )

        master.append(
            {
                "canonical_name": canonical,
                "variants": variants,
                "type": dominant_type(type_counts),
                "doc_ids": sorted(doc_ids),
                "doc_count": len(doc_ids),
                "example_mention": example_mention,
            }
        )

    master.sort(key=lambda e: (-e["doc_count"], e["canonical_name"].casefold()))
    return master


def render_markdown(master: list[dict[str, Any]], source_count: int, method: str) -> str:
    lines = [
        "# All Referenced Documents (deduplicated)",
        "",
        (
            f"_Aggregated across {source_count} source document(s); "
            f"{len(master)} distinct instrument(s) after deduplication._"
        ),
        (
            f"_Variant grouping method: {method}. Names are taken verbatim from the "
            "per-document extractions_"
        ),
        "",
    ]

    if not master:
        lines.append("_No references found._")
        lines.append("")
        return "\n".join(lines)

    for i, e in enumerate(master, start=1):
        docs = "document" if e["doc_count"] == 1 else "documents"
        lines.append(f"## {i}. {e['canonical_name']} — {e['doc_count']} {docs}")
        lines.append("")
        lines.append(f"- **Type:** {e['type']}")
        other_variants = [v for v in e["variants"] if norm_key(v) != norm_key(e["canonical_name"])]
        if other_variants:
            joined = ", ".join(f"\"{v}\"" for v in other_variants)
            lines.append(f"- **Also referenced as:** {joined}")
        lines.append(f"- **Referenced by:** {', '.join(e['doc_ids'])}")
        if e.get("example_mention"):
            lines.append(f"- **Example mention:** \"{e['example_mention']}\"")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Aggregate per-document references into one deduplicated list"
    )
    parser.add_argument(
        "--input", type=Path, default=Path("references"),
        help="Directory of per-document reference markdown files (default: references)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output markdown file (default: <input>/_all_references.md)",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="Deterministic string dedup only; skip the semantic clustering call",
    )
    args = parser.parse_args()

    output_path = args.output or (args.input / "_all_references.md")

    if not args.input.is_dir():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        return 1

    entries = collect_entries(args.input, output_path)
    if not entries:
        print(f"No reference entries found in {args.input}", file=sys.stderr)
        return 1

    source_count = len({e["doc_id"] for e in entries})
    forms = build_surface_forms(entries)
    surface_names = [f["name"] for f in forms.values()]
    print(f"Parsed {len(entries)} entries from {source_count} documents "
          f"({len(surface_names)} distinct spellings).")

    if args.no_llm:
        groups = [[n] for n in surface_names]
        method = "exact string match (no semantic merge)"
    else:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            print("MISTRAL_API_KEY not found (set it in .env or use --no-llm)", file=sys.stderr)
            return 1
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        temperature = float(os.getenv("TEMPERATURE", "0"))
        max_retries = int(os.getenv("MAX_RETRIES", "2"))

        print(f"Clustering {len(surface_names)} names via {model}...")
        with httpx.Client(timeout=6400.0) as client:
            groups = semantic_groups(
                surface_names, client, base_url, api_key, model, temperature, max_retries
            )
        method = f"semantic clustering via {model}"

    master = build_master_list(forms, groups)
    md = render_markdown(master, source_count, method)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md, encoding="utf-8")

    # Machine-readable sidecar (consumed by build_explorer_data.py for the website).
    json_path = output_path.with_suffix(".json")
    payload = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "method": method,
        "source_document_count": source_count,
        "instrument_count": len(master),
        "references": master,
    }
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"Wrote {len(master)} deduplicated instruments -> {output_path} (+ {json_path.name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
