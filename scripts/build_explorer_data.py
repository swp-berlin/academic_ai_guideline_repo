#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["pyyaml"]
# ///
"""Build explorer_data.json for the guidelines explorer website.

Combines guidelines.yaml metadata with coding output from run_coding_v2.py.
Outputs a single JSON file that the React explorer app can fetch.

Usage:
    uv run scripts/build_explorer_data.py
    uv run scripts/build_explorer_data.py --output docs/explorer_data.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GUIDELINES_FILE = ROOT / "guidelines.yaml"
OUTPUTS_DIR = ROOT / "outputs"
TOC_DIR = ROOT / "toc_markdown"
REFERENCES_JSON = ROOT / "references" / "_all_references.json"


def load_references() -> dict | None:
    """Load the deduplicated master reference list produced by
    aggregate_references.py, if present."""
    if not REFERENCES_JSON.is_file():
        return None
    with open(REFERENCES_JSON, encoding="utf-8") as f:
        return json.load(f)


def find_coding_file(slug: str, version: str) -> Path | None:
    """Exact (slug, version) coding output: outputs/{slug}_{version}.json"""
    p = OUTPUTS_DIR / f"{slug}_{version}.json"
    return p if p.is_file() else None


def load_toc(slug: str, version: str) -> list[dict] | None:
    """Parse toc_markdown/{slug}_{version}.md into a flat list of {depth, text}.

    The markdown files are bullet lists that use two-space indentation per level.
    Returns None when there is no TOC file or it has no entries.
    """
    path = TOC_DIR / f"{slug}_{version}.md"
    if not path.is_file():
        return None

    entries: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.lstrip(" ")
        if not stripped.startswith("- "):
            continue
        indent = len(raw) - len(stripped)
        text = stripped[2:].strip()
        if not text:
            continue
        entries.append({"depth": indent // 2, "text": text})

    return entries or None


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Build explorer data JSON")
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "explorer_data.json")
    args = parser.parse_args()

    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        guidelines = yaml.safe_load(f)

    docs = []
    skipped = 0

    for entry in guidelines:
        slug = entry["slug"]
        version = str(entry.get("version") or "1_0").replace(".", "_")
        coding_path = find_coding_file(slug, version)

        if coding_path is None:
            skipped += 1
            continue

        with open(coding_path, encoding="utf-8") as f:
            coding = json.load(f)

        validation = coding.get("validation", None)
        unmatched_ids = {
            unmatched["id"]
            for unmatched in (validation or {}).get("unmatched", [])
            if "id" in unmatched
        }
        segments = [
            seg
            for seg in coding.get("segments", [])
            if seg.get("id") not in unmatched_ids
        ]
        if not segments:
            skipped += 1
            continue

        docs.append({
            "doc_id": f"{slug}_{version}",
            "institution": entry["institution"],
            "version": version,
            "category": entry.get("category", ""),
            "date": entry.get("date", ""),
            "language": entry.get("language", ""),
            "url": entry.get("url", ""),
            "toc": load_toc(slug, version),
            "segments": [
                {
                    "id": seg["id"],
                    "B": seg["B"],
                    "B_label": seg.get("B_label", B_CODEBOOK.get(seg["B"], "")),
                    "text": seg["text"],
                }
                for seg in segments
            ],
            "projection": [
                {
                    "char_start": span["char_start"],
                    "char_end": span["char_end"],
                    "text": span["text"],
                    "B": span["B"],
                    "B_label": span.get("B_label", ""),
                    "segment_id": span.get("segment_id"),
                    "match_type": span.get("match_type", ""),
                }
                for span in coding.get("projection", [])
            ] or None,
            "validation": {
                "matched": validation.get("matched", 0),
                "total": validation.get("total", 0),
                "unmatched_count": validation.get("unmatched_count", 0),
                "source_coverage": validation.get("source_coverage", 0),
                "unmatched": [
                    {
                        "id": u["id"],
                        "B": u["B"],
                        "B_label": B_CODEBOOK.get(u["B"], ""),
                        "text": u["text"],
                        "matched_words": u.get("matched_words", ""),
                    }
                    for u in validation.get("unmatched", [])
                ],
            } if validation else None,
        })

    references_payload = load_references()

    data = {
        "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "codebook": B_CODEBOOK,
        "document_count": len(docs),
        "segment_count": sum(len(d["segments"]) for d in docs),
        "documents": docs,
        "references": (references_payload or {}).get("references", []),
        "references_meta": {
            "method": (references_payload or {}).get("method", ""),
            "source_document_count": (references_payload or {}).get("source_document_count", 0),
            "instrument_count": (references_payload or {}).get("instrument_count", 0),
        } if references_payload else None,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    ref_count = len(data["references"])
    print(f"Wrote {len(docs)} documents ({sum(len(d['segments']) for d in docs)} segments, "
          f"{ref_count} deduplicated references) to {args.output}")
    if skipped:
        print(f"Skipped {skipped} guidelines (no coding output)")
    if references_payload is None:
        print("Note: references/_all_references.json not found; References tab will be empty.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())