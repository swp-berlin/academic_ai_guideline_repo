#!/usr/bin/env python3
"""Build index.json from guidelines.yaml.

Checks which files have been downloaded and which texts extracted,
then generates a machine-readable index.
Fails fast if one slug has multiple download candidates, because that
makes the repository state ambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DOWNLOADS_DIR = ROOT / "downloads"
TEXTS_DIR = ROOT / "texts"
GUIDELINES_FILE = ROOT / "guidelines.yaml"


def find_versioned(directory: Path, slug: str, version: str,
                   suffixes: set[str] | None = None) -> Path | None:
    """Exact (slug, version) file in a directory: {dir}/{slug}_{version}.*"""
    stem = f"{slug}_{version}"
    cands = sorted(
        p for p in directory.glob(f"{stem}.*")
        if p.is_file() and (suffixes is None or p.suffix.lower() in suffixes)
    )
    if not cands:
        return None
    if len(cands) > 1:
        names = ", ".join(c.name for c in cands)
        raise RuntimeError(
            f"multiple files for '{stem}' in {directory.name}/: {names}; clean up first"
        )
    return cands[0]


def build_index(guidelines: list[dict]) -> list[dict]:
    index: list[dict] = []
    for entry in guidelines:
        slug = entry["slug"]
        version = str(entry.get("version") or "1_0").replace(".", "_")
        download = find_versioned(DOWNLOADS_DIR, slug, version)
        text_file = find_versioned(TEXTS_DIR, slug, version, {".txt"})
        index.append(
            {
                "slug": slug,
                "institution": entry["institution"],
                "url": entry["url"],
                "date": entry.get("date"),
                "version": entry.get("version"),
                "category": entry["category"],
                "language": entry.get("language"),
                "notes": entry.get("notes"),
                "downloaded": download is not None,
                "download_file": download.name if download else None,
                "text_extracted": text_file is not None,
                "text_file": text_file.name if text_file else None,
            }
        )
    return index


def main() -> int:
    with open(GUIDELINES_FILE, encoding="utf-8") as f:
        guidelines = yaml.safe_load(f)

    if not isinstance(guidelines, list):
        raise ValueError("guidelines.yaml must contain a YAML list")

    index = build_index(guidelines)
    (ROOT / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"index.json written ({len(index)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())