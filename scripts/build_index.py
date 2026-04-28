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


def find_downloaded_file(slug: str) -> Path | None:
    candidates = sorted(
        p for p in DOWNLOADS_DIR.glob(f"{slug}.*") if p.is_file()
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(candidate.name for candidate in candidates)
        raise RuntimeError(
            f"multiple downloaded files found for '{slug}': {names}; "
            "clean up downloads/ before rebuilding"
        )
    return candidates[0]


def build_index(guidelines: list[dict]) -> list[dict]:
    index: list[dict] = []
    for entry in guidelines:
        slug = entry["slug"]
        download = find_downloaded_file(slug)
        text_path = TEXTS_DIR / f"{slug}.txt"
        index.append(
            {
                "slug": slug,
                "institution": entry["institution"],
                "url": entry["url"],
                "date": entry.get("date"),
                "category": entry["category"],
                "language": entry.get("language"),
                "notes": entry.get("notes"),
                "downloaded": download is not None,
                "download_file": download.name if download else None,
                "text_extracted": text_path.exists(),
                "text_file": f"{slug}.txt" if text_path.exists() else None,
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