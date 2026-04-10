#!/usr/bin/env python3
"""Build README.md table and index.json from guidelines.yaml.

Checks which files have been downloaded and which texts extracted,
then generates a browsable README and a machine-readable index.
Fails fast if one slug has multiple download candidates, because that
makes the repository state ambiguous.
"""

from __future__ import annotations

import json
from datetime import datetime
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


def build_readme(guidelines: list[dict]) -> str:
    lines: list[str] = []
    lines.append("# AI Guidelines Collection")
    lines.append("")
    lines.append(
        "A curated collection of AI usage guidelines from research institutions, "
        "universities, and public organizations."
    )
    lines.append("")
    lines.append(
        "**Want to add a guideline?** See [CONTRIBUTING.md](CONTRIBUTING.md)."
    )
    lines.append("")

    lines.append("## Guidelines")
    lines.append("")
    lines.append("| Institution | Date | Language | Source | Text |")
    lines.append("| --- | --- | --- | --- | --- |")

    guideline_entries = [entry for entry in guidelines if entry["category"] == "guideline"]
    for entry in sorted(guideline_entries, key=lambda item: item["institution"].lower()):
        institution = entry["institution"]
        date = entry.get("date") or ""
        url = entry["url"]
        slug = entry["slug"]
        language = entry.get("language", "") or ""

        source_link = f"[Source]({url})" if str(url).startswith("http") else str(url)
        txt_path = TEXTS_DIR / f"{slug}.txt"
        text_link = f"[Text](texts/{slug}.txt)" if txt_path.exists() else ""

        notes = entry.get("notes") or ""
        if notes:
            institution = f"{institution} ({notes})"

        lines.append(
            f"| {institution} | {date} | {language} | {source_link} | {text_link} |"
        )

    template_entries = [entry for entry in guidelines if entry["category"] == "template"]
    if template_entries:
        lines.append("")
        lines.append("## Templates & Meta-Resources")
        lines.append("")
        lines.append("| Institution | Date | Notes | Source |")
        lines.append("| --- | --- | --- | --- |")
        for entry in sorted(template_entries, key=lambda item: item["institution"].lower()):
            institution = entry["institution"]
            date = entry.get("date") or ""
            notes = entry.get("notes") or ""
            url = entry["url"]
            source_link = f"[Source]({url})" if str(url).startswith("http") else str(url)
            lines.append(f"| {institution} | {date} | {notes} | {source_link} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d')} "
        f"({len(guidelines)} entries)*"
    )
    lines.append("")
    return "\n".join(lines)


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

    readme = build_readme(guidelines)
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print(f"README.md written ({len(guidelines)} entries)")

    index = build_index(guidelines)
    (ROOT / "index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"index.json written ({len(index)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())