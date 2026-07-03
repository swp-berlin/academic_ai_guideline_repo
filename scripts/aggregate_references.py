#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx", "python-dotenv"]
# ///
"""Aggregate per-document reference files into one deduplicated master list.

Reads the per-document markdown produced by ``run_references.py`` (one file per
guideline in ``references/``) and combines every referenced instrument into a
single, deduplicated list.

Deduplication runs in layers:
1. Exact/normalized string match (always applied, deterministic): identical
   spellings referenced by several documents collapse into one "surface form".
2. Deterministic folds (always applied): article/section citations
   ("Art. 5 DSGVO", "Anhang III der KI-Verordnung") and parenthetical
   abbreviations ("Datenschutzgrundverordnung (DSGVO)") fold into a parent
   surface form when one already exists.
3. Three-stage LLM merge (default; disable with --no-llm), replacing the old
   single name-only clustering call:
   - Stage 1 (Identify): each surface form is sent with its name, issuer(s) and
     a couple of quotes; the model returns an identity key {instrument, issuer}.
     These keys are grouping handles only (world knowledge allowed) and are
     cached in references/_identity_cache.json so re-runs are incremental.
   - Stage 2 (Group): surface forms are grouped by identical identity key, then
     ONE clustering call over the distinct keys absorbs wording drift between
     batches.
   - Stage 3 (Verify): for every resulting group of 2+ surface forms, a small
     call checks which members do NOT refer to the same underlying instrument
     as the majority (e.g. different universities' statutes) and splits them out.

Anti-hallucination contract (unchanged): the DISPLAYED canonical name and
variants are always verbatim source spellings. LLM identity keys are used only
to decide grouping; they never appear in the output. Any surface form the model
drops or alters survives as its own group, so nothing is lost or fabricated.

Usage:
    uv run scripts/aggregate_references.py                 # references/ -> references/_all_references.md
    uv run scripts/aggregate_references.py --no-llm        # deterministic string dedup only
    uv run scripts/aggregate_references.py --input references --output references/_all_references.md

Reads MISTRAL_API_KEY (and optionally MISTRAL_MODEL, MISTRAL_BASE_URL,
MAX_RETRIES, TEMPERATURE) from .env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


SYSTEM_PROMPT = (
    "You are a meticulous policy-document analyst. You return only valid JSON and "
    "follow the output schema exactly."
)

# ---------------------------------------------------------------------------
# Stage 1 — Identify: assign each surface form an {instrument, issuer} key.
# ---------------------------------------------------------------------------
IDENTIFY_SYSTEM_PROMPT = (
    "You are a meticulous legal/policy analyst who identifies which underlying "
    "instrument (a law, regulation, framework, guideline, code of conduct or "
    "standard) a reference points to. Return only valid JSON."
)

IDENTIFY_PROMPT_TEMPLATE = """
You are given references to documents/instruments extracted from many academic
AI-guideline documents. Each item has a verbatim NAME (possibly an abbreviation,
a partial title, or in German), optionally an ISSUER stated in the source, and a
few QUOTES showing how it was referenced.

For EACH item, return an identity key that lets us group different wordings of
the SAME underlying instrument together while keeping genuinely different
instruments apart. You MAY use world knowledge; the key is an internal grouping
handle and does NOT need to be verbatim.

Return, for each item:
- "instrument": the official SHORT name of the instrument. Use the established
  ENGLISH short name when one exists (e.g. "EU AI Act", "GDPR"). Append the
  official identifier in parentheses when you know it (e.g.
  "EU AI Act (Regulation (EU) 2024/1689)", "GDPR (Regulation (EU) 2016/679)").
  Use the SAME string for every wording of the same instrument, regardless of
  language or abbreviation.
- "issuer": the issuing organization or jurisdiction (e.g. "European Union",
  "DFG", "Universität Ulm", "State of Colorado"), or null if it cannot be
  determined. IMPORTANT: institution-specific documents (a single university's
  statute, a specific institute's guidelines) MUST carry that institution as the
  issuer so two different institutions' documents get different keys. If the
  stated issuer is empty and you cannot confidently infer one, use null and do
  NOT guess a specific institution.

CONVENTIONS (follow to reduce drift):
- Same instrument -> byte-identical "instrument" string across items.
- Two different national/state laws, or two different institutions' own
  policies, are DIFFERENT instruments and must get different keys.
- A generic phrase with no specific named document (e.g. "applicable data
  protection law") should get instrument equal to a short normalized form of the
  name itself and issuer null (it will simply stay on its own).

--- INPUT ITEMS (JSON) ---
{items_json}
--- END INPUT ITEMS ---

Return ONLY a JSON object of the form:
{{"results": [{{"id": <int>, "instrument": <string>, "issuer": <string or null>}}, ...]}}
with exactly one result per input id. No commentary, no markdown fences.
""".strip()


# ---------------------------------------------------------------------------
# Stage 2 — Group: cluster the distinct identity keys (absorb wording drift).
# ---------------------------------------------------------------------------
CLUSTER_PROMPT_TEMPLATE = """
Below is a JSON list of identity keys of the form "<instrument> || <issuer>".
Each names an underlying instrument (a law, regulation, framework, guideline,
code of conduct or standard). Because they were produced in separate batches,
the SAME instrument may appear under slightly different wordings.

TASK:
Find EVERY set of keys that refer to the SAME underlying instrument (same issuer
too) and put them in one group. Merge across:
- spelling / punctuation / abbreviation differences in the instrument part;
- an official long title vs a short name for the same instrument;
- trivially different issuer wordings for the same body ("EU" = "European Union").

STRICT RULES:
- Use ONLY the keys given in the input list, copied verbatim. Do NOT invent,
  translate, reword or expand a key.
- Output ONLY groups that contain TWO OR MORE keys. Any key you leave out is
  automatically kept on its own, so never list a singleton.
- Each key may appear in at most ONE group.
- Do NOT merge instruments that are genuinely different: two different national
  or state laws (e.g. an EU law vs a US state law), or two different
  institutions' own policies, even if the instrument words look similar.

OUTPUT:
Return one JSON object mapping "groups" to a list of groups, where each group is
a list of two or more verbatim input keys:
{{"groups": [[string, string, ...], ...]}}.

--- INPUT KEYS (JSON) ---
{names_json}
--- END INPUT KEYS ---

Return ONLY the JSON object. No commentary and no markdown fences.
""".strip()


# ---------------------------------------------------------------------------
# Stage 3 — Verify: split members that do not match the group's majority.
# ---------------------------------------------------------------------------
VERIFY_SYSTEM_PROMPT = (
    "You are a precise legal/policy analyst. You decide whether several "
    "references all point to the exact same underlying document. Return only "
    "valid JSON."
)

VERIFY_PROMPT_TEMPLATE = """
The following references were provisionally grouped as the SAME underlying
instrument. Check the grouping. Different institutions' own statutes/policies
(e.g. two universities' "Satzung zur Sicherung guter wissenschaftlicher Praxis")
are DIFFERENT documents and must NOT stay together.

For each member you are given its verbatim NAME, the ISSUER stated in the source
(may be "unknown"), and one QUOTE.

Decide which members do NOT refer to the same underlying document as the
MAJORITY of the group. List their ids. Members with an unknown issuer that could
plausibly be the majority instrument should be KEPT (do not split on uncertainty
alone); split a member only when its name/issuer/quote shows it is a different
document (e.g. a different institution, a different jurisdiction).

--- GROUP MEMBERS (JSON) ---
{members_json}
--- END GROUP MEMBERS ---

Return ONLY a JSON object:
{{"different": [<id>, ...]}}
listing the ids that should be split out as their own instruments (empty list if
they all match). No commentary, no markdown fences.
""".strip()


HEADING_RE = re.compile(r"^##\s+\d+\.\s+(.*)$")
FIELD_RE = re.compile(r"^-\s+\*\*(.+?):\*\*\s*(.*)$")

# Values the per-document renderer uses to mean "absent".
ABSENT_VALUES = {"", "none", "not stated"}

# Up to this many example quotes are kept per aggregated instrument.
MAX_EXAMPLE_MENTIONS = 4

# Surface forms per Stage-1 identify call.
IDENTIFY_BATCH_SIZE = 25
# Quotes shown to the model per surface form (Stage 1) / group member (Stage 3).
MAX_QUOTES_PER_ITEM = 2
# Sensible output budget for the small Stage 1/2/3 calls (they are tiny).
SMALL_MAX_TOKENS = 8192
# The Stage-2 clustering call can echo back a few dozen groups; give it more
# headroom so its JSON is never truncated (truncation -> parse error -> a needless
# fall back to un-clustered identity groups).
CLUSTER_MAX_TOKENS = 16384

CACHE_FILENAME = "_identity_cache.json"


# --- Deterministic article/section-citation folding -----------------------
# A leading legal-citation prefix, e.g. "Art. 5", "Artikel 28 Abs. 3 der",
# "Article 22 of the", "§ 60d", "Sec. 5", "Anhang III der",
# "Erwägungsgrund 25", "Art. 6, Anlage 1 und 3 der". Used to fold
# article/section/annex/recital-level references into their parent instrument
# (see fold_article_references).
_CITE_TOKEN = (
    r"(?:art(?:ikel|icle)?\.?|§{1,2}|sec(?:tion)?\.?|ziff(?:er)?\.?|nr\.?"
    r"|abs(?:atz)?\.?|para(?:graph)?\.?|lit\.?|satz"
    r"|anh(?:ang)?\.?|annex|anlage|appendix"
    r"|erw(?:\.|ägungsgrund|aegungsgrund)?|recital)"
)
# An article/annex number: arabic ("5", "60d") or a standalone roman numeral
# ("III"). The roman branch must not be followed by another letter so it never
# eats into a base name like "DSGVO".
_CITE_NUMBER = r"(?:\d+[a-z]?|(?=[ivxlcdm])[ivxlcdm]+(?![a-z]))"
_CITE_SUBPART = (
    r"(?:\s*(?:abs\.?|absatz|para\.?|paragraph|lit\.?|satz|nr\.?|s\.?|ff?\.?)\s*"
    + _CITE_NUMBER + r"?)*"
)
_CITE_UNIT = _CITE_TOKEN + r"(?:\s*" + _CITE_NUMBER + r")?" + _CITE_SUBPART
# Separators inside a multi-citation list: "Art. 6, Anlage 1 und 3 der ...".
_CITE_SEP = r"(?:\s*[,;]\s*|\s+(?:und|and|sowie|bis|to)\s+|\s*[&–—]\s*)"
ARTICLE_PREFIX_RE = re.compile(
    r"^(?:" + _CITE_UNIT + r")"
    r"(?:" + _CITE_SEP + r"(?:" + _CITE_UNIT + r"|" + _CITE_NUMBER + r"))*"
    r"\s*(?:der|des|of\s+the|of|the|zur|zum|im)?\s+",  # optional connector
    re.IGNORECASE,
)


def article_base(name: str) -> str | None:
    """If ``name`` is an article/section/annex/recital citation (e.g. "Art. 5
    DSGVO", "Anhang III der KI-Verordnung"), return the base instrument name
    ("DSGVO", "KI-Verordnung"); otherwise None."""
    m = ARTICLE_PREFIX_RE.match(name.strip())
    if not m:
        return None
    base = name.strip()[m.end():].strip()
    return base or None


# "Long Name (ABBR)" -> ("ABBR", "Long Name"). Abbreviation first: it is the more
# likely standalone reference to fold into.
PAREN_ABBR_RE = re.compile(r"^(.+?)\s*\(([^()]{2,}?)\)\s*$")


def paren_variants(name: str) -> tuple[str, str] | None:
    """For "Long Name (ABBR)" return ("ABBR", "Long Name"); otherwise None."""
    m = PAREN_ABBR_RE.match(name.strip())
    if not m:
        return None
    long, abbr = m.group(1).strip(), m.group(2).strip()
    if not long or not abbr:
        return None
    return abbr, long


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
    """Collapse exact/normalized-identical names into surface forms, tracking every
    raw spelling, the documents that use each, the stated issuer(s), and the
    (doc-attributed) mentions."""
    forms: dict[str, dict[str, Any]] = {}
    for e in entries:
        key = norm_key(e["name"])
        form = forms.get(key)
        if form is None:
            form = {
                "name": e["name"],          # representative spelling (fed to the LLM)
                "name_docs": {},             # raw spelling -> set of doc_ids using it
                "type_counts": {},
                "issuer_counts": {},         # stated issuer string -> count
                "doc_ids": set(),
                "mentions": [],              # list of {"doc_id":, "quote":}
                "_seen_mentions": set(),     # (doc_id, norm quote) for dedup
            }
            forms[key] = form
        form["name_docs"].setdefault(e["name"], set()).add(e["doc_id"])
        form["type_counts"][e["type"]] = form["type_counts"].get(e["type"], 0) + 1
        if e.get("issuer"):
            form["issuer_counts"][e["issuer"]] = form["issuer_counts"].get(e["issuer"], 0) + 1
        form["doc_ids"].add(e["doc_id"])
        add_mention(form, e["doc_id"], e.get("mention"))
    return forms


def add_mention(form: dict[str, Any], doc_id: str, quote: str | None) -> None:
    """Append a doc-attributed quote to a surface form, deduping by (doc, text)."""
    if not quote:
        return
    mk = (doc_id, norm_key(quote))
    if mk in form["_seen_mentions"]:
        return
    form["_seen_mentions"].add(mk)
    form["mentions"].append({"doc_id": doc_id, "quote": quote})


def merge_form(target: dict[str, Any], form: dict[str, Any]) -> None:
    """Fold ``form`` into ``target`` (spellings, doc_ids, types, issuers, mentions)."""
    for nm, ds in form["name_docs"].items():
        target["name_docs"].setdefault(nm, set()).update(ds)
    target["doc_ids"] |= form["doc_ids"]
    for t, c in form["type_counts"].items():
        target["type_counts"][t] = target["type_counts"].get(t, 0) + c
    for iss, c in form.get("issuer_counts", {}).items():
        target["issuer_counts"][iss] = target["issuer_counts"].get(iss, 0) + c
    for m in form["mentions"]:
        add_mention(target, m["doc_id"], m["quote"])


def form_issuers(form: dict[str, Any], limit: int = 2) -> list[str]:
    """Distinct stated issuers for a surface form, most frequent first."""
    counts = form.get("issuer_counts", {})
    return [
        iss for iss, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].casefold()))
    ][:limit]


def fold_article_references(forms: dict[str, dict[str, Any]]) -> None:
    """Deterministically fold article/section/annex/recital citations into their
    parent instrument, e.g. "Art. 5 DSGVO" / "Anhang III der KI-Verordnung" ->
    "DSGVO" / "KI-Verordnung", but only when the parent already exists as its own
    surface form. The citation spelling survives as a variant; nothing is
    invented. Mutates ``forms``."""
    for key in list(forms.keys()):
        form = forms.get(key)
        if form is None:
            continue
        base = article_base(form["name"])
        if not base:
            continue
        base_key = norm_key(base)
        target = forms.get(base_key)
        if target is None or base_key == key:
            continue
        merge_form(target, form)
        del forms[key]


def fold_parenthetical_abbreviations(forms: dict[str, dict[str, Any]]) -> None:
    """Deterministically fold "Long Name (ABBR)" into a standalone "ABBR" (or
    "Long Name") surface form when one exists, e.g. "Datenschutzgrundverordnung
    (DSGVO)" -> "DSGVO", "KI-Verordnung (KI-VO)" -> "KI-VO". Only folds into a name
    that is itself present as its own reference, so no knowledge is hardcoded and
    generic parentheticals stay put. Mutates ``forms``."""
    for key in list(forms.keys()):
        form = forms.get(key)
        if form is None:
            continue
        parts = paren_variants(form["name"])
        if not parts:
            continue
        for cand in parts:  # prefer the abbreviation, then the long form
            cand_key = norm_key(cand)
            if cand_key == key:
                continue
            target = forms.get(cand_key)
            if target is not None:
                merge_form(target, form)
                del forms[key]
                break


def select_mentions(
    mentions: list[dict[str, str]], limit: int
) -> list[dict[str, str]]:
    """Pick up to ``limit`` example quotes, preferring spread across documents
    (at most one per document first, then fill from the rest)."""
    chosen: list[dict[str, str]] = []
    used_docs: set[str] = set()
    for m in mentions:
        if len(chosen) >= limit:
            return chosen
        if m["doc_id"] not in used_docs:
            chosen.append(m)
            used_docs.add(m["doc_id"])
    for m in mentions:
        if len(chosen) >= limit:
            break
        if m not in chosen:
            chosen.append(m)
    return chosen


def dominant_type(type_counts: dict[str, int]) -> str:
    if not type_counts:
        return "other"
    return max(type_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


# --- Mistral client -------------------------------------------------------
@dataclass
class LLMConfig:
    client: httpx.Client
    base_url: str
    api_key: str
    model: str
    temperature: float
    max_retries: int
    # per-stage call counters (cold calls only; cache hits are not counted)
    calls: dict[str, int] = field(default_factory=lambda: {"identify": 0, "cluster": 0, "verify": 0})


def call_mistral(
    cfg: LLMConfig,
    prompt: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
    max_tokens: int = 8192,
) -> str:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": cfg.temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    url = f"{cfg.base_url}/chat/completions"

    for attempt in range(cfg.max_retries + 1):
        try:
            r = cfg.client.post(url, headers=headers, json=payload)
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
            if e.response.status_code in {429, 500, 502, 503, 504} and attempt < cfg.max_retries:
                wait = 2 * (attempt + 1)
                print(f"  HTTP {e.response.status_code}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            raise
        except Exception:
            if attempt < cfg.max_retries:
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise RuntimeError("Exhausted retries")


def parse_json_object(raw: str) -> Any:
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# --- Identity cache -------------------------------------------------------
def load_cache(path: Path, model: str) -> dict[str, Any]:
    """Load the identity/cluster/verify cache. Cache entries are invalidated when
    the model changes (a different model may key things differently)."""
    default = {"model": model, "identities": {}, "clusters": {}, "verify": {}}
    if not path.exists():
        return default
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if not isinstance(data, dict) or data.get("model") != model:
        return default
    for k in ("identities", "clusters", "verify"):
        if not isinstance(data.get(k), dict):
            data[k] = {}
    data["model"] = model
    return data


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hash(obj: Any) -> str:
    return hashlib.sha1(
        json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


# --- Stage 1: Identify ----------------------------------------------------
def identify_forms(
    forms: dict[str, dict[str, Any]],
    cfg: LLMConfig,
    cache: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Assign every surface form an identity key {instrument, issuer}. Cached by
    normalized surface name in ``cache['identities']``; only uncached names hit
    the model. Returns {form_key -> {"instrument":, "issuer":}}."""
    identities: dict[str, dict[str, Any]] = {}
    todo: list[tuple[str, dict[str, Any]]] = []

    for fkey, form in forms.items():
        cached = cache["identities"].get(fkey)
        if isinstance(cached, dict) and cached.get("instrument"):
            identities[fkey] = {
                "instrument": cached["instrument"],
                "issuer": cached.get("issuer"),
            }
        else:
            todo.append((fkey, form))

    print(
        f"Stage 1 (identify): {len(forms)} surface forms, "
        f"{len(forms) - len(todo)} cached, {len(todo)} to query."
    )

    for start in range(0, len(todo), IDENTIFY_BATCH_SIZE):
        batch = todo[start:start + IDENTIFY_BATCH_SIZE]
        items = []
        for i, (_fkey, form) in enumerate(batch):
            items.append({
                "id": i,
                "name": form["name"],
                "issuer": form_issuers(form) or None,
                "quotes": [m["quote"] for m in form["mentions"][:MAX_QUOTES_PER_ITEM]],
            })
        prompt = IDENTIFY_PROMPT_TEMPLATE.format(
            items_json=json.dumps(items, ensure_ascii=False, indent=2)
        )
        cfg.calls["identify"] += 1
        by_id: dict[int, dict[str, Any]] = {}
        batch_ok = True
        try:
            raw = call_mistral(
                cfg, prompt,
                system_prompt=IDENTIFY_SYSTEM_PROMPT,
                max_tokens=SMALL_MAX_TOKENS,
            )
            parsed = parse_json_object(raw)
            results = parsed.get("results", []) if isinstance(parsed, dict) else []
            for res in results:
                if isinstance(res, dict) and isinstance(res.get("id"), int):
                    by_id[res["id"]] = res
        except Exception as e:
            # A single flaky batch must not sink the whole run: fall back to
            # name-as-instrument for this batch and DON'T cache it, so a re-run
            # retries just these names.
            print(f"  identify batch failed ({e}); keeping names as singletons "
                  "for this batch (will retry on re-run).", file=sys.stderr)
            batch_ok = False

        for i, (fkey, form) in enumerate(batch):
            res = by_id.get(i)
            instrument = None
            issuer = None
            if isinstance(res, dict):
                inst = res.get("instrument")
                if isinstance(inst, str) and inst.strip():
                    instrument = normalize(inst)
                iss = res.get("issuer")
                if isinstance(iss, str) and iss.strip() and iss.strip().lower() not in ABSENT_VALUES:
                    issuer = normalize(iss)
            # Defensive fallback: if the model dropped/garbled an item, key it by
            # its own name so it stays a singleton (nothing is lost).
            if not instrument:
                instrument = form["name"]
            entry = {"instrument": instrument, "issuer": issuer}
            identities[fkey] = entry
            # Only cache genuine model answers, so a failed batch is retried.
            if batch_ok and res is not None:
                cache["identities"][fkey] = entry

    return identities


def identity_display(entry: dict[str, Any]) -> str:
    """Human-readable grouping key: "<instrument> || <issuer>"."""
    return f"{entry['instrument']} || {entry.get('issuer') or '—'}"


# --- Generic clustering (shared validation) -------------------------------
def cluster_names(
    names: list[str],
    cfg: LLMConfig,
    cache: dict[str, Any],
) -> list[list[str]]:
    """Ask the model to cluster ``names`` (identity keys) into merge-groups, then
    validate that the result only regroups the given names (nothing invented,
    nothing lost). Cached by the set of input names."""
    by_key = {norm_key(n): n for n in names}
    cache_key = _hash(sorted(by_key))
    cached = cache["clusters"].get(cache_key)
    if isinstance(cached, list):
        parsed = {"groups": cached}
    else:
        prompt = CLUSTER_PROMPT_TEMPLATE.format(
            names_json=json.dumps(names, ensure_ascii=False, indent=2)
        )
        cfg.calls["cluster"] += 1
        raw = call_mistral(cfg, prompt, max_tokens=CLUSTER_MAX_TOKENS)
        parsed = parse_json_object(raw)

    assigned: set[str] = set()
    groups: list[list[str]] = []
    validated_groups: list[list[str]] = []

    raw_groups = parsed.get("groups", []) if isinstance(parsed, dict) else []
    for grp in raw_groups:
        if isinstance(grp, dict):
            members_in = grp.get("variants", [])
        elif isinstance(grp, list):
            members_in = grp
        else:
            continue
        members: list[str] = []
        member_keys: list[str] = []
        seen_local: set[str] = set()
        for variant in members_in:
            key = norm_key(str(variant))
            # verbatim + not used by an earlier kept group + not repeated here
            if key in by_key and key not in assigned and key not in seen_local:
                seen_local.add(key)
                members.append(by_key[key])
                member_keys.append(key)
        # Only a real merge (2+ members) consumes its keys. A 1-member or invalid
        # "group" must NOT mark its key assigned, or that key would vanish from
        # both the output and the cache -> non-reproducible singleton handling.
        if len(members) >= 2:
            assigned.update(member_keys)
            groups.append(members)
            validated_groups.append(members)

    # Persist the validated (already sanitized) merge-groups for reproducibility.
    if not isinstance(cached, list):
        cache["clusters"][cache_key] = validated_groups

    # Every key the model did not group survives as its own singleton group.
    for key, name in by_key.items():
        if key not in assigned:
            groups.append([name])

    return groups


# --- Stage 3: Verify ------------------------------------------------------
def verify_group(
    member_forms: list[dict[str, Any]],
    cfg: LLMConfig,
    cache: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    """Precision gate: given a provisional group of 2+ surface forms, ask which do
    NOT match the majority instrument and split those into their own singletons.
    Parses defensively; on any malformed/failed response the group is kept
    as-is. Cached by the set of member names + issuers + one quote."""
    members = []
    for i, form in enumerate(member_forms):
        issuers = form_issuers(form)
        quote = form["mentions"][0]["quote"] if form["mentions"] else ""
        members.append({
            "id": i,
            "name": form["name"],
            "issuer": issuers[0] if issuers else "unknown",
            "quote": quote,
        })

    cache_key = _hash(members)
    cached = cache["verify"].get(cache_key)
    if isinstance(cached, list):
        different = cached
    else:
        prompt = VERIFY_PROMPT_TEMPLATE.format(
            members_json=json.dumps(members, ensure_ascii=False, indent=2)
        )
        cfg.calls["verify"] += 1
        try:
            raw = call_mistral(
                cfg, prompt,
                system_prompt=VERIFY_SYSTEM_PROMPT,
                max_tokens=SMALL_MAX_TOKENS,
            )
            parsed = parse_json_object(raw)
            raw_diff = parsed.get("different", []) if isinstance(parsed, dict) else []
            different = [d for d in raw_diff if isinstance(d, int) and 0 <= d < len(member_forms)]
        except Exception as e:
            # On a malformed/failed verify response, keep the group intact.
            print(f"  verify call failed ({e}); keeping group as-is.", file=sys.stderr)
            different = []
        cache["verify"][cache_key] = different

    if not different:
        return [member_forms]

    diff_set = set(different)
    kept = [f for i, f in enumerate(member_forms) if i not in diff_set]
    result: list[list[dict[str, Any]]] = []
    if kept:
        result.append(kept)
    for i in sorted(diff_set):
        result.append([member_forms[i]])
    return result


def llm_group_forms(
    forms: dict[str, dict[str, Any]],
    cfg: LLMConfig,
    cache: dict[str, Any],
) -> list[list[str]]:
    """Run the three-stage LLM merge and return groups as lists of surface names
    (the representative spelling of each form), ready for build_master_list."""
    # Stage 1: identify.
    identities = identify_forms(forms, cfg, cache)

    # Stage 2a: group surface forms by identical identity key.
    key_norm_to_display: dict[str, str] = {}
    key_norm_to_forms: dict[str, list[dict[str, Any]]] = {}
    for fkey, form in forms.items():
        display = identity_display(identities[fkey])
        nk = norm_key(display)
        key_norm_to_display.setdefault(nk, display)
        key_norm_to_forms.setdefault(nk, []).append(form)

    distinct_keys = sorted(key_norm_to_display.values(), key=str.casefold)
    print(
        f"Stage 2 (group): {len(forms)} forms -> {len(distinct_keys)} distinct "
        f"identity keys; clustering to absorb wording drift."
    )

    # Stage 2b: cluster the distinct identity keys to absorb wording drift. If
    # this single call fails, degrade gracefully to the (deterministic) exact
    # identity-key grouping rather than losing everything -- AI Act and DSGVO are
    # already unified by their exact keys at this point.
    try:
        key_clusters = cluster_names(distinct_keys, cfg, cache)
    except Exception as e:
        print(f"  clustering call failed ({e}); using exact identity-key grouping "
              "only (no cross-key merge).", file=sys.stderr)
        key_clusters = [[k] for k in distinct_keys]

    provisional_groups: list[list[dict[str, Any]]] = []
    for cluster in key_clusters:
        group_forms: list[dict[str, Any]] = []
        for display in cluster:
            group_forms.extend(key_norm_to_forms.get(norm_key(display), []))
        if group_forms:
            provisional_groups.append(group_forms)

    multi = [g for g in provisional_groups if len(g) >= 2]
    print(
        f"Stage 3 (verify): {len(provisional_groups)} provisional groups "
        f"({len(multi)} with 2+ members) to verify."
    )

    # Stage 3: verify each multi-member group; split mismatches.
    final_groups: list[list[dict[str, Any]]] = []
    for group_forms in provisional_groups:
        if len(group_forms) < 2:
            final_groups.append(group_forms)
            continue
        final_groups.extend(verify_group(group_forms, cfg, cache))

    return [[f["name"] for f in g] for g in final_groups]


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
        name_docs: dict[str, set[str]] = {}   # spelling -> set of doc_ids using it
        mentions: list[dict[str, str]] = []
        seen_mentions: set[tuple[str, str]] = set()
        for f in member_forms:
            doc_ids |= f["doc_ids"]
            for t, c in f["type_counts"].items():
                type_counts[t] = type_counts.get(t, 0) + c
            for nm, ds in f["name_docs"].items():
                name_docs.setdefault(nm, set()).update(ds)
            for m in f["mentions"]:
                mk = (m["doc_id"], norm_key(m["quote"]))
                if mk not in seen_mentions:
                    seen_mentions.add(mk)
                    mentions.append(m)

        # Canonical = spelling used by the most documents, then the longest.
        canonical = max(name_docs, key=lambda nm: (len(name_docs[nm]), len(nm)))
        # Order variants: canonical first, rest by descending document usage.
        variants = sorted(
            name_docs,
            key=lambda v: (v != canonical, -len(name_docs[v]), v.casefold()),
        )
        example_mentions = select_mentions(mentions, MAX_EXAMPLE_MENTIONS)

        master.append(
            {
                "canonical_name": canonical,
                "variants": variants,
                "type": dominant_type(type_counts),
                "doc_ids": sorted(doc_ids),
                "doc_count": len(doc_ids),
                "example_mentions": example_mentions,
                # Kept for backward compatibility with older consumers.
                "example_mention": example_mentions[0]["quote"] if example_mentions else None,
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
        mentions = e.get("example_mentions") or []
        if mentions:
            label = "Example mention" if len(mentions) == 1 else "Example mentions"
            lines.append(f"- **{label}:**")
            for m in mentions:
                lines.append(f"  - _{m['doc_id']}_: \"{m['quote']}\"")
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
        help="Deterministic string dedup only; skip the LLM merge stages",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if an LLM stage errors, instead of falling back to whatever "
             "grouping was already established",
    )
    parser.add_argument(
        "--cache", type=Path, default=None,
        help="Identity/cluster/verify cache file (default: <input>/_identity_cache.json)",
    )
    args = parser.parse_args()

    output_path = args.output or (args.input / "_all_references.md")
    cache_path = args.cache or (args.input / CACHE_FILENAME)

    if not args.input.is_dir():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        return 1

    entries = collect_entries(args.input, output_path)
    if not entries:
        print(f"No reference entries found in {args.input}", file=sys.stderr)
        return 1

    source_count = len({e["doc_id"] for e in entries})
    forms = build_surface_forms(entries)
    before_fold = len(forms)
    fold_article_references(forms)
    fold_parenthetical_abbreviations(forms)
    folded = before_fold - len(forms)
    surface_names = [f["name"] for f in forms.values()]
    print(f"Parsed {len(entries)} entries from {source_count} documents "
          f"({before_fold} distinct spellings; folded {folded} article/section and "
          f"parenthetical-abbreviation references into their parent instrument).")

    # Deterministic baseline: one group per surviving surface form. Used both for
    # --no-llm and as the graceful-fallback grouping if an LLM stage fails.
    groups = [[n] for n in surface_names]
    method = "exact string match (no semantic merge)"

    if not args.no_llm:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        if not api_key:
            print("MISTRAL_API_KEY not found (set it in .env or use --no-llm)", file=sys.stderr)
            return 1
        base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1").rstrip("/")
        model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        temperature = float(os.getenv("TEMPERATURE", "0"))
        max_retries = int(os.getenv("MAX_RETRIES", "2"))

        cache = load_cache(cache_path, model)
        try:
            with httpx.Client(timeout=6400.0) as client:
                cfg = LLMConfig(
                    client=client, base_url=base_url, api_key=api_key, model=model,
                    temperature=temperature, max_retries=max_retries,
                )
                groups = llm_group_forms(forms, cfg, cache)
            method = f"3-stage LLM merge (identify/group/verify) via {model}"
            print(
                f"Mistral calls -> identify: {cfg.calls['identify']}, "
                f"cluster: {cfg.calls['cluster']}, verify: {cfg.calls['verify']}"
            )
        except Exception as e:
            if args.strict:
                print(f"LLM merge failed: {e}", file=sys.stderr)
                # Still persist any cache progress before exiting.
                try:
                    save_cache(cache_path, cache)
                except Exception:
                    pass
                return 1
            print(
                f"LLM merge failed ({e}); falling back to deterministic string dedup. "
                "Re-run to retry (cached progress is reused), or pass --strict to fail.",
                file=sys.stderr,
            )
            groups = [[n] for n in surface_names]
            method = "exact string match (LLM merge unavailable — call failed)"
        finally:
            # Persist cache progress (identities/clusters/verify) even on partial runs.
            try:
                save_cache(cache_path, cache)
            except Exception as e:
                print(f"  warning: could not write cache {cache_path}: {e}", file=sys.stderr)

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
