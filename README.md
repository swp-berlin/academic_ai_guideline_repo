# AI Guidelines Repository

This repository collects institutional AI-guideline documents, mostly from the think tank and academic sector, extracts their text, codes them with a shared qualitative codebook (B1-B13), and publishes machine-readable artifacts for analysis and a local browser explorer.

The goal is to create a toolkit for the typical elements of AI guidelines, such as definitions, permitted uses, prohibited uses, scopes, revisions, etc. 

## Status And Validation Caveat

This project is still work in progress.

- The coding pipeline is automated (LLM-driven segmentation and coding).
- Results are not fully manually validated yet.
- Current quality assurance, specifically of the coding part is limited to cursory/manual spot checks plus automated matching/coverage diagnostics.
- Outputs should therefore be treated as provisional research material, not final validated annotations.

## Current Codebook (B1-B13)

Used in `scripts/run_coding.py`:

- `B1`: definition of AI
- `B2`: other definitions/terminology
- `B3`: scope/application of the document
- `B4`: purpose/rationale (why a guideline)/document status
- `B5`: principles/values underlying document or AI use
- `B6`: permitted or encouraged uses of AI
- `B7`: restricted or prohibited uses of AI
- `B8`: required safeguards/procedures for AI
- `B9`: roles/accountability/oversight
- `B10`: risks/limitations/concerns
- `B11`: training/support/learning resources
- `B12`: monitoring/revision/updating
- `B13`: other/not coded/metadata

## Coding Prompt (Transparency)

The coding prompt in `scripts/run_coding.py` currently instructs the model to:

- segment the full document into non-overlapping coherent units,
- assign exactly one primary B code per segment,
- copy segment text verbatim (no paraphrasing),
- avoid repeating shared list preambles,
- and skip obvious metadata/navigation lines.

Current coding guidance in the prompt:

- Prefer `B1`/`B2`/`B3`/`B4` for segments about what the document is, why it exists, or who it addresses.
- Prefer `B10` for risks/concerns and `B8` for concrete safeguards/actions in response.
- Use `B5` for high-level values/principles without concrete actions.
- Use `B9` when responsibility/accountability/oversight roles are assigned.

## What This Repo Contains

- Source metadata in `guidelines.yaml`.
- Downloaded original files in `downloads/` (PDF/HTML/HTM).
- Raw extracted text in `texts/`. (using mistral-ocr where possible)
- LLM-cleaned text in `texts_clean/`.
- Coding outputs (segments, validation, projection) in `outputs/`.
- A repository-wide status catalog in `index.json`.
- A static explorer app in `web/`.

## End-to-End Pipeline

### 1) Validate Metadata

Checks schema, URL safety constraints, slug rules, and duplicates:

```powershell
uv run scripts/validate.py
```

### 2) Download Source Documents

Conservative by default: download one slug unless `--all` is explicitly passed.

```powershell
uv run scripts/download.py --slug universitat-mannheim
```

Optional bulk run:

```powershell
uv run scripts/download.py --all
```

### 3) Extract Text

- PDF: primary path uses Mistral OCR API.
- PDF fallback: PyMuPDF if OCR fails.
- HTML/HTM: BeautifulSoup extraction.

```powershell
uv run scripts/extract_text.py --all
```

### 4) Clean Extracted Text (Optional but Recommended)

Creates cleaned text plus side-by-side HTML diffs for review. The side-by-side diffs are not uploaded to github, but we inspect them manually before continuing with the process of coding them. 

```powershell
uv run scripts/clean_texts.py --all
```

### 5) Run B-Code Segmentation/Coding

Generates one JSON per document in `outputs/` with:

- `segments`: coded units with B code and label.
- `validation`: segment-to-source matching and coverage metrics.
- `projection`: source-text span projection (coded spans plus structural gaps).

Single file:

```powershell
uv run scripts/run_coding.py --file texts_clean/universitat-mannheim.txt
```

Batch folder:

```powershell
uv run scripts/run_coding.py --dir texts_clean
```

### 6) Build Repository Index

Builds `index.json` from `guidelines.yaml` plus file presence checks in `downloads/` and `texts/`.

```powershell
uv run scripts/build_index.py
```

`index.json` fields per entry:

- `slug`, `institution`, `url`, `date`, `category`, `language`, `notes`
- `downloaded`, `download_file`
- `text_extracted`, `text_file`

### 7) Build Explorer Dataset

Combines `guidelines.yaml` and `outputs/*.json` into one explorer payload.

```powershell
uv run scripts/build_explorer_data.py --output web/explorer_data.json
```

## Local Explorer

After generating `web/explorer_data.json`, serve the repository root:

```powershell
cd web/ && python -m http.server 8000
```

Then open:

- `http://localhost:8000/web/`

The explorer expects `explorer_data.json` next to `web/index.html`.

## Environment

The OCR and coding scripts require a Mistral API key.

Create a `.env` in the repo root:

```env
MISTRAL_API_KEY=your_api_key_here
```

Optional overrides used by scripts:

- `MISTRAL_BASE_URL`
- `MISTRAL_MODEL`
- `REQUEST_SLEEP_SECONDS`
- `MAX_RETRIES`
- `OUTPUT_DIR`
- `TEMPERATURE`

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for metadata requirements and slug conventions.
