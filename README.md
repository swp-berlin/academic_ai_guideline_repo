# AI Guidelines Repository

This repository collects institutional AI-guideline documents, extracts their text, codes them with a shared qualitative codebook (B1-B11), and publishes machine-readable artifacts for analysis and a local browser explorer.

## What This Repo Contains

- Source metadata in `guidelines.yaml`.
- Downloaded original files in `downloads/` (PDF/HTML/HTM).
- Raw extracted text in `texts/`.
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
uv run scripts/extract_text.py --slug universitat-mannheim
```

### 4) Clean Extracted Text (Optional but Recommended)

Creates cleaned text plus side-by-side HTML diffs for review.

```powershell
uv run scripts/clean_texts.py --slug universitat-mannheim
```

### 5) Run B-Code Segmentation/Coding

Generates one JSON per document in `outputs/` with:

- `segments`: coded units with B code and label.
- `validation`: segment-to-source matching and coverage metrics.
- `projection`: source-text span projection (coded spans plus structural gaps).

Single file:

```powershell
uv run scripts/run_coding.py --file texts/universitat-mannheim.txt
```

Batch folder:

```powershell
uv run scripts/run_coding.py --dir texts
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
python -m http.server 8000
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
