# Contributing

Thanks for helping grow this collection! Adding a new AI guideline is straightforward.

## Adding a new guideline

1. **Fork** this repository and create a branch.
2. **Edit `guidelines.yaml`** and add your entry. Each entry needs these fields:

```yaml
- institution: "Name of the institution"
  url: "https://example.org/path/to/guideline.pdf"
  date: "2025-06-15"       # YYYY-MM-DD, YYYY-MM, or YYYY. Omit if unknown.
  notes: null               # Optional context, e.g. "Covers teaching only"
  category: guideline        # "guideline" or "template"
  slug: name-of-institution  # Lowercase, hyphens only, derived from institution name
  language: de               # ISO 639-1 code (de, en, it, nl, fr, ...)
```

3. **Generate the slug** from the institution name: lowercase, replace spaces and special characters with hyphens, drop accents. Examples:
   - "Deutsche Forschungsgemeinschaft" → `deutsche-forschungsgemeinschaft`
   - "ETH Zürich" → `eth-zurich`
   - "Universität Köln" → `universitat-koln`

4. **Open a Pull Request.** The CI will automatically validate your YAML entry.

## Field reference

| Field | Required | Description |
| --- | --- | --- |
| `institution` | yes | Full name of the publishing institution |
| `url` | yes | Direct link to the PDF or web page |
| `date` | no | Publication date (YYYY-MM-DD preferred) |
| `notes` | no | Short context or description |
| `category` | yes | `guideline` or `template` |
| `slug` | yes | URL-safe identifier, must be unique |
| `language` | no | ISO 639-1 language code |

## What happens after merge

A GitHub Actions workflow will automatically:

1. Download the PDF/HTML from the URL
2. Extract the full text
3. Update the README table and `index.json`

## Categories

- **guideline**: An institution's own AI usage guideline or policy
- **template**: Meta-resources like templates, overviews, or toolkits for creating guidelines

## Quality checks

Before submitting, please verify:

- The URL is a **direct link** to the document (not a landing page with a download button, unless the guideline *is* the web page)
- The document is **publicly accessible** (no login required)
- The institution hasn't already been added (check `guidelines.yaml` or the README)
- Your slug is unique

## Questions?

Open an issue if anything is unclear.
