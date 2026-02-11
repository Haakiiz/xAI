# Tifa Archivist

A Python 3.11+ CLI tool that searches for Tifa Lockhart images, downloads them,
classifies them with xAI/Grok vision, and sorts them into category folders.

Features:
- LLM-generated search queries + DuckDuckGo image search for URLs
- Async downloads and LLM classification with concurrency limits
- Dedup by URL and sha256 (no image decoding step)
- SQLite tracking for idempotent runs
- Progress bar with URLs found, downloaded, classified, kept, discarded
- NSFW allowed, but anything under 18 or ambiguous is discarded

## Install

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configure

Use `.env` (preferred) or set environment variables:

```bash
setx XAI_API_KEY "YOUR_KEY"
setx GEMINI_API_KEY "YOUR_KEY"
```

`.env` example:

```bash
XAI_API_KEY=your_xai_key
GOOGLE_API_KEY=your_gemini_key
```

If you edit `config.yaml`, make sure it is ignored by git.

## Run

Show CLI help:

```bash
python -m tifa_archivist --help
```

Basic run (PowerShell):

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset
```

Write manifest (default) / disable manifest:

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --manifest
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --no-manifest
```

Skip classification (downloads only, saves to `other/`):

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --skip-classify
```

Dataset stats:

```bash
python -m tifa_archivist stats --out ./tifa_dataset
```

Quality audit (metrics only):

```bash
python -m tifa_archivist audit --out ./tifa_dataset --limit 200 --report QUALITYREPORT.MD --json quality_report.json
```

Quality audit with LLM review:

```bash
python -m tifa_archivist audit --out ./tifa_dataset --limit 200 --llm --llm-limit 20
```

Use a custom config file:

```bash
python -m tifa_archivist --config .\config.yaml run --limit 20
python -m tifa_archivist --config .\config.yaml audit --limit 200
```

## Output

Categories created under the output directory:

- `original_game`
- `wallpaper`
- `cosplay`
- `fanart_2d`
- `render_3d`
- `sexy_sfw`
- `nsfw`
- `other`

Also created:
- `images.db` (SQLite)
- `manifest.json` (optional, enabled by default)

## Notes

- The run is idempotent: previously seen URLs, sha256 hashes, or near-duplicate phashes
  are skipped.
- The classifier discards images with underage or ambiguous subjects.
- To change models, update `gemini.model` for classification and `xai.model_primary`
  for search in `config.yaml`.
- LLM query generation (search) requires an xAI API key even when `--skip-classify`
  is used. Classification uses Gemini.
- If you hit xAI or DDG rate limits, lower `search_concurrency` or reduce categories.
- If you see lots of partial/gray images, lower `download_concurrency` and increase
  `decode_retry_attempts` in `config.yaml`.
- To search only specific categories, set `search_categories` in `config.yaml`
  (example: `["wallpaper"]`).
- To control cost, tune `max_urls_multiplier` (global URL cap = limit * multiplier).
- Only `.jpg/.jpeg/.png` URLs are queued; non-matching content-types are skipped.
- Small or corrupted files are rejected via `min_bytes` and strict decode checks.
- To reduce pixelated results, raise `min_side` in `config.yaml` (default is 512).

## Tests

```bash
pytest -q
```


<!-- codex:review:start -->
## Project Review (Auto)
Purpose: A Python 3.11+ CLI tool that searches for Tifa Lockhart images, downloads them, classifies them with xAI/Grok vision, and sorts them into category folders.
Entry points:
- module: `python -m tifa_archivist`
Install (inferred):
- `pip install -r requirements.txt`
Run (inferred):
- `python -m tifa_archivist`
Tests (inferred):
- `python -m pytest`
Configs:
- `requirements.txt`
Notable dirs:
- `scripts/`
- `tests/`
Dependencies (selected):
- `numpy`
- `pytest`
<!-- codex:review:end -->

Signed-off-by: gpt-5.3-codex — 2026-02-07
