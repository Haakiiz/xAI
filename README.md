# Tifa Archivist

A Python 3.11+ CLI tool that searches for Tifa Lockhart images, downloads them,
classifies them with xAI/Grok vision, and sorts them into category folders.

Features:
- DuckDuckGo image search (safesearch off)
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

Edit `config.yaml` or set an environment variable:

```bash
setx XAI_API_KEY "YOUR_KEY"
```

## Run

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset
python -m tifa_archivist stats --out ./tifa_dataset
```

Skip classification (downloads only, saves to `other/`):

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --skip-classify
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
- To change models, update `xai.model_primary` and `xai.model_fallback` in `config.yaml`.
- If you hit DDG rate limits, lower `search_concurrency` or reduce query count.
- To control cost, tune `max_urls_multiplier` (global URL cap = limit * multiplier).
- Only `.jpg/.jpeg/.png` URLs are queued; non-matching content-types are skipped.
- Small or truncated files are rejected via `min_bytes` and Content-Length checks.

## Tests

```bash
pytest -q
```

## Ekstra: analyser bankutskrift med Grok

Det finnes også et enkelt script for å analysere bankutskrift (CSV) og få en
oversikt over sannsynlige abonnementer og andre løpende kostnader:

```bash
python analyze_bank_statement.py ./bankutskrift.csv --env-file ./.env
```

Scriptet forventer `XAI_API_KEY` i miljøet eller `.env`.
