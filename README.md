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

Edit `config.yaml` or set an environment variable:

```bash
setx XAI_API_KEY "YOUR_KEY"
```

If you edit `config.yaml`, make sure it is ignored by git.

## Run

Basic run (PowerShell):

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset
python -m tifa_archivist stats --out ./tifa_dataset
```

Skip classification (downloads only, saves to `other/`):

```bash
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --skip-classify
```

First-time setup example:

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
setx XAI_API_KEY "YOUR_KEY"
python -m tifa_archivist run --limit 20 --out ./tifa_dataset
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
- LLM query generation requires an xAI API key even when `--skip-classify` is used.
- If you hit xAI or DDG rate limits, lower `search_concurrency` or reduce categories.
- To search only specific categories, set `search_categories` in `config.yaml`
  (example: `["wallpaper"]`).
- To control cost, tune `max_urls_multiplier` (global URL cap = limit * multiplier).
- Only `.jpg/.jpeg/.png` URLs are queued; non-matching content-types are skipped.
- Small, corrupted, or truncated files are rejected via `min_bytes`, decode checks, and
  Content-Length validation.
- To reduce pixelated results, raise `min_side` in `config.yaml` (default is 512).

## Tests

```bash
pytest -q
```
