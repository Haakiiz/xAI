# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**tifa_archivist** — a Python 3.11+ CLI that searches DuckDuckGo for Tifa Lockhart images, downloads them asynchronously, deduplicates by URL/sha256/phash, classifies them with the xAI/Grok vision API, and sorts them into category folders under an output directory. All state is persisted to a SQLite database (`images.db`) so runs are idempotent.

There is also **`tifa.py`** — a standalone script that generates new images via the xAI Grok Imagine API, configured entirely through `tifa.yaml`.

## Commands

```powershell
# Install (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run pipeline
python -m tifa_archivist run --limit 20 --out ./tifa_dataset
python -m tifa_archivist run --limit 20 --out ./tifa_dataset --skip-classify

# Show dataset stats
python -m tifa_archivist stats --out ./tifa_dataset

# Quality audit (CV metrics; optionally call Gemini LLM)
python -m tifa_archivist audit --out ./tifa_dataset
python -m tifa_archivist audit --out ./tifa_dataset --llm   # needs GEMINI_API_KEY

# Run all tests
pytest -q

# Run a single test file
pytest tests/test_orchestrator_smoke.py -q

# Generate images with Grok Imagine (edit tifa.yaml first)
python tifa.py
python tifa.py --prompt "custom prompt"
python tifa.py --config other.yaml
```

API keys: set `XAI_API_KEY` (and optionally `GEMINI_API_KEY`) as environment variables, or put them in `config.yaml` under `xai.api_key` / `gemini.api_key`.

## Architecture

### Pipeline flow (`orchestrator.py`)

`run_pipeline` wires everything together with asyncio producer/consumer concurrency:

1. **Search** (`search.py`) — `generate_queries` expands base queries with variant/filetype suffixes, then concurrent `search_ddg_images` calls (run in threads via `asyncio.to_thread`) feed image URLs into a bounded `asyncio.Queue`.
2. **URL filter** (`orchestrator.py: is_good_url`) — cheap pre-download gate: extension allowlist, `BLOCKED_HOSTS` set, and thumbnail URL pattern detection (`_looks_like_thumbnail`).
3. **Download** (`download.py`) — `fetch_image` validates content-type, Content-Length, min/max byte size, and rejects HTML responses. Gated by a `download_concurrency` semaphore.
4. **Dedup** (`dedupe.py`, `db.py`) — SHA-256 checked against SQLite before the quality gate; `PhashIndex` (perceptual hash via `imagehash`) provides near-duplicate detection after the quality gate, under a shared `phash_lock` to prevent racing workers from both admitting the same near-dup.
5. **Quality gate** (`utils.py: image_quality_metrics`, `orchestrator.py: _is_low_quality`) — 8 CV metrics computed inline before classification: luma stddev, saturation stddev/mean, colorfulness (Hasler & Süsstrunk), flat-tile ratio, Laplacian variance, edge density, entropy. Images failing compound threshold checks are discarded as `low_quality:<flags>`. A `None` luma_stddev means PIL couldn't decode the bytes (`decode_failed`).
6. **Classify** (`classify.py`) — `XAIClassifier` sends base64-encoded images to `xai.base_url/chat/completions`. It tries `model_primary` first, falling back to `model_fallback` on `ModelNoVisionError`. Structured output uses Pydantic `response_format` JSON schema; if the model rejects that, it retries without `response_format`. Images flagged `is_underage_or_ambiguous=True` are always discarded.
7. **Save** (`orchestrator.py: save_image`) — file named `{sha256[:12]}_{slug}{ext}` inside the category subdirectory.
8. **Manifest** — written to `manifest.json` at completion if `config.manifest=True`.

### Key modules

| Module | Responsibility |
|---|---|
| `config.py` | `AppConfig`, `XAIConfig`, `GeminiConfig` dataclasses; loaded from `config.yaml` + env vars; CLI flags override after loading |
| `models.py` | `ClassificationResult` Pydantic model — the structured output contract with the LLM |
| `db.py` | `ImageDB` — thread-safe SQLite wrapper; URL and sha256 uniqueness enforced at DB level |
| `source_intelligence.py` | `SourceIntelligence` — UCB-bandit scoring of queries and hosts across runs; persisted to `source_intelligence.json`; drives adaptive query variants and host filtering |
| `quality_audit.py` | Post-run CV quality audit using the same metrics as the runtime gate; outputs `QUALITYREPORT.MD` and `quality_report.json`; optional Gemini LLM second-opinion (`--llm`) |
| `dedupe.py` | `compute_sha256`, `compute_phash`, `PhashIndex` |
| `utils.py` | `image_quality_metrics` (8 CV metrics), `sniff_image_mime`, `slugify`, `utc_now_iso` |
| `search.py` | DDG image search with rate-limit retry + query expansion |
| `download.py` | Async HTTP fetch with content-type/size validation |
| `classify.py` | `XAIClassifier` — primary/fallback model selection, JSON schema structured output, exponential back-off retry |
| `logging_config.py` | Rotating file handler setup; log written under `log_dir` (not `out_dir`) |

### Concurrency model

The pipeline uses three semaphores: `search_sema` (search_concurrency), `download_sema` (download_concurrency), and one inside `XAIClassifier` (classify_concurrency). A `threading.Event` (`stop_flag`) bridges the async stop signal into the synchronous DDG search threads. `Counters` is an asyncio-locked counter class for progress tracking; `reserve_kept` / `release_kept` implement atomic limit enforcement so no worker overshoots `--limit`.

### Quality flag logic

Flags only fire on *compound* conditions to avoid false positives on legitimate official art with solid backgrounds:

| Flag | Condition |
|---|---|
| `low_luma+low_sat_std` | luma_stddev < threshold AND sat_stddev < threshold |
| `low_sat_mean+low_sat_std` | sat_mean < threshold AND sat_stddev < threshold |
| `low_color+low_sat_std` | colorfulness < threshold AND sat_stddev < threshold |
| `flat_ratio+low_entropy` | flat_ratio >= threshold AND entropy < threshold |
| `flat_ratio+blurry` | flat_ratio >= threshold AND laplacian_var < threshold AND edge_density < threshold |
| `low_lap+low_edge` | laplacian_var < threshold AND edge_density < threshold |

All thresholds live in `AppConfig` and map 1-to-1 to `config.yaml` keys.

### `tifa.py` — image generation script

Reads `tifa.yaml` for prompt, model, aspect_ratio, resolution, n, response_format, output_dir, and filename_prefix. Uses the OpenAI SDK pointed at `https://api.x.ai/v1`. Available models: `grok-imagine-image-quality` ($0.05/image) and `grok-imagine` ($0.02/image). Aspect ratios: `1:1`, `16:9`, `9:16`, `4:3`, `3:4`. Resolutions: `1k`, `2k`.

## Tracking files

- `CODINGPROGRESS.MD` — running log of what was tried and what was learned; update it when trying something new or getting stuck.
- `EXPERIMENTS.MD` — short entry per run/change with outcome.
- `SUCCESSRATE.MD` — per-run metrics (kept rate, failure breakdown).
- `RUNBOOK.md` — repeatable iteration checklist; read at the start of a session.
