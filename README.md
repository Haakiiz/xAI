# Tifa Archivist

A Python 3.11+ async CLI that searches for Tifa Lockhart images on DuckDuckGo,
downloads them, filters out low-quality / duplicate results with a computer-vision
quality gate, classifies the survivors with xAI/Grok vision, and sorts them into
category folders.

Also includes **`tifa.py`** — a standalone script that generates new images via
the xAI Grok Imagine API, configured through `tifa.yaml`.

## What the pipeline does

1. **Search** — DuckDuckGo image search, one task per query, safesearch off.
2. **URL filter** — rejects non-image extensions, blocked hosts, and thumbnail
   URL patterns (`/thumb`, `?w=150`, etc.).
3. **Download** — async with concurrency + size limits (`min_bytes`, `max_bytes`).
4. **Quality gate** (runtime, new) — three checks before classification:
   - **Decode check:** corrupted / half-rendered bytes → discarded as `decode_failed`.
   - **CV metrics:** 8 metrics (luma stddev, saturation, colorfulness, flat-tile ratio,
     Laplacian variance, edge density, entropy) → discarded as `low_quality:<flags>`.
   - **Perceptual-hash dedup:** near-duplicate of an already-kept image → discarded
     as `near_duplicate`.
5. **Classify** — xAI/Grok vision picks a category (or `--skip-classify` stores
   everything in `other/`).
6. **Store** — files written to the category folder; metadata written to
   `images.db` (SQLite) and optional `manifest.json`.

---

## Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## Environment variables

| Var | Required for | Purpose |
|---|---|---|
| `XAI_API_KEY` | `run` (unless `--skip-classify`) | xAI/Grok vision classification |
| `GEMINI_API_KEY` | `audit --llm` (optional) | Second-opinion quality audit |

On Windows:

```powershell
$env:XAI_API_KEY = "sk-..."
$env:GEMINI_API_KEY = "..."      # optional
```

You can also put the keys in `config.yaml` under `xai.api_key` / `gemini.api_key`,
but env vars take precedence.

---

## Commands

### `run` — search, download, filter, classify, store

```powershell
python -m tifa_archivist run --limit 20 --out .\tifa_dataset
python -m tifa_archivist run --limit 50 --out .\tifa_dataset --skip-classify
```

Flags:
- `--limit N` — stop after N images pass all gates and get kept.
- `--out DIR` — output directory (default `tifa_dataset`).
- `--manifest` / `--no-manifest` — write `manifest.json`.
- `--skip-classify` — skip the xAI call and put everything in `other/`.

### `stats` — count kept images per category

```powershell
python -m tifa_archivist stats --out .\tifa_dataset
```

### `audit` — post-hoc quality report on an existing dataset

```powershell
python -m tifa_archivist audit --out .\tifa_dataset
python -m tifa_archivist audit --out .\tifa_dataset --llm     # needs GEMINI_API_KEY
```

Writes `QUALITYREPORT.MD` (markdown summary) and `quality_report.json`
(per-image metrics). `--llm` additionally sends flagged images to Gemini for
a second opinion.

Flags:
- `--limit N` — audit the N most recent images (default 200).
- `--report PATH` — markdown report path (default `QUALITYREPORT.MD`).
- `--json PATH` — JSON report path (default `quality_report.json`).
- `--llm` — also run the Gemini LLM audit.
- `--llm-limit N` — max images to send to Gemini (default 30).

---

## Categories (output folders)

- `original_game` — screenshots or in-engine from FF7 / FFVII Remake
- `wallpaper` — high-res wallpapers, key art, official promotional images
- `cosplay` — real people cosplaying
- `fanart_2d` — 2D illustrations by fans
- `render_3d` — 3D renders, not from the game
- `sexy_sfw` — suggestive but safe-for-work
- `nsfw` — explicit
- `other` — anything else, or everything when `--skip-classify` is used

Anything underage or ambiguous is discarded, not stored.

## Output artifacts (all under `--out`)

| Path | Contents |
|---|---|
| `<category>/*.jpg` | Kept images, per category |
| `images.db` | SQLite with URL/sha256/phash dedup state |
| `manifest.json` | Per-image metadata (if `--manifest`) |
| `source_intelligence.json` | Per-host stats (currently unused at runtime) |
| `QUALITYREPORT.MD` | Post-hoc `audit` markdown report |
| `quality_report.json` | Post-hoc `audit` JSON report |
| `../logs/run.log` | Rotating log file (under `log_dir`, not `out_dir`) |

---

## Tuning quality

All thresholds live in `config.yaml`. The defaults ship in **balanced** mode;
adjust if you see too many rejections or too much slop.

### Too strict? (good images getting rejected as `low_quality`)

Lower these:

```yaml
min_luma_stddev: 10.0       # → 5.0   (accept lower-contrast images)
min_sat_stddev: 5.0         # → 3.0   (accept more monochrome-ish images)
min_colorfulness: 5.0       # → 3.0
flat_tile_ratio: 0.7        # → 0.85  (only reject very flat images)
min_laplacian_var: 10.0     # → 5.0   (accept softer images)
min_edge_density: 0.01      # → 0.005
min_entropy: 3.0            # → 2.5
```

### Too loose? (junk still getting through)

Raise the same knobs in the opposite direction, and/or:

```yaml
min_bytes: 50000            # → 100000 (reject small files)
phash_distance: 6           # → 10     (more aggressive dedup)
```

### Too few URLs making it through the pre-download filters?

`min_bytes`, the thumbnail regex, and the blocked host list are in
`orchestrator.py` (`BLOCKED_HOSTS`, `_looks_like_thumbnail`, `is_good_url`). To
restore low-quality-biased queries like bare `"cosplay"` / `"fanart"`, edit
`search_queries` in `config.yaml`.

---

## Typical workflow for a clean dataset

```powershell
$env:XAI_API_KEY = "sk-..."

# 1. Small run to smoke-test.
python -m tifa_archivist run --limit 10 --out .\tifa_dataset_v2

# 2. Audit it — target flagged rate <= 15%.
python -m tifa_archivist audit --out .\tifa_dataset_v2

# 3. Open QUALITYREPORT.MD.  If the flagged rate is too high, tighten
#    thresholds in config.yaml and rerun.

# 4. Real run.
python -m tifa_archivist run --limit 200 --out .\tifa_dataset_v2

# 5. Count what you got.
python -m tifa_archivist stats --out .\tifa_dataset_v2
```

The run is **idempotent**: re-running against the same `--out` will skip URLs,
SHA-256 hashes, and perceptual-hashes already stored in `images.db`.

---

## Tests

```powershell
pytest -q
```

Covers the URL filter, the quality gate, perceptual-hash near-duplicate
detection, and a full async-pipeline smoke test.

---

---

## Image generation (`tifa.py`)

Generate new images using the xAI Grok Imagine API:

```powershell
# Edit tifa.yaml first (prompt, model, size, etc.), then:
python tifa.py

# Override the prompt from the command line:
python tifa.py --prompt "Tifa Lockhart in snow, cinematic lighting"
```

Key options in `tifa.yaml`:

| Key | Options |
|---|---|
| `model` | `grok-imagine-image-quality` ($0.05) · `grok-imagine` ($0.02) |
| `aspect_ratio` | `1:1` · `16:9` · `9:16` · `4:3` · `3:4` |
| `resolution` | `1k` (≈1024 px) · `2k` (≈2048 px) |
| `n` | 1–10 images per call |
| `response_format` | `b64_json` (saved directly) · `url` (downloaded from temp link) |

Generated files land in `output_dir` (default `./generated`) named
`<prefix>_<timestamp>_<index>.png`.

---

## See also

- `CODINGPROGRESS.MD` — full development history, including quality-gate tuning notes.
- `EXPERIMENTS.MD` — notes on dataset experiments.
- `CLAUDE.md` — in-repo assistant briefing covering module layout and architecture.
