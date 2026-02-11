# CODINGPROGRESS.MD

This file tracks progress, experiments, and decisions while getting the
pipeline working reliably. Update it when we try something new, learn
something, or get stuck for too long.

## 2026-02-03
Goal: More images downloaded and classified, with fewer gray/low-quality files.

Update: moved success/error metrics into `SUCCESSRATE.MD` and added a refresh script
in `scripts/update_successrate.ps1`.

Update: strengthened download retry handling in `tifa_archivist/download.py` by
honoring Retry-After for 429/403/502/503/504 and failing fast on content-length
mismatches to avoid partial/gray images.

Update: New metrics in `SUCCESSRATE.MD` after the change show a big spike in
download failures, dominated by `content-length mismatch` (143 of 147) in
`logs/run_20260203_204215.log`. Kept rate nudged to 47.2% (42/89). This suggests
some hosts frequently report incorrect content-length; we need to decide whether
to keep strict rejection, allow a tolerance window, or selectively allow
mismatches for known hosts.

Update: Implemented a tolerance window for content-length mismatches in
`tifa_archivist/download.py` (allow small mismatches; fail only when short by
>10% or >64KB). This should reduce false failures while still rejecting likely
truncated downloads.

Update: After a new run, mismatches still dominated failures. Relaxed the
content-length mismatch handling to be non-fatal (log at debug and proceed) so
valid images can still decode and be filtered by size/quality. Also blocked
`previewsworld.com` (SSL errors) in `tifa_archivist/orchestrator.py`.

Update: Implemented a low-variance (gray) filter and stricter decode gate.
Added `min_luma_stddev` and `allow_undecoded` config options in
`tifa_archivist/config.py`; by default we now discard decode-failed images and
discard decoded images whose luma stddev falls below the threshold. This should
reduce gray/near-solid outputs while keeping valid images.

Update: Latest run shows heavy 429 rate limiting (35). Added per-host cooldown
on 429 in `tifa_archivist/download.py` to slow repeat hits and reduce wasted
retries. This should cut down on rate-limited failures and improve throughput.

Update: Fixed `UnboundLocalError` in `tifa_archivist/orchestrator.py` by
initializing `sha256` to None before invalid-image discard paths. Prevents worker
crashes when an image fails validation before hashing.

Update: Added a truncated-image decode fallback in `tifa_archivist/utils.py`
so decode failures get a second, lenient attempt. This should reduce
`decode_failed` discards while still enforcing min-side and low-variance checks.

Update: Added `classify_max_side` to downscale images before classification and
made `prepare_classification_bytes` retry with truncated-image loading. This
should reduce Gemini 400s and `image prep failed` events for large/truncated
files.

Update: Strengthened the gray/flat-image filter by adding saturation variance
and a configurable `variance_max_side` sampling size. We now discard images
only if both luma and saturation stddev are below thresholds, which should
target gray/blank outputs without killing textured grayscale art.

Update: Added a thumbnail/preview URL filter in `tifa_archivist/orchestrator.py`
to block obvious low-quality sources (thumb/preview/avatar/icon and small size
query patterns). This should cut down on tiny/gray images before download.

Update: Latest logs show heavy 403s from cdn.donmai.us and paheal mirrors.
Added those hosts (and rule34) to both search exclusions and runtime blocklist,
and made 403/404 non-retryable in `tifa_archivist/download.py` to reduce waste.

Update: Tightened gray-image filtering by adding `min_sat_mean` and an
`allow_truncated` guard. We now reject truncated decodes by default and discard
low-saturation images even when luma variance is higher, which should reduce
gray outputs that previously slipped through.

Update: Latest run shows most discards were `truncated` even at full
resolutions. Flipped `allow_truncated` default to True so valid images aren’t
thrown away, while keeping variance filters to catch gray placeholders.

Update: Added `RUNBOOK.md` (repeatable iteration loop) and `EXPERIMENTS.md`
to track each change/run outcome; updated `AGENTS.md` to require experiments
entries. This should make iteration faster and less error-prone.

Update: Added a colorfulness metric (`min_colorfulness`) to the gray filter and
expanded quality checks to include colorfulness alongside luma/saturation.
This should better catch gray/washed-out images that still have mild variance.

Update: Added a quality audit CLI (`python -m tifa_archivist audit`) with optional
Gemini review. It produces `QUALITYREPORT.MD` and `quality_report.json` so we can
measure gray/low-quality rates without manual inspection.

Update: New batch (`logs/run_20260203_221825.log`) after allowing truncated
decodes shows zero download failures and only one decode_failed discard; kept
rate ~41.7% (10/24) with discards mainly from Gemini/limit. Next knob is
raising saturation thresholds or `min_side`.

Update: Quality audit run shows 0/20 flagged but user still reports gray images,
so thresholds are too lenient. Next step is to raise `min_colorfulness` and/or
`min_sat_mean` and re-audit.

Update: Refreshed workflow docs (`RUNBOOK.md`, `AGENTS.md`) with start/end
checklists and explicit commands so a fresh agent can pick up the loop quickly.

Update: Added tqdm progress bars to quality audit (metrics pass + LLM audit) so
the CLI shows live progress instead of appearing idle.

Update: Fixed LLM audit progress loop crash (as_completed KeyError) by wrapping
each task to return its item alongside the result.

Update: LLM audit results are now surfaced in `QUALITYREPORT.MD` (summary +
sample list) and can mark items as flagged when the LLM says gray/low.

Update: LLM audit shows 95% gray/low in `QUALITYREPORT.MD`, indicating our
metrics are missing large flat/corrupted regions. Added a flat-tile ratio
metric (grid-based) to both runtime filtering and audits.

Update: Latest LLM audit still shows ~85% gray/low with high flat ratios.
Tightened runtime filter to discard any image with flat_ratio >= threshold
regardless of saturation variance.
Plan:
- Baseline: summarize the latest logs and record the most common failure reasons
  (too_small, decode_failed, 429, etc.) and top failing hosts.
- Download integrity: treat content-length mismatches (without encoding) as
  retryable failures; add stronger backoff for 429/403.
- Decode gate: stop saving undecoded images by default; add a config flag to
  allow "decode_failed_allowed" only if explicitly enabled.
- Decode robustness: allow a lenient decode fallback for truncated images; if
  decode_failed persists, consider webp-specific fallback or host blocklist.
- Classification robustness: downscale to a sane max side before sending to
  Gemini; keep an eye on prep failures and 400s.
- Quality filter: add saturation variance alongside luma; tune thresholds if
  gray images persist.
- URL filter: reject thumbnail/preview URLs early to avoid low-quality sources.
- Host filter: block 403-heavy hosts and avoid re-trying 403/404 responses.
- Gray filter: add saturation mean threshold and disable truncated-image saves
  by default.
- Gray filter: add colorfulness threshold and tune it if gray persists.
- Quality filter: reject URLs that look like thumbnails/previews and add a
  decoded-image variance/entropy check to discard gray or near-solid images.
- Search yield: tune queries and consider higher min_side only after the above
  steps stabilize; adjust concurrency for fewer failed downloads.
- Validate: run a small batch, compare kept count and visual quality, and log
  new metrics here.

Notes/Discoveries:
- (none yet)
## 2026-02-07
Update: Added Laplacian-variance, edge-density, and entropy metrics to runtime
filters and quality audit. Integrated new thresholds in config and reporting to
better detect flat/gray outputs.

Update: Runtime filter now treats truncated images as acceptable only when they
pass the stronger texture/entropy checks; otherwise they are discarded as
"truncated_low_quality".

Update: Quality audit output now includes laplacian variance, edge density, and
entropy per sample to align with the runtime gate.
Update: Refreshed README CLI usage examples to include run/stats/audit,
manifest toggles, and top-level --config usage.
