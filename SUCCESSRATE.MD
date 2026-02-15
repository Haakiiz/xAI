# SUCCESSRATE.MD

Last updated: 2026-02-14

## Baseline metrics (cumulative, from tifa_dataset\\images.db)
- Total records: 51
- Kept: 20 (39.2%)
- Discarded: 31 (60.8%)
- Top discard reasons (model_name):
  - gemini-3-flash-preview=21
  - limit_reached=5
  - decode_failed=3
  - low_variance=2
- Category counts:
  - discard=31
  - render_3d=10
  - fanart_2d=3
  - original_game=2
  - cosplay=2
  - wallpaper=1
  - sexy_sfw=1
  - other=1

## Latest run log metrics
- latest_log: C:\Users\Håkon\PycharmProjects\xAI\logs\run_20260203_232038.log
- download_failed: 11
- classification_failed: 1
- discarded_invalid: 3
- image_decode_skipped: 0
- rate_limited: 0
- search_failed: 0
- download_reasons:
  - image too small=11


## Maintenance updates
- 2026-02-14: Added `analyze_bank_statement.py` helper for Grok-based subscription/cost analysis from bank CSV exports.
