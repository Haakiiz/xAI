# RUNBOOK.md

Purpose: a short, repeatable loop for improving image quality and download yield.

## Quick loop (5-10 minutes)
1) Adjust one knob (config or code) at a time.
2) Run a small batch (limit 10-30) to validate quickly.
3) Refresh metrics: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update_successrate.ps1`
4) Log the change + outcome in `EXPERIMENTS.md` (1 row).
5) Update `CODINGPROGRESS.MD` with key learning or next step.
6) Optional: audit outputs for gray images (shows progress bars):
   `python -m tifa_archivist audit --out .\tifa_dataset --limit 200 --report QUALITYREPORT.MD --json quality_report.json`

## Start-of-session checklist
1) Read `SUCCESSRATE.MD` and `QUALITYREPORT.MD` to see current rates.
2) Skim the latest `logs\run_*.log` for dominant errors or hosts.
3) Review the last entry in `EXPERIMENTS.MD` to continue the thread.

## After-run checklist
1) Refresh metrics: `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update_successrate.ps1`
2) Audit for gray images:
   `python -m tifa_archivist audit --out .\tifa_dataset --limit 200 --report QUALITYREPORT.MD --json quality_report.json`
3) Update `EXPERIMENTS.MD` with the run and outcome.
4) Update `CODINGPROGRESS.MD` with the key learning and next step.

## Useful commands
- Run (downloads + classify): `python -m tifa_archivist run --limit 20 --out .\tifa_dataset`
- Metrics (updates SUCCESSRATE.MD): `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/update_successrate.ps1`
- Stats (category counts): `python -m tifa_archivist stats --out .\tifa_dataset`
- Audit (metrics only): `python -m tifa_archivist audit --out .\tifa_dataset --limit 200 --report QUALITYREPORT.MD --json quality_report.json`
- Audit (LLM sample): `python -m tifa_archivist audit --out .\tifa_dataset --llm --llm-limit 20`

Notes:
- LLM audit writes results into `quality_report.json` and summarizes them in `QUALITYREPORT.MD`.

## Where to look
- Logs: `logs\run_*.log`
- Dataset DB: `tifa_dataset\images.db`
- Output folders: `tifa_dataset\{category}\*.jpg/png`
