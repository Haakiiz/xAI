# EXPERIMENTS.MD

| Date | Change | Outcome |
| --- | --- | --- |
| 2026-02-11 | Redact xAI API key from config and plan history rewrite to satisfy GitHub push protection. | In progress. |
| 2026-02-14 | Added `analyze_bank_statement.py` for CSV bank statement analysis with Grok (subscriptions + unwanted recurring costs). | Script added and validated with `py_compile`. |
| 2026-02-15 | Updated CSV column detection in `analyze_bank_statement.py` to support `bokføring`, `Beløp`, `Mottaker`, `Navn`, `Tittel`, `Avsender`, and `Betalingstype`. | Header matching aligned with current export format; `py_compile` passes. |
| 2026-02-15 | Fixed semicolon CSV parsing and header normalization in `analyze_bank_statement.py`; switched to per-row first non-empty description across prioritized columns. | Verified by loading `Brukskonto 6201.11.64183 - 2026-02-15 11.35.47.csv` successfully (`932` rows, `0` empty descriptions). |
| 2026-02-15 | Investigated `xAI API-feil 403: error code: 1010`; compared request signatures. Added `User-Agent` + `Accept` headers and switched default model to `grok-4-1-fast-reasoning`. | Reproduced `403/1010` with urllib default; verified success (`200`) with browser-like UA and successful live `ask_grok(...)` response. |
| 2026-02-15 | Adjusted transaction description source and model prompt to treat `Tittel` as payee/merchant and avoid account-owner fields (`Avsender`, `Mottaker`). | Validation sample now resolves to merchant-like values (e.g. `APPLE.COM/BILL`) instead of account number strings. |
