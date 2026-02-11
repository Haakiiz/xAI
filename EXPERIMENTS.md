# EXPERIMENTS.md

Use this to record every change we try and the observed result. Keep entries short.

| Date | Goal | Change | Config/Code | Run/Log | Outcome | Next |
| ---- | ---- | ------ | ----------- | ------- | ------- | ---- |
| 2026-02-03 | Reduce gray images | Added variance filters | config + orchestrator | logs/run_20260203_214141.log | Fewer gray, but truncated discards spiked | Allow truncated with filters |
| 2026-02-03 | Validate allow_truncated | Ran batch after allow_truncated default true | config | logs/run_20260203_221825.log | No download failures; kept 41.7% (10/24); discards mostly gemini/limit | Tune sat thresholds or min_side |
| 2026-02-03 | Reduce gray images | Add colorfulness threshold | config + orchestrator | pending | Pending | Run batch and tune min_colorfulness |
| 2026-02-03 | Automate gray audit | Added quality audit CLI + optional LLM review | new module + CLI | QUALITYREPORT.MD | Flagged 0/20, but user still sees gray | Raise min_colorfulness/min_sat_mean |
| 2026-02-03 | Improve audit UX | Added progress bars to audit + LLM audit | quality_audit.py | pending | Pending | Re-run audit and confirm progress |
| 2026-02-03 | Fix LLM audit crash | Fix as_completed KeyError | quality_audit.py | pending | Pending | Re-run LLM audit |
| 2026-02-03 | Surface LLM results | Include LLM gray/low in report + flags | quality_audit.py | pending | Pending | Re-run audit and review QUALITYREPORT.MD |
| 2026-02-03 | LLM audit result | LLM flagged 19/20 gray/low | QUALITYREPORT.MD | logs/run_20260203_221825.log | Metrics missed gray blocks; need flat-tile ratio | Add flat-tile filter |
| 2026-02-04 | LLM audit result | LLM flagged 17/20 gray/low; flat_ratio high | QUALITYREPORT.MD | logs/run_20260203_232038.log | Gray blocks persist; flat_ratio useful | Discard on flat_ratio alone |
| 2026-02-04 | Tighten gray filter | Discard if flat_ratio >= threshold | orchestrator.py | pending | Pending | Run batch and re-audit |
