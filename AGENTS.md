# AGENTS.md

This file defines how automated coding agents should operate inside this
repository. It provides consistent expectations for style, safety, tooling,
and communication so agent changes are predictable and reviewable.

## Purpose in this project
- Keep agent edits aligned with existing code style and repo conventions.
- Reduce back-and-forth by documenting preferred workflows and commands.
- Make agent behavior safer (no destructive commands, no hidden assumptions).
- Improve collaboration by standardizing how agents report changes.

## Workspace context
- Root: C:\Users\Håkon\PycharmProjects\xAI
- Shell: PowerShell

## General rules
- Prefer small, targeted edits over broad refactors unless requested.
- Don’t introduce new dependencies without asking.
- Avoid destructive actions (e.g., `git reset --hard`, delete operations)
  unless explicitly requested.
- Keep new code readable and documented when non-obvious.
- Preserve existing file structure unless there is a clear benefit.

## Tooling and commands
- Use `rg` for searching when possible.
- If you run commands, summarize key results instead of dumping raw output.
- Ask before running long or potentially disruptive commands (tests, installs).
- At the start of a session, read `RUNBOOK.md`, the latest `SUCCESSRATE.MD`,
  and `QUALITYREPORT.MD` (if it exists) to understand the current state.
- Refresh `SUCCESSRATE.MD` after a new run so metrics track changes.
- When logs show a dominant failing host or error, note it in CODINGPROGRESS.MD
  and consider updating exclusions/blocklists.

## Code style and quality
- Follow existing patterns, naming, and formatting in the repository.
- Prefer straightforward, maintainable code over clever solutions.
- Add tests when behavior changes, if a relevant test location exists.

## Communication
- Explain what changed, where, and why.
- Call out any assumptions or open questions.
- Offer next steps (tests to run, files to review) when helpful.
- Always include "Next commands" with exact terminal commands and a brief
  explanation of what each command does.
- Update CODINGPROGRESS.MD when we try something new, learn something, or get stuck
  for too long; include the date and short notes.
- Add a short entry to EXPERIMENTS.MD for each run/change so outcomes are tracked.


<!-- codex:agents:start -->
## Agent Guidance (Auto)
Project purpose: A Python 3.11+ CLI tool that searches for Tifa Lockhart images, downloads them, classifies them with xAI/Grok vision, and sorts them into category folders.
Primary entry points:
- module: `python -m tifa_archivist`
Install:
- `pip install -r requirements.txt`
Run:
- `python -m tifa_archivist`
Test:
- `python -m pytest`
Lint/Format (inferred):
- Not specified
Conventions:
- Top-level packages: `tifa_archivist`
Configs to check:
- `requirements.txt`
Secrets/credentials:
- Check `.env` for required settings and keep secrets out of git
- Check `config.yaml` for required settings and keep secrets out of git
<!-- codex:agents:end -->

Signed-off-by: gpt-5.3-codex — 2026-02-07
