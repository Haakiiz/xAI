from __future__ import annotations

import re
from datetime import datetime, timezone


def slugify(text: str, max_len: int = 32) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        text = "image"
    return text[:max_len]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
