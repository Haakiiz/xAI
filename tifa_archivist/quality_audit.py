from __future__ import annotations

import asyncio
import base64
import json
import logging
import sqlite3
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image, ImageFilter, ImageStat, ImageFile
from tqdm import tqdm

from .config import AppConfig
from .utils import image_quality_metrics, sniff_image_mime


AUDIT_SYSTEM_PROMPT = (
    "You are a strict image quality auditor. Return ONLY JSON matching the schema."
)
AUDIT_USER_PROMPT = (
    "Assess whether this image is gray/washed out, a placeholder, or low-quality. "
    "Return JSON with: "
    "{\"quality\":\"good|gray|low\",\"is_gray\":true|false,\"reason\":\"...\"}"
)


@dataclass
class AuditItem:
    filepath: str
    url: str | None
    category: str | None
    width: int | None
    height: int | None
    luma_stddev: float | None
    sat_stddev: float | None
    sat_mean: float | None
    colorfulness: float | None
    flat_ratio: float | None
    laplacian_var: float | None
    edge_density: float | None
    entropy: float | None
    edge_mean: float | None
    brightness: float | None
    flagged: bool
    flags: list[str]
    llm: dict[str, Any] | None = None
    error: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise exc


def _guess_mime(data: bytes, fallback: str | None = None) -> str:
    if fallback:
        return fallback
    sniffed = sniff_image_mime(data)
    if sniffed:
        return sniffed
    try:
        with Image.open(BytesIO(data)) as image:
            kind = (image.format or "").lower()
    except Exception:
        return "image/jpeg"
    if kind == "jpeg":
        return "image/jpeg"
    if kind == "png":
        return "image/png"
    if kind == "gif":
        return "image/gif"
    if kind == "webp":
        return "image/webp"
    return "image/jpeg"


def _load_db_records(db_path: Path, limit: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.execute(
        """
        SELECT filepath, url, category, created_at
        FROM images
        WHERE filepath IS NOT NULL AND category != 'discard'
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    rows = [dict(row) for row in cur.fetchall()]
    conn.close()
    return rows


def _edge_mean(data: bytes, max_side: int) -> tuple[float | None, float | None, int | None, int | None]:
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
            working = image.copy()
            if max_side and max(working.size) > max_side:
                working.thumbnail((max_side, max_side), Image.BILINEAR)
            luma = working.convert("L")
            brightness = ImageStat.Stat(luma).mean[0]
            edges = luma.filter(ImageFilter.FIND_EDGES)
            edge_mean = ImageStat.Stat(edges).mean[0]
            return float(edge_mean), float(brightness), width, height
    except Exception:
        return None, None, None, None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def _flag_metrics(config: AppConfig, item: AuditItem) -> tuple[bool, list[str]]:
    flags: list[str] = []
    if item.luma_stddev is None or item.sat_stddev is None or item.sat_mean is None:
        flags.append("metrics_missing")
        return True, flags

    low_luma = item.luma_stddev < config.min_luma_stddev
    low_sat_std = item.sat_stddev < config.min_sat_stddev
    low_sat_mean = item.sat_mean < config.min_sat_mean
    low_color = (
        item.colorfulness is not None
        and item.colorfulness < config.min_colorfulness
    )
    high_flat_ratio = (
        item.flat_ratio is not None
        and item.flat_ratio >= config.flat_tile_ratio
    )
    low_lap = (
        item.laplacian_var is not None
        and item.laplacian_var < config.min_laplacian_var
    )
    low_edge = (
        item.edge_density is not None
        and item.edge_density < config.min_edge_density
    )
    low_entropy = item.entropy is not None and item.entropy < config.min_entropy

    if low_luma and low_sat_std:
        flags.append("low_luma+low_sat_std")
    if low_sat_mean and low_sat_std:
        flags.append("low_sat_mean+low_sat_std")
    if low_color and low_sat_std:
        flags.append("low_color+low_sat_std")
    if high_flat_ratio:
        flags.append("flat_ratio")
    if low_lap and low_edge:
        flags.append("low_lap+low_edge")
    if low_entropy:
        flags.append("low_entropy")

    return len(flags) > 0, flags


def _build_report(items: list[AuditItem]) -> str:
    total = len(items)
    flagged = sum(1 for item in items if item.flagged)
    flagged_rate = (flagged / total * 100.0) if total else 0.0
    llm_items = [item for item in items if item.llm]
    llm_gray = [
        item
        for item in llm_items
        if (item.llm.get("quality") in {"gray", "low"})
        or (item.llm.get("is_gray") is True)
    ]
    llm_rate = (len(llm_gray) / len(llm_items) * 100.0) if llm_items else 0.0
    lines = [
        "# QUALITYREPORT.MD",
        "",
        f"Audited images: {total}",
        f"Flagged as likely gray/low quality: {flagged} ({flagged_rate:.1f}%)",
        f"LLM audited: {len(llm_items)}",
        f"LLM flagged gray/low: {len(llm_gray)} ({llm_rate:.1f}%)",
        "",
        "## Sample flagged images",
    ]
    sample = [item for item in items if item.flagged][:20]
    if not sample:
        lines.append("- (none)")
    for item in sample:
        llm_quality = item.llm.get("quality") if item.llm else None
        llm_reason = item.llm.get("reason") if item.llm else None
        lines.append(
            "- {path} | luma_std={luma:.2f} sat_std={sat:.2f} sat_mean={mean:.2f} "
            "color={color:.2f} flat={flat:.2f} lap={lap:.2f} edge_den={edge_den:.4f} "
            "entropy={entropy:.2f} edge={edge:.2f} bright={bright:.2f} flags={flags}{llm}".format(
                path=item.filepath,
                luma=item.luma_stddev or 0.0,
                sat=item.sat_stddev or 0.0,
                mean=item.sat_mean or 0.0,
                color=item.colorfulness or 0.0,
                flat=item.flat_ratio or 0.0,
                lap=item.laplacian_var or 0.0,
                edge_den=item.edge_density or 0.0,
                entropy=item.entropy or 0.0,
                edge=item.edge_mean or 0.0,
                bright=item.brightness or 0.0,
                flags=",".join(item.flags) or "none",
                llm=f" | llm={llm_quality} ({llm_reason})" if llm_quality else "",
            )
        )
    lines.append("")
    lines.append("## Sample LLM-flagged images")
    llm_sample = llm_gray[:20]
    if not llm_sample:
        lines.append("- (none)")
    else:
        for item in llm_sample:
            llm_quality = item.llm.get("quality") if item.llm else None
            llm_reason = item.llm.get("reason") if item.llm else None
            lines.append(
                "- {path} | llm={llm} ({reason})".format(
                    path=item.filepath,
                    llm=llm_quality or "unknown",
                    reason=llm_reason or "no reason",
                )
            )
    return "\n".join(lines)


class GeminiAuditor:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float,
        max_retries: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.semaphore = semaphore
        self.logger = logging.getLogger(__name__)

    async def audit(
        self, session: aiohttp.ClientSession, image_bytes: bytes, mime_type: str | None
    ) -> dict[str, Any]:
        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    payload = self._build_payload(image_bytes, mime_type)
                    response = await self._post(session, payload)
                    return self._parse_response(response)
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    delay = (2 ** attempt) + (0.1 * attempt)
                    await asyncio.sleep(delay)
        raise RuntimeError("audit failed")

    def _build_payload(self, image_bytes: bytes, mime_type: str | None) -> dict[str, Any]:
        mime = _guess_mime(image_bytes, fallback=mime_type)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{AUDIT_SYSTEM_PROMPT}\n{AUDIT_USER_PROMPT}"},
                        {"inline_data": {"mime_type": mime, "data": image_b64}},
                    ],
                }
            ],
            "generation_config": {"temperature": 0},
        }

    async def _post(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("Gemini API key missing")
        url = f"{self.base_url}/models/{self.model}:generateContent"
        headers = {"x-goog-api-key": self.api_key}
        async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"Gemini API error {resp.status}: {text.strip()}")
            return json.loads(text)

    def _parse_response(self, data: dict[str, Any]) -> dict[str, Any]:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("no candidates in response")
        content = candidates[0].get("content", {})
        parts = content.get("parts") or []
        text = ""
        for part in parts:
            if isinstance(part, dict):
                text += part.get("text", "")
        return _extract_json(text)


async def _run_llm_audit(items: list[AuditItem], config: AppConfig, limit: int) -> None:
    if not config.gemini.api_key:
        raise RuntimeError("Gemini API key missing. Set GEMINI_API_KEY or config.yaml.")
    candidates = [item for item in items if item.flagged]
    if not candidates:
        candidates = items
    candidates = candidates[:limit]
    semaphore = asyncio.Semaphore(max(1, min(config.classify_concurrency, 4)))
    auditor = GeminiAuditor(
        api_key=config.gemini.api_key,
        base_url=config.gemini.base_url,
        model=config.gemini.model,
        timeout=config.request_timeout,
        max_retries=config.max_classify_retries,
        semaphore=semaphore,
    )
    async with aiohttp.ClientSession() as session:
        async def audit_one(
            item: AuditItem,
        ) -> tuple[AuditItem, dict[str, Any] | None, Exception | None]:
            data = Path(item.filepath).read_bytes()
            try:
                result = await auditor.audit(session, data, None)
                return item, result, None
            except Exception as exc:
                return item, None, exc

        tasks = [asyncio.create_task(audit_one(item)) for item in candidates]
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="LLM audit",
            unit="img",
        ):
            item, result, error = await task
            if error is not None:
                item.llm = {"quality": "error", "reason": str(error)}
            else:
                item.llm = result
                quality = str(result.get("quality", "")).lower()
                is_gray = result.get("is_gray") is True
                if quality in {"gray", "low"} or is_gray:
                    item.flagged = True
                    flag = "llm_gray" if quality == "gray" or is_gray else "llm_low"
                    if flag not in item.flags:
                        item.flags.append(flag)


def run_quality_audit(
    config: AppConfig,
    out_dir: Path,
    limit: int,
    report_path: Path,
    json_path: Path,
    llm: bool,
    llm_limit: int,
) -> None:
    logger = logging.getLogger(__name__)
    records = _load_db_records(out_dir / "images.db", limit)
    items: list[AuditItem] = []
    for record in tqdm(records, desc="Audit metrics", unit="img"):
        filepath = record.get("filepath")
        if not filepath:
            continue
        path = Path(filepath)
        if not path.exists():
            items.append(
                AuditItem(
                    filepath=str(path),
                    url=record.get("url"),
                    category=record.get("category"),
                    width=None,
                    height=None,
                    luma_stddev=None,
                    sat_stddev=None,
                    sat_mean=None,
                    colorfulness=None,
                    flat_ratio=None,
                    laplacian_var=None,
                    edge_density=None,
                    entropy=None,
                    edge_mean=None,
                    brightness=None,
                    flagged=True,
                    flags=["missing_file"],
                    error="missing_file",
                )
            )
            continue
        data = path.read_bytes()
        (
            luma_stddev,
            sat_stddev,
            sat_mean,
            colorfulness,
            flat_ratio,
            laplacian_var,
            edge_density,
            entropy,
        ) = image_quality_metrics(
            data,
            config.variance_max_side,
            config.flat_grid_size,
            config.flat_tile_stddev_max,
            config.edge_threshold,
        )
        edge_mean, brightness, width, height = _edge_mean(
            data, config.variance_max_side
        )
        item = AuditItem(
            filepath=str(path),
            url=record.get("url"),
            category=record.get("category"),
            width=width,
            height=height,
            luma_stddev=luma_stddev,
            sat_stddev=sat_stddev,
            sat_mean=sat_mean,
            colorfulness=colorfulness,
            flat_ratio=flat_ratio,
            laplacian_var=laplacian_var,
            edge_density=edge_density,
            entropy=entropy,
            edge_mean=edge_mean,
            brightness=brightness,
            flagged=False,
            flags=[],
        )
        flagged, flags = _flag_metrics(config, item)
        item.flagged = flagged
        item.flags = flags
        items.append(item)

    if llm and items:
        logger.info("running LLM audit on up to %s images", llm_limit)
        asyncio.run(_run_llm_audit(items, config, llm_limit))

    report_path.write_text(_build_report(items), encoding="utf-8")
    json_path.write_text(
        json.dumps([item.__dict__ for item in items], indent=2),
        encoding="utf-8",
    )
