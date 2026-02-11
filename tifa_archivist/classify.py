from __future__ import annotations

import asyncio
import base64
import json
import logging
import random
from typing import Any

import aiohttp
from PIL import Image
from io import BytesIO

from .models import ClassificationResult
from .utils import sniff_image_mime


SYSTEM_PROMPT = (
    "You are a strict image classifier. Return ONLY JSON matching the schema. "
    "If the subject appears under 18 or is ambiguous, set is_underage_or_ambiguous=true "
    "and label=discard. If uncertain, choose discard. Provide short reasons."
)

USER_PROMPT = (
    "Classify this image of Tifa Lockhart into exactly one label: "
    "original_game, wallpaper, cosplay, fanart_2d, render_3d, sexy_sfw, nsfw, other, discard. "
    "Return confidence 0..1 and brief reasons."
)


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


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise exc


class GeminiClassifier:
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

    async def classify(
        self, session: aiohttp.ClientSession, image_bytes: bytes, mime_type: str | None
    ) -> ClassificationResult:
        async with self.semaphore:
            for attempt in range(self.max_retries + 1):
                try:
                    payload = self._build_payload(image_bytes, mime_type)
                    response = await self._post(session, payload)
                    return self._parse_response(response)
                except Exception:
                    if attempt >= self.max_retries:
                        raise
                    delay = (2 ** attempt) + random.random()
                    await asyncio.sleep(delay)
        raise RuntimeError("classification failed")

    def _build_payload(self, image_bytes: bytes, mime_type: str | None) -> dict[str, Any]:
        mime = _guess_mime(image_bytes, fallback=mime_type)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": f"{SYSTEM_PROMPT}\n{USER_PROMPT}"},
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

    def _parse_response(self, data: dict[str, Any]) -> ClassificationResult:
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError("no candidates in response")
        content = candidates[0].get("content", {})
        parts = content.get("parts") or []
        text = ""
        for part in parts:
            if isinstance(part, dict):
                text += part.get("text", "")
        payload = _extract_json(text)
        result = ClassificationResult.model_validate(payload)
        if result.is_underage_or_ambiguous:
            result = result.model_copy(update={"label": "discard"})
        return result
