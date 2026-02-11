from __future__ import annotations

import asyncio
import base64
import imghdr
import json
import logging
import random
from typing import Any

import aiohttp

from .models import ClassificationResult


TEST_IMAGE_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

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


class ModelNoVisionError(RuntimeError):
    pass


class ImageDecodeError(RuntimeError):
    pass


def _guess_mime(data: bytes) -> str:
    kind = imghdr.what(None, data)
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


class XAIClassifier:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model_primary: str,
        model_fallback: str,
        timeout: float,
        max_retries: int,
        semaphore: asyncio.Semaphore,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model_primary = model_primary
        self.model_fallback = model_fallback
        self.model_name = model_primary
        self.timeout = timeout
        self.max_retries = max_retries
        self.semaphore = semaphore
        self.logger = logging.getLogger(__name__)

    async def check_vision_support(self, session: aiohttp.ClientSession) -> str:
        test_bytes = base64.b64decode(TEST_IMAGE_B64)
        payload = self._build_payload(self.model_primary, test_bytes, use_schema=False)
        try:
            await self._post(session, payload)
            self.model_name = self.model_primary
            return self.model_name
        except ModelNoVisionError:
            if not self.model_fallback:
                raise
            payload = self._build_payload(self.model_fallback, test_bytes, use_schema=False)
            await self._post(session, payload)
            self.model_name = self.model_fallback
            return self.model_name
        except ImageDecodeError as exc:
            self.logger.warning("vision check skipped: %s", exc)
            self.model_name = self.model_primary
            return self.model_name
        except (asyncio.TimeoutError, aiohttp.ClientError) as exc:
            self.logger.warning("vision check skipped (network): %s", exc)
            self.model_name = self.model_primary
            return self.model_name
        except RuntimeError as exc:
            self.logger.warning("vision check skipped (error): %s", exc)
            self.model_name = self.model_primary
            return self.model_name

    async def classify(
        self, session: aiohttp.ClientSession, image_bytes: bytes
    ) -> ClassificationResult:
        async with self.semaphore:
            try:
                return await self._classify_with_model(session, self.model_name, image_bytes)
            except ModelNoVisionError:
                if self.model_name != self.model_fallback and self.model_fallback:
                    self.model_name = self.model_fallback
                    return await self._classify_with_model(
                        session, self.model_name, image_bytes
                    )
                raise

    async def _classify_with_model(
        self, session: aiohttp.ClientSession, model: str, image_bytes: bytes
    ) -> ClassificationResult:
        use_schema = True
        for attempt in range(self.max_retries + 1):
            try:
                payload = self._build_payload(model, image_bytes, use_schema=use_schema)
                response = await self._post(session, payload)
                return self._parse_response(response)
            except ModelNoVisionError:
                raise
            except RuntimeError as exc:
                if use_schema and self._schema_unsupported(str(exc)):
                    use_schema = False
                    continue
                if attempt >= self.max_retries:
                    raise
                delay = (2 ** attempt) + random.random()
                await asyncio.sleep(delay)
            except Exception:
                if attempt >= self.max_retries:
                    raise
                delay = (2 ** attempt) + random.random()
                await asyncio.sleep(delay)
        raise RuntimeError("classification failed")

    def _build_payload(self, model: str, image_bytes: bytes, use_schema: bool) -> dict[str, Any]:
        mime = _guess_mime(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        schema = ClassificationResult.model_json_schema()
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": USER_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
                        },
                    ],
                },
            ],
            "temperature": 0,
        }
        if use_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "classification", "schema": schema, "strict": True},
            }
        return payload

    async def _post(self, session: aiohttp.ClientSession, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("XAI API key missing")
        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with session.post(url, json=payload, headers=headers, timeout=self.timeout) as resp:
            text = await resp.text()
            if resp.status >= 400:
                message = self._extract_error_message(text)
                if self._is_decode_error(message):
                    raise ImageDecodeError(message)
                if self._is_no_vision_error(message):
                    raise ModelNoVisionError(message)
                raise RuntimeError(f"XAI API error {resp.status}: {message}")
            return json.loads(text)

    @staticmethod
    def _extract_error_message(text: str) -> str:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text.strip()
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or text)
        return text.strip()

    @staticmethod
    def _is_no_vision_error(message: str) -> bool:
        msg = message.lower()
        return ("image" in msg or "vision" in msg) and (
            "not support" in msg or "unsupported" in msg
        )

    @staticmethod
    def _schema_unsupported(message: str) -> bool:
        msg = message.lower()
        return "response_format" in msg or "json_schema" in msg or (
            "schema" in msg and "unsupported" in msg
        )

    @staticmethod
    def _is_decode_error(message: str) -> bool:
        msg = message.lower()
        return "decode" in msg and ("image" in msg or "buffer" in msg)

    def _parse_response(self, data: dict[str, Any]) -> ClassificationResult:
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError("no choices in response")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    text_parts.append(part.get("text", ""))
            content = "".join(text_parts)
        payload = _extract_json(content)
        result = ClassificationResult.model_validate(payload)
        if result.is_underage_or_ambiguous:
            result = result.model_copy(update={"label": "discard"})
        return result
