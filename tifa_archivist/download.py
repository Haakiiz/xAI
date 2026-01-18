from __future__ import annotations

import asyncio
import logging
import random
from urllib.parse import urlparse


class FetchError(Exception):
    pass


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",
    "binary/octet-stream",
}


def _is_allowed_content_type(content_type: str) -> bool:
    if not content_type:
        return True
    if content_type in ALLOWED_CONTENT_TYPES:
        return True
    return content_type.startswith("image/")


async def fetch_image(
    url: str,
    session,
    sema: asyncio.Semaphore,
    timeout,
    max_bytes: int,
    max_retries: int,
    min_bytes: int,
) -> tuple[bytes | None, str | None]:
    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        try:
            async with sema:
                parsed = urlparse(url)
                headers = None
                if parsed.scheme and parsed.netloc:
                    headers = {"Referer": f"{parsed.scheme}://{parsed.netloc}/"}
                async with session.get(url, timeout=timeout, headers=headers) as resp:
                    if resp.status != 200:
                        raise FetchError(f"status {resp.status}")
                    content_type = (
                        resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    )
                    content_length = resp.headers.get("Content-Length")
                    expected = None
                    if content_length and content_length.isdigit():
                        expected = int(content_length)
                        if expected > max_bytes:
                            raise FetchError("image too large")
                    if content_type and not _is_allowed_content_type(content_type):
                        logger.debug(
                            "non-image content-type url=%s content_type=%s",
                            url,
                            content_type,
                        )
                    data = await resp.content.read(max_bytes + 1)
                    if expected is not None and len(data) != expected:
                        logger.debug(
                            "content-length mismatch url=%s expected=%s got=%s",
                            url,
                            expected,
                            len(data),
                        )
                        raise FetchError("content-length mismatch")
                    if len(data) > max_bytes:
                        raise FetchError("image too large")
                    if len(data) < min_bytes:
                        raise FetchError("image too small")
                    if data.lstrip().startswith(b"<"):
                        raise FetchError("html response")
                    return data, content_type
        except (asyncio.TimeoutError, FetchError, Exception) as exc:
            if attempt == max_retries - 1:
                logger.warning("download failed url=%s error=%s", url, exc)
                return None, None
            delay = (2 ** attempt) + random.random()
            await asyncio.sleep(delay)
    return None, None
