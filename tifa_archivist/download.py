from __future__ import annotations

import asyncio
import logging
import random


class FetchError(Exception):
    pass


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "application/octet-stream",
}


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
                async with session.get(url, timeout=timeout) as resp:
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
                        if expected < min_bytes:
                            raise FetchError("image too small")
                    if content_type:
                        if content_type not in ALLOWED_CONTENT_TYPES:
                            raise FetchError("unsupported content-type")
                    else:
                        # If the server does not send a content-type, rely on URL filtering.
                        pass
                    data = await resp.content.read(max_bytes + 1)
                    if expected is not None and len(data) != expected:
                        raise FetchError("truncated download")
                    if len(data) > max_bytes:
                        raise FetchError("image too large")
                    if len(data) < min_bytes:
                        raise FetchError("image too small")
                    if data.lstrip().startswith(b"<"):
                        raise FetchError("html response")
                    return data, content_type
        except (asyncio.TimeoutError, FetchError, Exception) as exc:
            if attempt == max_retries - 1:
                logger.debug("download failed url=%s error=%s", url, exc)
                return None, None
            delay = (2 ** attempt) + random.random()
            await asyncio.sleep(delay)
    return None, None
