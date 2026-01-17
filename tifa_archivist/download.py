from __future__ import annotations

import asyncio
import logging
import random


class FetchError(Exception):
    pass


async def fetch_image(
    url: str,
    session,
    sema: asyncio.Semaphore,
    timeout,
    max_bytes: int,
    max_retries: int,
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
                    if content_type and not content_type.startswith("image/"):
                        raise FetchError("not an image")
                    data = await resp.content.read(max_bytes + 1)
                    if len(data) > max_bytes:
                        raise FetchError("image too large")
                    return data, content_type
        except (asyncio.TimeoutError, FetchError, Exception) as exc:
            if attempt == max_retries - 1:
                logger.debug("download failed url=%s error=%s", url, exc)
                return None, None
            delay = (2 ** attempt) + random.random()
            await asyncio.sleep(delay)
    return None, None
