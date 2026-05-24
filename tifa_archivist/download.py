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
    "image/webp",
    "image/avif",
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
                    # Content-Length early-out: only reject if the declared size is
                    # clearly over the max.  Do NOT reject on declared-too-small here —
                    # many CDNs (tensorart, wixmp, wallpaperaccess) declare the
                    # *compressed* size and aiohttp transparently decompresses, making
                    # the actual body larger.  Let the post-download size check decide.
                    content_length = resp.headers.get("Content-Length")
                    if content_length and content_length.isdigit():
                        if int(content_length) > max_bytes:
                            raise FetchError("image too large")
                    if content_type:
                        if content_type not in ALLOWED_CONTENT_TYPES:
                            raise FetchError(f"unsupported content-type: {content_type}")
                    else:
                        # If the server does not send a content-type, rely on URL filtering.
                        pass
                    data = await resp.content.read(max_bytes + 1)
                    # Authoritative size checks on the actual bytes received.
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
            logger.debug("download retry %s/%s url=%s error=%s", attempt + 1, max_retries, url, exc)
            await asyncio.sleep(delay)
    return None, None
