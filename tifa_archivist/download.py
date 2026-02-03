from __future__ import annotations

import asyncio
import logging
import random
import time
from urllib.parse import urlparse


class FetchError(Exception):
    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.status_code = status_code


ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/pjpeg",
    "image/png",
    "image/webp",
    "application/octet-stream",
    "binary/octet-stream",
}

RETRY_STATUSES = {429, 502, 503, 504}
_HOST_COOLDOWN: dict[str, float] = {}
_HOST_COOLDOWN_LOCK = asyncio.Lock()


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
        if seconds >= 0:
            return seconds
    except ValueError:
        return None
    return None


async def _wait_for_host(host: str) -> None:
    if not host:
        return
    async with _HOST_COOLDOWN_LOCK:
        until = _HOST_COOLDOWN.get(host)
    if until:
        now = time.monotonic()
        if until > now:
            await asyncio.sleep(until - now)


async def _set_host_cooldown(host: str, seconds: float) -> None:
    if not host or seconds <= 0:
        return
    now = time.monotonic()
    until = now + seconds
    async with _HOST_COOLDOWN_LOCK:
        current = _HOST_COOLDOWN.get(host, 0.0)
        if until > current:
            _HOST_COOLDOWN[host] = until


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
        host = ""
        try:
            async with sema:
                parsed = urlparse(url)
                host = parsed.hostname or parsed.netloc or ""
                await _wait_for_host(host)
                headers = None
                if parsed.scheme and parsed.netloc:
                    headers = {"Referer": f"{parsed.scheme}://{parsed.netloc}/"}
                async with session.get(url, timeout=timeout, headers=headers) as resp:
                    if resp.status != 200:
                        retry_after = None
                        if resp.status in RETRY_STATUSES:
                            retry_after = _parse_retry_after(
                                resp.headers.get("Retry-After")
                            )
                        if resp.status in {403, 404}:
                            logger.debug(
                                "non-retryable status url=%s status=%s",
                                url,
                                resp.status,
                            )
                            raise FetchError(f"status {resp.status}")
                        raise FetchError(
                            f"status {resp.status}",
                            retry_after=retry_after,
                            status_code=resp.status,
                        )
                    content_type = (
                        resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                    )
                    content_encoding = resp.headers.get("Content-Encoding", "").strip().lower()
                    transfer_encoding = resp.headers.get("Transfer-Encoding", "").strip().lower()
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
                        if not content_encoding and "chunked" not in transfer_encoding:
                            delta = expected - len(data)
                            tolerance = max(64_000, int(expected * 0.10))
                            if delta > tolerance:
                                logger.debug(
                                    "content-length mismatch tolerated url=%s expected=%s got=%s",
                                    url,
                                    expected,
                                    len(data),
                                )
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
            if isinstance(exc, FetchError) and exc.retry_after:
                delay = max(delay, exc.retry_after)
                if exc.status_code == 429:
                    await _set_host_cooldown(host, delay)
            await asyncio.sleep(delay)
    return None, None
