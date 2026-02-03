from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from tqdm import tqdm

from .classify import GeminiClassifier
from .config import AppConfig
from .db import ImageDB
from .dedupe import compute_sha256
from .download import fetch_image
from .search import search_live_images
from .utils import (
    luma_stddev,
    prepare_classification_bytes,
    slugify,
    utc_now_iso,
    validate_image_bytes,
)


CATEGORIES = [
    "original_game",
    "wallpaper",
    "cosplay",
    "fanart_2d",
    "render_3d",
    "sexy_sfw",
    "nsfw",
    "other",
]
ALLOWED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".heic",
    ".heif",
    ".jxl",
    ".bmp",
    ".tif",
    ".tiff",
}
BLOCKED_HOSTS = {
    "rare-gallery.com",
    "www.rare-gallery.com",
    "wallpapersden.com",
    "images.wallpapersden.com",
    "previewsworld.com",
    "www.previewsworld.com",
}
CLASSIFY_SUPPORTED_MIME = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}


class Counters:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.urls_found = 0
        self.downloaded = 0
        self.classified = 0
        self.kept = 0
        self.discarded = 0

    async def inc(self, name: str, amount: int = 1) -> int:
        async with self._lock:
            value = getattr(self, name) + amount
            setattr(self, name, value)
            return value

    async def reserve_kept(self, limit: int) -> bool:
        async with self._lock:
            if self.kept >= limit:
                return False
            self.kept += 1
            return True

    async def release_kept(self) -> None:
        async with self._lock:
            if self.kept > 0:
                self.kept -= 1

    async def get(self, name: str) -> int:
        async with self._lock:
            return getattr(self, name)

    async def snapshot(self) -> dict[str, int]:
        async with self._lock:
            return {
                "urls_found": self.urls_found,
                "downloaded": self.downloaded,
                "classified": self.classified,
                "kept": self.kept,
                "discarded": self.discarded,
            }


def ensure_output_dirs(out_dir: Path) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name in CATEGORIES:
        path = out_dir / name
        path.mkdir(parents=True, exist_ok=True)
        dirs[name] = path
    return dirs


def guess_extension(content_type: str) -> str:
    ext = mimetypes.guess_extension(content_type or "")
    if ext == ".jpe":
        ext = ".jpg"
    if not ext and content_type:
        manual = {
            "image/avif": ".avif",
            "image/heic": ".heic",
            "image/heif": ".heif",
            "image/jxl": ".jxl",
        }
        ext = manual.get(content_type)
    return ext or ".jpg"


def save_image(
    data: bytes, category_dir: Path, sha256: str, url: str, content_type: str | None
) -> str:
    ext = guess_extension(content_type or "")
    parsed = urlparse(url)
    stem = Path(parsed.path).stem or parsed.netloc or "image"
    slug = slugify(stem)
    prefix_len = 12
    while True:
        filename = f"{sha256[:prefix_len]}_{slug}{ext}"
        path = category_dir / filename
        if not path.exists() or prefix_len >= len(sha256):
            break
        prefix_len += 4
    path.write_bytes(data)
    return str(path)


async def progress_loop(counters: Counters, done_event: asyncio.Event, limit: int) -> None:
    bar = tqdm(total=limit, desc="Kept", unit="img")
    while not done_event.is_set():
        snapshot = await counters.snapshot()
        bar.n = min(snapshot["kept"], limit)
        bar.set_postfix(
            {
                "urls": snapshot["urls_found"],
                "downloaded": snapshot["downloaded"],
                "classified": snapshot["classified"],
                "kept": snapshot["kept"],
                "discarded": snapshot["discarded"],
            }
        )
        bar.refresh()
        await asyncio.sleep(0.5)
    snapshot = await counters.snapshot()
    bar.n = min(snapshot["kept"], limit)
    bar.set_postfix(
        {
            "urls": snapshot["urls_found"],
            "downloaded": snapshot["downloaded"],
            "classified": snapshot["classified"],
            "kept": snapshot["kept"],
            "discarded": snapshot["discarded"],
        }
    )
    bar.refresh()
    bar.close()


async def run_pipeline(config: AppConfig) -> None:
    logger = logging.getLogger(__name__)

    if not config.gemini.api_key:
        raise ValueError("Gemini API key missing. Set GEMINI_API_KEY or config.yaml.")

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    category_dirs = ensure_output_dirs(out_dir)

    db = ImageDB(config.db_path)
    db.init()

    counters = Counters()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=config.queue_size)
    stop_event = asyncio.Event()
    done_event = asyncio.Event()
    stop_lock = asyncio.Lock()
    search_stop_event = asyncio.Event()
    max_urls = max(1, config.limit * config.max_urls_multiplier)
    categories_to_search = config.search_categories or CATEGORIES

    async def request_stop() -> None:
        async with stop_lock:
            if stop_event.is_set():
                return
            stop_event.set()
            for _ in range(config.download_concurrency):
                await queue.put(None)

    async def safe_put(url: str) -> bool:
        while not stop_event.is_set() and not search_stop_event.is_set():
            try:
                queue.put_nowait(url)
                return True
            except asyncio.QueueFull:
                await asyncio.sleep(0.1)
        return False

    seen_urls: set[str] = set()
    seen_lock = asyncio.Lock()

    async def is_new_url(url: str) -> bool:
        async with seen_lock:
            if url in seen_urls:
                return False
            seen_urls.add(url)
            return True

    def is_supported_url(url: str) -> bool:
        parsed = urlparse(url)
        if parsed.hostname and parsed.hostname.lower() in BLOCKED_HOSTS:
            return False
        ext = Path(parsed.path).suffix.lower()
        if ext and ext not in ALLOWED_EXTENSIONS:
            return False
        return True

    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    download_sema = asyncio.Semaphore(config.download_concurrency)
    search_sema = asyncio.Semaphore(config.search_concurrency)
    search_cap_lock = asyncio.Lock()

    async def mark_search_cap() -> None:
        async with search_cap_lock:
            if search_stop_event.is_set():
                return
            search_stop_event.set()
            logger.info("search cap reached urls=%s max_urls=%s", await counters.get("urls_found"), max_urls)

    if config.skip_classify:
        logger.warning("classification disabled; underage filtering is not enforced")

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": config.user_agent,
            "Accept": "image/jpeg,image/png,image/*;q=0.8,*/*;q=0.5",
            "Accept-Language": "en-US,en;q=0.9",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    ) as session:
        classifier: GeminiClassifier | None = None
        if not config.skip_classify:
            classifier = GeminiClassifier(
                api_key=config.gemini.api_key,
                base_url=config.gemini.base_url,
                model=config.gemini.model,
                timeout=config.request_timeout,
                max_retries=config.max_classify_retries,
                semaphore=asyncio.Semaphore(config.classify_concurrency),
            )

        async def search_task(category: str) -> None:
            if search_stop_event.is_set():
                return
            async with search_sema:
                if search_stop_event.is_set():
                    return
                try:
                    urls = await search_live_images(
                        session=session,
                        api_key=config.xai.api_key,
                        base_url=config.xai.base_url,
                        model=config.xai.model_primary,
                        categories=CATEGORIES,
                        target_category=category,
                        seed_queries=config.search_queries,
                        max_results=config.max_results_per_query,
                        timeout=config.request_timeout,
                        max_retries=config.max_search_retries,
                    )
                except Exception as exc:
                    logger.warning("search failed category=%s error=%s", category, exc)
                    return
            for url in urls:
                if stop_event.is_set() or search_stop_event.is_set():
                    break
                if not is_supported_url(url):
                    continue
                if not await is_new_url(url):
                    continue
                if db.has_url(url):
                    continue
                if await safe_put(url):
                    count = await counters.inc("urls_found")
                    if count >= max_urls:
                        await mark_search_cap()
                        break

        async def download_worker(worker_id: int) -> None:
            while True:
                if stop_event.is_set() and queue.empty():
                    break
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue
                if item is None:
                    queue.task_done()
                    break
                url = item
                try:
                    if db.has_url(url):
                        continue
                    if stop_event.is_set():
                        continue

                    data = None
                    content_type = None
                    sniffed_mime = None
                    decoded = False
                    reason = "download_failed"
                    size = None
                    for attempt in range(config.decode_retry_attempts + 1):
                        data, content_type = await fetch_image(
                            url,
                            session,
                            download_sema,
                            timeout,
                            config.max_bytes,
                            config.max_download_retries,
                            config.min_bytes,
                        )
                        if not data:
                            continue
                        await counters.inc("downloaded")
                        valid, reason, size, sniffed_mime, decoded = validate_image_bytes(
                            data, config.min_side
                        )
                        if valid:
                            break
                        if reason != "decode_failed":
                            break
                    if not data:
                        continue

                    phash = None
                    sha256 = None

                    if (
                        not valid
                        and reason == "decode_failed"
                        and sniffed_mime in CLASSIFY_SUPPORTED_MIME
                        and config.allow_undecoded
                    ):
                        valid = True
                        decoded = False
                        reason = "decode_failed_allowed"
                    if not valid:
                        db.insert_record(
                            url,
                            sha256,
                            phash,
                            None,
                            "discard",
                            reason,
                            0.0,
                        )
                        await counters.inc("discarded")
                        if size:
                            logger.info(
                                "discarded invalid image url=%s reason=%s size=%sx%s",
                                url,
                                reason,
                                size[0],
                                size[1],
                            )
                        else:
                            logger.info(
                                "discarded invalid image url=%s reason=%s", url, reason
                            )
                        continue

                    sha256 = compute_sha256(data)
                    if db.has_sha256(sha256):
                        db.insert_record(url, sha256, None, None, "discard", "dedupe", 1.0)
                        await counters.inc("discarded")
                        continue
                    if decoded and config.min_luma_stddev > 0:
                        stddev = luma_stddev(data)
                        if stddev is not None and stddev < config.min_luma_stddev:
                            db.insert_record(
                                url,
                                sha256,
                                None,
                                None,
                                "discard",
                                "low_variance",
                                0.0,
                            )
                            await counters.inc("discarded")
                            logger.info(
                                "discarded low variance url=%s stddev=%.2f",
                                url,
                                stddev,
                            )
                            continue
                    if sniffed_mime and (
                        not content_type
                        or content_type in {"application/octet-stream", "binary/octet-stream"}
                        or not content_type.startswith("image/")
                    ):
                        content_type = sniffed_mime
                    if not decoded:
                        logger.info(
                            "image decode skipped url=%s reason=%s mime=%s",
                            url,
                            reason,
                            sniffed_mime or "unknown",
                        )

                    if await counters.get("kept") >= config.limit:
                        db.insert_record(
                            url, sha256, phash, None, "discard", "limit_reached", 1.0
                        )
                        await counters.inc("discarded")
                        await request_stop()
                        continue

                    if config.skip_classify:
                        result_label = "other"
                        result_model = "unclassified"
                        result_confidence = 1.0
                        await counters.inc("classified")
                    else:
                        classify_bytes = None
                        classify_mime = sniffed_mime or content_type
                        if decoded:
                            classify_bytes = prepare_classification_bytes(data)
                            if classify_bytes is None:
                                decoded = False
                                logger.info(
                                    "image prep failed url=%s reason=decode_failed",
                                    url,
                                )
                        if not decoded:
                            if classify_mime in CLASSIFY_SUPPORTED_MIME:
                                assert classifier is not None
                                try:
                                    result = await classifier.classify(
                                        session, data, classify_mime
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "classification failed url=%s error=%s", url, exc
                                    )
                                    result_label = "other"
                                    result_model = "classify_error"
                                    result_confidence = 0.0
                                    await counters.inc("classified")
                                else:
                                    await counters.inc("classified")

                                    if result.is_underage_or_ambiguous or result.label == "discard":
                                        db.insert_record(
                                            url,
                                            sha256,
                                            phash,
                                            None,
                                            "discard",
                                            config.gemini.model,
                                            result.confidence,
                                        )
                                        await counters.inc("discarded")
                                        continue

                                    result_label = result.label
                                    result_model = config.gemini.model
                                    result_confidence = result.confidence
                            else:
                                result_label = "other"
                                result_model = "undecoded"
                                result_confidence = 1.0
                                await counters.inc("classified")
                        else:
                            assert classifier is not None
                            try:
                                result = await classifier.classify(
                                    session, classify_bytes or data, classify_mime
                                )
                            except Exception as exc:
                                logger.warning(
                                    "classification failed url=%s error=%s", url, exc
                                )
                                result_label = "other"
                                result_model = "classify_error"
                                result_confidence = 0.0
                                await counters.inc("classified")
                            else:
                                await counters.inc("classified")

                                if result.is_underage_or_ambiguous or result.label == "discard":
                                    db.insert_record(
                                        url,
                                        sha256,
                                        phash,
                                        None,
                                        "discard",
                                        config.gemini.model,
                                        result.confidence,
                                    )
                                    await counters.inc("discarded")
                                    continue

                                result_label = result.label
                                result_model = config.gemini.model
                                result_confidence = result.confidence

                    reserved = await counters.reserve_kept(config.limit)
                    if not reserved:
                        db.insert_record(
                            url, sha256, phash, None, "discard", "limit_reached", 1.0
                        )
                        await counters.inc("discarded")
                        await request_stop()
                        continue

                    try:
                        filepath = save_image(
                            data, category_dirs[result_label], sha256, url, content_type
                        )
                    except Exception:
                        await counters.release_kept()
                        raise

                    db.insert_record(
                        url,
                        sha256,
                        phash,
                        filepath,
                        result_label,
                        result_model,
                        result_confidence,
                    )

                    if await counters.get("kept") >= config.limit:
                        await request_stop()
                except Exception:
                    logger.exception("worker %s failed url=%s", worker_id, url)
                finally:
                    queue.task_done()

        progress_task = asyncio.create_task(progress_loop(counters, done_event, config.limit))

        search_tasks = [
            asyncio.create_task(search_task(category)) for category in categories_to_search
        ]
        download_tasks = [
            asyncio.create_task(download_worker(i))
            for i in range(config.download_concurrency)
        ]

        await asyncio.gather(*search_tasks, return_exceptions=True)

        if not stop_event.is_set():
            for _ in range(config.download_concurrency):
                await queue.put(None)

        await asyncio.gather(*download_tasks, return_exceptions=True)
        done_event.set()
        await progress_task

    if config.manifest:
        records = db.all_records()
        manifest = {
            "generated_at": utc_now_iso(),
            "count": len(records),
            "items": records,
        }
        (out_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    db.close()
