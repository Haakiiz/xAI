from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import threading
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
from tqdm import tqdm

from .classify import XAIClassifier
from .config import AppConfig
from .db import ImageDB
from .dedupe import PhashIndex, compute_phash, compute_sha256
from .download import fetch_image
from .search import generate_queries, search_ddg_images
from .utils import slugify, utc_now_iso


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

    if not config.xai.api_key:
        raise ValueError("XAI API key missing. Set XAI_API_KEY or config.yaml.")

    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    category_dirs = ensure_output_dirs(out_dir)

    db = ImageDB(config.db_path)
    db.init()

    phash_index = PhashIndex(db.get_all_phashes(), threshold=config.phash_distance)
    phash_lock = asyncio.Lock()

    counters = Counters()
    queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=config.queue_size)
    stop_event = asyncio.Event()
    done_event = asyncio.Event()
    stop_flag = threading.Event()
    stop_lock = asyncio.Lock()

    async def request_stop() -> None:
        async with stop_lock:
            if stop_event.is_set():
                return
            stop_event.set()
            stop_flag.set()
            for _ in range(config.download_concurrency):
                await queue.put(None)

    async def safe_put(url: str) -> bool:
        while not stop_event.is_set():
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

    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    download_sema = asyncio.Semaphore(config.download_concurrency)

    async with aiohttp.ClientSession(headers={"User-Agent": config.user_agent}) as session:
        classifier = XAIClassifier(
            api_key=config.xai.api_key,
            base_url=config.xai.base_url,
            model_primary=config.xai.model_primary,
            model_fallback=config.xai.model_fallback,
            timeout=config.request_timeout,
            max_retries=config.max_classify_retries,
            semaphore=asyncio.Semaphore(config.classify_concurrency),
        )

        await classifier.check_vision_support(session)

        async def search_task(query: str) -> None:
            urls = await asyncio.to_thread(
                search_ddg_images,
                query,
                config.max_results_per_query,
                stop_flag,
                config.max_search_retries,
            )
            for url in urls:
                if stop_event.is_set():
                    break
                if not await is_new_url(url):
                    continue
                if await safe_put(url):
                    await counters.inc("urls_found")

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

                    data, content_type = await fetch_image(
                        url,
                        session,
                        download_sema,
                        timeout,
                        config.max_bytes,
                        config.max_download_retries,
                    )
                    if not data:
                        continue
                    await counters.inc("downloaded")

                    sha256 = compute_sha256(data)
                    if db.has_sha256(sha256):
                        db.insert_record(url, sha256, None, None, "discard", "dedupe", 1.0)
                        await counters.inc("discarded")
                        continue

                    phash = compute_phash(data)
                    if not phash:
                        db.insert_record(
                            url, sha256, None, None, "discard", "decode_failed", 0.0
                        )
                        await counters.inc("discarded")
                        continue

                    async with phash_lock:
                        if phash_index.is_similar(phash):
                            db.insert_record(url, sha256, phash, None, "discard", "dedupe", 1.0)
                            await counters.inc("discarded")
                            continue

                    if await counters.get("kept") >= config.limit:
                        db.insert_record(
                            url, sha256, phash, None, "discard", "limit_reached", 1.0
                        )
                        await counters.inc("discarded")
                        await request_stop()
                        continue

                    try:
                        result = await classifier.classify(session, data)
                    except Exception as exc:
                        logger.warning("classification failed url=%s error=%s", url, exc)
                        db.insert_record(
                            url, sha256, phash, None, "discard", "classify_error", 0.0
                        )
                        await counters.inc("discarded")
                        continue

                    await counters.inc("classified")

                    if result.is_underage_or_ambiguous or result.label == "discard":
                        db.insert_record(
                            url,
                            sha256,
                            phash,
                            None,
                            "discard",
                            classifier.model_name,
                            result.confidence,
                        )
                        await counters.inc("discarded")
                        async with phash_lock:
                            phash_index.add(phash)
                        continue

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
                            data, category_dirs[result.label], sha256, url, content_type
                        )
                    except Exception:
                        await counters.release_kept()
                        raise

                    db.insert_record(
                        url,
                        sha256,
                        phash,
                        filepath,
                        result.label,
                        classifier.model_name,
                        result.confidence,
                    )
                    async with phash_lock:
                        phash_index.add(phash)

                    if await counters.get("kept") >= config.limit:
                        await request_stop()
                except Exception:
                    logger.exception("worker %s failed url=%s", worker_id, url)
                finally:
                    queue.task_done()

        progress_task = asyncio.create_task(progress_loop(counters, done_event, config.limit))

        queries = generate_queries(config.search_queries)
        search_tasks = [asyncio.create_task(search_task(q)) for q in queries]
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
