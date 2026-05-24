from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
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
from .utils import image_quality_metrics, slugify, utc_now_iso


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
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Hosts that consistently serve garbage (403, thumbnails, or near-duplicate floods).
# Extend via CODINGPROGRESS.MD when new offenders emerge.
BLOCKED_HOSTS = {
    # Requires login to serve full-size images; DDG returns auth-gated thumbnails.
    "zerochan.net",
    # Product-mockup storefronts — images are merchandise renders, not fan art.
    "redbubble.com",
    "merch.amazon.com",
    # DeviantArt CDN: JWTs expire within hours; DDG caches stale URLs.
    "wixmp.com",
    # Old DeviantArt image CDN — stale direct links.
    "deviantart.net",
    # Old NewGrounds CDN — frequent 404s on cached links.
    "ngfiles.com",
    # Rule-34 / eroge-scraper sites.
    "rule34.xxx",
    "rule34.paheal.net",
    "paheal.net",
    # Preview/promo sites with paywall full-res.
    "previewsworld.com",
    "cdn.donmai.us",
    # Poster / canvas print storefronts — serve product mockup JPEGs.
    "kaiteez.com",
    "byztee.com",
    "wallpaperaccess.com",
    # Reactor image CDN — low-res reposts.
    "reactor.cc",
}

# URL-path substrings that almost always indicate a thumbnail/preview/avatar.
_THUMB_PATH_MARKERS = (
    "/thumb",
    "/thumbs/",
    "/preview",
    "preview_",
    "_small",
    "_thumb",
    "/avatar",
    "/icon",
    "/sq/",
)
# Query-string size hints (e.g. "?w=150&h=150").  Values < 400 are treated as thumbnails.
_THUMB_SIZE_QUERY = re.compile(r"(?:w|h|size|sz)=(\d{1,4})", re.IGNORECASE)


def _normalize_host(netloc: str) -> str:
    host = netloc.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


def _looks_like_thumbnail(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    for marker in _THUMB_PATH_MARKERS:
        if marker in path:
            return True
    if parsed.query:
        for match in _THUMB_SIZE_QUERY.finditer(parsed.query):
            try:
                if int(match.group(1)) < 400:
                    return True
            except ValueError:
                continue
    return False


def _is_blocked_host(netloc: str) -> bool:
    """Return True if netloc matches any entry in BLOCKED_HOSTS (exact or subdomain)."""
    host = _normalize_host(netloc)
    return host in BLOCKED_HOSTS or any(
        host.endswith("." + blocked) for blocked in BLOCKED_HOSTS
    )


def is_good_url(url: str) -> bool:
    """Cheap URL-level filter: extension, host blocklist, thumbnail patterns."""
    parsed = urlparse(url)
    ext = Path(parsed.path).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False
    if _is_blocked_host(parsed.netloc):
        return False
    if _looks_like_thumbnail(url):
        return False
    return True


def _is_low_quality(
    config: AppConfig,
    metrics: tuple[
        float | None, float | None, float | None, float | None,
        float | None, float | None, float | None, float | None,
    ],
) -> tuple[bool, list[str]]:
    """Mirror of quality_audit._flag_metrics, used as runtime gate.

    Returns (is_low_quality, flags).  Flag combinations intentionally match
    the thresholds tuned in CODINGPROGRESS.MD (Feb 2026).
    """
    (
        luma_stddev,
        sat_stddev,
        sat_mean,
        colorfulness,
        flat_ratio,
        laplacian_var,
        edge_density,
        entropy,
    ) = metrics

    if luma_stddev is None or sat_stddev is None or sat_mean is None:
        return True, ["metrics_missing"]

    flags: list[str] = []
    low_luma = luma_stddev < config.min_luma_stddev
    low_sat_std = sat_stddev < config.min_sat_stddev
    low_sat_mean = sat_mean < config.min_sat_mean
    low_color = colorfulness is not None and colorfulness < config.min_colorfulness
    high_flat_ratio = flat_ratio is not None and flat_ratio >= config.flat_tile_ratio
    low_lap = laplacian_var is not None and laplacian_var < config.min_laplacian_var
    low_edge = edge_density is not None and edge_density < config.min_edge_density
    low_entropy = entropy is not None and entropy < config.min_entropy

    if low_luma and low_sat_std:
        flags.append("low_luma+low_sat_std")
    if low_sat_mean and low_sat_std:
        flags.append("low_sat_mean+low_sat_std")
    if low_color and low_sat_std:
        flags.append("low_color+low_sat_std")
    # flat_ratio and low_entropy alone are too noisy — official art commonly has
    # solid backgrounds.  Only reject when they fire together, or alongside a
    # sharpness failure.
    if high_flat_ratio and low_entropy:
        flags.append("flat_ratio+low_entropy")
    if high_flat_ratio and low_lap and low_edge:
        flags.append("flat_ratio+blurry")
    if low_lap and low_edge:
        flags.append("low_lap+low_edge")

    return len(flags) > 0, flags


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

    if not config.skip_classify and not config.xai.api_key:
        raise ValueError("XAI API key missing. Set XAI_API_KEY or config.yaml.")

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
    stop_flag = threading.Event()
    stop_lock = asyncio.Lock()
    search_stop_event = asyncio.Event()
    max_urls = max(1, config.limit * config.max_urls_multiplier)

    async def request_stop() -> None:
        async with stop_lock:
            if stop_event.is_set():
                return
            stop_event.set()
            stop_flag.set()
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

    phash_index = PhashIndex(db.get_all_phashes(), threshold=config.phash_distance)
    phash_lock = asyncio.Lock()

    timeout = aiohttp.ClientTimeout(total=config.request_timeout)
    download_sema = asyncio.Semaphore(config.download_concurrency)
    search_sema = asyncio.Semaphore(config.search_concurrency)
    search_cap_lock = asyncio.Lock()

    async def mark_search_cap() -> None:
        async with search_cap_lock:
            if search_stop_event.is_set():
                return
            search_stop_event.set()
            stop_flag.set()
            logger.info("search cap reached urls=%s max_urls=%s", await counters.get("urls_found"), max_urls)

    if config.skip_classify:
        logger.warning("classification disabled; underage filtering is not enforced")

    async with aiohttp.ClientSession(
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/jpeg,image/png,image/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Referer": "https://duckduckgo.com/",
        }
    ) as session:
        classifier: XAIClassifier | None = None
        if not config.skip_classify:
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
            if search_stop_event.is_set():
                return
            async with search_sema:
                if search_stop_event.is_set():
                    return
                urls = await asyncio.to_thread(
                    search_ddg_images,
                    query,
                    config.max_results_per_query,
                    stop_flag,
                    config.max_search_retries,
                )
            for url in urls:
                if stop_event.is_set() or search_stop_event.is_set():
                    break
                if not is_good_url(url):
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

                    sha256 = compute_sha256(data)
                    if db.has_sha256(sha256):
                        db.insert_record(url, sha256, None, None, "discard", "dedupe", 1.0)
                        await counters.inc("discarded")
                        continue

                    # --- Quality gate ---------------------------------------
                    # Step 1: decode-success + CV metrics.  A None luma_stddev
                    # means PIL couldn't parse the bytes at all.
                    metrics = await asyncio.to_thread(
                        image_quality_metrics,
                        data,
                        config.variance_max_side,
                        config.flat_grid_size,
                        config.flat_tile_stddev_max,
                        config.edge_threshold,
                    )
                    if metrics[0] is None:
                        db.insert_record(
                            url, sha256, None, None, "discard", "decode_failed", 0.0
                        )
                        await counters.inc("discarded")
                        continue

                    is_low, low_flags = _is_low_quality(config, metrics)
                    if is_low:
                        reason = "low_quality:" + "|".join(low_flags)
                        db.insert_record(url, sha256, None, None, "discard", reason, 0.0)
                        await counters.inc("discarded")
                        continue

                    # Step 2: perceptual-hash near-duplicate check.  Lock is
                    # held across check+add so two workers can't both admit
                    # the same near-dup image racing each other.
                    phash = await asyncio.to_thread(compute_phash, data)
                    if phash is not None:
                        async with phash_lock:
                            if phash_index.is_similar(phash):
                                db.insert_record(
                                    url, sha256, phash, None, "discard",
                                    "near_duplicate", 1.0,
                                )
                                await counters.inc("discarded")
                                continue
                            phash_index.add(phash)
                    # --------------------------------------------------------

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
                        assert classifier is not None
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
                            continue

                        result_label = result.label
                        result_model = classifier.model_name
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
