from __future__ import annotations

import logging
import random
import time

try:
    from ddgs import DDGS
    from ddgs.exceptions import RatelimitException
except Exception:  # pragma: no cover - fallback for older package
    from duckduckgo_search import DDGS
    try:
        from duckduckgo_search.exceptions import RatelimitException
    except Exception:  # pragma: no cover - last-resort fallback
        class RatelimitException(Exception):
            pass


DEFAULT_QUERIES = [
    "Tifa Lockhart FF7",
    "Tifa Lockhart wallpaper 4k",
    "Tifa Lockhart cosplay",
    "\u30c6\u30a3\u30d5\u30a1 \u30ed\u30c3\u30af\u30cf\u30fc\u30c8",
]

VARIANT_SUFFIXES = [
    "fanart",
    "2d art",
    "3d render",
    "key art",
    "screenshot",
    "official art",
    "wallpaper",
    "nsfw",
    "sfw",
    "remake",
]

FILETYPE_SUFFIXES = [
    "filetype:jpg",
    "filetype:png",
]

EXTRA_QUERIES = [
    "Tifa Lockhart FFVII remake",
    "Tifa Lockhart key art",
    "Tifa Lockhart portrait",
    "Tifa Lockhart 3d render",
]


def generate_queries(base_queries: list[str]) -> list[str]:
    queries: list[str] = []
    for query in base_queries:
        queries.append(query)
        if query.isascii():
            for suffix in VARIANT_SUFFIXES:
                queries.append(f"{query} {suffix}")
            for suffix in FILETYPE_SUFFIXES:
                queries.append(f"{query} {suffix}")
    queries.extend(EXTRA_QUERIES)

    seen = set()
    ordered: list[str] = []
    for query in queries:
        if query not in seen:
            seen.add(query)
            ordered.append(query)
    return ordered


def search_ddg_images(
    query: str, max_results: int, stop_flag, max_retries: int = 3
) -> list[str]:
    logger = logging.getLogger(__name__)

    for attempt in range(max_retries):
        try:
            urls: list[str] = []
            with DDGS() as ddgs:
                    for result in ddgs.images(
                        query,
                        safesearch="off",
                        max_results=max_results,
                    ):
                        if stop_flag.is_set():
                            break
                        url = result.get("image")
                        if url:
                            urls.append(url)
            return urls
        except RatelimitException as exc:
            if attempt == max_retries - 1:
                logger.warning("search rate-limited query=%s error=%s", query, exc)
                return []
            delay = (2 ** attempt) + random.uniform(1.0, 2.0)
            time.sleep(delay)
        except Exception as exc:
            if attempt == max_retries - 1:
                logger.warning("search failed query=%s error=%s", query, exc)
                return []
            delay = (2 ** attempt) + random.random()
            time.sleep(delay)
    return []
