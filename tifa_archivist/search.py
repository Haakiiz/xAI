from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any

import aiohttp
from pydantic import BaseModel, ConfigDict, Field

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

EXCLUDED_SITES = [
    "rare-gallery.com",
    "wallpapersden.com",
    "images.wallpapersden.com",
    "cdn.donmai.us",
    "donmai.us",
    "tulip.paheal.net",
    "lotus.paheal.net",
    "rule34.xxx",
    "us.rule34.xxx",
]

QUERY_SYSTEM_PROMPT = (
    "You create a single web image search query. Do not include URLs."
)

LIVE_SEARCH_SYSTEM_PROMPT = (
    "You are a web image search assistant. Use live search to find direct image URLs "
    "that are publicly accessible without cookies or referer. Create the search query yourself."
)


class LiveSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    query: str
    urls: list[str] = Field(default_factory=list)


class SearchQueryResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    query: str


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


def build_query_prompt(
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
) -> str:
    categories_text = ", ".join(categories)
    seeds_text = ", ".join(seed_queries) if seed_queries else "none"
    return (
        "Create one concise web image search query for the target category.\n"
        "Requirements: include 'Tifa Lockhart' and 'Final Fantasy VII'.\n"
        "Avoid sites that block hotlinking.\n"
        f"Categories to choose from: {categories_text}\n"
        f"Target category: {target_category}\n"
        f"Seed phrases (optional): {seeds_text}\n"
        "Return ONLY JSON with this shape:\n"
        '{"category":"...","query":"..."}'
    )


def build_live_search_prompt(
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
    max_results: int,
) -> str:
    categories_text = ", ".join(categories)
    seeds_text = ", ".join(seed_queries) if seed_queries else "none"
    return (
        "Find images of Tifa Lockhart (Final Fantasy VII).\n"
        f"Categories to choose from: {categories_text}\n"
        f"Target category for this search: {target_category}\n"
        f"Seed phrases (optional): {seeds_text}\n"
        "Create a single concise search query tailored to the target category and use live "
        "search to collect direct image URLs.\n"
        "Prefer full-resolution images (avoid thumbnails/previews like thumb, small, preview, "
        "avatar, icon, or size-suffixed URLs).\n"
        f"Return up to {max_results} results. Only include direct .jpg/.jpeg/.png/.webp URLs.\n"
        "Return ONLY JSON with this shape:\n"
        '{"category":"...","query":"...","urls":["https://...jpg", "..."]}'
    )


def search_ddg_images(
    query: str,
    max_results: int,
    stop_flag=None,
    max_retries: int = 3,
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
                        if stop_flag is not None and stop_flag.is_set():
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


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise exc


def _schema_unsupported(message: str) -> bool:
    msg = message.lower()
    return "response_format" in msg or "json_schema" in msg or (
        "schema" in msg and "unsupported" in msg
    )


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


def _parse_query_response(data: dict[str, Any]) -> SearchQueryResult:
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
    return SearchQueryResult.model_validate(payload)


def _parse_live_search_response(data: dict[str, Any]) -> LiveSearchResult:
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
    return LiveSearchResult.model_validate(payload)


def _dedupe_urls(urls: list[str], max_results: int) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for url in urls:
        if not isinstance(url, str):
            continue
        candidate = url.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
        if len(cleaned) >= max_results:
            break
    return cleaned


def _build_query_payload(
    model: str,
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
    use_schema: bool,
) -> dict[str, Any]:
    schema = SearchQueryResult.model_json_schema()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_query_prompt(categories, target_category, seed_queries),
            },
        ],
        "temperature": 0.2,
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "search_query", "schema": schema, "strict": True},
        }
    return payload


def _build_live_search_payload(
    model: str,
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
    max_results: int,
    use_schema: bool,
) -> dict[str, Any]:
    schema = LiveSearchResult.model_json_schema()
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": LIVE_SEARCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_live_search_prompt(
                    categories, target_category, seed_queries, max_results
                ),
            },
        ],
        "temperature": 0.2,
        "search_parameters": {"mode": "auto", "max_results": max_results},
    }
    if use_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "live_search", "schema": schema, "strict": True},
        }
    return payload


async def _post_live_search(
    session: aiohttp.ClientSession,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
        text = await resp.text()
        if resp.status >= 400:
            message = _extract_error_message(text)
            raise RuntimeError(f"XAI API error {resp.status}: {message}")
        return json.loads(text)


def _fallback_query(target_category: str) -> str:
    suffix = target_category.replace("_", " ")
    base = f"Tifa Lockhart {suffix} Final Fantasy VII".strip()
    if EXCLUDED_SITES:
        exclusions = " ".join(f"-site:{host}" for host in EXCLUDED_SITES)
        return f"{base} {exclusions}".strip()
    return base


async def generate_search_query(
    session: aiohttp.ClientSession,
    api_key: str,
    base_url: str,
    model: str,
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
    timeout: float,
    max_retries: int,
) -> str:
    if not api_key:
        raise RuntimeError("XAI API key missing")

    use_schema = True
    url = f"{base_url.rstrip('/')}/chat/completions"

    for attempt in range(max_retries + 1):
        try:
            payload = _build_query_payload(
                model,
                categories,
                target_category,
                seed_queries,
                use_schema,
            )
            response = await _post_live_search(session, url, api_key, payload, timeout)
            result = _parse_query_response(response)
            query = result.query.strip()
            if not query:
                raise RuntimeError("empty query")
            lower = query.lower()
            if "tifa" not in lower:
                query = f"Tifa Lockhart {query}".strip()
                lower = query.lower()
            if "ff7" not in lower and "final fantasy" not in lower:
                query = f"{query} Final Fantasy VII"
            if EXCLUDED_SITES:
                exclusions = " ".join(f"-site:{host}" for host in EXCLUDED_SITES)
                query = f"{query} {exclusions}".strip()
            return query
        except RuntimeError as exc:
            if use_schema and _schema_unsupported(str(exc)):
                use_schema = False
                continue
            if attempt >= max_retries:
                raise
            delay = (2 ** attempt) + random.random()
            await asyncio.sleep(delay)
        except Exception:
            if attempt >= max_retries:
                raise
            delay = (2 ** attempt) + random.random()
            await asyncio.sleep(delay)
    return _fallback_query(target_category)


async def search_live_images(
    session: aiohttp.ClientSession,
    api_key: str,
    base_url: str,
    model: str,
    categories: list[str],
    target_category: str,
    seed_queries: list[str],
    max_results: int,
    timeout: float,
    max_retries: int,
) -> list[str]:
    logger = logging.getLogger(__name__)
    try:
        query = await generate_search_query(
            session=session,
            api_key=api_key,
            base_url=base_url,
            model=model,
            categories=categories,
            target_category=target_category,
            seed_queries=seed_queries,
            timeout=timeout,
            max_retries=max_retries,
        )
    except Exception as exc:
        logger.warning(
            "query generation failed category=%s error=%s", target_category, exc
        )
        query = _fallback_query(target_category)

    if not query:
        query = _fallback_query(target_category)

    urls = await asyncio.to_thread(
        search_ddg_images,
        query,
        max_results,
        None,
        max_retries,
    )
    urls = _dedupe_urls(urls, max_results)
    logger.info(
        "image search category=%s query=%s results=%s source=ddg",
        target_category,
        query,
        len(urls),
    )
    return urls
