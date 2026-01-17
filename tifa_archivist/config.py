from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml


DEFAULT_SEARCH_QUERIES = [
    "Tifa Lockhart FF7",
    "Tifa Lockhart wallpaper 4k",
    "Tifa Lockhart cosplay",
    "\u30c6\u30a3\u30d5\u30a1 \u30ed\u30c3\u30af\u30cf\u30fc\u30c8",
]


@dataclass
class XAIConfig:
    api_key: str = ""
    base_url: str = "https://api.x.ai/v1"
    model_primary: str = "grok-4-1-fast-non-reasoning"
    model_fallback: str = "grok-2-vision-latest"


@dataclass
class AppConfig:
    out_dir: Path = Path("tifa_dataset")
    limit: int = 20
    db_path: Path = Path("images.db")
    log_dir: Path = Path("logs")
    manifest: bool = True
    download_concurrency: int = 8
    classify_concurrency: int = 3
    max_results_per_query: int = 80
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))
    phash_distance: int = 6
    request_timeout: float = 20.0
    max_bytes: int = 15_000_000
    queue_size: int = 200
    max_download_retries: int = 3
    max_classify_retries: int = 2
    max_search_retries: int = 3
    user_agent: str = "tifa-archivist/0.1"
    xai: XAIConfig = field(default_factory=XAIConfig)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None) -> AppConfig:
    data = {}
    if path and path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    xai_data = data.get("xai", {})
    api_key = os.getenv("XAI_API_KEY") or xai_data.get("api_key", "")
    xai = XAIConfig(
        api_key=api_key,
        base_url=xai_data.get("base_url", XAIConfig.base_url),
        model_primary=xai_data.get("model_primary", XAIConfig.model_primary),
        model_fallback=xai_data.get("model_fallback", XAIConfig.model_fallback),
    )

    out_dir = Path(data.get("out_dir", AppConfig.out_dir))
    db_path = Path(data.get("db_path", out_dir / "images.db"))
    log_dir = Path(data.get("log_dir", AppConfig.log_dir))

    return AppConfig(
        out_dir=out_dir,
        limit=int(data.get("limit", AppConfig.limit)),
        db_path=db_path,
        log_dir=log_dir,
        manifest=bool(data.get("manifest", AppConfig.manifest)),
        download_concurrency=int(data.get("download_concurrency", AppConfig.download_concurrency)),
        classify_concurrency=int(data.get("classify_concurrency", AppConfig.classify_concurrency)),
        max_results_per_query=int(data.get("max_results_per_query", AppConfig.max_results_per_query)),
        search_queries=list(data.get("search_queries", DEFAULT_SEARCH_QUERIES)),
        phash_distance=int(data.get("phash_distance", AppConfig.phash_distance)),
        request_timeout=float(data.get("request_timeout", AppConfig.request_timeout)),
        max_bytes=int(data.get("max_bytes", AppConfig.max_bytes)),
        queue_size=int(data.get("queue_size", AppConfig.queue_size)),
        max_download_retries=int(data.get("max_download_retries", AppConfig.max_download_retries)),
        max_classify_retries=int(data.get("max_classify_retries", AppConfig.max_classify_retries)),
        max_search_retries=int(data.get("max_search_retries", AppConfig.max_search_retries)),
        user_agent=str(data.get("user_agent", AppConfig.user_agent)),
        xai=xai,
    )
