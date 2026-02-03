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
class GeminiConfig:
    api_key: str = ""
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model: str = "gemini-3-flash-preview"


@dataclass
class AppConfig:
    out_dir: Path = Path("tifa_dataset")
    limit: int = 20
    db_path: Path = Path("images.db")
    log_dir: Path = Path("logs")
    manifest: bool = True
    download_concurrency: int = 3
    classify_concurrency: int = 3
    max_results_per_query: int = 80
    max_urls_multiplier: int = 10
    search_concurrency: int = 2
    skip_classify: bool = False
    min_bytes: int = 15_000
    min_side: int = 512
    search_categories: list[str] | None = None
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))
    phash_distance: int = 6
    request_timeout: float = 20.0
    max_bytes: int = 15_000_000
    queue_size: int = 200
    max_download_retries: int = 3
    max_classify_retries: int = 2
    max_search_retries: int = 3
    decode_retry_attempts: int = 2
    allow_undecoded: bool = False
    allow_truncated: bool = True
    min_luma_stddev: float = 3.0
    min_sat_stddev: float = 2.0
    min_sat_mean: float = 5.0
    min_colorfulness: float = 5.0
    variance_max_side: int = 256
    flat_grid_size: int = 8
    flat_tile_stddev_max: float = 4.0
    flat_tile_ratio: float = 0.35
    classify_max_side: int = 2048
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    xai: XAIConfig = field(default_factory=XAIConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


DEFAULT_CONFIG_PATH = Path("config.yaml")
DEFAULT_ENV_PATH = Path(".env")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path | None) -> AppConfig:
    _load_env_file(DEFAULT_ENV_PATH)
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
    gemini_data = data.get("gemini", {})
    gemini_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or gemini_data.get("api_key", "")
    )
    gemini = GeminiConfig(
        api_key=gemini_key,
        base_url=gemini_data.get("base_url", GeminiConfig.base_url),
        model=gemini_data.get("model", GeminiConfig.model),
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
            max_urls_multiplier=int(data.get("max_urls_multiplier", AppConfig.max_urls_multiplier)),
        search_concurrency=int(data.get("search_concurrency", AppConfig.search_concurrency)),
        skip_classify=bool(data.get("skip_classify", AppConfig.skip_classify)),
        min_bytes=int(data.get("min_bytes", AppConfig.min_bytes)),
        min_side=int(data.get("min_side", AppConfig.min_side)),
        search_categories=list(data["search_categories"])
        if data.get("search_categories")
        else AppConfig.search_categories,
        search_queries=list(data.get("search_queries", DEFAULT_SEARCH_QUERIES)),
        phash_distance=int(data.get("phash_distance", AppConfig.phash_distance)),
        request_timeout=float(data.get("request_timeout", AppConfig.request_timeout)),
        max_bytes=int(data.get("max_bytes", AppConfig.max_bytes)),
        queue_size=int(data.get("queue_size", AppConfig.queue_size)),
        max_download_retries=int(data.get("max_download_retries", AppConfig.max_download_retries)),
        max_classify_retries=int(data.get("max_classify_retries", AppConfig.max_classify_retries)),
        max_search_retries=int(data.get("max_search_retries", AppConfig.max_search_retries)),
        decode_retry_attempts=int(data.get("decode_retry_attempts", AppConfig.decode_retry_attempts)),
        allow_undecoded=bool(data.get("allow_undecoded", AppConfig.allow_undecoded)),
        allow_truncated=bool(data.get("allow_truncated", AppConfig.allow_truncated)),
        min_luma_stddev=float(data.get("min_luma_stddev", AppConfig.min_luma_stddev)),
        min_sat_stddev=float(data.get("min_sat_stddev", AppConfig.min_sat_stddev)),
        min_sat_mean=float(data.get("min_sat_mean", AppConfig.min_sat_mean)),
        min_colorfulness=float(data.get("min_colorfulness", AppConfig.min_colorfulness)),
        variance_max_side=int(data.get("variance_max_side", AppConfig.variance_max_side)),
        flat_grid_size=int(data.get("flat_grid_size", AppConfig.flat_grid_size)),
        flat_tile_stddev_max=float(
            data.get("flat_tile_stddev_max", AppConfig.flat_tile_stddev_max)
        ),
        flat_tile_ratio=float(data.get("flat_tile_ratio", AppConfig.flat_tile_ratio)),
        classify_max_side=int(data.get("classify_max_side", AppConfig.classify_max_side)),
        user_agent=str(data.get("user_agent", AppConfig.user_agent)),
        xai=xai,
        gemini=gemini,
    )
