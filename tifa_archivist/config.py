from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml

# Load .env if present (harmless if python-dotenv is not installed).
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv(override=False)   # env vars already in the shell take precedence
except ImportError:
    pass


# Defaults bias toward high-quality sources.  Low-quality anchors like bare
# "cosplay" / "fanart" / "nsfw" are dropped from defaults; set
# `search_queries` in config.yaml to restore them if needed.
DEFAULT_SEARCH_QUERIES = [
    "Tifa Lockhart FF7 official art",
    "Tifa Lockhart FFVII Remake key art",
    "Tifa Lockhart 4k wallpaper",
    "Tifa Lockhart 3d render",
    "Tifa Lockhart promotional art",
    "\u30c6\u30a3\u30d5\u30a1 \u30ed\u30c3\u30af\u30cf\u30fc\u30c8",  # ティファ ロックハート
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
    model: str = "gemini-2.0-flash"


@dataclass
class AppConfig:
    # --- Core pipeline ---
    out_dir: Path = Path("tifa_dataset")
    limit: int = 20
    db_path: Path = Path("images.db")
    log_dir: Path = Path("logs")
    manifest: bool = True

    # --- Concurrency ---
    download_concurrency: int = 8
    classify_concurrency: int = 3
    search_concurrency: int = 2

    # --- Search ---
    max_results_per_query: int = 80
    max_urls_multiplier: int = 2
    search_queries: list[str] = field(default_factory=lambda: list(DEFAULT_SEARCH_QUERIES))

    # --- Adaptive search / source intelligence ---
    adaptive_search_enabled: bool = True
    source_intelligence_path: Path = Path("source_intelligence.json")

    # --- Classification ---
    skip_classify: bool = False
    max_classify_retries: int = 2

    # --- Download ---
    min_bytes: int = 10_000  # Reject obvious thumbnails.  Raise to 50_000 to be stricter.
    max_bytes: int = 15_000_000
    request_timeout: float = 20.0
    max_download_retries: int = 3
    max_search_retries: int = 3
    queue_size: int = 200
    user_agent: str = "tifa-archivist/0.1"

    # --- Dedup ---
    phash_distance: int = 6

    # --- Quality-audit thresholds (used by `audit` command) ---
    min_luma_stddev: float = 10.0
    min_sat_stddev: float = 5.0
    min_sat_mean: float = 10.0
    min_colorfulness: float = 5.0
    flat_tile_ratio: float = 0.7       # flag image if flat_ratio >= this
    min_laplacian_var: float = 10.0
    min_edge_density: float = 0.01
    min_entropy: float = 3.0
    variance_max_side: int = 512       # resize to this before computing metrics
    flat_grid_size: int = 16           # NxN grid for flat-tile detection
    flat_tile_stddev_max: float = 8.0  # stddev below this → tile is "flat"
    edge_threshold: float = 10.0      # Laplacian magnitude threshold for edge px

    # --- API configs ---
    xai: XAIConfig = field(default_factory=XAIConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)


DEFAULT_CONFIG_PATH = Path("config.yaml")


def load_config(path: Path | None) -> AppConfig:
    data = {}
    if path and path.exists():
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    # xAI
    xai_data = data.get("xai", {})
    api_key = os.getenv("XAI_API_KEY") or xai_data.get("api_key", "")
    xai = XAIConfig(
        api_key=api_key,
        base_url=xai_data.get("base_url", XAIConfig.base_url),
        model_primary=xai_data.get("model_primary", XAIConfig.model_primary),
        model_fallback=xai_data.get("model_fallback", XAIConfig.model_fallback),
    )

    # Gemini
    gemini_data = data.get("gemini", {})
    gemini_api_key = os.getenv("GEMINI_API_KEY") or gemini_data.get("api_key", "")
    gemini = GeminiConfig(
        api_key=gemini_api_key,
        base_url=gemini_data.get("base_url", GeminiConfig.base_url),
        model=gemini_data.get("model", GeminiConfig.model),
    )

    out_dir = Path(data.get("out_dir", AppConfig.out_dir))
    db_path = Path(data.get("db_path", out_dir / "images.db"))
    log_dir = Path(data.get("log_dir", AppConfig.log_dir))
    source_intelligence_path = Path(
        data.get("source_intelligence_path", out_dir / "source_intelligence.json")
    )

    return AppConfig(
        out_dir=out_dir,
        limit=int(data.get("limit", AppConfig.limit)),
        db_path=db_path,
        log_dir=log_dir,
        manifest=bool(data.get("manifest", AppConfig.manifest)),
        download_concurrency=int(
            data.get("download_concurrency", AppConfig.download_concurrency)
        ),
        classify_concurrency=int(
            data.get("classify_concurrency", AppConfig.classify_concurrency)
        ),
        search_concurrency=int(
            data.get("search_concurrency", AppConfig.search_concurrency)
        ),
        max_results_per_query=int(
            data.get("max_results_per_query", AppConfig.max_results_per_query)
        ),
        max_urls_multiplier=int(
            data.get("max_urls_multiplier", AppConfig.max_urls_multiplier)
        ),
        skip_classify=bool(data.get("skip_classify", AppConfig.skip_classify)),
        min_bytes=int(data.get("min_bytes", AppConfig.min_bytes)),
        search_queries=list(data.get("search_queries", DEFAULT_SEARCH_QUERIES)),
        phash_distance=int(data.get("phash_distance", AppConfig.phash_distance)),
        request_timeout=float(data.get("request_timeout", AppConfig.request_timeout)),
        max_bytes=int(data.get("max_bytes", AppConfig.max_bytes)),
        queue_size=int(data.get("queue_size", AppConfig.queue_size)),
        max_download_retries=int(
            data.get("max_download_retries", AppConfig.max_download_retries)
        ),
        max_classify_retries=int(
            data.get("max_classify_retries", AppConfig.max_classify_retries)
        ),
        max_search_retries=int(
            data.get("max_search_retries", AppConfig.max_search_retries)
        ),
        user_agent=str(data.get("user_agent", AppConfig.user_agent)),
        adaptive_search_enabled=bool(
            data.get("adaptive_search_enabled", AppConfig.adaptive_search_enabled)
        ),
        source_intelligence_path=source_intelligence_path,
        min_luma_stddev=float(data.get("min_luma_stddev", AppConfig.min_luma_stddev)),
        min_sat_stddev=float(data.get("min_sat_stddev", AppConfig.min_sat_stddev)),
        min_sat_mean=float(data.get("min_sat_mean", AppConfig.min_sat_mean)),
        min_colorfulness=float(data.get("min_colorfulness", AppConfig.min_colorfulness)),
        flat_tile_ratio=float(data.get("flat_tile_ratio", AppConfig.flat_tile_ratio)),
        min_laplacian_var=float(data.get("min_laplacian_var", AppConfig.min_laplacian_var)),
        min_edge_density=float(data.get("min_edge_density", AppConfig.min_edge_density)),
        min_entropy=float(data.get("min_entropy", AppConfig.min_entropy)),
        variance_max_side=int(data.get("variance_max_side", AppConfig.variance_max_side)),
        flat_grid_size=int(data.get("flat_grid_size", AppConfig.flat_grid_size)),
        flat_tile_stddev_max=float(
            data.get("flat_tile_stddev_max", AppConfig.flat_tile_stddev_max)
        ),
        edge_threshold=float(data.get("edge_threshold", AppConfig.edge_threshold)),
        xai=xai,
        gemini=gemini,
    )
