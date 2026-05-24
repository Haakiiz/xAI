"""Runtime quality-gate unit tests.

Covers the three pure helpers in `orchestrator.py`:
  - `is_good_url`
  - `_looks_like_thumbnail`
  - `_is_low_quality` (paired with `utils.image_quality_metrics`)
"""
from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image

from tifa_archivist.config import AppConfig
from tifa_archivist.orchestrator import (
    BLOCKED_HOSTS,
    _is_low_quality,
    _looks_like_thumbnail,
    is_good_url,
)
from tifa_archivist.utils import image_quality_metrics


# --- Helpers ----------------------------------------------------------------


def _jpeg_from_array(arr: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _flat_gray_image(size: int = 256, value: int = 128) -> bytes:
    arr = np.full((size, size, 3), value, dtype=np.uint8)
    return _jpeg_from_array(arr)


def _textured_image(size: int = 256) -> bytes:
    # Deterministic noisy pattern: lots of edges, high entropy, no flat tiles.
    rng = np.random.default_rng(42)
    base = rng.integers(40, 220, size=(size, size, 3), dtype=np.int32)
    yy, xx = np.mgrid[0:size, 0:size]
    base[:, :, 0] = (base[:, :, 0] + (xx % 64)) % 256
    base[:, :, 1] = (base[:, :, 1] + (yy % 64)) % 256
    base[:, :, 2] = (base[:, :, 2] + ((xx ^ yy) % 64)) % 256
    return _jpeg_from_array(base)


# --- URL filter -------------------------------------------------------------


def test_is_good_url_accepts_normal_jpg() -> None:
    assert is_good_url("https://example.com/images/tifa.jpg")


def test_is_good_url_rejects_non_image_extension() -> None:
    assert not is_good_url("https://example.com/page.html")
    assert not is_good_url("https://example.com/image.gif")
    # .webp is now accepted (servers like Reddit return webp for .png URLs via ?auto=webp)
    assert is_good_url("https://example.com/image.webp")


def test_is_good_url_rejects_blocked_hosts() -> None:
    for host in BLOCKED_HOSTS:
        assert not is_good_url(f"https://{host}/tifa.jpg")
        assert not is_good_url(f"https://www.{host}/tifa.jpg")


def test_is_good_url_rejects_thumbnail_paths() -> None:
    assert not is_good_url("https://example.com/thumb/tifa.jpg")
    assert not is_good_url("https://example.com/thumbs/tifa.jpg")
    assert not is_good_url("https://example.com/preview_tifa.jpg")
    assert not is_good_url("https://example.com/tifa_small.jpg")
    assert not is_good_url("https://cdn.example.com/avatar/user.jpg")


def test_is_good_url_rejects_small_size_query() -> None:
    assert _looks_like_thumbnail("https://example.com/tifa.jpg?w=150&h=150")
    assert _looks_like_thumbnail("https://example.com/tifa.jpg?size=200")
    # 800 is large enough → not a thumbnail
    assert not _looks_like_thumbnail("https://example.com/tifa.jpg?w=800&h=600")


# --- Quality gate -----------------------------------------------------------


def test_quality_gate_rejects_flat_gray_image() -> None:
    cfg = AppConfig()
    data = _flat_gray_image()
    metrics = image_quality_metrics(
        data,
        cfg.variance_max_side,
        cfg.flat_grid_size,
        cfg.flat_tile_stddev_max,
        cfg.edge_threshold,
    )
    is_low, flags = _is_low_quality(cfg, metrics)
    assert is_low is True
    assert flags  # at least one flag fired (flat_ratio, low_luma+low_sat_std, etc.)


def test_quality_gate_accepts_textured_image() -> None:
    cfg = AppConfig()
    data = _textured_image()
    metrics = image_quality_metrics(
        data,
        cfg.variance_max_side,
        cfg.flat_grid_size,
        cfg.flat_tile_stddev_max,
        cfg.edge_threshold,
    )
    is_low, flags = _is_low_quality(cfg, metrics)
    assert is_low is False, f"expected textured image to pass, got flags={flags}"


def test_quality_gate_reports_decode_failure() -> None:
    cfg = AppConfig()
    # Garbage bytes → image_quality_metrics returns all None → gate must reject.
    metrics = image_quality_metrics(
        b"not an image at all",
        cfg.variance_max_side,
        cfg.flat_grid_size,
        cfg.flat_tile_stddev_max,
        cfg.edge_threshold,
    )
    assert metrics[0] is None
    is_low, flags = _is_low_quality(cfg, metrics)
    assert is_low is True
    assert "metrics_missing" in flags
