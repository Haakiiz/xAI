import asyncio
from io import BytesIO

import numpy as np
from PIL import Image

from tifa_archivist.config import AppConfig, XAIConfig
from tifa_archivist.orchestrator import run_pipeline


def _textured_jpeg(seed: int) -> bytes:
    """Deterministic noisy image that passes the runtime quality gate."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, size=(480, 640, 3), dtype=np.int32)
    yy, xx = np.mgrid[0:480, 0:640]
    arr[:, :, 0] = (arr[:, :, 0] + (xx % 64)) % 256
    arr[:, :, 1] = (arr[:, :, 1] + (yy % 64)) % 256
    arr[:, :, 2] = (arr[:, :, 2] + ((xx ^ yy) % 64)) % 256
    buf = BytesIO()
    Image.fromarray(arr.astype("uint8")).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _flat_gray_jpeg() -> bytes:
    """Solid-gray image that the quality gate should reject as low_quality."""
    arr = np.full((480, 640, 3), 128, dtype=np.uint8)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_run_pipeline_smoke(tmp_path, monkeypatch) -> None:
    """Three good textured images go in → two are kept (limit=2)."""
    def fake_search_ddg_images(query, max_results, stop_flag, max_retries):
        return [
            "https://example.com/a.jpg",
            "https://example.com/b.jpg",
            "https://example.com/c.jpg",
        ]

    async def fake_fetch_image(
        url, session, sema, timeout, max_bytes, max_retries, min_bytes
    ):
        # Last char of url picks a different seed so each image is unique.
        seed = ord(url[-5]) + ord(url[-6])
        return _textured_jpeg(seed), "image/jpeg"

    monkeypatch.setattr(
        "tifa_archivist.orchestrator.search_ddg_images", fake_search_ddg_images
    )
    monkeypatch.setattr(
        "tifa_archivist.orchestrator.fetch_image", fake_fetch_image
    )

    out_dir = tmp_path / "out"
    config = AppConfig(
        out_dir=out_dir,
        db_path=out_dir / "images.db",
        log_dir=tmp_path / "logs",
        manifest=False,
        limit=2,
        skip_classify=True,
        min_bytes=1,
        max_bytes=10_000_000,
        max_download_retries=1,
        max_search_retries=1,
        search_queries=["Tifa Lockhart FF7"],
        adaptive_search_enabled=False,
        source_intelligence_path=out_dir / "source_intelligence.json",
        xai=XAIConfig(api_key="test"),
    )

    asyncio.run(run_pipeline(config))

    saved = list((out_dir / "other").glob("*"))
    assert len(saved) == 2


def test_run_pipeline_rejects_flat_gray(tmp_path, monkeypatch) -> None:
    """Flat-gray images should be rejected by the quality gate → nothing saved."""
    def fake_search_ddg_images(query, max_results, stop_flag, max_retries):
        return [
            "https://example.com/flat_a.jpg",
            "https://example.com/flat_b.jpg",
            "https://example.com/flat_c.jpg",
        ]

    async def fake_fetch_image(
        url, session, sema, timeout, max_bytes, max_retries, min_bytes
    ):
        return _flat_gray_jpeg(), "image/jpeg"

    monkeypatch.setattr(
        "tifa_archivist.orchestrator.search_ddg_images", fake_search_ddg_images
    )
    monkeypatch.setattr(
        "tifa_archivist.orchestrator.fetch_image", fake_fetch_image
    )

    out_dir = tmp_path / "out"
    config = AppConfig(
        out_dir=out_dir,
        db_path=out_dir / "images.db",
        log_dir=tmp_path / "logs",
        manifest=False,
        limit=3,
        skip_classify=True,
        min_bytes=1,
        max_bytes=10_000_000,
        max_download_retries=1,
        max_search_retries=1,
        search_queries=["Tifa Lockhart FF7"],
        adaptive_search_enabled=False,
        source_intelligence_path=out_dir / "source_intelligence.json",
        xai=XAIConfig(api_key="test"),
    )

    asyncio.run(run_pipeline(config))

    saved = list((out_dir / "other").glob("*"))
    assert saved == [], f"expected zero kept images, found {saved}"
