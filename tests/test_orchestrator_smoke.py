import asyncio
from io import BytesIO

from PIL import Image

from tifa_archivist.config import AppConfig, XAIConfig
from tifa_archivist.orchestrator import run_pipeline


def test_run_pipeline_smoke(tmp_path, monkeypatch) -> None:
    def fake_search_ddg_images(
        query,
        max_results,
        stop_flag,
        max_retries,
    ):
        return [
            "https://example.com/a.jpg",
            "https://example.com/b.png",
            "https://example.com/c.jpg",
        ]

    async def fake_fetch_image(
        url,
        session,
        sema,
        timeout,
        max_bytes,
        max_retries,
        min_bytes,
    ):
        seed = ord(url[-5]) % 50
        image_buf = BytesIO()
        Image.new("RGB", (640, 480), color=(20 + seed, 30 + seed, 40 + seed)).save(
            image_buf, format="JPEG"
        )
        return image_buf.getvalue(), "image/jpeg"

    monkeypatch.setattr(
        "tifa_archivist.orchestrator.search_ddg_images",
        fake_search_ddg_images,
    )
    monkeypatch.setattr(
        "tifa_archivist.orchestrator.fetch_image",
        fake_fetch_image,
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
        max_bytes=10_000,
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
