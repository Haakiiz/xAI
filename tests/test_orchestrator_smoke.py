import asyncio
from io import BytesIO

from PIL import Image

from tifa_archivist.config import AppConfig, XAIConfig
from tifa_archivist.orchestrator import run_pipeline


def test_run_pipeline_smoke(tmp_path, monkeypatch) -> None:
    image_buf = BytesIO()
    Image.new("RGB", (640, 480), color=(20, 30, 40)).save(image_buf, format="JPEG")
    image_bytes = image_buf.getvalue()

    async def fake_search_live_images(**_kwargs):
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
        return image_bytes, "image/jpeg"

    monkeypatch.setattr(
        "tifa_archivist.orchestrator.search_live_images",
        fake_search_live_images,
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
        max_download_retries=0,
        max_search_retries=0,
        min_side=256,
        xai=XAIConfig(api_key="test"),
    )

    asyncio.run(run_pipeline(config))

    saved = list((out_dir / "other").glob("*"))
    assert len(saved) == 2
