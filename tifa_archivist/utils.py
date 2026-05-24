from __future__ import annotations

import re
from datetime import datetime, timezone
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def slugify(text: str, max_len: int = 32) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        text = "image"
    return text[:max_len]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sniff_image_mime(data: bytes) -> str | None:
    """Return MIME type from magic bytes, or None if unrecognised."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def image_quality_metrics(
    data: bytes,
    variance_max_side: int,
    flat_grid_size: int,
    flat_tile_stddev_max: float,
    edge_threshold: float,
) -> tuple[
    float | None,  # luma_stddev
    float | None,  # sat_stddev
    float | None,  # sat_mean
    float | None,  # colorfulness
    float | None,  # flat_ratio
    float | None,  # laplacian_var
    float | None,  # edge_density
    float | None,  # entropy
]:
    """Compute a suite of CV quality metrics from raw image bytes.

    Returns a tuple of 8 floats (or Nones on failure):
      luma_stddev, sat_stddev, sat_mean, colorfulness,
      flat_ratio, laplacian_var, edge_density, entropy
    """
    _NONE8: tuple = (None, None, None, None, None, None, None, None)
    try:
        import numpy as np
        from PIL import Image, ImageFile

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        try:
            with Image.open(BytesIO(data)) as img:
                img.load()
                rgb_img = img.convert("RGB")
                if variance_max_side > 0 and max(rgb_img.size) > variance_max_side:
                    rgb_img.thumbnail(
                        (variance_max_side, variance_max_side), Image.BILINEAR
                    )

                rgb = np.asarray(rgb_img, dtype=np.float32)
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = False

        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # --- Luma stddev -------------------------------------------------------
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        luma_stddev = float(np.std(luma))

        # --- HSV saturation (0-255 scale) --------------------------------------
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(maxc > 0, (maxc - minc) / maxc, 0.0) * 255.0
        sat_stddev = float(np.std(s))
        sat_mean = float(np.mean(s))

        # --- Colorfulness (Hasler & Süsstrunk 2003) ----------------------------
        rg = r - g
        yb = 0.5 * (r + g) - b
        colorfulness = float(
            (float(np.std(rg)) ** 2 + float(np.std(yb)) ** 2) ** 0.5
            + 0.3 * (float(np.mean(rg)) ** 2 + float(np.mean(yb)) ** 2) ** 0.5
        )

        # --- Flat-tile ratio ---------------------------------------------------
        h, w = luma.shape
        tile_h = max(1, h // max(1, flat_grid_size))
        tile_w = max(1, w // max(1, flat_grid_size))
        flat_count = 0
        total_count = 0
        for i in range(0, h - tile_h + 1, tile_h):
            for j in range(0, w - tile_w + 1, tile_w):
                tile = luma[i : i + tile_h, j : j + tile_w]
                total_count += 1
                if float(np.std(tile)) < flat_tile_stddev_max:
                    flat_count += 1
        flat_ratio = float(flat_count / total_count) if total_count > 0 else 0.0

        # --- Laplacian variance (sharpness) ------------------------------------
        # Discrete Laplacian via numpy rolls (edge wrap acceptable for quality
        # detection; avoids scipy dependency).
        lap = (
            np.roll(luma, -1, axis=0)
            + np.roll(luma, 1, axis=0)
            + np.roll(luma, -1, axis=1)
            + np.roll(luma, 1, axis=1)
            - 4.0 * luma
        )
        laplacian_var = float(np.var(lap))

        # --- Edge density ------------------------------------------------------
        edge_density = float(np.mean(np.abs(lap) > edge_threshold))

        # --- Entropy (8-bit grayscale histogram) -------------------------------
        gray8 = np.clip(luma, 0, 255).astype(np.uint8)
        hist, _ = np.histogram(gray8.flatten(), bins=256, range=(0, 256))
        total_px = hist.sum()
        if total_px > 0:
            probs = hist[hist > 0] / float(total_px)
            entropy = float(-np.sum(probs * np.log2(probs)))
        else:
            entropy = 0.0

        return (
            luma_stddev,
            sat_stddev,
            sat_mean,
            colorfulness,
            flat_ratio,
            laplacian_var,
            edge_density,
            entropy,
        )

    except Exception:
        return _NONE8
