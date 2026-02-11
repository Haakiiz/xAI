from __future__ import annotations

import re
from io import BytesIO
from datetime import datetime, timezone

import math
import numpy as np

from PIL import Image, ImageFile, ImageStat


def slugify(text: str, max_len: int = 32) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if not text:
        text = "image"
    return text[:max_len]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sniff_image_mime(data: bytes) -> str | None:
    if len(data) < 12:
        return None
    if data.startswith(b"\xFF\xD8\xFF"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data[4:12] in {b"ftypavif", b"ftypavis"}:
        return "image/avif"
    if data[4:12] in {
        b"ftypheic",
        b"ftypheix",
        b"ftyphevc",
        b"ftypheim",
        b"ftypmif1",
        b"ftypmsf1",
    }:
        return "image/heif"
    if data.startswith(b"\x00\x00\x00\x0cJXL ") or data.startswith(b"\xFF\x0A"):
        return "image/jxl"
    if data.startswith(b"BM"):
        return "image/bmp"
    if data.startswith(b"II*\x00") or data.startswith(b"MM\x00*"):
        return "image/tiff"
    return None


def validate_image_bytes(
    data: bytes, min_side: int
) -> tuple[bool, str, tuple[int, int] | None, str | None, bool]:
    mime = sniff_image_mime(data)
    if mime is None:
        return False, "unknown_format", None, None, False
    width = None
    height = None
    decoded = False
    reason = "ok"
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            width, height = image.size
        decoded = True
    except Exception:
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
            decoded = True
            reason = "truncated_ok"
        except Exception:
            return False, "decode_failed", None, mime, False
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
    if min_side > 0 and min(width, height) < min_side:
        return False, "too_small", (width, height), mime, decoded
    return True, reason, (width, height), mime, decoded


def prepare_classification_bytes(data: bytes, max_side: int | None = None) -> bytes | None:
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            image.load()
            image = image.convert("RGB")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=90, optimize=True)
            return output.getvalue()
    except Exception:
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(BytesIO(data)) as image:
                image.load()
                image = image.convert("RGB")
                if max_side and max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.LANCZOS)
                output = BytesIO()
                image.save(output, format="JPEG", quality=90, optimize=True)
                return output.getvalue()
        except Exception:
            return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def image_quality_metrics(
    data: bytes,
    max_side: int = 256,
    grid_size: int = 8,
    tile_stddev_max: float = 4.0,
    edge_threshold: float = 20.0,
) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        with Image.open(BytesIO(data)) as image:
            image.load()
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.BILINEAR)
            rgb = image.convert("RGB")
            luma = rgb.convert("L")
            luma_stat = ImageStat.Stat(luma)
            luma_stddev = float(luma_stat.stddev[0]) if luma_stat.stddev else None
            hsv = rgb.convert("HSV")
            sat = hsv.split()[1]
            sat_stat = ImageStat.Stat(sat)
            sat_stddev = float(sat_stat.stddev[0]) if sat_stat.stddev else None
            sat_mean = float(sat_stat.mean[0]) if sat_stat.mean else None
            flat_ratio = None
            laplacian_var = None
            edge_density = None
            entropy = None
            if grid_size > 0 and tile_stddev_max > 0:
                width, height = luma.size
                tiles_x = min(grid_size, max(1, width))
                tiles_y = min(grid_size, max(1, height))
                tile_w = max(1, width // tiles_x)
                tile_h = max(1, height // tiles_y)
                flat = 0
                total = 0
                for y in range(0, height, tile_h):
                    for x in range(0, width, tile_w):
                        box = (x, y, min(x + tile_w, width), min(y + tile_h, height))
                        tile = luma.crop(box)
                        stat = ImageStat.Stat(tile)
                        stddev = float(stat.stddev[0]) if stat.stddev else 0.0
                        if stddev < tile_stddev_max:
                            flat += 1
                        total += 1
                if total > 0:
                    flat_ratio = flat / total
            luma_array = np.asarray(luma, dtype=np.float32)
            if luma_array.size:
                lap = (
                    -4 * luma_array
                    + np.roll(luma_array, 1, axis=0)
                    + np.roll(luma_array, -1, axis=0)
                    + np.roll(luma_array, 1, axis=1)
                    + np.roll(luma_array, -1, axis=1)
                )
                laplacian_var = float(lap.var())
                gy, gx = np.gradient(luma_array)
                grad = np.hypot(gx, gy)
                if edge_threshold > 0:
                    edge_density = float((grad > edge_threshold).mean())
                else:
                    edge_density = float(grad.mean() / 255.0)
                bins = np.bincount(luma_array.astype(np.uint8).ravel(), minlength=256)
                total_bins = bins.sum()
                if total_bins > 0:
                    prob = bins / total_bins
                    prob = prob[prob > 0]
                    entropy = float(-(prob * np.log2(prob)).sum())
            pixels = list(rgb.getdata())
            if not pixels:
                return (
                    luma_stddev,
                    sat_stddev,
                    sat_mean,
                    None,
                    flat_ratio,
                    laplacian_var,
                    edge_density,
                    entropy,
                )
            sum_rg = 0.0
            sum_rg2 = 0.0
            sum_yb = 0.0
            sum_yb2 = 0.0
            count = 0
            for r, g, b in pixels:
                rg = float(r) - float(g)
                yb = 0.5 * (float(r) + float(g)) - float(b)
                sum_rg += rg
                sum_rg2 += rg * rg
                sum_yb += yb
                sum_yb2 += yb * yb
                count += 1
            mean_rg = sum_rg / count
            mean_yb = sum_yb / count
            var_rg = max(0.0, (sum_rg2 / count) - (mean_rg * mean_rg))
            var_yb = max(0.0, (sum_yb2 / count) - (mean_yb * mean_yb))
            std_rg = math.sqrt(var_rg)
            std_yb = math.sqrt(var_yb)
            colorfulness = math.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * math.sqrt(
                mean_rg * mean_rg + mean_yb * mean_yb
            )
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
        return None, None, None, None, None, None, None, None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
