from __future__ import annotations

import re
from io import BytesIO
from datetime import datetime, timezone

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
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False


def luma_stddev(data: bytes, max_side: int = 128) -> float | None:
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(BytesIO(data)) as image:
            image.load()
            image = image.convert("L")
            if max_side and max(image.size) > max_side:
                image.thumbnail((max_side, max_side), Image.BILINEAR)
            stat = ImageStat.Stat(image)
            if not stat.stddev:
                return None
            return float(stat.stddev[0])
    except Exception:
        return None
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
