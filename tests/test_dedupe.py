from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image

from tifa_archivist.dedupe import PhashIndex, compute_phash, compute_sha256, phash_distance


def make_image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (64, 64), color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_sha256_matches_hashlib() -> None:
    data = b"abc"
    assert compute_sha256(data) == hashlib.sha256(data).hexdigest()


def test_phash_distance_identical() -> None:
    data = make_image_bytes((255, 0, 0))
    h1 = compute_phash(data)
    h2 = compute_phash(data)
    assert h1 is not None
    assert h2 is not None
    assert phash_distance(h1, h2) == 0


def test_phash_distance_different() -> None:
    data1 = make_image_bytes((255, 0, 0))
    data2 = make_image_bytes((0, 0, 255))
    h1 = compute_phash(data1)
    h2 = compute_phash(data2)
    assert h1 is not None
    assert h2 is not None
    assert phash_distance(h1, h2) > 0


def test_phash_index_similarity() -> None:
    data = make_image_bytes((10, 20, 30))
    h1 = compute_phash(data)
    assert h1 is not None
    index = PhashIndex([h1], threshold=6)
    assert index.is_similar(h1)
