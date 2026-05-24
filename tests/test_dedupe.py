from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image

from tifa_archivist.dedupe import PhashIndex, compute_phash, compute_sha256, phash_distance


def make_image_bytes(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (64, 64), color)
    buf = BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _textured_jpeg(quality: int) -> bytes:
    """Deterministic textured image encoded as JPEG at a given quality."""
    rng = np.random.default_rng(7)
    arr = rng.integers(40, 220, size=(128, 128, 3), dtype=np.int32)
    yy, xx = np.mgrid[0:128, 0:128]
    arr[:, :, 0] = (arr[:, :, 0] + (xx % 32)) % 256
    arr[:, :, 1] = (arr[:, :, 1] + (yy % 32)) % 256
    arr[:, :, 2] = (arr[:, :, 2] + ((xx ^ yy) % 32)) % 256
    buf = BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="JPEG", quality=quality)
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
    # imagehash.phash uses median comparison: any solid non-black colour has a
    # non-zero DC term > median(0) → hash "8000000000000000".  Pure black has an
    # all-zero DCT → median = 0, so 0 > 0 is False → hash "0000000000000000".
    # These two are guaranteed to differ by Hamming distance 1.
    data1 = make_image_bytes((255, 255, 255))  # non-zero DC  → 8000000000000000
    data2 = make_image_bytes((0, 0, 0))        # all-zero DCT → 0000000000000000
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


def test_phash_index_near_duplicate_jpeg_recompression() -> None:
    """Same image recompressed at a different JPEG quality should hash-match
    as a near-duplicate through PhashIndex (distance <= phash_distance=6)."""
    original = _textured_jpeg(quality=92)
    recompressed = _textured_jpeg(quality=70)
    assert original != recompressed  # sanity: the bytes really differ
    h1 = compute_phash(original)
    h2 = compute_phash(recompressed)
    assert h1 is not None
    assert h2 is not None
    # Perceptual hash should be identical or very close for minor recompression.
    assert phash_distance(h1, h2) <= 6
    index = PhashIndex([h1], threshold=6)
    assert index.is_similar(h2)


def test_phash_index_rejects_unrelated_image() -> None:
    """A completely different image should NOT be flagged as a near-duplicate."""
    h1 = compute_phash(make_image_bytes((0, 0, 0)))
    h2 = compute_phash(make_image_bytes((255, 255, 255)))
    assert h1 is not None
    assert h2 is not None
    index = PhashIndex([h1], threshold=6)
    # Black vs white differ by exactly 1 bit (DC-only), which is <= 6 — not a
    # useful "unrelated" case.  Use a textured image instead.
    h_tex = compute_phash(_textured_jpeg(quality=92))
    assert h_tex is not None
    index2 = PhashIndex([h_tex], threshold=6)
    assert not index2.is_similar(h1)  # flat black should be far from textured
