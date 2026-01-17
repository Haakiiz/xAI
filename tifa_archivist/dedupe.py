from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image
import imagehash


def compute_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_phash(data: bytes) -> str | None:
    try:
        with Image.open(BytesIO(data)) as image:
            image = image.convert("RGB")
            return str(imagehash.phash(image))
    except Exception:
        return None


def phash_distance(phash_a: str, phash_b: str) -> int:
    return imagehash.hex_to_hash(phash_a) - imagehash.hex_to_hash(phash_b)


class PhashIndex:
    def __init__(self, existing: list[str] | None, threshold: int) -> None:
        self.threshold = threshold
        self._hashes = list(existing or [])

    def is_similar(self, phash: str) -> bool:
        for existing in self._hashes:
            if phash_distance(phash, existing) <= self.threshold:
                return True
        return False

    def add(self, phash: str) -> None:
        self._hashes.append(phash)
