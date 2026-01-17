from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path


class ImageDB:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()

    def init(self) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    sha256 TEXT,
                    phash TEXT,
                    filepath TEXT,
                    category TEXT,
                    created_at TEXT,
                    model_name TEXT,
                    confidence REAL
                )
                """
            )
            self.conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_images_sha256 ON images(sha256)"
            )
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash)")

    def has_url(self, url: str) -> bool:
        with self.lock:
            cur = self.conn.execute("SELECT 1 FROM images WHERE url = ? LIMIT 1", (url,))
            return cur.fetchone() is not None

    def has_sha256(self, sha256: str) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "SELECT 1 FROM images WHERE sha256 = ? LIMIT 1", (sha256,)
            )
            return cur.fetchone() is not None

    def get_all_phashes(self) -> list[str]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT phash FROM images WHERE phash IS NOT NULL AND phash != ''"
            )
            return [row["phash"] for row in cur.fetchall()]

    def insert_record(
        self,
        url: str,
        sha256: str | None,
        phash: str | None,
        filepath: str | None,
        category: str,
        model_name: str | None,
        confidence: float | None,
    ) -> None:
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO images
                (url, sha256, phash, filepath, category, created_at, model_name, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url,
                    sha256,
                    phash,
                    filepath,
                    category,
                    created_at,
                    model_name,
                    confidence,
                ),
            )

    def count_by_category(self) -> dict[str, int]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT category, COUNT(*) as count FROM images GROUP BY category"
            )
            return {row["category"]: int(row["count"]) for row in cur.fetchall()}

    def all_records(self) -> list[dict[str, object]]:
        with self.lock:
            cur = self.conn.execute(
                """
                SELECT url, sha256, phash, filepath, category, created_at, model_name, confidence
                FROM images
                ORDER BY created_at ASC
                """
            )
            return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        with self.lock:
            self.conn.close()
