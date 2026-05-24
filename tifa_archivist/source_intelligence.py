from __future__ import annotations

import json
import math
import random
from pathlib import Path
from urllib.parse import urlparse

from .utils import utc_now_iso


ADAPTIVE_SUFFIXES = [
    "4k wallpaper",
    "portrait",
    "official art",
    "high resolution",
    "render",
    "fanart",
    "cosplay",
]


def normalize_host(host_or_url: str) -> str:
    parsed = urlparse(host_or_url)
    host = parsed.netloc or host_or_url
    host = host.lower().strip()
    if host.startswith("www."):
        host = host[4:]
    return host


class SourceIntelligence:
    def __init__(
        self,
        state_path: Path,
        enabled: bool = True,
        exploration_weight: float = 0.35,
        max_variants: int = 12,
        max_site_variants: int = 4,
        min_host_samples: int = 6,
        bad_host_threshold: float = 0.05,
        bad_host_reject_probability: float = 0.9,
        rng_seed: int | None = None,
    ) -> None:
        self.state_path = Path(state_path)
        self.enabled = enabled
        self.exploration_weight = exploration_weight
        self.max_variants = max_variants
        self.max_site_variants = max_site_variants
        self.min_host_samples = min_host_samples
        self.bad_host_threshold = bad_host_threshold
        self.bad_host_reject_probability = bad_host_reject_probability
        self.rng = random.Random(rng_seed)

        self.query_stats: dict[str, dict[str, object]] = {}
        self.host_stats: dict[str, dict[str, object]] = {}
        self.query_host_stats: dict[str, dict[str, object]] = {}

    def load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        self.query_stats = self._sanitize_stats_map(payload.get("query_stats"))
        self.host_stats = self._sanitize_stats_map(payload.get("host_stats"))
        self.query_host_stats = self._sanitize_stats_map(payload.get("query_host_stats"))

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": utc_now_iso(),
            "query_stats": self.query_stats,
            "host_stats": self.host_stats,
            "query_host_stats": self.query_host_stats,
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def plan_queries(self, base_queries: list[str]) -> list[str]:
        ordered_base = self._unique(base_queries)
        if not self.enabled:
            return ordered_base

        variants = self._build_variants(ordered_base)
        combined = self._unique(ordered_base + variants)
        order_index = {query: idx for idx, query in enumerate(combined)}
        base_count = max(1, len(ordered_base))
        ranked = sorted(
            combined,
            key=lambda query: (
                self._query_ucb_score(query),
                -order_index[query],
            ),
            reverse=True,
        )
        cap = max(base_count, len(ordered_base) + self.max_variants)
        return ranked[:cap]

    def should_accept_host(self, query: str, host: str) -> bool:
        if not self.enabled:
            return True
        host = normalize_host(host)
        if not host:
            return True
        host_stats = self.host_stats.get(host) or {}
        pair_stats = self.query_host_stats.get(self._pair_key(query, host)) or {}
        host_is_bad = self._is_bad_source(host_stats)
        pair_is_bad = self._is_bad_source(pair_stats)
        if not host_is_bad and not pair_is_bad:
            return True
        keep_prob = 1.0 - self.bad_host_reject_probability
        # Keep a small random trickle for exploration so bad sources can recover.
        keep_prob = max(keep_prob, min(0.25, self.exploration_weight / 2.0))
        return self.rng.random() < keep_prob

    def record_search(self, query: str, urls_returned: int) -> None:
        stats = self._get_bucket(self.query_stats, query)
        self._inc(stats, "search_runs")
        self._inc(stats, "urls_returned", max(0, int(urls_returned)))

    def record_enqueued(self, query: str, host: str) -> None:
        host = normalize_host(host)
        if not host:
            return
        self._inc(self._get_bucket(self.query_stats, query), "enqueued")
        self._inc(self._get_bucket(self.host_stats, host), "enqueued")
        self._inc(
            self._get_bucket(self.query_host_stats, self._pair_key(query, host)),
            "enqueued",
        )

    def record_kept(self, query: str | None, host: str | None, label: str) -> None:
        self._record_outcome(query, host, "kept", f"kept:{label}")

    def record_discard(self, query: str | None, host: str | None, reason: str) -> None:
        self._record_outcome(query, host, "discarded", f"discard:{reason}")

    def record_download_failed(
        self, query: str | None, host: str | None, reason: str = "download_failed"
    ) -> None:
        self._record_outcome(query, host, "download_failed", f"download:{reason}")

    def record_classify_failed(
        self, query: str | None, host: str | None, reason: str = "classify_error"
    ) -> None:
        self._record_outcome(query, host, "classify_failed", f"classify:{reason}")

    def top_summary(self, limit: int = 5) -> list[tuple[str, float, int]]:
        items: list[tuple[str, float, int]] = []
        for host, stats in self.host_stats.items():
            attempts = self._attempts(stats)
            if attempts == 0:
                continue
            items.append((host, self._mean_success(stats), attempts))
        items.sort(key=lambda row: (row[1], row[2]), reverse=True)
        return items[:limit]

    @staticmethod
    def _sanitize_stats_map(raw: object) -> dict[str, dict[str, object]]:
        if not isinstance(raw, dict):
            return {}
        clean: dict[str, dict[str, object]] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            bucket: dict[str, object] = {}
            for field in (
                "search_runs",
                "urls_returned",
                "enqueued",
                "kept",
                "discarded",
                "download_failed",
                "classify_failed",
            ):
                bucket[field] = int(value.get(field, 0) or 0)
            reason_counts = value.get("reason_counts")
            if isinstance(reason_counts, dict):
                bucket["reason_counts"] = {
                    str(k): int(v)
                    for k, v in reason_counts.items()
                    if isinstance(k, str)
                }
            clean[key] = bucket
        return clean

    @staticmethod
    def _unique(items: list[str]) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _build_variants(self, base_queries: list[str]) -> list[str]:
        if self.max_variants <= 0:
            return []
        variants: list[str] = []
        base_set = set(base_queries)
        anchors = self._top_anchor_queries(base_queries, limit=3)
        good_hosts = self._top_hosts_for_site_queries(limit=self.max_site_variants)

        for query in anchors:
            if not query.isascii():
                continue
            for suffix in ADAPTIVE_SUFFIXES:
                candidate = f"{query} {suffix}"
                if candidate in base_set:
                    continue
                variants.append(candidate)
                if len(variants) >= self.max_variants:
                    return variants

        for query in anchors:
            for host in good_hosts:
                candidate = f"{query} site:{host}"
                if candidate in base_set:
                    continue
                variants.append(candidate)
                if len(variants) >= self.max_variants:
                    return variants
        return variants

    def _top_anchor_queries(self, base_queries: list[str], limit: int) -> list[str]:
        known = []
        for query, stats in self.query_stats.items():
            attempts = self._attempts(stats)
            if attempts <= 0:
                continue
            known.append((query, self._query_ucb_score(query)))
        known.sort(key=lambda item: item[1], reverse=True)
        anchors = [query for query, _ in known[:limit]]
        if not anchors:
            anchors = base_queries[:limit]
        return anchors

    def _top_hosts_for_site_queries(self, limit: int) -> list[str]:
        hosts: list[tuple[str, float, int]] = []
        for host, stats in self.host_stats.items():
            attempts = self._attempts(stats)
            if attempts < self.min_host_samples:
                continue
            success = self._mean_success(stats)
            if success <= self.bad_host_threshold:
                continue
            hosts.append((host, success, attempts))
        hosts.sort(key=lambda row: (row[1], row[2]), reverse=True)
        return [host for host, _, _ in hosts[:limit]]

    def _query_ucb_score(self, query: str) -> float:
        stats = self.query_stats.get(query) or {}
        mean = self._mean_success(stats)
        attempts = self._attempts(stats)
        total = 0
        for bucket in self.query_stats.values():
            total += self._attempts(bucket)
        total = max(1, total)
        explore = math.sqrt((2.0 * math.log(total + 1.0)) / (attempts + 1.0))
        return mean + (self.exploration_weight * explore)

    def _record_outcome(
        self, query: str | None, host: str | None, field: str, reason: str
    ) -> None:
        if not query or not host:
            return
        host = normalize_host(host)
        if not host:
            return
        query_bucket = self._get_bucket(self.query_stats, query)
        host_bucket = self._get_bucket(self.host_stats, host)
        pair_bucket = self._get_bucket(self.query_host_stats, self._pair_key(query, host))

        self._inc(query_bucket, field)
        self._inc(host_bucket, field)
        self._inc(pair_bucket, field)
        self._inc_reason(query_bucket, reason)
        self._inc_reason(host_bucket, reason)
        self._inc_reason(pair_bucket, reason)

    def _is_bad_source(self, stats: dict[str, object]) -> bool:
        attempts = self._attempts(stats)
        if attempts < self.min_host_samples:
            return False
        return self._mean_success(stats) <= self.bad_host_threshold

    @staticmethod
    def _attempts(stats: dict[str, object]) -> int:
        kept = int(stats.get("kept", 0) or 0)
        discarded = int(stats.get("discarded", 0) or 0)
        download_failed = int(stats.get("download_failed", 0) or 0)
        classify_failed = int(stats.get("classify_failed", 0) or 0)
        return kept + discarded + download_failed + classify_failed

    @staticmethod
    def _mean_success(stats: dict[str, object]) -> float:
        kept = int(stats.get("kept", 0) or 0)
        attempts = SourceIntelligence._attempts(stats)
        # Beta(1,1) posterior mean for stable ranking with sparse history.
        return (kept + 1.0) / (attempts + 2.0)

    @staticmethod
    def _pair_key(query: str, host: str) -> str:
        return f"{query}|||{host}"

    @staticmethod
    def _get_bucket(
        mapping: dict[str, dict[str, object]], key: str
    ) -> dict[str, object]:
        bucket = mapping.get(key)
        if bucket is None:
            bucket = {}
            mapping[key] = bucket
        return bucket

    @staticmethod
    def _inc(bucket: dict[str, object], key: str, amount: int = 1) -> None:
        bucket[key] = int(bucket.get(key, 0) or 0) + amount

    @staticmethod
    def _inc_reason(bucket: dict[str, object], reason: str) -> None:
        reason_counts = bucket.get("reason_counts")
        if not isinstance(reason_counts, dict):
            reason_counts = {}
            bucket["reason_counts"] = reason_counts
        reason_counts[reason] = int(reason_counts.get(reason, 0) or 0) + 1
