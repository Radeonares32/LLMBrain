"""Bounded RAM caching layer for LLMBrain project memory."""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import OrderedDict
from enum import StrEnum
from typing import Any, NamedTuple


class CacheEvictionReason(StrEnum):
    COUNT = "count"
    BYTE_SIZE = "byte_size"
    TTL = "ttl"
    EXPLICIT = "explicit"


class CacheStats(NamedTuple):
    hits: int
    misses: int
    evictions: int
    current_items: int
    current_bytes: int


class CacheEntry:
    def __init__(
        self,
        project_id: str,
        key: str,
        value: Any,
        size: int,
        ttl: float,
        source_version: str | None = None,
    ) -> None:
        self.project_id = project_id
        self.key = key
        self.value = value
        self.size = size
        self.created_at = time.time()
        self.last_access_at = self.created_at
        self.expiry_time = self.created_at + ttl
        self.source_version = source_version


def estimate_size(obj: Any) -> int:
    """Recursively estimate size of object in bytes."""
    try:
        if isinstance(obj, (str, bytes)):
            return len(obj)
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return sys.getsizeof(obj)
        elif isinstance(obj, dict):
            return sum(estimate_size(k) + estimate_size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple, set)):
            return sum(estimate_size(x) for x in obj)
        elif hasattr(obj, "model_dump"):
            return len(json.dumps(obj.model_dump()))
        elif hasattr(obj, "__dict__"):
            return sum(estimate_size(k) + estimate_size(v) for k, v in obj.__dict__.items())
        else:
            return sys.getsizeof(obj)
    except Exception:
        return 100  # Safe fallback size in bytes


class BrainCache:
    """Bounded, thread-safe LRU cache with item count, byte size and TTL expiration limits."""

    def __init__(
        self,
        max_items: int = 500,
        max_bytes: int = 134217728,  # Default: 128 MB
        ttl_seconds: float = 1800,  # Default: 30 minutes
    ) -> None:
        self.max_items = max_items
        self.max_bytes = max_bytes
        self.ttl = ttl_seconds
        self._cache: OrderedDict[tuple[str, str], CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.total_bytes = 0

    def get(
        self, project_id: str, key: str, current_source_version: str | None = None
    ) -> Any | None:
        """Retrieve value from cache. Invalidates entry if expired or version mismatch."""
        now = time.time()
        with self._lock:
            cache_key = (project_id, key)
            if cache_key in self._cache:
                entry = self._cache[cache_key]
                # Check TTL
                if now > entry.expiry_time:
                    self._evict_entry(cache_key, CacheEvictionReason.TTL)
                    self.misses += 1
                    return None
                # Check source version invalidation (e.g. file hash change)
                if (
                    current_source_version is not None
                    and entry.source_version != current_source_version
                ):
                    self._evict_entry(cache_key, CacheEvictionReason.EXPLICIT)
                    self.misses += 1
                    return None

                # Update LRU ordering
                entry.last_access_at = now
                self._cache.move_to_end(cache_key)
                self.hits += 1
                return entry.value

            self.misses += 1
            return None

    def set(self, project_id: str, key: str, value: Any, source_version: str | None = None) -> None:
        """Store value in cache. Evicts entries to respect capacity constraints."""
        size = estimate_size(value)
        if size > self.max_bytes:
            # Item is too large to fit in cache
            return

        cache_key = (project_id, key)
        with self._lock:
            if cache_key in self._cache:
                self._evict_entry(cache_key, CacheEvictionReason.EXPLICIT)

            self._ensure_space(size)

            entry = CacheEntry(project_id, key, value, size, self.ttl, source_version)
            self._cache[cache_key] = entry
            self.total_bytes += size

    def invalidate(self, project_id: str, key: str) -> None:
        """Explicitly invalidate a cache key."""
        with self._lock:
            cache_key = (project_id, key)
            if cache_key in self._cache:
                self._evict_entry(cache_key, CacheEvictionReason.EXPLICIT)

    def invalidate_project(self, project_id: str) -> None:
        """Invalidate all keys under a specific project namespace."""
        with self._lock:
            keys_to_remove = [k for k in self._cache.keys() if k[0] == project_id]
            for k in keys_to_remove:
                self._evict_entry(k, CacheEvictionReason.EXPLICIT)

    def clear(self) -> None:
        """Clear all cache contents."""
        with self._lock:
            self._cache.clear()
            self.total_bytes = 0

    def stats(self) -> CacheStats:
        """Return cache hits, misses, and current storage metrics."""
        with self._lock:
            # Prune expired entries to keep statistics accurate
            now = time.time()
            expired_keys = [k for k, entry in self._cache.items() if now > entry.expiry_time]
            for k in expired_keys:
                self._evict_entry(k, CacheEvictionReason.TTL)

            return CacheStats(
                hits=self.hits,
                misses=self.misses,
                evictions=self.evictions,
                current_items=len(self._cache),
                current_bytes=self.total_bytes,
            )

    def _evict_entry(self, cache_key: tuple[str, str], reason: CacheEvictionReason) -> None:
        entry = self._cache.pop(cache_key)
        self.total_bytes -= entry.size
        self.evictions += 1

    def _ensure_space(self, new_item_size: int) -> None:
        # Prune expired items
        now = time.time()
        expired_keys = [k for k, entry in self._cache.items() if now > entry.expiry_time]
        for k in expired_keys:
            self._evict_entry(k, CacheEvictionReason.TTL)

        # Evict LRU items until space is available
        while len(self._cache) >= self.max_items or (
            self.total_bytes + new_item_size > self.max_bytes
        ):
            if not self._cache:
                break
            first_key = next(iter(self._cache))
            reason = (
                CacheEvictionReason.COUNT
                if len(self._cache) >= self.max_items
                else CacheEvictionReason.BYTE_SIZE
            )
            self._evict_entry(first_key, reason)
