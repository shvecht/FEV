from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Dict, Tuple

import numpy as np

FetchFunc = Callable[[int, float, float], Tuple[np.ndarray, np.ndarray]]


def _tile_key(channel: int, start: float, duration: float) -> tuple[int, float, float]:
    return (channel, round(start, 6), round(duration, 6))


@dataclass
class CacheEntry:
    data: Tuple[np.ndarray, np.ndarray]
    last_access: float


@dataclass
class PrefetchConfig:
    tile_duration: float = 5.0
    max_tiles: int = 32
    max_bytes: float | None = None  # approximate budget in bytes


class PrefetchCache:
    def __init__(self, fetch: FetchFunc, config: PrefetchConfig | None = None):
        self.fetch = fetch
        self.config = config or PrefetchConfig()
        self._cache: OrderedDict[tuple[int, float, float], CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: list[tuple[int, float, float]] = []
        self._estimated_tile_bytes: float = 0.0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._pending.clear()

    def get_tile(self, channel: int, start: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        key = _tile_key(channel, start, duration)
        with self._lock:
            entry = self._cache.get(key)
            if entry:
                entry.last_access = time.time()
                self._cache.move_to_end(key)
                return entry.data
        data = self.fetch(channel, start, start + duration)
        with self._lock:
            self._cache[key] = CacheEntry(data=data, last_access=time.time())
            self._cache.move_to_end(key)
            self._evict_if_needed()
        return data

    def prefetch_window(self, channel: int, start: float, duration: float):
        tiles = self._tiles_for_window(channel, start, duration)
        with self._lock:
            for tile in tiles:
                if tile not in self._cache:
                    self._pending.append(tile)

    def _tiles_for_window(self, channel: int, start: float, duration: float):
        tiles = []
        t = start
        tile_duration = self.config.tile_duration
        while t < start + duration:
            tiles.append(_tile_key(channel, t, tile_duration))
            t += tile_duration
        return tiles

    def _evict_if_needed(self):
        while len(self._cache) > self._max_tiles_allowed():
            self._cache.popitem(last=False)

    def _worker(self):
        while not self._stop.is_set():
            tile = None
            with self._lock:
                if self._pending:
                    tile = self._pending.pop(0)
            if tile is None:
                self._stop.wait(0.1)
                continue
            channel, start, duration = tile
            key = _tile_key(channel, start, duration)
            with self._lock:
                if key in self._cache:
                    continue
            data = self.fetch(channel, start, start + duration)
            with self._lock:
                tile_bytes = sum(arr.nbytes for arr in data)
                if tile_bytes > 0:
                    if self._estimated_tile_bytes == 0.0:
                        self._estimated_tile_bytes = float(tile_bytes)
                    else:
                        self._estimated_tile_bytes = 0.8 * self._estimated_tile_bytes + 0.2 * float(tile_bytes)
                self._cache[key] = CacheEntry(data=data, last_access=time.time())
                self._cache.move_to_end(key)
                self._evict_if_needed()

    def _max_tiles_allowed(self) -> int:
        if self.config.max_bytes and self._estimated_tile_bytes > 0:
            tiles = int(self.config.max_bytes / self._estimated_tile_bytes)
            return max(1, tiles)
        return self.config.max_tiles

    def configure(self, config: PrefetchConfig):
        with self._lock:
            self.config = config
            self._evict_if_needed()


class PrefetchService:
    """Shared configuration for building prefetch caches."""

    def __init__(self, config: PrefetchConfig | None = None):
        self._config = config or PrefetchConfig()
        self._lock = threading.RLock()

    @property
    def config(self) -> PrefetchConfig:
        return self._config

    def configure(
        self,
        *,
        tile_duration: float | None = None,
        max_tiles: int | None = None,
        max_mb: float | None = None,
    ) -> None:
        cfg = self._config
        if tile_duration is not None:
            cfg = replace(cfg, tile_duration=float(tile_duration))
        if max_tiles is not None:
            cfg = replace(cfg, max_tiles=max(1, int(max_tiles)))
        if max_mb is not None:
            cfg = replace(cfg, max_bytes=float(max_mb) * 1024 * 1024)
        with self._lock:
            self._config = cfg

    def create_cache(self, fetch: FetchFunc) -> PrefetchCache:
        with self._lock:
            cfg = self._config
        return PrefetchCache(fetch, cfg)


prefetch_service = PrefetchService()
