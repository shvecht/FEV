from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence, Tuple

import numpy as np
import threading
import time

FetchFunc = Callable[[int, float, float], Tuple[np.ndarray, np.ndarray]]


def _tile_key(channel: int, start: float, duration: float) -> tuple[int, float, float]:
    return (channel, round(start, 6), round(duration, 6))


@dataclass
class CacheEntry:
    data: Tuple[np.ndarray, np.ndarray]
    last_access: float
    is_final: bool


@dataclass(frozen=True)
class _PrefetchTask:
    key: tuple[int, float, float]
    stage: str  # "preview" or "final"


@dataclass
class PrefetchConfig:
    tile_duration: float = 5.0
    max_tiles: int = 32
    max_bytes: float | None = None  # approximate budget in bytes


class PrefetchCache:
    def __init__(
        self,
        fetch: FetchFunc,
        config: PrefetchConfig | None = None,
        preview_fetch: FetchFunc | None = None,
    ):
        self.fetch = fetch
        self.preview_fetch = preview_fetch
        self.config = config or PrefetchConfig()
        self._cache: OrderedDict[tuple[int, float, float], CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pending: list[_PrefetchTask] = []
        self._estimated_tile_bytes: float = 0.0
        self._hits: int = 0
        self._misses: int = 0

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
                if not entry.is_final:
                    self._ensure_final_enqueued(key)
                self._hits += 1
                return entry.data
            self._misses += 1
        data = self.fetch(channel, start, start + duration)
        with self._lock:
            self._update_estimated_tile_bytes(data)
            self._cache[key] = CacheEntry(
                data=data,
                last_access=time.time(),
                is_final=True,
            )
            self._cache.move_to_end(key)
            self._evict_if_needed()
        return data

    def prefetch_window(self, channel: int, start: float, duration: float):
        tiles = self._tiles_for_window(channel, start, duration)
        with self._lock:
            for tile in tiles:
                entry = self._cache.get(tile)
                needs_final = entry is None or not entry.is_final
                if self.preview_fetch is not None and needs_final:
                    self._enqueue_task(_PrefetchTask(tile, "preview"))
                if needs_final:
                    self._enqueue_task(_PrefetchTask(tile, "final"))

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
            task: _PrefetchTask | None = None
            with self._lock:
                while self._pending:
                    candidate = self._pending.pop(0)
                    if candidate.stage == "preview" and self.preview_fetch is None:
                        continue
                    entry = self._cache.get(candidate.key)
                    if entry and entry.is_final:
                        continue
                    task = candidate
                    break
            if task is None:
                self._stop.wait(0.1)
                continue
            channel, start, duration = task.key
            key = _tile_key(channel, start, duration)
            fetch_fn = self.preview_fetch if task.stage == "preview" else self.fetch
            data = fetch_fn(channel, start, start + duration)
            with self._lock:
                entry = self._cache.get(key)
                is_final = task.stage == "final"
                if entry is not None and entry.is_final and not is_final:
                    continue
                self._cache[key] = CacheEntry(
                    data=data,
                    last_access=time.time(),
                    is_final=is_final,
                )
                self._cache.move_to_end(key)
                if is_final:
                    self._update_estimated_tile_bytes(data)
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

    @property
    def estimated_tile_bytes(self) -> float:
        with self._lock:
            return self._estimated_tile_bytes

    @property
    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self._hits, self._misses

    def _enqueue_task(self, task: _PrefetchTask) -> None:
        if task in self._pending:
            return
        self._pending.append(task)

    def _ensure_final_enqueued(self, key: tuple[int, float, float]) -> None:
        final_task = _PrefetchTask(key, "final")
        if final_task not in self._pending:
            self._pending.append(final_task)

    def _update_estimated_tile_bytes(self, data: Tuple[np.ndarray, np.ndarray]) -> None:
        tile_bytes = sum(arr.nbytes for arr in data)
        if tile_bytes <= 0:
            return
        if self._estimated_tile_bytes == 0.0:
            self._estimated_tile_bytes = float(tile_bytes)
        else:
            self._estimated_tile_bytes = 0.8 * self._estimated_tile_bytes + 0.2 * float(tile_bytes)


class PrefetchService:
    """Shared configuration for building prefetch caches."""

    def __init__(self, config: PrefetchConfig | None = None):
        self._config = config or PrefetchConfig()
        self._lock = threading.RLock()
        self._radius_multiple = 1.0
        self._hint_channels: tuple[int, ...] = ()
        self._hint_window_s: float = 60.0

    @property
    def config(self) -> PrefetchConfig:
        return self._config

    def configure(
        self,
        *,
        tile_duration: float | None = None,
        max_tiles: int | None = None,
        max_mb: float | None = None,
        radius_multiple: float | None = None,
    ) -> None:
        cfg = self._config
        if tile_duration is not None:
            cfg = replace(cfg, tile_duration=float(tile_duration))
        if max_tiles is not None:
            cfg = replace(cfg, max_tiles=max(1, int(max_tiles)))
        if max_mb is not None:
            cfg = replace(cfg, max_bytes=float(max_mb) * 1024 * 1024)
        if radius_multiple is not None and radius_multiple > 0:
            with self._lock:
                self._radius_multiple = float(radius_multiple)
        with self._lock:
            self._config = cfg

    def update_hints(
        self,
        *,
        channels: Iterable[int] | None = None,
        window_s: float | None = None,
    ) -> None:
        with self._lock:
            if channels is not None:
                ordered = []
                seen: set[int] = set()
                for value in channels:
                    idx = int(value)
                    if idx < 0 or idx in seen:
                        continue
                    seen.add(idx)
                    ordered.append(idx)
                self._hint_channels = tuple(ordered)
            if window_s is not None and window_s > 0:
                self._hint_window_s = float(window_s)

    def prefetch_radius(
        self,
        window_duration: float,
        *,
        channel_indices: Sequence[int] | None = None,
        sample_rates: Sequence[float] | None = None,
        estimated_tile_bytes: float | None = None,
    ) -> float:
        if window_duration <= 0:
            return 0.0
        with self._lock:
            cfg = self._config
            radius_multiple = self._radius_multiple
            hint_channels = self._hint_channels
        channels = tuple(channel_indices) if channel_indices is not None else hint_channels
        channel_count = len(channels) if channels else (len(hint_channels) or 1)
        base_radius = window_duration * radius_multiple
        tile_duration = float(cfg.tile_duration)
        max_span = float("inf")
        if channel_count > 0 and cfg.max_tiles:
            max_tiles_total = float(cfg.max_tiles)
            per_channel_tiles = max_tiles_total / float(channel_count)
            if per_channel_tiles > 0:
                max_span = min(max_span, per_channel_tiles * tile_duration)
        rates: list[float] = []
        if sample_rates is not None:
            rates = [float(rate) for rate in sample_rates if float(rate) > 0]
        if cfg.max_bytes and cfg.max_bytes > 0:
            if estimated_tile_bytes and estimated_tile_bytes > 0:
                per_tile_bytes = float(estimated_tile_bytes)
                if channel_count > 0:
                    tiles_allowed = float(cfg.max_bytes) / per_tile_bytes
                    if tiles_allowed > 0:
                        span_from_bytes = (tiles_allowed / max(1.0, float(channel_count))) * tile_duration
                        max_span = min(max_span, span_from_bytes)
            else:
                if not rates and hint_channels:
                    rates = [1.0 for _ in hint_channels]
                if rates:
                    bytes_per_sample = float(np.dtype(np.float64).itemsize + np.dtype(np.float32).itemsize)
                    bytes_per_second = sum(rate * bytes_per_sample for rate in rates)
                    if bytes_per_second > 0:
                        span_from_bytes = float(cfg.max_bytes) / bytes_per_second
                        max_span = min(max_span, span_from_bytes)
        if not np.isfinite(max_span):
            max_span = window_duration + base_radius * 2.0
        max_radius = max(0.0, (max_span - window_duration) * 0.5)
        return max(0.0, min(base_radius, max_radius))

    def create_cache(
        self, fetch: FetchFunc, *, preview_fetch: FetchFunc | None = None
    ) -> PrefetchCache:
        with self._lock:
            cfg = self._config
        return PrefetchCache(fetch, cfg, preview_fetch=preview_fetch)


prefetch_service = PrefetchService()
