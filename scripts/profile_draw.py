#!/usr/bin/env python3
"""Quick render benchmark for EDF/Zarr viewer pipeline."""
from __future__ import annotations

import argparse
import time

from core.edf_loader import EdfLoader
from core.zarr_loader import ZarrLoader
from core.prefetch import PrefetchCache, PrefetchConfig, PrefetchService
from core.overscan import SignalChunk


def load_loader(path: str):
    if path.endswith('.zarr'):
        return ZarrLoader(path)
    return EdfLoader(path)


def _fetch_tuple(loader, channel: int, start: float, end: float):
    chunk = loader.read(channel, start, end)
    if isinstance(chunk, SignalChunk):
        return chunk.as_tuple()
    return chunk


def _consume_window(cache: PrefetchCache, channel: int, start: float, duration: float):
    tile = cache.config.tile_duration
    t = start
    end = start + duration
    while t < end:
        cache.get_tile(channel, t, tile)
        t += tile


def _prefetch_scenario(
    loader,
    *,
    channel: int,
    window: float,
    config: PrefetchConfig,
    radius_fn,
    steps: int = 12,
) -> tuple[int, int]:
    cache = PrefetchCache(lambda c, s, e: _fetch_tuple(loader, c, s, e), config)
    cache.start()
    try:
        view_start = 0.0
        for _ in range(steps):
            radius = max(0.0, float(radius_fn(cache)))
            span = window + radius * 2.0
            prefetch_start = max(0.0, view_start - radius)
            cache.prefetch_window(channel, prefetch_start, span)
            time.sleep(0.02)
            _consume_window(cache, channel, view_start, window)
            view_start += window * 0.4
            if view_start + window > loader.duration_s:
                view_start = 0.0
        return cache.stats
    finally:
        cache.stop()


def run_profile(loader):
    windows = [10.0, 60.0, 120.0]
    channel = 0
    config = PrefetchConfig(tile_duration=5.0, max_tiles=240, max_bytes=128 * 1024 * 1024)
    service = PrefetchService(config)
    results = []
    fs = None
    fs_fn = getattr(loader, "fs", None)
    if callable(fs_fn):
        try:
            fs = float(fs_fn(channel))
        except Exception:
            fs = None
    for duration in windows:
        start = 0.0
        t0 = time.perf_counter()
        for _ in range(20):
            loader.read(channel, start, start + duration)
            start += duration / 10
            if start + duration > loader.duration_s:
                start = 0.0
        elapsed = time.perf_counter() - t0

        service.update_hints(channels=[channel], window_s=duration)

        def smart_radius(cache: PrefetchCache):
            est = cache.estimated_tile_bytes
            if est <= 0:
                est = None
            rates = [fs] if fs and fs > 0 else None
            return service.prefetch_radius(
                duration,
                channel_indices=[channel],
                sample_rates=rates,
                estimated_tile_bytes=est,
            )

        naive_radius = lambda _cache: duration  # noqa: E731

        smart_hits, smart_misses = _prefetch_scenario(
            loader,
            channel=channel,
            window=duration,
            config=config,
            radius_fn=smart_radius,
        )
        naive_hits, naive_misses = _prefetch_scenario(
            loader,
            channel=channel,
            window=duration,
            config=config,
            radius_fn=naive_radius,
        )
        results.append(
            (
                duration,
                elapsed,
                smart_hits,
                smart_misses,
                naive_hits,
                naive_misses,
            )
        )
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="EDF or Zarr file")
    args = parser.parse_args()

    loader = load_loader(args.path)
    try:
        results = run_profile(loader)
    finally:
        loader.close()

    for duration, elapsed, smart_hits, smart_misses, naive_hits, naive_misses in results:
        total_calls = smart_hits + smart_misses
        smart_rate = 0.0 if total_calls == 0 else smart_hits / total_calls
        naive_total = naive_hits + naive_misses
        naive_rate = 0.0 if naive_total == 0 else naive_hits / naive_total
        print(
            "window={:6.1f}s total={:0.3f}s avg={:0.4f}s smart_hit={:0.1%} naive_hit={:0.1%}".format(
                duration,
                elapsed,
                elapsed / 20,
                smart_rate,
                naive_rate,
            )
        )


if __name__ == "__main__":
    main()
