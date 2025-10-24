import time

import numpy as np
import pytest

from core.prefetch import PrefetchCache, PrefetchConfig, PrefetchService


def fake_fetch(channel: int, start: float, end: float):
    t = np.linspace(start, end, num=10, endpoint=False)
    x = np.full_like(t, fill_value=channel)
    return t, x


def test_prefetch_basic_get():
    cache = PrefetchCache(fake_fetch, PrefetchConfig(tile_duration=1.0, max_tiles=2))
    t, x = cache.get_tile(0, 0.0, 1.0)
    assert t[0] == 0.0
    assert np.all(x == 0)


def test_prefetch_enqueue_and_worker():
    cache = PrefetchCache(fake_fetch, PrefetchConfig(tile_duration=1.0, max_tiles=4))
    cache.start()
    try:
        cache.prefetch_window(1, 0.0, 3.0)
        time.sleep(0.2)
        with cache._lock:
            assert len(cache._cache) > 0
    finally:
        cache.stop()


def test_prefetch_preview_runs_before_final():
    calls: list[tuple[str, float, float]] = []

    def preview_fetch(channel: int, start: float, end: float):
        calls.append(("preview", start, end))
        t = np.linspace(start, end, num=4, endpoint=False)
        x = np.full_like(t, fill_value=-1)
        return t, x

    def final_fetch(channel: int, start: float, end: float):
        calls.append(("final", start, end))
        return fake_fetch(channel, start, end)

    cache = PrefetchCache(
        final_fetch,
        PrefetchConfig(tile_duration=1.0, max_tiles=4),
        preview_fetch=preview_fetch,
    )
    cache.start()
    try:
        cache.prefetch_window(0, 0.0, 2.0)
        time.sleep(0.3)
    finally:
        cache.stop()

    assert calls, "prefetch should invoke preview and final stages"
    assert calls[0][0] == "preview"
    assert any(stage == "final" for stage, *_ in calls)


def test_prefetch_service_configures_caches():
    service = PrefetchService()
    service.configure(tile_duration=2.0, max_tiles=5, max_mb=1.0)
    cache = service.create_cache(fake_fetch)
    assert cache.config.tile_duration == 2.0
    assert cache.config.max_tiles == 5
    assert cache.config.max_bytes == 1.0 * 1024 * 1024


def _heavy_fetch(channel: int, start: float, end: float):
    # Each tile is intentionally large so that size-based eviction should trigger.
    t = np.linspace(start, end, num=512, endpoint=False, dtype=np.float32)
    x = np.full_like(t, fill_value=channel, dtype=np.float32)
    return t, x


def test_prefetch_get_respects_max_bytes_budget():
    config = PrefetchConfig(tile_duration=1.0, max_tiles=10, max_bytes=1024.0)
    cache = PrefetchCache(_heavy_fetch, config)

    for idx in range(3):
        cache.get_tile(0, float(idx), 1.0)
        with cache._lock:
            assert len(cache._cache) == 1


def test_prefetch_cache_tracks_hits_and_misses():
    cache = PrefetchCache(fake_fetch, PrefetchConfig(tile_duration=1.0, max_tiles=4))
    cache.get_tile(0, 0.0, 1.0)
    cache.get_tile(0, 0.0, 1.0)
    hits, misses = cache.stats
    assert hits == 1
    assert misses == 1


def test_prefetch_service_radius_limits_span():
    service = PrefetchService(PrefetchConfig(tile_duration=5.0, max_tiles=6))
    radius = service.prefetch_radius(10.0, channel_indices=[0, 1])
    assert radius == pytest.approx(2.5)


def test_prefetch_service_radius_respects_byte_budget():
    config = PrefetchConfig(tile_duration=1.0, max_tiles=100, max_bytes=10_000.0)
    service = PrefetchService(config)
    radius = service.prefetch_radius(
        2.0,
        channel_indices=[0, 1],
        sample_rates=[200.0, 200.0],
    )
    assert radius == pytest.approx(0.041666, rel=1e-4)
