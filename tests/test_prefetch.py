import time

import numpy as np

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


def test_prefetch_service_configures_caches():
    service = PrefetchService()
    service.configure(tile_duration=2.0, max_tiles=5, max_mb=1.0)
    cache = service.create_cache(fake_fetch)
    assert cache.config.tile_duration == 2.0
    assert cache.config.max_tiles == 5
    assert cache.config.max_bytes == 1.0 * 1024 * 1024
