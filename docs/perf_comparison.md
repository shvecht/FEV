# Performance quick reference

This note tracks the trade-offs between the live EDF reader, RAM-backed int16 caches, and the Zarr background ingest path. Use it when deciding which toggles to enable for a given workstation.

## RAM-backed int16 cache

The viewer can promote an EDF into a float32-on-demand cache by first loading the raw digital samples into int16 arrays. This keeps the EDF reader idle while the UI serves decimated views from memory.

### Enabling the cache

Set the `[cache]` section in `config.ini`:

```ini
[cache]
enabled = true
max_mb = 768.0
memmap = false
```

* `enabled` gates the feature.
* `max_mb` is a hard cap: the EDF's on-disk size (or summed `n_samples × 2 bytes` when the filesystem check is unavailable) must fit inside this budget before the viewer builds the cache.
* `memmap` keeps the cache on disk via `numpy.memmap` if you would rather trade SSD bandwidth for RAM. Leave this `false` to keep everything hot in RAM.

The viewer logs a warning when the cap is too small, when the loader does not implement caching, or when an exception occurs during the build. Successful builds switch the source badge to **“Source: EDF (RAM cache)”** so operators can confirm the active mode at a glance.

### Sizing guidance

* Budget roughly **2 bytes per sample per channel**; add 10% headroom for PyEDFlib metadata copies while the cache is assembling.
* Disable the cache (set `enabled = false`) on machines with < 8 GiB of free RAM or when opening large multi-night studies that exceed the configured limit.
* Prefer `memmap = true` on shared desktops where RAM pressure is high but you can afford a fast NVMe scratch directory.

## When to rely on Zarr

The background Zarr ingest remains the long-term answer for skimming entire nights. Its chunked layout scales to arbitrarily large recordings and is resilient to mid-study restarts. Use the RAM cache when you need immediate responsiveness before the Zarr build completes or when the study is small enough to fit comfortably in memory.
