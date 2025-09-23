#!/usr/bin/env python3
"""Quick render benchmark for EDF/Zarr viewer pipeline."""
from __future__ import annotations

import argparse
import time

from core.edf_loader import EdfLoader
from core.zarr_loader import ZarrLoader


def load_loader(path: str):
    if path.endswith('.zarr'):
        return ZarrLoader(path)
    return EdfLoader(path)


def run_profile(loader):
    windows = [10.0, 60.0, 120.0]
    channel = 0
    results = []
    for duration in windows:
        start = 0.0
        t0 = time.perf_counter()
        for _ in range(20):
            loader.read(channel, start, start + duration)
            start += duration / 10
            if start + duration > loader.duration_s:
                start = 0.0
        elapsed = time.perf_counter() - t0
        results.append((duration, elapsed))
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

    for duration, elapsed in results:
        print(f"window={duration:6.1f}s total={elapsed:0.3f}s avg={elapsed/20:0.4f}s")


if __name__ == "__main__":
    main()
