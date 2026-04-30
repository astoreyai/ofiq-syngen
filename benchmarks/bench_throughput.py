#!/usr/bin/env python3
"""Per-component throughput benchmark.

Measures wall-clock per degradation across all 28 components on a
synthetic input. Reports image/sec for severity 0.5 with no FaceContext.

Usage:
    python benchmarks/bench_throughput.py
    python benchmarks/bench_throughput.py --iterations 200
    python benchmarks/bench_throughput.py --output benchmarks/results/throughput.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from ofiq_syngen.components import COMPONENT_REGISTRY


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Synthetic image side length (default 256).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rng = np.random.RandomState(0)
    img = rng.randint(80, 200, (args.image_size, args.image_size, 3), dtype=np.uint8)

    print(f"Iterations per component: {args.iterations}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Severity: {args.severity}")
    print()

    rows = []
    fmt = "{:<40s}  {:>8s}  {:>10s}  {:>12s}"
    print(fmt.format("component", "ms/img", "img/sec", "n_funcs"))
    print("-" * 80)

    for component, degs in sorted(COMPONENT_REGISTRY.items()):
        for d in degs:
            d.function(img, args.severity, 42, None)

        t0 = time.perf_counter()
        for _ in range(args.iterations):
            for d in degs:
                d.function(img, args.severity, 42, None)
        t1 = time.perf_counter()

        total_calls = args.iterations * len(degs)
        ms_per = (t1 - t0) * 1000.0 / total_calls
        img_per_s = 1000.0 / ms_per if ms_per > 0 else float("inf")
        print(fmt.format(component, f"{ms_per:.2f}", f"{img_per_s:.1f}", str(len(degs))))
        rows.append({
            "component": component,
            "n_funcs": len(degs),
            "iterations": args.iterations,
            "image_size": args.image_size,
            "severity": args.severity,
            "ms_per_call": ms_per,
            "calls_per_sec": img_per_s,
        })

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {len(rows)} rows to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
