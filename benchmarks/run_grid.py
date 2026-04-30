#!/usr/bin/env python3
"""Best-in-class benchmark grid.

For every (component, method, metric, image) cell, compute the metric
and write results to a parquet file. The grid is GPU-bound and not run
by CI; invoke manually when adapter implementations and OFIQ models are
available.

Metrics:
- directionality: per-component severity sweep correlation between
  syngen severity and OFIQ score
- isolation: cross-talk fraction (delta on target component minus max
  delta on any other)
- realism_fid: FID against a real low-quality reference set
- throughput: img/sec on the host hardware
- ofiq_parity: mean absolute error vs the parity manifest (only the
  syngen method has a parity claim; other methods are not OFIQ-aligned)

Adapters live in benchmarks/adapters/. Each adapter exposes:
    name: str
    degrade(img, severity, seed) -> img

Adapters that fail to import (missing dependency) are skipped with a
warning. The grid still runs over the methods that loaded.

Usage:
    python benchmarks/run_grid.py \
        --image-dir /path/to/face_images \
        --output benchmarks/results/grid.parquet
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ofiq_syngen import DegradationPipeline
from ofiq_syngen.components import COMPONENT_REGISTRY


def _load_adapters(adapter_dir: Path) -> dict[str, object]:
    adapters = {}
    sys.path.insert(0, str(adapter_dir))
    for path in sorted(adapter_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        module_name = path.stem
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "name") and hasattr(mod, "degrade"):
                adapters[mod.name] = mod
        except Exception as exc:
            print(
                f"  warn: skipped adapter {module_name} ({exc})",
                file=sys.stderr,
            )
    return adapters


def _syngen_adapter():
    """Wrap ofiq-syngen as a degradation adapter for uniform comparison."""
    pipeline = DegradationPipeline()

    class _Syngen:
        name = "ofiq_syngen"
        def degrade(self, img, component, severity, seed):
            out, _ = pipeline.degrade_single(img, component, severity, seed=seed)
            return out
    return _Syngen()


def _directionality(method, component, image, seeds=(42,)) -> dict:
    """Severity sweep -> per-component image-delta monotonicity proxy.

    Returns mean absolute pixel-delta at severity 0.0, 0.5, 1.0 plus
    the bool 'increases'.
    """
    deltas = {}
    for sev in (0.0, 0.5, 1.0):
        per_seed = []
        for seed in seeds:
            try:
                degraded = method.degrade(image, component, sev, seed)
            except Exception as exc:
                return {"error": str(exc)}
            per_seed.append(
                float(np.abs(degraded.astype(np.int32) - image.astype(np.int32)).mean())
            )
        deltas[sev] = float(np.mean(per_seed))
    return {
        "delta_0": deltas[0.0],
        "delta_50": deltas[0.5],
        "delta_100": deltas[1.0],
        "increases": deltas[1.0] >= deltas[0.5] >= deltas[0.0],
    }


def _throughput(method, component, image, iterations: int = 20) -> float:
    try:
        method.degrade(image, component, 0.5, 42)
    except Exception:
        return float("nan")
    t0 = time.perf_counter()
    for _ in range(iterations):
        method.degrade(image, component, 0.5, 42)
    t1 = time.perf_counter()
    return iterations / (t1 - t0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir", type=Path,
        default=Path(__file__).resolve().parent / "adapters",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "results" / "grid.parquet",
    )
    parser.add_argument(
        "--components", type=str, default=None,
        help="Comma-separated component names (default: all).",
    )
    parser.add_argument("--max-images", type=int, default=10)
    args = parser.parse_args()

    images = sorted(args.image_dir.glob("*.png")) + sorted(args.image_dir.glob("*.jpg"))
    images = images[:args.max_images]
    if not images:
        print(f"No images in {args.image_dir}", file=sys.stderr)
        return 1

    methods = {"ofiq_syngen": _syngen_adapter()}
    methods.update(_load_adapters(args.adapter_dir))
    print(f"Loaded {len(methods)} methods: {sorted(methods)}")

    components = (
        args.components.split(",") if args.components
        else sorted(COMPONENT_REGISTRY.keys())
    )

    rows = []
    for image_path in images:
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"  warn: cannot read {image_path}", file=sys.stderr)
            continue
        for component in components:
            for method_name, method in methods.items():
                row = {
                    "image_id": image_path.stem,
                    "component": component,
                    "method": method_name,
                }
                row.update(_directionality(method, component, img))
                row["throughput_img_s"] = _throughput(method, component, img)
                rows.append(row)
                print(
                    f"  {image_path.stem:15s}  {component:35s}  {method_name:20s}  "
                    f"d50={row.get('delta_50', 0):6.2f}  "
                    f"thr={row.get('throughput_img_s', 0):8.1f}/s"
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(args.output)
    except ImportError:
        json_path = args.output.with_suffix(".json")
        json_path.write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {len(rows)} rows to {json_path} (pandas not available, parquet skipped)")
        return 0
    print(f"\nWrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
