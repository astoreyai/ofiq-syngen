#!/usr/bin/env python3
"""Cross-talk matrix builder.

For each (perturbation, component) pair, compute the OFIQ score delta
on the targeted component and on every other component. Builds a
28x28 matrix where ideally diagonal entries are large (perturbation
moves its target) and off-diagonal entries are small (no side effects).

Requires:
- OFIQ binary or GPUOFIQScorer (set OFIQ_BINARY env or have models on disk).
- A directory of canonical face images.

Usage:
    python scripts/run_crosstalk.py \
        --image-dir /path/to/face_images \
        --severity 0.7 \
        --output benchmarks/results/crosstalk.csv

Outputs:
- benchmarks/results/crosstalk.csv: rows=perturbation, cols=measured component,
  cells = mean delta across input images.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

from ofiq_syngen import DegradationPipeline
from ofiq_syngen.standards import STANDARDS_REFS


def _score_with_binary(binary: Path, image_path: Path) -> dict[str, float]:
    """Run OFIQ binary and return component scores."""
    import subprocess
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        list_p = td_p / "list.csv"
        list_p.write_text(str(image_path) + "\n")
        out_p = td_p / "scores.csv"
        subprocess.run(
            [str(binary), "-l", str(list_p), "-o", str(out_p)],
            check=True, capture_output=True,
        )
        rows = list(csv.DictReader(out_p.open()))
        if not rows:
            return {}
        return {k: float(v) for k, v in rows[0].items() if v not in ("", None) and v.replace(".", "", 1).replace("-", "", 1).isdigit()}


def _score_with_python() -> "callable":
    """Fall back to GPUOFIQScorer when binary not available."""
    from ofiq_syngen.gpu_ofiq_scorer import GPUOFIQScorer
    scorer = GPUOFIQScorer()
    return scorer.score_image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/results/crosstalk.csv"),
    )
    parser.add_argument("--severity", type=float, default=0.7)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument(
        "--ofiq-binary", type=Path, default=None,
        help="Path to OFIQ binary; if unset uses OFIQ_BINARY env, then GPUOFIQScorer.",
    )
    args = parser.parse_args()

    binary_path = args.ofiq_binary
    if binary_path is None and os.environ.get("OFIQ_BINARY"):
        binary_path = Path(os.environ["OFIQ_BINARY"])

    if binary_path and binary_path.exists():
        print(f"Scoring via OFIQ binary at {binary_path}")
        score = lambda img_path: _score_with_binary(binary_path, img_path)
    else:
        print("Scoring via GPUOFIQScorer (OFIQ binary not provided)")
        try:
            score = _score_with_python()
        except Exception as exc:
            print(f"GPUOFIQScorer unavailable: {exc}", file=sys.stderr)
            return 1

    images = sorted(args.image_dir.glob("*.png")) + sorted(args.image_dir.glob("*.jpg"))
    images = images[:args.max_images]
    if not images:
        print(f"No images in {args.image_dir}", file=sys.stderr)
        return 1

    components = sorted(STANDARDS_REFS.keys())
    pipeline = DegradationPipeline()

    matrix: dict[str, dict[str, list[float]]] = {p: {m: [] for m in components} for p in components}

    for img_path in images:
        src = cv2.imread(str(img_path))
        if src is None:
            continue
        baseline = score(img_path) if callable(score) and "image_path" in score.__code__.co_varnames else score(src)
        if not baseline:
            print(f"  warn: no baseline scores for {img_path.name}", file=sys.stderr)
            continue

        for perturb in components:
            try:
                degraded, _ = pipeline.degrade_single(src, perturb, args.severity, seed=42)
            except Exception as exc:
                print(f"  warn: degrade failed {perturb} on {img_path.name}: {exc}", file=sys.stderr)
                continue

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                cv2.imwrite(tf.name, degraded)
                tf_path = Path(tf.name)
            try:
                deg_scores = score(tf_path) if callable(score) and "image_path" in score.__code__.co_varnames else score(degraded)
            finally:
                tf_path.unlink(missing_ok=True)

            for measured in components:
                base = baseline.get(measured)
                deg = deg_scores.get(measured) if deg_scores else None
                if base is not None and deg is not None:
                    matrix[perturb][measured].append(deg - base)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation"] + components)
        for perturb in components:
            row = [perturb]
            for measured in components:
                vals = matrix[perturb][measured]
                row.append(f"{np.mean(vals):.3f}" if vals else "")
            writer.writerow(row)

    print(f"\nWrote 28x28 cross-talk matrix to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
