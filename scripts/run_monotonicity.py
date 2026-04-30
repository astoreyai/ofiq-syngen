#!/usr/bin/env python3
"""Per-component monotonicity sweep.

For each component, sweeps severity 0.0 to 1.0 in 10 steps over a face
image set, scores each degraded image with OFIQ, and computes the
Spearman rank correlation between syngen severity and OFIQ component
score. Saturating components are flagged.

Closes the calibration gap identified in the FIQA systematic review:
"does moving the underlying property predictably move the OFIQ score
that claims to measure it?"

Usage:
    python scripts/run_monotonicity.py \
        --image-dir /path/to/face_images \
        --output benchmarks/results/monotonicity.csv

Output CSV: rows=(component, image), columns=severity values + spearman_r + monotonic.
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


SEVERITIES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. No scipy dependency."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    d2 = float(np.sum((rx - ry) ** 2))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path,
        default=Path("benchmarks/results/monotonicity.csv"),
    )
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
        import subprocess
        def score_image(image_path):
            with tempfile.TemporaryDirectory() as td:
                td_p = Path(td)
                list_p = td_p / "list.csv"
                list_p.write_text(str(image_path) + "\n")
                out_p = td_p / "scores.csv"
                subprocess.run(
                    [str(binary_path), "-l", str(list_p), "-o", str(out_p)],
                    check=True, capture_output=True,
                )
                rows = list(csv.DictReader(out_p.open()))
                if not rows:
                    return {}
                return {k: float(v) for k, v in rows[0].items() if v not in ("", None) and v.replace(".", "", 1).replace("-", "", 1).isdigit()}
        score_takes_path = True
    else:
        print("Scoring via GPUOFIQScorer")
        try:
            from ofiq_syngen.gpu_ofiq_scorer import GPUOFIQScorer
            scorer = GPUOFIQScorer()
            score_image = scorer.score_image
            score_takes_path = False
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["component", "image_id", "spearman_r", "monotonic"]
            + [f"sev_{s:.1f}" for s in SEVERITIES]
        )

        for component in components:
            for img_path in images:
                src = cv2.imread(str(img_path))
                if src is None:
                    continue
                scores = []
                for sev in SEVERITIES:
                    try:
                        degraded, _ = pipeline.degrade_single(src, component, sev, seed=42)
                    except Exception:
                        scores.append(None)
                        continue
                    if score_takes_path:
                        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                            cv2.imwrite(tf.name, degraded)
                            tf_p = Path(tf.name)
                        try:
                            s = score_image(tf_p).get(component)
                        finally:
                            tf_p.unlink(missing_ok=True)
                    else:
                        s = score_image(degraded).get(component)
                    scores.append(s)

                valid_pairs = [(s, sc) for s, sc in zip(SEVERITIES, scores) if sc is not None]
                if len(valid_pairs) < 3:
                    continue
                xs = [p[0] for p in valid_pairs]
                ys = [p[1] for p in valid_pairs]
                rho = _spearman(xs, ys)
                monotonic = (
                    all(ys[i] <= ys[i + 1] for i in range(len(ys) - 1))
                    or all(ys[i] >= ys[i + 1] for i in range(len(ys) - 1))
                )
                writer.writerow(
                    [component, img_path.stem, f"{rho:.3f}", str(monotonic)]
                    + [f"{s:.2f}" if s is not None else "" for s in scores]
                )
                print(f"  {component:35s}  {img_path.stem:15s}  rho={rho:+.3f}  mono={monotonic}")

    print(f"\nWrote per-component monotonicity sweep to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
