#!/usr/bin/env python3
"""Regenerate the OFIQ parity test manifest.

Runs each (image, component, severity) tuple through the syngen
pipeline, scores the degraded image with the OFIQ binary, and records
the expected score in the manifest. The test
``tests/test_ofiq_parity.py`` reads this manifest and asserts every
recorded score reproduces within tolerance.

Requires:
- OFIQ binary built and on disk (see tests/fixtures/ofiq_parity/REGENERATE.md).
- A directory of canonical face images.
- ofiq-syngen installed (`pip install -e .`).

Usage:
    python scripts/regenerate_parity_vectors.py \
        --ofiq-binary /path/to/OFIQSampleApp \
        --image-dir tests/fixtures/ofiq_parity/images \
        --output tests/fixtures/ofiq_parity/manifest.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import cv2

from ofiq_syngen import DegradationPipeline, __version__
from ofiq_syngen.standards import STANDARDS_REFS


def _image_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _run_ofiq(binary: Path, image_path: Path) -> dict[str, float]:
    """Invoke the OFIQ binary on a single image, return per-component scores.

    The OFIQ sample app produces a CSV per input. We pipe a single-image
    listing through it and parse the result.
    """
    # OFIQ 1.1.0 SampleApp: -c <configDir> -i <inputFile|Dir> -o <outputFile>
    # No -l (list) flag; invoke per image.
    config_dir = Path(
        "/mnt/projects/02_perception_biometrics/OFIQ-Project/data"
    )
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "scores.csv"
        cmd = [
            str(binary),
            "-c", str(config_dir),
            "-i", str(image_path),
            "-o", str(out_csv),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        # OFIQ outputs semicolon-separated CSV
        rows = list(csv.DictReader(out_csv.open(), delimiter=";"))
        if not rows:
            raise RuntimeError(f"OFIQ binary produced no rows for {image_path}")
        # Parse: keys are component names, values are stringified floats.
        out = {}
        for k, v in rows[0].items():
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                continue
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--ofiq-binary", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--severities", type=str, default="0.0,0.25,0.5,0.75,1.0",
        help="Comma-separated severity levels.",
    )
    parser.add_argument(
        "--components", type=str, default=None,
        help="Comma-separated component names. Defaults to all 28.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=5.0)
    args = parser.parse_args()

    if not args.ofiq_binary.exists():
        print(f"OFIQ binary not found at {args.ofiq_binary}", file=sys.stderr)
        return 1

    images = sorted(args.image_dir.glob("*.png")) + sorted(args.image_dir.glob("*.jpg"))
    if not images:
        print(f"No images in {args.image_dir}", file=sys.stderr)
        return 1

    severities = [float(s) for s in args.severities.split(",")]
    components = (
        args.components.split(",") if args.components
        else sorted(STANDARDS_REFS.keys())
    )

    pipeline = DegradationPipeline()
    image_set = []
    vectors = []

    for img_path in images:
        image_id = img_path.stem
        image_set.append({
            "id": image_id,
            "path": str(img_path.relative_to(args.image_dir.parent)),
            "sha256": _image_sha256(img_path),
        })
        src = cv2.imread(str(img_path))
        if src is None:
            print(f"  warn: cannot read {img_path}", file=sys.stderr)
            continue

        for component in components:
            for severity in severities:
                degraded, meta = pipeline.degrade_single(
                    src, component, severity, seed=args.seed,
                )
                with tempfile.NamedTemporaryFile(suffix=".png") as tf:
                    cv2.imwrite(tf.name, degraded)
                    scores = _run_ofiq(args.ofiq_binary, Path(tf.name))

                expected = scores.get(component)
                if expected is None:
                    print(
                        f"  warn: OFIQ produced no score for {component} on "
                        f"{image_id} sev={severity}", file=sys.stderr,
                    )
                    continue

                vectors.append({
                    "image_id": image_id,
                    "component": component,
                    "severity": severity,
                    "seed": args.seed,
                    "expected_score": expected,
                    "degradation_type": meta["degradation_type"],
                })

    manifest = {
        "schema_version": "1",
        "ofiq_version": "1.1.0",
        "syngen_version": __version__,
        "tolerance": args.tolerance,
        "tolerance_units": "raw OFIQ score (0-100)",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "generator_script": "scripts/regenerate_parity_vectors.py",
        "image_set": image_set,
        "vectors": vectors,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Wrote {len(vectors)} vectors over {len(image_set)} images to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
