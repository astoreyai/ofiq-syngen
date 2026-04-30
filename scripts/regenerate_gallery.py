#!/usr/bin/env python3
"""Regenerate the per-component severity grid gallery.

For each registered component, produces a 1x5 strip
(severity 0.0, 0.25, 0.5, 0.75, 1.0) on a canonical face image. Writes
PNG strips to docs/gallery/images/ and a Markdown index to
docs/gallery/<Component>.md.

Usage:
    python scripts/regenerate_gallery.py --face docs/gallery/canonical.png
    python scripts/regenerate_gallery.py --face docs/gallery/canonical.png --components Sharpness,CompressionArtifacts

The face image is not bundled (license caveats). See
docs/gallery/README.md for sourcing guidance.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ofiq_syngen import DegradationPipeline
from ofiq_syngen.components import COMPONENT_REGISTRY
from ofiq_syngen.standards import STANDARDS_REFS


SEVERITIES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _strip(images: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = images[0].shape[0]
    parts = []
    for i, img in enumerate(images):
        if i > 0:
            parts.append(np.full((h, gap, 3), 32, dtype=np.uint8))
        parts.append(img)
    return np.concatenate(parts, axis=1)


def _annotate(image: np.ndarray, severity: float) -> np.ndarray:
    out = image.copy()
    text = f"{severity:.2f}"
    cv2.rectangle(out, (4, 4), (60, 28), (0, 0, 0), -1)
    cv2.putText(
        out, text, (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
    )
    return out


def _component_md(component: str, strip_path: Path, gallery_dir: Path) -> str:
    refs = STANDARDS_REFS.get(component)
    rel = strip_path.relative_to(gallery_dir.parent)
    lines = [
        f"# {component}",
        "",
        f"![{component} severity strip]({rel})",
        "",
        "Severity: 0.00, 0.25, 0.50, 0.75, 1.00 left to right.",
        "",
    ]
    if refs is not None:
        lines.extend([
            "## Standards",
            "",
            f"- OFIQ section: `{refs.ofiq_section}`",
            f"- ISO/IEC 29794-5: {refs.iso_29794_5}",
            f"- ISO/IEC 19794-5: {refs.iso_19794_5}",
            f"- ICAO 9303 P9: {refs.icao_9303}",
            f"- Alignment: `{refs.alignment}`, confidence: `{refs.confidence}`",
            f"- OFIQ versions: {', '.join(refs.ofiq_versions)}",
            "",
        ])
    descs = [d.description for d in COMPONENT_REGISTRY[component]]
    lines.extend([
        "## Degradation method",
        "",
    ])
    for desc in descs:
        lines.append(f"- {desc}")
    lines.extend([
        "",
        "## Notes",
        "",
        "Add per-component caption text describing where the perturbation "
        "is visually convincing and where it betrays its synthetic origin.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--face", type=Path, required=True,
                        help="Canonical face image path (PNG or JPG)")
    parser.add_argument("--components", type=str, default=None,
                        help="Comma-separated component names (default: all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gallery-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "docs" / "gallery",
    )
    args = parser.parse_args()

    if not args.face.exists():
        print(f"Face image not found: {args.face}", file=sys.stderr)
        return 1

    src = cv2.imread(str(args.face))
    if src is None:
        print(f"Failed to read image: {args.face}", file=sys.stderr)
        return 1

    components = (
        args.components.split(",") if args.components
        else sorted(COMPONENT_REGISTRY.keys())
    )

    gallery_dir = args.gallery_dir
    images_dir = gallery_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DegradationPipeline()
    index_lines = ["# Gallery", "", "Per-component severity strips on a canonical face image.", ""]

    for component in components:
        try:
            frames = []
            for sev in SEVERITIES:
                degraded, _ = pipeline.degrade_single(src, component, sev, seed=args.seed)
                frames.append(_annotate(degraded, sev))
            strip = _strip(frames)
            short = component.replace(".scalar", "")
            strip_path = images_dir / f"{short}_strip.png"
            cv2.imwrite(str(strip_path), strip)
            md_path = gallery_dir / f"{short}.md"
            md_path.write_text(_component_md(component, strip_path, gallery_dir))
            print(f"  {component:40s} -> {strip_path.name}")
            index_lines.append(f"- [{component}]({short}.md)")
        except Exception as exc:
            print(f"  {component:40s} FAILED: {exc}", file=sys.stderr)

    (gallery_dir / "INDEX.md").write_text("\n".join(index_lines) + "\n")
    print(f"\nWrote {len(components)} component pages to {gallery_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
