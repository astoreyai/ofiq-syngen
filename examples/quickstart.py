"""Quickstart: run one operator from each tier and write a strip.

Tiers:
    1. 2D pure (always works) -- Sharpness via cv2 Gaussian blur
    2. SD inpaint / IP2P (needs [diffusion]) -- ExpressionNeutrality via IP2P
    3. 3D FLAME (needs [three_d] + assets) -- HeadPoseYaw via DECA + pyrender

Run:
    python examples/quickstart.py path/to/face.jpg
    OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p python examples/quickstart.py path/to/face.jpg

Output:
    quickstart_strip.png -- side-by-side composition of the four results.
"""

from __future__ import annotations

import sys

import cv2
import numpy as np

from ofiq_syngen import DegradationPipeline


def main(image_path: str) -> int:
    img = cv2.imread(image_path)
    if img is None:
        print(f"could not read {image_path}", file=sys.stderr)
        return 1

    pipeline = DegradationPipeline()

    examples = [
        ("Sharpness.scalar", "Sharpness sev=0.7"),
        ("ExpressionNeutrality.scalar", "Expression sev=0.7"),
        ("EyesVisible.scalar", "Sunglasses sev=0.8"),
        ("HeadPoseYaw.scalar", "Yaw sev=0.5"),
    ]

    panels = [_label(img, "original")]
    for component, label in examples:
        sev = 0.7 if "Sharpness" in component or "Expression" in component else 0.5
        if "Eyes" in component:
            sev = 0.8
        out, meta = pipeline.degrade_single(img, component, severity=sev, seed=42)
        panels.append(_label(out, label))
        print(f"{component:<35} ok ({meta.get('degradation_type', '?')})")

    strip = np.concatenate(panels, axis=1)
    out_path = "quickstart_strip.png"
    cv2.imwrite(out_path, strip)
    print(f"\nwrote {out_path}")
    return 0


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (4, 4), (180, 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python examples/quickstart.py path/to/face.jpg", file=sys.stderr)
        sys.exit(1)
    sys.exit(main(sys.argv[1]))
