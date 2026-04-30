"""OpenCV-only baseline adapter.

The simplest possible perturbations using nothing but OpenCV. Useful as
a sanity-check baseline: any method losing to opencv_baseline on a
component is doing something wrong.

Covers the components for which a one-line OpenCV op is meaningful;
raises NotImplementedError for the rest (handled by the grid runner as
a skip).
"""

from __future__ import annotations

import cv2
import numpy as np


name = "opencv_baseline"


def degrade(image: np.ndarray, component: str, severity: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    s = max(0.0, min(1.0, severity))
    if component == "Sharpness.scalar":
        sigma = 0.5 + 4.5 * s
        return cv2.GaussianBlur(image, (0, 0), sigma)
    if component == "CompressionArtifacts.scalar":
        q = max(1, int(95 - 90 * s))
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, q])
        if not ok:
            return image
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if component == "LuminanceMean.scalar":
        gamma = 1.0 - 0.85 * s if s <= 0.5 else 1.0 + 4.0 * (s - 0.5)
        gamma = max(0.05, gamma)
        table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
        return cv2.LUT(image, table)
    if component == "HeadPoseRoll.scalar":
        h, w = image.shape[:2]
        angle = (s - 0.5) * 60.0
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    raise NotImplementedError(f"opencv_baseline does not implement {component}")
