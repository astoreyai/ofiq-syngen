"""ExpressionNeutrality degradation: add non-neutral expression.

OFIQ measures: HSEmotion EfficientNet (B0 224x224 -> 1280d, B2 260x260 -> 1408d)
concatenated features -> AdaBoost classifier. Score via sigmoid.

Approach:
- Primary: Landmark TPS warp to simulate expression (raise eyebrows,
  smile curve, wide mouth). OFIQ's CNNs detect geometric face changes.
- V2: Diffusion-based face editing (InstructPix2Pix or ip-adapter).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


def add_expression(
    img: np.ndarray,
    severity: float,
    seed: int,
    ctx: FaceContext | None = None,
) -> np.ndarray:
    """Add non-neutral expression via landmark warping.

    Simulates expressions by warping facial landmarks:
    - Smile: mouth corners (76, 82) move up
    - Surprise: eyebrows (33-37, 42-46) move up, mouth opens
    - Combined at higher severity

    Args:
        img: BGR uint8 face image.
        severity: [0, 1] expression intensity.
        seed: random seed.
        ctx: FaceContext with landmarks for guided warping.
    """
    if severity < 0.05 or ctx is None:
        return img

    from ofiq_syngen.components import _apply_rbf_warp

    rng = np.random.RandomState(seed)
    landmarks = ctx.landmarks_98.astype(np.float64)
    t = ctx.t_metric

    src_points = landmarks.copy()
    dst_points = landmarks.copy()

    displacement = severity * t * 0.15

    # Choose expression type
    expr_type = rng.choice(["smile", "surprise", "frown"])

    if expr_type == "smile":
        # Raise mouth corners (76 = right corner, 82 = left corner)
        for idx in [76, 82]:
            dst_points[idx, 1] -= displacement * 0.8

        # Slightly raise cheeks (contour points near mouth)
        for idx in [5, 6, 7, 25, 26, 27]:
            dst_points[idx, 1] -= displacement * 0.3

        # Squint eyes slightly (upper eyelids move down a tiny bit)
        for idx in [61, 62, 63, 69, 70, 71]:
            dst_points[idx, 1] += displacement * 0.15

    elif expr_type == "surprise":
        # Raise eyebrows (contour indices 33-37 left brow area, 42-46 right)
        # ADNet 98-pt: eyebrow region is in contour landmarks
        # Upper contour (forehead/brow area): indices 33-42
        for idx in range(33, 43):
            if idx < 98:
                dst_points[idx, 1] -= displacement * 1.2

        # Open mouth wide (same mechanism as MouthClosed degradation)
        for idx in [88, 89, 90, 91]:
            dst_points[idx, 1] -= displacement * 0.6
        for idx in [92, 93, 94, 95]:
            dst_points[idx, 1] += displacement * 0.6

        # Widen eyes (upper eyelids move up)
        for idx in [61, 62, 63, 69, 70, 71]:
            dst_points[idx, 1] -= displacement * 0.4

    elif expr_type == "frown":
        # Lower mouth corners
        for idx in [76, 82]:
            dst_points[idx, 1] += displacement * 0.5

        # Furrow brows (inner brow points move down and together)
        for idx in range(37, 42):
            if idx < 98:
                dst_points[idx, 1] += displacement * 0.4
                # Move slightly inward
                center_x = (landmarks[37, 0] + landmarks[42, 0]) / 2
                direction = 1 if landmarks[idx, 0] > center_x else -1
                dst_points[idx, 0] -= direction * displacement * 0.2

    return _apply_rbf_warp(img, src_points, dst_points, seed)
