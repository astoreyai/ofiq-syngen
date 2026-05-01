"""NoHeadCoverings degradation: add hat/headwear to the forehead region.

OFIQ measures: proportion of BiSeNet class 16 (cloth) + class 18 (hat)
pixels in the upper head region (parsing map rows 0:196 of 400, after
cropping bottom 204 rows).

Approach:
- Primary: Composite textured hat/headband overlay onto the forehead
  region using landmark-guided positioning. The overlay uses colors and
  textures that BiSeNet is likely to classify as class 16/18.
- V2: Stable Diffusion inpainting with prompt "person wearing a hat".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


def add_head_covering(
    img: np.ndarray,
    severity: float,
    seed: int,
    ctx: FaceContext | None = None,
) -> np.ndarray:
    """Render a realistic hat on top of the head (Phase 3 [§7.4.13]).

    With FaceContext: delegates to ``occluders.render_hat()`` for a
    photorealistic dome-shaped hat with fabric texture, brim shadow on
    forehead, BiSeNet-trained colors, and feathered alpha edges.

    Without FaceContext: legacy heuristic colored band on forehead
    (lower fidelity).
    """
    if severity < 0.05:
        return img

    if ctx is not None:
        from ofiq_syngen.occluders import render_hat
        return render_hat(img, ctx, severity, seed)

    # Legacy fallback path retained for ctx=None calls
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    out = img.copy()

    # Determine the forehead/upper-head region
    if ctx is not None:
        landmarks = ctx.landmarks_98

        # Forehead region: above eyebrows, below top of head
        # Eyebrow top: min y of contour landmarks 33-46 (brow area)
        brow_indices = list(range(33, min(47, 98)))
        if brow_indices:
            brow_y = int(landmarks[brow_indices, 1].min())
        else:
            brow_y = int(h * 0.25)

        # Face contour gives the horizontal extent
        left_x = int(landmarks[0, 0])   # leftmost contour
        right_x = int(landmarks[32, 0]) # rightmost contour

        # Hat region: from top of image (or above brows) to brow line
        hat_top = max(0, brow_y - int((1 + severity) * ctx.t_metric * 0.4))
        hat_bottom = brow_y
    else:
        # Fallback: heuristic positioning
        hat_top = 0
        hat_bottom = int(h * 0.3)
        left_x = int(w * 0.1)
        right_x = int(w * 0.9)

    if hat_bottom <= hat_top or right_x <= left_x:
        return img

    # Expand coverage with severity
    coverage_height = int((hat_bottom - hat_top) * (0.3 + severity * 0.7))
    hat_top = max(0, hat_bottom - coverage_height)

    # Choose a hat-like color (solid fabric colors that BiSeNet recognizes)
    hat_colors = [
        (40, 40, 40),      # dark gray/black beanie
        (50, 50, 120),     # dark red/maroon
        (120, 80, 40),     # dark blue
        (30, 80, 30),      # dark green
        (60, 60, 100),     # brown
        (200, 200, 200),   # light gray
        (180, 180, 220),   # off-white/cream
    ]
    base_color = hat_colors[rng.randint(0, len(hat_colors))]

    # Create the hat mask with slight horizontal curvature
    hat_mask = np.zeros((h, w), dtype=np.uint8)

    # Draw filled region with curved bottom edge
    for x in range(max(0, left_x - 5), min(w, right_x + 5)):
        # Curved bottom: slight arc following head curvature
        center_x = (left_x + right_x) / 2
        dist_from_center = abs(x - center_x) / max(1, (right_x - left_x) / 2)
        curve_offset = int(dist_from_center ** 2 * coverage_height * 0.15)
        local_bottom = min(h, hat_bottom + curve_offset)

        hat_mask[hat_top:local_bottom, x] = 1

    # Add texture variation for realism
    texture = np.zeros_like(out, dtype=np.float32)
    for c in range(3):
        texture[:, :, c] = base_color[c]

    # Add subtle noise to simulate fabric texture
    fabric_noise = rng.randn(h, w) * 8
    for c in range(3):
        texture[:, :, c] = np.clip(texture[:, :, c] + fabric_noise, 0, 255)

    # Apply hat to image
    hat_region = hat_mask > 0
    alpha = 0.85 + severity * 0.1  # more opaque at higher severity
    for c in range(3):
        out[:, :, c][hat_region] = np.clip(
            out[:, :, c][hat_region].astype(np.float32) * (1 - alpha) +
            texture[:, :, c][hat_region] * alpha,
            0, 255,
        ).astype(np.uint8)

    # Add a slight edge/brim at the bottom for realism
    brim_thickness = max(1, int(severity * 4))
    for x in range(max(0, left_x - 5), min(w, right_x + 5)):
        center_x = (left_x + right_x) / 2
        dist_from_center = abs(x - center_x) / max(1, (right_x - left_x) / 2)
        curve_offset = int(dist_from_center ** 2 * coverage_height * 0.15)
        local_bottom = min(h - 1, hat_bottom + curve_offset)

        for by in range(max(0, local_bottom - brim_thickness), min(h, local_bottom + 1)):
            darker = [max(0, c - 30) for c in base_color]
            out[by, x] = darker

    return out
