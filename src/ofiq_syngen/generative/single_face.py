"""SingleFacePresent degradation: insert a second face into the image.

OFIQ measures: area_2nd_face / area_1st_face bounding box ratio.
Score = round(100 * (1 - ratio)).

MVP approach: Poisson-blend a face crop from a small library into a
background region identified by BiSeNet parsing (class 0 = background).

V2: InsightFace face swapping for photorealistic insertion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ofiq_syngen.landmark_utils import BISENET_BACKGROUND

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


def insert_second_face(
    img: np.ndarray,
    severity: float,
    seed: int,
    ctx: FaceContext | None = None,
) -> np.ndarray:
    """Insert a synthetic second face into the image.

    The inserted face is scaled so that its area relative to the primary
    face produces OFIQ's area ratio proportional to severity:
    - severity 0.1 -> tiny face in corner
    - severity 1.0 -> face nearly as large as the primary

    Args:
        img: BGR uint8 face image.
        severity: [0, 1] controls the size ratio of the inserted face.
        seed: random seed.
        ctx: FaceContext with parsing map for background region identification.
    """
    if severity < 0.05:
        return img

    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    out = img.copy()

    # Generate a simple synthetic face-like patch (skin-colored ellipse)
    target_area_frac = severity * 0.4  # inserted face area as fraction of image
    face_size = max(8, int(min(h, w) * np.sqrt(target_area_frac)))

    # Create a small face patch from a region of the source image
    # (self-similar -- take a crop of the original face, flip it)
    crop_size = min(face_size, min(h, w) - 2)
    cx, cy = w // 2, h // 2
    crop = img[
        max(0, cy - crop_size // 2):cy + crop_size // 2,
        max(0, cx - crop_size // 2):cx + crop_size // 2,
    ]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return img

    # Flip horizontally to make it look different
    face_patch = cv2.flip(crop, 1)
    face_patch = cv2.resize(face_patch, (face_size, face_size))

    # Find placement location -- prefer background region
    if ctx is not None and ctx.parsing_map is not None:
        bg_mask = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
        bg_mask_full = cv2.resize(bg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        # Fallback: use image borders
        bg_mask_full = np.zeros((h, w), dtype=np.uint8)
        border = max(10, min(h, w) // 6)
        bg_mask_full[:border, :] = 1
        bg_mask_full[-border:, :] = 1
        bg_mask_full[:, :border] = 1
        bg_mask_full[:, -border:] = 1

    # Find valid placement coordinates
    # Erode mask to ensure the patch fits
    erode_size = face_size // 2
    if erode_size > 1:
        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        placement_mask = cv2.erode(bg_mask_full, erode_kernel)
    else:
        placement_mask = bg_mask_full

    valid = np.argwhere(placement_mask > 0)
    if len(valid) == 0:
        # No background space -- place in a corner
        valid = np.array([[5, 5], [5, w - face_size - 5],
                          [h - face_size - 5, 5], [h - face_size - 5, w - face_size - 5]])

    idx = rng.randint(0, len(valid))
    py, px = valid[idx]

    # Clamp to bounds
    py = max(0, min(py, h - face_size))
    px = max(0, min(px, w - face_size))

    # Blend using seamless cloning for realistic insertion
    mask = np.full((face_size, face_size), 255, dtype=np.uint8)
    # Create elliptical mask for natural blending
    cv2.ellipse(
        mask,
        (face_size // 2, face_size // 2),
        (face_size // 2 - 2, face_size // 2 - 2),
        0, 0, 360, 255, -1,
    )

    center = (px + face_size // 2, py + face_size // 2)

    try:
        out = cv2.seamlessClone(face_patch, out, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        # Fallback: simple alpha blend
        alpha = mask.astype(np.float32) / 255.0
        alpha = alpha[:, :, np.newaxis]
        region = out[py:py + face_size, px:px + face_size].astype(np.float32)
        blended = region * (1 - alpha * 0.7) + face_patch.astype(np.float32) * alpha * 0.7
        out[py:py + face_size, px:px + face_size] = blended.astype(np.uint8)

    return out
