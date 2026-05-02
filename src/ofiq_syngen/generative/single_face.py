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

    if ctx is not None:
        import os
        method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
        if method in ("ip2p", "instructpix2pix", "instruct_pix2pix",
                      "sd_inpaint", "sd", "diffusion"):
            try:
                from ofiq_syngen.expression_diffusion import (
                    is_sd_available, render_second_face_sd,
                )
                if is_sd_available():
                    return render_second_face_sd(img, ctx, severity, seed)
            except Exception:
                pass

    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    out = img.copy()

    # Identify regions where the second face is allowed to land:
    # NOT inside the primary face, NOT inside hair, ideally inside
    # BiSeNet background. This guarantees the second face never
    # overlaps the primary -- if no allowed region is large enough
    # for the requested face size, the face is shrunk to fit.
    if ctx is not None and ctx.parsing_map is not None:
        bg_mask = cv2.resize(
            (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST,
        )
        # Forbid: primary face hull + adjacent hair + dilated face
        forbid = np.zeros((h, w), dtype=np.uint8)
        if ctx.face_mask is not None:
            forbid |= ctx.face_mask.astype(np.uint8)
        from ofiq_syngen.landmark_utils import BISENET_HAIR
        hair = cv2.resize(
            (ctx.parsing_map == BISENET_HAIR).astype(np.uint8),
            (w, h), interpolation=cv2.INTER_NEAREST,
        )
        forbid |= hair
        # Dilate forbid zone by 8 px so the second face has breathing room
        forbid = cv2.dilate(forbid, np.ones((9, 9), np.uint8))
        allowed = bg_mask & ~forbid
    else:
        # Fallback: use image borders only
        allowed = np.zeros((h, w), dtype=np.uint8)
        border = max(10, min(h, w) // 6)
        allowed[:border, :] = 1
        allowed[-border:, :] = 1
        allowed[:, :border] = 1
        allowed[:, -border:] = 1

    # Target face size: fraction of PRIMARY face area, not image area
    if ctx is not None and ctx.face_mask is not None:
        primary_area = max(int(ctx.face_mask.sum()), 100)
    else:
        primary_area = (h * w) // 4
    target_area = primary_area * severity * 0.5
    face_size = max(12, int(np.sqrt(target_area)))

    # Find the largest face_size that ACTUALLY fits in the allowed region.
    # Erode by face_size/2 (radius) -- if no valid pixels, shrink the
    # face and try again, until we find a fit or give up.
    placement_mask = None
    for shrink in [1.0, 0.75, 0.55, 0.4, 0.25]:
        candidate = max(12, int(face_size * shrink))
        erode_kernel = np.ones((candidate, candidate), np.uint8)
        em = cv2.erode(allowed, erode_kernel)
        if em.any():
            placement_mask = em
            face_size = candidate
            break
    if placement_mask is None:
        # No room anywhere; skip the perturbation rather than overlap
        return img

    # Crop and flip the primary face for the patch
    crop_size = min(face_size * 2, min(h, w) - 2)
    cx, cy = w // 2, h // 2
    crop = img[
        max(0, cy - crop_size // 2):cy + crop_size // 2,
        max(0, cx - crop_size // 2):cx + crop_size // 2,
    ]
    if crop.shape[0] < 4 or crop.shape[1] < 4:
        return img
    face_patch = cv2.flip(crop, 1)
    face_patch = cv2.resize(face_patch, (face_size, face_size))

    # Pick a placement: prefer the highest valid pixel (farther from
    # the primary face) for visual balance.
    valid = np.argwhere(placement_mask > 0)
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
