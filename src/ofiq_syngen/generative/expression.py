"""ExpressionNeutrality degradation: 3DMM-aligned expression morph [§7.4.11].

OFIQ measures: HSEmotion EfficientNet (B0 224x224 -> 1280d, B2 260x260 ->
1408d) concatenated -> AdaBoost. The CNN keys on mouth shape, brow
position, and eye opening.

Approach (Phase 5):
- Run 3DDFA-V2 (already in FaceContext) to get the 62-dim BFM-2009
  parameter fit for this face: 12 pose + 40 identity + 10 expression.
- Add an emotion-template delta to the 10 expression coefficients.
- Reproject the 68 sparse BFM landmarks at the original AND modified
  expression. The displacement field is anatomically constrained by
  the BFM model (smiling lifts cheeks AND raises mouth corners AND
  narrows eyes simultaneously, all coupled via the blendshape).
- Use the 68-landmark displacement as TPS warp targets, but only
  composite the warped pixels back into a feathered mouth-region mask
  so the cheeks and jaw don't show TPS distortion artifacts.

If the bundled BFM basis files are not present, falls back to the
sparse hand-picked landmark warp (lower fidelity, no anatomical
coupling). The fallback path triggers a one-time warning.

V2: Stable Diffusion inpainting with a "person smiling" prompt for
photorealistic expression edits with proper teeth and skin folds. Out
of scope for the current package; see PIPELINE_SPEC.md Phase 6.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


# Explicit ADNet-98 -> iBUG-68 index map. Both standards label mouth /
# eye / brow landmarks in a fixed order, so the correspondence is exact
# at the lip corners, eye corners, and brow tips. ADNet has more
# landmarks than iBUG (98 vs 68), so we only map the ones with a clean
# anatomical correspondence and let the TPS warp fill in the rest.
#
# Mouth (12 outer + 8 inner ADNet -> 12 outer + 8 inner iBUG, exact 1:1):
#   ADNet 76-87 (outer)  <- iBUG 48-59
#   ADNet 88-95 (inner)  <- iBUG 60-67
# Brows (5 each side iBUG -> 9 each side ADNet, sample subset):
# Eyes (6 each side iBUG -> 8 each side ADNet, sample subset):
_IBUG68_TO_ADNET98: dict[int, int] = {
    # outer mouth (76 = right corner, 82 = left corner)
    76: 48, 77: 49, 78: 50, 79: 51, 80: 52, 81: 53,
    82: 54, 83: 55, 84: 56, 85: 57, 86: 58, 87: 59,
    # inner mouth (88 = right inner corner, 92 = left inner corner)
    88: 60, 89: 61, 90: 62, 91: 63,
    92: 64, 93: 65, 94: 66, 95: 67,
    # right brow (subset: outer corner, peak, inner corner)
    33: 17, 37: 19, 41: 21,
    # left brow
    42: 22, 46: 24, 50: 26,
    # right eye corners (60 = outer, 64 = inner)
    60: 36, 64: 39,
    # left eye corners (68 = inner, 72 = outer)
    68: 42, 72: 45,
    # nose tip
    54: 30,
}


def add_expression(
    img: np.ndarray,
    severity: float,
    seed: int,
    ctx: FaceContext | None = None,
) -> np.ndarray:
    """Add non-neutral expression. Method dispatch via env var:

    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p`` -> InstructPix2Pix whole-image
      edit (strongest emotion change, mild identity drift)
    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=sd_inpaint`` -> Stable Diffusion
      masked inpainting (best identity preservation, weak frown/surprise)
    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=tps`` -> hand-picked landmark TPS
      warp (legacy, fast, lowest fidelity)
    - default (or ``=3dmm``) -> 3DMM-aligned landmark morph (Phase 5)
    """
    import os
    if severity < 0.01 or ctx is None:
        return img

    rng = np.random.RandomState(seed)
    emotion = str(rng.choice(["smile", "frown", "surprise"]))

    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()

    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_expression_ip2p,
            )
            if is_sd_available():
                return render_expression_ip2p(img, ctx, emotion, severity, seed)
            warnings.warn(
                "IP2P requested but diffusers not installed; falling back to 3DMM.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(f"IP2P failed ({exc}); falling back to 3DMM.", stacklevel=2)

    if method in ("sd_inpaint", "sd", "diffusion"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_expression_sd,
            )
            if is_sd_available():
                return render_expression_sd(img, ctx, emotion, severity, seed)
            warnings.warn(
                "SD inpainting requested but diffusers not installed; "
                "install with `pip install ofiq-syngen[diffusion]`. "
                "Falling back to 3DMM.",
                stacklevel=2,
            )
        except Exception as exc:
            warnings.warn(
                f"SD inpainting failed ({exc}); falling back to 3DMM.",
                stacklevel=2,
            )

    if method == "tps":
        return _fallback_warp(img, severity, seed, ctx, emotion)

    # Default: 3DMM
    try:
        from ofiq_syngen.face_3dmm import (
            is_3dmm_available,
            parse_3ddfa_params,
            project_3dmm_landmarks,
            expression_delta,
        )
        if not is_3dmm_available():
            return _fallback_warp(img, severity, seed, ctx, emotion)
        if not hasattr(ctx, "raw_3ddfa_params") or ctx.raw_3ddfa_params is None:
            return _fallback_warp(img, severity, seed, ctx, emotion)

        return _apply_3dmm_expression(img, severity, seed, ctx, emotion,
                                       parse_3ddfa_params,
                                       project_3dmm_landmarks,
                                       expression_delta)
    except (FileNotFoundError, ImportError) as exc:
        warnings.warn(
            f"3DMM expression unavailable ({exc}); falling back to "
            "hand-picked landmark TPS warp.",
            stacklevel=2,
        )
        return _fallback_warp(img, severity, seed, ctx, emotion)


def _apply_3dmm_expression(
    img: np.ndarray, severity: float, seed: int, ctx, emotion: str,
    parse_fn, project_fn, delta_fn,
) -> np.ndarray:
    """3DMM path: morph expression coefficients, reproject, warp via TPS."""
    import cv2
    from ofiq_syngen.components import _apply_rbf_warp
    from ofiq_syngen.landmark_utils import MOUTH_OUTER

    h, w = img.shape[:2]
    R, offset, alpha_shp, alpha_exp = parse_fn(ctx.raw_3ddfa_params)

    # Project BFM 68 landmarks at original and modified expression.
    # Coordinates are in 3DDFA-V2 model space (offset and scaled
    # differently from ADNet 98 image-space landmarks), so we cannot
    # match by nearest-neighbor. Instead, use the explicit iBUG-68 ->
    # ADNet-98 index mapping (both follow standard layouts) and scale
    # displacements by the face-size ratio to bring them into ADNet
    # image-space pixels.
    src_bfm = project_fn(R, offset, alpha_shp, alpha_exp, w, h)        # (68, 2)
    delta = delta_fn(emotion, severity, current_expr=alpha_exp)        # (10,)
    new_alpha_exp = alpha_exp + delta.reshape(-1, 1)
    dst_bfm = project_fn(R, offset, alpha_shp, new_alpha_exp, w, h)    # (68, 2)
    bfm_displacement = dst_bfm - src_bfm                               # (68, 2)

    landmarks_98 = ctx.landmarks_98.astype(np.float64)

    # Face-size ratio: (ADNet face span) / (BFM face span), used to
    # scale BFM displacement vectors into ADNet image-space pixels.
    bfm_span = np.array([
        src_bfm[:, 0].max() - src_bfm[:, 0].min(),
        src_bfm[:, 1].max() - src_bfm[:, 1].min(),
    ])
    ad_span = np.array([
        landmarks_98[:, 0].max() - landmarks_98[:, 0].min(),
        landmarks_98[:, 1].max() - landmarks_98[:, 1].min(),
    ])
    scale = ad_span / np.maximum(bfm_span, 1.0)
    bfm_displacement_scaled = bfm_displacement * scale[None, :]

    src_pts = landmarks_98.copy()
    dst_pts = landmarks_98.copy()
    for adnet_idx, bfm_idx in _IBUG68_TO_ADNET98.items():
        dst_pts[adnet_idx] = landmarks_98[adnet_idx] + bfm_displacement_scaled[bfm_idx]

    t = ctx.t_metric
    warped = _apply_rbf_warp(img, src_pts, dst_pts, seed)

    # v0.5.1: tighten composite. v0.5.0 dilated the union mouth hull
    # by ~22px and softened with sigma=t/35, producing visible
    # warped-tooth artifacts at the chin when the TPS pushed the
    # mouth far below source position. Cap dilation at 8px and drop
    # the soft sigma to t/80 so the composite stays inside the actual
    # lip region.
    mouth_idx = MOUTH_OUTER
    src_mouth = ctx.landmarks_98[mouth_idx].astype(np.int32)
    dst_mouth = dst_pts[mouth_idx].astype(np.int32)
    union_pts = np.vstack([src_mouth, dst_mouth])
    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(union_pts)
    cv2.fillConvexPoly(mouth_mask, hull, 1)
    dilate_px = max(4, min(8, int(t * 0.02 + 4)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mouth_mask = cv2.dilate(mouth_mask, kernel)
    soft = cv2.GaussianBlur(mouth_mask.astype(np.float32), (0, 0), max(1.5, t / 80))
    soft = np.clip(soft, 0.0, 1.0)[..., None]
    out = img.astype(np.float32) * (1 - soft) + warped.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def _fallback_warp(
    img: np.ndarray, severity: float, seed: int, ctx, emotion: str,
) -> np.ndarray:
    """Fallback path when the BFM basis isn't available.

    Uses hand-picked ADNet landmark displacement instead of 3DMM
    reprojection. Same TPS + mouth-region composite as the main path.
    """
    from ofiq_syngen.components import _apply_rbf_warp

    landmarks = ctx.landmarks_98.astype(np.float64)
    t = ctx.t_metric
    h, w = img.shape[:2]

    s_eff = float(np.sqrt(np.clip(severity, 0.0, 1.0)))
    # v0.5.1: dropped from 0.18 to 0.10. The TPS warp was displacing
    # mouth landmarks 36px at sev=1.0 (t≈200) which interpolated into
    # a visible chin-area artifact even with corner anchors.
    displacement = s_eff * t * 0.10

    # Restrict the warp's source/destination to MOUTH points only
    # (mouth corners + outer + inner) and pass the surrounding
    # anatomical landmarks (chin, jaw, nose tip, eye corners,
    # eyebrows) as ANCHORS that map to themselves. This forces the
    # TPS interpolation to localize near the mouth instead of bleeding
    # into the chin / jaw.
    mouth_indices = [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                     88, 89, 90, 91, 92, 93, 94, 95]
    anchor_indices = [
        16,  # chin
        0, 8, 16, 24, 32,  # jaw outline (sparse)
        54,  # nose tip
        51, 57,  # nose base sides
        60, 64, 68, 72,  # eye corners
        33, 38, 42, 46, 50,  # eyebrows (sparse)
    ]
    # Build src/dst arrays: mouth points get displaced, anchors stay put
    src_points = []
    dst_points = []
    for idx in mouth_indices:
        src_points.append(landmarks[idx].copy())
        dst_points.append(landmarks[idx].copy())
    if emotion == "smile":
        for i, idx in enumerate(mouth_indices):
            if idx in (76, 82):
                dst_points[i][1] -= displacement
            elif idx in (77, 78, 86, 87):
                dst_points[i][1] -= displacement * 0.4
    elif emotion == "frown":
        for i, idx in enumerate(mouth_indices):
            if idx in (76, 82):
                dst_points[i][1] += displacement
            elif idx in (77, 78, 86, 87):
                dst_points[i][1] += displacement * 0.4
    else:  # surprise
        for i, idx in enumerate(mouth_indices):
            if idx in (88, 89, 90, 91):
                dst_points[i][1] -= displacement * 0.6
            elif idx in (92, 93, 94, 95):
                dst_points[i][1] += displacement * 0.6
    # Anchors map to themselves
    for idx in anchor_indices:
        src_points.append(landmarks[idx].copy())
        dst_points.append(landmarks[idx].copy())
    src_points = np.array(src_points, dtype=np.float64)
    dst_points = np.array(dst_points, dtype=np.float64)

    warped = _apply_rbf_warp(img, src_points, dst_points, seed)

    # v0.5.1c: with anchor-pinned TPS, the warp is already localized
    # to the mouth — areas outside the mouth-anchor convex hull
    # should be near-pixel-identical to the source. Skip the soft
    # mouth-mask composite entirely; the previous composite produced
    # a "chin ghost" artifact at the soft-blend boundary even with
    # tiny dilation, because sub-pixel warp interpolation differed
    # from the source through the alpha gradient.
    return warped
