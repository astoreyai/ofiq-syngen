"""ISO/IEC 29794-5 Component-Aligned Degradation Functions.

Each degradation targets a specific OFIQ quality component by perturbing
exactly what OFIQ measures, using the same regions and analysis pipeline.
Severity is normalized to [0, 1] where 0 = no degradation, 1 = maximum.
All functions are deterministic given a seed.

Functions that require face analysis receive a FaceContext parameter
(landmarks, parsing map, occlusion mask, head pose, derived metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from ofiq_syngen.landmark_utils import (
    BISENET_BACKGROUND,
    MOUTH_OUTER,
    PAIRS_LEFT_EYE,
    PAIRS_RIGHT_EYE,
)
from ofiq_syngen.standards import STANDARDS_REFS, StandardRefs

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


@dataclass
class ComponentDegradation:
    """Registration entry for a degradation function."""

    ofiq_component: str
    function: Callable
    description: str
    severity_range: str
    requires_context: bool = False
    standard_refs: StandardRefs | None = None


# =========================================================================
# Section 6 -- Capture-Related Components
# =========================================================================

def _background_clutter_segmented(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Add structured noise to the segmented background region [S6.1].

    OFIQ measures Sobel gradient magnitude on the BiSeNet-segmented
    background (class 0), eroded with a 4x4 kernel. We add edge-creating
    noise within that exact mask.
    """
    if ctx is None or ctx.parsing_map is None:
        return _background_clutter_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]

    # Get background mask from parsing map (class 0)
    bg_mask_small = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
    bg_mask = cv2.resize(bg_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)

    # Erode with 4x4 kernel (matching OFIQ's BackgroundUniformity.cpp)
    kernel = np.ones((4, 4), np.uint8)
    bg_mask = cv2.erode(bg_mask, kernel)

    if bg_mask.sum() == 0:
        return img

    # Add structured noise that creates Sobel gradients (edges, not just pixel noise)
    out = img.copy()
    noise_intensity = int(severity * 200)
    if noise_intensity < 1:
        return img

    # Random rectangles and edges within background to create gradient spikes
    n_patches = max(3, int(severity * 20))
    bg_coords = np.argwhere(bg_mask > 0)
    if len(bg_coords) == 0:
        return img

    for _ in range(n_patches):
        idx = rng.randint(0, len(bg_coords))
        cy, cx = bg_coords[idx]
        patch_size = rng.randint(3, max(4, int(severity * 40)))
        rng.randint(0, 256, 3).tolist()

        y1, y2 = max(0, cy - patch_size // 2), min(h, cy + patch_size // 2)
        x1, x2 = max(0, cx - patch_size // 2), min(w, cx + patch_size // 2)

        patch_mask = bg_mask[y1:y2, x1:x2]
        for c in range(3):
            channel = out[y1:y2, x1:x2, c]
            channel[patch_mask > 0] = np.clip(
                channel[patch_mask > 0].astype(np.int16) +
                rng.randint(-noise_intensity, noise_intensity + 1, channel[patch_mask > 0].shape),
                0, 255,
            ).astype(np.uint8)

    return out


def _background_clutter_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: border-based background noise when no context available."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    out = img.copy()
    border = int(min(h, w) * 0.15)
    noise = rng.randint(0, int(severity * 200 + 1), (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    out[mask] = np.clip(
        out[mask].astype(np.int16) + noise[mask].astype(np.int16), 0, 255
    ).astype(np.uint8)
    return out


def _uneven_illumination_roi(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Darken one ROI zone to create left-right illumination asymmetry [S6.2].

    OFIQ measures histogram intersection between left and right ROI zones
    (computed from eye centers and IED). We darken one zone to reduce the
    histogram overlap.
    """
    if ctx is None:
        return _uneven_illumination_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    out = img.astype(np.float32)
    h, w = img.shape[:2]

    # Pick which ROI to darken
    target_roi = ctx.left_roi if rng.random() < 0.5 else ctx.right_roi
    rx, ry, rw, rh = target_roi

    # Clamp to image bounds
    x1, y1 = max(0, rx), max(0, ry)
    x2, y2 = min(w, rx + rw), min(h, ry + rh)

    if x2 <= x1 or y2 <= y1:
        return img

    factor = 1.0 - severity * 0.8
    out[y1:y2, x1:x2] *= factor

    return np.clip(out, 0, 255).astype(np.uint8)


def _uneven_illumination_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: gradient lighting."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    direction = rng.choice(["left", "right"])
    gradient = np.linspace(1.0 - severity * 0.8, 1.0, w)
    if direction == "right":
        gradient = gradient[::-1]
    mask = np.tile(gradient, (h, 1))
    mask = np.stack([mask] * 3, axis=-1)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _darken_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Darken within the face mask region [S6.3, S6.4].

    OFIQ measures luminance histogram within the face landmark mask.
    """
    factor = 1.0 - severity * 0.85
    darkened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if ctx is not None and ctx.face_mask is not None:
        mask3 = ctx.face_mask[:, :, np.newaxis] > 0
        return np.where(mask3, darkened, img)
    return darkened


def _darken_face_with_occlusion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Darken within face mask AND occlusion mask [S6.4 UnderExposure].

    OFIQ's CalculateExposure uses bitwise_and(faceMask, occlusionMask).
    """
    factor = 1.0 - severity * 0.85
    darkened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if ctx is not None and ctx.face_mask is not None:
        combined = cv2.bitwise_and(ctx.face_mask, ctx.occlusion_mask)
        mask3 = combined[:, :, np.newaxis] > 0
        return np.where(mask3, darkened, img)
    return darkened


def _brighten_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Brighten within the face mask region [S6.4 OverExposure].

    Pushes face pixels toward luminance 247-255 range.
    """
    factor = 1.0 + severity * 2.5
    brightened = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    if ctx is not None and ctx.face_mask is not None:
        mask3 = ctx.face_mask[:, :, np.newaxis] > 0
        return np.where(mask3, brightened, img)
    return brightened


def _reduce_luminance_variance_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Compress luminance variance within face region [S6.3].

    OFIQ measures luminance histogram variance within the face mask.
    """
    factor = 1.0 - severity * 0.9
    result = img.astype(np.float32)

    if ctx is not None and ctx.face_mask is not None:
        mask = ctx.face_mask > 0
        for c in range(3):
            face_pixels = result[:, :, c][mask]
            if len(face_pixels) == 0:
                continue
            ch_mean = face_pixels.mean()
            result[:, :, c][mask] = ch_mean + (face_pixels - ch_mean) * factor
    else:
        for c in range(3):
            ch_mean = result[:, :, c].mean()
            result[:, :, c] = ch_mean + (result[:, :, c] - ch_mean) * factor

    return np.clip(result, 0, 255).astype(np.uint8)


def _reduce_dynamic_range(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Compress dynamic range toward mid-gray [S6.5]. Keep as-is (EXCELLENT fidelity)."""
    mid = 128.0
    factor = 1.0 - severity * 0.9
    return np.clip(mid + (img.astype(np.float32) - mid) * factor, 0, 255).astype(np.uint8)


def _blur(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Gaussian blur [S6.6]. Keep as-is (EXCELLENT fidelity)."""
    sigma = severity * 10.0 + 0.5
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _motion_blur(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Motion blur [S6.6]. Keep as-is (EXCELLENT fidelity)."""
    ksize = max(int(severity * 30) + 1, 3)
    kernel = np.zeros((ksize, ksize))
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 180)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for i in range(ksize):
        x = int(ksize // 2 + (i - ksize // 2) * cos_a)
        y = int(ksize // 2 + (i - ksize // 2) * sin_a)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def _gaussian_noise(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Additive Gaussian noise [S6.6]. Keep as-is (EXCELLENT fidelity)."""
    rng = np.random.RandomState(seed)
    sigma = severity * 80
    noise = rng.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg_compression(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """JPEG re-encoding [S6.7]. Keep as-is (EXCELLENT fidelity)."""
    quality = max(int((1 - severity) * 95) + 5, 1)
    _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def _color_cast_cielab(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Push skin color outside natural CIELAB range in the ROI zones [S6.8].

    OFIQ measures CIELAB distance from ideal ranges a* in [5,25], b* in [5,35]
    in the left/right ROI zones. We shift chromaticity within those zones.
    """
    if ctx is None:
        return _color_cast_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    out = img.copy()
    h, w = img.shape[:2]

    for roi in [ctx.left_roi, ctx.right_roi]:
        rx, ry, rw, rh = roi
        x1, y1 = max(0, rx), max(0, ry)
        x2, y2 = min(w, rx + rw), min(h, ry + rh)
        if x2 <= x1 or y2 <= y1:
            continue

        region = out[y1:y2, x1:x2].astype(np.float32)

        # Convert to LAB using OpenCV (close enough for perturbation)
        lab = cv2.cvtColor(region.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)

        # Push a* and b* channels outside natural range
        # Direction: randomly push toward cold (blue) or warm (red/yellow)
        direction = rng.choice([-1, 1])
        shift_a = direction * severity * 60  # push a* far from [5,25]
        shift_b = direction * severity * 60  # push b* far from [5,35]

        lab[:, :, 1] = np.clip(lab[:, :, 1] + shift_a, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + shift_b, 0, 255)

        modified = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        out[y1:y2, x1:x2] = modified

    return out


def _color_cast_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: global RGB channel boost."""
    rng = np.random.RandomState(seed)
    channel = rng.randint(0, 3)
    out = img.astype(np.float32)
    out[:, :, channel] = np.clip(out[:, :, channel] + severity * 80, 0, 255)
    return out.astype(np.uint8)


def _radial_distortion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Barrel/pincushion distortion [S6.9]. Keep as-is."""
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    k = severity * 0.5
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = (x_idx - cx) / cx
    y_norm = (y_idx - cy) / cy
    r2 = x_norm ** 2 + y_norm ** 2
    scale = 1 + k * r2
    map_x = (cx + x_norm * scale * cx).astype(np.float32)
    map_y = (cy + y_norm * scale * cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# =========================================================================
# Section 7 -- Subject-Related Components
# =========================================================================

def _eyes_close_warp(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Warp upper eyelid landmarks toward lower to simulate eye closure [S7.2].

    OFIQ measures min(max_pair_dist(LEFT_EYE), max_pair_dist(RIGHT_EYE)) / t.
    Pairs: Left=(61,67),(62,66),(63,65); Right=(69,75),(70,74),(71,73).
    We move upper eyelid landmarks toward their lower counterparts.
    """
    if ctx is None:
        return _eye_occlusion_fallback(img, severity, seed)

    landmarks = ctx.landmarks_98.astype(np.float64)
    h, w = img.shape[:2]

    # Collect control points: all 98 landmarks as identity, then
    # override the upper eyelid landmarks with displaced versions
    src_points = landmarks.copy()
    dst_points = landmarks.copy()

    all_pairs = PAIRS_LEFT_EYE + PAIRS_RIGHT_EYE
    for upper_idx, lower_idx in all_pairs:
        upper = landmarks[upper_idx]
        lower = landmarks[lower_idx]
        gap = upper - lower
        # Move upper toward lower by severity fraction
        dst_points[upper_idx] = upper - gap * severity * 0.9

    return _apply_rbf_warp(img, src_points, dst_points, seed)


def _mouth_open_warp(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Warp inner lip landmarks apart to simulate mouth opening [S7.3].

    OFIQ measures max_pair_dist(MOUTH_INNER_pairs) / t.
    Pairs: (89,95),(90,94),(91,93).
    We move upper inner lip UP and lower inner lip DOWN.
    """
    if ctx is None:
        return _mouth_occlusion_fallback(img, severity, seed)

    landmarks = ctx.landmarks_98.astype(np.float64)
    t = ctx.t_metric

    src_points = landmarks.copy()
    dst_points = landmarks.copy()

    displacement = severity * t * 0.25

    # Upper inner lip landmarks: 88, 89, 90, 91
    for idx in [88, 89, 90, 91]:
        dst_points[idx, 1] -= displacement  # move up

    # Lower inner lip landmarks: 92, 93, 94, 95
    for idx in [92, 93, 94, 95]:
        dst_points[idx, 1] += displacement  # move down

    # Also move outer mouth landmarks proportionally (smaller displacement)
    outer_displacement = displacement * 0.3
    # Upper outer: 76, 77, 78 (right side), 82, 83, 84 (left side)
    for idx in [77, 78, 83, 84]:
        dst_points[idx, 1] -= outer_displacement
    # Lower outer: 79, 80, 81, 85, 86, 87
    for idx in [80, 81, 86, 87]:
        dst_points[idx, 1] += outer_displacement

    return _apply_rbf_warp(img, src_points, dst_points, seed)


def _eye_occlusion_evz(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Occlude within the Eye Visibility Zone [S7.4].

    OFIQ measures occlusion proportion within EVZ rectangles
    (eye bounding rect expanded by floor(IED/20) pixels).
    """
    if ctx is None:
        return _eye_occlusion_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    out = img.copy()
    h, w = img.shape[:2]

    for evz in [ctx.left_evz, ctx.right_evz]:
        ex, ey, ew, eh = evz
        # Clamp to image
        x1, y1 = max(0, ex), max(0, ey)
        x2, y2 = min(w, ex + ew), min(h, ey + eh)
        if x2 <= x1 or y2 <= y1:
            continue

        # Occlusion covers severity fraction of the EVZ
        occ_h = max(1, int((y2 - y1) * severity * 0.8))
        occ_y = y1 + (y2 - y1 - occ_h) // 2

        color = rng.randint(0, 60, 3).tolist()  # dark, like sunglasses
        out[occ_y:occ_y + occ_h, x1:x2] = color

    return out


def _eye_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: band-based eye occlusion."""
    h, w = img.shape[:2]
    band_h = int(h * 0.15 * (1 + severity))
    y_start = int(h * 0.25)
    out = img.copy()
    rng = np.random.RandomState(seed)
    color = rng.randint(0, 60, 3).tolist()
    out[y_start:y_start + band_h, :] = color
    return out


def _mouth_occlusion_polygon(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Occlude within the mouth landmark polygon [S7.5].

    OFIQ measures occlusion proportion within the convex polygon
    of mouth outer landmarks [76..87].
    """
    if ctx is None:
        return _mouth_occlusion_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    out = img.copy()
    h, w = img.shape[:2]

    mouth_pts = ctx.landmarks_98[MOUTH_OUTER].astype(np.int32)
    hull = cv2.convexHull(mouth_pts)

    # Create mask from the mouth polygon
    mouth_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mouth_mask, hull, 1)

    # Apply occlusion proportional to severity
    color = [rng.randint(180, 220)] * 3  # surgical mask color
    occ_mask = mouth_mask > 0

    # Scale occlusion area by severity (erode the mask for lower severity)
    if severity < 0.9:
        erode_size = max(1, int((1 - severity) * 10))
        erode_kernel = np.ones((erode_size, erode_size), np.uint8)
        eroded = cv2.erode(mouth_mask, erode_kernel)
        occ_mask = eroded > 0

    out[occ_mask] = color

    return out


def _mouth_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: band-based mouth occlusion."""
    h, w = img.shape[:2]
    band_h = int(h * 0.2 * (1 + severity))
    y_start = int(h * 0.55)
    out = img.copy()
    rng = np.random.RandomState(seed)
    color = [rng.randint(180, 220)] * 3
    out[y_start:y_start + band_h, :] = color
    return out


def _face_region_occlusion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Place rectangular occlusion within the face mask region [S7.6].

    OFIQ measures occlusion proportion over the full face landmark mask.
    """
    if ctx is None:
        return _rect_occlusion_fallback(img, severity, seed)

    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]

    # Find bounding box of the face mask
    face_coords = np.argwhere(ctx.face_mask > 0)
    if len(face_coords) == 0:
        return img

    y_min, x_min = face_coords.min(axis=0)
    y_max, x_max = face_coords.max(axis=0)
    face_h = y_max - y_min
    face_w = x_max - x_min

    # Occlusion size scales with severity
    frac = severity * 0.6
    oh, ow = max(2, int(face_h * frac)), max(2, int(face_w * frac))

    # Place within face region
    y = rng.randint(y_min, max(y_min + 1, y_max - oh + 1))
    x = rng.randint(x_min, max(x_min + 1, x_max - ow + 1))

    out = img.copy()
    color = rng.randint(0, 256, 3).tolist()

    # Only occlude pixels within the face mask
    region_mask = ctx.face_mask[y:y + oh, x:x + ow] > 0
    for c in range(3):
        channel = out[y:y + oh, x:x + ow, c]
        channel[region_mask] = color[c]

    return out


def _rect_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: random rectangular occlusion."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    frac = severity * 0.6
    oh, ow = int(h * frac), int(w * frac)
    if oh < 2 or ow < 2:
        return img
    y = rng.randint(0, max(h - oh, 1))
    x = rng.randint(0, max(w - ow, 1))
    out = img.copy()
    color = rng.randint(0, 256, 3).tolist()
    out[y:y + oh, x:x + ow] = color
    return out


def _reduce_ied(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shrink face in frame to reduce inter-eye distance [S7.7].

    OFIQ measures euclidean distance between eye centers (pixels) / cos(yaw).
    Downscale-and-upscale doesn't change this on aligned crops.
    Instead: shrink image and embed in padded canvas.
    """
    h, w = img.shape[:2]
    scale = max(0.3, 1.0 - severity * 0.7)

    new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center in canvas with border replication
    canvas = np.zeros_like(img)
    # Fill canvas with border-replicated content
    pad_top = (h - new_h) // 2
    pad_left = (w - new_w) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = small

    # Fill borders by replicating edge pixels
    if pad_top > 0:
        canvas[:pad_top, pad_left:pad_left + new_w] = small[0:1, :]
    if pad_top + new_h < h:
        canvas[pad_top + new_h:, pad_left:pad_left + new_w] = small[-1:, :]
    if pad_left > 0:
        canvas[:, :pad_left] = canvas[:, pad_left:pad_left + 1]
    if pad_left + new_w < w:
        canvas[:, pad_left + new_w:] = canvas[:, pad_left + new_w - 1:pad_left + new_w]

    return canvas


def _reduce_head_size(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shrink face in frame to reduce t/imageHeight ratio [S7.8].

    OFIQ measures |t/imageHeight - 0.45| (optimal ~0.45).
    Same pad-and-shrink mechanism as IED.
    """
    return _reduce_ied(img, severity, seed, ctx)


# =========================================================================
# Section 8 -- Geometric/Pose Components
# =========================================================================

def _yaw_rotation(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Simulated yaw via perspective warp [S8]. Kept with calibration note."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    direction = rng.choice([-1, 1])
    squeeze = severity * 0.5 * direction
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if squeeze > 0:
        dst = np.float32([[0, int(h * squeeze)], [w, 0], [w, h], [0, int(h * (1 - squeeze))]])
    else:
        dst = np.float32([[0, 0], [w, int(h * (-squeeze))], [w, int(h * (1 + squeeze))], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _pitch_tilt(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Simulated pitch via perspective warp [S8]. Kept with calibration note."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    direction = rng.choice([-1, 1])
    squeeze = severity * 0.4 * direction
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if squeeze > 0:
        dst = np.float32([[int(w * squeeze), 0], [int(w * (1 - squeeze)), 0], [w, h], [0, h]])
    else:
        dst = np.float32([[0, 0], [w, 0], [int(w * (1 + squeeze)), h], [int(w * (-squeeze)), h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _roll_rotation(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """In-plane rotation [S8]. Keep as-is (EXCELLENT fidelity)."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    angle = severity * 30 * rng.choice([-1, 1])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _crop_left(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image LEFT to push face toward left edge (degrades LeftwardCrop) [§7.4.10.1].

    OFIQ measures q_l = rightEyeCenter.x / IED. To DEGRADE this scalar we
    must DECREASE q_l, which means moving the face toward the LEFT edge of
    the image (small X_R). cv2.warpAffine with NEGATIVE shift_x translates
    the source image content LEFT.
    """
    h, w = img.shape[:2]
    shift_x = -int(severity * w * 0.4)  # negative = content moves left = face moves left
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _crop_right(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image RIGHT to push face toward right edge (degrades RightwardCrop) [§7.4.10.2].

    OFIQ measures q_r = (imgW - leftEyeCenter.x) / IED. To DEGRADE this scalar
    we must DECREASE q_r, which means moving the face toward the RIGHT edge
    of the image (X_L close to imgW). Positive shift_x translates content RIGHT.
    """
    h, w = img.shape[:2]
    shift_x = int(severity * w * 0.4)  # positive = content moves right = face moves right
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _margin_above(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image UP to push face toward top edge (degrades MarginAbove) [§7.4.10.3].

    OFIQ measures q_a = eyeMidPoint.y / t. To DEGRADE this scalar we must
    DECREASE q_a, which means moving the face toward the TOP of the image
    (small Y_C). Negative shift_y translates content UP.
    """
    h, w = img.shape[:2]
    shift_y = -int(severity * h * 0.4)  # negative = content moves up = face moves up
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def _margin_below(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image DOWN to push face toward bottom edge (degrades MarginBelow) [§7.4.10.4].

    OFIQ measures q_b = (imgH - eyeMidPoint.y) / t. To DEGRADE this scalar
    we must DECREASE q_b, which means moving the face toward the BOTTOM of
    the image (Y_C close to imgH). Positive shift_y translates content DOWN.
    """
    h, w = img.shape[:2]
    shift_y = int(severity * h * 0.4)  # positive = content moves down = face moves down
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# =========================================================================
# RBF warp helper (used by EyesOpen, MouthClosed landmark warps)
# =========================================================================

def _apply_rbf_warp(
    img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply smooth image warp using RBF interpolation.

    Warps img so that src_points move to dst_points, with smooth
    interpolation for surrounding pixels.
    """
    h, w = img.shape[:2]

    # Only use points that actually moved
    diffs = dst_points - src_points
    moved_mask = np.abs(diffs).sum(axis=1) > 0.5
    if not moved_mask.any():
        return img

    # Add corner anchors to keep the warp bounded
    corners = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2],
    ], dtype=np.float64)

    all_src = np.vstack([src_points, corners])
    all_dst = np.vstack([dst_points, corners])  # corners map to themselves

    # Compute displacement field using RBF
    dx = all_dst[:, 0] - all_src[:, 0]
    dy = all_dst[:, 1] - all_src[:, 1]

    rbf_x = RBFInterpolator(all_src, dx, kernel="thin_plate_spline", smoothing=1.0)
    rbf_y = RBFInterpolator(all_src, dy, kernel="thin_plate_spline", smoothing=1.0)

    # Build remap grid (subsample for performance, then upscale)
    step = 4
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    disp_x = rbf_x(grid_pts).reshape(grid_y.shape)
    disp_y = rbf_y(grid_pts).reshape(grid_y.shape)

    # Upscale displacement to full resolution
    map_x_small = (grid_x + disp_x).astype(np.float32)
    map_y_small = (grid_y + disp_y).astype(np.float32)

    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# =========================================================================
# Component Registry
# =========================================================================

COMPONENT_REGISTRY: dict[str, list[ComponentDegradation]] = {}


def _register(component: str, fn, desc: str, srange: str, needs_ctx: bool = False):
    if component not in COMPONENT_REGISTRY:
        COMPONENT_REGISTRY[component] = []
    COMPONENT_REGISTRY[component].append(
        ComponentDegradation(
            ofiq_component=component, function=fn,
            description=desc, severity_range=srange,
            requires_context=needs_ctx,
            standard_refs=STANDARDS_REFS.get(component),
        )
    )


# === Capture-related (OFIQ Report Section 6) ===

# S6.1 Background Uniformity
_register("BackgroundUniformity.scalar", _background_clutter_segmented,
          "Segmented background noise [S6.1]", "gradient: 0 -> 200", needs_ctx=True)

# S6.2 Illumination Uniformity
_register("IlluminationUniformity.scalar", _uneven_illumination_roi,
          "ROI illumination asymmetry [S6.2]", "factor: 1.0 -> 0.2", needs_ctx=True)

# S6.3 Moments of Luminance Distribution
_register("LuminanceMean.scalar", _darken_face,
          "Face-masked darkening [S6.3]", "factor: 1.0 -> 0.15", needs_ctx=True)
_register("LuminanceVariance.scalar", _reduce_luminance_variance_face,
          "Face-masked variance compression [S6.3]", "factor: 1.0 -> 0.1", needs_ctx=True)

# S6.4 Over- and Under-Exposure Prevention
_register("UnderExposurePrevention.scalar", _darken_face_with_occlusion,
          "Face+occlusion masked under-exposure [S6.4]", "factor: 1.0 -> 0.15", needs_ctx=True)
_register("OverExposurePrevention.scalar", _brighten_face,
          "Face-masked over-exposure [S6.4]", "factor: 1.0 -> 3.5", needs_ctx=True)

# S6.5 Dynamic Range
_register("DynamicRange.scalar", _reduce_dynamic_range,
          "Dynamic range compression [S6.5]", "range: 100% -> 10%")

# S6.6 Sharpness
_register("Sharpness.scalar", _blur,
          "Gaussian blur [S6.6]", "sigma: 0.5 -> 10.5")
_register("Sharpness.scalar", _motion_blur,
          "Motion blur [S6.6]", "kernel: 3 -> 31px")
_register("Sharpness.scalar", _gaussian_noise,
          "Additive Gaussian noise [S6.6]", "sigma: 0 -> 80")

# S6.7 No Compression Artefacts
_register("CompressionArtifacts.scalar", _jpeg_compression,
          "JPEG compression [S6.7]", "Q: 100 -> 5")

# S6.8 Natural Colour
_register("NaturalColour.scalar", _color_cast_cielab,
          "CIELAB color shift in ROI zones [S6.8]", "shift: 0 -> 60", needs_ctx=True)

# S6.9 Radial Distortion Prevention
_register("RadialDistortion.scalar", _radial_distortion,
          "Barrel distortion [S6.9]", "k: 0 -> 0.5")

# === Subject-related (OFIQ Report Section 7) ===

# S7.1 Single Face Present -- placeholder (generative, Phase 3)
# Registered below after generative imports

# S7.2 Eyes Open
_register("EyesOpen.scalar", _eyes_close_warp,
          "Landmark-warped eye closure [S7.2]", "closure: 0% -> 90%", needs_ctx=True)

# S7.3 Mouth Closed
_register("MouthClosed.scalar", _mouth_open_warp,
          "Landmark-warped mouth opening [S7.3]", "opening: 0 -> 0.25t", needs_ctx=True)

# S7.4 Eyes Visible
_register("EyesVisible.scalar", _eye_occlusion_evz,
          "EVZ-targeted eye occlusion [S7.4]", "coverage: 0% -> 80%", needs_ctx=True)

# S7.5 Mouth Occlusion Prevention
_register("MouthOcclusionPrevention.scalar", _mouth_occlusion_polygon,
          "Polygon-targeted mouth occlusion [S7.5]", "coverage: 0% -> 100%", needs_ctx=True)

# S7.6 Face Occlusion Prevention
_register("FaceOcclusionPrevention.scalar", _face_region_occlusion,
          "Face-masked rectangular occlusion [S7.6]", "area: 0% -> 60%", needs_ctx=True)

# S7.7 Inter-Eye Distance
_register("InterEyeDistance.scalar", _reduce_ied,
          "Pad-and-shrink to reduce IED [S7.7]", "scale: 1.0 -> 0.3")

# S7.8 Head Size
_register("HeadSize.scalar", _reduce_head_size,
          "Pad-and-shrink to reduce head size [S7.8]", "scale: 1.0 -> 0.3")

# === Geometric (OFIQ Report Sections 8+) ===

# Head Pose
_register("HeadPoseYaw.scalar", _yaw_rotation,
          "Perspective yaw rotation [S8]", "squeeze: 0% -> 50%")
_register("HeadPosePitch.scalar", _pitch_tilt,
          "Perspective pitch tilt [S8]", "squeeze: 0% -> 40%")
_register("HeadPoseRoll.scalar", _roll_rotation,
          "In-plane rotation [S8]", "angle: 0 -> +/-30 deg")

# Crop / margins (4 separate directional functions)
_register("LeftwardCropOfTheFaceImage.scalar", _crop_left,
          "Rightward shift [S8 CropLeft]", "shift: 0% -> 40%")
_register("RightwardCropOfTheFaceImage.scalar", _crop_right,
          "Leftward shift [S8 CropRight]", "shift: 0% -> 40%")
_register("MarginAboveOfTheFaceImage.scalar", _margin_above,
          "Downward shift [S8 MarginAbove]", "shift: 0% -> 40%")
_register("MarginBelowOfTheFaceImage.scalar", _margin_below,
          "Upward shift [S8 MarginBelow]", "shift: 0% -> 40%")

# === Generative components (Section 7.1, Section 8) ===

from ofiq_syngen.generative.single_face import insert_second_face
from ofiq_syngen.generative.expression import add_expression
from ofiq_syngen.generative.head_covering import add_head_covering

_register("SingleFacePresent.scalar", insert_second_face,
          "Face insertion via Poisson blending [S7.1]",
          "area ratio: 0 -> 0.4", needs_ctx=True)
_register("ExpressionNeutrality.scalar", add_expression,
          "Landmark-warped expression [S8]",
          "displacement: 0 -> 0.15t", needs_ctx=True)
_register("NoHeadCoverings.scalar", add_head_covering,
          "Synthetic hat overlay [S8]",
          "coverage: 0% -> 100%", needs_ctx=True)


def list_supported_components() -> list[str]:
    """Return all OFIQ components that have at least one degradation function."""
    return sorted(COMPONENT_REGISTRY.keys())


def list_all_degradations() -> list[tuple[str, str, str]]:
    """Return (component, description, severity_range) for all registered degradations."""
    result = []
    for comp, degs in sorted(COMPONENT_REGISTRY.items()):
        for d in degs:
            result.append((comp, d.description, d.severity_range))
    return result
