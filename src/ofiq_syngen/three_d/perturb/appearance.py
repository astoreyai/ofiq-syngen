"""Appearance-tier perturbations: photometric, mask-aware.

Components in this tier are illumination, exposure, color, and luminance
operations. They modify pixel values inside (or outside) the OFIQ-defined
face region. They do NOT need a FLAME mesh — only a face mask.

Honest framing: most photometric components have no rich 3D meaning. A
luminance shift, a JPEG-style color cast, or a dynamic-range compression
is a global per-pixel transform. We could lift+render+post-process, but the
result would be numerically identical to a direct mask-aware 2D operation.
3dsyn applies these directly to scene.image (the input) inside the face
mask, mirroring ofiq-syngen's approach by intent.

The one exception is IlluminationUniformity, which CAN be modeled as
asymmetric scene lighting in 3D. The current implementation uses the same
two-zone darkening as ofiq-syngen (§7.3.3); a future renderer-driven
version that places real point lights in scene.lighting and re-renders is
a natural follow-up.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ofiq_syngen.three_d.registry import register
from ofiq_syngen.three_d.scene.state import SceneState


def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = img.clip(0, 255).astype(np.uint8)
    return img


def _require_scene(scene: Optional[SceneState], component: str) -> SceneState:
    if scene is None:
        raise RuntimeError(
            f"{component} requires a SceneState (face mask). Pipeline must build a "
            "mask-only scene before dispatching appearance-tier perturbations."
        )
    return scene


def _eye_zone_rois(scene: SceneState) -> Optional[tuple[tuple[int, int, int, int],
                                                        tuple[int, int, int, int]]]:
    """Return (left_roi, right_roi) from ofiq-syngen FaceContext if present."""
    fa = scene.face_analysis
    if fa is None:
        return None
    left = getattr(fa, "left_roi", None)
    right = getattr(fa, "right_roi", None)
    if left is None or right is None:
        return None
    return left, right


# =========================================================================
# Background uniformity
# =========================================================================

def _background_uniformity(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Add structured noise to the segmented background [§7.3.2].

    OFIQ measures Sobel gradient magnitude on the BiSeNet-segmented
    background. We add gradient-creating noise outside the face mask.
    """
    scene = _require_scene(scene, "BackgroundUniformity.scalar")
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]

    bg_mask = (scene.face_mask == 0).astype(np.uint8)
    if bg_mask.sum() == 0:
        return img

    intensity = int(float(severity) * 200)
    if intensity < 1:
        return img

    noise = rng.randn(h, w).astype(np.float32) * intensity
    edges = cv2.Sobel(noise, cv2.CV_32F, 1, 1, ksize=3)
    out = img.astype(np.float32)
    for c in range(3):
        out[..., c] = np.where(bg_mask > 0, out[..., c] + edges, out[..., c])
    return _ensure_uint8(out)


# =========================================================================
# Illumination uniformity (asymmetric eye-zone darkening)
# =========================================================================

def _illumination_uniformity(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Asymmetric L/R illumination via eye-ROI darkening [§7.3.3].

    Severity 0 -> 1 maps to a darken factor 1.0 -> 0.2 applied to one of two
    eye-region ROIs picked randomly under the seed. Mirrors ofiq-syngen.
    """
    scene = _require_scene(scene, "IlluminationUniformity.scalar")
    rois = _eye_zone_rois(scene)
    if rois is None:
        return img

    rng = np.random.RandomState(seed)
    pick = int(rng.choice([0, 1]))
    x, y, w, h = rois[pick]
    factor = 1.0 - float(severity) * 0.8

    out = img.copy()
    region = out[y:y + h, x:x + w].astype(np.float32) * factor
    out[y:y + h, x:x + w] = region.clip(0, 255).astype(np.uint8)
    return out


# =========================================================================
# Luminance / exposure (mask-aware scaling)
# =========================================================================

def _apply_face_scale(
    img: np.ndarray,
    severity: float,
    component: str,
    scene: Optional[SceneState],
    factor_min: float,
    factor_max: float,
) -> np.ndarray:
    """Multiplicative tone shift inside the face mask. severity=0 -> factor_max,
    severity=1 -> factor_min (we scan the range linearly)."""
    scene = _require_scene(scene, component)
    factor = factor_max + (factor_min - factor_max) * float(severity)
    mask = (scene.face_mask > 0)[..., None]
    out = img.astype(np.float32)
    out = np.where(mask, out * factor, out)
    return _ensure_uint8(out)


def _luminance_mean(img, severity, seed, scene):
    """Face-masked darkening [§7.3.4.2]. factor 1.0 -> 0.15."""
    return _apply_face_scale(img, severity, "LuminanceMean.scalar", scene, 0.15, 1.0)


def _under_exposure_prevention(img, severity, seed, scene):
    """Face-masked under-exposure [§7.3.5]. factor 1.0 -> 0.15."""
    return _apply_face_scale(img, severity, "UnderExposurePrevention.scalar", scene, 0.15, 1.0)


def _over_exposure_prevention(img, severity, seed, scene):
    """Face-masked over-exposure [§7.3.6]. factor 1.0 -> 3.5."""
    return _apply_face_scale(img, severity, "OverExposurePrevention.scalar", scene, 3.5, 1.0)


def _luminance_variance(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Compress per-channel variance toward the face-region mean [§7.3.4.3].

    severity 0 = unchanged, severity 1 = compress by 90% (factor 0.1).
    """
    scene = _require_scene(scene, "LuminanceVariance.scalar")
    factor = 1.0 - float(severity) * 0.9
    mask = (scene.face_mask > 0)
    if not mask.any():
        return img

    out = img.astype(np.float32).copy()
    for c in range(3):
        ch = out[..., c]
        face_pixels = ch[mask]
        if face_pixels.size == 0:
            continue
        mean = float(face_pixels.mean())
        ch_compressed = mean + (ch - mean) * factor
        out[..., c] = np.where(mask, ch_compressed, ch)
    return _ensure_uint8(out)


def _dynamic_range(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Compress global pixel range toward mid-gray [§7.3.7].

    severity 0 = unchanged, severity 1 = compress to 10% range (factor 0.1).
    Applied globally; mask presence not required but kept consistent with
    appearance-tier signature.
    """
    factor = 1.0 - float(severity) * 0.9
    mid = 128.0
    out = (img.astype(np.float32) - mid) * factor + mid
    return _ensure_uint8(out)


# =========================================================================
# Natural colour (CIELAB shift in eye-region ROIs)
# =========================================================================

def _natural_colour(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Shift CIELAB a*/b* in eye-zone ROIs [§7.3.10].

    severity 0 -> shift 0; severity 1 -> shift +/- 60 LAB units. Direction
    randomized under the seed. Same mechanism as ofiq-syngen.
    """
    scene = _require_scene(scene, "NaturalColour.scalar")
    rois = _eye_zone_rois(scene)
    if rois is None:
        return img

    rng = np.random.RandomState(seed)
    shift_a = float(severity) * 60.0 * float(rng.choice([-1, 1]))
    shift_b = float(severity) * 60.0 * float(rng.choice([-1, 1]))

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    for x, y, w, h in rois:
        lab[y:y + h, x:x + w, 1] = (lab[y:y + h, x:x + w, 1] + shift_a).clip(0, 255)
        lab[y:y + h, x:x + w, 2] = (lab[y:y + h, x:x + w, 2] + shift_b).clip(0, 255)
    out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return _ensure_uint8(out)


# =========================================================================
# Registration
# =========================================================================

register("BackgroundUniformity.scalar", _background_uniformity,
         "APP::structured noise outside face mask [§7.3.2]", "noise: 0 -> 200",
         tier="appearance")
register("IlluminationUniformity.scalar", _illumination_uniformity,
         "APP::asymmetric eye-ROI darkening [§7.3.3]", "factor: 1.0 -> 0.2",
         tier="appearance")
register("LuminanceMean.scalar", _luminance_mean,
         "APP::face-masked darkening [§7.3.4.2]", "factor: 1.0 -> 0.15",
         tier="appearance")
register("LuminanceVariance.scalar", _luminance_variance,
         "APP::face-masked variance compression [§7.3.4.3]", "factor: 1.0 -> 0.1",
         tier="appearance")
register("UnderExposurePrevention.scalar", _under_exposure_prevention,
         "APP::face-occlusion darkening [§7.3.5]", "factor: 1.0 -> 0.15",
         tier="appearance")
register("OverExposurePrevention.scalar", _over_exposure_prevention,
         "APP::face-masked brightening [§7.3.6]", "factor: 1.0 -> 3.5",
         tier="appearance")
register("DynamicRange.scalar", _dynamic_range,
         "APP::global range compression toward mid-gray [§7.3.7]", "range: 100% -> 10%",
         tier="appearance")
register("NaturalColour.scalar", _natural_colour,
         "APP::CIELAB shift in eye-zone ROIs [§7.3.10]", "shift: 0 -> 60",
         tier="appearance")
