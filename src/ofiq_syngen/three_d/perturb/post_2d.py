"""Post-render 2D tier: sensor / codec / lens artifacts.

These perturbations have no meaningful 3D representation. JPEG quantization,
Gaussian blur, motion blur, and barrel distortion are properties of the
camera + sensor + codec stack, applied AFTER the 3D scene is rendered to a
2D image.

Implementing them as needs_scene=False keeps the path light: the pipeline
skips the lift + render step entirely and applies the operator directly to
the input image. Numerically identical to ofiq-syngen's 2D operators by
intent.
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


# =========================================================================
# Sharpness — Gaussian blur, motion blur, additive noise
# =========================================================================

def _gaussian_blur(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Gaussian blur [§7.3.8]. sigma 0.5 -> 10.5 across severity."""
    sigma = 0.5 + float(severity) * 10.0
    ksize = int(2 * round(3.0 * sigma) + 1)
    return _ensure_uint8(cv2.GaussianBlur(img, (ksize, ksize), sigma))


def _motion_blur(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Linear motion blur [§7.3.8]. kernel 3 -> 31 px, random direction under seed."""
    rng = np.random.RandomState(seed)
    ksize = int(3 + round(float(severity) * 28.0))
    if ksize < 3:
        return img
    if ksize % 2 == 0:
        ksize += 1
    angle_deg = float(rng.uniform(0.0, 180.0))

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / float(ksize)
    M = cv2.getRotationMatrix2D((ksize / 2.0, ksize / 2.0), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    kernel /= kernel.sum() if kernel.sum() != 0 else 1.0
    return _ensure_uint8(cv2.filter2D(img, -1, kernel))


def _additive_gaussian_noise(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Additive Gaussian noise [§7.3.8]. sigma 0 -> 80."""
    rng = np.random.RandomState(seed)
    sigma = float(severity) * 80.0
    if sigma <= 0:
        return img
    noise = rng.randn(*img.shape).astype(np.float32) * sigma
    out = img.astype(np.float32) + noise
    return _ensure_uint8(out)


# =========================================================================
# Compression — JPEG re-encode
# =========================================================================

def _jpeg_compression(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """JPEG quantization artifacts [§7.3.9]. quality 100 -> 5."""
    quality = int(round(100 - float(severity) * 95))
    quality = max(1, min(100, quality))
    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return img
    return _ensure_uint8(decoded)


# =========================================================================
# Lens distortion — radial (barrel)
# =========================================================================

def _radial_distortion(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Barrel distortion [Annex D.2.1]. k coefficient 0 -> 0.5.

    Applies the inverse of the canonical radial-distortion mapping so the
    output looks barrel-distorted relative to the input. Keeps input dims.
    """
    h, w = img.shape[:2]
    k = float(severity) * 0.5
    if k <= 0:
        return img

    fx = fy = max(h, w)
    cx, cy = w / 2.0, h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k, 0.0, 0.0, 0.0], dtype=np.float64)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0.0, newImgSize=(w, h))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, new_K, (w, h), cv2.CV_32FC1)
    distorted = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return _ensure_uint8(distorted)


# =========================================================================
# Registration
# =========================================================================

register("Sharpness.scalar", _gaussian_blur,
         "POST::Gaussian blur [§7.3.8]", "sigma: 0.5 -> 10.5",
         tier="post_2d")
register("Sharpness.scalar", _motion_blur,
         "POST::Motion blur [§7.3.8]", "kernel: 3 -> 31 px",
         tier="post_2d")
register("Sharpness.scalar", _additive_gaussian_noise,
         "POST::Additive Gaussian noise [§7.3.8]", "sigma: 0 -> 80",
         tier="post_2d")

register("CompressionArtifacts.scalar", _jpeg_compression,
         "POST::JPEG compression [§7.3.9]", "Q: 100 -> 5",
         tier="post_2d")

register("RadialDistortion.scalar", _radial_distortion,
         "POST::barrel distortion [Annex D.2.1]", "k: 0 -> 0.5",
         tier="post_2d")
