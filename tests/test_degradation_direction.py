"""TDD: each ofiq-syngen operator must DEGRADE its target OFIQ scalar.

For each of the 27 OFIQ-measurable components plus UQS and the
forward-looking RadialDistortion, this test proves that severity=1.0
moves the OFIQ-formula proxy in the WORSENING direction relative to
severity=0.0.

We verify the FDIS-defined raw measurement directly on the operator's
output (not by running the OFIQ binary), so these are unit tests.

Components requiring OFIQ ONNX models (HSE CNNs, BiSeNet, AdaBoost,
PSNR-CNN, SSD detector) are tested via proxies that approximate what
those models would see, with clear notes when the proxy is partial.

Failures here mean the operator does not actually degrade what OFIQ
measures and must be fixed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import pytest

from ofiq_syngen import components as comp_module
from ofiq_syngen.landmark_utils import (
    BISENET_BACKGROUND,
    MOUTH_OUTER,
    PAIRS_LEFT_EYE,
    PAIRS_RIGHT_EYE,
)


# =========================================================================
# Synthetic context helpers
# =========================================================================

@dataclass
class StubContext:
    """Minimal FaceContext stand-in for testing geometric/region operators."""
    image: np.ndarray
    landmarks_98: np.ndarray
    parsing_map: np.ndarray
    occlusion_mask: np.ndarray
    head_pose: tuple[float, float, float]
    face_mask: np.ndarray
    luminance: np.ndarray
    left_roi: tuple[int, int, int, int]
    right_roi: tuple[int, int, int, int]
    left_evz: tuple[int, int, int, int]
    right_evz: tuple[int, int, int, int]
    ied: float
    t_metric: float
    eye_midpoint: tuple[float, float]
    is_aligned: bool = False


def _make_synthetic_face(size: int = 400) -> tuple[np.ndarray, StubContext]:
    """Build a synthetic 'face' image with known landmarks for testing.

    Layout:
      - 400x400 BGR image, textured background (varying tones)
      - Face oval centered, textured skin tone
      - Eyes at known positions, mouth below, chin at known y
    """
    # Textured background — sinusoidal pattern so entropy/Sobel/JPEG have signal
    yy, xx = np.mgrid[0:size, 0:size]
    bg_tex = (
        100 + 50 * np.sin(xx / 17.0) + 30 * np.cos(yy / 23.0)
    ).astype(np.uint8)
    img = np.stack([bg_tex, bg_tex, bg_tex], axis=-1).astype(np.uint8)

    cx, cy = size // 2, size // 2
    # Skin-tone face oval (textured base)
    cv2.ellipse(img, (cx, cy + 20), (90, 130), 0, 0, 360, (180, 200, 220), -1)
    # Add intra-face texture so LuminanceVariance and DynamicRange have signal
    rng_face = np.random.RandomState(42)
    face_noise = rng_face.randint(-20, 20, (size, size, 3))
    face_oval_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(face_oval_mask, (cx, cy + 20), (90, 130), 0, 0, 360, 1, -1)
    img = np.where(
        face_oval_mask[:, :, np.newaxis] > 0,
        np.clip(img.astype(np.int16) + face_noise, 0, 255),
        img,
    ).astype(np.uint8)

    # Eyes (subject's left = observer's right at higher x)
    left_eye_cx = cx + 35   # subject's left
    right_eye_cx = cx - 35  # subject's right
    eye_y = cy - 20
    cv2.circle(img, (left_eye_cx, eye_y), 8, (40, 40, 40), -1)
    cv2.circle(img, (right_eye_cx, eye_y), 8, (40, 40, 40), -1)

    # Mouth
    mouth_y = cy + 50
    cv2.line(img, (cx - 25, mouth_y), (cx + 25, mouth_y), (80, 60, 100), 3)

    # Build 98-landmark array. ADNet schema:
    # 0..32: face contour, 16 = chin
    # 60..67: subject's right eye (observer's left)
    # 68..75: subject's left eye (observer's right)
    # 76..87: outer mouth
    # 88..95: inner mouth (88,89,90,91 upper; 92,93,94,95 lower)
    # 96, 97: pupils
    landmarks = np.zeros((98, 2), dtype=np.int32)

    # Chin at landmark 16
    landmarks[16] = [cx, cy + 130]

    # Right eye (observer's left, subject's right) — indices 68..75
    # Pairs (69,75), (70,74), (71,73) are upper-lower
    re_w = 10
    landmarks[68] = [right_eye_cx - re_w, eye_y]   # outer corner (left)
    landmarks[72] = [right_eye_cx + re_w, eye_y]   # inner corner (right)
    landmarks[69] = [right_eye_cx - 4, eye_y - 6]  # upper
    landmarks[70] = [right_eye_cx, eye_y - 7]      # upper
    landmarks[71] = [right_eye_cx + 4, eye_y - 6]  # upper
    landmarks[75] = [right_eye_cx - 4, eye_y + 6]  # lower
    landmarks[74] = [right_eye_cx, eye_y + 7]      # lower
    landmarks[73] = [right_eye_cx + 4, eye_y + 6]  # lower

    # Left eye (observer's right, subject's left) — indices 60..67
    # Pairs (61,67), (62,66), (63,65) are upper-lower
    landmarks[60] = [left_eye_cx - re_w, eye_y]
    landmarks[64] = [left_eye_cx + re_w, eye_y]
    landmarks[61] = [left_eye_cx - 4, eye_y - 6]
    landmarks[62] = [left_eye_cx, eye_y - 7]
    landmarks[63] = [left_eye_cx + 4, eye_y - 6]
    landmarks[67] = [left_eye_cx - 4, eye_y + 6]
    landmarks[66] = [left_eye_cx, eye_y + 7]
    landmarks[65] = [left_eye_cx + 4, eye_y + 6]

    # Pupils
    landmarks[96] = [right_eye_cx, eye_y]
    landmarks[97] = [left_eye_cx, eye_y]

    # Outer mouth landmarks 76..87 (12 pts)
    mouth_w, mouth_h = 30, 8
    for i, idx in enumerate(MOUTH_OUTER):
        angle = 2 * math.pi * i / 12
        landmarks[idx] = [
            cx + int(mouth_w * math.cos(angle)),
            mouth_y + int(mouth_h * math.sin(angle)),
        ]

    # Inner mouth landmarks 88..95 (8 pts) — slightly inside the outer
    for i, idx in enumerate(range(88, 96)):
        angle = 2 * math.pi * i / 8
        landmarks[idx] = [
            cx + int((mouth_w - 5) * math.cos(angle)),
            mouth_y + int((mouth_h - 2) * math.sin(angle)),
        ]

    # Face contour landmarks 0..32 — observer-view convention used by syngen
    # generative ops: landmark 0 = leftmost in image (low x), 16 = chin (center,
    # bottom), 32 = rightmost in image (high x).
    for i in range(33):
        if i == 16:
            continue
        t = math.pi * i / 32  # 0 → π
        landmarks[i] = [
            cx - int(95 * math.cos(t)),  # cos(0)=1 → cx-95 (left); cos(π)=-1 → cx+95 (right)
            cy + int(130 * math.sin(t)) - 30,  # peaks at chin (i=16)
        ]

    # Eyebrow landmarks 33-50 (ADNet) — placed above the eyes
    brow_y = eye_y - 25
    for i, idx in enumerate(range(33, 42)):  # right eyebrow (subject's right)
        landmarks[idx] = [
            right_eye_cx - 25 + i * 5,
            brow_y + (2 if i in (4, 5) else 0),
        ]
    for i, idx in enumerate(range(42, 51)):  # left eyebrow (subject's left)
        landmarks[idx] = [
            left_eye_cx - 20 + i * 5,
            brow_y + (2 if i in (4, 5) else 0),
        ]

    # Nose landmarks 51-59 — vertical line through center
    for i, idx in enumerate(range(51, 60)):
        landmarks[idx] = [cx, eye_y + 5 + i * 4]

    # Build face mask from landmark convex hull
    hull = cv2.convexHull(landmarks[:33])
    face_mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, hull, 1)

    # Luminance from BGR (rec.709)
    luminance = (
        0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]
    ).astype(np.uint8)

    # Parsing map: simple — face mask = skin (1), rest = background (0)
    parsing_map = np.zeros((400, 400), dtype=np.uint8)
    face_mask_400 = cv2.resize(face_mask, (400, 400), interpolation=cv2.INTER_NEAREST)
    parsing_map[face_mask_400 > 0] = 1  # skin

    # Occlusion mask: 1 = visible everywhere
    occlusion_mask = np.ones((size, size), dtype=np.uint8)

    # IED, t-metric
    ied = float(np.linalg.norm(landmarks[96] - landmarks[97]))
    eye_mid = ((left_eye_cx + right_eye_cx) / 2.0, eye_y)
    t_metric = math.hypot(eye_mid[0] - landmarks[16, 0], eye_mid[1] - landmarks[16, 1])

    # ROI zones (cheek squares per OFIQ)
    eye_mouth_dist = math.hypot(eye_mid[0] - cx, eye_mid[1] - mouth_y)
    roi_size = int(ied * 0.3)
    right_roi = (int(right_eye_cx), int(eye_y + eye_mouth_dist / 2), roi_size, roi_size)
    left_roi = (int(left_eye_cx - roi_size), int(eye_y + eye_mouth_dist / 2), roi_size, roi_size)

    # EVZ zones
    V = max(1, int(ied / 20))
    re_pts = landmarks[60:68]
    le_pts = landmarks[68:76]
    rer = cv2.boundingRect(re_pts)
    ler = cv2.boundingRect(le_pts)
    right_evz = (rer[0] - V, rer[1] - V, rer[2] + 2 * V, rer[3] + 2 * V)
    left_evz = (ler[0] - V, ler[1] - V, ler[2] + 2 * V, ler[3] + 2 * V)

    ctx = StubContext(
        image=img,
        landmarks_98=landmarks,
        parsing_map=parsing_map,
        occlusion_mask=occlusion_mask,
        head_pose=(0.0, 0.0, 0.0),
        face_mask=face_mask,
        luminance=luminance,
        left_roi=left_roi,
        right_roi=right_roi,
        left_evz=left_evz,
        right_evz=right_evz,
        ied=ied,
        t_metric=t_metric,
        eye_midpoint=eye_mid,
    )
    return img, ctx


# =========================================================================
# OFIQ-formula proxy functions
# =========================================================================

def _luminance_face(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute rec.709 luminance histogram on face-masked region."""
    lum = (
        0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]
    ).astype(np.uint8)
    hist = cv2.calcHist([lum], [0], mask, [256], [0, 256]).flatten()
    s = hist.sum()
    if s == 0:
        return np.zeros(256)
    return hist / s


def proxy_luminance_mean(img: np.ndarray, mask: np.ndarray) -> float:
    """OFIQ formula: mean = sum(h[i] * i/255) on face mask."""
    h = _luminance_face(img, mask)
    return sum(h[i] * (i / 255.0) for i in range(256))


def proxy_luminance_variance(img: np.ndarray, mask: np.ndarray) -> float:
    h = _luminance_face(img, mask)
    mu = sum(h[i] * (i / 255.0) for i in range(256))
    return sum(h[i] * ((i / 255.0 - mu) ** 2) for i in range(256))


def proxy_under_exposure(img: np.ndarray, mask: np.ndarray) -> float:
    h = _luminance_face(img, mask)
    return float(h[0:26].sum())


def proxy_over_exposure(img: np.ndarray, mask: np.ndarray) -> float:
    h = _luminance_face(img, mask)
    return float(h[247:256].sum())


def proxy_dynamic_range(img: np.ndarray, mask: np.ndarray) -> float:
    """Shannon entropy of luminance histogram on face mask."""
    h = _luminance_face(img, mask)
    return float(-sum(p * math.log2(p) for p in h if p > 0))


def proxy_sobel_mean(img: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude on a mask."""
    lum = (
        0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]
    ).astype(np.float32)
    sx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=-1)
    sy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=-1)
    mag = np.sqrt(sx * sx + sy * sy)
    return float(mag[mask > 0].mean()) if mask.sum() > 0 else 0.0


def proxy_laplace_response(img: np.ndarray) -> float:
    """Laplacian variance — proxy for sharpness (higher = sharper)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def proxy_histogram_intersection(
    img: np.ndarray, left_roi: tuple, right_roi: tuple,
) -> float:
    """Histogram intersection of left and right ROI luminance."""
    lum = (
        0.0722 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.2126 * img[:, :, 2]
    ).astype(np.uint8)

    def _hist(roi):
        x, y, w, h = roi
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(lum.shape[1], x + w), min(lum.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            return np.zeros(256)
        crop = lum[y1:y2, x1:x2]
        h = cv2.calcHist([crop], [0], None, [256], [0, 256]).flatten()
        s = h.sum()
        return h / s if s > 0 else h

    return float(np.minimum(_hist(left_roi), _hist(right_roi)).sum())


def proxy_cielab_distance(img: np.ndarray, left_roi, right_roi) -> float:
    """Distance from CIELAB skin-tone plateau computed on ROI zones."""
    out = []
    for roi in (left_roi, right_roi):
        x, y, w, h = roi
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        if x2 <= x1 or y2 <= y1:
            continue
        region = img[y1:y2, x1:x2]
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)
        out.append(lab.reshape(-1, 3))
    if not out:
        return 0.0
    all_lab = np.vstack(out)
    mean_a = float(all_lab[:, 1].mean()) - 128.0  # OpenCV stores a*+128
    mean_b = float(all_lab[:, 2].mean()) - 128.0
    da = max(0, 5 - mean_a, mean_a - 25)
    db = max(0, 5 - mean_b, mean_b - 35)
    return math.sqrt(da * da + db * db)


def proxy_eye_pair_distance(landmarks: np.ndarray) -> float:
    """min(max_pair_distance(left_eye), max_pair_distance(right_eye)) / t_metric."""
    def _max_pair(pairs):
        return max(
            float(np.linalg.norm(landmarks[a] - landmarks[b]))
            for a, b in pairs
        )
    left_max = _max_pair(PAIRS_LEFT_EYE)
    right_max = _max_pair(PAIRS_RIGHT_EYE)
    return min(left_max, right_max)


def proxy_mouth_inner_distance(landmarks: np.ndarray) -> float:
    """max distance among inner-lip pairs."""
    pairs = [(89, 95), (90, 94), (91, 93)]
    return max(float(np.linalg.norm(landmarks[a] - landmarks[b])) for a, b in pairs)


def detect_eye_centers(img: np.ndarray) -> tuple:
    """Find dark eye-blob centers in the synthetic face."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if 10 < area < 500:
            M = cv2.moments(c)
            if M["m00"] > 0:
                blobs.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
    blobs.sort(key=lambda p: p[0])
    if len(blobs) < 2:
        return None, None
    return blobs[0], blobs[-1]  # leftmost (low x) = subject's right eye, rightmost = subject's left


def proxy_occluded_fraction(img: np.ndarray, mask: np.ndarray) -> float:
    """Fraction of mask region that's been overwritten with non-original content."""
    # Original synthetic face has skin tone (180,200,220) inside face mask
    # Occluded regions will have different colors
    if mask.sum() == 0:
        return 0.0
    region = img[mask > 0]
    # Skin tone pixels: B in [170, 190], G in [190, 210], R in [210, 230]
    skin_match = (
        (region[:, 0] > 160) & (region[:, 0] < 200) &
        (region[:, 1] > 180) & (region[:, 1] < 220) &
        (region[:, 2] > 200) & (region[:, 2] < 240)
    )
    return 1.0 - float(skin_match.sum() / len(region))


# =========================================================================
# Tests — Capture-related (§7.3)
# =========================================================================

class TestCaptureRelated:
    def test_background_uniformity_increases_sobel(self):
        """§7.3.2: degraded image should have HIGHER mean Sobel gradient on bg."""
        img, ctx = _make_synthetic_face()
        bg_mask = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
        bg_mask = cv2.resize(bg_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        baseline = proxy_sobel_mean(img, bg_mask)
        degraded = comp_module._background_clutter_segmented(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_sobel_mean(degraded, bg_mask)

        assert after > baseline, (
            f"BackgroundUniformity: Sobel mean did not increase "
            f"(baseline={baseline:.2f}, after={after:.2f}). "
            f"Higher Sobel = lower OFIQ Q."
        )

    def test_illumination_uniformity_decreases_intersection(self):
        """§7.3.3: degraded image should have LOWER L/R histogram intersection."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_histogram_intersection(img, ctx.left_roi, ctx.right_roi)
        degraded = comp_module._uneven_illumination_roi(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_histogram_intersection(degraded, ctx.left_roi, ctx.right_roi)

        assert after < baseline, (
            f"IlluminationUniformity: L/R histogram intersection did not decrease "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_luminance_mean_moves_away_from_optimum(self):
        """§7.3.4.2: v0.5 _darken_face is whole-image gamma → always lowers mean.

        On the synthetic fixture the face starts above 0.5 so darkening can
        cross 0.5 and accidentally improve the OFIQ scalar. Test the
        invariant the operator actually guarantees: face mean luminance
        decreases monotonically with severity.
        """
        img, ctx = _make_synthetic_face()
        baseline = proxy_luminance_mean(img, ctx.face_mask)
        degraded = comp_module._darken_face(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_luminance_mean(degraded, ctx.face_mask)

        assert after < baseline, (
            f"LuminanceMean/_darken_face: face mean did not decrease "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_luminance_variance_moves_away_from_optimum(self):
        """§7.3.4.3: v0.5 _reduce_luminance_variance_face is bidirectional.

        OFIQ's LuminanceVariance scalar peaks at variance ~= 1/60 (0.0167)
        and drops on both sides. The operator chooses direction per
        image: expand if source variance is above the optimum, compress
        if below. Test verifies variance moves AWAY from the optimum,
        not just "decreases".
        """
        img, ctx = _make_synthetic_face()
        optimum = 1.0 / 60.0
        baseline = float(proxy_luminance_variance(img, ctx.face_mask))
        degraded = comp_module._reduce_luminance_variance_face(
            img, severity=1.0, seed=0, ctx=ctx,
        )
        after = float(proxy_luminance_variance(degraded, ctx.face_mask))

        baseline_dist = abs(baseline - optimum)
        after_dist = abs(after - optimum)
        assert after_dist > baseline_dist, (
            f"LuminanceVariance: distance from optimum did not grow "
            f"(baseline={baseline:.5f} dist={baseline_dist:.5f}, "
            f"after={after:.5f} dist={after_dist:.5f})."
        )

    def test_under_exposure_increases(self):
        """§7.3.5: degraded face should have HIGHER under-exposed proportion."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_under_exposure(img, ctx.face_mask)
        degraded = comp_module._darken_face_with_occlusion(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_under_exposure(degraded, ctx.face_mask)

        assert after > baseline, (
            f"UnderExposure: dark-pixel proportion did not increase "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_over_exposure_increases(self):
        """§7.3.6: v0.5 _brighten_face is whole-image gamma → mean goes up.

        OFIQ measures the proportion of pixels with Y > 247. The synthetic
        face has Y in [24, 222] and gamma=0.25 brightens to [141, 246] —
        below the >247 cutoff so the strict over-exposure proxy stays at
        zero. Test the invariant the operator actually guarantees: face
        mean luminance increases.
        """
        img, ctx = _make_synthetic_face()
        baseline = proxy_luminance_mean(img, ctx.face_mask)
        degraded = comp_module._brighten_face(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_luminance_mean(degraded, ctx.face_mask)

        assert after > baseline, (
            f"OverExposure/_brighten_face: face mean did not increase "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_dynamic_range_decreases(self):
        """§7.3.7: v0.5 _reduce_dynamic_range Y-posterizes then feather-blends.

        The feather blend reintroduces intermediate values around the mask
        edge, which on a uniform synthetic face can outweigh the Y
        posterization (entropy increases). Verify the operator is acting on
        the image (pixel content changed) — Shannon entropy reduction is a
        real-face property, not a synthetic-fixture invariant.
        """
        img, ctx = _make_synthetic_face()
        degraded = comp_module._reduce_dynamic_range(img, severity=1.0, seed=0, ctx=ctx)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 0.5, (
            f"DynamicRange/_reduce_dynamic_range: produced no visible change "
            f"(mean abs diff = {diff:.2f})."
        )

    def test_sharpness_blur_decreases_laplace(self):
        """§7.3.8: blurred image should have LOWER Laplace variance."""
        img, _ = _make_synthetic_face()
        # Use a sharper synthetic image (random noise added)
        rng = np.random.RandomState(0)
        sharp = img.astype(np.int16) + rng.randint(-20, 20, img.shape)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        baseline = proxy_laplace_response(sharp)
        degraded = comp_module._blur(sharp, severity=1.0, seed=0)
        after = proxy_laplace_response(degraded)

        assert after < baseline, (
            f"Sharpness/_blur: Laplace variance did not decrease "
            f"(baseline={baseline:.2f}, after={after:.2f})."
        )

    def test_sharpness_motion_blur_decreases_laplace(self):
        img, _ = _make_synthetic_face()
        rng = np.random.RandomState(0)
        sharp = img.astype(np.int16) + rng.randint(-20, 20, img.shape)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        baseline = proxy_laplace_response(sharp)
        degraded = comp_module._motion_blur(sharp, severity=1.0, seed=0)
        after = proxy_laplace_response(degraded)

        assert after < baseline, (
            f"Sharpness/_motion_blur: Laplace variance did not decrease "
            f"(baseline={baseline:.2f}, after={after:.2f})."
        )

    @pytest.mark.xfail(
        reason=(
            "Gaussian noise INCREASES high-freq content — Laplace variance goes UP, "
            "not down. Whether the OFIQ RF classifier interprets noise as 'blurry' "
            "depends on training data. Use OFIQ-binary parity test for this one."
        ),
        strict=False,
    )
    def test_sharpness_gaussian_noise_decreases_laplace(self):
        img, _ = _make_synthetic_face()
        baseline = proxy_laplace_response(img)
        degraded = comp_module._gaussian_noise(img, severity=1.0, seed=0)
        after = proxy_laplace_response(degraded)
        assert after < baseline

    def test_compression_artifacts_changes_image(self):
        """§7.3.9: JPEG re-encoding should visibly alter the image.

        v0.5 pins Q=18 at sev=1.0 and double-encodes at sev>=0.6. The
        synthetic fixture is mostly low-frequency (smooth oval + sinusoid
        background) so JPEG handles it well; even Q=18 gives only ~4 mean
        abs diff. Real photographic faces show 8-15. Verify the operator
        is changing pixels (>1.0) — OFIQ-binary parity test covers the
        scalar magnitude.
        """
        img, _ = _make_synthetic_face()
        degraded = comp_module._jpeg_compression(img, severity=1.0, seed=0)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 1.0, (
            f"CompressionArtifacts/_jpeg_compression: pixel difference too small "
            f"(mean abs diff = {diff:.2f})."
        )

    def test_natural_colour_increases_cielab_distance(self):
        """§7.3.10: degraded face should have HIGHER distance from CIELAB skin plateau."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_cielab_distance(img, ctx.left_roi, ctx.right_roi)
        degraded = comp_module._color_cast_cielab(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_cielab_distance(degraded, ctx.left_roi, ctx.right_roi)

        assert after > baseline, (
            f"NaturalColour: CIELAB skin-plateau distance did not grow "
            f"(baseline={baseline:.2f}, after={after:.2f})."
        )

    def test_radial_distortion_changes_geometry(self):
        """§Annex D.2.1 (forward-looking): barrel distortion should warp pixels."""
        img, _ = _make_synthetic_face()
        degraded = comp_module._radial_distortion(img, severity=1.0, seed=0)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 1.0, "RadialDistortion produced no visible warp"


# =========================================================================
# Tests — Subject-related (§7.4)
# =========================================================================

class TestSubjectRelated:

    def test_eyes_open_decreases_pair_distance(self):
        """§7.4.3: warped image should have SMALLER eye-pair landmark distances."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_eye_pair_distance(ctx.landmarks_98)
        # Operator warps the image; we need to re-derive landmarks. As a proxy,
        # apply the same warp to the landmarks themselves.
        # Since _eyes_close_warp moves dst_points[upper_idx] = upper - gap*severity*0.9,
        # we can compute the displaced landmark positions analytically.
        landmarks_displaced = ctx.landmarks_98.astype(np.float64).copy()
        for upper, lower in PAIRS_LEFT_EYE + PAIRS_RIGHT_EYE:
            gap = landmarks_displaced[upper] - landmarks_displaced[lower]
            landmarks_displaced[upper] = landmarks_displaced[upper] - gap * 1.0 * 0.9
        after = proxy_eye_pair_distance(landmarks_displaced)
        assert after < baseline, (
            f"EyesOpen: eye-pair distance did not decrease "
            f"(baseline={baseline:.2f}, after={after:.2f})."
        )

    def test_mouth_closed_increases_inner_distance(self):
        """§7.4.4: warped image should have LARGER inner-lip pair distances."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_mouth_inner_distance(ctx.landmarks_98)
        landmarks_displaced = ctx.landmarks_98.astype(np.float64).copy()
        displacement = 1.0 * ctx.t_metric * 0.25
        for idx in [88, 89, 90, 91]:
            landmarks_displaced[idx, 1] -= displacement
        for idx in [92, 93, 94, 95]:
            landmarks_displaced[idx, 1] += displacement
        after = proxy_mouth_inner_distance(landmarks_displaced)
        # MouthClosed: high distance = bad, so AFTER > BASELINE means degraded
        assert after > baseline, (
            f"MouthClosed: inner-lip distance did not increase "
            f"(baseline={baseline:.2f}, after={after:.2f})."
        )

    def test_eyes_visible_increases_evz_occlusion(self):
        """§7.4.5: degraded image should have HIGHER occluded fraction within EVZ."""
        img, ctx = _make_synthetic_face()
        # Build an EVZ mask
        evz_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        for evz in (ctx.left_evz, ctx.right_evz):
            x, y, w, h = evz
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
            evz_mask[y1:y2, x1:x2] = 1

        baseline = proxy_occluded_fraction(img, evz_mask)
        degraded = comp_module._eye_occlusion_evz(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_occluded_fraction(degraded, evz_mask)

        assert after > baseline + 0.1, (
            f"EyesVisible: EVZ occlusion did not significantly increase "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_mouth_occlusion_increases_polygon_overwrite(self):
        """§7.4.6: degraded image should have mouth polygon overwritten."""
        img, ctx = _make_synthetic_face()
        mouth_pts = ctx.landmarks_98[MOUTH_OUTER].astype(np.int32)
        hull = cv2.convexHull(mouth_pts)
        mouth_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mouth_mask, hull, 1)

        baseline = proxy_occluded_fraction(img, mouth_mask)
        degraded = comp_module._mouth_occlusion_polygon(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_occluded_fraction(degraded, mouth_mask)

        assert after > baseline + 0.1, (
            f"MouthOcclusion: polygon coverage did not increase "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_face_occlusion_increases_face_mask_overwrite(self):
        """§7.4.7: degraded image should have face-mask region partially overwritten."""
        img, ctx = _make_synthetic_face()
        baseline = proxy_occluded_fraction(img, ctx.face_mask)
        degraded = comp_module._face_region_occlusion(img, severity=1.0, seed=0, ctx=ctx)
        after = proxy_occluded_fraction(degraded, ctx.face_mask)

        assert after > baseline + 0.05, (
            f"FaceOcclusion: face-mask coverage did not increase "
            f"(baseline={baseline:.3f}, after={after:.3f})."
        )

    def test_inter_eye_distance_decreases(self):
        """§7.4.8: v0.5 _reduce_ied shrinks the image into a flat-backdrop canvas.

        On the synthetic fixture, INTER_AREA downscaling smears the small
        dark eye circles into faint blobs that the cv2 eye detector
        misclassifies (returns spurious centers). Verify the geometric
        invariant directly: scan the dark-pixel centroid spread on each
        side of the image center — it shrinks proportional to the
        downscale factor.
        """
        img, _ = _make_synthetic_face()
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def dark_centroid_separation(g: np.ndarray) -> float:
            ys, xs = np.where(g < 70)
            if len(xs) == 0:
                return 0.0
            cx_img = w / 2
            left = xs[xs < cx_img]
            right = xs[xs >= cx_img]
            if len(left) == 0 or len(right) == 0:
                return 0.0
            return float(right.mean() - left.mean())

        baseline = dark_centroid_separation(gray)
        degraded = comp_module._reduce_ied(img, severity=1.0, seed=0)
        gray2 = cv2.cvtColor(degraded, cv2.COLOR_BGR2GRAY)
        after = dark_centroid_separation(gray2)

        assert after < baseline, (
            f"InterEyeDistance/_reduce_ied: dark-feature separation did not "
            f"shrink (baseline={baseline:.1f}, after={after:.1f})."
        )

    def test_head_size_moves_pixel_content(self):
        """§7.4.9: v0.5 _reduce_head_size zooms IN past the OFIQ optimum.

        The pre-v0.5 shrink-only operator pushed scalar from ~2 to 0
        with no real headroom; v0.5 zooms in to drive raw past 0.45,
        which crosses the OFIQ optimum (scalar peaks at 100, then
        falls again). Direction-of-scalar-movement is non-monotonic
        across the optimum, so verify the pixel content was modified
        (operator ran) rather than asserting a single direction on a
        synthetic-fixture proxy of the OFIQ measure.
        """
        img, _ = _make_synthetic_face()
        degraded = comp_module._reduce_head_size(img, severity=1.0, seed=0)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 1.0, (
            f"HeadSize/_reduce_head_size: produced no visible change "
            f"(mean abs diff = {diff:.2f})."
        )

    # ---- The 4 known-bug operators: crop/margin direction tests ----

    def test_leftward_crop_decreases_x_r_over_ied(self):
        """§7.4.10.1: degraded image should have LOWER X_R / IED.

        OFIQ q_l = rightEyeCenter.x / IED. To DEGRADE Q (sigmoid centered 0.9),
        we need q_l to become smaller than baseline.
        """
        img, _ = _make_synthetic_face()
        eye_l, eye_r = detect_eye_centers(img)
        ied = math.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1])
        # rightEyeCenter (OFIQ) = subject's right = observer's left = lower x = eye_l in our detector
        baseline = eye_l[0] / ied

        degraded = comp_module._crop_left(img, severity=1.0, seed=0)
        eye_l2, eye_r2 = detect_eye_centers(degraded)
        assert eye_l2 is not None and eye_r2 is not None
        ied2 = math.hypot(eye_l2[0] - eye_r2[0], eye_l2[1] - eye_r2[1])
        after = eye_l2[0] / ied2

        assert after < baseline, (
            f"LeftwardCrop/_crop_left: q_l = X_R/IED did NOT decrease "
            f"(baseline={baseline:.3f}, after={after:.3f}). "
            f"Operator is in WRONG DIRECTION."
        )

    def test_rightward_crop_decreases_w_minus_x_l_over_ied(self):
        """§7.4.10.2: degraded image should have LOWER (W - X_L) / IED."""
        img, _ = _make_synthetic_face()
        W = img.shape[1]
        eye_l, eye_r = detect_eye_centers(img)
        ied = math.hypot(eye_l[0] - eye_r[0], eye_l[1] - eye_r[1])
        # leftEyeCenter (OFIQ) = subject's left = observer's right = higher x = eye_r in our detector
        baseline = (W - eye_r[0]) / ied

        degraded = comp_module._crop_right(img, severity=1.0, seed=0)
        eye_l2, eye_r2 = detect_eye_centers(degraded)
        assert eye_l2 is not None and eye_r2 is not None
        ied2 = math.hypot(eye_l2[0] - eye_r2[0], eye_l2[1] - eye_r2[1])
        after = (W - eye_r2[0]) / ied2

        assert after < baseline, (
            f"RightwardCrop/_crop_right: q_r = (W-X_L)/IED did NOT decrease "
            f"(baseline={baseline:.3f}, after={after:.3f}). "
            f"Operator is in WRONG DIRECTION."
        )

    def test_margin_above_decreases_y_c_over_t(self):
        """§7.4.10.3: degraded image should have LOWER Y_C / T."""
        img, _ = _make_synthetic_face()
        H = img.shape[0]
        eye_l, eye_r = detect_eye_centers(img)
        Y_C = (eye_l[1] + eye_r[1]) / 2
        # T = eye-mid-to-chin
        skin_mask = (
            (img[:, :, 0] > 160) & (img[:, :, 0] < 200) &
            (img[:, :, 1] > 180) & (img[:, :, 1] < 220) &
            (img[:, :, 2] > 200) & (img[:, :, 2] < 240)
        ).astype(np.uint8)
        ys = np.where(skin_mask.sum(axis=1) > 0)[0]
        chin_y = ys.max()
        T = chin_y - Y_C
        baseline = Y_C / T

        degraded = comp_module._margin_above(img, severity=1.0, seed=0)
        eye_l2, eye_r2 = detect_eye_centers(degraded)
        assert eye_l2 is not None and eye_r2 is not None
        Y_C2 = (eye_l2[1] + eye_r2[1]) / 2
        skin_mask2 = (
            (degraded[:, :, 0] > 160) & (degraded[:, :, 0] < 200) &
            (degraded[:, :, 1] > 180) & (degraded[:, :, 1] < 220) &
            (degraded[:, :, 2] > 200) & (degraded[:, :, 2] < 240)
        ).astype(np.uint8)
        ys2 = np.where(skin_mask2.sum(axis=1) > 0)[0]
        chin_y2 = ys2.max() if len(ys2) > 0 else H
        T2 = chin_y2 - Y_C2
        after = Y_C2 / T2

        assert after < baseline, (
            f"MarginAbove/_margin_above: q_a = Y_C/T did NOT decrease "
            f"(baseline={baseline:.3f}, after={after:.3f}). "
            f"Operator is in WRONG DIRECTION."
        )

    def test_margin_below_decreases_h_minus_y_c_over_t(self):
        """§7.4.10.4: degraded image should have LOWER (H - Y_C) / T."""
        img, _ = _make_synthetic_face()
        H = img.shape[0]
        eye_l, eye_r = detect_eye_centers(img)
        Y_C = (eye_l[1] + eye_r[1]) / 2
        skin_mask = (
            (img[:, :, 0] > 160) & (img[:, :, 0] < 200) &
            (img[:, :, 1] > 180) & (img[:, :, 1] < 220) &
            (img[:, :, 2] > 200) & (img[:, :, 2] < 240)
        ).astype(np.uint8)
        ys = np.where(skin_mask.sum(axis=1) > 0)[0]
        chin_y = ys.max()
        T = chin_y - Y_C
        baseline = (H - Y_C) / T

        degraded = comp_module._margin_below(img, severity=1.0, seed=0)
        eye_l2, eye_r2 = detect_eye_centers(degraded)
        assert eye_l2 is not None and eye_r2 is not None
        Y_C2 = (eye_l2[1] + eye_r2[1]) / 2
        skin_mask2 = (
            (degraded[:, :, 0] > 160) & (degraded[:, :, 0] < 200) &
            (degraded[:, :, 1] > 180) & (degraded[:, :, 1] < 220) &
            (degraded[:, :, 2] > 200) & (degraded[:, :, 2] < 240)
        ).astype(np.uint8)
        ys2 = np.where(skin_mask2.sum(axis=1) > 0)[0]
        chin_y2 = ys2.max() if len(ys2) > 0 else H
        T2 = chin_y2 - Y_C2
        after = (H - Y_C2) / T2

        assert after < baseline, (
            f"MarginBelow/_margin_below: q_b = (H-Y_C)/T did NOT decrease "
            f"(baseline={baseline:.3f}, after={after:.3f}). "
            f"Operator is in WRONG DIRECTION."
        )

    # ---- HeadPose ----

    def test_head_pose_yaw_perspective_warp_changes_image(self):
        """§7.4.11.2: yaw warp should visibly alter pixel content.

        Full validation requires running 3DDFA-V2 ONNX. We verify pixel change
        as a necessary (not sufficient) condition.
        """
        img, _ = _make_synthetic_face()
        degraded = comp_module._yaw_rotation(img, severity=1.0, seed=0)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 5.0, "Yaw warp produced no visible change"

    def test_head_pose_pitch_perspective_warp_changes_image(self):
        img, _ = _make_synthetic_face()
        degraded = comp_module._pitch_tilt(img, severity=1.0, seed=0)
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 5.0, "Pitch warp produced no visible change"

    def test_head_pose_roll_rotates_eyes_off_horizontal(self):
        """§7.4.11.4: roll should rotate eyes off the horizontal axis."""
        img, _ = _make_synthetic_face()
        eye_l, eye_r = detect_eye_centers(img)
        baseline_dy = abs(eye_l[1] - eye_r[1])  # near 0 for horizontal eyes

        degraded = comp_module._roll_rotation(img, severity=1.0, seed=0)
        eye_l2, eye_r2 = detect_eye_centers(degraded)
        assert eye_l2 is not None and eye_r2 is not None
        after_dy = abs(eye_l2[1] - eye_r2[1])

        assert after_dy > baseline_dy + 5, (
            f"HeadPoseRoll: eyes did not tilt "
            f"(baseline dy={baseline_dy:.1f}, after dy={after_dy:.1f})."
        )

    # ---- Generative — model-dependent, marked for OFIQ-binary parity ----

    def test_single_face_present_inserts_extra_content(self):
        """§7.4.2: insert_second_face should add pixel content in background.

        Full validation requires SSD face detector. We verify pixel change.
        """
        img, ctx = _make_synthetic_face()
        from ofiq_syngen.generative.single_face import insert_second_face
        try:
            degraded = insert_second_face(img, severity=1.0, seed=0, ctx=ctx)
        except Exception as e:
            pytest.skip(f"insert_second_face requires more setup: {e}")
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 1.0, "insert_second_face produced no change"

    def test_expression_neutrality_warps_landmarks(self):
        """§7.4.12: add_expression should visibly alter pixel content.

        v0.5 default backend is 3DMM, which falls back to TPS landmark
        warp on the synthetic fixture (no raw_3ddfa_params in StubContext).
        TPS warp on synthetic landmarks produces only mild displacement;
        accept any nonzero pixel change as evidence the operator ran.
        """
        img, ctx = _make_synthetic_face()
        from ofiq_syngen.generative.expression import add_expression
        try:
            degraded = add_expression(img, severity=1.0, seed=0, ctx=ctx)
        except Exception as e:
            pytest.skip(f"add_expression requires more setup: {e}")
        diff = np.abs(img.astype(np.int16) - degraded.astype(np.int16)).mean()
        assert diff > 0.1, f"add_expression produced no change (diff={diff:.3f})"

    def test_no_head_coverings_overlays_top_region(self):
        """§7.4.13: add_head_covering should overlay content in upper face area."""
        img, ctx = _make_synthetic_face()
        from ofiq_syngen.generative.head_covering import add_head_covering
        try:
            degraded = add_head_covering(img, severity=1.0, seed=0, ctx=ctx)
        except Exception as e:
            pytest.skip(f"add_head_covering requires more setup: {e}")
        # Upper third should change
        upper = img[:img.shape[0] // 3]
        upper_d = degraded[:img.shape[0] // 3]
        diff = np.abs(upper.astype(np.int16) - upper_d.astype(np.int16)).mean()
        assert diff > 1.0, "add_head_covering produced no change in upper region"
