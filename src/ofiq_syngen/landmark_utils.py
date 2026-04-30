"""OFIQ-aligned landmark utilities for ADNet 98-point landmarks.

Exact Python port of OFIQ's adnet_FaceMap.h, FaceMeasures.cpp, and
image_utils.cpp. All index maps, pair definitions, and metric formulas
match the BSI reference implementation.

Reference: https://arxiv.org/pdf/2109.05721.pdf Appendix A, Figure 6.
"""

from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# ADNet 98-point landmark index map (from adnet_FaceMap.h)
# "Left" = left as seen in the image = person's right eye (physically)
# ---------------------------------------------------------------------------

LEFT_EYE = list(range(60, 68))           # [60..67]
RIGHT_EYE = list(range(68, 76))          # [68..75]
LEFT_EYE_CORNERS = [60, 64]
RIGHT_EYE_CORNERS = [68, 72]
MOUTH_OUTER = list(range(76, 88))        # [76..87]
MOUTH_INNER = list(range(88, 96))        # [88..95]
CONTOUR = list(range(0, 33))             # [0..32]
CHIN = [16]
NOSETIP = [54]
FOREHEAD: list[int] = []                 # empty for ADNet

# ---------------------------------------------------------------------------
# Landmark pair indices for openness/closure measurement (adnet_FaceMap.h)
# ---------------------------------------------------------------------------

PAIRS_LEFT_EYE = [(61, 67), (62, 66), (63, 65)]
PAIRS_RIGHT_EYE = [(69, 75), (70, 74), (71, 73)]
PAIRS_MOUTH_INNER = [(89, 95), (90, 94), (91, 93)]
PAIRS_MOUTH_CENTER = [(90, 94)]

# Alignment reference points for 616x616 aligned face (from utils.cpp)
ALIGNMENT_REF_POINTS = np.float32([
    [251, 272],   # left eye center
    [364, 272],   # right eye center
    [308, 336],   # nose tip
    [262, 402],   # right mouth corner
    [355, 402],   # left mouth corner
])

# BiSeNet face parsing class indices (from FaceParsing.h)
BISENET_BACKGROUND = 0
BISENET_SKIN = 1
BISENET_LEFT_EYEBROW = 2
BISENET_RIGHT_EYEBROW = 3
BISENET_LEFT_EYE = 4
BISENET_RIGHT_EYE = 5
BISENET_EYEGLASSES = 6
BISENET_LEFT_EAR = 7
BISENET_RIGHT_EAR = 8
BISENET_EARRING = 9
BISENET_NOSE = 10
BISENET_MOUTH = 11
BISENET_UPPER_LIP = 12
BISENET_LOWER_LIP = 13
BISENET_NECK = 14
BISENET_NECKLACE = 15
BISENET_CLOTH = 16
BISENET_HAIR = 17
BISENET_HAT = 18


# ---------------------------------------------------------------------------
# Core geometry functions (ported from FaceMeasures.cpp)
# ---------------------------------------------------------------------------

def get_middle(landmarks: np.ndarray, indices: Sequence[int] | None = None) -> tuple[float, float]:
    """Compute the mean (x, y) of selected landmarks.

    Port of FaceMeasures::GetMiddle.
    """
    if indices is not None:
        pts = landmarks[indices]
    else:
        pts = landmarks
    if len(pts) == 0:
        return (0.0, 0.0)
    mx = round(float(pts[:, 0].sum()) / len(pts))
    my = round(float(pts[:, 1].sum()) / len(pts))
    return (mx, my)


def get_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Euclidean distance between two points.

    Port of FaceMeasures::GetDistance.
    """
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return math.sqrt(dx * dx + dy * dy)


def get_max_pair_distance(
    landmarks: np.ndarray,
    pairs: Sequence[tuple[int, int]],
) -> float:
    """Maximum distance among landmark pairs.

    Port of FaceMeasures::GetMaxPairDistance.
    Used for eye openness (LEFT_EYE/RIGHT_EYE pairs) and mouth opening
    (MOUTH_INNER pairs).
    """
    max_dist = 0.0
    for i, j in pairs:
        p1 = (float(landmarks[i, 0]), float(landmarks[i, 1]))
        p2 = (float(landmarks[j, 0]), float(landmarks[j, 1]))
        max_dist = max(max_dist, get_distance(p1, p2))
    return max_dist


def calculate_eye_centers(
    landmarks: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Compute left and right eye centers from corner landmarks.

    Port of calculateEyeCenter (utils.cpp lines 303-316).
    Uses midpoint of eye corner pairs [60,64] and [68,72].
    """
    lc0, lc1 = landmarks[60], landmarks[64]
    left_cx = round(lc0[0] + 0.5 * (lc1[0] - lc0[0]))
    left_cy = round(lc0[1] + 0.5 * (lc1[1] - lc0[1]))

    rc0, rc1 = landmarks[68], landmarks[72]
    right_cx = round(rc0[0] + 0.5 * (rc1[0] - rc0[0]))
    right_cy = round(rc0[1] + 0.5 * (rc1[1] - rc0[1]))

    return (float(left_cx), float(left_cy)), (float(right_cx), float(right_cy))


def tmetric(landmarks: np.ndarray) -> float:
    """T-metric: distance from eye midpoint to chin.

    Port of tmetric (utils.cpp lines 319-331).
    """
    left_eye_center, right_eye_center = calculate_eye_centers(landmarks)
    eye_mid = (
        (left_eye_center[0] + right_eye_center[0]) / 2.0,
        (left_eye_center[1] + right_eye_center[1]) / 2.0,
    )
    chin = (float(landmarks[16, 0]), float(landmarks[16, 1]))
    return get_distance(chin, eye_mid)


def inter_eye_distance(landmarks: np.ndarray, yaw: float) -> float:
    """Yaw-corrected inter-eye distance.

    Port of FaceMeasures::InterEyeDistance (FaceMeasures.cpp lines 73-95).
    """
    EPS = 1e-6
    left_center, right_center = calculate_eye_centers(landmarks)
    cos_yaw = math.cos(yaw * math.pi / 180.0)
    if abs(cos_yaw) < EPS:
        return float("nan")
    dist = get_distance(left_center, right_center)
    return dist / cos_yaw


def calculate_reference_points(
    landmarks: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], float, float]:
    """Compute reference points for ROI calculation.

    Port of CalculateReferencePoints (image_utils.cpp lines 114-132).

    Returns:
        (leftEyeCenter, rightEyeCenter, interEyeDistance, eyeMouthDistance)
    """
    left_eye_center, right_eye_center = calculate_eye_centers(landmarks)
    ied = get_distance(left_eye_center, right_eye_center)
    eye_mid = (
        (left_eye_center[0] + right_eye_center[0]) / 2.0,
        (left_eye_center[1] + right_eye_center[1]) / 2.0,
    )
    # Mouth center = middle of MOUTH_CENTER pair (90, 94)
    mouth_center = get_middle(landmarks, [90, 94])
    eye_mouth_dist = get_distance(eye_mid, mouth_center)
    return left_eye_center, right_eye_center, ied, eye_mouth_dist


def calculate_roi(
    left_eye_center: tuple[float, float],
    right_eye_center: tuple[float, float],
    ied: float,
    eye_mouth_dist: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Compute left and right regions of interest for NaturalColour / IlluminationUniformity.

    Port of CalculateRegionOfInterest (image_utils.cpp lines 134-150).

    Returns:
        (right_roi, left_roi) each as (x, y, width, height).
        Note: OFIQ's "right" ROI is near the right eye center (left side of image).
    """
    zone_size = int(ied * 0.3)
    if zone_size < 1:
        zone_size = 1

    right_roi = (
        int(right_eye_center[0]) - zone_size,
        int(right_eye_center[1]) + int(eye_mouth_dist / 2),
        zone_size,
        zone_size,
    )
    left_roi = (
        int(left_eye_center[0]),
        int(left_eye_center[1]) + int(eye_mouth_dist / 2),
        zone_size,
        zone_size,
    )
    return right_roi, left_roi


# ---------------------------------------------------------------------------
# Face mask (ported from FaceMeasures::GetFaceMask, FaceMeasures.cpp lines 98-228)
# ---------------------------------------------------------------------------

def get_face_mask(
    landmarks: np.ndarray,
    height: int,
    width: int,
    alpha: float = 0.0,
) -> np.ndarray:
    """Generate convex-hull face mask from 98-point ADNet landmarks.

    Port of FaceMeasures::GetFaceMask. When alpha > 0, extends the hull
    to include the forehead region using a fitted ellipse.

    Returns:
        uint8 mask of shape (height, width), 1 = face, 0 = background.
    """
    pts = landmarks.astype(np.int32).tolist()
    landmark_points = [tuple(p) for p in pts]

    if alpha > 0:
        # Compute eye midpoint and chin for forehead extension
        eye_corners = [landmarks[i] for i in LEFT_EYE_CORNERS + RIGHT_EYE_CORNERS]
        eyes_mid = np.mean(eye_corners, axis=0)

        chin_pt = landmarks[16].astype(np.float32)
        contour_indices = [0, 7, 25, 32]
        contour_pts = [landmarks[i].astype(np.int32) for i in contour_indices]

        chin_mid_vec = eyes_mid - chin_pt  # vector from chin to eye midpoint

        top_of_forehead = (
            int(eyes_mid[0] + alpha * chin_mid_vec[0]),
            int(eyes_mid[1] + alpha * chin_mid_vec[1]),
        )

        ellipse_points = [tuple(p) for p in contour_pts]
        ellipse_points.append((int(chin_pt[0]), int(chin_pt[1])))
        ellipse_points.append(top_of_forehead)

        ellipse_np = np.array(ellipse_points, dtype=np.int32)
        if len(ellipse_np) >= 5:
            rotated_rect = cv2.fitEllipse(ellipse_np)
            center = (int(rotated_rect[0][0]), int(rotated_rect[0][1]))
            axes = (int(rotated_rect[1][0] / 2), int(rotated_rect[1][1] / 2))
            angle = int(rotated_rect[2])
            poly_points = cv2.ellipse2Poly(center, axes, angle, 0, 360, 10)

            # Keep only points on the forehead side
            chin_mid_dot = float(chin_mid_vec[0] * chin_mid_vec[0] + chin_mid_vec[1] * chin_mid_vec[1])
            forehead_pts = []
            for p in poly_points:
                vec = (float(p[0]) - chin_pt[0], float(p[1]) - chin_pt[1])
                dot = vec[0] * chin_mid_vec[0] + vec[1] * chin_mid_vec[1]
                if dot > 1.1 * chin_mid_dot:
                    forehead_pts.append(tuple(p))

            landmark_points.extend(forehead_pts)

    all_pts = np.array(landmark_points, dtype=np.int32)
    hull = cv2.convexHull(all_pts)

    # Compute bounding rect and normalize to 224x224 intermediate
    rect = cv2.boundingRect(hull)
    b = int(rect[1] - rect[3] * 0.05)
    d = int(rect[1] + rect[3] * 1.05)
    a = int(rect[0] + rect[2] / 2.0 - (d - b) / 2.0)
    c = int(rect[0] + rect[2] / 2.0 + (d - b) / 2.0)

    img_size = 224
    scale = d - b if d - b > 0 else 1
    hull_scaled = hull.copy().astype(np.float32)
    for i in range(len(hull_scaled)):
        hull_scaled[i, 0, 0] = (hull_scaled[i, 0, 0] - a) / scale * img_size
        hull_scaled[i, 0, 1] = (hull_scaled[i, 0, 1] - b) / scale * img_size
    hull_scaled = hull_scaled.astype(np.int32)

    mask_small = np.zeros((img_size, img_size), dtype=np.uint8)
    cv2.fillConvexPoly(mask_small, hull_scaled, 1)

    # Resize back to the bounding region and embed in full image
    if c - a > 0 and d - b > 0:
        mask_rescaled = cv2.resize(
            mask_small, (c - a, d - b), interpolation=cv2.INTER_NEAREST
        )
    else:
        return np.zeros((height, width), dtype=np.uint8)

    face_region = np.zeros((height, width), dtype=np.uint8)

    # Clip to image bounds
    src_top = max(0, -b)
    src_left = max(0, -a)
    src_bottom = min(mask_rescaled.shape[0], mask_rescaled.shape[0] - max(0, d - height))
    src_right = min(mask_rescaled.shape[1], mask_rescaled.shape[1] - max(0, c - width))

    dst_top = max(0, b)
    dst_left = max(0, a)
    dst_bottom = min(height, d)
    dst_right = min(width, c)

    h_slice = slice(dst_top, dst_bottom)
    w_slice = slice(dst_left, dst_right)
    sh = dst_bottom - dst_top
    sw = dst_right - dst_left

    if sh > 0 and sw > 0:
        crop = mask_rescaled[src_top:src_top + sh, src_left:src_left + sw]
        face_region[h_slice, w_slice] = crop

    return face_region


# ---------------------------------------------------------------------------
# Color and luminance (ported from image_utils.cpp)
# ---------------------------------------------------------------------------

def _color_convert(x: float) -> float:
    """sRGB linearization. Port of ColorConvert (image_utils.cpp line 43)."""
    if x <= 0.04045:
        return x / 12.92
    return ((x + 0.055) / 1.055) ** 2.4


# Pre-compute LUT (matches prepare_COLOR_CVT_LUT)
_COLOR_CVT_LUT = np.array([_color_convert(i / 255.0) for i in range(256)], dtype=np.float64)


def get_luminance_image(bgr: np.ndarray) -> np.ndarray:
    """Rec.709 linearized luminance image.

    Port of GetLuminanceImageFromBGR (image_utils.cpp lines 99-112).
    NOT equivalent to cv2.cvtColor(BGR2GRAY) which uses a different formula.

    Returns:
        uint8 luminance image of shape (H, W).
    """
    b_lin = _COLOR_CVT_LUT[bgr[:, :, 0]]  # B
    g_lin = _COLOR_CVT_LUT[bgr[:, :, 1]]  # G
    r_lin = _COLOR_CVT_LUT[bgr[:, :, 2]]  # R

    y = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return np.floor(y * 255 + 0.5).astype(np.uint8)


def _cubic(x: float, k: float, eps: float) -> float:
    """Port of Cubic (image_utils.cpp lines 58-66)."""
    if x <= eps:
        return (k * x + 16) / 116
    return x ** (1.0 / 3.0)


def convert_bgr_to_cielab(bgr_region: np.ndarray) -> tuple[float, float]:
    """Convert a BGR region to CIELAB and return mean (a*, b*).

    Port of ConvertBGRToCIELAB (image_utils.cpp lines 68-97).
    Uses OFIQ's exact D50 illuminant and sRGB matrix, NOT cv2.cvtColor.

    Args:
        bgr_region: BGR uint8 image region.

    Returns:
        (mean_a_star, mean_b_star) in CIELAB coordinates.
    """
    k = 24289.0 / 27.0
    eps = 216.0 / 24389.0

    # Mean channel values normalized to [0, 1]
    R = float(bgr_region[:, :, 2].mean()) / 255.0
    G = float(bgr_region[:, :, 1].mean()) / 255.0
    B = float(bgr_region[:, :, 0].mean()) / 255.0

    R_L = _color_convert(R)
    G_L = _color_convert(G)
    B_L = _color_convert(B)

    X = R_L * 0.43605 + G_L * 0.38508 + B_L * 0.14309
    Y = R_L * 0.22249 + G_L * 0.71689 + B_L * 0.06062
    Z = R_L * 0.01393 + G_L * 0.09710 + B_L * 0.71419

    X_R = X / 0.964221
    Y_R = Y
    Z_R = Z / 0.825211

    F_X = _cubic(X_R, k, eps)
    F_Y = _cubic(Y_R, k, eps)
    F_Z = _cubic(Z_R, k, eps)

    a_star = 500.0 * (F_X - F_Y)
    b_star = 200.0 * (F_Y - F_Z)
    return a_star, b_star


# ---------------------------------------------------------------------------
# EVZ (Eye Visibility Zone) computation (from EyesVisible.cpp)
# ---------------------------------------------------------------------------

def compute_evz_rects(
    landmarks: np.ndarray,
    ied: float,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Compute Eye Visibility Zone rectangles.

    Port of EyesVisible.cpp EVZ computation.
    V = floor(IED / 20). Each eye's bounding rect is expanded by V pixels.

    Returns:
        (left_eye_evz, right_eye_evz) each as (x, y, w, h).
    """
    v = int(math.floor(ied / 20.0))

    def _eye_evz(eye_indices: list[int]) -> tuple[int, int, int, int]:
        pts = landmarks[eye_indices].astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        return (x - v, y - v, w + 2 * v, h + 2 * v)

    left_evz = _eye_evz(LEFT_EYE)
    right_evz = _eye_evz(RIGHT_EYE)
    return left_evz, right_evz
