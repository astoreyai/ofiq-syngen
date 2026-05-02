"""FaceContext: OFIQ-aligned face analysis computed once per image.

Runs OFIQ's own ONNX models (ADNet, BiSeNet, occlusion segmentation,
HeadPose3DDFAV2) with their exact preprocessing pipelines, then caches
all derived metrics (IED, t-metric, face mask, luminance, ROIs).

Each degradation function receives the FaceContext and uses it to target
the same regions OFIQ analyzes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from ofiq_syngen.landmark_utils import (
    calculate_eye_centers,
    calculate_reference_points,
    calculate_roi,
    compute_evz_rects,
    get_face_mask,
    get_luminance_image,
    inter_eye_distance,
    tmetric,
)
from ofiq_syngen.models import OFIQModels, get_models

# HeadPose3DDFAV2 denormalization parameters (from HeadPose3DDFAV2.cpp lines 41-56)
_PARAM_MEAN = np.array(
    [3.4926363e-04, 2.5279013e-07, -6.8751979e-07,
     6.0167957e+01, -6.2955132e-07, 5.7572004e-04, -5.0853912e-05],
    dtype=np.float32,
)
_PARAM_STD = np.array(
    [1.76321526e-04, 6.73794348e-05, 4.47084894e-04,
     2.65502319e+01, 1.23137695e-04, 4.49302170e-05, 7.92367064e-05],
    dtype=np.float32,
)

# BiSeNet normalization (from FaceParsing.cpp lines 65-84)
_BISENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
_BISENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0


@dataclass
class FaceContext:
    """All OFIQ-derived face analysis for a single image.

    Computed once, passed to every degradation function that needs it.
    """

    image: np.ndarray                         # original BGR uint8
    is_aligned: bool                          # True if 616x616 OFIQ-aligned
    landmarks_98: np.ndarray                  # (98, 2) int, ADNet landmarks
    parsing_map: np.ndarray                   # (400, 400) uint8, BiSeNet classes 0-18
    occlusion_mask: np.ndarray                # image-sized uint8, 1=visible 0=occluded
    head_pose: tuple[float, float, float]     # (yaw, pitch, roll) degrees
    raw_3ddfa_params: np.ndarray              # (62,) full 3DDFA-V2 output (denormalized)

    # Derived metrics (computed in __post_init__)
    left_eye_center: tuple[float, float] = field(init=False)
    right_eye_center: tuple[float, float] = field(init=False)
    eye_midpoint: tuple[float, float] = field(init=False)
    chin: tuple[float, float] = field(init=False)
    ied: float = field(init=False)            # yaw-corrected inter-eye distance
    t_metric: float = field(init=False)       # eye-midpoint to chin
    eye_mouth_dist: float = field(init=False)
    face_mask: np.ndarray = field(init=False) # convex hull uint8
    luminance: np.ndarray = field(init=False) # rec709 luminance uint8
    right_roi: tuple[int, int, int, int] = field(init=False)  # (x, y, w, h)
    left_roi: tuple[int, int, int, int] = field(init=False)
    left_evz: tuple[int, int, int, int] = field(init=False)
    right_evz: tuple[int, int, int, int] = field(init=False)

    def __post_init__(self) -> None:
        h, w = self.image.shape[:2]

        self.left_eye_center, self.right_eye_center = calculate_eye_centers(
            self.landmarks_98
        )
        self.eye_midpoint = (
            (self.left_eye_center[0] + self.right_eye_center[0]) / 2.0,
            (self.left_eye_center[1] + self.right_eye_center[1]) / 2.0,
        )
        self.chin = (
            float(self.landmarks_98[16, 0]),
            float(self.landmarks_98[16, 1]),
        )

        yaw = self.head_pose[0]
        self.ied = inter_eye_distance(self.landmarks_98, yaw)
        self.t_metric = tmetric(self.landmarks_98)

        _, _, raw_ied, self.eye_mouth_dist = calculate_reference_points(
            self.landmarks_98
        )
        self.right_roi, self.left_roi = calculate_roi(
            self.left_eye_center, self.right_eye_center,
            raw_ied, self.eye_mouth_dist,
        )
        self.left_evz, self.right_evz = compute_evz_rects(
            self.landmarks_98, self.ied if not math.isnan(self.ied) else raw_ied,
        )

        self.face_mask = get_face_mask(self.landmarks_98, h, w, alpha=1.0)
        self.luminance = get_luminance_image(self.image)

    @classmethod
    def from_image(
        cls,
        image: np.ndarray,
        models: OFIQModels | None = None,
        is_aligned: bool | None = None,
    ) -> FaceContext:
        """Build FaceContext by running all OFIQ models on the image.

        Args:
            image: BGR uint8 face image.
            models: OFIQModels instance (uses global singleton if None).
            is_aligned: Whether image is 616x616 OFIQ-aligned.
                Auto-detected from shape if None.
        """
        if models is None:
            models = get_models()

        h, w = image.shape[:2]
        if is_aligned is None:
            is_aligned = (h == 616 and w == 616)

        landmarks = _run_adnet(image, models, is_aligned)
        parsing = _run_bisenet(image, models, is_aligned)
        occlusion = _run_occlusion_seg(image, models, is_aligned)
        pose, raw_params = _run_headpose(image, models, is_aligned)

        return cls(
            image=image,
            is_aligned=is_aligned,
            landmarks_98=landmarks,
            parsing_map=parsing,
            occlusion_mask=occlusion,
            head_pose=pose,
            raw_3ddfa_params=raw_params,
        )


# ---------------------------------------------------------------------------
# Model inference functions (exact OFIQ preprocessing)
# ---------------------------------------------------------------------------

def _run_adnet(
    image: np.ndarray, models: OFIQModels, is_aligned: bool,
) -> np.ndarray:
    """Run ADNet landmark extraction.

    Preprocessing (from adnet_landmarks.cpp):
    - For aligned images: feed the full 616x616 image
    - Normalize: 2/255 * pixel - 1 (range [-1, 1])
    - HWC -> CHW
    - Output: (landmark + 1) / 2 * 255, then scale by face_height/256
    """
    session = models.adnet

    # Get expected input shape from model
    input_shape = session.get_inputs()[0].shape  # [1, C, H, W]
    _, _, model_h, model_w = input_shape

    if is_aligned:
        # For aligned images, the full image IS the face crop
        face_crop = image
        offset_x, offset_y = 0, 0
        face_h = image.shape[0]
    else:
        # For uncropped images, use the whole image as the face region
        # (simplified: in practice OFIQ uses SSD face detection first)
        face_crop = image
        offset_x, offset_y = 0, 0
        face_h = image.shape[0]

    # Resize to model input size
    resized = cv2.resize(face_crop, (model_w, model_h), interpolation=cv2.INTER_LINEAR)

    # Normalize: 2/255 * x - 1
    normalized = resized.astype(np.float32) * (2.0 / 255.0) - 1.0

    # HWC -> CHW, then add batch dimension
    chw = np.transpose(normalized, (2, 0, 1))  # (3, H, W)
    batch = chw[np.newaxis, ...]  # (1, 3, H, W)

    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    # Take last output (matching C++ implementation)
    raw_landmarks = outputs[-1].flatten()

    # Denormalize: (landmark + 1) / 2 * 255
    denorm = (raw_landmarks + 1.0) / 2.0 * 255.0

    # Scale to image coordinates
    scaling_factor = face_h / 256.0
    landmarks = np.zeros((98, 2), dtype=np.int32)
    for i in range(98):
        x = int(round(denorm[i * 2] * scaling_factor + offset_x))
        y = int(round(denorm[i * 2 + 1] * scaling_factor + offset_y))
        landmarks[i] = [x, y]

    return landmarks


def _run_bisenet(
    image: np.ndarray, models: OFIQModels, is_aligned: bool,
) -> np.ndarray:
    """Run BiSeNet face parsing segmentation.

    Preprocessing (from FaceParsing.cpp lines 62-106):
    - Input: aligned face image (616x616)
    - Crop: rows [0, h-60], cols [30, w-30]
    - Convert BGR -> RGB
    - Normalize: (pixel - mean*255) / (std*255)
    - Resize to 400x400
    - Run model
    - Output: argmax across 19 channels -> class ID per pixel

    For non-aligned images, use the full image as approximation.
    """
    session = models.bisenet

    if is_aligned:
        h, w = image.shape[:2]
        cropped = image[0:h - 60, 30:w - 30]
    else:
        cropped = image

    # BGR -> RGB
    rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Normalize
    rgb = (rgb - _BISENET_MEAN) / _BISENET_STD

    # Resize to 400x400
    resized = cv2.resize(rgb, (400, 400), interpolation=cv2.INTER_LINEAR)

    # HWC -> CHW, add batch
    chw = np.transpose(resized, (2, 0, 1))
    batch = chw[np.newaxis, ...].astype(np.float32)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    # Output is (1, 19, 400, 400) -- argmax across class dimension
    logits = outputs[0]  # (1, C, H, W)
    parsing_map = np.argmax(logits[0], axis=0).astype(np.uint8)  # (400, 400)

    return parsing_map


def _run_occlusion_seg(
    image: np.ndarray, models: OFIQModels, is_aligned: bool,
) -> np.ndarray:
    """Run face occlusion segmentation.

    Preprocessing (from FaceOcclusionSegmentation.cpp lines 57-102):
    - Input: aligned face image (616x616)
    - Crop: 96px on all sides -> 424x424
    - Resize to 224x224
    - blobFromImage: scale=1/255, no mean subtraction, swap channels
    - Run model
    - Post-process: multiply by -1, threshold at 0 (THRESH_BINARY_INV),
      resize back, embed in 616x616

    Output: uint8 mask, 1=visible, 0=occluded.
    """
    session = models.occlusion

    h, w = image.shape[:2]

    if is_aligned:
        pad = 96
        cropped = image[pad:h - pad, pad:w - pad]
        crop_h, crop_w = cropped.shape[:2]
    else:
        # For non-aligned, crop proportionally
        pad_y = int(h * 96 / 616)
        pad_x = int(w * 96 / 616)
        cropped = image[pad_y:h - pad_y, pad_x:w - pad_x]
        crop_h, crop_w = cropped.shape[:2]
        pad = pad_y  # approximate

    # Resize to 224x224
    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Normalize: scale by 1/255, swap BGR->RGB
    blob = cv2.dnn.blobFromImage(
        resized, scalefactor=1.0 / 255.0, size=(224, 224),
        mean=(0, 0, 0), swapRB=True, crop=False,
    )

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob.astype(np.float32)})

    # Post-process: multiply by -1, threshold
    raw = outputs[-1].squeeze()
    if raw.ndim == 3:
        raw = raw[0]  # take first channel if multi-channel
    raw = raw * -1.0

    # Threshold at 0 (THRESH_BINARY_INV equivalent)
    mask_224 = (raw > 0).astype(np.uint8)

    # Resize back to cropped dimensions
    mask_crop = cv2.resize(mask_224, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)

    # Embed in full image
    occlusion_mask = np.zeros((h, w), dtype=np.uint8)
    if is_aligned:
        occlusion_mask[pad:pad + crop_h, pad:pad + crop_w] = mask_crop
    else:
        pad_y_actual = (h - crop_h) // 2
        pad_x_actual = (w - crop_w) // 2
        occlusion_mask[pad_y_actual:pad_y_actual + crop_h,
                       pad_x_actual:pad_x_actual + crop_w] = mask_crop

    return occlusion_mask


def _run_headpose(
    image: np.ndarray, models: OFIQModels, is_aligned: bool,
) -> tuple[tuple[float, float, float], np.ndarray]:
    """Run HeadPose3DDFAV2 to get Euler angles AND full 62-dim params.

    The model outputs 62 parameters: 12 pose (R + translation) + 40 shape
    + 10 expression. OFIQ uses only the first 7 for Euler angles. We
    return both: the (yaw, pitch, roll) tuple AND the full 62-dim
    denormalized vector for downstream 3DMM-based perturbations
    (ExpressionNeutrality).

    Returns:
        ((yaw, pitch, roll) in degrees, raw_params_62d as np.ndarray).
    """
    session = models.headpose

    h, w = image.shape[:2]

    if is_aligned:
        # For aligned images, the face is centered -- use full image as crop
        face_crop = image
    else:
        # Approximate: use center region
        face_crop = image

    # Resize to 120x120
    resized = cv2.resize(face_crop, (120, 120), interpolation=cv2.INTER_LINEAR)

    # Normalize: (pixel - 127.5) / 128.0
    normalized = (resized.astype(np.float32) - 127.5) / 128.0

    # HWC -> CHW
    chw = np.transpose(normalized, (2, 0, 1))
    batch = chw[np.newaxis, ...]  # (1, 3, 120, 120)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})

    raw_full = outputs[0].flatten()  # (62,)

    # Denormalize all 62 params using the bundled 62-dim mean/std table.
    # Falls back to the OFIQ-style 7-dim subset if the full table isn't
    # available (e.g. tests that don't ship the data file).
    try:
        from ofiq_syngen.face_3dmm import load_param_mean_std
        mean62, std62 = load_param_mean_std()
        raw_full_denorm = raw_full * std62 + mean62
    except FileNotFoundError:
        # Fall back to the OFIQ subset; raw_3ddfa_params will only be
        # partially valid but Euler angles still work correctly.
        raw_full_denorm = raw_full.copy()
        raw_full_denorm[:7] = raw_full[:7] * _PARAM_STD + _PARAM_MEAN

    # OFIQ uses just the first 7 (after denormalization) for Euler angles.
    params = raw_full_denorm[:7]

    # Build rotation matrix
    r0 = params[0:3].copy()
    r1 = params[4:7].copy()
    r0 /= np.linalg.norm(r0) + 1e-10
    r1 /= np.linalg.norm(r1) + 1e-10
    r2 = np.cross(r0, r1)

    rot = np.stack([r0, r1, r2], axis=0).T.astype(np.float64)

    # Extract Euler angles (from HeadPose3DDFAV2.cpp lines 177-213)
    thres = 0.9975
    r11, r12, r13 = rot[0, 0], rot[0, 1], rot[0, 2]
    r21 = rot[1, 0]
    r31, r32, r33 = rot[2, 0], rot[2, 1], rot[2, 2]

    if -thres < r31 < thres:
        phi_pitch = math.asin(r31)
        s = 1.0 / math.cos(phi_pitch)
        phi_yaw = -math.atan2(s * r32, s * r33)
        phi_roll = -math.atan2(s * r21, s * r11)
    elif r31 <= -thres:
        phi_pitch = -0.5 * math.pi
        phi_yaw = -math.atan2(r12, r13)
        phi_roll = 0.0
    else:
        phi_pitch = 0.5 * math.pi
        phi_yaw = math.atan2(r12, r13)
        phi_roll = 0.0

    yaw = phi_yaw * 180.0 / math.pi
    pitch = phi_pitch * 180.0 / math.pi
    roll = phi_roll * 180.0 / math.pi

    return (yaw, pitch, roll), raw_full_denorm
