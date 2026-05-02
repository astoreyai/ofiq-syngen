"""FaceAnalysis adapter wrapping ofiq-syngen.FaceContext.

The 3D scene needs the same 2D analysis ofiq-syngen uses: ADNet 98-pt
landmarks, BiSeNet face parsing, occlusion mask, HeadPose3DDFAV2 angles.
Rather than re-implement, we wrap. If ofiq-syngen is not installed we
return a typed stub so the scaffold remains importable for plumbing tests.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class FaceAnalysis:
    """Minimal interface 3dsyn relies on. Satisfied by ofiq_syngen.FaceContext.

    Field set is the subset of FaceContext used by 3D perturbations and
    background compositing. Adding fields here means updating the adapter,
    not duplicating the whole FaceContext schema.
    """

    image: np.ndarray
    landmarks_98: np.ndarray
    parsing_map: Optional[np.ndarray]
    occlusion_mask: Optional[np.ndarray]
    head_pose: tuple[float, float, float]
    face_mask: np.ndarray


def build_face_analysis(image: np.ndarray) -> Optional[FaceAnalysis]:
    """Return an ofiq-syngen FaceContext for the image, or None if unavailable.

    We never raise: callers handle None by falling back to a Mock pipeline
    or by raising a strict-mode error themselves.
    """
    try:
        from ofiq_syngen.face_context import FaceContext
        from ofiq_syngen.models import get_models
    except ImportError:
        return None

    try:
        models = get_models()
    except (FileNotFoundError, RuntimeError):
        return None

    try:
        return FaceContext.from_image(image, models)  # type: ignore[return-value]
    except Exception:
        return None


def synthesize_mock_face_analysis(image: np.ndarray) -> Any:
    """Build a minimal stub face analysis for tests without OFIQ models.

    Lands a single fake landmark set centered on the image, an all-ones
    face mask, frontal head pose, and skips parsing/occlusion. Sufficient
    for the dimension-equality contract test and registry round-trip.
    """
    h, w = image.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    r = min(h, w) * 0.25

    landmarks = np.zeros((98, 2), dtype=np.float32)
    for i in range(98):
        theta = 2.0 * np.pi * i / 98.0
        landmarks[i, 0] = cx + r * np.cos(theta)
        landmarks[i, 1] = cy + r * np.sin(theta)

    class _StubFaceAnalysis:
        pass

    stub = _StubFaceAnalysis()
    stub.image = image
    stub.landmarks_98 = landmarks
    stub.parsing_map = None
    stub.occlusion_mask = None
    stub.head_pose = (0.0, 0.0, 0.0)
    stub.face_mask = np.ones((h, w), dtype=np.uint8) * 255
    return stub
