"""OFIQ ONNX model loader with lazy initialization.

Loads ADNet (landmarks), BiSeNet (face parsing), occlusion segmentation,
and HeadPose3DDFAV2 models from the OFIQ-Project model directory.
Models are cached as singletons -- initialized once on first use.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore[assignment]

# Default model directory — resolved at runtime from env or common install paths
_DEFAULT_MODEL_DIR: Path | None = None

def _find_default_model_dir() -> Path | None:
    """Search common OFIQ model locations."""
    candidates = [
        Path.home() / "OFIQ-Project" / "data" / "models",
        Path("/opt/ofiq/models"),
        Path("/usr/local/share/ofiq/models"),
    ]
    for p in candidates:
        if p.exists() and (p / "face_landmark_estimation").exists():
            return p
    return None

# Relative paths within the model directory
_MODEL_PATHS = {
    "adnet": "face_landmark_estimation/ADNet.onnx",
    "bisenet": "face_parsing/bisenet_400.onnx",
    "occlusion": "face_occlusion_segmentation/face_occlusion_segmentation_ort.onnx",
    "headpose": "head_pose_estimation/mb1_120x120.onnx",
}


class OFIQModels:
    """Singleton-like ONNX model manager for OFIQ models.

    Resolution order for model directory:
    1. Explicit ``model_dir`` constructor argument.
    2. ``OFIQ_MODEL_DIR`` environment variable.
    3. Auto-detected from common install paths (~/OFIQ-Project, /opt/ofiq, etc.).
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        """Initialize the OFIQ ONNX model singleton.

        Resolution order: explicit ``model_dir``, then ``OFIQ_MODEL_DIR``
        environment variable, then auto-detection in common install
        paths.

        Raises:
            FileNotFoundError: if no model directory can be resolved.
            ImportError: if ``onnxruntime`` is not installed.
        """
        if model_dir is not None:
            self._model_dir = Path(model_dir)
        else:
            env = os.environ.get("OFIQ_MODEL_DIR")
            if env:
                self._model_dir = Path(env)
            else:
                auto = _find_default_model_dir()
                if auto is None:
                    raise FileNotFoundError(
                        "OFIQ models not found. Set OFIQ_MODEL_DIR environment variable "
                        "or pass model_dir to OFIQModels(). Expected directory containing "
                        "face_landmark_estimation/ADNet.onnx and other OFIQ ONNX models."
                    )
                self._model_dir = auto

        if ort is None:
            raise ImportError(
                "onnxruntime is required for OFIQ model inference. "
                "Install with: pip install onnxruntime"
            )

        self._sessions: dict[str, ort.InferenceSession] = {}
        self._lock = Lock()

        # Configure session options
        self._sess_opts = ort.SessionOptions()
        self._sess_opts.inter_op_num_threads = 1
        self._sess_opts.intra_op_num_threads = 1
        self._sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    @property
    def model_dir(self) -> Path:
        """Resolved OFIQ model directory."""
        return self._model_dir

    def _get_session(self, key: str) -> ort.InferenceSession:
        if key not in self._sessions:
            with self._lock:
                if key not in self._sessions:
                    path = self._model_dir / _MODEL_PATHS[key]
                    if not path.exists():
                        raise FileNotFoundError(
                            f"OFIQ model not found: {path}. "
                            f"Set OFIQ_MODEL_DIR or pass model_dir to OFIQModels()."
                        )
                    self._sessions[key] = ort.InferenceSession(
                        str(path), self._sess_opts
                    )
        return self._sessions[key]

    @property
    def adnet(self) -> ort.InferenceSession:
        """ADNet 98-point facial landmark estimation."""
        return self._get_session("adnet")

    @property
    def bisenet(self) -> ort.InferenceSession:
        """BiSeNet face parsing (19-class segmentation, 400x400)."""
        return self._get_session("bisenet")

    @property
    def occlusion(self) -> ort.InferenceSession:
        """Face occlusion segmentation (binary mask)."""
        return self._get_session("occlusion")

    @property
    def headpose(self) -> ort.InferenceSession:
        """HeadPose3DDFAV2 (Euler angles from face crop)."""
        return self._get_session("headpose")


# Module-level singleton (lazily populated)
_global_models: OFIQModels | None = None
_global_lock = Lock()


def get_models(model_dir: str | Path | None = None) -> OFIQModels:
    """Get or create the global OFIQModels singleton."""
    global _global_models
    if _global_models is None or (
        model_dir is not None and Path(model_dir) != _global_models.model_dir
    ):
        with _global_lock:
            if _global_models is None or (
                model_dir is not None and Path(model_dir) != _global_models.model_dir
            ):
                _global_models = OFIQModels(model_dir)
    return _global_models
