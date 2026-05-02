"""3DMM utilities for FACS-aligned expression simulation.

Wraps the 3DDFA-V2 model's full 62-dim parameter output (12 pose + 40
shape + 10 expression) using a sparse 68-landmark BFM-2009 basis. The
basis files (~40KB) are bundled with the package; no external download
is required at inference time.

Pipeline:
    raw_3ddfa_params (62,) -> parse_3ddfa_params -> (R, offset, shape, expr)
    expr_modified = expr + delta(emotion, severity)
    src_landmarks = project_3dmm_landmarks(R, offset, shape, expr,        img_w, img_h)
    dst_landmarks = project_3dmm_landmarks(R, offset, shape, expr_modified, img_w, img_h)
    warped = TPS_warp(image, src_landmarks, dst_landmarks)

The 68 landmarks are the iBUG-300W standard subset of the BFM-2009
mean shape. Modifying expression coefficients then re-projecting yields
an anatomically-plausible displacement field constrained by real face
geometry, instead of the hand-picked landmark perturbations TPS warps
otherwise rely on.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


_DATA_DIR = Path(__file__).resolve().parent / "data" / "3dmm"
_BFM_CACHE: dict | None = None
_PARAM_MS_CACHE: tuple[np.ndarray, np.ndarray] | None = None


def _load_bfm_sparse() -> dict:
    """Load the 68-landmark sparse BFM-2009 basis (cached)."""
    global _BFM_CACHE
    if _BFM_CACHE is not None:
        return _BFM_CACHE
    path = _DATA_DIR / "bfm_sparse.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"BFM sparse basis not found at {path}. "
            "Run scripts/fetch_3dmm_basis.py to install."
        )
    d = np.load(path)
    _BFM_CACHE = {
        "u_base": d["u_base"].astype(np.float64),         # (68, 3)
        "w_shp_base": d["w_shp_base"].astype(np.float64), # (68, 3, 40)
        "w_exp_base": d["w_exp_base"].astype(np.float64), # (68, 3, 10)
    }
    return _BFM_CACHE


def load_param_mean_std() -> tuple[np.ndarray, np.ndarray]:
    """Load the 62-dim parameter mean and std for 3DDFA-V2 denormalization."""
    global _PARAM_MS_CACHE
    if _PARAM_MS_CACHE is not None:
        return _PARAM_MS_CACHE
    path = _DATA_DIR / "param_mean_std_62d_120x120.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"3DDFA-V2 param mean/std not found at {path}. "
            "Run scripts/fetch_3dmm_basis.py to install."
        )
    with open(path, "rb") as f:
        pms = pickle.load(f)
    _PARAM_MS_CACHE = (pms["mean"].astype(np.float32), pms["std"].astype(np.float32))
    return _PARAM_MS_CACHE


def parse_3ddfa_params(
    params_62d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a 62-dim 3DDFA-V2 parameter vector into pose + 3DMM coefficients.

    Args:
        params_62d: (62,) DENORMALIZED 3DDFA-V2 output (raw * std + mean).
            Layout: [12 pose | 40 shape | 10 expression].
            Pose is reshaped to (3, 4) row-major as [R | translation].

    Returns:
        R: (3, 3) rotation+scale matrix.
        offset: (3, 1) translation in 120x120 space.
        alpha_shp: (40, 1) identity shape coefficients.
        alpha_exp: (10, 1) expression blendshape coefficients.
    """
    if params_62d.shape != (62,):
        raise ValueError(f"Expected (62,) params, got {params_62d.shape}")
    pose = params_62d[:12].reshape(3, 4).astype(np.float64)
    R = pose[:, :3]
    offset = pose[:, -1:].copy()
    alpha_shp = params_62d[12:52].reshape(-1, 1).astype(np.float64)
    alpha_exp = params_62d[52:62].reshape(-1, 1).astype(np.float64)
    return R, offset, alpha_shp, alpha_exp


def project_3dmm_landmarks(
    R: np.ndarray,
    offset: np.ndarray,
    alpha_shp: np.ndarray,
    alpha_exp: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Project the 68 BFM-2009 sparse landmarks to image coordinates.

    Args:
        R, offset, alpha_shp, alpha_exp: as from ``parse_3ddfa_params``.
        img_w, img_h: original image width and height (the image was
            resized to 120x120 for the 3DDFA-V2 forward pass; we undo
            that scale here, including the non-uniform aspect).

    Returns:
        (68, 2) array of landmark positions in original image pixel space.
    """
    bfm = _load_bfm_sparse()
    # 3D shape at 68 landmarks
    shape_3d = (
        bfm["u_base"]
        + (bfm["w_shp_base"].reshape(68 * 3, 40) @ alpha_shp).reshape(68, 3)
        + (bfm["w_exp_base"].reshape(68 * 3, 10) @ alpha_exp).reshape(68, 3)
    )
    # Apply pose transform (in 120x120 input space)
    proj_120 = (R @ shape_3d.T + offset).T  # (68, 3)
    pts_120 = proj_120[:, :2]
    # Scale back to original image space (3DDFA-V2 took a stretched 120x120 input)
    sx = img_w / 120.0
    sy = img_h / 120.0
    pts_img = pts_120 * np.array([sx, sy], dtype=np.float64)
    return pts_img


def expression_delta(
    emotion: str, severity: float, current_expr: np.ndarray | None = None,
) -> np.ndarray:
    """Return a (10,) delta to add to alpha_exp for the requested emotion.

    The 10 BFM-2009 expression modes don't correspond 1:1 to FACS Action
    Units, but the first few PCA modes empirically encode (in order of
    influence): jaw open, smile, brow raise, lip purse, eye open. The
    deltas here were chosen by inspecting per-mode landmark displacements.

    Severity is sqrt-front-loaded so even sev=0.25 produces visible change.

    Args:
        emotion: one of 'smile', 'frown', 'surprise', 'disgust'.
        severity: [0, 1] strength.
        current_expr: optional current (10, 1) expression coefficients;
            if provided, the delta is reduced where the current value
            already aligns with the target direction (avoids over-shooting
            already-expressive faces).

    Returns:
        (10,) float delta vector to add to alpha_exp.
    """
    s_eff = float(np.sqrt(np.clip(severity, 0.0, 1.0)))

    # Per-emotion 10-dim deltas. Magnitudes are tuned for visible change
    # at sev=1.0 without breaking realism. Signs were chosen by looking
    # at the BFM-2009 PCA mode interpretations from the 3DDFA-V2 docs.
    # Magnitudes calibrated empirically so each emotion produces ~20-25 px
    # max landmark displacement at sev=1.0 on a 350x300 image. Per-mode
    # pixel response varies (mode 6 ~3 px/unit, others ~1.5-2 px/unit) so
    # the template values are scaled per-mode to give comparable visible
    # impact across emotions. Mouth-corner movement at sev=1.0 is ~10-12 px
    # which matches a moderate smile/frown anatomically.
    # v0.5.1: surprise mode-0 (jaw open) reduced from 8 -> 4. The
    # original magnitude pushed mouth landmarks ~25px down at sev=1.0,
    # which dragged the source's white teeth pixels onto the chin via
    # the TPS warp + soft-mask composite. Halving keeps the expression
    # change visible (~12px max landmark displacement) without producing
    # the "tooth-on-chin" artifact.
    templates = {
        "smile":    np.array([ 0.0,   8.0,  0.0,  0.0,  0.0,   5.0,  0.0,  0.0,  0.0,  0.0]),
        "frown":    np.array([ 0.0,  -6.0,  0.0,  0.0,  0.0,  -3.0,  5.0,  0.0,  0.0,  0.0]),
        "surprise": np.array([ 4.0,   1.0,  3.0,  0.0,  2.5,   0.0,  0.0,  0.0,  0.0,  0.0]),
        "disgust":  np.array([ 0.0,  -3.0, -5.0,  6.0,  0.0,   0.0,  3.0,  0.0,  0.0,  0.0]),
    }
    if emotion not in templates:
        raise ValueError(f"Unknown emotion '{emotion}'. Use one of {list(templates)}.")
    delta = templates[emotion] * s_eff

    if current_expr is not None:
        # If current expression already trends toward this emotion,
        # reduce the delta so we don't push past anatomical limits.
        current = current_expr.flatten()
        same_sign = np.sign(current) == np.sign(delta)
        delta = np.where(
            same_sign, delta * np.maximum(0.2, 1.0 - 0.4 * np.abs(current)), delta,
        )
    return delta.astype(np.float64)


def is_3dmm_available() -> bool:
    """Whether the bundled BFM basis + param-norm files are present."""
    return (
        (_DATA_DIR / "bfm_sparse.npz").exists()
        and (_DATA_DIR / "param_mean_std_62d_120x120.pkl").exists()
    )
