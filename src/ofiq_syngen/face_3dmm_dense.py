"""Dense 3DDFA-V2 / BFM-2009 head pose re-rendering via global TPS warp.

Renders new yaw/pitch by:
  1. Parse 62-dim 3DDFA-V2 params into (R, offset, shape, expression).
  2. Compute the dense 3D mesh: u + w_shp @ shape + w_exp @ expression.
  3. Project mesh to 2D at original pose (src_2d) and rotated pose (dst_2d).
  4. Use ~200 sub-sampled vertex pairs as TPS control points for an
     INVERSE displacement field (dst -> src), with image-corner anchors
     keeping the background fixed.
  5. cv2.remap the source image through the field.

The TPS interpolation is C^infty smooth, so there are no per-triangle
seams or rasterization artifacts. Quality is photographic up to the
~15-degree rotation range that the underlying 2D-to-2D image warp can
plausibly represent without proper occlusion handling. Beyond that
rotation a TPS warp cannot synthesize the side of the face that becomes
visible -- a true 3D rasterizer (pytorch3d/nvdiffrast) would be needed.

The dense BFM mesh (38,365 vertices, 76,073 triangles) is bundled at
``data/3dmm/bfm_dense.npz`` (~22 MB).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ofiq_syngen.face_3dmm import parse_3ddfa_params

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


_DATA_DIR = Path(__file__).resolve().parent / "data" / "3dmm"
_BFM_DENSE_CACHE: dict | None = None


def _load_bfm_dense() -> dict:
    """Load the dense BFM-2009 mesh (cached singleton, ~22 MB)."""
    global _BFM_DENSE_CACHE
    if _BFM_DENSE_CACHE is not None:
        return _BFM_DENSE_CACHE
    path = _DATA_DIR / "bfm_dense.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dense BFM not found at {path}. "
            "Run scripts/fetch_3dmm_basis.py to install."
        )
    d = np.load(path)
    n_verts = d["u"].shape[0] // 3
    # The BFM-2009 "noneck" pkl removes neck vertices but keeps the
    # original triangle list, so many triangles reference indices that
    # no longer exist. Filter to triangles whose three vertices all
    # survived the noneck cut. This drops ~60% of triangles but keeps
    # the entire face surface.
    tri_raw = d["tri"].astype(np.int32)
    valid = (tri_raw < n_verts).all(axis=0)
    tri_filtered = tri_raw[:, valid]
    _BFM_DENSE_CACHE = {
        "u": d["u"].reshape(n_verts, 3).astype(np.float64),
        "w_shp": d["w_shp"].reshape(n_verts, 3, -1).astype(np.float64),
        "w_exp": d["w_exp"].reshape(n_verts, 3, -1).astype(np.float64),
        "tri": tri_filtered,  # (3, T_valid)
        "n_verts": n_verts,
    }
    return _BFM_DENSE_CACHE


def is_dense_available() -> bool:
    """Whether the bundled dense BFM mesh is available."""
    return (_DATA_DIR / "bfm_dense.npz").exists()


def _compute_mesh_3d(
    R: np.ndarray, offset: np.ndarray,
    alpha_shp: np.ndarray, alpha_exp: np.ndarray,
) -> np.ndarray:
    """Compute the 3D mesh vertices in camera space.

    Returns (N, 3) array of vertex 3D positions after pose transform.
    """
    bfm = _load_bfm_dense()
    # 3D shape in model space: u + w_shp @ alpha_shp + w_exp @ alpha_exp
    # (each w basis is shape (n_verts, 3, modes); contract on modes)
    shp_contrib = bfm["w_shp"] @ alpha_shp.flatten()  # (n_verts, 3)
    exp_contrib = bfm["w_exp"] @ alpha_exp.flatten()  # (n_verts, 3)
    verts = bfm["u"] + shp_contrib + exp_contrib  # (n, 3)
    # Apply pose transform
    verts_cam = (R @ verts.T + offset).T  # (n, 3)
    return verts_cam


def _yaw_matrix(deg: float) -> np.ndarray:
    """Rotation matrix about the Y (vertical) axis."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _pitch_matrix(deg: float) -> np.ndarray:
    """Rotation matrix about the X (horizontal, ear-to-ear) axis."""
    r = np.deg2rad(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def render_pose_dense(
    img: np.ndarray,
    ctx: FaceContext,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Re-render the face at a new yaw / pitch using a global TPS warp
    seeded from the dense BFM mesh.

    Args:
        img: source BGR image.
        ctx: FaceContext with raw_3ddfa_params populated.
        yaw_deg: additional yaw rotation in degrees.
        pitch_deg: additional pitch rotation in degrees.
        seed: unused (kept for API consistency).

    Returns:
        BGR image with the face area re-rendered at the new pose.
        The TPS field is anchored at the image corners so hair,
        background, and shoulders remain identical to the source.
    """
    if not hasattr(ctx, "raw_3ddfa_params") or ctx.raw_3ddfa_params is None:
        return img
    if abs(yaw_deg) < 0.1 and abs(pitch_deg) < 0.1:
        return img

    from scipy.interpolate import RBFInterpolator

    R, offset, alpha_shp, alpha_exp = parse_3ddfa_params(ctx.raw_3ddfa_params)
    h, w = img.shape[:2]

    # Source and destination 3D meshes, projected back to image-pixel space
    # (BFM was fit to a 120x120 input, so we apply the inverse scaling).
    verts_src_3d = _compute_mesh_3d(R, offset, alpha_shp, alpha_exp)
    R_extra = _yaw_matrix(yaw_deg) @ _pitch_matrix(pitch_deg)
    R_dst = R_extra @ R
    verts_dst_3d = _compute_mesh_3d(R_dst, offset, alpha_shp, alpha_exp)

    sx, sy = w / 120.0, h / 120.0
    verts_src_2d = verts_src_3d[:, :2] * np.array([sx, sy])
    verts_dst_2d = verts_dst_3d[:, :2] * np.array([sx, sy])

    # Subsample BFM vertices to ~200 control points (TPS solver scales
    # cubically; 200 is the sweet spot between accuracy and speed).
    n_total = verts_src_2d.shape[0]
    step = max(1, n_total // 200)
    ctrl_src = verts_src_2d[::step]
    ctrl_dst = verts_dst_2d[::step]

    # Add hair control points so hair rotates WITH the head (otherwise
    # the face appears to "peel out" of the hair frame at high yaw).
    # For each sampled hair pixel, inherit the displacement of its
    # nearest face vertex -- approximates rigid-body attachment.
    if ctx.parsing_map is not None:
        from ofiq_syngen.landmark_utils import BISENET_HAIR
        hair_small = (ctx.parsing_map == BISENET_HAIR).astype(np.uint8)
        hair = cv2.resize(hair_small, (w, h), interpolation=cv2.INTER_NEAREST)
        # Erode slightly so we don't pick hair pixels far from the face
        hair_inside = cv2.erode(hair, np.ones((5, 5), np.uint8))
        ys, xs = np.where(hair_inside > 0)
        if len(xs) > 0:
            # Sample ~60 hair points (more would slow the TPS solver)
            n_hair = min(60, len(xs))
            sel = np.linspace(0, len(xs) - 1, n_hair).astype(int)
            hair_src = np.column_stack([xs[sel], ys[sel]]).astype(np.float64)
            # Find nearest face vertex (2D L2) and inherit its displacement
            face_disp = verts_dst_2d - verts_src_2d  # (N_face, 2)
            # KD-tree-free vectorized nearest neighbor
            d2 = ((hair_src[:, None, :] - verts_src_2d[None, :, :]) ** 2).sum(axis=2)
            nearest = np.argmin(d2, axis=1)
            hair_dst = hair_src + face_disp[nearest]
            ctrl_src = np.vstack([ctrl_src, hair_src])
            ctrl_dst = np.vstack([ctrl_dst, hair_dst])

    # Anchor the four corners and edge midpoints to themselves so the
    # background stays fixed during the warp.
    corners = np.array(
        [[0, 0], [w, 0], [0, h], [w, h],
         [w // 2, 0], [w // 2, h], [0, h // 2], [w, h // 2]],
        dtype=np.float64,
    )
    all_src = np.vstack([ctrl_src, corners])
    all_dst = np.vstack([ctrl_dst, corners])

    # Inverse displacement field: dst -> src, so cv2.remap can sample
    # from the source image at the correct location.
    dx = all_src[:, 0] - all_dst[:, 0]
    dy = all_src[:, 1] - all_dst[:, 1]
    rbf_x = RBFInterpolator(all_dst, dx, kernel="thin_plate_spline", smoothing=1.0)
    rbf_y = RBFInterpolator(all_dst, dy, kernel="thin_plate_spline", smoothing=1.0)

    # Sample the displacement field on a downsampled grid and upscale
    # for performance (the TPS field is smooth so downsampling is lossless).
    step_grid = 4
    gy, gx = np.mgrid[0:h:step_grid, 0:w:step_grid].astype(np.float32)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    disp_x_lo = rbf_x(pts).reshape(gx.shape).astype(np.float32)
    disp_y_lo = rbf_y(pts).reshape(gx.shape).astype(np.float32)
    map_x_lo = (gx + disp_x_lo).astype(np.float32)
    map_y_lo = (gy + disp_y_lo).astype(np.float32)
    map_x = cv2.resize(map_x_lo, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_lo, (w, h), interpolation=cv2.INTER_LINEAR)

    return cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )

