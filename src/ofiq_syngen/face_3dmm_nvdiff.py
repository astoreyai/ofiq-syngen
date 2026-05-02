"""nvdiffrast-based dense 3DMM head pose re-rendering.

Replaces the TPS approximation with proper differentiable rasterization:
per-pixel z-buffer, sub-pixel triangle coverage, true texture sampling.
This eliminates the residual artifacts (hair drift, foreshortening
errors, occlusion smearing) that the TPS approach cannot fix.

Pipeline:
    1. Load dense BFM mesh (38K vertices, 60K valid triangles).
    2. Compute source 3D vertices and project to source 2D positions
       (these become the "UV map" pointing into the source image).
    3. Compute rotated 3D vertices via R_extra @ R.
    4. nvdiffrast rasterizes the destination mesh -> triangle_id +
       barycentric coords per pixel (with depth test).
    5. dr.interpolate barycentric coords with source 2D positions ->
       per-pixel source pixel locations.
    6. dr.texture samples the source image at those locations.
    7. Composite over the original image so hair / background outside
       the face stay put.

Requires:
    nvdiffrast + torch with CUDA. Falls back to TPS renderer if
    unavailable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from ofiq_syngen.face_3dmm import parse_3ddfa_params
from ofiq_syngen.face_3dmm_dense import (
    _compute_mesh_3d, _load_bfm_dense, _yaw_matrix, _pitch_matrix,
)

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


_DR_CTX = None  # lazy-init nvdiffrast context


def is_nvdiff_available() -> bool:
    """Whether nvdiffrast + CUDA torch are usable."""
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        import nvdiffrast.torch  # noqa: F401
        return True
    except ImportError:
        return False


def _get_dr_context():
    """Lazy-init the nvdiffrast CUDA rasterization context."""
    global _DR_CTX
    if _DR_CTX is None:
        import nvdiffrast.torch as dr
        _DR_CTX = dr.RasterizeCudaContext()
    return _DR_CTX


def render_pose_nvdiff(
    img: np.ndarray,
    ctx: FaceContext,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Re-render the face at a new yaw / pitch using nvdiffrast.

    Args:
        img: source BGR image.
        ctx: FaceContext with raw_3ddfa_params populated.
        yaw_deg: additional yaw rotation in degrees.
        pitch_deg: additional pitch rotation in degrees.
        seed: unused (kept for API consistency).

    Returns:
        BGR image with the face area re-rendered at the new pose,
        composited over the original (so hair / background outside
        the face mask are unchanged).
    """
    if not hasattr(ctx, "raw_3ddfa_params") or ctx.raw_3ddfa_params is None:
        return img
    if abs(yaw_deg) < 0.1 and abs(pitch_deg) < 0.1:
        return img

    import torch
    import nvdiffrast.torch as dr

    R, offset, alpha_shp, alpha_exp = parse_3ddfa_params(ctx.raw_3ddfa_params)
    h, w = img.shape[:2]

    # 3D meshes at source and rotated poses (in 120x120 BFM space)
    verts_src_3d = _compute_mesh_3d(R, offset, alpha_shp, alpha_exp)
    R_extra = _yaw_matrix(yaw_deg) @ _pitch_matrix(pitch_deg)
    verts_dst_3d = _compute_mesh_3d(R_extra @ R, offset, alpha_shp, alpha_exp)

    # Scale projections back to original image pixel space
    sx, sy = w / 120.0, h / 120.0
    verts_src_2d = verts_src_3d[:, :2] * np.array([sx, sy])
    verts_dst_2d_xy = verts_dst_3d[:, :2] * np.array([sx, sy])

    # Convert destination positions to nvdiffrast clip space:
    #   x_clip = (x_pixel / w) * 2 - 1
    #   y_clip = -((y_pixel / h) * 2 - 1)  (image y is top-down, clip is bottom-up)
    #   z is preserved for depth test, w = 1 (orthographic)
    dst_x_clip = (verts_dst_2d_xy[:, 0] / w) * 2.0 - 1.0
    dst_y_clip = -((verts_dst_2d_xy[:, 1] / h) * 2.0 - 1.0)
    # Normalize z to a reasonable range for the depth buffer
    dst_z = verts_dst_3d[:, 2]
    z_min, z_max = dst_z.min(), dst_z.max()
    dst_z_clip = (dst_z - z_min) / max(z_max - z_min, 1e-6) * 0.8 + 0.1

    verts_clip = np.column_stack([
        dst_x_clip, dst_y_clip, dst_z_clip, np.ones_like(dst_z_clip),
    ]).astype(np.float32)

    # Source 2D positions in pixel space (NOT normalized): we attach
    # these as per-vertex "attributes" and have nvdiffrast interpolate
    # them barycentrically per pixel, then use cv2.remap to sample the
    # source image at the resulting pixel coordinates. This avoids the
    # OpenGL-vs-OpenCV UV convention mismatch.
    src_uv = verts_src_2d.astype(np.float32)  # (V, 2) pixel coords

    bfm = _load_bfm_dense()
    tri_all = bfm["tri"].T.astype(np.int32)  # (T, 3)

    # Backface culling: drop triangles whose normal points away from
    # the camera in the destination pose. nvdiffrast does not cull
    # automatically and back-facing triangles otherwise leak through
    # (showing the back of the head over the face).
    tri_verts = verts_dst_3d[tri_all]  # (T, 3, 3)
    e1 = tri_verts[:, 1] - tri_verts[:, 0]
    e2 = tri_verts[:, 2] - tri_verts[:, 0]
    normals_z = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    front_mask = normals_z < 0  # CCW vertices in screen space
    tri = tri_all[front_mask]

    # Move to CUDA tensors
    device = "cuda"
    verts_t = torch.from_numpy(verts_clip).unsqueeze(0).to(device).contiguous()
    tri_t = torch.from_numpy(tri).to(device).contiguous()
    uv_t = torch.from_numpy(src_uv).unsqueeze(0).to(device).contiguous()

    dr_ctx = _get_dr_context()
    # Rasterize destination mesh at image resolution. nvdiffrast uses
    # OpenGL Y-up convention so the rasterized buffer is bottom-up; we
    # flip vertically when converting back to numpy below.
    rast, _ = dr.rasterize(dr_ctx, verts_t, tri_t, resolution=[h, w])

    # Per-pixel barycentric interpolation of the SOURCE pixel coordinates
    # we attached as vertex attributes -> per-pixel (src_x, src_y) lookup.
    pix_src, _ = dr.interpolate(uv_t, rast, tri_t)

    # Mask: where the destination mesh covers the pixel
    triangle_id = rast[..., 3]
    face_mask_t = (triangle_id > 0).float()

    # Pull back to numpy and flip Y (OpenGL bottom-up -> numpy top-down)
    map_xy = pix_src.squeeze(0).cpu().numpy()  # (H, W, 2)
    map_xy = np.ascontiguousarray(map_xy[::-1])  # flip Y
    mask = face_mask_t.squeeze(0).cpu().numpy()
    mask = np.ascontiguousarray(mask[::-1]).astype(np.uint8) * 255

    # Sample source image at the per-pixel source coordinates
    map_x = map_xy[..., 0].astype(np.float32)
    map_y = map_xy[..., 1].astype(np.float32)
    face_bgr = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT,
    )

    # Composite the rendered face over the original image with a
    # feathered alpha so hair / background blend invisibly.
    feather = max(4.0, ctx.t_metric / 25.0)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather)
    soft = np.clip(soft / 255.0, 0.0, 1.0)[..., None]
    out = img.astype(np.float32) * (1 - soft) + face_bgr.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8)
