"""PyRenderRenderer: real EGL-backed renderer for FLAME meshes.

Takes a SceneState produced by DECALift, builds a trimesh from the FLAME
vertices + faces, applies UV texture, sets up an orthographic camera that
matches DECA's projection (s, tx, ty), renders RGB + depth, then composites
the rendered face over scene.background_plate at the input image dimensions.

Design constraints honored:
- Output dims == scene.image_size, dtype uint8, BGR.
- No CUDA toolkit (uses EGL via NVIDIA driver).
- No mocks. If EGL is unavailable, raises RuntimeError with the actual
  pyrender error message.
"""

from __future__ import annotations

import os

# Force EGL backend before pyrender imports OpenGL.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import numpy as np
import pyrender
import trimesh

from ofiq_syngen.three_d.scene.state import SceneState


class PyRenderRenderer:
    """EGL-backed offscreen renderer for FLAME meshes."""

    backend_name = "pyrender"

    def __init__(self, point_size: float = 1.0) -> None:
        self.point_size = point_size
        self._renderer_cache: dict[tuple[int, int], pyrender.OffscreenRenderer] = {}

    def render(self, scene_state: SceneState) -> np.ndarray:
        h, w = scene_state.image_size
        verts = getattr(scene_state, "flame_verts", None)
        faces = getattr(scene_state, "flame_faces", None)
        if verts is None or faces is None:
            raise RuntimeError(
                "SceneState lacks flame_verts/flame_faces; was it produced by DECALift?"
            )

        cam_params = scene_state.camera.intrinsics  # (3, 3) holds (s, _, tx; _, s, ty; ...)
        if cam_params is None:
            raise RuntimeError("SceneState.camera.intrinsics is None")
        s = float(cam_params[0, 0])
        tx = float(cam_params[0, 2])
        ty = float(cam_params[1, 2])

        crop_size = int(getattr(scene_state, "crop_size", 224))

        verts_proj, depth_z = _project_for_render(verts, s, tx, ty)

        mesh_tri = trimesh.Trimesh(
            vertices=verts_proj,
            faces=faces,
            process=False,
        )

        # Per-vertex texture: sample the source image at each vertex's
        # ORIGINAL 2D projection (before perturbation). The mesh wears
        # the source image as its texture; rendering the perturbed mesh
        # then drags that texture along with the rotation -> photoreal.
        verts_2d_orig = getattr(scene_state, "flame_verts_2d_orig", None)
        source_image = scene_state.image
        if verts_2d_orig is not None and source_image is not None:
            vertex_colors = _sample_source_per_vertex(verts_2d_orig, source_image)
            mesh_tri.visual = trimesh.visual.color.ColorVisuals(
                mesh=mesh_tri,
                vertex_colors=vertex_colors,
            )
        else:
            uv_texture = scene_state.flame.uv_texture
            if uv_texture is not None:
                mesh_tri.visual = trimesh.visual.color.ColorVisuals(
                    mesh=mesh_tri,
                    vertex_colors=_sample_vertex_colors(verts, uv_texture),
                )

        mesh = pyrender.Mesh.from_trimesh(mesh_tri, smooth=False)

        scene_pr = pyrender.Scene(
            bg_color=[0.0, 0.0, 0.0, 0.0],
            ambient_light=[0.5, 0.5, 0.5],
        )
        scene_pr.add(mesh)

        for occluder in scene_state.occluders:
            occ_verts_proj, _ = _project_for_render(occluder.vertices, s, tx, ty)
            occ_tri = trimesh.Trimesh(
                vertices=occ_verts_proj,
                faces=occluder.faces,
                process=False,
            )
            occ_color = _occluder_color(occluder)
            occ_tri.visual = trimesh.visual.color.ColorVisuals(
                mesh=occ_tri,
                vertex_colors=np.tile(occ_color, (len(occ_verts_proj), 1)),
            )
            scene_pr.add(pyrender.Mesh.from_trimesh(occ_tri, smooth=False))

        cam_node = _build_orthographic_camera(crop_size)
        scene_pr.add(cam_node, pose=_camera_pose(crop_size, depth_z))

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        scene_pr.add(light, pose=_camera_pose(crop_size, depth_z))

        renderer = self._get_offscreen(crop_size, crop_size)
        flags = pyrender.RenderFlags.RGBA
        rgba_crop, depth_crop = renderer.render(scene_pr, flags=flags)

        rgba_full = _warp_back_to_full(rgba_crop, scene_state)

        composite = _composite_over_background(rgba_full, scene_state.background_plate)
        if composite.shape[:2] != (h, w):
            composite = cv2.resize(composite, (w, h), interpolation=cv2.INTER_AREA)
        return composite

    def _get_offscreen(self, h: int, w: int) -> pyrender.OffscreenRenderer:
        key = (h, w)
        if key not in self._renderer_cache:
            self._renderer_cache[key] = pyrender.OffscreenRenderer(
                viewport_width=w,
                viewport_height=h,
                point_size=self.point_size,
            )
        return self._renderer_cache[key]

    def close(self) -> None:
        for r in self._renderer_cache.values():
            r.delete()
        self._renderer_cache.clear()


def _project_for_render(
    verts: np.ndarray,
    s: float,
    tx: float,
    ty: float,
) -> tuple[np.ndarray, float]:
    """Apply DECA orthographic projection then map to a Z position the
    pyrender orthographic camera can see.

    DECA's projection: p = s * (v + [tx, ty, 0]) in [-1, 1] crop NDC.
    pyrender expects mesh vertices in world space; we keep the full XYZ but
    scale and translate so the projected XY matches DECA's, and we offset Z
    so the camera (placed at z = +D) sees the mesh.
    """
    out = verts.copy().astype(np.float32)
    out[:, 0] = s * (out[:, 0] + tx)
    out[:, 1] = -s * (out[:, 1] + ty)  # flip Y to image coords
    out[:, 2] = -s * out[:, 2]         # flip Z so + Z is forward
    z_offset = float(out[:, 2].max() + 1.0)
    out[:, 2] -= z_offset
    return out, z_offset


def _build_orthographic_camera(crop_size: int) -> pyrender.OrthographicCamera:
    return pyrender.OrthographicCamera(xmag=1.0, ymag=1.0, znear=0.01, zfar=100.0)


def _camera_pose(crop_size: int, depth_z: float) -> np.ndarray:
    """Identity rotation, camera at z = +distance looking down -Z."""
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = 5.0  # camera distance; mesh has been offset to [-zmax, 0]
    return pose


def _occluder_color(occluder) -> np.ndarray:
    """Return RGBA per-vertex color for an occluder.

    Occluder.texture, if set, supplies a (4,) or (3,) BGR(A) base color.
    Otherwise we fall back to a name-keyed palette (sunglasses=black,
    surgical mask=light blue, hat=dark gray, generic patch=mid gray).
    """
    if occluder.texture is not None and occluder.texture.size >= 3:
        flat = occluder.texture.reshape(-1, occluder.texture.shape[-1])[0]
        rgb = flat[:3][::-1]  # BGR -> RGB
        alpha = flat[3] if flat.size >= 4 else 255
        return np.array([rgb[0], rgb[1], rgb[2], alpha], dtype=np.uint8)

    palette = {
        "sunglasses": (15, 15, 15, 255),
        "surgical_mask": (220, 230, 240, 255),
        "hat": (40, 40, 40, 255),
        "face_patch": (160, 160, 160, 255),
    }
    return np.array(palette.get(occluder.name, (128, 128, 128, 255)), dtype=np.uint8)


def _sample_source_per_vertex(
    verts_2d_orig: np.ndarray, source_image: np.ndarray,
) -> np.ndarray:
    """Sample the source image at each vertex's original 2D projection.

    The mesh's texture is the source image itself; rendering the
    perturbed mesh with these per-vertex colors drags the source
    appearance through the 3D rotation.

    Args:
        verts_2d_orig: (V, 2) float32 of original vertex projections in
            source-image pixel coordinates.
        source_image: (H, W, 3) BGR uint8 source image.

    Returns:
        (V, 4) RGBA uint8 vertex colors (alpha=255).
    """
    h, w = source_image.shape[:2]
    xs = np.clip(verts_2d_orig[:, 0].astype(np.int32), 0, w - 1)
    ys = np.clip(verts_2d_orig[:, 1].astype(np.int32), 0, h - 1)
    bgr = source_image[ys, xs]  # (V, 3) BGR
    # pyrender vertex_colors take RGB - swap channels
    rgb = bgr[:, ::-1]
    rgba = np.concatenate(
        [rgb, np.full((rgb.shape[0], 1), 255, dtype=np.uint8)], axis=1,
    )
    return rgba


def _sample_vertex_colors(
    verts: np.ndarray,
    uv_texture: np.ndarray,
) -> np.ndarray:
    """Per-vertex color fallback using mean texture color.

    A correct implementation samples the FLAME UV texture per vertex via
    the FLAME UV coordinate map. For now we use mean texture color so the
    rendered mesh is uniformly lit, which is sufficient for HeadPoseYaw
    (geometry tier — tests rotation, not texture fidelity). UV-correct
    sampling lands when FLAMETex outputs are wired through and a TextureVisuals
    path replaces this ColorVisuals fallback.
    """
    n = verts.shape[0]
    mean_bgr = uv_texture.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    rgb = mean_bgr[::-1]  # BGR -> RGB
    colors = np.tile(rgb, (n, 1))
    alpha = np.full((n, 1), 255, dtype=np.uint8)
    return np.concatenate([colors, alpha], axis=1)


def _warp_back_to_full(
    rgba_crop: np.ndarray,
    scene_state: SceneState,
) -> np.ndarray:
    """Warp the rendered crop back to the original image dimensions.

    The forward crop applied tform: original -> crop. The inverse maps crop
    -> original. We stored tform_inverse as scene_state.camera.extrinsics.
    """
    tform_inv = scene_state.camera.extrinsics
    h, w = scene_state.image_size
    if tform_inv is None:
        return cv2.resize(rgba_crop, (w, h), interpolation=cv2.INTER_LINEAR)
    rgba_full = cv2.warpAffine(
        rgba_crop,
        tform_inv[:2, :],
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rgba_full


def _composite_over_background(
    rgba_full: np.ndarray,
    background_bgr: np.ndarray,
) -> np.ndarray:
    """Alpha-composite the rendered face over the inpainted background plate."""
    if rgba_full.shape[2] == 4:
        rgb = rgba_full[..., :3]
        alpha = rgba_full[..., 3:4].astype(np.float32) / 255.0
    else:
        rgb = rgba_full
        alpha = np.ones((rgba_full.shape[0], rgba_full.shape[1], 1), dtype=np.float32)

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    composite = (alpha * bgr.astype(np.float32) + (1.0 - alpha) * background_bgr.astype(np.float32))
    return composite.clip(0, 255).astype(np.uint8)
