"""Geometry-tier perturbations: 3D rotations, jaw, expression, framing.

Each function takes (img, severity, seed, scene) and returns BGR uint8 with
shape == img.shape. Severity in [0, 1]. Deterministic given seed.

Three operator families in this module:

1. Pose rotations — modify scene.flame.pose (axis-angle) AND rotate
   scene.flame_verts in 3D so the renderer projects the rotated mesh.
   Pose vector layout (DECA convention):
       pose[0:3] = global rotation (axis-angle), [pitch, yaw, roll]
       pose[3:6] = jaw rotation (axis-angle)

2. Camera transforms — mutate scene.camera.intrinsics where (s, tx, ty) are
   stored at [0,0], [0,2], [1,2]. DECA's orthographic projection:
       p = s * (v[:,:2] + [tx, ty])
   Crop/margin shifts the face by translating (tx, ty); HeadSize and
   InterEyeDistance shrink the face by reducing s.

3. Expression — mutate scene.flame.expression (50-d FLAME coefficients).
   Perturbed coefficients flow through FLAME's forward pass, so we re-run
   the FLAME decoder to regenerate vertices before render.

The pipeline binds the renderer at startup via set_renderer().
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ofiq_syngen.three_d.registry import register
from ofiq_syngen.three_d.render import Renderer
from ofiq_syngen.three_d.scene.state import SceneState


_RENDERER: Optional[Renderer] = None


def set_renderer(renderer: Renderer) -> None:
    global _RENDERER
    _RENDERER = renderer


def _require_renderer() -> Renderer:
    if _RENDERER is None:
        raise RuntimeError(
            "No renderer is bound. Build a DegradationPipeline first; "
            "it sets the renderer via three_d_syn.perturb.geometry.set_renderer()."
        )
    return _RENDERER


def _require_scene(scene: Optional[SceneState], component: str) -> SceneState:
    if scene is None:
        raise RuntimeError(
            f"{component} requires a SceneState. The pipeline must lift the "
            "image before dispatching this perturbation."
        )
    return scene


def _finalize(out: np.ndarray, img: np.ndarray) -> np.ndarray:
    if out.shape != img.shape:
        import cv2
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    if out.dtype != np.uint8:
        out = out.clip(0, 255).astype(np.uint8)
    return out


def _rotate_verts(verts: np.ndarray, axis: int, angle_rad: float) -> np.ndarray:
    """Rotate (V, 3) vertices about a principal axis."""
    c, s = float(np.cos(angle_rad)), float(np.sin(angle_rad))
    R = np.eye(3, dtype=np.float32)
    if axis == 0:
        R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    elif axis == 1:
        R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c
    elif axis == 2:
        R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    else:
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")
    return (verts.astype(np.float32) @ R.T)


def _clone_for_perturb(scene: SceneState) -> SceneState:
    perturbed = scene.clone()
    if hasattr(scene, "flame_verts"):
        perturbed.flame_verts = scene.flame_verts.copy()  # type: ignore[attr-defined]
    if hasattr(scene, "flame_faces"):
        perturbed.flame_faces = scene.flame_faces  # type: ignore[attr-defined]
    if hasattr(scene, "crop_size"):
        perturbed.crop_size = scene.crop_size  # type: ignore[attr-defined]
    # Carry the original 2D vertex projections through the perturbation
    # so the renderer can sample per-vertex texture from the source.
    if hasattr(scene, "flame_verts_2d_orig"):
        perturbed.flame_verts_2d_orig = scene.flame_verts_2d_orig.copy()  # type: ignore[attr-defined]
    if hasattr(scene, "flame_landmarks_3d"):
        perturbed.flame_landmarks_3d = scene.flame_landmarks_3d.copy()  # type: ignore[attr-defined]
    if hasattr(scene, "flame_landmarks_2d"):
        perturbed.flame_landmarks_2d = scene.flame_landmarks_2d.copy()  # type: ignore[attr-defined]
    return perturbed


# =========================================================================
# Pose-axis rotations (yaw, pitch, roll)
# =========================================================================

def _apply_pose_rotation(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState],
    axis: int,
    max_deg: float,
    component: str,
) -> np.ndarray:
    scene = _require_scene(scene, component)
    if scene.flame.pose is None or scene.flame.pose.size < 3:
        raise RuntimeError("FLAME pose vector missing; lift backend produced an incomplete scene.")

    # OFIQ HeadPoseYaw / Pitch / Roll scalars all use sigmoid(|angle|),
    # so the operator must rotate AWAY from upright to degrade. If we
    # pick a random direction we'll cancel the source pose half the time
    # (e.g., source yaw=+22deg, random=-1 -> rotated toward 0 -> scalar
    # IMPROVES). Detect source pose sign on this axis and rotate in the
    # same direction; only when source is near-upright (|pose|<0.05 rad
    # ~ 3deg) do we fall back to seed-deterministic random direction.
    src_pose = float(scene.flame.pose[axis])
    if abs(src_pose) >= 0.05:
        direction = 1.0 if src_pose >= 0 else -1.0
    else:
        rng = np.random.RandomState(seed)
        direction = float(rng.choice([-1, 1]))
    angle_deg = float(severity) * max_deg * direction
    angle_rad = np.deg2rad(angle_deg)

    perturbed = _clone_for_perturb(scene)
    perturbed.flame.pose[axis] += angle_rad

    if hasattr(perturbed, "flame_verts"):
        perturbed.flame_verts = _rotate_verts(  # type: ignore[attr-defined]
            perturbed.flame_verts, axis=axis, angle_rad=angle_rad,  # type: ignore[attr-defined]
        )

    out = _require_renderer().render(perturbed)
    return _finalize(out, img)


def _head3d_pose_yaw(img, severity, seed, scene):
    """Real 3D yaw via FLAME global pose [§7.4.11.2]. Severity 0->1 maps 0deg->35deg."""
    return _apply_pose_rotation(img, severity, seed, scene, axis=1, max_deg=35.0,
                                component="HeadPoseYaw.scalar")


def _head3d_pose_pitch(img, severity, seed, scene):
    """Real 3D pitch via FLAME global pose [§7.4.11.3]. Severity 0->1 maps 0deg->25deg."""
    return _apply_pose_rotation(img, severity, seed, scene, axis=0, max_deg=25.0,
                                component="HeadPosePitch.scalar")


def _head3d_pose_roll(img, severity, seed, scene):
    """Real 3D roll via FLAME global pose [§7.4.11.4]. Severity 0->1 maps 0deg->30deg."""
    return _apply_pose_rotation(img, severity, seed, scene, axis=2, max_deg=30.0,
                                component="HeadPoseRoll.scalar")


# =========================================================================
# Camera translations (crop / margin)
# =========================================================================

def _apply_camera_shift(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState],
    component: str,
    delta_x: float = 0.0,
    delta_y: float = 0.0,
) -> np.ndarray:
    """Shift the face in image space via DECA's orthographic camera (tx, ty).

    DECA projection: p = s * (v[:,:2] + [tx, ty]). Increasing tx moves the
    projected face right; increasing ty moves it down (image y grows
    downward in DECA's NDC after our renderer's Y-flip).
    """
    scene = _require_scene(scene, component)
    perturbed = _clone_for_perturb(scene)
    intr = perturbed.camera.intrinsics
    if intr is None:
        raise RuntimeError("Scene camera intrinsics missing; cannot apply camera shift.")
    intr[0, 2] += float(severity) * delta_x
    intr[1, 2] += float(severity) * delta_y

    out = _require_renderer().render(perturbed)
    return _finalize(out, img)


def _leftward_crop(img, severity, seed, scene):
    """Face shifts toward the LEFT image edge [§7.4.10.1]. Severity 0->1: tx 0->-0.4."""
    return _apply_camera_shift(img, severity, seed, scene,
                               "LeftwardCropOfTheFaceImage.scalar", delta_x=-0.4)


def _rightward_crop(img, severity, seed, scene):
    """Face shifts toward the RIGHT image edge [§7.4.10.2]. Severity 0->1: tx 0->+0.4."""
    return _apply_camera_shift(img, severity, seed, scene,
                               "RightwardCropOfTheFaceImage.scalar", delta_x=+0.4)


def _margin_above(img, severity, seed, scene):
    """Face shifts UP (less margin above) [§7.4.10.3]. Severity 0->1: ty 0->+0.4 (NDC)."""
    return _apply_camera_shift(img, severity, seed, scene,
                               "MarginAboveOfTheFaceImage.scalar", delta_y=+0.4)


def _margin_below(img, severity, seed, scene):
    """Face shifts DOWN (less margin below) [§7.4.10.4]. Severity 0->1: ty 0->-0.4 (NDC)."""
    return _apply_camera_shift(img, severity, seed, scene,
                               "MarginBelowOfTheFaceImage.scalar", delta_y=-0.4)


# =========================================================================
# Camera scale (head size, inter-eye distance)
# =========================================================================

def _apply_camera_scale(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState],
    component: str,
    min_factor: float = 0.3,
) -> np.ndarray:
    """Shrink the face in frame by reducing camera scale s.

    Severity 0 leaves s unchanged. Severity 1 multiplies s by min_factor.
    """
    scene = _require_scene(scene, component)
    perturbed = _clone_for_perturb(scene)
    intr = perturbed.camera.intrinsics
    if intr is None:
        raise RuntimeError("Scene camera intrinsics missing; cannot apply camera scale.")
    factor = 1.0 - float(severity) * (1.0 - min_factor)
    intr[0, 0] *= factor
    intr[1, 1] *= factor

    out = _require_renderer().render(perturbed)
    return _finalize(out, img)


def _head_size(img, severity, seed, scene):
    """Reduce head size in frame [§7.4.9]. Severity 0->1: scale 1.0->0.3."""
    return _apply_camera_scale(img, severity, seed, scene, "HeadSize.scalar", min_factor=0.3)


def _inter_eye_distance(img, severity, seed, scene):
    """Reduce IED by shrinking the face [§7.4.8]. Severity 0->1: scale 1.0->0.3."""
    return _apply_camera_scale(img, severity, seed, scene, "InterEyeDistance.scalar", min_factor=0.3)


# =========================================================================
# Jaw and expression
# =========================================================================

def _mouth_closed(img, severity, seed, scene):
    """Open the jaw via FLAME jaw pose [§7.4.4]. Severity 0->1 maps 0->30deg jaw drop.

    Degrades MouthClosed.scalar (high score = closed, low = open). FLAME jaw
    is pose[3:6] in axis-angle radians; jaw open = positive rotation about
    the X axis -> pose[3].

    Because the jaw rotation propagates through FLAME's linear blend skinning,
    we rerun the FLAME decoder to regenerate vertices. Falls back to ignoring
    the jaw if the lift backend (e.g. mock) did not provide a callable
    decoder reference.
    """
    scene = _require_scene(scene, "MouthClosed.scalar")
    angle_rad = np.deg2rad(float(severity) * 30.0)

    perturbed = _clone_for_perturb(scene)
    if perturbed.flame.pose is None or perturbed.flame.pose.size < 6:
        raise RuntimeError("FLAME pose vector lacks jaw axis; lift backend produced incomplete scene.")
    perturbed.flame.pose[3] += angle_rad

    flame_module = getattr(scene, "flame_module", None)
    if flame_module is not None and hasattr(scene, "flame_verts"):
        verts = _redecode_flame(flame_module, perturbed.flame)
        perturbed.flame_verts = verts  # type: ignore[attr-defined]

    out = _require_renderer().render(perturbed)
    return _finalize(out, img)


def _expression_neutrality(img, severity, seed, scene):
    """Add expression amplitude to drive the face away from neutral [§7.4.12].

    Degrades ExpressionNeutrality.scalar (high = neutral, low = expressive).
    Adds Gaussian noise to FLAME expression coefficients with stddev scaled
    by severity. Re-decodes FLAME to regenerate vertices.
    """
    scene = _require_scene(scene, "ExpressionNeutrality.scalar")
    rng = np.random.RandomState(seed)

    perturbed = _clone_for_perturb(scene)
    if perturbed.flame.expression is None:
        raise RuntimeError(
            "FLAME expression vector missing; lift backend produced an incomplete scene."
        )
    noise = rng.randn(*perturbed.flame.expression.shape).astype(np.float32)
    perturbed.flame.expression += float(severity) * 2.0 * noise

    flame_module = getattr(scene, "flame_module", None)
    if flame_module is not None and hasattr(scene, "flame_verts"):
        verts = _redecode_flame(flame_module, perturbed.flame)
        perturbed.flame_verts = verts  # type: ignore[attr-defined]

    out = _require_renderer().render(perturbed)
    return _finalize(out, img)


def _redecode_flame(flame_module, params) -> np.ndarray:
    """Run FLAME forward pass with mutated params to regenerate (V, 3) vertices."""
    import torch
    device = next(flame_module.parameters()).device

    def _to_tensor(arr):
        return torch.from_numpy(arr).float().unsqueeze(0).to(device)

    with torch.no_grad():
        verts, _, _ = flame_module(
            shape_params=_to_tensor(params.shape),
            expression_params=_to_tensor(params.expression),
            pose_params=_to_tensor(params.pose),
        )
    return verts[0].cpu().numpy()


# =========================================================================
# Registration
# =========================================================================

register("HeadPoseYaw.scalar", _head3d_pose_yaw,
         "3D::FLAME global yaw rotation [§7.4.11.2]", "yaw: 0deg -> 35deg",
         tier="geometry")
register("HeadPosePitch.scalar", _head3d_pose_pitch,
         "3D::FLAME global pitch rotation [§7.4.11.3]", "pitch: 0deg -> 25deg",
         tier="geometry")
register("HeadPoseRoll.scalar", _head3d_pose_roll,
         "3D::FLAME global roll rotation [§7.4.11.4]", "roll: 0deg -> 30deg",
         tier="geometry")

register("LeftwardCropOfTheFaceImage.scalar", _leftward_crop,
         "3D::camera tx shift toward left edge [§7.4.10.1]", "tx: 0 -> -0.4 NDC",
         tier="geometry")
register("RightwardCropOfTheFaceImage.scalar", _rightward_crop,
         "3D::camera tx shift toward right edge [§7.4.10.2]", "tx: 0 -> +0.4 NDC",
         tier="geometry")
register("MarginAboveOfTheFaceImage.scalar", _margin_above,
         "3D::camera ty shift up [§7.4.10.3]", "ty: 0 -> +0.4 NDC",
         tier="geometry")
register("MarginBelowOfTheFaceImage.scalar", _margin_below,
         "3D::camera ty shift down [§7.4.10.4]", "ty: 0 -> -0.4 NDC",
         tier="geometry")

register("HeadSize.scalar", _head_size,
         "3D::camera scale shrink [§7.4.9]", "scale: 1.0 -> 0.3",
         tier="geometry")
register("InterEyeDistance.scalar", _inter_eye_distance,
         "3D::camera scale shrink [§7.4.8]", "scale: 1.0 -> 0.3",
         tier="geometry")

register("MouthClosed.scalar", _mouth_closed,
         "3D::FLAME jaw rotation [§7.4.4]", "jaw: 0deg -> 30deg",
         tier="geometry")
register("ExpressionNeutrality.scalar", _expression_neutrality,
         "3D::FLAME expression coeff perturbation [§7.4.12]", "stddev: 0 -> 2.0",
         tier="geometry")
