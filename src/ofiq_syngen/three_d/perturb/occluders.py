"""Occluder-tier perturbations: 3D occluder meshes attached to FLAME landmarks.

Each function builds a procedural occluder mesh in FLAME mesh space, attaches
it to the relevant landmark region, and appends it to scene.occluders before
rendering. The renderer iterates scene.occluders and composes them with the
FLAME mesh under pyrender's depth buffer — depth ordering happens for free.

Occluder geometry is built with trimesh primitives (no external assets, no
license). Severity drives occluder SIZE (and therefore pixel coverage), since
OFIQ measures occlusion as a coverage fraction.

Attachment uses iBUG 68-point 3D landmarks stored on scene.flame_landmarks_3d
by DECALift. Indices follow the canonical iBUG-68 layout:
    eyes:        right corners 36-41, left corners 42-47
    nose tip:    30
    mouth:       48-67 (outer 48-59, inner 60-67)

These are stable across FLAME 2020 — they come from FLAME's own landmark
embedding, not from any 2D detector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ofiq_syngen.three_d.registry import register
from ofiq_syngen.three_d.render import Renderer
from ofiq_syngen.three_d.scene.state import Occluder, SceneState


_RENDERER: Optional[Renderer] = None


def set_renderer(renderer: Renderer) -> None:
    global _RENDERER
    _RENDERER = renderer


def _require_renderer() -> Renderer:
    if _RENDERER is None:
        raise RuntimeError(
            "No renderer is bound. Build a DegradationPipeline first; "
            "it sets the renderer via three_d_syn.perturb.occluders.set_renderer()."
        )
    return _RENDERER


def _require_scene_with_landmarks(
    scene: Optional[SceneState],
    component: str,
) -> SceneState:
    if scene is None:
        raise RuntimeError(
            f"{component} requires a SceneState. Pipeline must lift the image first."
        )
    landmarks = getattr(scene, "flame_landmarks_3d", None)
    if landmarks is None or landmarks.size == 0:
        raise RuntimeError(
            f"{component} requires scene.flame_landmarks_3d (iBUG-68 in FLAME space). "
            "DECALift should populate this; lift backend appears incomplete."
        )
    return scene


def _finalize(out: np.ndarray, img: np.ndarray) -> np.ndarray:
    if out.shape != img.shape:
        import cv2
        out = cv2.resize(out, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    if out.dtype != np.uint8:
        out = out.clip(0, 255).astype(np.uint8)
    return out


def _clone_with_3d_attrs(scene: SceneState) -> SceneState:
    perturbed = scene.clone()
    for attr in ("flame_verts", "flame_faces", "flame_landmarks_3d",
                 "flame_landmarks_2d", "crop_size", "flame_module"):
        if hasattr(scene, attr):
            setattr(perturbed, attr, getattr(scene, attr))
    return perturbed


# =========================================================================
# Mesh builders (trimesh primitives in FLAME mesh space)
# =========================================================================

def _build_glasses_mesh(
    landmarks_3d: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Sunglasses: two ellipsoid lenses + bridge bar.

    Lenses sit at the iBUG eye-region centers (avg of corner landmarks).
    Severity scales lens radii (severity=0 -> tiny invisible, =1 -> full coverage).

    Returns (vertices (V, 3), faces (F, 3) int32).
    """
    import trimesh

    right_eye = landmarks_3d[36:42].mean(axis=0)
    left_eye = landmarks_3d[42:48].mean(axis=0)
    eye_span = float(np.linalg.norm(left_eye - right_eye))

    base_r = max(eye_span * 0.30, 0.01)
    lens_radius_x = base_r * (0.4 + 0.6 * float(severity))
    lens_radius_y = lens_radius_x * 0.7
    lens_thickness = lens_radius_x * 0.15

    def _disc(center: np.ndarray) -> trimesh.Trimesh:
        cylinder = trimesh.creation.cylinder(
            radius=1.0, height=lens_thickness, sections=24,
        )
        cylinder.apply_scale([lens_radius_x, lens_radius_y, 1.0])
        cylinder.apply_translation(center)
        return cylinder

    right_lens = _disc(right_eye)
    left_lens = _disc(left_eye)

    bridge_center = (right_eye + left_eye) / 2.0
    bridge = trimesh.creation.box(extents=[eye_span * 0.25, lens_thickness * 1.5, lens_thickness])
    bridge.apply_translation(bridge_center)

    glasses = trimesh.util.concatenate([right_lens, left_lens, bridge])
    return glasses.vertices.astype(np.float32), glasses.faces.astype(np.int32)


def _build_mask_mesh(
    landmarks_3d: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Surgical mask: extruded oval covering nose + mouth.

    Spans from below-eyes to chin; width scales with severity to control
    coverage of the lower face.
    """
    import trimesh

    nose_tip = landmarks_3d[30]
    mouth_outer = landmarks_3d[48:60]
    chin = landmarks_3d[8]
    face_y_top = float((landmarks_3d[27] + nose_tip)[1] / 2.0)
    face_y_bot = float(chin[1])
    face_x_left = float(landmarks_3d[3, 0])
    face_x_right = float(landmarks_3d[13, 0])

    cx = (face_x_left + face_x_right) / 2.0
    cy = (face_y_top + face_y_bot) / 2.0
    cz = float(mouth_outer[:, 2].mean())

    half_w = (face_x_right - face_x_left) / 2.0 * (0.4 + 0.6 * float(severity))
    half_h = (face_y_top - face_y_bot) / 2.0 * 0.55 * (0.4 + 0.6 * float(severity))
    thickness = max(half_w, half_h) * 0.05

    box = trimesh.creation.box(extents=[half_w * 2.0, half_h * 2.0, thickness])
    box.apply_translation([cx, cy, cz + thickness * 0.5])
    return box.vertices.astype(np.float32), box.faces.astype(np.int32)


def _build_hat_mesh(
    landmarks_3d: np.ndarray,
    severity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Hat: cylinder crown + flat brim.

    Anchored above the iBUG forehead (landmarks[19:24]). Height scales with
    severity. The brim is a flat annulus around the crown base.
    """
    import trimesh

    brow = landmarks_3d[19:24].mean(axis=0)
    eye_center = (landmarks_3d[36:42].mean(axis=0) + landmarks_3d[42:48].mean(axis=0)) / 2.0
    span = float(np.linalg.norm(landmarks_3d[16] - landmarks_3d[0]))

    crown_radius = span * 0.40
    crown_height = span * (0.20 + 0.30 * float(severity))
    brim_radius = crown_radius * 1.4
    brim_thickness = span * 0.02

    crown_base_y = brow[1] + span * 0.05
    crown_base = np.array([brow[0], crown_base_y, eye_center[2]], dtype=np.float32)

    crown = trimesh.creation.cylinder(radius=crown_radius, height=crown_height, sections=24)
    crown.apply_translation([crown_base[0], crown_base[1] + crown_height / 2.0, crown_base[2]])

    brim = trimesh.creation.cylinder(radius=brim_radius, height=brim_thickness, sections=24)
    brim.apply_translation([crown_base[0], crown_base[1], crown_base[2]])

    hat = trimesh.util.concatenate([crown, brim])
    return hat.vertices.astype(np.float32), hat.faces.astype(np.int32)


def _build_face_patch_mesh(
    landmarks_3d: np.ndarray,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic face patch: small flat box at a random face position.

    Position is sampled uniformly inside the iBUG facial polygon (jaw 0-16
    + brow 17-26). Size scales with severity to control coverage.
    """
    import trimesh

    rng = np.random.RandomState(seed)
    contour_idx = list(range(17)) + list(range(17, 27))[::-1]
    contour = landmarks_3d[contour_idx]
    centroid = contour.mean(axis=0)

    span = float(np.linalg.norm(landmarks_3d[16] - landmarks_3d[0]))
    half_w = span * (0.05 + 0.20 * float(severity))
    half_h = half_w
    thickness = span * 0.02

    jitter = rng.uniform(-0.2, 0.2, size=2) * span
    cx = float(centroid[0]) + jitter[0]
    cy = float(centroid[1]) + jitter[1]
    cz = float(contour[:, 2].max()) + thickness

    patch = trimesh.creation.box(extents=[half_w * 2.0, half_h * 2.0, thickness])
    patch.apply_translation([cx, cy, cz])
    return patch.vertices.astype(np.float32), patch.faces.astype(np.int32)


# =========================================================================
# Perturbation functions
# =========================================================================

def _eyes_visible_glasses(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Sunglasses occluder over both eyes [§7.4.5].

    OFIQ measures EyesVisible.scalar high when eyes are unoccluded. We
    degrade by attaching a sunglasses occluder; lens radii scale with
    severity so coverage grows monotonically.
    """
    scene = _require_scene_with_landmarks(scene, "EyesVisible.scalar")
    if float(severity) <= 0:
        return _finalize(_require_renderer().render(scene), img)

    landmarks_3d = scene.flame_landmarks_3d  # type: ignore[attr-defined]
    verts, faces = _build_glasses_mesh(landmarks_3d, float(severity))

    perturbed = _clone_with_3d_attrs(scene)
    perturbed.occluders.append(Occluder(name="sunglasses", vertices=verts, faces=faces))
    return _finalize(_require_renderer().render(perturbed), img)


def _mouth_occlusion_mask(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Surgical mask covering mouth + nose [§7.4.6].

    Degrades MouthOcclusionPrevention.scalar (high score = unoccluded).
    Mask spans from below-eye to chin with extent scaled by severity.
    """
    scene = _require_scene_with_landmarks(scene, "MouthOcclusionPrevention.scalar")
    if float(severity) <= 0:
        return _finalize(_require_renderer().render(scene), img)

    landmarks_3d = scene.flame_landmarks_3d  # type: ignore[attr-defined]
    verts, faces = _build_mask_mesh(landmarks_3d, float(severity))

    perturbed = _clone_with_3d_attrs(scene)
    perturbed.occluders.append(Occluder(name="surgical_mask", vertices=verts, faces=faces))
    return _finalize(_require_renderer().render(perturbed), img)


def _face_occlusion_patch(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Generic face patch [§7.4.7].

    Random rectangular occluder placed inside the facial polygon. Size
    scales with severity. Position randomized under seed.
    """
    scene = _require_scene_with_landmarks(scene, "FaceOcclusionPrevention.scalar")
    if float(severity) <= 0:
        return _finalize(_require_renderer().render(scene), img)

    landmarks_3d = scene.flame_landmarks_3d  # type: ignore[attr-defined]
    verts, faces = _build_face_patch_mesh(landmarks_3d, float(severity), seed)

    perturbed = _clone_with_3d_attrs(scene)
    perturbed.occluders.append(Occluder(name="face_patch", vertices=verts, faces=faces))
    return _finalize(_require_renderer().render(perturbed), img)


def _no_head_coverings_hat(
    img: np.ndarray,
    severity: float,
    seed: int,
    scene: Optional[SceneState] = None,
) -> np.ndarray:
    """Hat covering crown + brim shading forehead [§7.4.13].

    Degrades NoHeadCoverings.scalar (high score = uncovered). Hat height
    scales with severity.
    """
    scene = _require_scene_with_landmarks(scene, "NoHeadCoverings.scalar")
    if float(severity) <= 0:
        return _finalize(_require_renderer().render(scene), img)

    landmarks_3d = scene.flame_landmarks_3d  # type: ignore[attr-defined]
    verts, faces = _build_hat_mesh(landmarks_3d, float(severity))

    perturbed = _clone_with_3d_attrs(scene)
    perturbed.occluders.append(Occluder(name="hat", vertices=verts, faces=faces))
    return _finalize(_require_renderer().render(perturbed), img)


# =========================================================================
# Registration
# =========================================================================

register("EyesVisible.scalar", _eyes_visible_glasses,
         "3D::sunglasses occluder over eyes [§7.4.5]", "lens radius: 0.4x -> 1.0x eye span",
         tier="geometry")
register("MouthOcclusionPrevention.scalar", _mouth_occlusion_mask,
         "3D::surgical mask occluder over mouth+nose [§7.4.6]",
         "mask extent: 0.4x -> 1.0x lower face",
         tier="geometry")
register("FaceOcclusionPrevention.scalar", _face_occlusion_patch,
         "3D::random rectangular patch inside facial polygon [§7.4.7]",
         "patch half-width: 0.05x -> 0.25x face span",
         tier="geometry")
register("NoHeadCoverings.scalar", _no_head_coverings_hat,
         "3D::hat (cylinder crown + brim) above forehead [§7.4.13]",
         "crown height: 0.20x -> 0.50x face span",
         tier="geometry")
