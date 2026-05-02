"""SceneState: the 3D scene a perturbation operates on.

Layered to match ofiq-syngen.FaceContext: 2D face analysis comes first
(landmarks, parsing map, occlusion mask, head pose), then 3D fields are
populated by a Lift backend (FLAME params, fitted camera, lighting, UV
texture, optional occluder meshes).

A degradation function mutates a copy of SceneState, hands it to a
Renderer, and the renderer composites the result over the original
background plate at the input image dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class FlameParams:
    """FLAME morphable model parameters.

    Shape and expression coefficients are 100-d and 50-d respectively in the
    canonical FLAME-2020 release. Pose is (global_rot[3], jaw[3], neck[3],
    eye_l[3], eye_r[3]) in axis-angle radians. All fields are optional so a
    MockLift can populate only what it needs.
    """

    shape: Optional[np.ndarray] = None       # (100,)
    expression: Optional[np.ndarray] = None  # (50,)
    pose: Optional[np.ndarray] = None        # (15,) axis-angle radians
    texture: Optional[np.ndarray] = None     # (50,) DECA texture coeffs
    detail: Optional[np.ndarray] = None      # (128,) DECA detail coeffs
    uv_texture: Optional[np.ndarray] = None  # (H, W, 3) uint8, BGR sampled UV map


@dataclass
class Camera:
    """Pinhole camera fitted to the input image.

    The renderer must reproduce the input image dimensions exactly.
    """

    intrinsics: Optional[np.ndarray] = None  # (3, 3)
    extrinsics: Optional[np.ndarray] = None  # (4, 4) world-to-camera
    image_size: tuple[int, int] = (0, 0)     # (H, W) — must match input


@dataclass
class Lighting:
    """Scene lighting. Spherical harmonics (DECA convention) plus optional
    point/directional lights for asymmetric illumination perturbations.
    """

    sh_coeffs: Optional[np.ndarray] = None       # (9, 3) order-2 SH, RGB
    point_lights: list[dict[str, Any]] = field(default_factory=list)
    directional_lights: list[dict[str, Any]] = field(default_factory=list)
    ambient: float = 0.0


@dataclass
class Occluder:
    """A 3D occluder mesh placed in scene coordinates.

    Used for EyesVisible (sunglasses), MouthOcclusion (surgical mask),
    NoHeadCoverings (hat), FaceOcclusion (generic patch). Depth-ordered
    against the FLAME mesh by the renderer.
    """

    name: str
    vertices: np.ndarray                    # (V, 3)
    faces: np.ndarray                       # (F, 3) int
    texture: Optional[np.ndarray] = None    # (H, W, 3) uint8
    transform: Optional[np.ndarray] = None  # (4, 4) attach to FLAME landmark


@dataclass
class SceneState:
    """Full 3D scene reconstructed from a single 2D face image.

    A perturbation typically clones SceneState, mutates one tier, and asks
    the Renderer to flatten back to a 2D image at image_size.
    """

    image: np.ndarray                       # original BGR uint8 (H, W, 3)
    image_size: tuple[int, int]             # (H, W)
    background_plate: np.ndarray            # original image with face region removed
    face_mask: np.ndarray                   # (H, W) uint8 from BiSeNet skin/face
    flame: FlameParams = field(default_factory=FlameParams)
    camera: Camera = field(default_factory=Camera)
    lighting: Lighting = field(default_factory=Lighting)
    occluders: list[Occluder] = field(default_factory=list)
    face_analysis: Optional[Any] = None     # FaceAnalysis (ofiq-syngen FaceContext)
    lift_backend: str = "unset"             # "deca", "emoca", "mock"

    def clone(self) -> SceneState:
        """Shallow clone safe for mutation. Image and background plate are
        copied; FLAME arrays are copied; occluder meshes are not deep-copied
        because they are not mutated by perturbations.
        """
        return SceneState(
            image=self.image.copy(),
            image_size=self.image_size,
            background_plate=self.background_plate.copy(),
            face_mask=self.face_mask.copy(),
            flame=FlameParams(
                shape=self.flame.shape.copy() if self.flame.shape is not None else None,
                expression=self.flame.expression.copy() if self.flame.expression is not None else None,
                pose=self.flame.pose.copy() if self.flame.pose is not None else None,
                texture=self.flame.texture.copy() if self.flame.texture is not None else None,
                detail=self.flame.detail.copy() if self.flame.detail is not None else None,
                uv_texture=self.flame.uv_texture.copy() if self.flame.uv_texture is not None else None,
            ),
            camera=Camera(
                intrinsics=self.camera.intrinsics.copy() if self.camera.intrinsics is not None else None,
                extrinsics=self.camera.extrinsics.copy() if self.camera.extrinsics is not None else None,
                image_size=self.camera.image_size,
            ),
            lighting=Lighting(
                sh_coeffs=self.lighting.sh_coeffs.copy() if self.lighting.sh_coeffs is not None else None,
                point_lights=list(self.lighting.point_lights),
                directional_lights=list(self.lighting.directional_lights),
                ambient=self.lighting.ambient,
            ),
            occluders=list(self.occluders),
            face_analysis=self.face_analysis,
            lift_backend=self.lift_backend,
        )
