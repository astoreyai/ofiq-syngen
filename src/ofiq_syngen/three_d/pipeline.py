"""DegradationPipeline: tie lift + renderer + registry into one entry point.

API mirrors ofiq-syngen.DegradationPipeline so a downstream trainer can swap
ofiq_syngen.DegradationPipeline -> three_d_syn.DegradationPipeline with one
import change.

Tier-aware scene building:
- geometry  -> full FLAME lift via DECALift (needs FLAME assets).
- appearance -> mask-only scene via ofiq-syngen FaceContext (no DECA needed).
- post_2d   -> no scene built; perturbation runs directly on the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ofiq_syngen.three_d.registry import COMPONENT_REGISTRY, ComponentDegradation
from ofiq_syngen.three_d.scene.analysis import build_face_analysis
from ofiq_syngen.three_d.scene.state import (
    Camera,
    FlameParams,
    Lighting,
    SceneState,
)


@dataclass
class DegradationConfig:
    severity_levels: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    seed: int = 42
    output_format: str = "jpg"
    jpg_quality: int = 95
    device: str = "cuda"
    deca_dir: Optional[str] = None


class DegradationPipeline:
    """3D-grounded ISO-29794-5-aligned degradation pipeline."""

    def __init__(
        self,
        config: Optional[DegradationConfig] = None,
        lift: Optional[Any] = None,
        renderer: Optional[Any] = None,
    ) -> None:
        self.config = config or DegradationConfig()
        self._lift = lift
        self._renderer = renderer
        self._supported = sorted(COMPONENT_REGISTRY.keys())

    @property
    def supported(self) -> list[str]:
        return list(self._supported)

    def _ensure_lift(self):
        if self._lift is None:
            from ofiq_syngen.three_d.lift.deca import DECALift
            self._lift = DECALift(deca_dir=self.config.deca_dir, device=self.config.device)
        return self._lift

    def _ensure_renderer(self):
        if self._renderer is None:
            from ofiq_syngen.three_d.render.pyrender_renderer import PyRenderRenderer
            self._renderer = PyRenderRenderer()
        from ofiq_syngen.three_d.perturb.geometry import set_renderer as _set_geo
        from ofiq_syngen.three_d.perturb.occluders import set_renderer as _set_occ
        _set_geo(self._renderer)
        _set_occ(self._renderer)
        return self._renderer

    def degrade_single(
        self,
        image: np.ndarray,
        component: str,
        severity: float,
        degradation_index: int = 0,
        seed: Optional[int] = None,
        scene: Optional[SceneState] = None,
    ) -> tuple[np.ndarray, dict]:
        if component not in COMPONENT_REGISTRY:
            raise ValueError(
                f"No 3D degradation registered for component '{component}'. "
                f"Supported: {self.supported}"
            )

        degs = COMPONENT_REGISTRY[component]
        deg = degs[degradation_index % len(degs)]
        s = seed if seed is not None else self.config.seed

        if scene is None:
            scene = self._build_scene_for_tier(image, deg)

        if deg.tier == "geometry":
            self._ensure_renderer()

        degraded = deg.function(image, severity, s, scene)

        metadata = {
            "target_component": component,
            "degradation_type": deg.description,
            "tier": deg.tier,
            "severity": severity,
            "seed": s,
            "lift_backend": getattr(self._lift, "backend_name", "unset" if deg.tier == "geometry" else "n/a"),
            "renderer": getattr(self._renderer, "backend_name", "unset" if deg.tier == "geometry" else "n/a"),
        }
        return degraded, metadata

    def degrade_all_components(
        self,
        image: np.ndarray,
        severity: float = 0.5,
        seed: Optional[int] = None,
    ) -> list[tuple[np.ndarray, dict]]:
        """Apply one degradation per registered component, reusing scenes.

        Builds a full FLAME-lifted scene if any geometry-tier component is
        registered, else builds only a mask-only scene if any appearance-tier
        component is registered. post_2d components reuse neither.
        """
        s = seed if seed is not None else self.config.seed
        any_geometry = any(
            entries[0].tier == "geometry" for entries in COMPONENT_REGISTRY.values()
        )
        any_appearance = any(
            entries[0].tier == "appearance" for entries in COMPONENT_REGISTRY.values()
        )

        full_scene: Optional[SceneState] = None
        mask_scene: Optional[SceneState] = None
        if any_geometry:
            try:
                full_scene = self._build_full_scene(image)
                self._ensure_renderer()
            except Exception:
                full_scene = None
        if any_appearance:
            mask_scene = self._build_mask_only_scene(image)

        results = []
        for component in self.supported:
            deg = COMPONENT_REGISTRY[component][0]
            scene = self._scene_for_tier(deg.tier, full_scene, mask_scene)
            try:
                degraded = deg.function(image, severity, s, scene)
            except Exception as exc:
                degraded = image.copy()
                results.append(
                    (
                        degraded,
                        {
                            "target_component": component,
                            "degradation_type": deg.description,
                            "tier": deg.tier,
                            "severity": severity,
                            "seed": s,
                            "error": f"{exc.__class__.__name__}: {exc}",
                        },
                    )
                )
                continue
            results.append(
                (
                    degraded,
                    {
                        "target_component": component,
                        "degradation_type": deg.description,
                        "tier": deg.tier,
                        "severity": severity,
                        "seed": s,
                    },
                )
            )
        return results

    def _build_scene_for_tier(
        self,
        image: np.ndarray,
        deg: ComponentDegradation,
    ) -> Optional[SceneState]:
        if deg.tier == "geometry":
            return self._build_full_scene(image)
        if deg.tier == "appearance":
            return self._build_mask_only_scene(image)
        return None

    def _scene_for_tier(
        self,
        tier: str,
        full_scene: Optional[SceneState],
        mask_scene: Optional[SceneState],
    ) -> Optional[SceneState]:
        if tier == "geometry":
            return full_scene
        if tier == "appearance":
            return mask_scene
        return None

    def _build_full_scene(self, image: np.ndarray) -> SceneState:
        face_analysis = build_face_analysis(image)
        lift = self._ensure_lift()
        return lift.lift(image, face_analysis=face_analysis)

    def _build_mask_only_scene(self, image: np.ndarray) -> SceneState:
        """Build a SceneState carrying only the face mask + image.

        No DECA, no FLAME, no renderer. Used by appearance-tier perturbations
        which need only the mask. If ofiq-syngen is not available, falls back
        to a whole-image mask (every pixel = face).
        """
        face_analysis = build_face_analysis(image)
        h, w = image.shape[:2]
        if face_analysis is not None and getattr(face_analysis, "face_mask", None) is not None:
            face_mask = face_analysis.face_mask
            if face_mask.shape[:2] != (h, w):
                face_mask = np.full((h, w), 255, dtype=np.uint8)
        else:
            face_mask = np.full((h, w), 255, dtype=np.uint8)

        return SceneState(
            image=image,
            image_size=(h, w),
            background_plate=image.copy(),
            face_mask=face_mask,
            flame=FlameParams(),
            camera=Camera(image_size=(h, w)),
            lighting=Lighting(),
            occluders=[],
            face_analysis=face_analysis,
            lift_backend="mask_only",
        )
