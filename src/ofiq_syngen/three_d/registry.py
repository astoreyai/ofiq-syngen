"""Component registry mirroring ofiq-syngen.

Same dataclass shape and same register() signature so a downstream trainer
swaps registries with one import. Registered functions take (img, severity,
seed, scene) where scene is a SceneState carrying as much of the 3D scene
as the perturbation's tier requires.

Tiers (selected by `tier` arg to register()):
- geometry  — needs full FLAME lift. SceneState has flame_verts, flame_module,
              camera. Pipeline runs DECALift.
- appearance — needs only face mask. SceneState has image + face_mask.
              Pipeline builds via FaceContext (BiSeNet) without DECA.
- post_2d   — needs nothing. SceneState passed as None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ofiq_syngen.three_d.standards import STANDARDS_REFS, StandardRefs


VALID_TIERS = ("geometry", "appearance", "post_2d")


@dataclass
class ComponentDegradation:
    """Registration entry for a 3dsyn degradation function.

    Mirrors ofiq_syngen.components.ComponentDegradation field-for-field, plus
    a `tier` field that drives pipeline scene-building.
    """

    ofiq_component: str
    function: Callable
    description: str
    severity_range: str
    tier: str = "geometry"
    requires_scene: bool = True
    standard_refs: Optional[StandardRefs] = None

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(
                f"tier must be one of {VALID_TIERS}; got {self.tier!r}"
            )


COMPONENT_REGISTRY: dict[str, list[ComponentDegradation]] = {}


def register(
    component: str,
    fn: Callable,
    desc: str,
    srange: str,
    tier: str = "geometry",
    needs_scene: Optional[bool] = None,
) -> None:
    """Register a 3dsyn degradation function.

    Args:
        component: OFIQ component name (e.g. "HeadPoseYaw.scalar").
        fn: callable with signature (img, severity, seed, scene) -> np.ndarray.
        desc: short description; conventionally prefixed with the tier
            marker ("3D::", "APP::", "POST::").
        srange: human-readable severity range (e.g. "yaw: 0deg -> 35deg").
        tier: "geometry" | "appearance" | "post_2d". Drives pipeline behavior.
        needs_scene: legacy alias. Ignored if tier is given explicitly.
            Inferred from tier when omitted: True for geometry/appearance,
            False for post_2d.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}; got {tier!r}")

    requires_scene = (tier != "post_2d") if needs_scene is None else needs_scene

    if component not in COMPONENT_REGISTRY:
        COMPONENT_REGISTRY[component] = []
    COMPONENT_REGISTRY[component].append(
        ComponentDegradation(
            ofiq_component=component,
            function=fn,
            description=desc,
            severity_range=srange,
            tier=tier,
            requires_scene=requires_scene,
            standard_refs=STANDARDS_REFS.get(component),
        )
    )


def list_supported_components() -> list[str]:
    """Return registered OFIQ component names sorted alphabetically."""
    return sorted(COMPONENT_REGISTRY.keys())
