"""3dsyn: ISO/IEC 29794-5 component-aligned 3D face image quality degradation.

Sister library to ofiq-syngen. Lifts a 2D face to a 3D scene (FLAME mesh,
fitted camera, lights, occluders), perturbs in 3D, re-renders to 2D at the
original dimensions. Each perturbation registers under the same OFIQ
component name as ofiq-syngen and uses the same severity in [0, 1] interface.
"""

from __future__ import annotations

from ofiq_syngen.three_d.registry import (
    COMPONENT_REGISTRY,
    ComponentDegradation,
    list_supported_components,
    register,
)

__version__ = "0.0.1"

# Importing the perturb subpackage triggers registration of all built-in
# components into COMPONENT_REGISTRY.
from ofiq_syngen.three_d import perturb  # noqa: F401, E402

__all__ = [
    "COMPONENT_REGISTRY",
    "ComponentDegradation",
    "list_supported_components",
    "register",
    "__version__",
]
