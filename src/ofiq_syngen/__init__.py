"""ofiq-syngen: ISO/IEC 29794-5 Component-Aligned Synthetic Face Image Quality Degradation Pipeline.

Generates controlled quality degradations mapped to specific ISO/IEC 29794-5
quality components. Each degradation targets a known OFIQ component using the
same face analysis pipeline OFIQ uses for measurement (ADNet landmarks, BiSeNet
parsing, occlusion segmentation, HeadPose3DDFAV2).

Usage:
    from ofiq_syngen import DegradationPipeline, DegradationConfig

    pipeline = DegradationPipeline()
    result = pipeline.degrade_single(image, component="Sharpness.scalar", severity=0.5)
"""

from ofiq_syngen.pipeline import DegradationPipeline, DegradationConfig, DegradationResult
from ofiq_syngen.components import (
    COMPONENT_REGISTRY,
    ComponentDegradation,
    list_supported_components,
    list_all_degradations,
)

__version__ = "0.5.2"

__all__ = [
    "DegradationPipeline",
    "DegradationConfig",
    "DegradationResult",
    "COMPONENT_REGISTRY",
    "ComponentDegradation",
    "list_supported_components",
    "list_all_degradations",
    "__version__",
]
