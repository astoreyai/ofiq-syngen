"""Perturbation tiers.

Importing this package registers all built-in perturbations into
three_d_syn.registry.COMPONENT_REGISTRY. The four submodules correspond
to the tier taxonomy in the README:

- geometry: real 3D wins (pose, expression, identity, framing)
- appearance: 3D-meaningful (lighting, exposure, colour, occluders)
- post_2d: sensor / codec, no 3D meaning (sharpness, JPEG)
"""

# Import order matters only for registration ordering; functionally
# independent.
from ofiq_syngen.three_d.perturb import geometry    # noqa: F401
from ofiq_syngen.three_d.perturb import occluders   # noqa: F401
from ofiq_syngen.three_d.perturb import appearance  # noqa: F401
from ofiq_syngen.three_d.perturb import post_2d     # noqa: F401
