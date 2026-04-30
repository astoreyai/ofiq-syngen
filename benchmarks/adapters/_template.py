"""Adapter template for benchmarks/run_grid.py.

Copy this file to benchmarks/adapters/<your_method>.py and fill in
``degrade``. The grid runner imports any non-underscore-prefixed
module from this directory.

Adapters are responsible for:
- Lazy-loading their dependencies inside ``degrade`` so a missing
  optional dependency only skips the adapter, not the whole grid.
- Returning a ``uint8`` BGR image with the same shape as input.
- Honoring the seed for determinism where the underlying method
  supports it.

For methods that do not directly correspond to an OFIQ component name,
implement a mapping (e.g., GFPGAN's combined blur+noise+JPEG can be
mapped to Sharpness.scalar with severity = blur strength).
"""

from __future__ import annotations

import numpy as np


name = "TEMPLATE"  # Replace with the method name; e.g., "gfpgan_ffhq", "real_esrgan"


def degrade(image: np.ndarray, component: str, severity: float, seed: int) -> np.ndarray:
    """Apply this method's perturbation matching `component` at `severity`.

    Args:
        image: BGR uint8, shape (H, W, 3).
        component: target OFIQ component name (e.g., "Sharpness.scalar").
        severity: 0.0 (no degradation) to 1.0 (max degradation).
        seed: RNG seed; the method should be deterministic for fixed seed.

    Returns:
        Degraded image, same shape and dtype as input.
    """
    raise NotImplementedError(
        f"Adapter {name} has no implementation for component {component}. "
        "Either implement it or skip it (raise to fall through to the next adapter)."
    )
