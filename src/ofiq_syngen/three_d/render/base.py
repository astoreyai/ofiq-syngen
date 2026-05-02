"""Renderer protocol: SceneState -> 2D BGR uint8 at scene.image_size.

The contract is strict: output dims must equal scene.image_size, dtype
uint8, BGR channel order. Renderers composite the rasterized FLAME mesh
(plus occluders, with depth ordering) over scene.background_plate using
scene.face_mask.

Concrete implementations:
- MockRenderer (mock.py): warps the input image as a stand-in for testing.
- NvDiffRenderer (nvdiff.py, stub): nvdiffrast differentiable rasterizer.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ofiq_syngen.three_d.scene.state import SceneState


@runtime_checkable
class Renderer(Protocol):
    """Flatten a SceneState back to a 2D image."""

    backend_name: str

    def render(self, scene: SceneState) -> np.ndarray:
        """Return BGR uint8 image with shape (H, W, 3) where (H, W) == scene.image_size."""
        ...
