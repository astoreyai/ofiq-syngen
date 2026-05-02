"""Lift protocol: 2D face image -> 3D SceneState.

Concrete implementations:
- MockLift (mock.py): trivial proxy used for tests, no neural net.
- DECALift (deca.py, stub): DECA single-image FLAME fitting.
- EMOCALift (deca.py, stub): EMOCA v2 with stronger expression head.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np

from ofiq_syngen.three_d.scene.state import SceneState


@runtime_checkable
class Lift(Protocol):
    """Lift a 2D face image into a 3D SceneState.

    Implementations must:
    - Populate SceneState.flame, .camera, .lighting consistent with the input.
    - Set SceneState.image_size to (H, W) of the input exactly.
    - Set SceneState.background_plate to the input with the face region
      removed (NaN or zero where the face will be re-rendered).
    - Set SceneState.face_mask to a binary uint8 mask (255 inside face).
    - Set SceneState.lift_backend to a short string identifying themselves.
    """

    backend_name: str

    def lift(self, image: np.ndarray, face_analysis: Optional[object] = None) -> SceneState:
        """Build a SceneState from a BGR uint8 image."""
        ...
