"""Lift backends: 2D image -> 3D SceneState.

Available real backends:
- DECALift (deca.py): single-image FLAME fitting via DECA. Requires FLAME 2020
  model file + DECA pretrained weights. See ASSETS.md.

Future real backends (drop-in behind the Lift protocol):
- EMOCALift: EMOCA v2 with stronger expression head.
"""

from ofiq_syngen.three_d.lift.base import Lift
from ofiq_syngen.three_d.lift.deca import DECALift, MissingAssetError

__all__ = ["Lift", "DECALift", "MissingAssetError"]
