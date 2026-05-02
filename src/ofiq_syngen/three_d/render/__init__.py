"""Renderer backends: SceneState -> 2D BGR uint8 at scene image_size.

Available real backends:
- PyRenderRenderer (pyrender_renderer.py): pyrender via EGL, no CUDA toolkit
  needed. Renders FLAME mesh + UV texture + SH lighting and composites over
  the scene's background plate.
"""

from ofiq_syngen.three_d.render.base import Renderer
from ofiq_syngen.three_d.render.pyrender_renderer import PyRenderRenderer

__all__ = ["Renderer", "PyRenderRenderer"]
