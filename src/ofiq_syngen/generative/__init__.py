"""Generative degradation components requiring external models.

These components cannot be implemented with simple pixel operations:
- SingleFacePresent: face insertion via Poisson blending
- ExpressionNeutrality: expression manipulation via landmark warping + diffusion
- NoHeadCoverings: hat/headwear overlay via diffusion inpainting

Optional dependencies: insightface, diffusers, torch
"""
