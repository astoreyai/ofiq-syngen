"""DECA-native textured renderer for the 3D pipeline.

Uses DECA's own ``SRenderY.render_shape`` which projects the FLAME mesh
back to image space with proper UV-texture sampling and composites
over a background image. This is the renderer DECA itself uses for its
``shape_images`` output, so it's the correct path for photoreal output.

The renderer needs the DECA instance from the lift step to access its
internal renderer object and image_size. We cache it on the scene
state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ofiq_syngen.three_d.scene.state import SceneState


class DECARenderer:
    """Renders the perturbed FLAME mesh using DECA's textured renderer."""

    name = "deca_native"

    def render(self, scene_state: SceneState) -> np.ndarray:
        import torch

        deca = getattr(scene_state, "_deca_instance", None)
        codedict = getattr(scene_state, "_deca_codedict", None)
        if deca is None or codedict is None:
            raise RuntimeError(
                "DECARenderer requires the DECA instance and codedict to be "
                "cached on scene_state by the lift step."
            )

        background = scene_state.background_plate
        if background is None:
            raise RuntimeError("DECARenderer requires a background plate.")
        h, w = background.shape[:2]

        # The perturbed FLAME vertices override the lift's verts.
        verts_np = scene_state.flame_verts  # (V, 3)
        device = next(deca.E_flame.parameters()).device
        verts = torch.from_numpy(verts_np.astype(np.float32))[None].to(device)

        # Project verts using the original camera (s, tx, ty) from codedict
        from decalib.utils import util
        trans_verts = util.batch_orth_proj(verts, codedict["cam"])
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]

        # If lift saved an inverse transform back to original image
        # coordinates, apply it so the rendered mesh aligns with the
        # source background (not the cropped DECA input).
        tform = getattr(scene_state, "_deca_tform", None)
        original_image = getattr(scene_state, "_deca_original_image", None)
        points_scale = getattr(scene_state, "_deca_points_scale", None)
        if tform is not None and original_image is not None:
            from decalib.utils.tensor_cropper import transform_points
            trans_verts = transform_points(
                trans_verts, tform, points_scale, [h, w],
            )

        # Background: DECA expects a torch tensor (1, 3, h, w) in [0, 1] RGB
        bg_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        bg_t = torch.from_numpy(bg_rgb.transpose(2, 0, 1))[None].to(device)

        # Render the textured shape over the background
        with torch.no_grad():
            shape_images, _, _, alpha_images = deca.render.render_shape(
                verts, trans_verts, h=h, w=w, images=bg_t, return_grid=True,
            )

        # shape_images is (1, 3, h, w) RGB float [0, 1]; convert to BGR uint8
        out_rgb = shape_images[0].cpu().numpy().transpose(1, 2, 0)
        out_rgb = np.clip(out_rgb * 255.0, 0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

        return out_bgr
