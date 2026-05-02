"""DECALift: real single-image FLAME fitting via DECA's encoder.

We use DECA's components piecemeal so we can swap the renderer:
- ResnetEncoder (decalib.models.encoders) for image -> 236-d FLAME params
- FLAME (decalib.models.FLAME) for params -> 3D mesh vertices + landmarks
- FLAMETex (decalib.models.FLAME) for texture coeffs -> UV albedo

We deliberately do NOT instantiate the full decalib.deca.DECA class because
its __init__ calls SRenderY which imports pytorch3d. We don't have pytorch3d
on this box. PyRenderRenderer renders the mesh instead.

Asset preconditions (see ASSETS.md):
- third_party/DECA/data/generic_model.pkl  (FLAME 2020, license-gated)
- third_party/DECA/data/deca_model.tar     (DECA pretrained encoder + decoder)
- third_party/DECA/data/landmark_embedding.npy
- third_party/DECA/data/uv_face_eye_mask.png

If any are missing, DECALift.__init__ raises MissingAssetError naming the
exact path it expected. This is real failure, not a mock.
"""

from __future__ import annotations

# Import order matters: chumpy compat must run before any FLAME pkl unpickle.
from ofiq_syngen.three_d.lift import _chumpy_compat  # noqa: F401  (Py3.11 inspect.getargspec shim)

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from ofiq_syngen.three_d.scene.state import (
    Camera,
    FlameParams,
    Lighting,
    SceneState,
)


_DECA_REPO = Path(__file__).resolve().parents[3] / "third_party" / "DECA"


class MissingAssetError(FileNotFoundError):
    """Raised when DECA cannot construct because a required asset is missing."""


def _ensure_deca_on_path() -> None:
    """Make decalib importable without installing it as a package."""
    if str(_DECA_REPO) not in sys.path:
        sys.path.insert(0, str(_DECA_REPO))


def _check_assets(deca_dir: Path) -> None:
    """Verify every asset DECA needs exists. Raise with a precise message."""
    required = {
        "FLAME 2020 model (license-gated, register at https://flame.is.tue.mpg.de/)":
            deca_dir / "data" / "generic_model.pkl",
        "DECA pretrained weights (Google Drive, see fetch_data.sh)":
            deca_dir / "data" / "deca_model.tar",
        "FLAME landmark embedding":
            deca_dir / "data" / "landmark_embedding.npy",
        "head template OBJ":
            deca_dir / "data" / "head_template.obj",
        "UV face-eye mask":
            deca_dir / "data" / "uv_face_eye_mask.png",
        "UV face mask":
            deca_dir / "data" / "uv_face_mask.png",
        "fixed displacement":
            deca_dir / "data" / "fixed_displacement_256.npy",
        "texture data":
            deca_dir / "data" / "texture_data_256.npy",
        "mean texture":
            deca_dir / "data" / "mean_texture.jpg",
    }
    missing = {label: path for label, path in required.items() if not path.exists()}
    if missing:
        msg_lines = ["DECA cannot construct. Missing assets:"]
        for label, path in missing.items():
            msg_lines.append(f"  - {label}: {path}")
        msg_lines.append("")
        msg_lines.append("Run: cd third_party/DECA && bash fetch_data.sh")
        msg_lines.append("See ASSETS.md for FLAME license registration.")
        raise MissingAssetError("\n".join(msg_lines))


@dataclass
class DECALiftConfig:
    """Mirrors decalib.utils.config.cfg.model fields used by the encoder + FLAME."""

    n_shape: int = 100
    n_tex: int = 50
    n_exp: int = 50
    n_pose: int = 6   # axis-angle: global rotation [0:3] + jaw [3:6]
    n_cam: int = 3    # ortho: scale, tx, ty
    n_light: int = 27
    n_detail: int = 128
    image_size: int = 224
    uv_size: int = 256
    crop_scale: float = 1.25  # bbox expansion factor
    use_tex: bool = True

    @property
    def n_param(self) -> int:
        return self.n_shape + self.n_tex + self.n_exp + self.n_pose + self.n_cam + self.n_light

    @property
    def param_list(self) -> list[str]:
        return ["shape", "tex", "exp", "pose", "cam", "light"]

    @property
    def num_dict(self) -> dict[str, int]:
        return {
            "shape": self.n_shape,
            "tex": self.n_tex,
            "exp": self.n_exp,
            "pose": self.n_pose,
            "cam": self.n_cam,
            "light": self.n_light,
        }


class DECALift(nn.Module):
    """Real DECA-based FLAME fitter. No mocks, no stubs."""

    backend_name = "deca"

    def __init__(
        self,
        deca_dir: Optional[str] = None,
        device: str = "cuda",
        config: Optional[DECALiftConfig] = None,
    ) -> None:
        super().__init__()
        self.deca_dir = Path(deca_dir) if deca_dir else _DECA_REPO
        _check_assets(self.deca_dir)
        _ensure_deca_on_path()

        self.cfg = config or DECALiftConfig()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        from decalib.models.encoders import ResnetEncoder
        from decalib.models.FLAME import FLAME, FLAMETex

        flame_cfg = _build_flame_cfg(self.deca_dir, self.cfg)

        self.E_flame = ResnetEncoder(outsize=self.cfg.n_param).to(self.device)
        self.flame = FLAME(flame_cfg).to(self.device)
        if self.cfg.use_tex and (self.deca_dir / "data" / "FLAME_albedo_from_BFM.npz").exists():
            self.flametex: Optional[nn.Module] = FLAMETex(flame_cfg).to(self.device)
        else:
            self.flametex = None

        ckpt_path = self.deca_dir / "data" / "deca_model.tar"
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        _copy_state_dict(self.E_flame, ckpt["E_flame"])

        self.E_flame.eval()
        self.flame.eval()
        if self.flametex is not None:
            self.flametex.eval()

        self._mean_texture = _load_mean_texture(self.deca_dir)

        # FAN landmark detector for face cropping. Lazy-loaded on first lift().
        self._face_detector = None

    @torch.no_grad()
    def lift(
        self,
        image: np.ndarray,
        face_analysis: Optional[object] = None,
    ) -> SceneState:
        """Lift a BGR uint8 image into a SceneState with FLAME mesh + camera + lighting."""
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError(
                f"DECALift expects BGR uint8 (H, W, 3), got shape={image.shape} dtype={image.dtype}"
            )
        h, w = image.shape[:2]

        crop, tform_inv = self._crop_to_224(image, face_analysis)
        crop_tensor = torch.from_numpy(crop.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device) / 255.0

        params = self.E_flame(crop_tensor)
        codes = self._decompose(params)

        verts, ldm2d, ldm3d = self.flame(
            shape_params=codes["shape"],
            expression_params=codes["exp"],
            pose_params=codes["pose"],
        )

        if self.flametex is not None:
            albedo = self.flametex(codes["tex"])  # (1, 3, uv, uv) in [0, 1]
            uv_texture = (albedo[0].cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
            uv_texture = cv2.cvtColor(uv_texture, cv2.COLOR_RGB2BGR)
        else:
            uv_texture = self._mean_texture.copy()

        flame_params = FlameParams(
            shape=codes["shape"][0].cpu().numpy(),
            expression=codes["exp"][0].cpu().numpy(),
            pose=codes["pose"][0].cpu().numpy(),
            texture=codes["tex"][0].cpu().numpy() if self.flametex is not None else None,
            uv_texture=uv_texture,
        )

        cam_params = codes["cam"][0].cpu().numpy()  # (3,) ortho s, tx, ty
        camera = Camera(
            intrinsics=_ortho_intrinsics(cam_params, h, w),
            extrinsics=tform_inv.astype(np.float32),
            image_size=(h, w),
        )

        lighting = Lighting(
            sh_coeffs=codes["light"][0].cpu().numpy().reshape(9, 3),
            ambient=0.0,
        )

        face_mask = self._build_face_mask(image, face_analysis)
        background_plate = self._build_background_plate(image, face_mask)

        scene = SceneState(
            image=image,
            image_size=(h, w),
            background_plate=background_plate,
            face_mask=face_mask,
            flame=flame_params,
            camera=camera,
            lighting=lighting,
            occluders=[],
            face_analysis=face_analysis,
            lift_backend=self.backend_name,
        )
        scene.flame_verts = verts[0].cpu().numpy()  # type: ignore[attr-defined]
        scene.flame_faces = self.flame.faces_tensor.cpu().numpy()  # type: ignore[attr-defined]
        scene.flame_landmarks_3d = ldm3d[0].cpu().numpy()  # type: ignore[attr-defined]  # (68, 3) iBUG
        scene.flame_landmarks_2d = ldm2d[0].cpu().numpy()  # type: ignore[attr-defined]  # (68, 2) NDC
        scene.crop_size = self.cfg.image_size  # type: ignore[attr-defined]
        scene.flame_module = self.flame  # type: ignore[attr-defined]

        # Save the ORIGINAL (un-perturbed) per-vertex 2D image-space
        # projection. Renderers use this to sample per-vertex texture
        # from the source image, then apply that texture when rendering
        # the perturbed mesh -- gives proper photoreal output instead
        # of flat skin color.
        from decalib.utils import util
        with torch.no_grad():
            trans_verts = util.batch_orth_proj(verts, codes["cam"])
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
            # Map NDC [-1, 1] back to crop pixel coordinates [0, crop_size]
            crop_size = self.cfg.image_size
            trans_xy = (trans_verts[0, :, :2].cpu().numpy() + 1.0) * 0.5 * crop_size
        # Use the inverse transform to map from crop-space back to original
        # image coordinates (so the colors come from the actual full image).
        # tform_inv is the affine that maps (crop xy 1) -> (image xy 1).
        ones = np.ones((trans_xy.shape[0], 1), dtype=np.float32)
        trans_xy1 = np.concatenate([trans_xy, ones], axis=1)
        if tform_inv.shape == (3, 3):
            trans_image_xy = (tform_inv @ trans_xy1.T).T[:, :2]
        else:
            trans_image_xy = trans_xy
        scene.flame_verts_2d_orig = trans_image_xy.astype(np.float32)  # type: ignore[attr-defined]
        return scene

    def _decompose(self, code: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        start = 0
        for key, n in self.cfg.num_dict.items():
            out[key] = code[:, start:start + n]
            start += n
        return out

    def _crop_to_224(
        self,
        image_bgr: np.ndarray,
        face_analysis: Optional[object],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (224x224 RGB crop, tform_inverse 3x3) following DECA's preprocessing."""
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        bbox = self._bbox_from_analysis(face_analysis, h, w)
        if bbox is None:
            bbox = self._bbox_from_fan(rgb)
        if bbox is None:
            bbox = (0, 0, w - 1, h - 1)

        left, top, right, bottom = bbox
        old_size = max(right - left, bottom - top)
        center_x = (right + left) / 2.0
        center_y = (bottom + top) / 2.0 + old_size * 0.06  # DECA's vertical bias for kpt-style boxes
        size = old_size * self.cfg.crop_scale

        from skimage.transform import estimate_transform, warp

        src_pts = np.array(
            [
                [center_x - size / 2, center_y - size / 2],
                [center_x - size / 2, center_y + size / 2],
                [center_x + size / 2, center_y - size / 2],
            ]
        )
        dst = self.cfg.image_size - 1
        dst_pts = np.array([[0, 0], [0, dst], [dst, 0]])
        tform = estimate_transform("similarity", src_pts, dst_pts)
        crop = warp(rgb / 255.0, tform.inverse, output_shape=(self.cfg.image_size, self.cfg.image_size))
        crop_uint8 = (crop * 255.0).clip(0, 255).astype(np.uint8)
        tform_inv = np.linalg.inv(tform.params)
        return crop_uint8, tform_inv

    def _bbox_from_analysis(
        self,
        face_analysis: Optional[object],
        h: int,
        w: int,
    ) -> Optional[tuple[float, float, float, float]]:
        if face_analysis is None:
            return None
        landmarks = getattr(face_analysis, "landmarks_98", None)
        if landmarks is None or landmarks.size == 0:
            return None
        xs = landmarks[:, 0].astype(np.float32)
        ys = landmarks[:, 1].astype(np.float32)
        return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())

    def _bbox_from_fan(self, rgb: np.ndarray) -> Optional[tuple[float, float, float, float]]:
        if self._face_detector is None:
            try:
                import face_alignment
                self._face_detector = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    flip_input=False,
                    device=str(self.device),
                )
            except Exception:
                return None
        try:
            preds = self._face_detector.get_landmarks_from_image(rgb)
        except Exception:
            return None
        if not preds:
            return None
        kpt = preds[0]
        return float(kpt[:, 0].min()), float(kpt[:, 1].min()), float(kpt[:, 0].max()), float(kpt[:, 1].max())

    def _build_face_mask(
        self,
        image: np.ndarray,
        face_analysis: Optional[object],
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if face_analysis is not None:
            mask = getattr(face_analysis, "face_mask", None)
            if mask is not None and mask.shape[:2] == (h, w):
                return mask.astype(np.uint8)
        return np.full((h, w), 255, dtype=np.uint8)

    def _build_background_plate(
        self,
        image: np.ndarray,
        face_mask: np.ndarray,
    ) -> np.ndarray:
        plate = image.copy()
        face_bool = face_mask > 0
        if not face_bool.any():
            return plate
        plate_inpaint = cv2.inpaint(image, face_mask, 5, cv2.INPAINT_TELEA)
        return plate_inpaint


def _build_flame_cfg(deca_dir: Path, cfg: DECALiftConfig):
    """Construct the YACS-like config object FLAME() expects."""
    from yacs.config import CfgNode as CN

    flame_cfg = CN()
    flame_cfg.flame_model_path = str(deca_dir / "data" / "generic_model.pkl")
    flame_cfg.flame_lmk_embedding_path = str(deca_dir / "data" / "landmark_embedding.npy")
    flame_cfg.tex_path = str(deca_dir / "data" / "FLAME_albedo_from_BFM.npz")
    flame_cfg.tex_type = "BFM"
    flame_cfg.uv_size = cfg.uv_size
    flame_cfg.n_shape = cfg.n_shape
    flame_cfg.n_tex = cfg.n_tex
    flame_cfg.n_exp = cfg.n_exp
    flame_cfg.n_pose = cfg.n_pose
    flame_cfg.n_cam = cfg.n_cam
    flame_cfg.n_light = cfg.n_light
    flame_cfg.use_tex = cfg.use_tex
    flame_cfg.jaw_type = "aa"
    return flame_cfg


def _copy_state_dict(module: nn.Module, src: dict) -> None:
    """Strip a 'module.' prefix if present and load."""
    own = module.state_dict()
    cleaned = {k.replace("module.", ""): v for k, v in src.items()}
    matched = {k: v for k, v in cleaned.items() if k in own and own[k].shape == v.shape}
    own.update(matched)
    module.load_state_dict(own)


def _ortho_intrinsics(cam_params: np.ndarray, h: int, w: int) -> np.ndarray:
    """Build a 3x3 intrinsic matrix in image pixels for DECA's orthographic camera.

    DECA's camera params are (s, tx, ty) with projection
        p_x = s * (v_x + tx) * crop_size / 2 + crop_size / 2
        p_y = s * (v_y + ty) * crop_size / 2 + crop_size / 2
    in the 224x224 crop space. We return an intrinsic that pyrender can use
    after we set up an OrthographicCamera; the actual scale/offset is applied
    in the renderer.
    """
    s, tx, ty = float(cam_params[0]), float(cam_params[1]), float(cam_params[2])
    return np.array(
        [[s, 0.0, tx], [0.0, s, ty], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _load_mean_texture(deca_dir: Path) -> np.ndarray:
    path = deca_dir / "data" / "mean_texture.jpg"
    img = cv2.imread(str(path))
    if img is None:
        raise MissingAssetError(f"Cannot read mean texture: {path}")
    return img
