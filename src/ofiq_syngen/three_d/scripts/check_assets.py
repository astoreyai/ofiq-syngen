"""Report which 3D pipeline assets are present.

Run:
    python -m ofiq_syngen.three_d.scripts.check_assets

Or via the CLI:
    ofiq-syngen check-assets
"""

from __future__ import annotations

# Apply Py3.11 inspect.getargspec shim before any chumpy import.
from ofiq_syngen.three_d.lift import _chumpy_compat  # noqa: F401

import importlib
import sys
from pathlib import Path

DECA_REPO = Path(__file__).resolve().parents[3] / "third_party" / "DECA"

REQUIRED = [
    ("FLAME 2020 model (license-gated)", DECA_REPO / "data" / "generic_model.pkl"),
    ("DECA pretrained weights", DECA_REPO / "data" / "deca_model.tar"),
    ("FLAME landmark embedding", DECA_REPO / "data" / "landmark_embedding.npy"),
    ("head template OBJ", DECA_REPO / "data" / "head_template.obj"),
    ("UV face-eye mask", DECA_REPO / "data" / "uv_face_eye_mask.png"),
    ("UV face mask", DECA_REPO / "data" / "uv_face_mask.png"),
    ("fixed displacement", DECA_REPO / "data" / "fixed_displacement_256.npy"),
    ("texture data", DECA_REPO / "data" / "texture_data_256.npy"),
    ("mean texture", DECA_REPO / "data" / "mean_texture.jpg"),
]

OPTIONAL = [
    ("FLAME albedo from BFM (FLAMETex)", DECA_REPO / "data" / "FLAME_albedo_from_BFM.npz"),
]

DEPS = ["torch", "torchvision", "cv2", "numpy", "skimage", "yacs", "kornia",
        "fvcore", "chumpy", "face_alignment", "pyrender", "trimesh", "OpenGL"]


def _check_module(name: str) -> tuple[bool, str]:
    try:
        m = importlib.import_module(name)
        version = getattr(m, "__version__", "?")
        return True, f"{name} {version}"
    except Exception as exc:
        return False, f"{name} (import failed: {exc.__class__.__name__}: {exc})"


def main() -> int:
    print("=== Required assets ===")
    missing = 0
    for label, path in REQUIRED:
        ok = path.exists()
        size_mb = (path.stat().st_size / 1e6) if ok else 0
        marker = "OK " if ok else "MISS"
        print(f"  [{marker}] {label}")
        print(f"         {path}  ({size_mb:.1f} MB)" if ok else f"         {path}  (missing)")
        if not ok:
            missing += 1

    print("\n=== Optional assets ===")
    for label, path in OPTIONAL:
        ok = path.exists()
        marker = "OK " if ok else " - "
        print(f"  [{marker}] {label}: {path}")

    print("\n=== Python deps ===")
    dep_missing = 0
    for dep in DEPS:
        ok, info = _check_module(dep)
        marker = "OK " if ok else "MISS"
        print(f"  [{marker}] {info}")
        if not ok:
            dep_missing += 1

    print("\n=== Renderer probe (EGL) ===")
    try:
        import os
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        import numpy as np
        import pyrender
        import trimesh

        m = trimesh.creation.icosphere(subdivisions=2)
        sc = pyrender.Scene(ambient_light=[0.5, 0.5, 0.5])
        sc.add(pyrender.Mesh.from_trimesh(m))

        cam = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        cam_pose = np.eye(4, dtype=np.float32)
        cam_pose[2, 3] = 3.0
        sc.add(cam, pose=cam_pose)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
        sc.add(light, pose=cam_pose)

        r = pyrender.OffscreenRenderer(64, 64)
        img, _ = r.render(sc)
        r.delete()
        print(f"  [OK ] pyrender EGL render works ({img.shape})")
    except Exception as exc:
        print(f"  [MISS] pyrender EGL render failed: {exc.__class__.__name__}: {exc}")
        dep_missing += 1

    print()
    if missing == 0 and dep_missing == 0:
        print("READY: 3dsyn can lift, render, and degrade.")
        return 0
    print(f"NOT READY: {missing} required asset(s), {dep_missing} dep(s) missing.")
    print("See ASSETS.md for next steps.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
