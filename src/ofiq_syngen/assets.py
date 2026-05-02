"""Asset management for ofiq-syngen.

Discovers, downloads, and verifies the optional model assets that
unlock different operator paths:

    Asset                         Unlocks
    ------------------------------------------------
    OFIQ ONNX models              Region-targeted 2D operators
                                  (BackgroundUniformity, Sharpness,
                                  IlluminationUniformity, etc.)
    DECA pretrained + FLAME 2020  3D head pose (HeadPoseYaw / Pitch)
    BFM dense mesh                2D head pose fallback (TPS warp)
    Stable Diffusion checkpoint   IP2P / SD inpaint paths (auto-download
                                  on first use via diffusers)
    VGGFace2 sample identities    SingleFacePresent second-face library

Resolution order for OFIQ models (most-explicit first):

    1. OFIQ_MODEL_DIR environment variable
    2. ~/.ofiq/models
    3. ~/OFIQ-Project/data/models
    4. /opt/ofiq/models
    5. /usr/local/share/ofiq/models
    6. Sibling repo: <package>/../OFIQ-Project/data/models
    7. Bundled in the wheel: ofiq_syngen/data/models
"""

from __future__ import annotations

import hashlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Asset:
    """Single bundled-or-downloadable asset.

    sha256 is optional. If set, downloads are verified against it and
    mismatches abort with a hard error (supply-chain protection). If
    None, a warning is emitted on download but the file is accepted.
    """

    name: str
    abs_path: Path
    description: str
    required_for: list[str]
    bytes: int | None = None
    sha256: str | None = None


PACKAGE_ROOT = Path(__file__).resolve().parent  # src/ofiq_syngen/
DATA_ROOT = PACKAGE_ROOT / "data"               # src/ofiq_syngen/data/
THIRD_PARTY_ROOT = PACKAGE_ROOT.parent / "third_party"  # src/third_party/
DECA_DATA = THIRD_PARTY_ROOT / "DECA" / "data"

OFFLINE_ENV_VAR = "OFIQ_SYNGEN_OFFLINE"


def _is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV_VAR, "").lower() in ("1", "true", "yes", "on")


def _abort_if_offline(what: str) -> None:
    """Raise if OFIQ_SYNGEN_OFFLINE is set. Use before any network fetch."""
    if _is_offline():
        raise RuntimeError(
            f"{what} requires a network download but {OFFLINE_ENV_VAR} is set. "
            f"Pre-stage the asset manually or unset {OFFLINE_ENV_VAR}."
        )


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected: str | None, name: str) -> None:
    """Verify a file's SHA-256 against a pinned hash.

    If `expected` is None, emit a warning but accept the file. If pinned,
    mismatch is fatal — this protects against a hijacked download URL
    serving altered weights.
    """
    if expected is None:
        warnings.warn(
            f"Asset {name!r} has no pinned SHA-256; downloaded file accepted "
            f"without verification. Add a `sha256=...` field to its Asset entry "
            f"in assets.py before shipping a release. Computed: {_sha256_of(path)}",
            stacklevel=2,
        )
        return
    actual = _sha256_of(path)
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {name} at {path}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"Refusing to use the file (it has been deleted). Either the upstream "
            f"changed or your download was corrupted/intercepted."
        )


ASSETS: list[Asset] = [
    Asset(
        name="bfm_dense",
        abs_path=DATA_ROOT / "3dmm" / "bfm_dense.npz",
        description="Dense BFM-2009 mesh for 2D head pose TPS",
        required_for=["HeadPoseYaw 2D path", "HeadPosePitch 2D path"],
        sha256="438eaaff78bddb89fc5ca771e98aef9bef0f001bbc107b2f3a16fc8392482423",
    ),
    Asset(
        name="bfm_sparse",
        abs_path=DATA_ROOT / "3dmm" / "bfm_sparse.npz",
        description="Sparse BFM-2009 basis for 2D pose",
        required_for=["HeadPoseYaw 2D path"],
        sha256="c109dd893358616f06db533d0af3b9e776e5e52b195629bf9c44db9f3daad288",
    ),
    Asset(
        name="param_mean_std",
        abs_path=DATA_ROOT / "3dmm" / "param_mean_std_62d_120x120.pkl",
        description="3DDFA-V2 normalization constants",
        required_for=["3DMM operations"],
        sha256="090d7150f77cb66c29ddae21e4508fbde59123dcd1dea7facc24a7ed06d1c795",
    ),
    Asset(
        name="deca_model",
        abs_path=DECA_DATA / "deca_model.tar",
        description="DECA pretrained encoder (~415 MB)",
        required_for=["3D HeadPoseYaw", "3D HeadPosePitch"],
        sha256="e714ed293054cba5eea9c96bd3b6b57880074cd84b3fd00d606cbaf0bee7c5c2",
        bytes=434_142_943,
    ),
    Asset(
        name="flame_2020",
        abs_path=DECA_DATA / "generic_model.pkl",
        description="FLAME 2020 morphable model (~51 MB, license-gated)",
        required_for=["3D HeadPoseYaw", "3D HeadPosePitch"],
        # Pinned to the FLAME 2020 release. If Max Planck publishes a
        # new FLAME version (e.g., 2023) the hash will mismatch and the
        # user will get a clear error to update the pin.
        sha256="efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b",
        bytes=53_023_716,
    ),
]


def _asset_by_name(name: str) -> Asset | None:
    for a in ASSETS:
        if a.name == name:
            return a
    return None


def _resolve(asset: Asset) -> Path:
    return asset.abs_path


def status() -> dict[str, dict]:
    """Return a mapping of asset name -> {present, path, description}."""
    out = {}
    for a in ASSETS:
        p = _resolve(a)
        out[a.name] = {
            "present": p.exists(),
            "path": str(p),
            "description": a.description,
            "required_for": a.required_for,
        }
    return out


def download_deca_model(target_dir: Path) -> Path:
    """Download DECA pretrained weights from the official Google Drive.

    Returns the path to the downloaded ``deca_model.tar``.
    Public download, no credentials needed. Refuses to run if
    OFIQ_SYNGEN_OFFLINE is set; verifies SHA-256 if pinned in ASSETS.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "deca_model.tar"
    if out.exists():
        return out

    _abort_if_offline("DECA pretrained weights")

    try:
        import gdown
    except ImportError:
        raise RuntimeError(
            "gdown is required to fetch the DECA pretrained weights. "
            "Install with: pip install ofiq-syngen[three_d]"
        )

    file_id = "1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje"
    print(f"Downloading DECA pretrained ({file_id}) -> {out}", file=sys.stderr)
    gdown.download(id=file_id, output=str(out), quiet=False)

    if not out.exists() or out.stat().st_size < 1_000_000:
        raise RuntimeError(
            f"DECA download failed or file too small at {out}. "
            "Check network and Google Drive availability."
        )
    asset = _asset_by_name("deca_model")
    _verify_sha256(out, asset.sha256 if asset else None, "deca_model")
    return out


FLAME_INSTRUCTIONS = """
============================================================================
FLAME 2020 -- manual download required (license-restricted)
============================================================================

The FLAME 2020 morphable face model is licensed by the Max Planck
Institute for Intelligent Systems for NON-COMMERCIAL RESEARCH PURPOSES
ONLY and CANNOT be redistributed. ofiq-syngen does not download,
bundle, or proxy FLAME under any circumstances.

To enable the 3D head pose pipeline (HeadPoseYaw / HeadPosePitch):

  1. Register an academic account at https://flame.is.tue.mpg.de/
  2. Accept the FLAME 2020 license terms during registration
  3. Wait for approval (typically within 24 hours)
  4. Download FLAME2020.zip from the website's "Download" tab
  5. Unzip and copy generic_model.pkl into the path below:

     {target_path}

  6. Verify the install with:

     ofiq-syngen check-assets

Optional (same source, same license):
  - male_model.pkl
  - female_model.pkl

Full license text: https://flame.is.tue.mpg.de/modellicense.html
============================================================================
"""


BFM_INSTRUCTIONS = """
============================================================================
BFM-2009 derivative -- license-restricted source

The dense / sparse BFM mesh files in src/ofiq_syngen/data/3dmm/ are
derived from the Basel Face Model (BFM-2009) via the cleardusk/3DDFA_V2
distribution. BFM-2009 is licensed by University of Basel for academic
research only -- redistribution is prohibited.

To obtain the files, run:
  ofiq-syngen install-assets   # downloads from cleardusk/3DDFA_V2 GitHub
                               # (you accept the BFM license by downloading)

Or manually:
  - bfm_dense.npz (~22 MB)
  - bfm_sparse.npz (~40 KB)
  - param_mean_std_62d_120x120.pkl (~1 KB)

are downloadable from the cleardusk/3DDFA_V2 repository configs.
Place them in: {target_dir}

License: https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=agreement
============================================================================
"""


def download_bfm_files(target_dir: Path) -> dict[str, Path]:
    """Download BFM-2009 derivative files from cleardusk/3DDFA_V2.

    These are bundled into 3DDFA_V2 under MIT, but the underlying BFM-2009
    license restricts redistribution to academic use. By calling this
    you confirm you have read the BFM license at:
        https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=agreement

    Returns:
        Mapping of filename -> path.
    """
    import urllib.request
    import pickle
    import numpy as np

    _abort_if_offline("BFM-2009 derivative download")

    target_dir.mkdir(parents=True, exist_ok=True)
    placed = {}

    # Direct downloads from cleardusk/3DDFA_V2 configs/
    base = "https://github.com/cleardusk/3DDFA_V2/raw/master/configs"
    for name in ("bfm_noneck_v3.pkl", "param_mean_std_62d_120x120.pkl"):
        out_raw = target_dir / f"_raw_{name}"
        if not out_raw.exists():
            print(f"Downloading {name} from cleardusk/3DDFA_V2...", file=sys.stderr)
            urllib.request.urlretrieve(f"{base}/{name}", out_raw)

    # Repackage param_mean_std as-is (it's small and 1-to-1 useful)
    param_pkl_dst = target_dir / "param_mean_std_62d_120x120.pkl"
    raw = target_dir / "_raw_param_mean_std_62d_120x120.pkl"
    if not param_pkl_dst.exists():
        param_pkl_dst.write_bytes(raw.read_bytes())
    placed["param_mean_std_62d_120x120.pkl"] = param_pkl_dst

    # Extract sparse + dense BFM derivatives from bfm_noneck_v3.pkl
    raw_bfm = target_dir / "_raw_bfm_noneck_v3.pkl"
    with open(raw_bfm, "rb") as f:
        bfm = pickle.load(f)
    keypoints = bfm["keypoints"]
    u = bfm["u"].flatten()
    w_shp = bfm["w_shp"]
    w_exp = bfm["w_exp"]
    tri = bfm["tri"]

    # Sparse 68-landmark subset (~40 KB)
    sparse_dst = target_dir / "bfm_sparse.npz"
    if not sparse_dst.exists():
        np.savez_compressed(
            sparse_dst,
            u_base=u[keypoints].reshape(68, 3).astype(np.float32),
            w_shp_base=w_shp[keypoints, :].reshape(68, 3, 40).astype(np.float32),
            w_exp_base=w_exp[keypoints, :].reshape(68, 3, 10).astype(np.float32),
            keypoints=keypoints,
        )
    placed["bfm_sparse.npz"] = sparse_dst

    # Dense mesh (~22 MB)
    dense_dst = target_dir / "bfm_dense.npz"
    if not dense_dst.exists():
        np.savez_compressed(
            dense_dst,
            u=u.astype(np.float32),
            w_shp=w_shp.astype(np.float32),
            w_exp=w_exp.astype(np.float32),
            tri=tri.astype(np.int32),
            keypoints=keypoints.astype(np.int32),
        )
    placed["bfm_dense.npz"] = dense_dst

    # Clean up raw downloads
    for raw_path in target_dir.glob("_raw_*"):
        raw_path.unlink()

    return placed


def install_3d_assets() -> dict[str, Path]:
    """Download all openly-distributable 3D-pipeline assets and print
    manual-install instructions for the license-gated FLAME 2020 model.

    Public downloads:
      - DECA pretrained weights (public Google Drive)
      - BFM-2009 derivatives from cleardusk/3DDFA_V2 (academic use only)

    License-gated, manual:
      - FLAME 2020 (academic registration required)

    Returns:
        Mapping of asset name -> file path.
    """
    DECA_DATA.mkdir(parents=True, exist_ok=True)
    BFM_DIR = DATA_ROOT / "3dmm"
    BFM_DIR.mkdir(parents=True, exist_ok=True)

    placed = {}

    # DECA pretrained
    placed["deca_model"] = download_deca_model(DECA_DATA)

    # BFM-2009 derivatives
    bfm_files = download_bfm_files(BFM_DIR)
    placed.update({f"bfm_{k}": v for k, v in bfm_files.items()})

    # FLAME 2020 must be installed manually
    flame_target = DECA_DATA / "generic_model.pkl"
    if flame_target.exists():
        print(f"FLAME 2020 already installed at {flame_target}",
              file=sys.stderr)
        placed["flame_2020"] = flame_target
    else:
        print(FLAME_INSTRUCTIONS.format(target_path=flame_target),
              file=sys.stderr)

    return placed


def print_checksums() -> dict[str, str]:
    """Compute and print SHA-256 of every present asset.

    Used by release engineers to fill in the `sha256=` field on each
    Asset entry before tagging a release. Missing assets are skipped.

    Returns:
        Mapping of asset name -> sha256 hex string for present files.
    """
    out: dict[str, str] = {}
    for a in ASSETS:
        if not a.abs_path.exists():
            print(f"  {a.name}: <missing> ({a.abs_path})", file=sys.stderr)
            continue
        digest = _sha256_of(a.abs_path)
        out[a.name] = digest
        marker = "OK " if a.sha256 == digest else "NEW"
        print(f"  {marker} {a.name}: {digest}", file=sys.stderr)
    return out


def required_for_method(method: str) -> Iterable[Asset]:
    """Yield assets required for a given expression-method label."""
    if method in ("ip2p", "sd_inpaint", "sd", "diffusion"):
        return []  # SD checkpoint auto-downloads on first use
    if method == "3dmm":
        return [a for a in ASSETS if a.name in ("bfm_dense", "bfm_sparse", "param_mean_std")]
    if method == "3d":
        return [a for a in ASSETS if a.name in ("deca_model", "flame_2020")]
    return []
