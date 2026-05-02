# Installing ofiq-syngen

The package has tiered functionality. Pick the install that matches what you need.

## Tier 1 — basic 2D operators (always works)

```bash
pip install ofiq-syngen
```

Unlocks: every operator that uses pure cv2 / numpy paths
(BackgroundUniformity, Sharpness, JPEG, IlluminationUniformity, all
crop / margin / head-size operators with flat-backdrop fill, etc.).

You also need OFIQ's ONNX models for face context. Set the env var or
let the package auto-discover them at one of the standard paths
(see `OFIQ_MODEL_DIR` resolution order in `models.py`).

```bash
export OFIQ_MODEL_DIR=/path/to/OFIQ-Project/data/models
```

## Tier 2 — Stable Diffusion / InstructPix2Pix paths

```bash
pip install "ofiq-syngen[diffusion]"
```

Unlocks: photorealistic occluders (sunglasses, surgical mask, hat,
hand), photographic expression edits (smile / frown / surprise),
realistic side lighting, mouth opening with teeth, second-face
insertion via SD inpaint. Requires CUDA-capable GPU for usable speed
(~1 s per inference; CPU is ~60 s per inference).

Activate at runtime:
```bash
export OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p
```

The Stable Diffusion checkpoint (~5 GB) auto-downloads on first call
to `~/.cache/ofiq_syngen_sd/`.

## Tier 3 — full 3D head pose (FLAME / DECA)

```bash
pip install "ofiq-syngen[three_d]"
ofiq-syngen install-assets   # downloads public DECA pretrained
```

`install-assets` fetches the public DECA pretrained weights (~415 MB)
into `src/third_party/DECA/data/deca_model.tar`.

**FLAME 2020 must be installed manually.** It is license-gated under
the [FLAME 2020 Model License](https://flame.is.tue.mpg.de/modellicense.html)
(non-commercial, non-redistributable). ofiq-syngen does NOT download,
bundle, or proxy FLAME under any circumstances — each user must
register individually.

### FLAME 2020 manual install steps

1. Register an academic account at https://flame.is.tue.mpg.de/
2. Accept the FLAME 2020 license terms during registration
3. Wait for approval email (typically within 24 hours)
4. Sign in and click the "Download" tab
5. Download `FLAME2020.zip`
6. Unzip and copy `generic_model.pkl` to:

   ```
   <ofiq-syngen-install>/src/third_party/DECA/data/generic_model.pkl
   ```

   You can find the exact path by running `ofiq-syngen check-assets`.

7. Verify with `ofiq-syngen check-assets` — `flame_2020` should show
   `yes`.

Optional (same source, same license — only needed for gendered FLAME variants):
- `male_model.pkl`
- `female_model.pkl`

Without FLAME, `HeadPoseYaw` / `HeadPosePitch` fall back to the 2D
TPS warp on the dense BFM mesh (shipped with the package, not
license-gated).

## All tiers + dev tooling

```bash
pip install "ofiq-syngen[all,dev]"
ofiq-syngen install-assets   # downloads DECA only
# then manually install FLAME 2020 (see Tier 3 above)
```

## Verifying the install

```bash
ofiq-syngen --version
ofiq-syngen check-assets
ofiq-syngen list-components
```

`check-assets` shows every optional asset, whether it's installed,
and what operator paths it unlocks.

## Quick test

```bash
ofiq-syngen degrade input.jpg --component Sharpness.scalar --severity 0.6 --output blurred.jpg
ofiq-syngen sweep input.jpg --component HeadPoseYaw.scalar --levels 5 --output-dir ./sweep
```

## CUDA toolkit (only for nvdiffrast)

The optional `nvdiff` extra requires CUDA toolkit (`nvcc`) at install
time:

```bash
pip install "ofiq-syngen[nvdiff]"  # builds nvdiffrast from source
```

You can install nvcc without sudo from NVIDIA's pip-distributed
toolkit:

```bash
pip install nvidia-cuda-nvcc-cu12
# or, for a self-contained install, download CUDA toolkit redistributable
# archive from https://developer.download.nvidia.com/compute/cuda/redist/
```
