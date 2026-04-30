# Competitor catalog

For every component, the alternatives `ofiq-syngen` chose its
perturbation against. Used as the input set for the head-to-head
benchmark grid (`benchmarks/run_grid.py`).

## Catalog convention

Per component:

- **Method**: name of the alternative perturbation approach.
- **Origin**: paper / library where the method appears.
- **License**: of the reference implementation.
- **Cost**: relative ops/parameter cost (S=small, M=medium, L=large).
- **Determinism**: whether the method is seed-deterministic.
- **Honest assessment**: where the alternative likely beats syngen,
  where syngen likely beats it. Filled in after the benchmark grid
  runs (`benchmarks/results/grid.parquet`); placeholder for now.

Methods not in this catalog were considered and discarded for one of:

- Unfair comparison (a 200M-parameter diffusion model vs a 3-line
  OpenCV operation is not a like-for-like cost comparison; documented
  in `BENCHMARK_NARRATIVE.md` once that file exists).
- License-incompatible reference implementations.
- No public reference implementation we could adapt.

## Capture-related (S6.x)

### `BackgroundUniformity.scalar` (S6.1)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Border-strip noise (syngen v0.1) | this package, deprecated | MIT | S | yes | Less precise than BiSeNet-segmented; kept as ctx-free fallback |
| BiSeNet-segmented noise (syngen v0.2+) | this package | MIT | S+ONNX | yes | Current implementation |
| 3D rendered backgrounds | DigiFace-1M | research-only | L | no | High realism, low control |
| Diffusion background swap | Stable Diffusion + ControlNet | CreativeML | XL | no | Very high realism, very low control |

### `IlluminationUniformity.scalar` (S6.2)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| ROI-zone darkening (syngen) | this package | MIT | S+ONNX | yes | Current |
| Global gamma | albumentations | MIT | S | yes | Simpler; less precise region targeting |
| Spherical-harmonics relighting | DECA / FLAME 3DMM | non-commercial | L | partial | Physics-based, requires 3DMM fit |
| Diffusion-based relight | LightGlue / IC-Light | non-commercial | XL | no | High realism |

### `LuminanceMean.scalar`, `LuminanceVariance.scalar` (S6.3)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Face-mask gamma (syngen) | this package | MIT | S+ONNX | yes | Current |
| Whole-image gamma | OpenCV / albumentations | MIT | S | yes | Easier, less precise |
| Histogram matching | scikit-image | BSD | S | yes | Different mechanism, similar effect |

### `Sharpness.scalar` (S6.6)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Gaussian blur (syngen) | this package | MIT | S | yes | Current; one of three syngen variants |
| Motion blur (syngen) | this package | MIT | S | yes | Current |
| Additive Gaussian noise (syngen) | this package | MIT | S | yes | Current |
| Real-ESRGAN second-order degradation | Real-ESRGAN | BSD | M | yes | Multi-pass; higher realism |
| GFPGAN ffhq_degradation | GFPGAN | Apache 2.0 | M | yes | Reference template; we follow its structure |
| BSRGAN shuffled order | BSRGAN | Apache 2.0 | M | yes | Random pipeline ordering |
| Diffusion blur | DiffJPEG, DDIM-Inversion | various | L | no | Generative blur |

### `CompressionArtifacts.scalar` (S6.7)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| JPEG re-encode (syngen) | this package | MIT | S | yes | Current |
| JPEG2000 re-encode | OpenCV | MIT | S | yes | Different codec, OFIQ may respond differently |
| HEIC / WebP re-encode | various | various | S | yes | Modern codecs, less artifact at same bitrate |
| Differentiable JPEG | DiffJPEG | MIT | M | yes | Differentiable for adversarial training |

### `NaturalColour.scalar` (S6.8)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| CIELAB ROI-zone shift (syngen) | this package | MIT | S+ONNX | yes | Current |
| Whole-image CIELAB shift | OpenCV | MIT | S | yes | Less precise region |
| Color-temperature shift | albumentations | MIT | S | yes | Simulates white-balance error |
| Diffusion color edit | Stable Diffusion img2img | CreativeML | XL | no | High realism |

## Subject-related (S7.x)

### `EyesOpen.scalar`, `MouthClosed.scalar` (S7.2 / S7.3)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| RBF landmark warp (syngen) | this package | MIT | S+ONNX | yes | Current; requires FaceContext |
| StyleGAN latent edit | StyleGAN3 + InterFaceGAN | non-commercial | L | no | High realism, non-deterministic |
| Eyelid mesh deform | DECA | non-commercial | L | yes | Physics-based |

### `EyesVisible.scalar`, `MouthOcclusionPrevention.scalar`, `FaceOcclusionPrevention.scalar` (S7.4-S7.6)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Mask-targeted occlusion (syngen) | this package | MIT | S+ONNX | yes | Current |
| NatOcc (sunglasses, hands) | face-occlusion-generation | research | M | yes | Naturalistic accessories |
| RandOcc (random patches) | face-occlusion-generation | research | S | yes | Synthetic patches |
| Diffusion inpainting | Stable Diffusion inpaint | CreativeML | XL | no | Removes face region with realistic fill |

### `InterEyeDistance.scalar`, `HeadSize.scalar` (S7.7 / S7.8)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Pad-and-shrink (syngen) | this package | MIT | S | yes | Current |
| Bicubic downsample-upsample | OpenCV | MIT | S | yes | Same effect on IED, may also affect Sharpness |

### `ExpressionNeutrality.scalar` (S7.9)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| RBF landmark warp (syngen) | this package | MIT | S+ONNX | yes | Current |
| 3DMM expression transfer | DECA / FLAME | non-commercial | L | yes | Expression code from another image |
| Diffusion expression edit | InstructPix2Pix | CreativeML | XL | no | Text-guided edit |

### `NoHeadCoverings.scalar` (S7.10)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Texture overlay (syngen fallback) | this package | MIT | S | yes | Ctx-free fallback |
| Parsing-map-targeted hat overlay (syngen) | this package | MIT | S+ONNX | yes | Current with FaceContext |
| StyleGAN attribute edit | StyleGAN + InterFaceGAN | non-commercial | L | no | "+hat" direction in latent space |
| Diffusion accessory inpaint | Stable Diffusion + ControlNet | CreativeML | XL | no | High realism |

### `SingleFacePresent.scalar` (S6 ?)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Poisson-blend insert (syngen) | this package | MIT | S+insightface | partial | Current; requires insightface |
| Copy-paste insert (syngen fallback) | this package | MIT | S | yes | Ctx-free fallback; visually obvious |
| Diffusion add-second-face | Stable Diffusion + face-condition | CreativeML | XL | no | Highest realism |

## Geometric / pose (S8.x)

### `HeadPoseYaw.scalar`, `HeadPosePitch.scalar` (S8)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Perspective warp (syngen) | this package | MIT | S | yes | Current; MODERATE alignment |
| 3DMM rotation render | DECA / FLAME | non-commercial | L | yes | True 3D, expensive |
| StyleGAN3 pose edit | EG3D / StyleGAN3 | non-commercial | L | no | Generative |

### `HeadPoseRoll.scalar` (S8)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| 2D affine rotation (syngen) | this package | MIT | S | yes | Current; exact for roll |

### `LeftwardCropOfTheFaceImage`, `RightwardCropOfTheFaceImage`, `MarginAboveOfTheFaceImage`, `MarginBelowOfTheFaceImage` (S8)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Directional shift (syngen) | this package | MIT | S | yes | Current; one function per direction |
| Affine translation | OpenCV | MIT | S | yes | Equivalent mechanism |

### `RadialDistortion.scalar` (S6.9 forward-looking)

| Method | Origin | License | Cost | Det | Notes |
|---|---|---|---|---|---|
| Barrel distortion remap (syngen) | this package | MIT | S | yes | Current |
| Brown-Conrady model | OpenCV undistort inverse | MIT | S | yes | Same family, different parameterization |

## Notes on selection bias

This catalog is curated from the perspective of "what alternatives
exist for a degrader targeting OFIQ alignment." It is biased toward:

- Lightweight methods (the package's design constraint is low
  dependency footprint).
- Deterministic methods (calibration sweeps need reproducibility).
- Methods with a published reference implementation.

Excluded categories:

- **Heavy generative-only methods** for which a fair head-to-head
  cost comparison is not meaningful at the per-image scale this
  package operates on. These are surveyed in `docs/CITING.md` and
  the systematic-review companion paper.
- **Closed-source SDKs** (TONO, ONOT, commercial OFIQ implementations).
- **Methods without parametric severity control**, which cannot be
  swept the way `ofiq-syngen` is.

## What the benchmark grid will tell us

For each (component, method, metric) cell:

- WIN: ours beats the alternative within statistical significance.
- TIE: within noise band.
- LOSE: alternative wins; rationale documented in
  `BENCHMARK_NARRATIVE.md` (e.g., "alternative wins on visual realism
  but requires 200 MB of model weights and is non-deterministic").

Honest LOSE entries are not weaknesses; they are where the package
chose simplicity / determinism / dependency-light over absolute
fidelity.

The grid run is `benchmarks/run_grid.py`. Adapter implementations live
in `benchmarks/adapters/`. The full grid is GPU-bound and not run by
CI; results land in `benchmarks/results/grid.parquet` when regenerated.
