# Third-Party License Notices

ofiq-syngen itself is licensed under MIT (see `LICENSE`). It depends on
several third-party models and datasets with restrictive licenses. None
of these are bundled in the wheel or sdist; users obtain their own copy
via `ofiq-syngen install-assets` (public sources) or by manual
registration (license-gated sources).

## License-gated assets (manual download required)

### FLAME 2020 (Max Planck Institute)

- **License**: Non-commercial research only — no redistribution
  https://flame.is.tue.mpg.de/modellicense.html
- **What it unlocks**: 3D HeadPoseYaw / HeadPosePitch via FLAME mesh fit
- **How to obtain**: Register at https://flame.is.tue.mpg.de/ with an
  academic email, accept the license, download `FLAME2020.zip`,
  manually place `generic_model.pkl` at the path shown by
  `ofiq-syngen check-assets`.
- **ofiq-syngen does NOT download, bundle, cache, or proxy FLAME.**

## Academic-license assets (public download, license still applies)

### DECA pretrained weights (Yao Feng et al., MIT)

- **License**: MIT (CODE) — model weights ship under research use
  https://github.com/YadiraF/DECA/blob/master/LICENSE
- **What it unlocks**: 3D face fit for HeadPose pipeline
- **How to obtain**: `ofiq-syngen install-assets` downloads from the
  official Google Drive link (file ID `1rp8kdyLPvErw2dTmqtjISRVvQLj6Yzje`).

### BFM-2009 (Basel Face Model, University of Basel)

- **License**: Academic research only — no redistribution
  https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=agreement
- **What it unlocks**: 2D head pose TPS warp (fallback path), 3DMM
  parameter normalization
- **How to obtain**: `ofiq-syngen install-assets` downloads via the
  3DDFA-V2 distribution (cleardusk/3DDFA_V2, MIT-distributed but
  underlying BFM license still applies). By running `install-assets`
  you confirm you have read and accepted the BFM-2009 license.

## Public datasets used as runtime inputs

These are not redistributed by ofiq-syngen but are referenced in
examples and the `SingleFacePresent` operator:

### CelebA / CelebA-HQ (Liu et al.)

- **License**: Academic research only
  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- **Used by**: example notebooks, parity test fixtures (3 sample images
  bundled in `tests/fixtures/ofiq_parity/images/`)

### VGGFace2 (Cao et al.)

- **License**: Non-commercial research only
- **Used by**: `SingleFacePresent.scalar` operator references VGGFace2
  identities via the `OFIQ_SYNGEN_SECOND_FACE_DIR` env var. ofiq-syngen
  does NOT bundle VGGFace2 — users provide their own license-compliant
  copy.

### FFHQ (Karras et al.)

- **License**: Creative Commons BY-NC-SA 4.0
- **Status**: Not currently used; reserved for future face-library work.

## Optional GPU model checkpoints (auto-download from HuggingFace)

When `OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p` (or `sd_inpaint`) is set, the
package's `expression_diffusion.py` lazily loads:

### Stable Diffusion Inpainting (RunwayML)

- **HF model id**: `runwayml/stable-diffusion-inpainting`
- **License**: CreativeML Open RAIL-M (research / non-malicious use)
  https://huggingface.co/runwayml/stable-diffusion-inpainting
- **Auto-cached at**: `~/.cache/ofiq_syngen_sd/` on first use

### InstructPix2Pix (Tim Brooks)

- **HF model id**: `timbrooks/instruct-pix2pix`
- **License**: CreativeML Open RAIL-M
  https://huggingface.co/timbrooks/instruct-pix2pix
- **Auto-cached at**: `~/.cache/ofiq_syngen_sd/`

To run fully offline (no auto-downloads), set `OFIQ_SYNGEN_OFFLINE=1`
and ensure all required assets are pre-installed via
`ofiq-syngen install-assets`.

## OFIQ ONNX models (BSI)

`ofiq_syngen.face_context` calls the OFIQ ONNX models (ADNet, BiSeNet,
HeadPose3DDFAV2, occlusion segmentation) for region-targeted operators.
These are part of the BSI OFIQ project:

- **Source**: https://github.com/BSI-OFIQ/OFIQ-Project
- **License**: Per OFIQ-Project repository (Apache 2.0 for code; model
  weights have their own constraints)
- **How to obtain**: OFIQ ships these models in its `data/models/`
  directory. Set `OFIQ_MODEL_DIR=/path/to/OFIQ-Project/data/models`.

## Summary of attribution requirements

| Asset | If used in publication |
|---|---|
| FLAME 2020 | Cite Li et al. 2017 (FLAME). |
| DECA | Cite Feng et al. 2021 (DECA). |
| BFM-2009 | Cite Paysan et al. 2009 (BFM). |
| CelebA | Cite Liu et al. 2015 (CelebA). |
| VGGFace2 | Cite Cao et al. 2018 (VGGFace2). |
| Stable Diffusion | Cite Rombach et al. 2022 (LDM). |
| InstructPix2Pix | Cite Brooks et al. 2023 (IP2P). |
| OFIQ | Cite BSI OFIQ-Project; ISO/IEC 29794-5:2024. |

Full BibTeX in `CITATIONS.bib` (TODO).
