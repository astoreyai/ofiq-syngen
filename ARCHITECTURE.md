# ofiq-syngen Architecture

## What this package does

For each of the 28 ISO/IEC 29794-5 (OFIQ) face image quality components,
`ofiq-syngen` ships a **degradation operator**: a pure function that
takes a source face image and a severity in `[0, 1]`, and produces a
degraded image whose OFIQ scalar moves in the targeted direction. The
package is the substrate for the four-link causal chain in the
`world/` thesis (Link 1 = degrade, Link 2 = re-score, Link 3 = re-embed,
Link 4 = leaderboard).

## Module map

```
src/ofiq_syngen/
├── __init__.py          public API (DegradationPipeline, DegradationConfig)
├── cli.py               argparse front-end (degrade / sweep / list / install-assets)
├── components.py        registry: name -> degradation function (28 entries)
├── pipeline.py          DegradationPipeline (single / sweep / all-components)
├── face_context.py      FaceContext: cached landmarks, masks, eye centers
├── landmark_utils.py    98-pt ADNet landmark schema, BiSeNet parsing labels
├── proxies.py           non-binary OFIQ scalar proxies (Sobel, entropy, etc.)
├── standards.py         multi-standard cross-reference (29794, 19794, ICAO)
├── assets.py            asset registry, OFFLINE switch, SHA-256 verification
├── face_3dmm.py         62-dim 3DDFA-V2 parameter parser + sparse landmark proj
├── face_3dmm_dense.py   dense BFM mesh + TPS warp for 2D head pose
├── expression_diffusion.py    InstructPix2Pix / SD inpaint expression edit
├── occluders.py         medical mask, sunglasses, head covering overlays
├── generative/
│   ├── expression.py    method dispatch (3dmm / ip2p / sd_inpaint / tps)
│   ├── single_face.py   second-face Poisson blending
│   └── head_covering.py synthetic hat overlays
└── three_d/             3D FLAME head pose pipeline (optional [three_d] extra)
    ├── pipeline.py      DegradationPipeline (3D variant)
    ├── lift/deca.py     DECA encoder wrapper (image -> FLAME params)
    ├── render/          pyrender + nvdiffrast back-ends
    └── standards.py     three_d-specific dispatch
```

## Tier dispatch (3D head pose)

`HeadPoseYaw` and `HeadPosePitch` operators try three back-ends in order:

```
                 +--------+        +-------+        +-------+
   image, sev -> |  3D    | -OK--> | done  |        |       |
                 |  FLAME |        +-------+        |       |
                 +---+----+                         |       |
                     | fail                         |       |
                     v                              |       |
                 +--------+        +-------+        |       |
                 |  2D    | -OK--> | done  |        |       |
                 |  TPS   |        +-------+        |       |
                 +---+----+                         |       |
                     | fail                         |       |
                     v                              |       |
                 +--------+        +-------+        |       |
                 | persp- | -OK--> | done  |        |       |
                 | ective |        +-------+        +-------+
                 | squeeze|
                 +--------+
```

Tier 1 (3D FLAME): true 3D rotation via DECA mesh fit + pyrender.
Source-textured mesh, real anatomical rotation, sev 0..1 → 0..35°.
Requires `[three_d]` extra and FLAME 2020 + DECA pretrained.

Tier 2 (2D TPS dense BFM): warps the source image through a thin-plate
spline field seeded by the projected BFM mesh at the original vs.
rotated pose. Capped at ±10° because a 2D warp cannot synthesize
disocclusions. Requires the BFM-2009 derivatives (`bfm_dense.npz`).

Tier 3 (perspective squeeze): pure 2D affine that simulates yaw by
non-uniformly compressing the image horizontally. Capped at ±0.5
severity because the geometry stops looking like a head past that.
Always available — no extras.

The same tier pattern applies to expression (`add_expression`):
3DMM landmark morph → IP2P / SD inpaint → TPS landmark warp.

## Asset lifecycle

Assets fall into three categories:

| Asset | Distribution | License gate |
|---|---|---|
| OFIQ ONNX models | Sibling repo or `~/.ofiq/models/` | none (BSI public) |
| BFM-2009 derivatives | Downloaded from cleardusk/3DDFA_V2 GitHub | academic (BFM-2009) |
| DECA pretrained | Downloaded from public Google Drive | academic (DECA) |
| FLAME 2020 | **User-installed only** | non-commercial (Max Planck) |
| Stable Diffusion | Auto-download on first IP2P / SD use | CreativeML OpenRAIL |

The `assets.py` module owns the asset table, OFFLINE kill switch, and
SHA-256 verification. Pinned hashes for the BFM derivatives ship in
`ASSETS`; `check-assets --print-checksums` computes hashes for the
release engineer to fill in for new assets before tagging a release.

`OFIQ_SYNGEN_OFFLINE=1` aborts every network fetch with a clear error
naming the asset to pre-stage. Use this in air-gapped environments and
in CI to verify nothing implicitly hits the network.

## Determinism

Every operator accepts a `seed: int`. Given identical `(img, severity, seed)`
the output is byte-identical across runs on the same platform/Python
version. The pipeline does not consume entropy outside what callers
provide — no `time.time()`, no `os.urandom()`, no `np.random` global
state. This is load-bearing for the reproducibility of the four-link
causal chain in the thesis.

## Test layout

| Test file | What it covers |
|---|---|
| `test_degradation_direction.py` | Each operator moves its target OFIQ scalar in the correct direction on a synthetic face fixture |
| `test_pipeline.py` | DegradationPipeline single / sweep / all-components |
| `test_components.py` | COMPONENT_REGISTRY completeness (all 28 components present) |
| `test_standards.py` | ICAO / ISO 19794 / ISO 29794 cross-reference integrity |
| `test_landmark_utils.py` | ADNet 98-point schema, BiSeNet parsing labels |
| `test_assets.py` | OFFLINE switch, SHA-256 verification |
| `tests/fixtures/ofiq_parity/` | Reference vectors from the OFIQ binary (V1–V4 parity tests; require OFIQ install) |

OFIQ V1–V4 parity vectors (synthetic-fixture vs. the OFIQ binary) are a
release-blocker but cannot be exercised without an OFIQ install — see
`OFIQ_UPSTREAM.md` for the local install procedure.

## Extension points

Adding a new degradation operator:

1. Implement `def _new_operator(img, severity, seed, ctx) -> np.ndarray`
   in `components.py`.
2. Add an entry to `COMPONENT_REGISTRY` mapping the OFIQ component name
   to the operator and a one-line description.
3. Add a row to `docs/standards/MAPPING.csv` and regenerate
   `docs/standards/MAPPING.md`.
4. Add a directional test to `test_degradation_direction.py`.
5. Render a 5-step severity strip into `docs/gallery/images/<Component>_strip.png`
   and write the matching `docs/gallery/<Component>.md`.
