# Per-Component Status

Single-page rollup of every component, combining standards mapping
(from [`MAPPING.md`](../standards/MAPPING.md)), OFIQ source provenance
(from [`PROVENANCE.md`](../standards/PROVENANCE.md)), test coverage,
and parity-vector status.

This is the table to consult when:
- Triaging a regression: which dimensions are checked, which are not.
- Drafting a paper section: what defends the per-component claim.
- Planning future work: where coverage is thin.

For algorithm details, see [`ANALYSIS.md`](../ANALYSIS.md).
For citation guidance, see [`../CITING.md`](../CITING.md).

## Status table

`P` = parametrized per-component test row.
`X` = covered by global iterate-inside test.
`-` = not covered yet.
`?` = uncertainty flagged (verify against source standard).

### Capture-related (FDIS §7.3)

| Component | Section | Align | Conf | OFIQ src lines | Smoke | Determ | Sev change | Sev mono | Parity vec |
|---|---|---|---|---|---|---|---|---|---|
| BackgroundUniformity | §7.3.2 | exact | verified | BackgroundUniformity.cpp:55-168 | P | P | P | P | - |
| IlluminationUniformity | §7.3.3 | exact | verified | IlluminationUniformity.cpp:1-113 | P | P | P | P | - |
| LuminanceMean | §7.3.4.2 | exact | verified | Luminance.cpp:41-76 | P | P | P | P | - |
| LuminanceVariance | §7.3.4.3 | partial | verified | Luminance.cpp:41-76 | P | P | P | P | - |
| UnderExposurePrevention | §7.3.5 | exact | verified | UnderExposurePrevention.cpp:1-72 | P | P | P | P | - |
| OverExposurePrevention | §7.3.6 | exact | verified | OverExposurePrevention.cpp:1-71 | P | P | P | P | - |
| DynamicRange | §7.3.7 | partial | verified | DynamicRange.cpp:1-88 | P | P | P | P | - |
| Sharpness (3 variants) | §7.3.8 | exact | verified | Sharpness.cpp:1-214 | P | P | P | P | - |
| CompressionArtifacts | §7.3.9 | exact | verified | CompressionArtifacts.cpp:1-111 | P | P | P | P | - |
| NaturalColour | §7.3.10 | exact | verified | NaturalColour.cpp:1-176 | P | P | P | P | - |
| RadialDistortion | Annex D.2.1 | exact | verified | (not in OFIQ 1.1.0) | P | P | P | P | - |
| SingleFacePresent | S6 ? | exact | uncertain | SingleFacePresent.cpp:1-100 | P | P | P | exempt | - |

### Subject-related (FDIS §7.4)

| Component | Section | Align | Conf | OFIQ src lines | Smoke | Determ | Sev change | Sev mono | Parity vec |
|---|---|---|---|---|---|---|---|---|---|
| EyesOpen | §7.4.3 | exact | verified | EyesOpen.cpp:1-61 | P | P | ctx-skip | ctx-skip | - |
| MouthClosed | §7.4.4 | exact | verified | MouthClosed.cpp:1-70 | P | P | ctx-skip | ctx-skip | - |
| EyesVisible | §7.4.5 | exact | verified | EyesVisible.cpp:1-123 | P | P | ctx-skip | ctx-skip | - |
| MouthOcclusionPrevention | §7.4.6 | exact | verified | MouthOcclusionPrevention.cpp:1-67 | P | P | ctx-skip | ctx-skip | - |
| FaceOcclusionPrevention | §7.4.7 | exact | verified | FaceOcclusionPrevention.cpp:1-74 | P | P | P | P | - |
| InterEyeDistance | §7.4.8 | exact | verified | InterEyeDistance.cpp:1-71 | P | P | P | exempt | - |
| HeadSize | §7.4.9 | exact | verified | HeadSize.cpp:1-61 | P | P | P | exempt | - |
| ExpressionNeutrality | §7.4.12 | exact | uncertain | ExpressionNeutrality.cpp:1-154 | P | P | ctx-skip | ctx-skip | - |
| NoHeadCoverings | §7.4.13 | partial | uncertain | NoHeadCoverings.cpp:1-123 | P | P | P | exempt | - |

### Geometric / pose (S8.x)

| Component | Section | Align | Conf | OFIQ src lines | Smoke | Determ | Sev change | Sev mono | Parity vec |
|---|---|---|---|---|---|---|---|---|---|
| HeadPoseYaw | §7.4.11.2 | exact | verified | HeadPose.cpp:49-59 | P | P | P | P | - |
| HeadPosePitch | §7.4.11.3 | exact | verified | HeadPose.cpp:49-59 | P | P | P | P | - |
| HeadPoseRoll | §7.4.11.4 | exact | verified | HeadPose.cpp:49-59 | P | P | P | P | - |
| LeftwardCropOfTheFaceImage | §7.4.10.1 | exact | verified | CropOfTheFaceImage.cpp:69-113 | P | P | P | exempt | - |
| RightwardCropOfTheFaceImage | §7.4.10.2 | exact | verified | CropOfTheFaceImage.cpp:69-113 | P | P | P | exempt | - |
| MarginAboveOfTheFaceImage | §7.4.10.3 | exact | verified | CropOfTheFaceImage.cpp:69-113 | P | P | P | exempt | - |
| MarginBelowOfTheFaceImage | §7.4.10.4 | exact | verified | CropOfTheFaceImage.cpp:69-113 | P | P | P | exempt | - |

## Coverage rollup

| Dimension | 28 / 28 | Notes |
|---|---|---|
| Smoke (function runs, shape preserved) | 28 / 28 | Test rows in `test_components_per_component.py::test_smoke` |
| Determinism (same seed, same output) | 28 / 28 | `test_deterministic_same_seed` |
| Severity changes output | 23 / 28 | 5 ctx-noop components (Eyes/Mouth/Expression) skipped without FaceContext |
| Severity monotonicity (image delta) | 15 / 28 | 5 ctx-skip + 8 monotonicity-exempt (geometric translations + content-overlay) |
| Output finite, in-range | 28 / 28 | `test_output_finite_and_in_range` |
| Size invariance (64 / 112 / 256) | 28 / 28 | `test_works_on_small_image` and `test_works_on_large_image` |
| OFIQ parity vectors | 0 / 28 | Manifest empty; see [`tests/fixtures/ofiq_parity/REGENERATE.md`](https://github.com/astoreyai/ofiq-syngen/blob/main/tests/fixtures/ofiq_parity/REGENERATE.md) |

## Known limitations

1. **`?`-confidence components** (`SingleFacePresent`, `ExpressionNeutrality`,
   `NoHeadCoverings`): clause numbers in ISO/IEC 29794-5:2024 inferred,
   not sourced from the Algorithm Book PDF. Remove the `?` once verified.
2. **`partial`-alignment components** (`LuminanceVariance`, `DynamicRange`,
   `NoHeadCoverings`): the perturbation reaches the measurement region
   but does not perfectly invert the metric. Documented in `MAPPING.md`.
3. **`MODERATE` algorithm fidelity** (per [`ANALYSIS.md`](../ANALYSIS.md)):
   `SingleFacePresent`, `HeadPoseYaw`, `HeadPosePitch`,
   `ExpressionNeutrality`, `NoHeadCoverings`. The perturbation method is
   honest about its approximation: e.g., perspective warp for yaw rather
   than true 3D rotation. Where a higher-fidelity alternative exists
   (DDPM-based generative editing) the docs note the trade-off.
4. **No OFIQ-binary parity vectors yet.** The manifest is scaffolded;
   regenerating requires a built OFIQ binary plus a face image set.

## How to extend this table

When adding a new degrader:

1. Add the component to `src/ofiq_syngen/standards.py` `STANDARDS_REFS`.
2. Register the function in `src/ofiq_syngen/components.py`.
3. The parametrized test suite picks it up automatically (no test
   editing required for smoke / determinism / severity-change /
   monotonicity / finiteness / size invariance).
4. Update `MAPPING.csv` with the new row.
5. Append a row to the appropriate section of this status table with
   the OFIQ source citation.
6. Regenerate the parity manifest if a binary is available.
