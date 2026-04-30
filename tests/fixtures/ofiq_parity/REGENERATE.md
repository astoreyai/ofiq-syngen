# Regenerating OFIQ parity vectors

This fixture set pairs every (component, severity) tuple with the OFIQ
score it should produce when measured by the OFIQ binary. The test
`tests/test_ofiq_parity.py` reads this manifest and asserts that every
recorded score is within tolerance.

The manifest is **empty by default**. To populate it:

## Prerequisites

1. **OFIQ binary**, built or downloaded. Build steps:
   ```bash
   cd OFIQ-Project
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   cmake --build . -j
   ```
   The resulting binary is at `OFIQ-Project/build/install_x86_64_linux/Release/bin/OFIQSampleApp`.

2. **OFIQ models** in the model directory (typically `OFIQ-Project/data/models/`). These ship with the OFIQ source tree.

3. **A canonical face image set** (10 images recommended). Use FFHQ
   samples or any license-clean source. Place under
   `tests/fixtures/ofiq_parity/images/`.

## Regenerate

```bash
python scripts/regenerate_parity_vectors.py \
    --ofiq-binary /path/to/OFIQSampleApp \
    --image-dir tests/fixtures/ofiq_parity/images \
    --severities 0.0,0.25,0.5,0.75,1.0 \
    --output tests/fixtures/ofiq_parity/manifest.json
```

For each (image, component, severity) triple the script:
1. Runs `ofiq_syngen.degrade_single` to produce a degraded image.
2. Pipes the degraded image through the OFIQ binary.
3. Records `(image_id, component, severity, expected_score)` in the manifest.

## Verify

```bash
pytest tests/test_ofiq_parity.py -v
```

## Tolerance

Default tolerance is `+/- 5` raw OFIQ score (0-100 scale). Tighten in
`manifest.json` once empirical noise is characterized; the field is
`"tolerance"` at top level. Setting it to `2.0` or lower is reasonable
for deterministic non-context-dependent components; CNN-based components
(Sharpness, CompressionArtifacts, ExpressionNeutrality) have run-to-run
variance that justifies a wider tolerance.

## What this test catches

- Drift in a degrader's perturbation that moves the OFIQ score outside
  the recorded envelope.
- OFIQ release changes that move the measured score for the same input.
- Bugs in `landmark_utils` or `face_context` that change which region
  the perturbation targets.

## What this test does NOT catch

- Visual realism. A perturbation that produces the right OFIQ score but
  unrealistic pixels still passes.
- Cross-component side effects. See the cross-talk benchmark for that.
- Out-of-distribution inputs (extreme lighting, partial faces, etc.).

## Conformance bundle

The same manifest is exported by `ofiq-syngen export-conformance` so
external OFIQ-aligned implementations (US-FIQA, future ports) can run
the same parity assertions against their own measurement chain.
