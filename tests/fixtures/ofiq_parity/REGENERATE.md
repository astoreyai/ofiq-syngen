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

The script and the test both invoke the OFIQ 1.1.0 SampleApp directly.
The OFIQ shared libraries live next to the binary, so set
`LD_LIBRARY_PATH` accordingly. **Also set `OFIQ_MODEL_DIR`** so the
syngen pipeline can build a FaceContext — several operators (e.g.,
`DynamicRange`, `UnderExposurePrevention`) use `ctx.face_mask` for
masked / autoscaled effects, and the OFIQ scores will differ from
the no-ctx defaults if you skip this.

```bash
OFIQ_BIN=OFIQ-Project/install_x86_64_linux/Release/bin
LD_LIBRARY_PATH=$OFIQ_BIN \
OFIQ_MODEL_DIR=OFIQ-Project/data/models \
python scripts/regenerate_parity_vectors.py \
    --ofiq-binary $OFIQ_BIN/OFIQSampleApp \
    --image-dir tests/fixtures/ofiq_parity/images \
    --severities 0.0,0.5,1.0 \
    --output tests/fixtures/ofiq_parity/manifest.json
```

To regenerate only a subset (e.g., after changing a single operator)
pass `--components`:

```bash
... --components LuminanceMean.scalar,DynamicRange.scalar
```

For each (image, component, severity) triple the script:
1. Runs `ofiq_syngen.degrade_single` to produce a degraded image.
2. Pipes the degraded image through the OFIQ binary.
3. Records `(image_id, component, severity, expected_score)` in the manifest.

## Verify

The test reads `OFIQ_BINARY` from the environment and skips entirely if
unset:

```bash
OFIQ_BINARY=OFIQ-Project/install_x86_64_linux/Release/bin/OFIQSampleApp \
LD_LIBRARY_PATH=OFIQ-Project/install_x86_64_linux/Release/bin \
pytest tests/test_ofiq_parity.py -v
```

In CI the test is opt-in: builds without OFIQ pre-installed see all
parity vectors as `SKIPPED`.

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
