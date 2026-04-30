# ofiq-syngen

[![PyPI version](https://img.shields.io/pypi/v/ofiq-syngen.svg)](https://pypi.org/project/ofiq-syngen/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ISO/IEC 29794-5 component-aligned synthetic face image quality degradation pipeline.**

`ofiq-syngen` generates controlled quality degradations mapped to specific ISO/IEC 29794-5 (OFIQ) quality components. Each degradation function targets a known OFIQ scalar component -- Sharpness, CompressionArtifacts, HeadPoseYaw, etc. -- producing ground-truth (image, degradation, delta-OFIQ) triplets for training quality-aware biometric systems. The pipeline covers 25 of the 27 OFIQ scalar components with 27 deterministic, seed-reproducible degradation functions.

## Installation

```bash
pip install ofiq-syngen
```

With pandas support (required for `generate_dataset` and influence matrix):

```bash
pip install ofiq-syngen[pandas]
```

## Quick Start

```python
import cv2
from ofiq_syngen import DegradationPipeline, DegradationConfig

# Load a face image (BGR, uint8)
image = cv2.imread("face.jpg")

# Create pipeline with default settings
pipeline = DegradationPipeline()

# Apply a single degradation
degraded, meta = pipeline.degrade_single(
    image, component="Sharpness.scalar", severity=0.6
)
cv2.imwrite("degraded.jpg", degraded)
print(meta)
# {'target_component': 'Sharpness.scalar', 'degradation_type': 'Gaussian blur [S6.6]',
#  'severity': 0.6, 'seed': 42}

# Sweep severity levels
config = DegradationConfig(severity_levels=[0.0, 0.25, 0.5, 0.75, 1.0])
pipeline = DegradationPipeline(config)
results = pipeline.degrade_sweep(image, "CompressionArtifacts.scalar")

# Apply one degradation per component
all_results = pipeline.degrade_all_components(image, severity=0.5)
```

## CLI Usage

```bash
# Apply a single degradation
ofiq-syngen degrade --component Sharpness --severity 0.6 --output degraded.jpg input.jpg

# Sweep 10 severity levels
ofiq-syngen sweep --component Sharpness --levels 10 --output-dir ./output input.jpg

# List all supported components
ofiq-syngen list-components

# List components in a specific standards preset
ofiq-syngen list-components --preset icao-strict

# Print the multi-standard cross-reference (29794-5, 19794-5, ICAO 9303 P9)
ofiq-syngen show-standards
ofiq-syngen show-standards --preset icao-strict

# Generate a full dataset (requires pandas)
ofiq-syngen generate-dataset --images-dir ./faces --output-dir ./degraded --max-images 100

# Generate a dataset filtered to a standards preset
ofiq-syngen generate-dataset --images-dir ./faces --preset icao-strict --output-dir ./icao
```

### Standards presets

`--preset` selects a curated component subset. Three presets ship today:

| Preset | Components | Source |
|---|---|---|
| `iso-29794-5` | All 28 | ISO/IEC 29794-5:2024 (default scope of the package) |
| `iso-19794-5` | All 28 | ISO/IEC 19794-5:2011 (every component maps to a 19794-5 clause) |
| `icao-strict` | 22 | Components with `alignment=exact` against ICAO 9303 Part 9 §3.2 |

Preset definitions live in `src/ofiq_syngen/standards.py` and trace back to
[`docs/standards/MAPPING.csv`](docs/standards/MAPPING.csv) (CSV is the source
of truth). See [`docs/standards/MAPPING.md`](docs/standards/MAPPING.md) for
the full per-component cross-reference.

## Component Coverage

25 of 27 OFIQ scalar components are covered. Each row maps to the OFIQ Algorithm Book (BSI V1.2) section reference.

| OFIQ Component | Section | Degradation | Severity Range |
|---|---|---|---|
| `BackgroundUniformity.scalar` | S6.1 | Background noise | intensity: 0 -> 200 |
| `IlluminationUniformity.scalar` | S6.2 | Gradient lighting | gradient: 0% -> 80% drop |
| `LuminanceMean.scalar` | S6.3 | Brightness reduction | factor: 1.0 -> 0.15 |
| `LuminanceVariance.scalar` | S6.3 | Variance compression | factor: 1.0 -> 0.1 |
| `UnderExposurePrevention.scalar` | S6.4 | Under-exposure | factor: 1.0 -> 0.15 |
| `OverExposurePrevention.scalar` | S6.4 | Over-exposure | factor: 1.0 -> 3.5 |
| `DynamicRange.scalar` | S6.5 | Dynamic range compression | range: 100% -> 10% |
| `Sharpness.scalar` | S6.6 | Gaussian blur | sigma: 0.5 -> 10.5 |
| `Sharpness.scalar` | S6.6 | Motion blur | kernel: 3 -> 31px |
| `Sharpness.scalar` | S6.6 | Additive Gaussian noise | sigma: 0 -> 80 |
| `CompressionArtifacts.scalar` | S6.7 | JPEG compression | Q: 100 -> 5 |
| `NaturalColour.scalar` | S6.8 | Color channel cast | offset: 0 -> 80 |
| `RadialDistortion.scalar` | S6.9 | Barrel distortion | k: 0 -> 0.5 |
| `EyesOpen.scalar` | S7.2 | Eye-region occlusion | band: 15% -> 30% |
| `MouthClosed.scalar` | S7.3 | Mouth-region occlusion | band: 20% -> 40% |
| `EyesVisible.scalar` | S7.4 | Eye-region occlusion | band: 15% -> 30% |
| `MouthOcclusionPrevention.scalar` | S7.5 | Mouth occlusion/mask | band: 20% -> 40% |
| `FaceOcclusionPrevention.scalar` | S7.6 | Rectangular occlusion | area: 0% -> 60% |
| `InterEyeDistance.scalar` | S7.7 | Resolution reduction | factor: 1.0 -> 0.1 |
| `HeadSize.scalar` | S7.8 | Resolution reduction | factor: 1.0 -> 0.1 |
| `HeadPoseYaw.scalar` | S8 | Simulated yaw rotation | squeeze: 0% -> 50% |
| `HeadPosePitch.scalar` | S8 | Simulated pitch tilt | squeeze: 0% -> 40% |
| `HeadPoseRoll.scalar` | S8 | In-plane rotation | angle: 0 -> +/-30 deg |
| `LeftwardCropOfTheFaceImage.scalar` | S8 | Crop offset | shift: 0% -> 30% |
| `RightwardCropOfTheFaceImage.scalar` | S8 | Crop offset | shift: 0% -> 30% |
| `MarginAboveOfTheFaceImage.scalar` | S8 | Crop offset | shift: 0% -> 30% |
| `MarginBelowOfTheFaceImage.scalar` | S8 | Crop offset | shift: 0% -> 30% |

### Not Covered (2/27)

| Component | Reason |
|---|---|
| `ExpressionNeutrality.scalar` | Requires facial Action Unit manipulation |
| `NoHeadCoverings.scalar` | Requires realistic overlay generation |

(`SingleFacePresent.scalar` requires face insertion and is also out of scope.)

## Severity Levels

All degradation functions accept a `severity` parameter in the range `[0.0, 1.0]`:

- **0.0** -- No degradation (identity or near-identity transform)
- **0.5** -- Moderate degradation (good default for influence matrix calibration)
- **1.0** -- Maximum degradation (image heavily distorted)

The mapping from severity to physical parameters is function-specific. For example, `Sharpness.scalar` with Gaussian blur maps severity 0.0 to sigma=0.5 and severity 1.0 to sigma=10.5. See the coverage table above for each function's range.

All functions are **deterministic** given a seed. Use the `seed` parameter for reproducible experiments.

## Entanglement and the Influence Matrix

A key design insight: degradations are **not independent**. Applying Gaussian blur (targeting Sharpness) also affects NaturalColour, DynamicRange, and other components. Pretending degradations are orthogonal leads to incorrect training signals.

The pipeline addresses this with an **empirical influence matrix** `W[d, c]`:

```
W[d, c] = mean(|delta_OFIQ_c|) when degradation d is applied at severity 0.5
```

To build it:

1. Generate a degradation dataset with `generate_dataset()`
2. Run OFIQ on all original and degraded images (externally)
3. Call `build_influence_matrix()` with the manifest and OFIQ scores

```python
from pathlib import Path
from ofiq_syngen import DegradationPipeline, DegradationConfig

pipeline = DegradationPipeline(DegradationConfig())

# Step 1: Generate degraded images
manifest = pipeline.generate_dataset(
    image_dir=Path("./faces"),
    output_dir=Path("./degraded"),
    max_images=100,
)

# Step 2: Run OFIQ externally, producing a CSV with scalar scores
# ofiq_scores = pd.read_csv("ofiq_results.csv")

# Step 3: Build the influence matrix
# matrix = DegradationPipeline.build_influence_matrix(
#     manifest, ofiq_scores, ofiq_scalar_cols=[...]
# )
```

The resulting matrix reveals cross-component effects and can be used to weight multi-task training losses appropriately.

## Standards Mapping

Every component carries a triple cross-reference: ISO/IEC 29794-5 (the
quality measurement standard OFIQ implements), ISO/IEC 19794-5 (the face
image data interchange format ICAO references), and ICAO Doc 9303 Part 9
(the travel-document specification).

| Doc | Path | Purpose |
|---|---|---|
| Mapping table (rendered) | [`docs/standards/MAPPING.md`](docs/standards/MAPPING.md) | Human-readable per-component cross-reference, grouped by OFIQ section |
| Mapping table (machine) | [`docs/standards/MAPPING.csv`](docs/standards/MAPPING.csv) | CSV source of truth used by the test suite |
| Source pinning | [`docs/standards/SOURCES.md`](docs/standards/SOURCES.md) | Standard editions referenced, OFIQ version pin, mapping confidence convention |
| OFIQ source provenance | [`docs/standards/PROVENANCE.md`](docs/standards/PROVENANCE.md) | Per-component citation of the OFIQ C++ `Execute()` line ranges |
| Per-component status | [`docs/theory/COMPONENT_STATUS.md`](docs/theory/COMPONENT_STATUS.md) | Single-page rollup: standards + provenance + test + parity status |
| ISO 29794-5 coverage | [`docs/ISO_COVERAGE.md`](docs/ISO_COVERAGE.md) | Coverage matrix per clause with update protocol |
| Failure modes | [`docs/FAILURE_MODES.md`](docs/FAILURE_MODES.md) | Per-component edge cases, ctx-fallback behavior, known limitations |
| Citation guide | [`docs/CITING.md`](docs/CITING.md) | The four canonical references to cite together when using this package |
| Upstream policy | [`OFIQ_UPSTREAM.md`](OFIQ_UPSTREAM.md) | Relationship to BSI OFIQ-Project; what would be candidates to upstream |
| Benchmarks | [`benchmarks/README.md`](benchmarks/README.md) | Throughput + best-in-class adapter framework + competitor catalog |
| Gallery | [`docs/gallery/README.md`](docs/gallery/README.md) | Per-component severity-grid visualization (regenerated locally) |
| Real-time capture demo | [`examples/README_realtime.md`](examples/README_realtime.md) | Webcam capture with live OFIQ-aligned quality feedback |
| JOSS paper | [`paper/paper.md`](paper/paper.md) | Submission-ready ~1000-word software paper |

Programmatic access:

```python
from ofiq_syngen.standards import (
    STANDARDS_REFS,           # dict[component_name, StandardRefs]
    ICAO_STRICT_COMPONENTS,   # list of component names in the icao-strict preset
    get_refs,                 # accessor: get_refs("Sharpness.scalar")
    components_by_alignment,  # filter: by 'exact' / 'partial' / 'absent'
    components_by_confidence, # filter: by 'verified' / 'derived' / 'uncertain'
)

refs = get_refs("BackgroundUniformity.scalar")
print(refs.iso_29794_5, refs.icao_9303, refs.alignment)
# Background uniformity   3.2.3 background   exact
```

The `ComponentDegradation` dataclass returned by the registry also carries
the `standard_refs` field for in-pipeline access.

## References

- **OFIQ Algorithm Book**: BSI Technical Report, Open Source Face Image Quality (OFIQ), Algorithm Book V1.2.
  Available at: [https://www.bsi.bund.de/OFIQ](https://www.bsi.bund.de/OFIQ)
- **ISO/IEC 29794-5**: Information technology -- Biometric sample quality -- Part 5: Face image data.

### Citation

If you use this package in your research, please cite:

```bibtex
@software{storey2026ofiqsyngen,
  author = {Storey, Aaron},
  title = {ofiq-syngen: ISO/IEC 29794-5 Component-Aligned Synthetic Face Image Quality Degradation Pipeline},
  year = {2026},
  url = {https://github.com/astoreyai/ofiq-syngen},
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
