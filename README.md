# ofiq-syngen

[![PyPI version](https://img.shields.io/pypi/v/ofiq-syngen.svg)](https://pypi.org/project/ofiq-syngen/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.PENDING.svg)](https://doi.org/10.5281/zenodo.PENDING)
[![CI](https://github.com/astoreyai/ofiq-syngen/actions/workflows/ci.yml/badge.svg)](https://github.com/astoreyai/ofiq-syngen/actions/workflows/ci.yml)

**ISO/IEC 29794-5 component-aligned synthetic face image quality degradation pipeline.**

`ofiq-syngen` generates controlled quality degradations mapped to specific ISO/IEC 29794-5 (OFIQ) quality components. Each degradation function targets a known OFIQ scalar component -- Sharpness, CompressionArtifacts, HeadPoseYaw, etc. -- producing ground-truth (image, degradation, delta-OFIQ) triplets for training quality-aware biometric systems. **All 27 OFIQ-measurable components are covered** with deterministic, seed-reproducible degradation operators; OFIQ-binary parity vectors verify each operator degrades its target scalar in the correct direction.

**Visual gallery**: see [`docs/gallery/INDEX.md`](docs/gallery/INDEX.md) for severity strips of every component on a canonical face image.

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
# {'target_component': 'Sharpness.scalar', 'degradation_type': 'Gaussian blur [§7.3.8]',
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

All 27 OFIQ scalar components are covered. Section references follow ISO/IEC FDIS 29794-5:2024 (= IS:2025).

| OFIQ Component | FDIS § | Degradation | Severity Range |
|---|---|---|---|
| `BackgroundUniformity.scalar` | §7.3.2 | Background noise | intensity: 0 -> 200 |
| `IlluminationUniformity.scalar` | §7.3.3 | Gradient lighting | gradient: 0% -> 80% drop |
| `LuminanceMean.scalar` | §7.3.4.2 | Brightness reduction | factor: 1.0 -> 0.15 |
| `LuminanceVariance.scalar` | §7.3.4.3 | Variance compression | factor: 1.0 -> 0.1 |
| `UnderExposurePrevention.scalar` | §7.3.5 | Under-exposure | factor: 1.0 -> 0.15 |
| `OverExposurePrevention.scalar` | §7.3.6 | Over-exposure | factor: 1.0 -> 3.5 |
| `DynamicRange.scalar` | §7.3.7 | Dynamic range compression | range: 100% -> 10% |
| `Sharpness.scalar` | §7.3.8 | Gaussian blur | sigma: 0.5 -> 10.5 |
| `Sharpness.scalar` | §7.3.8 | Motion blur | kernel: 3 -> 31px |
| `Sharpness.scalar` | §7.3.8 | Additive Gaussian noise | sigma: 0 -> 80 |
| `CompressionArtifacts.scalar` | §7.3.9 | JPEG compression | Q: 100 -> 5 |
| `NaturalColour.scalar` | §7.3.10 | CIELAB a\*/b\* shift in ROI zones | shift: 0 -> 60 |
| `RadialDistortion.scalar` | Annex D.2.1 | Barrel distortion (no QAA in IS:2025) | k: 0 -> 0.5 |
| `EyesOpen.scalar` | §7.4.3 | Landmark-warped eye closure | closure: 0% -> 90% |
| `MouthClosed.scalar` | §7.4.4 | Landmark-warped mouth opening | opening: 0 -> 0.25t |
| `EyesVisible.scalar` | §7.4.5 | EVZ-targeted eye occlusion | coverage: 0% -> 80% |
| `MouthOcclusionPrevention.scalar` | §7.4.6 | Polygon-targeted mouth occlusion | coverage: 0% -> 100% |
| `FaceOcclusionPrevention.scalar` | §7.4.7 | Face-masked rectangular occlusion | area: 0% -> 60% |
| `InterEyeDistance.scalar` | §7.4.8 | Pad-and-shrink (face smaller in frame) | scale: 1.0 -> 0.3 |
| `HeadSize.scalar` | §7.4.9 | Pad-and-shrink (same mechanism as IED) | scale: 1.0 -> 0.3 |
| `HeadPoseYaw.scalar` | §7.4.11.2 | Perspective yaw rotation | squeeze: 0% -> 50% |
| `HeadPosePitch.scalar` | §7.4.11.3 | Perspective pitch tilt | squeeze: 0% -> 40% |
| `HeadPoseRoll.scalar` | §7.4.11.4 | In-plane rotation | angle: 0 -> +/-30 deg |
| `LeftwardCropOfTheFaceImage.scalar` | §7.4.10.1 | Leftward image shift (v0.4.0 fix) | shift: 0% -> 40% |
| `RightwardCropOfTheFaceImage.scalar` | §7.4.10.2 | Rightward image shift (v0.4.0 fix) | shift: 0% -> 40% |
| `MarginAboveOfTheFaceImage.scalar` | §7.4.10.3 | Upward image shift (v0.4.0 fix) | shift: 0% -> 40% |
| `MarginBelowOfTheFaceImage.scalar` | §7.4.10.4 | Downward image shift (v0.4.0 fix) | shift: 0% -> 40% |
| `SingleFacePresent.scalar` | §7.4.2 | Face insertion via Poisson blending | area ratio: 0 -> 0.4 |
| `ExpressionNeutrality.scalar` | §7.4.12 | Landmark-warped expression | displacement: 0 -> 0.15t |
| `NoHeadCoverings.scalar` | §7.4.13 | Synthetic hat overlay | coverage: 0% -> 100% |

### Coverage notes

All 27 measurable OFIQ components have direct degraders. Three subject-related
components — `ExpressionNeutrality`, `NoHeadCoverings`, `SingleFacePresent` —
use generative operators (landmark RBF warps, synthetic overlays, Poisson-blended
face inserts respectively) that depend on OFIQ ONNX models classifying the
synthetic content correctly. Effectiveness against the OFIQ classifier is verified
via the parity manifest (`tests/fixtures/ofiq_parity/`).

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
