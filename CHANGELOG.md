# Changelog

All notable changes to the ofiq-syngen package.

## [0.3.2] - 2026-04-29

### Added

- `scripts/run_crosstalk.py`: builds the 28x28 cross-talk matrix
  (perturbation x measured-component delta) when an OFIQ binary or
  GPUOFIQScorer is available. Runs against a user-supplied face image
  set; outputs CSV.
- `scripts/run_monotonicity.py`: per-component severity sweep (10
  steps), Spearman rank correlation between syngen severity and OFIQ
  score per component, monotonicity flag. Closes the calibration gap
  identified in the FIQA systematic review companion paper.
- `scripts/audit_docstrings.py`: CI-friendly walk over every public
  symbol in src/ofiq_syngen/; reports missing docstrings.

### Changed

- Docstring audit pass: 0 public symbols without docstrings (was 4).
  `GPUOFIQScorer.__init__`, `OFIQModels.__init__`, and
  `OFIQModels.model_dir` documented.
- README "Standards Mapping" section expanded to link the per-component
  status page, ISO coverage matrix, failure modes catalog, benchmarks,
  gallery scaffold, real-time capture demo, and JOSS paper.

## [0.3.1] - 2026-04-29

### Added

- **`paper/`**: JOSS submission scaffold with `paper.md` (statement of
  need + functionality + validation, ~1000 words) and `paper.bib`
  (OFIQ + ISO/IEC 29794-5 + ISO/IEC 19794-5 + ICAO 9303 + competitor
  references).
- **`docs/FAILURE_MODES.md`**: per-component documentation of input
  conditions and edge cases; the source of truth for "where it
  breaks" claims.
- **`docs/ISO_COVERAGE.md`**: ISO/IEC 29794-5 coverage matrix with
  status per clause and an update protocol for new OFIQ releases.
- **`docs/gallery/`**: scaffold for the per-component severity-grid
  visual gallery, plus `scripts/regenerate_gallery.py` to populate it
  from a user-supplied canonical face image.
- **`benchmarks/`**: `bench_throughput.py` (per-component wall-clock
  characterization), `run_grid.py` (best-in-class adapter framework),
  `adapters/` (template + opencv_baseline sanity adapter),
  `COMPETITORS.md` (per-component catalog of alternatives), `README.md`.
- Initial throughput baseline at `benchmarks/results/throughput.csv`
  (synthetic 256x256, no FaceContext).

### Cross-link

- Paper P32 (`world/papers/paper32_experimental/`) gains a
  `\cite{ofiq_syngen}` reference at the synthetic-degradation pipeline
  description; bibliography entry added.

## [0.3.0] - 2026-04-29

### Added

- **Multi-standard mapping** for every component (28 of 28).
  - `src/ofiq_syngen/standards.py` exposes `STANDARDS_REFS` (dict keyed by
    component name) plus `ICAO_STRICT_COMPONENTS`,
    `ISO_19794_5_COMPONENTS`, `ISO_29794_5_COMPONENTS` subset helpers and
    `get_refs`, `components_by_alignment`, `components_by_confidence`
    accessors.
  - `ComponentDegradation.standard_refs` field populated automatically at
    registration time.
  - Reference docs in `docs/standards/`: `MAPPING.csv` (machine-readable
    source of truth), `MAPPING.md` (rendered with section grouping),
    `SOURCES.md` (standards editions and version pinning), `PROVENANCE.md`
    (OFIQ C++ source-line citations per component plus helper-utility
    ports).
- **CLI standards presets**: `--preset {icao-strict,iso-19794-5,iso-29794-5}`
  on `list-components` and `generate-dataset`.
- **`ofiq-syngen show-standards`** subcommand prints the full triple
  cross-reference table; supports `--preset` filter.
- **`examples/realtime_capture.py`** + `examples/README_realtime.md`:
  runnable webcam-capture demo with live OFIQ-aligned quality feedback.
  Background-thread scoring through `GPUOFIQScorer`, 28-component to
  5-panel aggregation (LIGHTING, FOCUS, POSE, EXPRESSION, OCCLUSION),
  auto-trigger when all panels stay green for N consecutive scored
  frames. Mock fallback when OFIQ models are unavailable.
- **Per-component thorough test suite** in `tests/test_components_per_component.py`.
  Parametrized over all 28 components across 7 dimensions: smoke,
  determinism, seed sensitivity, severity-changes-output, severity
  monotonicity, output finiteness, size invariance (small / medium /
  large fixtures). Adds 216 passing test rows; intentional skips for
  context-noop components and seed-insensitive degraders are documented
  inline.
- **`tests/test_standards_mapping.py`**: 10 tests covering full coverage,
  field well-formedness, partition completeness, and CSV-vs-Python sync.
- **`tests/test_cli_presets.py`**: 11 tests covering preset registry,
  preset filtering on `show-standards` and `list-components`, and
  mutual-exclusivity validation on `generate-dataset`.
- `CITATION.cff`, `docs/CITING.md`, and `OFIQ_UPSTREAM.md`.

### Changed

- `cmd_generate_dataset` validates argument mutex (`--components` vs
  `--preset`) before filesystem checks.
- Test count: 63 (v0.2.0) -> 300 passing in v0.3.0.

### Documentation

- README adds a "Standards Mapping" section with programmatic-access
  examples and a CLI section for the new preset flags.
- USAGE adds a "Standards presets" walkthrough and "OFIQ source
  provenance" reference.

### Standards alignment

- OFIQ pin: `1.1.0` (per `OFIQ-Project/Version.txt`). Discrepancy with
  v0.2.0 CHANGELOG ("BSI V1.2") flagged in `docs/standards/SOURCES.md`
  for resolution before v0.4.0.

## [0.2.0] - 2026-04-08

### Breaking Changes
- All degradation functions now take 4 args: `fn(img, severity, seed, ctx)` (was 3)
- `ComponentDegradation` has new field `requires_context: bool`
- `DegradationPipeline.__init__` accepts optional `models: OFIQModels`
- `degrade_single` accepts optional `ctx: FaceContext` parameter
- `src/degradation/` shim removed — all imports must use `ofiq_syngen.*`

### Added
- **FaceContext**: runs OFIQ's ADNet, BiSeNet, occlusion seg, and HeadPose3DDFAV2 once per image
- **OFIQModels**: lazy-loading singleton for OFIQ ONNX models
- **landmark_utils.py**: exact Python port of OFIQ's adnet_FaceMap.h, FaceMeasures, image_utils (98-pt index maps, tmetric, IED, ROI zones, EVZ rects, face mask, custom CIELAB, rec709 luminance)
- 28/28 OFIQ components implemented (was 25/27):
  - SingleFacePresent: Poisson-blended face insertion
  - ExpressionNeutrality: landmark RBF warp (smile/surprise/frown)
  - NoHeadCoverings: fabric-textured hat overlay
- RadialDistortion (S6.9) — forward-looking ISO 29794-5 component
- 63 unit tests (was 29)

### Changed — OFIQ-Aligned Degradation Functions
- **BackgroundUniformity**: border strip noise → BiSeNet class-0 segmented background with 4x4 erosion
- **IlluminationUniformity**: global gradient → ROI zone darkening (landmark-derived L/R zones)
- **LuminanceMean/Variance/UnderExposure/OverExposure**: whole image → face mask targeting
- **UnderExposurePrevention**: face mask only → face mask ∩ occlusion mask (matches OFIQ)
- **NaturalColour**: RGB channel boost → CIELAB a*/b* shift in landmark-derived ROI zones
- **EyesOpen**: dark band → RBF landmark warp (upper eyelid pairs 61/67, 62/66, 63/65)
- **MouthClosed**: light band → RBF landmark warp (inner lip pairs 89/95, 90/94, 91/93)
- **EyesVisible**: dark band → EVZ-targeted occlusion (IED/20 expansion)
- **MouthOcclusionPrevention**: horizontal band → mouth polygon fill (landmarks 76-87)
- **FaceOcclusionPrevention**: random rect → face-mask-constrained rect
- **InterEyeDistance**: downscale+upscale (zero effect) → pad-and-shrink
- **HeadSize**: downscale+upscale → pad-and-shrink
- **CropLeft/Right/MarginAbove/Below**: 1 shared random-direction → 4 deterministic single-direction

### Dependencies
- Added: `onnxruntime>=1.14`, `scipy>=1.7`
- Optional: `onnxruntime-gpu`, `insightface`, `diffusers`, `torch`

## [0.1.0] - 2026-04-06

### Added
- Initial release
- 25/27 OFIQ quality component degradation functions
- CLI: `ofiq-syngen degrade`, `sweep`, `list-components`, `generate-dataset`
- Influence matrix builder
- 29 unit tests
- MIT license
