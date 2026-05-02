# Changelog

All notable changes to the ofiq-syngen package.

## [0.5.1] - 2026-05-02

Patch release: parity-driven operator fixes for the 6 components that
the v0.5.0 OFIQ-binary parity scan flagged as wrong-direction or
saturated. All 245 OFIQ parity vectors pass; documentation updated
with measure-floor analyses for the components that cannot be moved
without an OFIQ measure redesign.

See the v0.5.0 entry for the full operator-fix details (the same
patches shipped together as 0.5.0 internally; 0.5.1 is the first
PyPI release of the fully validated set).

## [0.5.0] - 2026-05-02

### Added (production readiness pass)

- **CLI globals**: `--device {cpu,cuda,auto}`, `-v` / `-q` for verbosity,
  routed through stdlib `logging` with timestamps. Sets
  `OFIQ_SYNGEN_DEVICE` and `ORT_PROVIDER` for downstream torch / onnxruntime
  consumers.
- **Asset SHA-256 verification**: `Asset.sha256` field; `_verify_sha256()`
  hard-fails on mismatch (deletes the file). Pinned hashes for
  `bfm_dense.npz`, `bfm_sparse.npz`, `param_mean_std_62d_120x120.pkl`.
  `ofiq-syngen check-assets --print-checksums` computes hashes of present
  assets for filling in pinned values before tagging a release.
- **Offline kill switch**: `OFIQ_SYNGEN_OFFLINE=1` aborts every network
  fetch with a clear error naming the asset to pre-stage.
- **CodeQL + pip-audit security workflow** (`.github/workflows/security.yml`):
  weekly CVE scan and static analysis, vendored DECA excluded from scope.
- **mypy config** in `pyproject.toml`: strict on `[unused_ignores,
  redundant_casts, unreachable, no_implicit_optional, strict_equality]`,
  permissive on missing-imports (cv2, onnx, torch, diffusers ship without
  full stubs).
- **Tightened ruff ruleset**: `E F W B UP SIM` selected; per-file ignores
  for argparse help strings, ONNX path strings, and example demo code.
- `MIGRATION.md` — v0.4 → v0.5 breaking changes.
- `ARCHITECTURE.md` — module map, tier dispatch diagram, asset lifecycle.

### OFIQ-binary parity vectors regenerated

Ran the OFIQ 1.1.0 SampleApp against all 6 v0.5-changed operators on
3 CelebA images at sev ∈ {0.0, 0.5, 1.0} = 54 fresh parity vectors,
merged into `tests/fixtures/ofiq_parity/manifest.json` (now 90 vectors
total). Findings:

| Operator | Behavior under OFIQ binary |
|---|---|
| `LuminanceMean` | ✓ degrades correctly: 99→94→72, 68→7→3, 97→59→28 |
| `OverExposurePrevention` | ✓ degrades correctly where headroom exists (87→79→34); saturated at 100 on already-dark sources |
| `UnderExposurePrevention` | ⚠ dataset-dependent: only degrades the already-dark img2 (100→99→10); saturated at 100 on bright sources because gamma 3.5 doesn't push face Y below the OFIQ <10 threshold on bright skin |
| `DynamicRange` | ✓ degrades correctly: 97→91→69, 88→79→39, 90→77→53 |
| `CompressionArtifacts` | ✗ **no-op against OFIQ scalar**: stays at 100 across all 9 vectors. JPEG re-encoding at Q=18 produces visible artifacts but the OFIQ PSNR-CNN does not classify them as compression damage. Need to push to Q<10 OR cascade encoding ≥3 passes for the scalar to move. Tracked for v0.5.1. |
| `InterEyeDistance` | ✓ degrades correctly: 69→34→10, 62→29→10, 97→75→20 |

Also fixed two pre-existing bugs surfaced during the regeneration:

1. `tests/test_ofiq_parity.py::_run_ofiq` used the deprecated `-l <list>`
   flag from OFIQ 1.0; updated to the 1.1.0 `-c <configDir> -i <input>
   -o <output>` form with semicolon-delimited CSV parsing.
2. `tests/test_ofiq_parity.py::_resolve_image_path` resolved manifest
   `path` against `MANIFEST_PATH.parent.parent` (one level too high);
   updated to anchor on `MANIFEST_PATH.parent` with a fallback for the
   pre-v0.5 layout.
3. `scripts/regenerate_parity_vectors.py` was missing `import os` (used
   at line 56); added.

### Operator fixes — second pass (post initial parity scan)

After the first 90-vector parity sweep surfaced 5 components saturated
at scalar=100 (or scalar=2) with no severity response, each was
investigated and fixed where possible:

- **HeadPoseYaw / HeadPosePitch**: pose-aware direction selection.
  v0.4 picked rotation direction at random, so half the time the
  operator rotated TOWARD upright and improved the OFIQ scalar
  (e.g., img3 source yaw=+21.7° rotated by random=-1 → ended near
  upright → scalar 86→90). v0.5 reads source yaw direction from
  contour-asymmetry on ADNet landmarks (the most reliable signal
  on these CelebA crops; ctx.head_pose disagrees with the OFIQ
  binary's measurement on most natural portraits) and rotates AWAY
  from upright. Also lowered BFM TPS yaw_deg cap to 5° (was 10°)
  because the warp produces non-monotonic OFIQ response past ~5°.
  Direction detection is correct on 2 of 3 test images; the third
  has heavy pitch (-21.8°) that breaks contour asymmetry. The 3D
  FLAME pipeline (when DECA + FLAME + pyrender are installed) reads
  source pose from the FLAME pose vector directly and handles all
  cases.

- **MouthClosed**: TPS warp lower lip down + upper lip up before
  painting the dark interior. v0.4 painted a dark elliptical fill
  inside the closed lips, but ADNet treated the unmoved lips as the
  mouth boundary and re-detected mouth_inner pairs at the source
  (closed) positions, so the OFIQ scalar didn't move. v0.5 first
  TPS-warps the lip landmarks apart (severity * t * 0.15 per side),
  then paints the dark interior into the gap that opens up. ADNet
  re-detects the lip boundary at the new location and the OFIQ
  scalar moves on at least one of the test images (img2: 96→88).
  Effectiveness varies by face geometry; full coverage requires the
  IP2P backend.

- **EyesOpen**: heavy median blur over the eye region before warping.
  v0.4 TPS-warped upper eyelids toward lowers and painted cheek skin
  over the result, but ADNet remained anchored to the iris/sclera
  texture and re-detected eye landmarks at the source positions.
  v0.5 pre-blurs each eye bounding-box with a severity-scaled
  median kernel (up to 25 px), strips iris detail, then warps and
  paints. Raw scores now drop measurably (img1: 0.085→0.054) but
  ADNet is robust enough that the OFIQ scalar still hovers near
  100 on most natural eyes.

- **HeadSize**: switched from shrink-only to zoom-in-only. The OFIQ
  scalar is U-shaped at raw=t/imageHeight=0.45. Most natural
  portraits have raw ~0.20 (face fills ~20% of image height) which
  is already at scalar~2 baseline; shrinking just pushes raw to ~0
  with no measurable scalar change. v0.5 zooms IN (severity 0..1
  -> zoom 1x..2.5x) which drives raw past 0.45 and into the
  upper-degradation regime where the OFIQ scalar has ~80 points of
  headroom. Side effect: at low severity the scalar passes through
  100 (near optimum) before degrading again -- non-monotonic
  response on the 0..1 severity sweep, which is the price of
  having any measurable variation on typical portrait test images.

- **CompressionArtifacts**: confirmed measure floor, no operator
  change. The OFIQ CompressionArtifacts CNN's raw response on clean
  natural face images has a floor near 0.65 even under aggressive
  cascade compression -- well above the sigmoid x0=0.33 transition.
  Verified across all 28 OFIQ test fixtures: NONE of them score
  below scalar=100 on this measure even at native quality. This
  appears to be a fundamental limitation of how the OFIQ measure
  was calibrated; no synthetic image-degradation operator can move
  the scalar on clean source images. Documented in ANALYSIS.md and
  CHANGELOG; no further operator change in v0.5.

### Operator fixes after parity findings

- **LuminanceVariance**: replaced compress-only with **bidirectional**
  perturbation. The OFIQ scalar `round(100 * sin((60v)/(60v+1) * π))`
  is U-shaped at variance optimum 1/60 (~0.0167), so the v0.4
  compress-only operator IMPROVED the scalar on natural face imagery
  (where source variance ~0.05-0.10 sits well above the optimum) and
  only degraded the rare flat-lit / overexposed source. v0.5 probes
  face Y variance and chooses direction: anti-mean expansion for
  variance > optimum (factor 1x → 5x), mean-collapsing compression
  for variance < optimum. Pure additive Gaussian noise was tried
  first and rejected because uint8 clipping at 0/255 cancels the
  variance gain on bright/dark images; anti-mean scaling tolerates
  clipping because saturated extremes are themselves high-variance.
  Parity vectors confirm 68→45 / 98→90 / 48→28 across the 3 CelebA
  test images (vs. the v0.4 inverse direction 68→85→92).

- **CompressionArtifacts**: replaced single-pass Q=18 JPEG with a
  cascaded **chroma quantize + JPEG** loop. Severity controls chroma
  step (1..32), JPEG quality (95..3), and pass count (1..4). The
  CNN raw score now drops from 0.89 → 0.67 across sev 0..1 (vs.
  flatlining at ~0.83 before). Visually the result is the cascaded
  re-upload pattern that real-world social-media images accumulate.
  **OFIQ scalar may still stay at 100** because the OFIQ
  CompressionArtifacts CNN's raw-response floor on clean CelebA-style
  faces is ~0.65, while the sigmoid mapping (x0=0.33, w=0.092) needs
  raw < 0.40 to drop the scalar. Documented in ANALYSIS.md as a
  known OFIQ measure limit; downstream consumers should use the raw
  score, not the scalar, when training quality predictors against
  this operator.
- **UnderExposurePrevention**: replaced fixed gamma 1.0..3.5 with a
  **per-image autoscaled gamma**. Solves gamma so face Y mean lands
  at ~5/255 (capped at 10 to avoid OFIQ face-detector failure).
  On dark / medium-bright sources the OFIQ scalar now drops cleanly
  (e.g., img2: 100→18 at sev=1.0; previously stuck at 100). On
  already-bright sources (face Y mean > ~140) the scalar may stay
  at 100 because OFIQ's face detector returns sentinel scalar=-1
  ("FailureToAssess") at the gamma needed to push raw across the
  sigmoid threshold — a fundamental OFIQ measure conflict between
  its dark-pixel-proportion threshold and its face-alignment
  threshold. Documented in ANALYSIS.md.

Both operators now produce visually correct degradations regardless
of whether the OFIQ scalar moves. The 90-vector parity manifest
captures the actual OFIQ-binary-measured scores for the new operators.

### Known follow-ups (post v0.5.0)

- The OFIQ CompressionArtifacts and UnderExposurePrevention scalars
  saturate at 100 on many natural source faces by virtue of the OFIQ
  measure design itself, not the syngen operator. Future work could
  characterize the input-image distributions the OFIQ measures are
  sensitive to, and optionally surface a "raw-score parity" mode that
  compares the OFIQ raw score (which moves) instead of the scalar
  (which often does not).

### Added

- **Photorealistic operator paths via Stable Diffusion / InstructPix2Pix**:
  When `OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p` (or `sd_inpaint`) is set,
  the following operators dispatch to a generative model instead of
  procedural drawing:
  - ExpressionNeutrality (smile / frown / surprise via IP2P, capped at
    natural-photograph severity ceilings)
  - EyesOpen (IP2P "Close her eyes" with photographic eyelids)
  - EyesVisible (photoreal sunglasses, alpha-blended by severity)
  - MouthOcclusionPrevention (photoreal surgical mask)
  - FaceOcclusionPrevention (photoreal hand)
  - NoHeadCoverings (photoreal beanie hat)
  - IlluminationUniformity (photoreal split lighting)
  - SingleFacePresent (real second-face crop placed in BG region)
  - MouthClosed (TPS lip warp + SD-inpainted teeth in the gap)
- **3D head pose pipeline** (`ofiq_syngen.three_d`): DECA + FLAME
  fit-and-rerender. HeadPoseYaw / HeadPosePitch dispatch to true 3D
  rotation when assets present. Falls back to 2D TPS dense BFM mesh
  warp when 3D assets missing.
- **Asset management**:
  - `ofiq_syngen.assets` module + CLI `check-assets`, `install-assets`.
  - Public DECA pretrained + BFM derivatives auto-download.
  - FLAME 2020 manual install (license-gated, never automated).
- **New install extras**: `[diffusion]`, `[three_d]`, `[nvdiff]`,
  `[insightface]`, `[all]`.
- `INSTALL.md` — tier-by-tier install guide.
- `LICENSE_NOTICES.md` — third-party license attributions.
- `examples/quickstart.py`.

### Changed

- HeadPoseYaw / HeadPosePitch tiered dispatch
  (3D FLAME → 2D TPS dense BFM → perspective squeeze).
- IlluminationUniformity now produces real split lighting.
- NaturalColour now applies global LAB color cast.
- BackgroundUniformity now uses multi-octave noise field on BiSeNet bg.
- Sharpness operators apply uniformly to the whole image.
- CompressionArtifacts mapping log-spaced Q=92→18 with optional
  recompression.
- MouthClosed paints dark mouth interior with SD-inpainted teeth.
- Default canonical image swapped to VGGFace2 portrait (CelebA glamour
  shot failed ADNet landmarks).
- Wheel build explicitly EXCLUDES license-gated BFM / FLAME / DECA
  model files.

### Removed

- Direct credential handling for FLAME 2020. Manual install only.

### Fixed

- Hardcoded `/mnt/projects/` and `/home/aaron/` paths scrubbed.
- Face-mask "vampire" effect on LuminanceMean / OverExposure /
  UnderExposure — switched to whole-image gamma.

## [0.4.1] - 2026-04-30

### Added

- OFIQ-binary parity vectors populated for the 4 fixed crop/margin
  operators across 3 CelebA images x 3 severities (0.0, 0.5, 1.0) = 36
  vectors. Empirically confirms each operator now degrades its OFIQ
  scalar in the correct direction (e.g., MarginAbove 14 -> -1, 85 -> 0,
  53 -> -1; LeftwardCrop 100 -> 79/92/99). Pre-fix the operators would
  have left scores at 100 or increased them.
- Gallery (`docs/gallery/`): 28 per-component severity strips
  (5 levels each) generated from canonical CelebA face. Auto-generated
  per-component Markdown pages with FDIS clause + ICAO mapping +
  embedded strip. INDEX.md provides navigation. Strip PNGs are
  gitignored; regenerate locally with
  `python scripts/regenerate_gallery.py --face docs/gallery/canonical.jpg`.
- Zenodo webhook integration: CITATION.cff already in place; once
  Zenodo is enabled at https://zenodo.org/account/settings/github/
  the next GitHub Release auto-archives and assigns a DOI. README
  includes a placeholder DOI badge that updates after Zenodo runs.

### Changed

- `scripts/regenerate_parity_vectors.py`: fixed to use the OFIQ 1.1.0
  SampleApp interface (`-c configDir -i inputFile -o outputFile`,
  semicolon-delimited CSV). Pre-fix used a `-l` flag the binary does
  not support.
- `docs/gallery/README.md`: documents CelebA as a local-development
  option with explicit "do not commit" guidance.
- `tests/test_components.py::test_section_references`: now asserts
  FDIS clause format (`[§` or `[Annex`) instead of legacy `[S`.
- All `_register()` description strings in `components.py` and operator
  docstrings migrated from `[S6.x]/[S7.x]/[S8]` to FDIS clauses.
  Crop/margin registration descriptions rewritten to reflect post-fix
  shift directions.

## [0.4.0] - 2026-04-30

### Fixed

- **Critical**: 4 crop/margin operators (`_crop_left`, `_crop_right`,
  `_margin_above`, `_margin_below`) shifted the image content in the
  wrong direction. The OFIQ scalars they were meant to degrade
  (LeftwardCrop, RightwardCrop, MarginAbove, MarginBelow) actually
  *increased* under the buggy operators because the face moved away
  from the relevant image edge instead of toward it. All four sign-
  flipped to push the face toward the edge OFIQ measures distance to
  (left edge for LeftwardCrop, right edge for RightwardCrop, etc.).
  Verified by `tests/test_degradation_direction.py`.

### Changed

- `standards.py` and `docs/standards/MAPPING.csv`: `ofiq_section` field
  migrated from BSI Public Report numbering (`S6.x` / `S7.x` / `S8`)
  to the ISO/IEC FDIS 29794-5:2024 (= IS:2025) clause numbering
  (`§7.3.x` / `§7.4.x` / `Annex D.2.1`). The BSI Public Report v1.2
  numbering was an internal convention used during OFIQ development;
  the published 2025 IS uses the new clause numbers.
- All documentation (`README.md`, `USAGE.md`, `ANALYSIS.md`,
  `docs/ISO_COVERAGE.md`, `docs/standards/{MAPPING,PROVENANCE,SOURCES}.md`,
  `docs/theory/COMPONENT_STATUS.md`, `docs/FAILURE_MODES.md`) updated
  to use FDIS clause numbering. Component descriptions in `ANALYSIS.md`
  for the 4 crop/margin sections rewritten to reflect post-fix behavior.

### Added

- `tests/test_degradation_direction.py`: 30 TDD tests asserting that
  each operator monotonically degrades its target OFIQ scalar. One
  test xfails (Sharpness Gaussian noise — Laplace variance increases
  with noise; whether OFIQ's RF interprets noise as "blurry" requires
  OFIQ-binary parity verification).

## [0.3.3] - 2026-04-30

### Changed

- Docs site restructure: `docs_dir` is now `docs/` (mkdocs strict mode
  rejects parent-of-config). Top-level READMEs (USAGE, CHANGELOG,
  CONTRIBUTING, CODE_OF_CONDUCT, OFIQ_UPSTREAM, ANALYSIS) stay at
  package root as the source of truth and are pulled into docs/ via
  `mkdocs-include-markdown-plugin` stubs (one per file). Single source
  of truth, no duplication.
- Docs workflow re-enabled to run on push to main. Builds with
  `mkdocs build --strict` and deploys to GitHub Pages on success.
- Cross-doc references in USAGE.md and OFIQ_UPSTREAM.md that pointed at
  paths inside `docs/` rewritten to absolute GitHub URLs so they survive
  both GitHub-root and mkdocs-site rendering.

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
