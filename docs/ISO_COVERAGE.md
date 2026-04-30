# ISO/IEC 29794-5 Coverage Matrix

Per-component coverage of the components defined in ISO/IEC 29794-5:2024.

## Status

| Status | Count | Definition |
|---|---|---|
| **Implemented + OFIQ-aligned** | 27 | Degrader exists, region targeting matches OFIQ measurement, multi-standard mapping verified or uncertain (uncertain flagged) |
| **Implemented + forward-looking** | 1 | Degrader exists for a 29794-5 component not yet measured by OFIQ 1.1.0 (`RadialDistortion`) |
| **Not implemented** | 0 | Components in 29794-5 with no syngen counterpart |

## Per-component status

### Capture-related (Section 6)

| Clause | Component | Status | Notes |
|---|---|---|---|
| S6.1 | BackgroundUniformity | implemented | exact alignment to OFIQ BiSeNet-segmented background |
| S6.2 | IlluminationUniformity | implemented | exact alignment to OFIQ L/R ROI zones |
| S6.3 | LuminanceMean | implemented | face-mask targeting |
| S6.3 | LuminanceVariance | implemented | partial alignment (OFIQ uses face mask; degrader same region but different metric) |
| S6.4 | UnderExposurePrevention | implemented | exact alignment to face mask intersect occlusion mask |
| S6.4 | OverExposurePrevention | implemented | exact alignment to face mask |
| S6.5 | DynamicRange | implemented | partial alignment |
| S6.6 | Sharpness | implemented | three degraders: blur, motion blur, additive noise |
| S6.7 | CompressionArtifacts | implemented | JPEG re-compression at variable quality |
| S6.8 | NaturalColour | implemented | CIELAB shift in landmark ROI zones (or whole image without context) |
| S6.9 | RadialDistortion | implemented (forward-looking) | no OFIQ measurement counterpart in v1.1.0 |
| S6 ? | SingleFacePresent | implemented | OFIQ section number unverified; ICAO clause confirmed |

### Subject-related (Section 7)

| Clause | Component | Status | Notes |
|---|---|---|---|
| S7.2 | EyesOpen | implemented | RBF landmark warp; requires FaceContext |
| S7.3 | MouthClosed | implemented | RBF landmark warp; requires FaceContext |
| S7.4 | EyesVisible | implemented | EVZ-rectangle occlusion |
| S7.5 | MouthOcclusionPrevention | implemented | mouth-polygon occlusion |
| S7.6 | FaceOcclusionPrevention | implemented | face-mask-constrained rectangular occlusion |
| S7.7 | InterEyeDistance | implemented | pad-and-shrink to reduce IED |
| S7.8 | HeadSize | implemented | pad-and-shrink to reduce face proportion |
| S7.9 ? | ExpressionNeutrality | implemented | OFIQ section unverified; landmark warp |
| S7.10 ? | NoHeadCoverings | implemented | partial alignment (no religious/medical exemption modeling) |

### Geometric / pose (Section 8)

| Clause | Component | Status | Notes |
|---|---|---|---|
| S8 | HeadPoseYaw | implemented | perspective warp (not true 3D); MODERATE algorithm fidelity |
| S8 | HeadPosePitch | implemented | perspective warp (not true 3D); MODERATE algorithm fidelity |
| S8 | HeadPoseRoll | implemented | exact 2D rotation |
| S8 | LeftwardCropOfTheFaceImage | implemented | exact directional shift |
| S8 | RightwardCropOfTheFaceImage | implemented | exact directional shift |
| S8 | MarginAboveOfTheFaceImage | implemented | exact directional shift |
| S8 | MarginBelowOfTheFaceImage | implemented | exact directional shift |

## Where the package extends beyond OFIQ 1.1.0

`RadialDistortion.scalar` (S6.9) implements perturbation for a component
defined in ISO/IEC 29794-5:2024 but not yet measured in OFIQ 1.1.0. When
OFIQ adds the corresponding measurement, the degrader is ready.

## Where the package falls short of OFIQ 1.1.0 components

None. Every component implemented in OFIQ 1.1.0 has a corresponding
degrader in syngen 0.3.0.

## Verification path

The coverage table above is human-curated against the OFIQ Algorithm
Book and `OFIQ-Project/OFIQlib/modules/measures/src/`. To verify
mechanically:

```bash
ls /path/to/OFIQ-Project/OFIQlib/modules/measures/src/*.cpp | wc -l
# Expect: 25 .cpp files (some cover multiple measures, hence 27 measured components)

ofiq-syngen list-components | wc -l
# Expect: 30 lines (28 components + header + footer)

ofiq-syngen show-standards --preset iso-29794-5 | grep -c '^[A-Z]'
# Expect: 28
```

Discrepancies between the OFIQ source measure count and the syngen
component count must be investigated and either fixed (add a missing
degrader) or documented (mark a component as unimplemented).

## Update protocol

When OFIQ ships a new release with new measured components:

1. Add the component to `STANDARDS_REFS` in `src/ofiq_syngen/standards.py`.
2. Add the row to `docs/standards/MAPPING.csv` (CSV / Python sync test
   asserts they match).
3. Implement and register the degrader.
4. Update this document and `docs/theory/COMPONENT_STATUS.md`.
5. Add an entry to `CHANGELOG.md` under "Standards alignment."
