# ISO/IEC 29794-5 Coverage Matrix

Per-component coverage of the components defined in ISO/IEC 29794-5:2024.

## Status

| Status | Count | Definition |
|---|---|---|
| **Implemented + OFIQ-aligned** | 27 | Degrader exists, region targeting matches OFIQ measurement, multi-standard mapping verified or uncertain (uncertain flagged) |
| **Implemented + forward-looking** | 1 | Degrader exists for a 29794-5 component not yet measured by OFIQ 1.1.0 (`RadialDistortion`) |
| **Not implemented** | 0 | Components in 29794-5 with no syngen counterpart |

## Per-component status

### Capture-related (FDIS §7.3)

| Clause | Component | Status | Notes |
|---|---|---|---|
| §7.3.2 | BackgroundUniformity | implemented | exact alignment to OFIQ BiSeNet-segmented background |
| §7.3.3 | IlluminationUniformity | implemented | exact alignment to OFIQ L/R ROI zones |
| §7.3.4.2 | LuminanceMean | implemented | face-mask targeting |
| §7.3.4.3 | LuminanceVariance | implemented | partial alignment (OFIQ uses face mask; degrader same region but different metric) |
| §7.3.5 | UnderExposurePrevention | implemented | exact alignment to face mask intersect occlusion mask |
| §7.3.6 | OverExposurePrevention | implemented | exact alignment to face mask |
| §7.3.7 | DynamicRange | implemented | partial alignment |
| §7.3.8 | Sharpness | implemented | three degraders: blur, motion blur, additive noise |
| §7.3.9 | CompressionArtifacts | implemented | JPEG re-compression at variable quality |
| §7.3.10 | NaturalColour | implemented | CIELAB shift in landmark ROI zones (or whole image without context) |
| Annex D.2.1 | RadialDistortion | implemented (forward-looking) | no OFIQ measurement counterpart in v1.1.0 |
| §7.4.2 | SingleFacePresent | implemented | OFIQ section number unverified; ICAO clause confirmed |

### Subject-related (FDIS §7.4)

| Clause | Component | Status | Notes |
|---|---|---|---|
| §7.4.3 | EyesOpen | implemented | RBF landmark warp; requires FaceContext |
| §7.4.4 | MouthClosed | implemented | RBF landmark warp; requires FaceContext |
| §7.4.5 | EyesVisible | implemented | EVZ-rectangle occlusion |
| §7.4.6 | MouthOcclusionPrevention | implemented | mouth-polygon occlusion |
| §7.4.7 | FaceOcclusionPrevention | implemented | face-mask-constrained rectangular occlusion |
| §7.4.8 | InterEyeDistance | implemented | pad-and-shrink to reduce IED |
| §7.4.9 | HeadSize | implemented | pad-and-shrink to reduce face proportion |
| §7.4.12 | ExpressionNeutrality | implemented | OFIQ section unverified; landmark warp |
| §7.4.13 | NoHeadCoverings | implemented | partial alignment (no religious/medical exemption modeling) |

### Geometric / pose (Section 8)

| Clause | Component | Status | Notes |
|---|---|---|---|
| §7.4.11.2 | HeadPoseYaw | implemented | perspective warp (not true 3D); MODERATE algorithm fidelity |
| §7.4.11.3 | HeadPosePitch | implemented | perspective warp (not true 3D); MODERATE algorithm fidelity |
| §7.4.11.4 | HeadPoseRoll | implemented | exact 2D rotation |
| §7.4.10.1 | LeftwardCropOfTheFaceImage | implemented | exact directional shift |
| §7.4.10.2 | RightwardCropOfTheFaceImage | implemented | exact directional shift |
| §7.4.10.3 | MarginAboveOfTheFaceImage | implemented | exact directional shift |
| §7.4.10.4 | MarginBelowOfTheFaceImage | implemented | exact directional shift |

## Where the package extends beyond OFIQ 1.1.0

`RadialDistortion.scalar` (Annex D.2.1) implements perturbation for a component
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
