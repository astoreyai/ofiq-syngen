# Multi-Standard Component Mapping

Cross-reference for every `ofiq-syngen` component against the three standards
that define face-image-quality requirements: ISO/IEC 29794-5 (measurement),
ISO/IEC 19794-5 (data interchange), and ICAO Doc 9303 Part 9 (travel
documents).

**Sources and edition pins**: see [`SOURCES.md`](SOURCES.md).
**Machine-readable form**: [`MAPPING.csv`](MAPPING.csv).

## Confidence flags

- (no flag) = `verified`: clause text in hand or referenced directly in OFIQ Algorithm Book.
- `?` = `uncertain`: best inference; please verify against the standard before citing.

## Capture-related components (S6.x)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `BackgroundUniformity.scalar` | S6.1 | Background uniformity | §8.9 lighting/scene | §3.2.3 background | exact |
| `IlluminationUniformity.scalar` | S6.2 | Illumination uniformity | §8.9 lighting | §3.2.3 lighting | exact |
| `LuminanceMean.scalar` | S6.3 | Luminance mean | §8.9 exposure | §3.2.3 exposure | exact |
| `LuminanceVariance.scalar` | S6.3 | Luminance variance | §8.9 exposure | §3.2.3 contrast | partial |
| `UnderExposurePrevention.scalar` | S6.4 | Under-exposure prevention | §8.9 exposure | §3.2.3 exposure | exact |
| `OverExposurePrevention.scalar` | S6.4 | Over-exposure prevention | §8.9 exposure | §3.2.3 exposure | exact |
| `DynamicRange.scalar` | S6.5 | Dynamic range | §8.9 lighting | §3.2.3 contrast | partial |
| `Sharpness.scalar` | S6.6 | Sharpness (focus) | §8.10 focus | §3.2.3 focus | exact |
| `CompressionArtifacts.scalar` | S6.7 | Compression artifacts | §9.2 JPEG/JPEG2000 | §3.2.5 compression | exact |
| `NaturalColour.scalar` | S6.8 | Natural colour | §9.1 colour space | §3.2.4 true colour | exact |
| `RadialDistortion.scalar` | S6.9 | Radial/lens distortion | §8.10 lens | §3.2.3 lens distortion | exact |
| `SingleFacePresent.scalar` | S6 ? | Single face present | §8.1 subject | §3.2.3 one subject | exact |

## Subject-related components (S7.x)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `EyesOpen.scalar` | S7.2 | Eyes open | §8.5 eye state | §3.2.3 eyes open | exact |
| `MouthClosed.scalar` | S7.3 | Mouth closed | §8.6 mouth state | §3.2.3 mouth closed | exact |
| `EyesVisible.scalar` | S7.4 | Eyes visible (no occlusion) | §8.7 eye occlusion | §3.2.3 no eye covering | exact |
| `MouthOcclusionPrevention.scalar` | S7.5 | Mouth visible | §8.7 mouth occlusion | §3.2.3 no mouth covering | exact |
| `FaceOcclusionPrevention.scalar` | S7.6 | Face visible | §8.7 face occlusion | §3.2.3 face visible | exact |
| `InterEyeDistance.scalar` | S7.7 | Inter-eye distance | §9.3 resolution | §3.2.2 eye distance min | exact |
| `HeadSize.scalar` | S7.8 | Head size in frame | §7.4 token formats | §3.2.2 face size | exact |
| `ExpressionNeutrality.scalar` | S7.9 ? | Neutral expression | §8.3 expression | §3.2.3 neutral expression | exact |
| `NoHeadCoverings.scalar` | S7.10 ? | No head coverings | §8.8 head coverings | §3.2.3 head coverings | partial |

> `NoHeadCoverings` alignment is **partial**: ICAO 9303 §3.2.3 carves out
> religious/medical exemptions that OFIQ does not model. Document this in any
> compliance preset built around the component.

## Geometric / pose components (S8.x)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `HeadPoseYaw.scalar` | S8 | Yaw | §8.2 pose | §3.2.3 frontal | exact |
| `HeadPosePitch.scalar` | S8 | Pitch | §8.2 pose | §3.2.3 frontal | exact |
| `HeadPoseRoll.scalar` | S8 | Roll | §8.2 pose | §3.2.3 frontal | exact |
| `LeftwardCropOfTheFaceImage.scalar` | S8 | Left margin | §7.4 token formats | §3.2.2 centring | exact |
| `RightwardCropOfTheFaceImage.scalar` | S8 | Right margin | §7.4 token formats | §3.2.2 centring | exact |
| `MarginAboveOfTheFaceImage.scalar` | S8 | Top margin | §7.4 token formats | §3.2.2 head position | exact |
| `MarginBelowOfTheFaceImage.scalar` | S8 | Bottom margin | §7.4 token formats | §3.2.2 head position | exact |

## Coverage summary

| Standard | Components covered | Notes |
|---|---|---|
| ISO/IEC 29794-5 | **28 / 28** | Full coverage of OFIQ-implemented components plus forward-looking RadialDistortion. |
| ISO/IEC 19794-5 | **27 / 28** | RadialDistortion not directly addressed by 19794-5 (predates the lens-distortion clause in 29794-5). |
| ICAO 9303 Part 9 | **27 / 28** | Same — RadialDistortion is informative-only at ICAO level. |

## CLI presets enabled by this mapping

Once `1.Y.4` lands in `cli.py`, these become available:

```bash
# Generate dataset filtering to ICAO-strict requirements
ofiq-syngen generate-dataset --preset icao-strict --images-dir ./faces --output-dir ./out

# Filter to ISO/IEC 19794-5 token-format requirements
ofiq-syngen generate-dataset --preset iso-19794-5

# Default: all 28 components (ISO 29794-5)
ofiq-syngen generate-dataset --preset iso-29794-5
```

Preset definitions live in `src/ofiq_syngen/standards.py` and read from this
mapping table; CSV is the source of truth.

## Known unknowns

The four `?`-flagged rows above (`SingleFacePresent`, `ExpressionNeutrality`,
`NoHeadCoverings` clauses) need verification against the OFIQ Algorithm Book and
the published ISO/IEC 29794-5:2024 PDF. Once verified, remove the `?` and bump
`confidence` to `verified` in `MAPPING.csv`.
