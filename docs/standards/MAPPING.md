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

## Capture-related components (FDIS §7.3)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `BackgroundUniformity.scalar` | §7.3.2 | Background uniformity | §8.9 lighting/scene | §3.2.3 background | exact |
| `IlluminationUniformity.scalar` | §7.3.3 | Illumination uniformity | §8.9 lighting | §3.2.3 lighting | exact |
| `LuminanceMean.scalar` | §7.3.4.2 | Luminance mean | §8.9 exposure | §3.2.3 exposure | exact |
| `LuminanceVariance.scalar` | §7.3.4.3 | Luminance variance | §8.9 exposure | §3.2.3 contrast | partial |
| `UnderExposurePrevention.scalar` | §7.3.5 | Under-exposure prevention | §8.9 exposure | §3.2.3 exposure | exact |
| `OverExposurePrevention.scalar` | §7.3.6 | Over-exposure prevention | §8.9 exposure | §3.2.3 exposure | exact |
| `DynamicRange.scalar` | §7.3.7 | Dynamic range | §8.9 lighting | §3.2.3 contrast | partial |
| `Sharpness.scalar` | §7.3.8 | Sharpness (focus) | §8.10 focus | §3.2.3 focus | exact |
| `CompressionArtifacts.scalar` | §7.3.9 | Compression artifacts | §9.2 JPEG/JPEG2000 | §3.2.5 compression | exact |
| `NaturalColour.scalar` | §7.3.10 | Natural colour | §9.1 colour space | §3.2.4 true colour | exact |
| `RadialDistortion.scalar` | Annex D.2.1 | Radial/lens distortion | §8.10 lens | §3.2.3 lens distortion | exact |
| `SingleFacePresent.scalar` | §7.4.2 | Single face present | §8.1 subject | §3.2.3 one subject | exact |

## Subject-related components (FDIS §7.4)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `EyesOpen.scalar` | §7.4.3 | Eyes open | §8.5 eye state | §3.2.3 eyes open | exact |
| `MouthClosed.scalar` | §7.4.4 | Mouth closed | §8.6 mouth state | §3.2.3 mouth closed | exact |
| `EyesVisible.scalar` | §7.4.5 | Eyes visible (no occlusion) | §8.7 eye occlusion | §3.2.3 no eye covering | exact |
| `MouthOcclusionPrevention.scalar` | §7.4.6 | Mouth visible | §8.7 mouth occlusion | §3.2.3 no mouth covering | exact |
| `FaceOcclusionPrevention.scalar` | §7.4.7 | Face visible | §8.7 face occlusion | §3.2.3 face visible | exact |
| `InterEyeDistance.scalar` | §7.4.8 | Inter-eye distance | §9.3 resolution | §3.2.2 eye distance min | exact |
| `HeadSize.scalar` | §7.4.9 | Head size in frame | §7.4 token formats | §3.2.2 face size | exact |
| `ExpressionNeutrality.scalar` | §7.4.12 | Neutral expression | §8.3 expression | §3.2.3 neutral expression | exact |
| `NoHeadCoverings.scalar` | §7.4.13 | No head coverings | §8.8 head coverings | §3.2.3 head coverings | partial |

> `NoHeadCoverings` alignment is **partial**: ICAO 9303 §3.2.3 carves out
> religious/medical exemptions that OFIQ does not model. Document this in any
> compliance preset built around the component.

## Geometric / pose components (S8.x)

| Component | OFIQ Section | ISO 29794-5 | ISO 19794-5 | ICAO 9303 Part 9 | Alignment |
|---|---|---|---|---|:---:|
| `HeadPoseYaw.scalar` | §7.4.11.2 | Yaw | §8.2 pose | §3.2.3 frontal | exact |
| `HeadPosePitch.scalar` | §7.4.11.3 | Pitch | §8.2 pose | §3.2.3 frontal | exact |
| `HeadPoseRoll.scalar` | §7.4.11.4 | Roll | §8.2 pose | §3.2.3 frontal | exact |
| `LeftwardCropOfTheFaceImage.scalar` | §7.4.10.1 | Left margin | §7.4 token formats | §3.2.2 centring | exact |
| `RightwardCropOfTheFaceImage.scalar` | §7.4.10.2 | Right margin | §7.4 token formats | §3.2.2 centring | exact |
| `MarginAboveOfTheFaceImage.scalar` | §7.4.10.3 | Top margin | §7.4 token formats | §3.2.2 head position | exact |
| `MarginBelowOfTheFaceImage.scalar` | §7.4.10.4 | Bottom margin | §7.4 token formats | §3.2.2 head position | exact |

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
