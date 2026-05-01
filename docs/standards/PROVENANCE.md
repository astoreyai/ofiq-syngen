# OFIQ Source-Line Provenance

For every `ofiq-syngen` degradation function, the OFIQ-Project C++ source it
was ported from. Citations follow the format
`OFIQ-Project/OFIQlib/modules/measures/src/<File>.cpp:Lstart-Lend (OFIQ v<ver>)`.

**OFIQ version pinned**: `1.1.0` (per `OFIQ-Project/Version.txt` at time of
this document; see [`SOURCES.md`](SOURCES.md) for full version notes).

**Convention**: `Execute(Session&)` is the entry-point method on every OFIQ
measure class; line ranges below cover that method body. Helper utilities
(`landmark_utils`, `gpu_ofiq_scorer`, `face_context`) port from
`OFIQlib/modules/landmarks/`, `OFIQlib/modules/utils/`, and the ONNX model
loaders — those are catalogued in a separate provenance pass.

## Capture-related (FDIS §7.3)

| Component | OFIQ Source | Lines | Notes |
|---|---|---|---|
| `BackgroundUniformity.scalar` | `BackgroundUniformity.cpp` | L55-L168 | Sobel gradient on rec.709 luminance, masked by BiSeNet class 0 eroded 4×4. Steps 1-9 commented in source. |
| `IlluminationUniformity.scalar` | `IlluminationUniformity.cpp` | L1-L113 | 256-bin histogram intersection of L/R eye ROI zones (zoneSize = IED × 0.3). |
| `LuminanceMean.scalar` | `Luminance.cpp` | L41-L76 | Weighted Y-channel mean, sigmoid peak at 0.5. **Multi-component file**: also covers LuminanceVariance. |
| `LuminanceVariance.scalar` | `Luminance.cpp` | L41-L76 | `sin((60·var)/(60·var+1)·π)` on Y-channel variance. Same `Execute` as above. |
| `UnderExposurePrevention.scalar` | `UnderExposurePrevention.cpp` | L1-L72 | Fraction of face∩occlusion-mask pixels with Y ∈ [0,25]. |
| `OverExposurePrevention.scalar` | `OverExposurePrevention.cpp` | L1-L71 | Fraction of face-mask pixels with Y ∈ [247,255]. |
| `DynamicRange.scalar` | `DynamicRange.cpp` | L1-L88 | `12.5 × Shannon_entropy(Y_histogram)` on face mask. |
| `Sharpness.scalar` (3 variants) | `Sharpness.cpp` | L1-L214 | Random Forest on 26 features: Laplacian + MeanDiff + Sobel at 5 kernel sizes. |
| `CompressionArtifacts.scalar` | `CompressionArtifacts.cpp` | L1-L111 | ONNX neural network on 248×248 center crop, sigmoid output. |
| `NaturalColour.scalar` | `NaturalColour.cpp` | L1-L176 | CIE Lab distance from skin range a*∈[5,25], b*∈[5,35] in L/R ROI zones. |
| `RadialDistortion.scalar` | *(not in OFIQ 1.1.0)* | — | Forward-looking ISO 29794-5 component; no upstream measurement to align against. |
| `SingleFacePresent.scalar` | `SingleFacePresent.cpp` | L1-L100 | Binary face count from ADNet detector. |

## Subject-related (FDIS §7.4)

| Component | OFIQ Source | Lines | Notes |
|---|---|---|---|
| `EyesOpen.scalar` | `EyesOpen.cpp` | L1-L61 | Min eye-opening ratio / t-metric from landmarks 61/67, 62/66, 63/65. |
| `MouthClosed.scalar` | `MouthClosed.cpp` | L1-L70 | Mouth opening magnitude from inner-lip landmarks 89/95, 90/94, 91/93. |
| `EyesVisible.scalar` | `EyesVisible.cpp` | L1-L123 | Eye visibility within EVZ rectangle (IED/20 expansion). |
| `MouthOcclusionPrevention.scalar` | `MouthOcclusionPrevention.cpp` | L1-L67 | Occlusion ratio inside mouth-polygon landmarks 76-87. |
| `FaceOcclusionPrevention.scalar` | `FaceOcclusionPrevention.cpp` | L1-L74 | Occlusion ratio inside face-mask landmarks. |
| `InterEyeDistance.scalar` | `InterEyeDistance.cpp` | L1-L71 | Pixel distance between eye centers from ADNet landmarks. |
| `HeadSize.scalar` | `HeadSize.cpp` | L1-L61 | t-metric / image height ratio. |
| `ExpressionNeutrality.scalar` | `ExpressionNeutrality.cpp` | L1-L154 | Dual CNN ensemble + AdaBoost. |
| `NoHeadCoverings.scalar` | `NoHeadCoverings.cpp` | L1-L123 | BiSeNet class 16 + 18 (hat + cloth) pixel ratio. |

## Geometric / pose (S8.x)

| Component | OFIQ Source | Lines | Notes |
|---|---|---|---|
| `HeadPoseYaw.scalar` | `HeadPose.cpp` | L49-L59 | 3DDFA-V2 → `100 × cos²(yaw)`. **Multi-component file**: covers Yaw, Pitch, Roll. |
| `HeadPosePitch.scalar` | `HeadPose.cpp` | L49-L59 | Same `Execute`; pitch axis. |
| `HeadPoseRoll.scalar` | `HeadPose.cpp` | L49-L59 | Same `Execute`; roll axis. |
| `LeftwardCropOfTheFaceImage.scalar` | `CropOfTheFaceImage.cpp` | L69-L113 | `rightEyeCenter.x / IED`. **Multi-component file**: covers all 4 crop/margin measures. |
| `RightwardCropOfTheFaceImage.scalar` | `CropOfTheFaceImage.cpp` | L69-L113 | `(imgW - leftEyeCenter.x) / IED`. |
| `MarginAboveOfTheFaceImage.scalar` | `CropOfTheFaceImage.cpp` | L69-L113 | `eyeMidPoint.y / t`. |
| `MarginBelowOfTheFaceImage.scalar` | `CropOfTheFaceImage.cpp` | L69-L113 | `(imgH - eyeMidPoint.y) / t`. |

## Helper-utility provenance

| `ofiq-syngen` symbol | OFIQ-Project source | Lines | Notes |
|---|---|---|---|
| `landmark_utils.PAIRS_LEFT_EYE`, `PAIRS_RIGHT_EYE`, `PAIRS_MOUTH_INNER`, `LEFT_EYE`, `RIGHT_EYE`, `MOUTH_INNER`, `MOUTH_OUTER` | `OFIQlib/modules/landmarks/adnet_FaceMap.h` | L1-L159 | 98-point ADNet landmark index map. `ofiq-syngen` ports the full table. |
| `landmark_utils.LEFT_EYE_CORNERS`, `RIGHT_EYE_CORNERS` | `OFIQlib/modules/landmarks/adnet_FaceMap.h` | L1-L159 | Eye-corner index pairs derived from same map. |
| `landmark_utils.tmetric`, `IED` | `OFIQlib/modules/landmarks/src/FaceMeasures.cpp` | L1-L244 | `static double InterEyeDistance(landmarks, yaw)` declared at `FaceMeasures.h:66`. |
| `landmark_utils.face_mask`, `evz_rects`, ROI-zone helpers | `OFIQlib/modules/utils/src/image_utils.cpp` | L1-L198 | Face mask construction; eye-visibility-zone rectangle math; left/right ROI zone derivation. |
| `landmark_utils.luminance_rec709` | `OFIQlib/modules/utils/src/image_utils.cpp` `GetLuminanceImageFromBGR` | decl at `image_utils.h:74` | Rec.709 luminance conversion. |
| `landmark_utils.convert_bgr_to_cielab` | `OFIQlib/modules/utils/src/image_utils.cpp` `ConvertBGRToCIELAB` | decl at `image_utils.h:65` | OFIQ's custom CIELAB implementation. |
| `landmark_utils.normalized_histogram` | `OFIQlib/modules/utils/src/image_utils.cpp` `GetNormalizedHistogram` | decl at `image_utils.h:116` | 256-bin normalized luminance histogram on a mask. |
| `landmark_utils.BISENET_BACKGROUND`, `BISENET_HAT`, `BISENET_CLOTH` | `OFIQlib/modules/segmentations/` BiSeNet class indices | (see segmentations module) | Class index constants for the 19-class BiSeNet face-parsing model. |
| `gpu_ofiq_scorer.OFIQModels` (ONNX loader) | `OFIQlib/modules/measures/src/CompressionArtifacts.cpp` ONNX loading pattern | L1-L111 | Pattern reused across all OFIQ ONNX model wrappers. |
| `face_context._run_adnet`, `face_context.FaceContext.from_image` | `OFIQlib/modules/landmarks/src/adnet_landmarks.cpp` | L1-L331 | ADNet landmark inference + post-processing. |
| `face_context._run_bisenet` | `OFIQlib/modules/segmentations/` BiSeNet wrapper | (see segmentations module) | 19-class face-parsing inference. |
| `face_context._run_occlusion_seg` | `OFIQlib/modules/segmentations/` occlusion segmenter | (see segmentations module) | Binary face/non-face occlusion mask. |
| `face_context._run_headpose` | `OFIQlib/modules/poseEstimators/` HeadPose3DDFAV2 wrapper | (see poseEstimators module) | 3DDFA-V2 pose inference. |
| `face_context.FaceContext` (top-level dataclass) | Composition of the four above | n/a | Wraps all per-image OFIQ computation as a single shared context. Used by all `needs_ctx=True` degraders to avoid redundant model invocations. |

## Reciprocal flag — OFIQ Algorithm Book quotes

`ANALYSIS.md` contains direct paraphrases of OFIQ Algorithm Book step
descriptions (e.g., "Step 1. Create a completely black (0) grey scale image
A…" in BackgroundUniformity). These are clearly cited as paraphrases of OFIQ
documentation. Verbatim quotes — if any are added — must be marked as such
under fair-use provisions of MIT-licensed code, and the OFIQ Algorithm Book
PDF citation included.

## Verification protocol

To verify any single line range above:

```bash
# From the workspace root
SRC=02_perception_biometrics/OFIQ-Project/OFIQlib/modules/measures/src
sed -n '55,168p' $SRC/BackgroundUniformity.cpp     # see Execute body
grep -n '::Execute' $SRC/BackgroundUniformity.cpp  # confirm method header line
```

When OFIQ ships a new release, line numbers will shift. Update this document
in the same commit that bumps the OFIQ pin in `SOURCES.md`.
