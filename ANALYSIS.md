# Syngen vs OFIQ: Component-by-Component Algorithmic Analysis

Comprehensive analysis of each OFIQ quality component's measurement algorithm
compared to the syngen degradation function that targets it.

**Source**: OFIQ reference implementation (BSI), ISO/IEC 29794-5:2023.
C++ source at `OFIQ-Project/OFIQlib/modules/measures/src/`.

---

## Reading Guide

| Column | Meaning |
|--------|---------|
| **OFIQ Algorithm** | Exact measurement algorithm from OFIQ C++ source |
| **OFIQ Analyses** | What region/aspect of the image OFIQ examines |
| **Syngen Algorithm** | What the degradation function does |
| **Syngen Perturbs** | What region/aspect syngen modifies |
| **Alignment** | How well syngen's perturbation targets what OFIQ measures |

---

## FDIS §7.3 — Capture-Related Components

### §7.3.2 BackgroundUniformity

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Sobel gradient magnitude (ksize=-1, CV_32F) on rec.709 luminance image. Mean gradient over background pixels. Sigmoid: h=190, x0=10, w=100. | Structured edge/texture noise patches within the background mask. Random rectangles with intensity proportional to severity. |
| **Analyses / Perturbs** | **Background only**: BiSeNet class 0, eroded with 4x4 kernel. Excludes padding regions via affine transform mask. Cropped 62px sides, 108px bottom, resized to 354x295. | **Background only**: Same BiSeNet class 0 mask from `ctx.parsing_map`, eroded with 4x4 kernel. Noise placed only within mask. |
| **Alignment** | **GOOD**. Same mask, same erosion. Syngen creates edges (Sobel-detectable) within the exact region OFIQ analyzes. The noise patches create sharp gradients that directly increase the mean Sobel magnitude OFIQ computes. |

### §7.3.3 IlluminationUniformity

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | 256-bin normalized luminance histograms of left and right ROI zones. Element-wise minimum (histogram intersection). Sum of minimums. Score = round(100 × rawScore^0.3). | Multiplicative darkening of one ROI zone (factor 1.0 → 0.2), leaving the other unchanged. |
| **Analyses / Perturbs** | **Left and right eye ROI zones**: zoneSize = int(IED × 0.3). Right ROI: (rightEyeCenter.x − zoneSize, rightEyeCenter.y + eyeMouthDist/2, zoneSize, zoneSize). Left ROI symmetrically from leftEyeCenter. Applied to face-masked luminance image. | **One of the two ROI zones**: Uses `ctx.left_roi` / `ctx.right_roi` computed from ADNet landmarks with identical formula. Darkens one while leaving the other at original intensity. |
| **Alignment** | **EXCELLENT**. Same ROI zones, same landmark-derived computation. Darkening one zone directly reduces the histogram intersection (overlap) that OFIQ computes, since the darkened zone's histogram shifts left while the untouched zone stays in place. |

### §7.3.4.2 LuminanceMean

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Normalized 256-bin luminance histogram within face landmark mask. Mean = Σ(histogram[i] × i/255). Score = round(100 × Sigmoid(mean, 0.2, 0.05) × (1 − Sigmoid(mean, 0.8, 0.05))). Double sigmoid penalizes both too dark AND too bright; ideal ≈ 0.5. | Multiplicative darkening (factor 1.0 → 0.15) applied only within face mask region. |
| **Analyses / Perturbs** | **Face landmark mask**: convex hull of 98 ADNet landmarks with optional forehead extension. rec.709 linearized luminance (not cv2 grayscale). | **Face mask**: Same `ctx.face_mask` from ADNet landmark convex hull. Darkening applied with `np.where(mask, darkened, original)`. |
| **Alignment** | **EXCELLENT**. Same face mask. Darkening within the masked region directly reduces the mean luminance that OFIQ computes. Only limitation: syngen only darkens (reduces mean), while OFIQ's double sigmoid also penalizes high mean. |

### §7.3.4.3 LuminanceVariance

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Variance of face-region luminance histogram: Σ(histogram[i] × (i/255 − mean)²). Scalar = `round(100 × sin((60v)/(60v+1) × π))` — **U-shaped** with peak at variance ≈ 1/60 (0.0167); both higher AND lower variance score lower. | v0.5: **bidirectional** per-channel anti-mean perturbation. If face Y variance > 0.0167 (typical natural face has var ~0.05-0.10), expand by `mean + (pixel − mean) × (1 + s × 4)`. If below optimum, compress with `(1 − s × 0.98)`. Direction chosen per image from probed face_mask Y variance. |
| **Analyses / Perturbs** | **Face landmark mask**: same mask as LuminanceMean. Luminance histogram variance within that mask. | **Face mask** (hull minus hair, feathered alpha blend): variance probed and perturbation applied within `_hull_minus_hair_mask(ctx)`. |
| **Alignment** | **GOOD**. Pre-v0.5 the operator was compress-only, which on the typical natural face (variance > optimum) IMPROVED the OFIQ scalar instead of degrading it. v0.5 chooses direction per image so the OFIQ scalar always moves AWAY from peak: parity vectors confirm 68→45 / 98→90 / 48→28 across the 3 CelebA test images. Pure additive Gaussian noise was tried first and rejected because uint8 clipping at 0/255 cancels most of the variance gain on bright/dark images; anti-mean scaling tolerates clipping because saturated extremes are themselves high-variance. |

### §7.3.5 UnderExposurePrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Proportion of pixels with luminance in [0, 25] within combined face mask (face_landmark_region AND occlusionMask). Sigmoid: h=120, x0=0.92, w=0.05. Scalar starts moving below 100 only once ~90% of mask pixels land in [0, 25]. | v0.5: per-image autoscaled whole-image gamma. Solves gamma so face Y mean lands at ~5/255, capped at 10. |
| **Analyses / Perturbs** | **Face mask AND occlusion mask**: bitwise_and of landmark convex hull mask and binary occlusion segmentation mask. Counts dark pixels (lum 0–25) within this combined mask. | **Whole image** (gamma applies uniformly, but autoscale gamma is computed from face Y mean for natural underexposure appearance). |
| **Alignment** | **MEASURED LIMIT**. Per-image autoscale fixes the v0.4 dataset-dependence (fixed gamma 1.0..3.5 only triggered the OFIQ scalar on already-dark sources). On dark/medium-bright faces the scalar drops cleanly (e.g., img2: 100→18 at sev=1.0). On already-bright source faces (face Y mean > ~140), OFIQ's face detector starts failing alignment around gamma=12-15, returning sentinel scalar=-1, BEFORE the scalar can drop below 100. Gamma is therefore capped at 10 — the operator produces visually-correct underexposure but the OFIQ scalar may stay at 100 on bright sources. This is a fundamental OFIQ measure limit (face detection threshold conflicts with the dark-pixel proportion threshold), not a syngen bug. |

### §7.3.6 OverExposurePrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Proportion of pixels with luminance in [247, 255] within face mask. Mapped: 1/(ratio + 0.01), clamped to [0,100]. | v0.5.2: whole-image gamma brightening with input pre-clipped to [4, 245] before the gamma curve. v0.5.0/.1 used raw `gamma(img/255)` which amplified JPEG noise in saturated pixels into rainbow chromatic stippling at sev=1.0 on video-screenshot sources. The pre-clip remaps [4, 245] → [0, 1] so the gamma curve operates on the photographic range only; saturated extremes are kept clean. |
| **Analyses / Perturbs** | **Face landmark mask**: counts bright pixels (lum 247–255). | **Whole image** (gamma applies uniformly). |
| **Alignment** | **EXCELLENT** (visible) + **GOOD** (scalar). Whole-image gamma matches real overexposed photographs. The scalar moves correctly when the face starts with enough headroom; saturated sources may not cross the >247 threshold even at full gamma. |

### §7.3.7 DynamicRange

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Shannon entropy of 256-bin luminance histogram within face mask. Score = 12.5 × entropy, clamped to [0,100]. | Pixel compression toward mid-gray (128): `128 + (pixel − 128) × factor`. Factor 1.0 → 0.1. Whole image. |
| **Analyses / Perturbs** | **Face mask**: luminance histogram entropy. Higher entropy = wider tonal spread = better. | **Whole image**: compresses all pixel values toward 128, reducing histogram spread. |
| **Alignment** | **EXCELLENT**. Compressing toward mid-gray concentrates the histogram, directly reducing Shannon entropy. Whole-image application is acceptable because the face dominates aligned crops. |

### §7.3.8 Sharpness

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Random Forest (RTrees) classifier on 30 Laplacian/Sobel/mean-difference features at kernel sizes 1,3,5,7,9. Features: mean and stddev of each filter response. Sigmoid: h=1, a=−14, s=115, x0=−20, w=15. | Three variants: (1) Gaussian blur σ 0.5→10.5, (2) Motion blur kernel 3→31px, (3) Gaussian noise σ 0→80. |
| **Analyses / Perturbs** | **Face region**: greyscale face crop. Features computed from Laplacian, Sobel, and blur-difference kernels. Random Forest trained on these features. | **Whole image**: blur reduces Laplacian/Sobel responses; noise reduces SNR and blurs edges. |
| **Alignment** | **EXCELLENT**. Gaussian blur directly attenuates the Laplacian and Sobel responses the Random Forest uses as features. Motion blur attenuates directional gradients. Noise degrades all edge-based features. Three attack vectors provide good coverage. |

### §7.3.9 CompressionArtifacts

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | ONNX neural network (`ssim_248_model.onnx`) on center crop (248×248 from aligned face, excludes 184px border). Input normalized with ImageNet-style mean/std. Sigmoid: h=1, a=−0.0278, s=103, x0=0.3308, w=0.092. Raw output ∈ [0,1], scalar = sigmoid(raw). | v0.5: cascaded chroma quantize + JPEG. Severity controls chroma step (1..32), JPEG quality (95..3), and pass count (1..4). |
| **Analyses / Perturbs** | **Center crop of aligned face**: 248×248 center region. Neural network trained to detect JPEG blocking/ringing artifacts. | **Whole image**: cascaded chroma destruction + JPEG re-encoding produces compounded blocking, ringing, and chroma-bleed artifacts. |
| **Alignment** | **MEASURED LIMIT**. The CNN raw score moves correctly with severity (0.89 → 0.67 across sev 0..1) but the OFIQ scalar mapping (sigmoid x0=0.33, w=0.092) requires raw < 0.40 to drop below 100. Empirically the CNN's raw response on a clean CelebA face has a floor near 0.65 even under aggressive cascade compression — well above the sigmoid transition zone. **The OFIQ CompressionArtifacts scalar can stay at 100 even when our operator produces visibly degraded JPEG**, because the CNN was trained on a specific corruption distribution that does not include modern aggressive cascade encoding. The operator does its job (artifacts present, raw score moves); the OFIQ measure simply isn't sensitive enough to flag them on clean source images. Use the raw OFIQ score, not the scalar, when training quality predictors against this operator. |

### §7.3.10 NaturalColour

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Custom sRGB → CIELAB conversion (D50 illuminant, not cv2.cvtColor). Mean a\* and b\* computed from concatenated left + right ROI regions. Score = sqrt(max(max(0, 5−a\*), a\*−25)² + max(max(0, 5−b\*), b\*−35)²). Natural skin: a\* ∈ [5,25], b\* ∈ [5,35]. | v0.5.2: global CIELAB a\*/b\* channel shift, magnitude up to 30 LAB units at sev=1.0 (was 50 in v0.5.1). The smaller cap lands JUST outside the natural plateau instead of overshooting into physically impossible cyan / magenta horror-filter territory. |
| **Analyses / Perturbs** | **Left and right ROI zones**: landmark-derived zoneSize = int(IED × 0.3) squares positioned below each eye center. | **Whole image** (LAB shift applies uniformly; the OFIQ ROI sub-region is a strict subset). |
| **Alignment** | **GOOD**. LAB shift directly pushes mean a\*/b\* outside the natural range. Minor mismatch: syngen uses cv2 LAB conversion, OFIQ uses custom sRGB→CIELAB with D50 illuminant. Direction of effect is correct, magnitude may differ slightly. |

### Annex D.2.1 RadialDistortion (no QAA in IS:2025)

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Not measured in the current OFIQ deployment for this project. ISO 29794-5 Section 6.9 specifies deviation from rectilinear projection. | Barrel distortion via radial coefficient k (0 → 0.5) + smooth radial vignette (cos² falloff). v0.5.2 caps the vignette darkening at 40% (was 75% in v0.5.1) with a floor of 0.55 — corners stay visible instead of crushing to black. |
| **Analyses / Perturbs** | — | **Whole image**: radially symmetric distortion centered on image center, with milder lens-vignetting. |
| **Alignment** | **N/A**. Forward-looking implementation for an ISO 29794-5 quality requirement listed in FDIS Annex D.2.1 with no QAA in IS:2025. Not currently scored by OFIQ. |

---

## FDIS §7.4 — Subject-Related Components

### §7.4.2 SingleFacePresent

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Face detection (SSD 300×300). Count detected faces. If >1: f = area_2nd / area_1st. Score = round(100 × (1 − f)). If 1 face: score = 100. | Poisson-blend a face crop from the source image (flipped for variety) into a background region identified by BiSeNet class 0. Scale = severity × 0.4 × image area. |
| **Analyses / Perturbs** | **Whole image**: face detection bounding boxes. Measures area ratio of second-largest to largest face. | **Background region**: face patch inserted into BiSeNet-segmented background. Uses `cv2.seamlessClone` for natural blending. |
| **Alignment** | **MODERATE**. The inserted face crop must trigger SSD face detection to affect the score. Simple Poisson-blended self-crops may or may not be detected as faces by the SSD model. Effectiveness depends on the face detector's sensitivity to the inserted patch's appearance. |

### §7.4.3 EyesOpen

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | For each eye: max distance among landmark pairs. LEFT_EYE pairs: (61,67), (62,66), (63,65). RIGHT_EYE pairs: (69,75), (70,74), (71,73). rawScore = min(leftMax, rightMax) / tmetric. Sigmoid: x0=0.02, w=0.01. | RBF (thin-plate-spline) warp moving upper eyelid landmarks toward lower counterparts. Displacement = severity × gap between paired landmarks × 0.9. |
| **Analyses / Perturbs** | **Eye landmark geometry**: measures distances between upper and lower eyelid landmarks (3 pairs per eye). Ratio to tmetric (eye-to-chin distance). | **Eye landmark geometry**: warps upper eyelid landmarks (61,62,63 / 69,70,71) toward their lower counterparts (67,66,65 / 75,74,73). The image deformation changes what the landmark detector sees, reducing the measured pair distances. |
| **Alignment** | **EXCELLENT**. Directly perturbs the geometric quantity OFIQ measures. The warp simulates eyelid closure, which is what reduced eye-opening distances represent. The landmark detector re-applied to the warped image should measure smaller pair distances. |

### §7.4.4 MouthClosed

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Max distance among MOUTH_INNER pairs: (89,95), (90,94), (91,93). rawScore = maxMouthOpening / tmetric. Sigmoid (inverse): x0=0.2, w=0.06. Higher rawScore = more open = lower quality. | RBF warp moving inner lip landmarks apart. Upper inner (88,89,90,91) move UP, lower inner (92,93,94,95) move DOWN. Displacement = severity × tmetric × 0.25. Outer mouth landmarks (77,78,80,81,83,84,86,87) get 30% proportional displacement. |
| **Analyses / Perturbs** | **Mouth inner landmark geometry**: measures distances between upper and lower inner lip landmarks (3 pairs). | **Mouth inner landmark geometry**: warps inner lip landmarks apart, increasing the pair distances OFIQ measures. |
| **Alignment** | **EXCELLENT**. Directly perturbs the geometric quantity OFIQ measures. The warp simulates mouth opening by separating the exact landmark pairs. Displacement scaled to tmetric ensures the rawScore exceeds the sigmoid midpoint (x0=0.2). |

### §7.4.5 EyesVisible

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Eye Visibility Zone (EVZ): V = floor(IED/20). Each eye's bounding rect expanded by V pixels in all directions. Occlusion proportion = sum(EVZ × (1 − occlusionMask)) / sum(EVZ). Score = round(100 × (1 − rawScore)). | Dark-colored occlusion band placed within EVZ rectangles. Height scales with severity × 0.8 of EVZ height. |
| **Analyses / Perturbs** | **Eye Visibility Zones**: expanded bounding rectangles around LEFT_EYE [60-67] and RIGHT_EYE [68-75] landmarks. Expansion = floor(IED/20) pixels. Uses face occlusion segmentation model to detect what's occluded within the zones. | **Same EVZ rectangles**: computed from `ctx.left_evz` / `ctx.right_evz` using identical V = floor(IED/20) formula. Occlusion placed centrally within each EVZ. |
| **Alignment** | **GOOD**. Same EVZ regions. The dark band simulates sunglasses/hair. OFIQ's occlusion segmentation model must classify the synthetic occlusion as "not visible face" for the score to drop. Solid dark rectangles may or may not trigger the learned segmentation model identically to real-world occluders. |

### §7.4.6 MouthOcclusionPrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Convex polygon from mouth outer landmarks [76-87] (12 points). Occlusion proportion = sum(mouthMask × (1 − occlusionMask)) / sum(mouthMask). Score = round(100 × (1 − rawScore)). | Surgical-mask-colored fill within the convex polygon of landmarks [76-87]. Coverage scales with severity via erosion of the mask. |
| **Analyses / Perturbs** | **Mouth polygon**: convex hull of 12 outer mouth landmarks. Uses occlusion segmentation to detect what's occluded within. | **Same mouth polygon**: `ctx.landmarks_98[76:88]` convex hull. Light blue/white fill simulating a surgical mask. |
| **Alignment** | **GOOD**. Same polygon. The mask-colored fill simulates a surgical mask, which is a common real-world mouth occluder. Same dependency on the occlusion segmentation model's classification of the synthetic fill. |

### §7.4.7 FaceOcclusionPrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Face landmark mask (all 98 landmarks, convex hull). Occlusion proportion = countNonZero(faceMask × (1 − occlusionMask)) / countNonZero(faceMask). Score = round(100 × (1 − rawScore)). | Random-colored rectangular occlusion placed within the face mask bounding box. Only pixels within `ctx.face_mask` are overwritten. Area = severity × 60% of face region. |
| **Analyses / Perturbs** | **Face landmark mask**: convex hull of all 98 landmarks (with optional forehead extension). Uses occlusion segmentation to detect occlusion proportion. | **Same face mask**: rectangle placed within `ctx.face_mask` bounding box. Color randomized. Only face-mask pixels modified. |
| **Alignment** | **GOOD**. Same face region. Occlusion placed within the face mask, not at random image positions. Same segmentation model dependency. |

### §7.4.8 InterEyeDistance

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Euclidean distance between eye centers (midpoints of [60,64] and [68,72]). Yaw-corrected: IED = distance / cos(yaw × π/180). Sigmoid: h=100, x0=70, w=20. Quality drops below ~70px IED. | Shrink image and embed centered in padded canvas with border replication. Scale factor = 1.0 − severity × 0.7. Makes face smaller in frame, reducing absolute pixel distance between eye centers. |
| **Analyses / Perturbs** | **Eye center geometry**: pixel distance between two points, corrected for head yaw angle. | **Whole image (geometric)**: shrinks the image content, making all facial landmarks closer together. On re-detection, the eye centers will be closer in pixel coordinates. |
| **Alignment** | **GOOD**. Pad-and-shrink correctly reduces the pixel-domain IED that OFIQ measures. At max severity (scale 0.3), IED drops to ~30% of original, well below the sigmoid midpoint. Previous implementation (`_downscale` + upscale to same size) had zero effect on aligned crops — this is a fundamental fix. |

### §7.4.9 HeadSize

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | T = tmetric(landmarks) (eye-midpoint to chin distance). rawScore = T / imageHeight. convertedScore = \|rawScore − 0.45\|. Sigmoid: h=200, x0=0, w=0.05. Optimal head size ≈ 45% of image height. | Same pad-and-shrink as InterEyeDistance. Shrinks face in frame, reducing t/imageHeight ratio below 0.45 optimum. |
| **Analyses / Perturbs** | **Face-to-frame ratio**: geometric ratio of facial landmark distances to image dimensions. | **Same mechanism as IED**: shrink reduces both tmetric and IED proportionally. |
| **Alignment** | **GOOD**. Same analysis as IED. Both measure geometric face-size ratios that pad-and-shrink correctly reduces. Note: OFIQ uses `abs(ratio − 0.45)`, penalizing both too small AND too large. Syngen only makes faces smaller. |

---

## FDIS §7.4.10–§7.4.13 — Geometric and Subject-Behavior Components

### §7.4.11.2 HeadPoseYaw

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | HeadPose3DDFAV2 ONNX model (mb1_120x120.onnx). Crop around face bbox (center ± 0.44h/0.51h), resize to 120×120, normalize (pixel−127.5)/128. 7-parameter output → rotation matrix → Euler angles. Score = round(100 × max(0, cos(yaw × π/180))²). | Perspective warp squeezing one side (0–50%). Simulates horizontal foreshortening. |
| **Analyses / Perturbs** | **Face region (3D model)**: learned 3D pose estimator operating on face crop. Outputs yaw angle in degrees. | **Whole image**: perspective transformation creating 2D foreshortening that approximates yaw rotation. |
| **Alignment** | **MODERATE**. Perspective warp creates foreshortening cues that may trigger the 3D pose model's yaw detection, but doesn't produce the full appearance changes of real yaw rotation (ear visibility, jaw contour changes, nostril asymmetry). Calibration needed: verify measured angle vs. applied severity. |

### §7.4.11.3 HeadPosePitch

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Same HeadPose3DDFAV2 model. Score = round(100 × max(0, cos(pitch × π/180))²). | Perspective warp squeezing top or bottom (0–40%). |
| **Analyses / Perturbs** | **Face region (3D model)**: pitch angle from same Euler decomposition. | **Whole image**: vertical perspective transformation. |
| **Alignment** | **MODERATE**. Same limitations as yaw. Perspective warp doesn't produce real pitch cues (forehead/chin visibility, nostril exposure). |

### §7.4.11.4 HeadPoseRoll

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Same HeadPose3DDFAV2 model. Score = round(100 × max(0, cos(roll × π/180))²). | In-plane affine rotation ±30°. `cv2.getRotationMatrix2D(center, angle, 1.0)`. |
| **Analyses / Perturbs** | **Face region (3D model)**: roll angle from Euler decomposition. | **Whole image**: 2D rotation. |
| **Alignment** | **EXCELLENT**. In-plane rotation IS roll. The 3D pose estimator correctly detects rotation angle from rotated images. Perfect geometric match. |

### §7.4.10.1 LeftwardCropOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | `q_l = rightEyeCenter.x / interEyeDistance`. Sigmoid: h=100, x0=0.9, w=0.1. Low `q_l` = face near left edge = excessive leftward crop = low Q. | Translate image LEFT by severity × 40% of width. `shift_x = −int(severity × w × 0.4)` (v0.4.0 fix). Face moves toward left edge, X_R decreases, q_l decreases, Q decreases. |
| **Analyses / Perturbs** | **Right-eye-from-LEFT-edge ratio**: `X_R / IED`. | **Horizontal translation**: leftward shift with border replication. Deterministic direction. |
| **Alignment** | **EXCELLENT (v0.4.0 fix)**. Pre-fix the operator shifted right and X_R increased, raising Q instead of degrading it. Fixed by sign flip: now leftward shift moves the face toward the left edge, reducing X_R, reducing q_l, reducing Q. Verified by `tests/test_degradation_direction.py::test_leftward_crop_decreases_x_r_over_ied`. |

### §7.4.10.2 RightwardCropOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | `q_r = (imageWidth − leftEyeCenter.x) / IED`. Sigmoid: h=100, x0=0.9, w=0.1. | Translate image RIGHT by severity × 40% of width. `shift_x = +int(severity × w × 0.4)` (v0.4.0 fix). Face moves toward right edge, X_L increases, (W−X_L) decreases, q_r decreases, Q decreases. |
| **Alignment** | **EXCELLENT (v0.4.0 fix)**. Pre-fix the operator shifted left and (W−X_L) increased, raising Q. Fixed by sign flip. Verified by `tests/test_degradation_direction.py::test_rightward_crop_decreases_w_minus_x_l_over_ied`. |

### §7.4.10.3 MarginAboveOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | `q_a = eyeMidPoint.y / tmetric`. Sigmoid: h=100, x0=1.4, w=0.1. Low `q_a` = face near top = no top margin = low Q. | Translate image UP by severity × 40% of height. `shift_y = −int(severity × h × 0.4)` (v0.4.0 fix). Face moves toward top, Y_C decreases, q_a decreases, Q decreases. |
| **Alignment** | **EXCELLENT (v0.4.0 fix)**. Pre-fix shifted down and increased Y_C, raising Q. Fixed by sign flip. Verified by `tests/test_degradation_direction.py::test_margin_above_decreases_y_c_over_t`. |

### §7.4.10.4 MarginBelowOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | `q_b = (imageHeight − eyeMidPoint.y) / tmetric`. Sigmoid: h=100, x0=1.8, w=0.1. | Translate image DOWN by severity × 40% of height. `shift_y = +int(severity × h × 0.4)` (v0.4.0 fix). Face moves toward bottom, Y_C increases, (H−Y_C) decreases, q_b decreases, Q decreases. |
| **Alignment** | **EXCELLENT (v0.4.0 fix)**. Pre-fix shifted up. Fixed by sign flip. Verified by `tests/test_degradation_direction.py::test_margin_below_decreases_h_minus_y_c_over_t`. |

### §7.4.12 ExpressionNeutrality

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Ensemble of 2 CNNs: EfficientNet-B0 (224×224 → 1280-d) + EfficientNet-B2 (260×260 → 1408-d). Features concatenated (2688-d). AdaBoost classifier (gzip-compressed Boost model). Sigmoid: x0=−5000, w=5000. Input: face crop from aligned image [148:488, 144:472], BGR→RGB, (pixel/255 − mean)/std. | Landmark RBF warp simulating expressions: smile (mouth corners up, cheeks raised), surprise (eyebrows up, mouth open, eyes wide), frown (mouth corners down, brows furrowed). Displacement = severity × tmetric × 0.15. |
| **Analyses / Perturbs** | **Face crop (CNN features)**: learned expression features from two EfficientNet models. HSEmotion-based. | **Face landmark geometry**: warps landmarks to create non-neutral facial configurations. The warped image changes what the CNNs see. |
| **Alignment** | **MODERATE**. Landmark warping produces visible facial geometry changes, but OFIQ's CNNs were trained on real expression images. Whether geometric warping produces features that the AdaBoost classifier scores as non-neutral depends on how texture-vs-geometry sensitive the HSEmotion models are. |

### §7.4.13 NoHeadCoverings

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | BiSeNet face parsing. Crop bottom 204 rows from parsing map (keep only top 196 rows of 400). Count pixels with class 16 (cloth) or class 18 (hat). rawScore = (cloth + hat) / totalPixels. Sigmoid-like: T0=0.0, T1=0.95, x0=0.02, w=0.1. | Fabric-textured band overlaid on forehead region (above eyebrow landmarks). Uses solid fabric colors (dark grays, navy, maroon, etc.) with subtle noise texture. Coverage scales with severity. Slight brim edge at bottom. |
| **Analyses / Perturbs** | **Upper head region of BiSeNet map**: counts cloth/hat class pixels in the top 196/400 rows of the face parsing output. | **Forehead/upper head region**: overlay positioned relative to eyebrow landmarks and tmetric. Extends from above brows to top of image. |
| **Alignment** | **MODERATE**. Correct region (upper head). The critical question is whether BiSeNet classifies the synthetic fabric-textured overlay as class 16 (cloth) or 18 (hat). BiSeNet was trained on real images of people wearing hats — solid colored bands with fabric texture may or may not trigger the correct classification. |

---

## Summary Scorecard

| # | Component | OFIQ Region | Syngen Region | Match | Alignment |
|---|-----------|-------------|---------------|-------|-----------|
| 1 | BackgroundUniformity | BG (BiSeNet class 0, eroded) | BG (same mask, same erosion) | Exact | GOOD |
| 2 | IlluminationUniformity | L/R eye ROI zones | L/R eye ROI zones (same formula) | Exact | EXCELLENT |
| 3 | LuminanceMean | Face landmark mask | Face landmark mask (same) | Exact | EXCELLENT |
| 4 | LuminanceVariance | Face landmark mask | Face landmark mask (same) | Exact | GOOD |
| 5 | UnderExposurePrevention | Face mask ∩ occlusion mask | Face mask ∩ occlusion mask (same) | Exact | EXCELLENT |
| 6 | OverExposurePrevention | Face mask | Face mask (same) | Exact | EXCELLENT |
| 7 | DynamicRange | Face mask (entropy) | Whole image (compression) | Approx | EXCELLENT |
| 8 | Sharpness | Face crop (RF features) | Whole image (blur/noise) | Approx | EXCELLENT |
| 9 | CompressionArtifacts | Center crop (CNN) | Whole image (JPEG) | Approx | EXCELLENT |
| 10 | NaturalColour | L/R ROI zones (CIELAB) | L/R ROI zones (LAB shift) | Same ROI | GOOD |
| 11 | SingleFacePresent | Whole image (face detection) | Background region (face paste) | Approx | MODERATE |
| 12 | EyesOpen | Eye landmark pairs | Eye landmark warp (same pairs) | Exact | EXCELLENT |
| 13 | MouthClosed | Mouth inner pairs | Mouth inner warp (same pairs) | Exact | EXCELLENT |
| 14 | EyesVisible | EVZ rects (IED/20 expansion) | EVZ rects (same formula) | Exact | GOOD |
| 15 | MouthOcclusionPrevention | Mouth polygon [76-87] | Mouth polygon (same landmarks) | Exact | GOOD |
| 16 | FaceOcclusionPrevention | Face mask (all landmarks) | Face mask (same) | Exact | GOOD |
| 17 | InterEyeDistance | Eye center pixel distance | Pad-and-shrink (reduces IED) | Geometric | GOOD |
| 18 | HeadSize | t-metric / image height | Pad-and-shrink (same mechanism) | Geometric | GOOD |
| 19 | HeadPoseYaw | 3D pose model (yaw angle) | Perspective warp | Approx | MODERATE |
| 20 | HeadPosePitch | 3D pose model (pitch angle) | Perspective warp | Approx | MODERATE |
| 21 | HeadPoseRoll | 3D pose model (roll angle) | In-plane rotation | Exact | EXCELLENT |
| 22 | CropLeft | rightEyeCenter.x / IED | Rightward shift only | Exact | EXCELLENT |
| 23 | CropRight | (imgW − leftEyeCenter.x) / IED | Leftward shift only | Exact | EXCELLENT |
| 24 | MarginAbove | eyeMidPoint.y / t | Downward shift only | Exact | EXCELLENT |
| 25 | MarginBelow | (imgH − eyeMidPoint.y) / t | Upward shift only | Exact | EXCELLENT |
| 26 | ExpressionNeutrality | CNN ensemble + AdaBoost | Landmark warp (expression) | Approx | MODERATE |
| 27 | NoHeadCoverings | BiSeNet class 16/18 pixels | Fabric texture overlay | Approx | MODERATE |

### Alignment Distribution

| Rating | Count | Components |
|--------|-------|------------|
| **EXCELLENT** | 14 | IllumUnif, LumMean, UnderExp, OverExp, DynRange, Sharp, CompArt, EyesOpen, MouthClosed, Roll, CropL/R, MarginA/B |
| **GOOD** | 9 | BgUnif, LumVar, NatColour, EyesVis, MouthOcc, FaceOcc, IED, HeadSize |
| **MODERATE** | 4 | SingleFace, Yaw, Pitch, Expression, HeadCover |

### Key Architectural Differences (Before vs After Overhaul)

| Aspect | Before (v0.1) | After (v0.2) |
|--------|---------------|--------------|
| **Face analysis** | None — all heuristic | OFIQ's own ONNX models (ADNet, BiSeNet, occlusion seg, HeadPose3DDFAV2) |
| **Region targeting** | Border strips, fixed y-bands, whole image | Exact OFIQ regions: BiSeNet masks, EVZ rects, landmark polygons, ROI zones |
| **EyesOpen** | Dark band over y=25-40% | RBF warp of upper eyelid landmarks toward lower |
| **MouthClosed** | Light band over y=55-75% | RBF warp of inner lip landmarks apart |
| **IED/HeadSize** | Downscale+upscale (zero effect on aligned crops) | Pad-and-shrink (geometrically reduces IED) |
| **Crop margins** | 1 random-direction function for all 4 | 4 deterministic single-direction functions |
| **Background** | Outer 15% border strip | BiSeNet class 0, eroded with 4×4 kernel |
| **Illumination** | Global L/R/T/B gradient | Specific ROI zone darkening |
| **NaturalColour** | RGB channel boost, whole image | CIELAB shift in landmark-derived ROI zones |
| **Missing components** | 3 unimplemented | All 27 implemented (+ RadialDistortion) |
| **Context sharing** | N/A | FaceContext computed once, shared across all degradations |
