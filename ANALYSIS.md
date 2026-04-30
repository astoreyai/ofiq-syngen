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

## Section 6 — Capture-Related Components

### 6.1 BackgroundUniformity

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Sobel gradient magnitude (ksize=-1, CV_32F) on rec.709 luminance image. Mean gradient over background pixels. Sigmoid: h=190, x0=10, w=100. | Structured edge/texture noise patches within the background mask. Random rectangles with intensity proportional to severity. |
| **Analyses / Perturbs** | **Background only**: BiSeNet class 0, eroded with 4x4 kernel. Excludes padding regions via affine transform mask. Cropped 62px sides, 108px bottom, resized to 354x295. | **Background only**: Same BiSeNet class 0 mask from `ctx.parsing_map`, eroded with 4x4 kernel. Noise placed only within mask. |
| **Alignment** | **GOOD**. Same mask, same erosion. Syngen creates edges (Sobel-detectable) within the exact region OFIQ analyzes. The noise patches create sharp gradients that directly increase the mean Sobel magnitude OFIQ computes. |

### 6.2 IlluminationUniformity

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | 256-bin normalized luminance histograms of left and right ROI zones. Element-wise minimum (histogram intersection). Sum of minimums. Score = round(100 × rawScore^0.3). | Multiplicative darkening of one ROI zone (factor 1.0 → 0.2), leaving the other unchanged. |
| **Analyses / Perturbs** | **Left and right eye ROI zones**: zoneSize = int(IED × 0.3). Right ROI: (rightEyeCenter.x − zoneSize, rightEyeCenter.y + eyeMouthDist/2, zoneSize, zoneSize). Left ROI symmetrically from leftEyeCenter. Applied to face-masked luminance image. | **One of the two ROI zones**: Uses `ctx.left_roi` / `ctx.right_roi` computed from ADNet landmarks with identical formula. Darkens one while leaving the other at original intensity. |
| **Alignment** | **EXCELLENT**. Same ROI zones, same landmark-derived computation. Darkening one zone directly reduces the histogram intersection (overlap) that OFIQ computes, since the darkened zone's histogram shifts left while the untouched zone stays in place. |

### 6.3 LuminanceMean

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Normalized 256-bin luminance histogram within face landmark mask. Mean = Σ(histogram[i] × i/255). Score = round(100 × Sigmoid(mean, 0.2, 0.05) × (1 − Sigmoid(mean, 0.8, 0.05))). Double sigmoid penalizes both too dark AND too bright; ideal ≈ 0.5. | Multiplicative darkening (factor 1.0 → 0.15) applied only within face mask region. |
| **Analyses / Perturbs** | **Face landmark mask**: convex hull of 98 ADNet landmarks with optional forehead extension. rec.709 linearized luminance (not cv2 grayscale). | **Face mask**: Same `ctx.face_mask` from ADNet landmark convex hull. Darkening applied with `np.where(mask, darkened, original)`. |
| **Alignment** | **EXCELLENT**. Same face mask. Darkening within the masked region directly reduces the mean luminance that OFIQ computes. Only limitation: syngen only darkens (reduces mean), while OFIQ's double sigmoid also penalizes high mean. |

### 6.3 LuminanceVariance

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Variance of face-region luminance histogram: Σ(histogram[i] × (i/255 − mean)²). Score = round(100 × sin((60v)/(60v+1) × π)). Sine mapping means ideal variance ≈ 0.03. | Per-channel pixel compression toward face-region channel mean: `mean + (pixel − mean) × factor`. Factor 1.0 → 0.1. Applied within face mask. |
| **Analyses / Perturbs** | **Face landmark mask**: same mask as LuminanceMean. Luminance histogram variance within that mask. | **Face mask**: Same `ctx.face_mask`. Compresses toward per-channel mean computed only from face pixels. |
| **Alignment** | **GOOD**. Same face mask. Compressing RGB channels toward their means reduces luminance variance. Minor mismatch: OFIQ computes variance on rec.709 luminance, syngen compresses per-channel in RGB. The effect is directionally correct — pulling RGB channels toward their means necessarily compresses the resulting luminance distribution. |

### 6.4 UnderExposurePrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Proportion of pixels with luminance in [0, 25] within combined face mask (faceMask AND occlusionMask). Sigmoid: h=120, x0=0.92, w=0.05. | Multiplicative darkening (factor 1.0 → 0.15) within combined face + occlusion mask. |
| **Analyses / Perturbs** | **Face mask AND occlusion mask**: bitwise_and of landmark convex hull mask and binary occlusion segmentation mask. Counts dark pixels (lum 0–25) within this combined mask. | **Face mask AND occlusion mask**: Same `cv2.bitwise_and(ctx.face_mask, ctx.occlusion_mask)`. Darkening pushes pixels into the [0,25] range that OFIQ penalizes. |
| **Alignment** | **EXCELLENT**. Same combined mask. Darkening directly increases the proportion of dark pixels OFIQ counts. |

### 6.4 OverExposurePrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Proportion of pixels with luminance in [247, 255] within face mask. Mapped: 1/(ratio + 0.01), clamped to [0,100]. | Multiplicative brightening (factor 1.0 → 3.5) within face mask. |
| **Analyses / Perturbs** | **Face landmark mask**: counts bright pixels (lum 247–255). | **Face mask**: Same `ctx.face_mask`. Brightening pushes pixels into saturation (247–255 range). |
| **Alignment** | **EXCELLENT**. Same mask. Brightening directly increases the proportion of saturated pixels OFIQ counts. |

### 6.5 DynamicRange

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Shannon entropy of 256-bin luminance histogram within face mask. Score = 12.5 × entropy, clamped to [0,100]. | Pixel compression toward mid-gray (128): `128 + (pixel − 128) × factor`. Factor 1.0 → 0.1. Whole image. |
| **Analyses / Perturbs** | **Face mask**: luminance histogram entropy. Higher entropy = wider tonal spread = better. | **Whole image**: compresses all pixel values toward 128, reducing histogram spread. |
| **Alignment** | **EXCELLENT**. Compressing toward mid-gray concentrates the histogram, directly reducing Shannon entropy. Whole-image application is acceptable because the face dominates aligned crops. |

### 6.6 Sharpness

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Random Forest (RTrees) classifier on 30 Laplacian/Sobel/mean-difference features at kernel sizes 1,3,5,7,9. Features: mean and stddev of each filter response. Sigmoid: h=1, a=−14, s=115, x0=−20, w=15. | Three variants: (1) Gaussian blur σ 0.5→10.5, (2) Motion blur kernel 3→31px, (3) Gaussian noise σ 0→80. |
| **Analyses / Perturbs** | **Face region**: greyscale face crop. Features computed from Laplacian, Sobel, and blur-difference kernels. Random Forest trained on these features. | **Whole image**: blur reduces Laplacian/Sobel responses; noise reduces SNR and blurs edges. |
| **Alignment** | **EXCELLENT**. Gaussian blur directly attenuates the Laplacian and Sobel responses the Random Forest uses as features. Motion blur attenuates directional gradients. Noise degrades all edge-based features. Three attack vectors provide good coverage. |

### 6.7 CompressionArtifacts

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | ONNX neural network (`ssim_248_model.onnx`) on center crop (248×248 from aligned face, excludes 184px border). Input normalized with ImageNet-style mean/std. Sigmoid: h=1, a=−0.0278, s=103, x0=0.3308, w=0.092. | JPEG re-encoding at quality 100 → 5. |
| **Analyses / Perturbs** | **Center crop of aligned face**: 248×248 center region. Neural network trained to detect JPEG blocking/ringing artifacts. | **Whole image**: JPEG encode/decode at degraded quality produces exactly the blocking and ringing artifacts the neural network was trained to detect. |
| **Alignment** | **EXCELLENT**. JPEG re-encoding produces the exact artifact type OFIQ's neural network was designed to detect. |

### 6.8 NaturalColour

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Custom sRGB → CIELAB conversion (D50 illuminant, not cv2.cvtColor). Mean a\* and b\* computed from concatenated left + right ROI regions (same zones as IlluminationUniformity). Score = sqrt(max(max(0, 5−a\*), a\*−25)² + max(max(0, 5−b\*), b\*−35)²). Natural skin: a\* ∈ [5,25], b\* ∈ [5,35]. Sigmoid: h=200, x0=0, w=10. | CIELAB a\*/b\* channel shift (±60 per channel) within the two ROI zones. Pushes chromaticity outside [5,25]/[5,35]. |
| **Analyses / Perturbs** | **Left and right ROI zones** (same as IlluminationUniformity): landmark-derived zoneSize = int(IED × 0.3) squares positioned below each eye center. ROIs concatenated, converted to CIELAB. | **Left and right ROI zones**: Same `ctx.left_roi` / `ctx.right_roi`. Converts to LAB, shifts a\*/b\* channels, converts back. Applied within face-masked ROI. |
| **Alignment** | **GOOD**. Same ROI zones. LAB shift directly pushes mean a\*/b\* outside the natural range OFIQ measures. Minor mismatch: syngen uses cv2 LAB conversion, OFIQ uses custom sRGB→CIELAB with D50 illuminant and different matrix coefficients. The direction of effect is correct, magnitude may differ slightly. |

### 6.9 RadialDistortion

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Not measured in the current OFIQ deployment for this project (no entry in `OFIQ_COMPONENTS`). ISO 29794-5 Section 6.9 specifies deviation from rectilinear projection. | Barrel distortion via radial coefficient k (0 → 0.5). Uses cv2.remap with `scale = 1 + k × r²`. |
| **Analyses / Perturbs** | — | **Whole image**: radially symmetric distortion centered on image center. |
| **Alignment** | **N/A**. Forward-looking implementation for ISO 29794-5 S6.9. Not currently scored by OFIQ in this project. |

---

## Section 7 — Subject-Related Components

### 7.1 SingleFacePresent

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Face detection (SSD 300×300). Count detected faces. If >1: f = area_2nd / area_1st. Score = round(100 × (1 − f)). If 1 face: score = 100. | Poisson-blend a face crop from the source image (flipped for variety) into a background region identified by BiSeNet class 0. Scale = severity × 0.4 × image area. |
| **Analyses / Perturbs** | **Whole image**: face detection bounding boxes. Measures area ratio of second-largest to largest face. | **Background region**: face patch inserted into BiSeNet-segmented background. Uses `cv2.seamlessClone` for natural blending. |
| **Alignment** | **MODERATE**. The inserted face crop must trigger SSD face detection to affect the score. Simple Poisson-blended self-crops may or may not be detected as faces by the SSD model. Effectiveness depends on the face detector's sensitivity to the inserted patch's appearance. |

### 7.2 EyesOpen

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | For each eye: max distance among landmark pairs. LEFT_EYE pairs: (61,67), (62,66), (63,65). RIGHT_EYE pairs: (69,75), (70,74), (71,73). rawScore = min(leftMax, rightMax) / tmetric. Sigmoid: x0=0.02, w=0.01. | RBF (thin-plate-spline) warp moving upper eyelid landmarks toward lower counterparts. Displacement = severity × gap between paired landmarks × 0.9. |
| **Analyses / Perturbs** | **Eye landmark geometry**: measures distances between upper and lower eyelid landmarks (3 pairs per eye). Ratio to tmetric (eye-to-chin distance). | **Eye landmark geometry**: warps upper eyelid landmarks (61,62,63 / 69,70,71) toward their lower counterparts (67,66,65 / 75,74,73). The image deformation changes what the landmark detector sees, reducing the measured pair distances. |
| **Alignment** | **EXCELLENT**. Directly perturbs the geometric quantity OFIQ measures. The warp simulates eyelid closure, which is what reduced eye-opening distances represent. The landmark detector re-applied to the warped image should measure smaller pair distances. |

### 7.3 MouthClosed

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Max distance among MOUTH_INNER pairs: (89,95), (90,94), (91,93). rawScore = maxMouthOpening / tmetric. Sigmoid (inverse): x0=0.2, w=0.06. Higher rawScore = more open = lower quality. | RBF warp moving inner lip landmarks apart. Upper inner (88,89,90,91) move UP, lower inner (92,93,94,95) move DOWN. Displacement = severity × tmetric × 0.25. Outer mouth landmarks (77,78,80,81,83,84,86,87) get 30% proportional displacement. |
| **Analyses / Perturbs** | **Mouth inner landmark geometry**: measures distances between upper and lower inner lip landmarks (3 pairs). | **Mouth inner landmark geometry**: warps inner lip landmarks apart, increasing the pair distances OFIQ measures. |
| **Alignment** | **EXCELLENT**. Directly perturbs the geometric quantity OFIQ measures. The warp simulates mouth opening by separating the exact landmark pairs. Displacement scaled to tmetric ensures the rawScore exceeds the sigmoid midpoint (x0=0.2). |

### 7.4 EyesVisible

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Eye Visibility Zone (EVZ): V = floor(IED/20). Each eye's bounding rect expanded by V pixels in all directions. Occlusion proportion = sum(EVZ × (1 − occlusionMask)) / sum(EVZ). Score = round(100 × (1 − rawScore)). | Dark-colored occlusion band placed within EVZ rectangles. Height scales with severity × 0.8 of EVZ height. |
| **Analyses / Perturbs** | **Eye Visibility Zones**: expanded bounding rectangles around LEFT_EYE [60-67] and RIGHT_EYE [68-75] landmarks. Expansion = floor(IED/20) pixels. Uses face occlusion segmentation model to detect what's occluded within the zones. | **Same EVZ rectangles**: computed from `ctx.left_evz` / `ctx.right_evz` using identical V = floor(IED/20) formula. Occlusion placed centrally within each EVZ. |
| **Alignment** | **GOOD**. Same EVZ regions. The dark band simulates sunglasses/hair. OFIQ's occlusion segmentation model must classify the synthetic occlusion as "not visible face" for the score to drop. Solid dark rectangles may or may not trigger the learned segmentation model identically to real-world occluders. |

### 7.5 MouthOcclusionPrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Convex polygon from mouth outer landmarks [76-87] (12 points). Occlusion proportion = sum(mouthMask × (1 − occlusionMask)) / sum(mouthMask). Score = round(100 × (1 − rawScore)). | Surgical-mask-colored fill within the convex polygon of landmarks [76-87]. Coverage scales with severity via erosion of the mask. |
| **Analyses / Perturbs** | **Mouth polygon**: convex hull of 12 outer mouth landmarks. Uses occlusion segmentation to detect what's occluded within. | **Same mouth polygon**: `ctx.landmarks_98[76:88]` convex hull. Light blue/white fill simulating a surgical mask. |
| **Alignment** | **GOOD**. Same polygon. The mask-colored fill simulates a surgical mask, which is a common real-world mouth occluder. Same dependency on the occlusion segmentation model's classification of the synthetic fill. |

### 7.6 FaceOcclusionPrevention

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Face landmark mask (all 98 landmarks, convex hull). Occlusion proportion = countNonZero(faceMask × (1 − occlusionMask)) / countNonZero(faceMask). Score = round(100 × (1 − rawScore)). | Random-colored rectangular occlusion placed within the face mask bounding box. Only pixels within `ctx.face_mask` are overwritten. Area = severity × 60% of face region. |
| **Analyses / Perturbs** | **Face landmark mask**: convex hull of all 98 landmarks (with optional forehead extension). Uses occlusion segmentation to detect occlusion proportion. | **Same face mask**: rectangle placed within `ctx.face_mask` bounding box. Color randomized. Only face-mask pixels modified. |
| **Alignment** | **GOOD**. Same face region. Occlusion placed within the face mask, not at random image positions. Same segmentation model dependency. |

### 7.7 InterEyeDistance

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Euclidean distance between eye centers (midpoints of [60,64] and [68,72]). Yaw-corrected: IED = distance / cos(yaw × π/180). Sigmoid: h=100, x0=70, w=20. Quality drops below ~70px IED. | Shrink image and embed centered in padded canvas with border replication. Scale factor = 1.0 − severity × 0.7. Makes face smaller in frame, reducing absolute pixel distance between eye centers. |
| **Analyses / Perturbs** | **Eye center geometry**: pixel distance between two points, corrected for head yaw angle. | **Whole image (geometric)**: shrinks the image content, making all facial landmarks closer together. On re-detection, the eye centers will be closer in pixel coordinates. |
| **Alignment** | **GOOD**. Pad-and-shrink correctly reduces the pixel-domain IED that OFIQ measures. At max severity (scale 0.3), IED drops to ~30% of original, well below the sigmoid midpoint. Previous implementation (`_downscale` + upscale to same size) had zero effect on aligned crops — this is a fundamental fix. |

### 7.8 HeadSize

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | T = tmetric(landmarks) (eye-midpoint to chin distance). rawScore = T / imageHeight. convertedScore = \|rawScore − 0.45\|. Sigmoid: h=200, x0=0, w=0.05. Optimal head size ≈ 45% of image height. | Same pad-and-shrink as InterEyeDistance. Shrinks face in frame, reducing t/imageHeight ratio below 0.45 optimum. |
| **Analyses / Perturbs** | **Face-to-frame ratio**: geometric ratio of facial landmark distances to image dimensions. | **Same mechanism as IED**: shrink reduces both tmetric and IED proportionally. |
| **Alignment** | **GOOD**. Same analysis as IED. Both measure geometric face-size ratios that pad-and-shrink correctly reduces. Note: OFIQ uses `abs(ratio − 0.45)`, penalizing both too small AND too large. Syngen only makes faces smaller. |

---

## Section 8 — Geometric/Pose Components

### 8.1 HeadPoseYaw

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | HeadPose3DDFAV2 ONNX model (mb1_120x120.onnx). Crop around face bbox (center ± 0.44h/0.51h), resize to 120×120, normalize (pixel−127.5)/128. 7-parameter output → rotation matrix → Euler angles. Score = round(100 × max(0, cos(yaw × π/180))²). | Perspective warp squeezing one side (0–50%). Simulates horizontal foreshortening. |
| **Analyses / Perturbs** | **Face region (3D model)**: learned 3D pose estimator operating on face crop. Outputs yaw angle in degrees. | **Whole image**: perspective transformation creating 2D foreshortening that approximates yaw rotation. |
| **Alignment** | **MODERATE**. Perspective warp creates foreshortening cues that may trigger the 3D pose model's yaw detection, but doesn't produce the full appearance changes of real yaw rotation (ear visibility, jaw contour changes, nostril asymmetry). Calibration needed: verify measured angle vs. applied severity. |

### 8.2 HeadPosePitch

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Same HeadPose3DDFAV2 model. Score = round(100 × max(0, cos(pitch × π/180))²). | Perspective warp squeezing top or bottom (0–40%). |
| **Analyses / Perturbs** | **Face region (3D model)**: pitch angle from same Euler decomposition. | **Whole image**: vertical perspective transformation. |
| **Alignment** | **MODERATE**. Same limitations as yaw. Perspective warp doesn't produce real pitch cues (forehead/chin visibility, nostril exposure). |

### 8.3 HeadPoseRoll

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Same HeadPose3DDFAV2 model. Score = round(100 × max(0, cos(roll × π/180))²). | In-plane affine rotation ±30°. `cv2.getRotationMatrix2D(center, angle, 1.0)`. |
| **Analyses / Perturbs** | **Face region (3D model)**: roll angle from Euler decomposition. | **Whole image**: 2D rotation. |
| **Alignment** | **EXCELLENT**. In-plane rotation IS roll. The 3D pose estimator correctly detects rotation angle from rotated images. Perfect geometric match. |

### 8.4 LeftwardCropOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | rawScore = rightEyeCenter.x / interEyeDistance. Sigmoid: h=100, x0=0.9, w=0.1. Measures how much margin exists to the left of the face. | Translate image RIGHT by severity × 40% of width. `shift_x = +int(severity × w × 0.4)`. Face moves left, left margin shrinks. |
| **Analyses / Perturbs** | **Left margin**: ratio of right eye center's x-position to IED. | **Horizontal translation only**: rightward shift with border replication. Deterministic direction (not random). |
| **Alignment** | **EXCELLENT**. Rightward image shift moves all landmarks right, increasing rightEyeCenter.x. But since the face is now further right, the LEFT margin of the frame is filled with replicated border content. OFIQ re-detecting the face on the shifted image will find reduced left margin. Previous implementation used random-direction shifts shared across all 4 crop components — this is a targeted fix. |

### 8.5 RightwardCropOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | rawScore = (imageWidth − leftEyeCenter.x) / IED. Sigmoid: h=100, x0=0.9, w=0.1. | Translate image LEFT by severity × 40% of width. `shift_x = −int(severity × w × 0.4)`. |
| **Alignment** | **EXCELLENT**. Mirror of LeftwardCrop. Leftward shift reduces right margin. |

### 8.6 MarginAboveOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | rawScore = eyeMidPoint.y / tmetric. Sigmoid: h=100, x0=1.4, w=0.1. Measures top margin. | Translate image DOWN by severity × 40% of height. `shift_y = +int(severity × h × 0.4)`. |
| **Alignment** | **EXCELLENT**. Downward shift moves face down, reducing top margin. |

### 8.7 MarginBelowOfTheFaceImage

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | rawScore = (imageHeight − eyeMidPoint.y) / tmetric. Sigmoid: h=100, x0=1.8, w=0.1. | Translate image UP by severity × 40% of height. `shift_y = −int(severity × h × 0.4)`. |
| **Alignment** | **EXCELLENT**. Upward shift reduces bottom margin. |

### 8.8 ExpressionNeutrality

| | OFIQ | Syngen |
|---|---|---|
| **Algorithm** | Ensemble of 2 CNNs: EfficientNet-B0 (224×224 → 1280-d) + EfficientNet-B2 (260×260 → 1408-d). Features concatenated (2688-d). AdaBoost classifier (gzip-compressed Boost model). Sigmoid: x0=−5000, w=5000. Input: face crop from aligned image [148:488, 144:472], BGR→RGB, (pixel/255 − mean)/std. | Landmark RBF warp simulating expressions: smile (mouth corners up, cheeks raised), surprise (eyebrows up, mouth open, eyes wide), frown (mouth corners down, brows furrowed). Displacement = severity × tmetric × 0.15. |
| **Analyses / Perturbs** | **Face crop (CNN features)**: learned expression features from two EfficientNet models. HSEmotion-based. | **Face landmark geometry**: warps landmarks to create non-neutral facial configurations. The warped image changes what the CNNs see. |
| **Alignment** | **MODERATE**. Landmark warping produces visible facial geometry changes, but OFIQ's CNNs were trained on real expression images. Whether geometric warping produces features that the AdaBoost classifier scores as non-neutral depends on how texture-vs-geometry sensitive the HSEmotion models are. |

### 8.9 NoHeadCoverings

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
