# ofiq-syngen Pipeline Specification (v1.0 design target)

Rigorous per-component pipeline specification. Replaces the current
fallback-based architecture with **mandatory FaceContext** — every operator
requires the OFIQ ONNX models (ADNet landmarks, BiSeNet face parsing, face
occlusion segmentation, HeadPose3DDFAV2). When a caller does not supply a
`FaceContext`, the pipeline builds one automatically from the bundled OFIQ
models. There are no ctx-free fallback paths.

Every perturbation is **region-targeted** to exactly the image region OFIQ
measures, with **alpha-feathered boundaries** so the output looks
photographically natural rather than rectangular-overlay-flagged.

## Cross-cutting design rules

1. **Mandatory FaceContext.** Every operator's signature is
   `(img, severity, seed, ctx: FaceContext)`. `ctx` is required; passing
   `None` raises `MissingFaceContextError`. The pipeline auto-builds `ctx`
   if the caller didn't.
2. **OFIQ-region targeting.** Every perturbation operates on the exact
   region OFIQ measures: BiSeNet background mask, cheek ROI zones, EVZ
   rectangles, mouth polygon, face landmarked region, etc. No whole-image
   perturbations except where OFIQ's measurement IS whole-frame (head
   pose, crop/margin geometry, IED, head size, radial distortion).
3. **Feathered masks.** Hard mask boundaries produce visible rectangular
   artifacts. Every region mask is Gaussian-blurred (sigma scales with
   region size) before alpha-blending the perturbation with the original.
4. **Photometric realism.** Perturbations match the scene illumination:
   color-correct, texture-aware, shadow-consistent. Use sRGB → linear →
   perturb → sRGB to avoid gamma artifacts.
5. **Background extension via inpainting.** Operators that change the
   face position in frame (crop/margin, IED, head size) inpaint the
   exposed background region using `cv2.INPAINT_TELEA` (or
   `cv2.xphoto.inpaint` if available) instead of `BORDER_REPLICATE`
   streaks.
6. **High-quality resize.** All resize operations use Lanczos
   (`cv2.INTER_LANCZOS4`) for downscale, bicubic for upscale.
7. **Deterministic.** Every operator is fully deterministic given
   `(img, severity, seed)`. Random color choices, occluder placement,
   etc. are seeded.

## Mask feathering helper (used everywhere)

```python
def feather(mask: np.ndarray, sigma: float = None) -> np.ndarray:
    """Soft alpha mask from binary mask. Sigma defaults to mask_diameter/30."""
    if sigma is None:
        h, w = mask.shape
        sigma = max(2.0, min(h, w) / 60)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma)
    return np.clip(soft, 0.0, 1.0)


def alpha_blend(orig: np.ndarray, perturbed: np.ndarray,
                soft_mask: np.ndarray) -> np.ndarray:
    """Smoothly blend perturbed pixels onto orig using soft alpha."""
    a = soft_mask[..., None]
    return (orig.astype(np.float32) * (1 - a) +
            perturbed.astype(np.float32) * a).clip(0, 255).astype(np.uint8)
```

---

## Capture-related — FDIS §7.3

### §7.3.2 BackgroundUniformity

**OFIQ measures:** mean Sobel gradient magnitude on BiSeNet background mask,
eroded 4×4, after warp-and-crop to 354×292.

**Current:** scattered random rectangles with delta noise.

**Specification (v1):**
1. Get BiSeNet background mask `B = (parsing_map == 0)`, erode 4×4.
2. Generate **Perlin-noise gradient texture** at the image resolution.
3. Add edge-creating elements (random brush strokes, scanlines, gradient
   bands) tuned so mean Sobel magnitude scales linearly with severity.
4. Composite over background only via `B * texture + (1-B) * orig`.
5. Soft-feather `B` boundary so background-to-face transition is smooth.

**Severity calibration:** map severity to target mean-Sobel gradient
`m = severity * 200`. Use binary search over Perlin amplitude to hit `m`.

**Dependencies:** BiSeNet (already in FaceContext).

---

### §7.3.3 IlluminationUniformity

**OFIQ measures:** histogram intersection between left and right cheek ROI
zones (D = Σ min(h_L, h_R)).

**Current:** multiply one ROI by `(1 - severity * 0.8)`.

**Specification (v1):**
1. Get left and right ROI rectangles from `ctx.left_roi`, `ctx.right_roi`.
2. Apply a **soft photometric gradient** centered on one ROI (random which):
   - Build a radial Gaussian alpha mask centered on the chosen cheek.
   - Sigma matches the cheek size so the lighting falls off naturally.
3. Apply the gradient as a multiplicative luminance darkening
   (in YCbCr Y channel only, preserve color), severity-scaled.
4. The gradient extends slightly into the face (cheek region) but
   feathers smoothly toward the unaffected side.

**Severity calibration:** `D_target = (1 - severity * 0.7)`. Apply
darkening factor that produces this drop in measured intersection.

**Dependencies:** ADNet landmarks (in FaceContext).

---

### §7.3.4.2 LuminanceMean

**OFIQ measures:** mean of normalized luminance histogram on face mask.

**Current:** multiply face-mask pixels by `(1 - severity * 0.85)`.

**Specification (v1):**
1. Convert image sRGB → linear RGB (gamma 2.2).
2. Within feathered face mask, multiply by darkening factor in linear space.
3. Convert back to sRGB.
4. **Two-sided perturbation:** randomly choose darkening or brightening
   (per `seed`), since OFIQ band-pass mapping penalizes both.

**Severity calibration:** target mean luminance offset of `±severity * 0.4`
from the original mean.

**Dependencies:** ADNet landmarks for face mask (in FaceContext).

---

### §7.3.4.3 LuminanceVariance

**OFIQ measures:** variance of luminance histogram on face mask.

**Current:** per-channel compression toward mean.

**Specification (v1):**
1. Convert to YCbCr (preserves color).
2. On Y channel, within feathered face mask, compress toward mean:
   `Y' = Y_mean + (Y - Y_mean) * (1 - severity * 0.9)`.
3. Convert back to BGR.

**Severity calibration:** target variance reduction of `severity * 0.85`.

---

### §7.3.5 UnderExposurePrevention

**OFIQ measures:** proportion of pixels in [0, 25] within face ∩ occlusion mask.

**Current:** uniform multiplicative darkening.

**Specification (v1):**
1. Get combined mask `M = ctx.face_mask & ctx.occlusion_mask`, feather.
2. Apply **directional shadow gradient** (random direction per seed):
   - Build a directional gradient alpha (linear from one side to the other)
   - Multiply into M to get the shadow region
3. Within the shadow region, darken sRGB → linear → multiply → sRGB.
4. Mimics natural side-shadow from a window or single-source lighting.

**Severity calibration:** target dark-pixel proportion `v = severity * 0.5`
(since OFIQ sigmoid is centered at 0.92 — need v close to 1.0 for max
degradation).

---

### §7.3.6 OverExposurePrevention

**OFIQ measures:** proportion of pixels in [247, 255] within face mask.

**Current:** multiply by 3.5 then clip.

**Specification (v1):**
1. Get face mask, feather.
2. Apply **highlight bloom**:
   - Pick a random "hot spot" location within face (typically forehead or
     cheekbone for plausibility)
   - Build a Gaussian alpha centered there
   - Add brightness in linear-RGB space
3. Optional: add subtle bloom (low-pass filter the bright region and add
   back) for cinematic look.

**Severity calibration:** target blown-out proportion `v = severity * 0.5`.

---

### §7.3.7 DynamicRange

**OFIQ measures:** Shannon entropy of luminance histogram on face mask.

**Current:** WHOLE-IMAGE compression toward mid-gray (BUG — perturbs
background).

**Specification (v1):**
1. Within feathered face mask only, apply S-curve flattening:
   `Y' = 128 + (Y - 128) * (1 - severity * 0.9)`.
2. Background unchanged.

**Severity calibration:** target entropy reduction of `severity * 6` bits.

---

### §7.3.8 Sharpness — three variants

**OFIQ measures:** RF classifier on Sobel/Laplace/MeanDiff features over
greyscale face crop.

**Current (all 3):** WHOLE-IMAGE perturbation (BUG — also blurs background).

**Specification (v1) — `_blur`:**
1. Within feathered face mask only, apply Gaussian blur with
   `sigma = 0.5 + severity * 10`.
2. Background remains sharp.

**Specification (v1) — `_motion_blur`:**
1. Within feathered face mask only, apply directional kernel
   (random angle per seed) of size `3 + severity * 28`.
2. Background sharp.

**Specification (v1) — `_gaussian_noise`:**
1. Within feathered face mask only, add Gaussian noise
   (sigma = `severity * 80`) to luminance channel.
2. Use luminance-preserving noise (subtract mean, scale, add back).
3. Background unchanged.

---

### §7.3.9 No compression artefacts

**OFIQ measures:** PSNR-CNN on face center crop (248×248).

**Current:** whole-image JPEG re-encode.

**Specification (v1):** **Keep whole-image** — JPEG artifacts are
inherently whole-frame in real scenarios. Document this as a deliberate
choice. Severity → JPEG quality `Q = max(5, int(100 - severity * 95))`.

**Note:** The "face-only JPEG" alternative (extract crop, encode/decode,
paste back) creates non-physical images. Real compression affects the
whole frame.

---

### §7.3.10 NaturalColour

**OFIQ measures:** CIELAB skin-tone-plateau distance on cheek ROI zones.

**Current:** OpenCV LAB shift in ROI rectangles (hard edges, wrong gamma).

**Specification (v1):**
1. Convert sRGB → linear RGB → XYZ → CIELAB (D50 illuminant, matching OFIQ
   §6.13). NOT cv2.cvtColor (which uses 8-bit LAB with simplified math).
2. Within feathered cheek ROI mask, shift a* and b* channels by
   `(direction * severity * 50)` to push outside skin plateau
   `[5,25] × [5,35]`.
3. Convert back through XYZ → linear RGB → sRGB.

**Severity calibration:** target distance from plateau `D = severity * 30`.

---

### Annex D.2.1 RadialDistortion

**Status:** Forward-looking; OFIQ does not measure this in IS:2025.

**Current:** whole-image radial remap. **Correct as-is** — lens distortion
is physically whole-frame.

**Specification (v1):** keep current implementation; whole-frame is
appropriate for this measure.

---

## Subject-related — FDIS §7.4

### §7.4.2 SingleFacePresent

**OFIQ measures:** area ratio of 2nd-largest to largest detected face
(SSD detector).

**Current:** Poisson-blend a face crop into BiSeNet background.

**Specification (v1):**
1. Pick a face from a curated **distractor face library** (license-clean
   thumbnails bundled with package, or synthesized via FaceContext-derived
   self-crops with rotation/flip).
2. Determine target area = `severity * 0.4 * primary_face_area`.
3. Place in BiSeNet background region, away from primary face (via
   distance transform).
4. Color-correct distractor to match scene illumination (mean LAB shift).
5. Poisson blend with feathered alpha.
6. Verify with optional SSD pre-check that the inserted region is
   detectable as a face (otherwise increase contrast/sharpness).

**Dependencies:** BiSeNet (FaceContext), optional SSD verification.

---

### §7.4.3 EyesOpen

**OFIQ measures:** smaller eye palpebral aperture / chin distance,
computed from ADNet landmarks.

**Current:** RBF warp moving upper eyelids toward lower (OK when ctx
exists). No fallback when ctx is missing — operator does nothing.

**Specification (v1):**
1. **Mandatory FaceContext.** No fallback.
2. RBF warp upper eyelid landmarks (61,62,63 / 69,70,71) toward their
   lower counterparts (67,66,65 / 75,74,73) by `severity * gap * 0.9`.
3. **Texture-aware blending**: where the eye was visible, paint a smooth
   eyelid skin gradient (sample skin tone from surrounding pixels) so the
   closure looks natural rather than just a warp.
4. Add subtle eyelash darkening at the new lid line (1-pixel-wide dark
   stroke along the warped upper-lower lid junction).
5. Preserve eyebrow / forehead untouched.

**Severity calibration:** at severity=1.0, palpebral aperture should drop
to <10% of original (ω → ~0.002, well below sigmoid x0=0.02).

---

### §7.4.4 MouthClosed

**OFIQ measures:** max inner-lip pair distance / chin distance.

**Current:** RBF warp inner lip landmarks apart. No fallback if no ctx.

**Specification (v1):**
1. **Mandatory FaceContext.**
2. RBF warp inner lip landmarks (88-91 up, 92-95 down) by
   `severity * t_metric * 0.25`.
3. **Inpaint mouth interior** with a soft dark color (sampled from the
   shadowed inner lip region of the original) to simulate the dark cavity
   that appears when the mouth opens. NO white "teeth" simulation
   (avoids uncanny valley).
4. Subtle warp of cheek/jaw landmarks so the open mouth deforms the
   surrounding face naturally, not just the lips.

**Severity calibration:** at severity=1.0, inner-lip distance should
reach ω > 0.4 (well above sigmoid x0=0.2).

---

### §7.4.5 EyesVisible

**OFIQ measures:** occlusion fraction of EVZ via face-occlusion seg.

**Current:** dark band placed centrally in EVZ rectangle.

**Specification (v1):**
1. **Mandatory FaceContext.**
2. Render **realistic sunglasses** over the EVZ:
   - Build elliptical lens shapes matching EVZ aspect ratio
   - Lens color: dark RGB ~(20, 25, 30) with subtle gradient (top darker)
   - Add specular highlight (small bright Gaussian in upper-left of each lens)
   - Frame: 2-3 px dark line around lens edge
   - Bridge between lenses
3. Severity controls **opacity** (sev=0.3 → 60%, sev=1.0 → 95%) and
   **lens darkness** so progression is visible.
4. Verify with face occlusion segmentation that the rendered sunglasses
   would be classified as occluded.

**Severity calibration:** at severity=1.0, EVZ occlusion fraction should
exceed 0.8 (Q drops to <20).

---

### §7.4.6 MouthOcclusionPrevention

**OFIQ measures:** occlusion fraction of mouth polygon.

**Current:** mouth-color polygon fill.

**Specification (v1):**
1. **Mandatory FaceContext.**
2. Render **realistic surgical mask**:
   - Pleated horizontal fabric texture (procedural noise + light gradient)
   - Mask color: light blue (~200, 220, 230) with subtle variation
   - Conform to face contour: top edge follows nose bridge landmarks,
     bottom edge follows chin contour, sides follow jaw landmarks
   - Soft drop shadow under the mask edge (subtle darkening on jaw)
   - Optional: hint of ear-loop strap (1-px line from mask side toward
     where ear would be)
3. Severity controls **mask coverage**: at sev=0.3, only the mouth covered;
   at sev=1.0, full nose-to-chin coverage.

---

### §7.4.7 FaceOcclusionPrevention

**OFIQ measures:** occlusion fraction of landmarked region.

**Current:** random colored rectangle within face mask bounding box.

**Specification (v1):**
1. **Mandatory FaceContext.**
2. Render **realistic occluder**, randomly chosen from:
   - **Hand**: silhouette PNG (skin-tone matched), placed over lower face
   - **Microphone**: cylindrical shape with specular highlight, lower-mouth area
   - **Hair strand**: dark wisp following a curved Bezier path across face
   - **Hat brim**: arc shadow over upper face
3. Severity controls **occluder size** and **opacity**.
4. Edge-feather everything; cast soft shadow under the occluder for depth.

**Severity calibration:** at severity=1.0, occlusion fraction > 0.6.

---

### §7.4.8 InterEyeDistance

**OFIQ measures:** pixel inter-eye distance in original frame.

**Current:** pad-and-shrink with `BORDER_REPLICATE` (visible streaky
borders).

**Specification (v1):**
1. Lanczos downsample the entire image by `(1 - severity * 0.7)`.
2. Place centered in original-size canvas.
3. Inpaint the exposed border region using `cv2.INPAINT_TELEA` with a
   reasonable radius. This produces plausible background continuation
   instead of streaks.
4. Optional: blur the inpainted border to ~3 px so the stitch line is
   invisible.

**Severity calibration:** scale factor `(1 - severity * 0.7)`,
producing IED reduction proportional to scale.

---

### §7.4.9 HeadSize

**OFIQ measures:** chin-to-eye-mid / image height.

Same mechanism as InterEyeDistance. **Specification:** identical to §7.4.8.

---

### §7.4.10.1 LeftwardCrop

**OFIQ measures:** rightEyeCenter.x / IED.

**Current:** affine shift left with `BORDER_REPLICATE`.

**Specification (v1):**
1. Affine shift image left by `severity * w * 0.4`.
2. Inpaint exposed right border via `cv2.INPAINT_TELEA`.
3. Soft-blend the inpaint stitch line.

### §7.4.10.2 RightwardCrop, §7.4.10.3 MarginAbove, §7.4.10.4 MarginBelow

Same pattern as Leftward — replace `BORDER_REPLICATE` with inpainting.

---

### §7.4.11.2 HeadPoseYaw

**OFIQ measures:** 3DDFA-V2 yaw angle.

**Current:** 2D perspective squeeze (foreshortening proxy).

**Specification (v1):**
1. **3D-aware rotation**: use the 3DDFA-V2 model (already in FaceContext)
   to extract 3D face mesh, rotate by target yaw, re-render.
2. Fall back to perspective warp only if 3DDFA reconstruction fails
   (raise warning).
3. Inpaint background revealed by the rotation.

**Dependencies:** 3DDFA-V2 (in FaceContext) for proper 3D rendering.
Without 3D mesh export, perspective-warp is the best 2D approximation;
keep current implementation but document the limitation.

### §7.4.11.3 HeadPosePitch — same pattern as Yaw.

### §7.4.11.4 HeadPoseRoll

**OFIQ measures:** roll angle.

**Specification (v1):**
1. Affine rotation by `severity * 30 deg` (random sign per seed).
2. Inpaint exposed corners (rotation creates triangular blank regions).

---

### §7.4.12 ExpressionNeutrality

**OFIQ measures:** HSEmotion CNN ensemble.

**Current:** RBF warp on landmarks. **No fallback if ctx is None.**

**Specification (v1):**
1. **Mandatory FaceContext.**
2. Apply **FACS-aware action unit warps** based on chosen expression
   (random per seed):
   - **Smile (AU12)**: zygomaticus major — warp mouth corners up,
     cheeks raised
   - **Surprise (AU2 + AU5 + AU26)**: brow raise, upper lid raise, jaw drop
   - **Frown (AU15 + AU4)**: depressor anguli oris + brow lower
3. Texture-aware: add subtle wrinkle hints at expected locations (crow's
   feet for smile, forehead lines for surprise) via low-amplitude
   structured noise.
4. Severity controls warp magnitude.

**Optional v2:** integrate a small expression model (DECA, EMOCA) for
photorealism.

---

### §7.4.13 NoHeadCoverings

**OFIQ measures:** BiSeNet hat (class 18) + clothing (class 16) pixel
fraction in upper face region.

**Current:** solid colored band on forehead.

**Specification (v1):**
1. **Mandatory FaceContext.**
2. Render **realistic hat** from procedural template:
   - Pick hat type from {beanie, cap, fedora, headscarf} based on seed
   - Conform to head shape using forehead landmark curvature
   - Apply fabric texture (procedural Perlin noise)
   - Cast soft shadow on forehead (darkening gradient under hat brim)
   - Color: muted earth tones (brown, gray, navy) — the colors BiSeNet
     was trained on
3. Severity controls **hat coverage extent** (sev=0.3 → covers top 20%
   of upper face; sev=1.0 → covers full upper region down to brows).
4. Verify with BiSeNet that the rendered hat is classified as class 18.

**Optional v2:** use a small hat overlay library (PNG sprites with
alpha) for higher photorealism.

---

## §7.2 UnifiedQualityScore

**OFIQ measures:** MagFace-derived ‖V‖₂.

**Specification:** this is a composite score; no operator. Cannot be
directly degraded; degrade through the constituent components.

---

## Implementation phases

### Phase 1: face-mask feathering for over-perturbing operators (immediate)

Fix the 4 operators that currently perturb the whole image when they
should be face-only:
- DynamicRange
- Sharpness `_blur`, `_motion_blur`, `_gaussian_noise`

Mechanical edit: wrap each in feathered face mask, blend.

### Phase 2: remove fallback paths (week 1)

Make FaceContext mandatory. Auto-build from OFIQ ONNX models when not
supplied. Raise `MissingFaceContextError` if models unavailable.

Affected operators: every `_*_fallback` function gets removed.

### Phase 3: photorealistic occluders (weeks 2–4)

Replace solid-fill occluders with procedural realistic renderings:
- Sunglasses (EyesVisible)
- Surgical mask (MouthOcclusion)
- Hand/microphone/hat/hair (FaceOcclusion)
- Realistic hat with fabric texture (NoHeadCoverings)

### Phase 4: inpainting-based geometric ops (weeks 5–6)

Replace `BORDER_REPLICATE` with `cv2.INPAINT_TELEA`:
- LeftwardCrop, RightwardCrop, MarginAbove, MarginBelow
- InterEyeDistance, HeadSize
- HeadPoseRoll

### Phase 5: 3D-aware pose (weeks 7–10)

Use 3DDFA-V2 mesh extraction for proper yaw/pitch rendering.

### Phase 6: FACS-aware expression (weeks 11–12)

Replace generic landmark warp with FACS action unit warps for
ExpressionNeutrality. Optional: integrate DECA/EMOCA.

---

## Testing requirements

Every operator MUST have:
1. **Direction test** (`tests/test_degradation_direction.py`): the OFIQ
   scalar measured by binary parity must DECREASE monotonically with
   severity.
2. **Visual smoke test**: severity=1.0 output must look photographically
   plausible (no rectangular artifacts, no hard mask edges, no
   non-physical colors).
3. **Determinism test**: same `(img, severity, seed, ctx)` produces
   bit-identical output.
4. **Region locality test**: pixels OUTSIDE the OFIQ measurement region
   must be unchanged (modulo feathering boundary).

---

## What this replaces

The current architecture has:
- Ctx-required and ctx-free paths (~30 fallback functions)
- Whole-image perturbations that confound background (4 ops)
- Hard-edged solid-fill occluders (5 ops)
- `BORDER_REPLICATE` streaky borders (7 ops)

The v1.0 specification removes all of these.
