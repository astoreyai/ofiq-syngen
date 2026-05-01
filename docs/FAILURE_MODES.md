# Failure modes

Per-component documentation of input conditions and edge cases that
cause each degrader to behave unexpectedly. Where a failure has a clean
fallback, the fallback behavior is named. Where it raises, the
exception type is given.

This document is the source of truth for the package's "where it
breaks" claims. Reviewers cite these expected behaviors; users debug
against them.

## Convention

| Severity | Behavior class | Action |
|---|---|---|
| **Hard error** | Function raises | Exception type and message named here |
| **Silent fallback** | Function returns input unchanged or a degraded variant | Documented; tests assert the fallback path |
| **Visual artifact** | Output is not visually convincing but mathematically valid | Documented; OFIQ score still moves predictably |
| **Score deviation** | OFIQ score does not move as expected for this severity | Documented; affects calibration assumptions |

## All components

### Common to every degrader

- **Non-uint8 input**: raises `cv2.error`. Functions assume `uint8` BGR.
- **Non-3-channel input** (grayscale, BGRA): may raise or produce
  unexpected output. Functions assume 3 channels.
- **Empty image** (0 size): raises in OpenCV before reaching syngen
  code. Validate input shape before calling.
- **Severity outside [0.0, 1.0]**: not validated by the function. Most
  degraders extrapolate gracefully (severity > 1 produces stronger
  effect, severity < 0 produces inverse effect or no effect). Use
  severity in `[0, 1]` for documented behavior.
- **`ctx=None` for `requires_context=True` components**: degrader falls
  back to a context-free path that often produces no change. See
  per-component notes below.

### Capture-related (FDIS §7.3)

#### `BackgroundUniformity.scalar`

- **`ctx=None`**: falls back to border-strip noise (outer 15% of image).
  Less precise than the BiSeNet-segmented background but still moves
  the OFIQ score in the right direction.
- **All-uniform input**: noise is added regardless; OFIQ Sobel gradient
  rises predictably.

#### `IlluminationUniformity.scalar`

- **`ctx=None`**: silent fallback to global L/R gradient. The
  perturbation reaches the wrong region (whole face instead of OFIQ's
  ROI zones), so OFIQ score may not respond as expected. Treat as
  "smoke test only" without context.

#### `LuminanceMean.scalar`, `LuminanceVariance.scalar`

- Whole-image fallback when `ctx=None`. Functional but less precise
  than face-mask targeting.
- **Already-dark inputs** (`LuminanceMean`): severity 1 cannot push
  pixels below 0; severity sweep saturates.

#### `UnderExposurePrevention.scalar`, `OverExposurePrevention.scalar`

- Whole-image fallback when `ctx=None`.
- **Severity 1.0 produces black/white image**: this is intentional. OFIQ
  measures the fraction of clipped pixels; full clipping is the maximum
  signal.

#### `DynamicRange.scalar`

- No context required. Reduces histogram entropy uniformly across the
  image. Always works.

#### `Sharpness.scalar` (3 variants: blur, motion blur, gaussian noise)

- **Already-blurry input**: blur variant has diminishing returns;
  motion blur still moves the score; noise reliably reduces sharpness.
- **Very small images** (< 32 px): kernel sizes scale, output may be
  near-uniform. Use medium / large fixtures for meaningful tests.

#### `CompressionArtifacts.scalar`

- **PNG / lossless input**: severity > 0 introduces JPEG artifacts as
  expected.
- **Already-compressed input**: re-compression may produce non-monotonic
  OFIQ score because OFIQ's CNN was not trained on doubly-compressed
  inputs. Document any such case in your validation.

#### `NaturalColour.scalar`

- **`ctx=None`**: silent fallback to whole-image CIELAB shift. Less
  precise than landmark-derived ROI zones but moves OFIQ score in the
  right direction.
- **Already-tinted skin** (sunburn, makeup): perturbation compounds;
  baseline OFIQ score may already be low.

#### `RadialDistortion.scalar`

- **No OFIQ measurement counterpart in v1.1.0**: this is a forward-
  looking component. OFIQ score after distortion is undefined until the
  measurement is added upstream.
- Always works at the pixel level; produces visible barrel distortion.

#### `SingleFacePresent.scalar`

- **Generative**: requires `insightface` to be installed for the
  Poisson-blended second-face insertion. Without it, falls back to a
  copy-paste rectangle insertion. The copy-paste fallback is
  detectable; treat as a smoke test only.
- Score-monotonicity is not guaranteed because the inserted face's
  position is seed-dependent.

### Subject-related (FDIS §7.4)

#### `EyesOpen.scalar`, `MouthClosed.scalar`

- **`ctx=None`**: silent no-op (returns input unchanged). The
  perturbation is a landmark-based RBF warp that requires eye/mouth
  landmark coordinates from FaceContext.
- **Off-frontal pose**: warp produces visible distortion. Baseline
  OFIQ score for `EyesOpen` may already be uncertain at severe yaw.

#### `EyesVisible.scalar`, `MouthOcclusionPrevention.scalar`

- **`ctx=None`**: silent fallback to band-shaped occlusion (less
  precise than EVZ rectangles or mouth polygon).
- **Already-occluded input** (sunglasses, mask): perturbation compounds;
  OFIQ score may saturate.

#### `FaceOcclusionPrevention.scalar`

- Falls back to random rectangle within image when `ctx=None`. Without
  the face mask the rectangle may land outside the face entirely.

#### `InterEyeDistance.scalar`, `HeadSize.scalar`

- Pad-and-shrink operations are deterministic, no context required.
- **Aspect-ratio-preserving**: the function pads to maintain aspect
  ratio. Severely non-square inputs produce padded output with visible
  borders.

#### `ExpressionNeutrality.scalar`

- **`ctx=None`**: silent no-op. RBF warp requires landmark coordinates.
- Generative: when context is available, requires `diffusers` /
  `torch` if using diffusion-based variants.

#### `NoHeadCoverings.scalar`

- **`ctx=None`**: falls back to fabric-textured rectangle pasted at
  upper-image y-coordinates. Less precise than parsing-map-targeted
  forehead overlay.
- **Generative variant** requires the StyleGAN edit pipeline; fallback
  is the texture overlay.

### Geometric / pose (S8.x)

#### `HeadPoseYaw.scalar`, `HeadPosePitch.scalar`

- **Perspective warp, not true 3D rotation**: documented limitation.
  Below severity 0.5 the result looks plausible. At severity > 0.7
  geometric shear becomes visible. For research requiring true 3D pose,
  use a 3DMM-based degrader (DECA, FLAME) outside this package.

#### `HeadPoseRoll.scalar`

- Pure 2D affine rotation. Always works.

#### `LeftwardCropOfTheFaceImage.scalar`, `RightwardCropOfTheFaceImage.scalar`, `MarginAboveOfTheFaceImage.scalar`, `MarginBelowOfTheFaceImage.scalar`

- Deterministic shifts. Always work.
- **Severity 1.0**: face content may shift entirely off-frame. OFIQ
  score is undefined when the face is gone.
- **Image-delta monotonicity**: not guaranteed for translation
  operations; documented as `MONOTONICITY_EXEMPT` in the test suite.

## Common error recovery

If a degrader raises:

1. Confirm input is `uint8` BGR with shape `(H, W, 3)`.
2. Confirm severity is finite and in `[0, 1]`.
3. If `requires_context=True`, build a `FaceContext` (see
   `examples/realtime_capture.py` for a working example) and pass it.
4. Check `tests/test_components_per_component.py` for the parametrized
   row covering this component; it should reproduce on the test
   fixtures.

If a degrader silently returns the input unchanged:

1. Check `requires_context`. Most no-ops are ctx-free fallbacks for
   landmark-dependent components.
2. Add the component to your call site's "needs context" list.

If OFIQ score does not move as expected after degradation:

1. Confirm the degraded image was saved with no further re-compression
   (PNG, not JPEG).
2. Confirm the OFIQ score reading is for the same component (not the
   `unified_quality_score`).
3. Check the parity test vectors (once populated) for the expected
   score envelope.
