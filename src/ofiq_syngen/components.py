"""ISO/IEC 29794-5 Component-Aligned Degradation Functions.

Each degradation targets a specific OFIQ quality component by perturbing
exactly what OFIQ measures, using the same regions and analysis pipeline.
Severity is normalized to [0, 1] where 0 = no degradation, 1 = maximum.
All functions are deterministic given a seed.

Functions that require face analysis receive a FaceContext parameter
(landmarks, parsing map, occlusion mask, head pose, derived metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

from ofiq_syngen.landmark_utils import (
    BISENET_BACKGROUND,
    BISENET_HAIR,
    LEFT_EYE,
    PAIRS_LEFT_EYE,
    PAIRS_RIGHT_EYE,
    RIGHT_EYE,
)
from ofiq_syngen.standards import STANDARDS_REFS, StandardRefs

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


@dataclass
class ComponentDegradation:
    """Registration entry for a degradation function."""

    ofiq_component: str
    function: Callable
    description: str
    severity_range: str
    requires_context: bool = False
    standard_refs: StandardRefs | None = None


# =========================================================================
# Section 6 -- Capture-Related Components
# =========================================================================

def _background_clutter_segmented(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Add structured non-uniformity to the segmented background [§7.3.2].

    OFIQ measures the L1 norm of Sobel gradients on the BiSeNet
    background (class 0), eroded with a 4x4 kernel. We modulate the
    background luminance with a multi-octave noise field that produces
    strong, naturally-shaped gradients instead of confetti-like patches.

    Octaves: 8x8 (low-freq lighting gradient), 32x32 (surface texture),
    128x128 (fine detail). Cubic upsample yields smooth-but-edged
    transitions that read as wall texture, fabric weave, or lighting
    falloff rather than synthetic noise.
    """
    if ctx is None or ctx.parsing_map is None:
        return _background_clutter_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]

    bg_small = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
    bg_mask = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_NEAREST)
    bg_mask = cv2.erode(bg_mask, np.ones((4, 4), np.uint8))
    if int(bg_mask.sum()) == 0:
        return img

    field = np.zeros((h, w), dtype=np.float32)
    for tile, weight in [(8, 0.6), (32, 0.3), (128, 0.1)]:
        n = rng.randn(tile, tile).astype(np.float32)
        field += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * weight
    field /= float(np.abs(field).max() + 1e-8)

    amplitude = severity * 90.0
    soft = _feather_mask(bg_mask, sigma=2.0)[..., None]
    delta = (field * amplitude)[..., None].astype(np.float32)

    out = img.astype(np.float32) + delta * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def _background_clutter_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: border-based background noise when no context available."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    out = img.copy()
    border = int(min(h, w) * 0.15)
    noise = rng.randint(0, int(severity * 200 + 1), (h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    out[mask] = np.clip(
        out[mask].astype(np.int16) + noise[mask].astype(np.int16), 0, 255
    ).astype(np.uint8)
    return out


def _uneven_illumination_roi(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Side lighting: smooth left-right luminance gradient on the face [§7.3.3].

    OFIQ measures histogram intersection between left and right ROI zones
    (small cheek patches near each eye). The histogram intersection drops
    when the two sides have different luminance distributions, which is
    what real side-lighting (a window from one direction, off-camera flash,
    cast shadow) produces. We render that look directly: a smooth gradient
    across the face mask darkening one side.
    """
    if ctx is None:
        return _uneven_illumination_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    # Prefer IP2P (photoreal side lighting with shadows that follow face
    # geometry) when available.
    import os
    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_side_lighting_ip2p,
            )
            if is_sd_available():
                return render_side_lighting_ip2p(img, ctx, severity, seed)
        except Exception:
            pass

    rng = np.random.RandomState(seed)
    direction = rng.choice([-1, 1])  # which side gets the shadow

    h, w = img.shape[:2]
    img_f = img.astype(np.float32)

    # Build a horizontal gradient that goes from full brightness on the
    # lit side to (1 - severity * 0.7) on the shadow side. Power 1.5 makes
    # the falloff slightly more concentrated on the shadow side, which
    # looks more like a real off-axis light source.
    xs = np.linspace(0, 1, w, dtype=np.float32)
    if direction > 0:
        ramp = xs  # 0 on left, 1 on right -> shadow on left
    else:
        ramp = 1.0 - xs  # shadow on right
    shadow_strength = severity * 0.7
    gradient = 1.0 - shadow_strength * (1.0 - ramp) ** 1.5  # (W,)
    gradient_2d = np.broadcast_to(gradient[None, :, None], (h, w, 3))

    perturbed = img_f * gradient_2d

    if ctx.face_mask is not None:
        clean_mask = _hull_minus_hair_mask(ctx, h, w)
        # Wide feather so the gradient blends invisibly into hair / neck
        sigma = max(8.0, min(h, w) / 12)
        soft = _feather_mask(clean_mask, sigma=sigma)
        return _alpha_blend(img, np.clip(perturbed, 0, 255).astype(np.uint8), soft)
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def _uneven_illumination_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: gradient lighting."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    direction = rng.choice(["left", "right"])
    gradient = np.linspace(1.0 - severity * 0.8, 1.0, w)
    if direction == "right":
        gradient = gradient[::-1]
    mask = np.tile(gradient, (h, 1))
    mask = np.stack([mask] * 3, axis=-1)
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def _darken_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Whole-image gamma darkening to lower face mean luminance [§7.3.4].

    OFIQ measures luminance histogram within the face landmark mask.
    Face-only darkening looked unnatural (bright hair against dim face,
    like a stage spotlight). Uniform whole-image gamma is honest about
    being an underexposure perturbation: face pixels darken (degrading
    the LuminanceMean scalar correctly) AND hair / neck / clothing /
    background darken with them, matching real underexposed photographs.
    """
    if severity < 0.01:
        return img
    # Gamma in [1.0 .. 3.5] across severity. Gamma > 1 photographically
    # darkens midtones while preserving extremes. Cap at 3.5 so the
    # whole image stays readable at sev=1.0 (no full black silhouette).
    gamma = 1.0 + severity * 2.5
    img_f = img.astype(np.float32) / 255.0
    darkened = np.power(img_f, gamma) * 255.0
    return np.clip(darkened, 0, 255).astype(np.uint8)


def _darken_face_with_occlusion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Per-image autoscaled gamma underexposure for §7.3.5.

    OFIQ measures the proportion of face pixels with Y < 10 inside the
    face mask. A fixed gamma curve (e.g., 1.0..3.5) only crosses that
    threshold on already-dark sources -- bright skin survives gamma=3.5
    with face Y mean ~110 and a near-zero <10 proportion, so the OFIQ
    scalar stays at 100. We instead autoscale gamma per image so that
    at sev=1.0 the face Y mean lands at ~5 (well below 10), regardless
    of source brightness.

    Solve mean(face_Y / 255 ** gamma) = 5/255 for gamma. With y = mean
    face Y as a float in [0, 1], gamma_target = log(5/255) / log(y).
    Severity interpolates linearly between gamma 1.0 (no change) and
    gamma_target. Clamped to [1.0, 8.0] so image stays decodable.
    Whole-image gamma is preserved (face-only darkening looked spotlit;
    real underexposed photographs darken everything together).
    """
    if severity < 0.01:
        return img

    # OFIQ UnderExposure raw = proportion of (face_landmark_region AND
    # occlusion_mask) pixels with luminance in [0, 25] (inclusive),
    # mapped through sigmoid x0=0.92, w=0.05 to the [0, 100] scalar.
    # The scalar starts moving below 100 only once ~90% of OFIQ's mask
    # lands in [0, 25].
    #
    # Per-image autoscale: solve gamma so face Y mean lands at target=5.
    # For face_y_mean already < target, no darkening is needed. Cap
    # gamma at 10 because beyond that OFIQ's face detector starts
    # failing alignment (returning scalar=-1 instead of a valid score),
    # which is worse than a stuck-at-100 scalar -- the latter is honest
    # about "the OFIQ measure cannot judge this face as underexposed".
    if ctx is not None and ctx.face_mask is not None and (ctx.face_mask > 0).any():
        m = ctx.face_mask > 0
        lum = (
            0.0722 * img[..., 0]
            + 0.7152 * img[..., 1]
            + 0.2126 * img[..., 2]
        )
        face_y_mean = float(lum[m].mean()) / 255.0
    else:
        face_y_mean = float(img.mean()) / 255.0

    target_y = 5.0 / 255.0
    if face_y_mean <= target_y:
        gamma_target = 1.0
    elif face_y_mean >= 0.999:
        gamma_target = 10.0
    else:
        gamma_target = float(np.log(target_y) / np.log(face_y_mean))
    # Cap at 10: beyond ~12, OFIQ face alignment fails on the darkened
    # image and returns a -1 sentinel rather than a valid scalar.
    gamma_target = max(1.0, min(10.0, gamma_target))
    gamma = 1.0 + severity * (gamma_target - 1.0)

    img_f = img.astype(np.float32) / 255.0
    darkened = np.power(img_f, gamma) * 255.0
    return np.clip(darkened, 0, 255).astype(np.uint8)


def _brighten_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Whole-image gamma brightening to push face toward overexposure [§7.3.6].

    OFIQ OverExposurePrevention measures the proportion of face pixels
    pushed into the high-luminance (>247) range. Face-only brightening
    looked unnatural (blown-out face spotlit against normal hair).
    Uniform whole-image gamma matches real overexposed photographs:
    face gets blown out (degrading the OFIQ scalar correctly) AND
    hair / neck / clothing brighten with it.

    v0.5.1: pre-clip to [4, 245] before gamma so JPEG noise in the
    near-saturated extremes doesn't get amplified into chromatic
    rainbow stippling (visible on video screenshots / saturated
    sources at sev=1.0).
    """
    if severity < 0.01:
        return img
    # Gamma in [1.0 .. 0.25]. Gamma < 1 photographically brightens
    # midtones while preserving extremes. Cap at 0.25 so the image
    # stays readable at sev=1.0 (no full-white blowout).
    gamma = 1.0 - severity * 0.75
    # Pre-clip the input to avoid amplifying JPEG noise in pixels
    # already at saturation; remap [4, 245] -> [0, 1] so gamma acts
    # on the photographic range, then push back to [0, 255].
    img_clip = np.clip(img.astype(np.float32), 4.0, 245.0)
    img_f = (img_clip - 4.0) / 241.0
    brightened = np.power(img_f, gamma) * 255.0
    return np.clip(brightened, 0, 255).astype(np.uint8)


def _reduce_luminance_variance_face(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Bidirectional face luminance variance perturbation [§7.3.4].

    OFIQ scalar = round(100 * sin((60v) / (60v+1) * pi)) where v is the
    face Y variance (normalized [0, 1]). The scalar peaks at v ~= 1/60
    (~0.0167) and DROPS BOTH WAYS -- raising or lowering variance from
    that optimum degrades the scalar. So a one-directional "compress
    variance" operator improves the OFIQ scalar on already-high-variance
    sources (which is most natural face imagery, var ~ 0.05-0.10) and
    only degrades the rare low-variance source.

    Direction is chosen per image: if face Y variance > optimum, compress
    toward face mean (existing behavior); if below, expand by injecting
    deterministic per-pixel offsets to push variance away from optimum.
    Both branches honor the hull-minus-hair feather blend so the
    boundary stays invisible.
    """
    if severity < 0.01:
        return img
    img_f = img.astype(np.float32)
    h, w = img.shape[:2]
    optimum = 1.0 / 60.0  # OFIQ peak

    # Probe face variance to decide direction
    if ctx is not None and ctx.face_mask is not None:
        m = ctx.face_mask > 0
    else:
        m = None
    if m is not None and m.any():
        face_lum = (
            img_f[..., 0] * 0.0722
            + img_f[..., 1] * 0.7152
            + img_f[..., 2] * 0.2126
        ) / 255.0
        var = float(face_lum[m].var())
    else:
        var = 0.05  # plausible photographic default

    s_eff = float(np.sqrt(np.clip(severity, 0.0, 1.0)))

    if var >= optimum:
        # Source variance is ABOVE the OFIQ optimum (0.0167) — most
        # natural face imagery (typical var ~0.05-0.1). EXPAND variance
        # further by anti-mean scaling: push each pixel away from the
        # face mean by factor 1+s_eff*N. Additive Gaussian noise was
        # tried first but uint8 clipping at 0/255 kills the variance
        # gain on bright/dark images. Anti-mean scaling tolerates
        # clipping because saturated extremes ARE high-variance.
        if m is not None and ctx is not None and ctx.face_mask is not None:
            clean_mask = _hull_minus_hair_mask(ctx, h, w)
            mask = clean_mask > 0
            face_pixels = img_f[mask].reshape(-1, 3)
            if face_pixels.size == 0:
                return img
            face_mean = face_pixels.mean(axis=0)
            factor = 1.0 + s_eff * 4.0  # 1..5x expansion
            expanded = face_mean + (img_f - face_mean) * factor
            sigma = max(8.0, min(h, w) / 12)
            return _alpha_blend(
                img,
                np.clip(expanded, 0, 255).astype(np.uint8),
                _feather_mask(clean_mask, sigma=sigma),
            )
        # No mask: expand around 128
        factor = 1.0 + s_eff * 4.0
        expanded = 128.0 + (img_f - 128.0) * factor
        return np.clip(expanded, 0, 255).astype(np.uint8)

    # Source variance is BELOW the OFIQ optimum (rare for natural faces
    # — would require flat-lit / overexposed input). COMPRESS toward
    # the per-channel face mean to crush variance even lower.
    factor = 1.0 - s_eff * 0.98
    sat_factor = 1.0 - s_eff * 0.55
    if m is not None and ctx is not None and ctx.face_mask is not None:
        clean_mask = _hull_minus_hair_mask(ctx, h, w)
        mask = clean_mask > 0
        face_pixels = img_f[mask].reshape(-1, 3)
        if face_pixels.size == 0:
            return img
        face_mean = face_pixels.mean(axis=0)
        compressed = face_mean + (img_f - face_mean) * factor
        luma = (compressed[..., 0] * 0.0722 + compressed[..., 1] * 0.7152
                + compressed[..., 2] * 0.2126)
        gray3 = np.stack([luma, luma, luma], axis=-1)
        perturbed = gray3 + (compressed - gray3) * sat_factor
        sigma = max(8.0, min(h, w) / 12)
        return _alpha_blend(
            img,
            np.clip(perturbed, 0, 255).astype(np.uint8),
            _feather_mask(clean_mask, sigma=sigma),
        )
    perturbed = 128.0 + (img_f - 128.0) * factor
    return np.clip(perturbed, 0, 255).astype(np.uint8)


def _feather_mask(
    mask: np.ndarray, sigma: float | None = None, inward_only: bool = True,
) -> np.ndarray:
    """Convert binary uint8 mask to a soft alpha (0..1 float32) via Gaussian.

    Sigma defaults to mask_diameter / 60 (visually invisible mask edge).

    With ``inward_only=True`` (default), the binary mask is eroded by
    ~sigma before blurring so the soft edge sits INSIDE the original
    boundary. This prevents the perturbation from bleeding past the
    OFIQ measurement region (e.g. face_mask perturbations leaking onto
    neck or dress). Pass ``inward_only=False`` for symmetric edge
    transitions (rare).
    """
    h, w = mask.shape[:2]
    if sigma is None:
        sigma = max(2.0, min(h, w) / 60)
    if inward_only:
        erode_px = max(1, int(round(sigma * 2)))
        eroded = cv2.erode(mask, np.ones((erode_px, erode_px), np.uint8))
        src = eroded.astype(np.float32)
    else:
        src = mask.astype(np.float32)
    soft = cv2.GaussianBlur(src, (0, 0), sigma)
    return np.clip(soft, 0.0, 1.0)


def _alpha_blend(
    orig: np.ndarray, perturbed: np.ndarray, soft_mask: np.ndarray,
) -> np.ndarray:
    """Blend perturbed pixels onto orig using the soft alpha mask."""
    a = soft_mask[..., None] if soft_mask.ndim == 2 else soft_mask
    blended = (
        orig.astype(np.float32) * (1.0 - a) +
        perturbed.astype(np.float32) * a
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def _sample_backdrop_color(
    img: np.ndarray, ctx: FaceContext | None,
) -> tuple[int, int, int]:
    """Sample a uniform backdrop color from the source image.

    Used by Category C operators (HeadSize, InterEyeDistance, the four
    crop/margin operators) to fill regions emptied by reframing the face.

    Strategy:
        1. If ctx has a BiSeNet parsing map, average the source pixels
           classified as background (class 0).
        2. Otherwise, average a 5-pixel-wide border ring around the
           image edge (likely background in a portrait composition).

    Returns (B, G, R) tuple of uint8.
    """
    h, w = img.shape[:2]
    if ctx is not None and ctx.parsing_map is not None:
        bg_small = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
        bg_mask = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_NEAREST)
        if bg_mask.sum() > 100:
            bg_pixels = img[bg_mask > 0]
            mean = bg_pixels.mean(axis=0)
            return tuple(int(v) for v in mean)
    # Fallback: average a 5-pixel ring around the image edge
    edge = np.concatenate([
        img[:5].reshape(-1, 3), img[-5:].reshape(-1, 3),
        img[:, :5].reshape(-1, 3), img[:, -5:].reshape(-1, 3),
    ])
    mean = edge.mean(axis=0)
    return tuple(int(v) for v in mean)


def _hull_minus_hair_mask(ctx: FaceContext, h: int, w: int) -> np.ndarray:
    """Return the OFIQ face_mask convex hull with BiSeNet hair pixels removed.

    OFIQ's face_mask is a convex hull of the 98 ADNet landmarks; on most
    portraits this hull sweeps through hair flowing past the temples and
    forehead. Subtracting BiSeNet's hair class (17) inside the hull
    preserves all real face features (eyes, brows, nose, lips, skin)
    while removing the hair pixels that cause visible bleed when the
    perturbation is rendered. ~66% of the original hull pixels survive,
    enough for OFIQ's hull-wide measurements to register the
    perturbation cleanly.
    """
    if ctx.face_mask is None or ctx.parsing_map is None:
        return ctx.face_mask if ctx.face_mask is not None else np.zeros((h, w), np.uint8)
    hair_small = (ctx.parsing_map == BISENET_HAIR).astype(np.uint8)
    hair = cv2.resize(hair_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return ((ctx.face_mask > 0) & (hair == 0)).astype(np.uint8)


def _warp_with_inpaint(
    img: np.ndarray, warp_fn: Callable,
) -> np.ndarray:
    """Apply a geometric warp and fill the empty source-out-of-bounds region.

    ``warp_fn(src, border_mode, border_value)`` must perform the transform.
    Empty regions (where the warp pulled in pixels outside the source
    image) are filled with cv2.BORDER_REFLECT — re-rendering the warp
    once with reflect-mode and compositing into the holes. v0.4 used
    cv2.inpaint with INPAINT_TELEA which collapsed to a flat-color
    smear on smooth backgrounds (sky, water, walls); reflect-fill
    extends the existing image content into the hole, giving a
    plausible continuation that matches the source texture.
    """
    h, w = img.shape[:2]
    warped = warp_fn(img, cv2.BORDER_CONSTANT, 0)
    marker = np.full((h, w), 255, dtype=np.uint8)
    marker_warped = warp_fn(marker, cv2.BORDER_CONSTANT, 0)
    hole_mask = (marker_warped < 128).astype(np.uint8) * 255
    if int(hole_mask.sum()) == 0:
        return warped
    # Re-warp with reflect-fill and composite into holes.
    reflect_fill = warp_fn(img, cv2.BORDER_REFLECT, 0)
    out = warped.copy()
    hole = hole_mask > 0
    out[hole] = reflect_fill[hole]
    return out


def _warp_with_flat_backdrop(
    img: np.ndarray, warp_fn: Callable, ctx: FaceContext | None,
) -> np.ndarray:
    """Apply a geometric warp and fill the empty region.

    Used by Category C operators (crop/margin shifts) where the OFIQ
    metric only cares about landmark positions. v0.5.0 used a uniform
    flat-color fill to avoid confounding BackgroundUniformity with
    hallucinated texture, but the result looked like a brown rectangle
    pasted on the image.

    v0.5.1 uses BORDER_REFLECT for the empty region (mirror-extends the
    existing image content into the hole) blended with the sampled
    backdrop color via a wide gaussian alpha. The reflect texture
    matches the source palette so BackgroundUniformity drift stays
    small, while the visual continuation is plausible instead of a
    flat color bar.
    """
    h, w = img.shape[:2]
    warped = warp_fn(img, cv2.BORDER_CONSTANT, 0)
    marker = np.full((h, w), 255, dtype=np.uint8)
    marker_warped = warp_fn(marker, cv2.BORDER_CONSTANT, 0)
    hole_mask = (marker_warped < 128)
    if not hole_mask.any():
        return warped
    bg_color = np.array(_sample_backdrop_color(img, ctx), dtype=np.float32)
    reflect_fill = warp_fn(img, cv2.BORDER_REFLECT, 0).astype(np.float32)
    flat_fill = np.full_like(reflect_fill, bg_color)
    # Soften reflect with bg_color: 70% reflect texture, 30% flat
    # backdrop. Keeps the visual continuity dominant while pulling
    # the hue toward the OFIQ-friendly uniform-bg target.
    blended = (reflect_fill * 0.7 + flat_fill * 0.3).astype(np.uint8)
    warped[hole_mask] = blended[hole_mask]
    return warped


def _reduce_dynamic_range(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Compress dynamic range around the face mean within the face mask [§7.3.7].

    OFIQ measures Shannon entropy of luminance histogram on the face
    landmark mask. To degrade, we crush the histogram around its own
    centre of mass (per-channel mean of face pixels) instead of pulling
    toward absolute mid-gray, so skin tonality is preserved while
    texture variation collapses. Capped at 85% compression so even
    sev=1.0 retains a hint of detail rather than going ghostly flat.
    """
    if severity < 0.01:
        return img
    # OFIQ DynamicRange measures Shannon entropy of the face LUMINANCE
    # histogram (not chroma). Quantize the Y channel of YCbCr and
    # leave Cb / Cr at full precision: destroys the luminance entropy
    # OFIQ scores while preserving natural skin / lip color (per-RGB
    # posterization shifts hue toward magenta / yellow at low levels).
    # Levels go from 64 (subtle banding) to 4 (heavy banding).
    # LuminanceVariance uses the related "compress around mean"
    # technique; the two operators are intentionally different to
    # target the specific statistic OFIQ measures for each.
    s_eff = float(np.sqrt(np.clip(severity, 0.0, 1.0)))
    n_levels = max(4, int(round(64.0 - s_eff * 60.0)))
    step = 255.0 / max(n_levels - 1, 1)

    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    y = ycbcr[..., 0]
    ycbcr[..., 0] = np.clip(np.round(y / step) * step, 0, 255)
    posterized_u8 = cv2.cvtColor(
        ycbcr.astype(np.uint8), cv2.COLOR_YCrCb2BGR,
    )
    h, w = img.shape[:2]

    if ctx is not None and ctx.face_mask is not None:
        clean_mask = _hull_minus_hair_mask(ctx, h, w)
        sigma = max(8.0, min(h, w) / 12)
        return _alpha_blend(
            img, posterized_u8, _feather_mask(clean_mask, sigma=sigma),
        )
    return posterized_u8


def _blur(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Whole-image Gaussian blur (out-of-focus simulation) [§7.3.8].

    OFIQ Sharpness measures Sobel/Laplace features inside the face crop.
    Real out-of-focus images blur the WHOLE scene, not just the face.
    Face-only blur left the hair pin-sharp against a fuzzy face which
    looked unnatural; whole-image blur matches real shallow-depth-of-
    field / wrong-focus shots and degrades the OFIQ scalar correctly.
    """
    if severity < 0.01:
        return img
    # v0.5.1: cap sigma at 6.5 (was 10.5). Past sigma=7 the face is
    # unrecognizable mush — no longer useful for ML training where
    # the operator is supposed to test "reduced sharpness", not
    # "complete identity loss". The OFIQ Sharpness scalar curve
    # plateaus past sigma=5 anyway.
    sigma = severity * 6.0 + 0.5
    ksize = int(6 * sigma + 1) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)


def _motion_blur(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Whole-image directional motion blur (camera shake) [§7.3.8].

    Real motion blur affects the entire frame uniformly; face-only
    motion blur looked like a stage spotlight effect.
    """
    if severity < 0.01:
        return img
    ksize = max(int(severity * 30) + 1, 3)
    kernel = np.zeros((ksize, ksize))
    rng = np.random.RandomState(seed)
    angle = rng.uniform(0, 180)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    for i in range(ksize):
        x = int(ksize // 2 + (i - ksize // 2) * cos_a)
        y = int(ksize // 2 + (i - ksize // 2) * sin_a)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1
    kernel /= kernel.sum() + 1e-8
    return cv2.filter2D(img, -1, kernel)


def _gaussian_noise(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Whole-image additive Gaussian noise (sensor noise) [§7.3.8].

    Real sensor noise (low-light, high ISO, cheap webcam) appears
    uniformly across the frame, not selectively on the face.
    """
    if severity < 0.01:
        return img
    rng = np.random.RandomState(seed)
    sigma = severity * 80
    noise = rng.randn(*img.shape) * sigma
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _jpeg_compression(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Cascaded JPEG + chroma quantization for §7.3.9 CompressionArtifacts.

    OFIQ's CompressionArtifacts is a CNN that returns a raw score in
    [0, 1] then maps it through a sigmoid (x0=0.33, w=0.092, rounded
    to int) to the [0, 100] scalar. Empirically the CNN's raw response
    on a clean CelebA face has a floor near 0.65 even under Q=1 JPEG
    -- well above the sigmoid's transition zone -- so single-pass JPEG
    leaves the scalar at 100 regardless of Q.

    The combination that DOES move the raw score below the floor is
    cascaded (chroma quantize + JPEG) passes. Severity controls both
    the chroma quantization step (1..32) and JPEG quality (95..3),
    cascaded for max(1, ceil(severity * 4)) passes. This drives the
    raw score down by ~0.25 per pass beyond the single-pass floor,
    eventually crossing the sigmoid transition and pulling the scalar
    below 100. Visually the result is the cascaded re-upload pattern
    real-world images accumulate (camera JPEG -> social-media re-encode
    -> screenshot -> re-upload).
    """
    if severity < 0.01:
        return img
    # v0.5.1: dialed back from "1995-web-GIF" to "social-media
    # re-upload" intensity. The OFIQ scalar floor is unmovable on
    # clean faces regardless (CNN raw response 0.65 well above
    # sigmoid x0=0.33), so we optimize for visible-but-recognizable
    # compression damage instead of maximum chroma destruction.
    # Caps: chroma_step ≤ 8 (was 32), JPEG Q ≥ 8 (was 3),
    # passes ≤ 2 (was 4). The face stays identifiable but JPEG
    # blocking + chroma bleed are unmistakable at sev=1.0.
    chroma_step = max(1, int(round(1 + severity * 7)))
    jpeg_q = max(8, int(round(95 - severity * 87)))
    passes = max(1, int(round(severity * 2)))
    out = img
    for _ in range(passes):
        ycc = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        ycc[..., 1] = (ycc[..., 1] // chroma_step) * chroma_step
        ycc[..., 2] = (ycc[..., 2] // chroma_step) * chroma_step
        out = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
        _, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return out


def _color_cast_cielab(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Apply a global CIELAB color cast (white-balance shift) [§7.3.10].

    OFIQ measures CIELAB distance from natural skin ranges (a* in
    [5,25], b* in [5,35]) within left / right cheek ROI zones near the
    eyes. A global white-balance shift naturally drives those ROI zones
    out of range while looking like a real photographic color cast
    (wrong white balance, mixed-light scene, color filter) instead of
    rectangle stickers on the cheeks.

    Direction is randomized per seed: positive = warm cast (yellow /
    orange), negative = cool cast (blue / cyan).
    """
    if severity < 0.01:
        return img
    rng = np.random.RandomState(seed)
    direction = rng.choice([-1, 1])

    # Shift a* / b* uniformly. v0.5.1 caps magnitude at 30 LAB units
    # (was 50) so the result lands JUST outside the natural plateau
    # (a* ∈ [5,25], b* ∈ [5,35]) instead of overshooting into
    # physically impossible cyan / magenta. 30 units is enough to
    # push the OFIQ scalar to ~0 without producing horror-movie
    # filter results.
    shift = direction * severity * 30.0

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] = np.clip(lab[..., 1] + shift, 0, 255)
    lab[..., 2] = np.clip(lab[..., 2] + shift, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _color_cast_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: global RGB channel boost."""
    rng = np.random.RandomState(seed)
    channel = rng.randint(0, 3)
    out = img.astype(np.float32)
    out[:, :, channel] = np.clip(out[:, :, channel] + severity * 80, 0, 255)
    return out.astype(np.uint8)


def _radial_distortion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Barrel distortion + smooth lens vignetting [Annex D.2.1].

    Uses cv2.remap with a quadratic radial term to produce true lens
    distortion. The corners that fall outside the source image after
    distortion are then darkened by a smooth Gaussian-falloff vignette
    (matching real wide-angle lens vignetting) instead of being filled
    with hard inpainted texture or a black mask. The combined result
    looks like a real wide-angle / fish-eye photograph.
    """
    if severity < 0.01:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2, h / 2
    k = severity * 0.5
    y_idx, x_idx = np.mgrid[0:h, 0:w].astype(np.float32)
    x_norm = (x_idx - cx) / cx
    y_norm = (y_idx - cy) / cy
    r2 = x_norm ** 2 + y_norm ** 2
    scale = 1 + k * r2
    map_x = (cx + x_norm * scale * cx).astype(np.float32)
    map_y = (cy + y_norm * scale * cy).astype(np.float32)
    distorted = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    # Smooth radial vignette: cos^2 falloff scaled by severity. v0.5.1
    # caps the corner darkening at 40% (was 75%) so the corners stay
    # visible — past the v0.5.0 cap the result looked like binoculars
    # / camera-obscura, not natural lens vignette.
    r = np.sqrt(r2)  # 0 at center, ~sqrt(2) at corners
    vignette = np.cos(np.clip(r, 0, 1) * np.pi / 2) ** 2  # 1 -> 0 at r=1
    falloff = 1.0 - severity * 0.40 * (1.0 - vignette)
    falloff = np.clip(falloff, 0.55, 1.0)[..., None]
    out = distorted.astype(np.float32) * falloff
    return np.clip(out, 0, 255).astype(np.uint8)


# =========================================================================
# Section 7 -- Subject-Related Components
# =========================================================================

def _eyes_close_warp(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Close the subject's eyes [§7.4.3].

    OFIQ measures min(max_pair_dist(LEFT_EYE), max_pair_dist(RIGHT_EYE)) / t.

    Method dispatch via env var (mirrors ExpressionNeutrality):

    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p`` -> InstructPix2Pix renders
      photorealistic closed eyelids with proper eyelashes/skin (Phase 6).
    - default -> TPS warp + skin painting fallback (Phase 5, fast,
      lower fidelity).

    The TPS path warps upper eyelid landmarks toward their lower pair
    counterparts then composites a feathered skin patch (sampled from
    the cheek) over the eye polygon to fake closed lids. Realistic at
    low severity; at high severity the lack of real eyelid texture
    becomes visible.
    """
    if ctx is None:
        return _eye_occlusion_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    import os
    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_eyes_closed_ip2p,
            )
            if is_sd_available():
                return render_eyes_closed_ip2p(img, ctx, severity, seed)
        except Exception:
            pass  # fall through to TPS

    landmarks = ctx.landmarks_98.astype(np.float64)
    h, w = img.shape[:2]

    # Pre-process: heavily blur the eye region so ADNet cannot use iris
    # detail (sclera/iris boundary, pupil center, eyelash texture) to
    # find the original eye landmarks. Without this the warp + paint
    # below is overruled by ADNet's iris-anchored re-detection.
    img = _blur_eye_regions(img, ctx.landmarks_98, severity)

    src_points = landmarks.copy()
    dst_points = landmarks.copy()

    # Move upper eyelid all the way to the lower at sev=1.0 (gap*1.0).
    # v0.4 used 0.9 which left a small slit ADNet would still detect.
    # Also pull lower lid up by a small fraction so the eye fully
    # collapses on a wider region, not just a single pixel line.
    all_pairs = PAIRS_LEFT_EYE + PAIRS_RIGHT_EYE
    for upper_idx, lower_idx in all_pairs:
        upper = landmarks[upper_idx]
        lower = landmarks[lower_idx]
        gap = upper - lower
        dst_points[upper_idx] = upper - gap * severity * 1.0
        dst_points[lower_idx] = lower + gap * severity * 0.15

    warped = _apply_rbf_warp(img, src_points, dst_points, seed)

    # Skin painting: ramp opacity in much earlier (severity > 0.1)
    # and reach full opacity by 0.6, so ADNet sees skin (not warped
    # iris) where the eye used to be, even at moderate severity.
    paint_alpha = float(np.clip((severity - 0.1) / 0.5, 0.0, 1.0))
    if paint_alpha <= 0.0:
        return warped

    return _paint_lid_skin(warped, ctx.landmarks_98, paint_alpha)


def _blur_eye_regions(
    img: np.ndarray, landmarks_98: np.ndarray, severity: float,
) -> np.ndarray:
    """Heavy median blur over each eye bounding-box, ramping with severity.

    Strips iris/pupil/sclera detail so ADNet cannot anchor eye landmarks
    to the original eye boundary after the close-warp. Without this the
    warp + skin paint do not move OFIQ's EyesOpen scalar on most CelebA
    crops -- ADNet just re-detects the eye at the unwarped spread.
    """
    if severity < 0.05:
        return img
    out = img.copy()
    h, w = img.shape[:2]
    ksize = max(3, int(severity * 25)) | 1  # odd, up to ~25
    for eye_indices in (LEFT_EYE, RIGHT_EYE):
        eye_pts = landmarks_98[eye_indices].astype(np.int32)
        x, y, ew, eh = cv2.boundingRect(eye_pts)
        # Pad bounding box outward so the blur covers eyelashes
        pad_x, pad_y = ew // 2, eh // 2
        x0 = max(0, x - pad_x); y0 = max(0, y - pad_y)
        x1 = min(w, x + ew + pad_x); y1 = min(h, y + eh + pad_y)
        roi = out[y0:y1, x0:x1]
        if roi.size:
            out[y0:y1, x0:x1] = cv2.medianBlur(roi, ksize)
    return out


def _paint_lid_skin(
    img: np.ndarray, landmarks_98: np.ndarray, alpha_max: float,
) -> np.ndarray:
    """Sample local skin tone and composite a soft patch over each eye polygon."""
    h, w = img.shape[:2]
    out = img.astype(np.float32)
    img_f = img.astype(np.float32)

    for eye_indices in (LEFT_EYE, RIGHT_EYE):
        eye_pts = landmarks_98[eye_indices].astype(np.int32)
        x, y, ew, eh = cv2.boundingRect(eye_pts)
        if ew < 4 or eh < 2:
            continue

        # Sample skin tone from a band just below the eye (cheek), which
        # tends to be skin-only and well-lit. Fall back to the band above.
        sample_h = max(2, eh // 2)
        cheek_y1 = min(h, y + eh + max(2, eh // 4))
        cheek_y2 = min(h, cheek_y1 + sample_h)
        cheek_x1 = max(0, x)
        cheek_x2 = min(w, x + ew)
        if cheek_y2 - cheek_y1 < 1 or cheek_x2 - cheek_x1 < 1:
            continue

        cheek = img_f[cheek_y1:cheek_y2, cheek_x1:cheek_x2]
        skin_bgr = cheek.reshape(-1, 3).mean(axis=0)

        # Build a soft mask from the eye polygon. Inflate vertically a
        # touch so the closed lid covers the full eye opening.
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        hull = cv2.convexHull(eye_pts)
        cv2.fillConvexPoly(eye_mask, hull, 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, max(3, eh // 3)))
        eye_mask = cv2.dilate(eye_mask, kernel)
        sigma = max(1.5, eh / 4)
        soft = cv2.GaussianBlur(eye_mask.astype(np.float32), (0, 0), sigma)
        soft = np.clip(soft, 0.0, 1.0) * alpha_max

        # Build a per-pixel skin patch with subtle horizontal lash-line
        # shading so the closed lid does not look like a flat decal.
        patch = np.empty_like(img_f)
        patch[..., 0] = skin_bgr[0]
        patch[..., 1] = skin_bgr[1]
        patch[..., 2] = skin_bgr[2]

        # Lash line: a faint dark horizontal stripe near eye center
        ys = np.arange(h, dtype=np.float32)
        eye_cy = float(y + eh / 2)
        # Lash darkening: keep subtle. v0.5.0 used 0.18 + sigma=eh/6
        # which produced a visible "bandaid" pink stripe across the
        # eye. v0.5.1 drops to 0.06 + sigma=eh/3 (wider falloff) so
        # the lash hint stays inside the realistic eyelid-shadow range.
        lash = np.exp(-((ys - eye_cy) ** 2) / max(1.0, (eh / 3) ** 2))
        lash_strength = 0.06
        patch -= patch * (lash[:, None, None] * lash_strength)

        a = soft[..., None]
        out = out * (1 - a) + patch * a

    return np.clip(out, 0, 255).astype(np.uint8)


def _mouth_open_warp(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Open the mouth via TPS lip-warp + dark interior fill [§7.4.4].

    OFIQ measures max_pair_dist(MOUTH_INNER_pairs) / t with ADNet
    landmarks 88-95 (upper inner 88-91, lower inner 92-95). To move
    that scalar we need ADNet's re-detection on the warped image to
    place the upper-inner and lower-inner landmarks farther apart.

    A dark ellipse painted inside the closed lips is NOT enough --
    ADNet treats unmoved lips as the mouth boundary regardless of
    painted-on shadows, so max_pair_dist stays at its source value.
    The fix: TPS-warp the lip landmarks themselves (push lower lip
    down, upper lip up by severity * t * 0.20), then paint the dark
    interior into the gap that opens up. The TPS displacement lets
    ADNet re-detect the lip boundary at the new location, and the
    dark fill keeps the synthesized gap from looking like a glitch.
    """
    if ctx is None:
        return _mouth_occlusion_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    h, w = img.shape[:2]
    t = ctx.t_metric
    landmarks = ctx.landmarks_98

    # ----- Step 1: TPS warp lower lip down + upper lip up so ADNet
    # re-detects mouth_inner at the new (wider) positions.
    # 15% of t per side: max OFIQ raw of ~0.30 (well past sigmoid x0=0.20,
    # w=0.06). Past 0.18 per side the warp creates impossible geometry
    # that confuses ADNet, causing non-monotonic scalar response.
    open_amt = float(severity) * t * 0.15
    # Source landmarks (lip outer + inner)
    upper_outer = landmarks[[76, 77, 78, 79, 80, 81, 82]]
    lower_outer = landmarks[[82, 83, 84, 85, 86, 87, 76]]
    upper_inner = landmarks[[88, 89, 90, 91]]
    lower_inner = landmarks[[92, 93, 94, 95]]
    # Target: move uppers up, lowers down
    src_pts = np.vstack([upper_outer, lower_outer, upper_inner, lower_inner]).astype(np.float32)
    dst_pts = src_pts.copy()
    dst_pts[:7, 1] -= open_amt * 0.5  # upper outer up
    dst_pts[7:14, 1] += open_amt * 0.5  # lower outer down
    dst_pts[14:18, 1] -= open_amt  # upper inner up (more)
    dst_pts[18:22, 1] += open_amt  # lower inner down (more)
    # Anchor non-mouth region with corner + edge midpoint pins
    anchors = np.array(
        [[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
         [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2]],
        dtype=np.float32,
    )
    src_pts = np.vstack([src_pts, anchors])
    dst_pts = np.vstack([dst_pts, anchors])
    try:
        from scipy.interpolate import RBFInterpolator
        # Inverse displacement field: (dst -> src) so cv2.remap can
        # sample the source image at the right location.
        dx = src_pts[:, 0] - dst_pts[:, 0]
        dy = src_pts[:, 1] - dst_pts[:, 1]
        rbf_x = RBFInterpolator(dst_pts, dx, kernel="thin_plate_spline", smoothing=1.0)
        rbf_y = RBFInterpolator(dst_pts, dy, kernel="thin_plate_spline", smoothing=1.0)
        gy, gx = np.mgrid[0:h:4, 0:w:4].astype(np.float32)
        pts = np.column_stack([gx.ravel(), gy.ravel()])
        disp_x = rbf_x(pts).reshape(gx.shape).astype(np.float32)
        disp_y = rbf_y(pts).reshape(gx.shape).astype(np.float32)
        map_x_lo = (gx + disp_x).astype(np.float32)
        map_y_lo = (gy + disp_y).astype(np.float32)
        map_x = cv2.resize(map_x_lo, (w, h), interpolation=cv2.INTER_LINEAR)
        map_y = cv2.resize(map_y_lo, (w, h), interpolation=cv2.INTER_LINEAR)
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    except Exception:
        pass

    # Geometry: center of the mouth opening = midpoint of inner lip pairs
    upper_inner = landmarks[[88, 89, 90, 91]].mean(axis=0)
    lower_inner = landmarks[[92, 93, 94, 95]].mean(axis=0)
    center = (upper_inner + lower_inner) / 2

    # Width = distance between mouth corners (76 outer right, 82 outer left)
    corners = landmarks[[76, 82]]
    mouth_width = float(np.linalg.norm(corners[1] - corners[0]))

    # Ellipse axes:
    #   Horizontal: ~50% of mouth width (sits inside lips, not on them)
    #   Vertical: severity * t * 0.30 (target raw=0.30 at sev=1.0, well
    #   past OFIQ MouthClosed sigmoid x0=0.2, w=0.06 — so the scalar
    #   actually drops). v0.4 used 0.14 which kept raw at ~0.10
    #   regardless of severity; the painted ellipse was too small to
    #   move ADNet's mouth_inner landmark detection. Cap at the full
    #   mouth_width/2 (round-mouth limit) instead of 35% (under-cap
    #   that matched the v0.4 too-small-ellipse problem).
    ax_x = max(4, int(mouth_width * 0.45))
    ax_y_max = max(3, int(mouth_width * 0.55))
    ax_y = int(min(ax_y_max, severity * t * 0.30))
    if ax_y < 2:
        return img

    # Sample the natural mouth-shadow color from a 3x3 patch at the
    # current lip-line center (preserves white balance).
    cy_int, cx_int = int(center[1]), int(center[0])
    patch = img[max(0, cy_int - 1):cy_int + 2, max(0, cx_int - 1):cx_int + 2]
    if patch.size:
        base_color = patch.reshape(-1, 3).mean(axis=0) * 0.35  # darken
    else:
        base_color = np.array([20, 15, 25], dtype=np.float32)
    base_color = np.clip(base_color, 0, 60).astype(int).tolist()

    # Build the dark mouth interior with a subtle dental hint:
    #   1. Dominant dark fill (the OFIQ-detectable mouth opening)
    #   2. Thin lighter band near the top (suggests upper teeth)
    #   3. Subtle red gum line at the very top edge (anatomical)
    # The dark fill dominates so ADNet still detects the lip / interior
    # boundary correctly; the dental hint just makes it look less like
    # a flat black hole.
    interior = np.full_like(img, base_color, dtype=np.uint8)

    # Teeth band: top ~25% of ellipse, slightly lighter than interior.
    teeth_color = (np.array(base_color, dtype=np.float32) * 1.6
                   + np.array([180, 180, 175], dtype=np.float32) * 0.45)
    teeth_color = np.clip(teeth_color, 0, 165).astype(np.uint8)
    teeth_mask = np.zeros((h, w), dtype=np.uint8)
    teeth_y_top = cy_int - ax_y + max(1, ax_y // 6)
    teeth_y_bot = cy_int - max(0, int(ax_y * 0.15))
    cv2.ellipse(
        teeth_mask,
        (cx_int, cy_int),
        (max(2, ax_x - 2), ax_y),
        0, 0, 360, 255, -1,
    )
    # Restrict teeth band vertically
    teeth_band = np.zeros((h, w), dtype=np.uint8)
    teeth_band[teeth_y_top:teeth_y_bot, :] = teeth_mask[teeth_y_top:teeth_y_bot, :]

    interior_with_teeth = interior.copy()
    teeth_alpha = (cv2.GaussianBlur(teeth_band.astype(np.float32), (0, 0), 1.0)
                   / 255.0)[..., None] * 0.55
    interior_with_teeth = (interior.astype(np.float32) * (1 - teeth_alpha)
                            + teeth_color.astype(np.float32) * teeth_alpha)

    # Gum hint: 1-2 px red ring at the very top
    gum_color = np.array([60, 30, 95], dtype=np.uint8)  # dark red-pink (BGR)
    gum_band = np.zeros((h, w), dtype=np.uint8)
    gum_y = cy_int - ax_y + 1
    cv2.line(gum_band, (cx_int - ax_x, gum_y), (cx_int + ax_x, gum_y), 255, 1)
    gum_alpha = (cv2.GaussianBlur(gum_band.astype(np.float32), (0, 0), 1.0)
                 / 255.0)[..., None] * 0.45
    interior_with_teeth = (interior_with_teeth.astype(np.float32) * (1 - gum_alpha)
                           + gum_color.astype(np.float32) * gum_alpha)

    # Composite the styled interior into the image via the ellipse mask
    ellipse_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        ellipse_mask,
        (cx_int, cy_int),
        (ax_x, ax_y),
        0, 0, 360, 255, -1,
    )
    soft = cv2.GaussianBlur(ellipse_mask.astype(np.float32), (0, 0), 1.5)
    soft = np.clip(soft / 255.0, 0.0, 1.0)[..., None]

    out = (img.astype(np.float32) * (1 - soft)
           + interior_with_teeth.astype(np.float32) * soft)
    return np.clip(out, 0, 255).astype(np.uint8)


def _eye_occlusion_evz(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Render sunglasses over the Eye Visibility Zone [§7.4.5].

    Method dispatch via env var:

    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p`` -> InstructPix2Pix renders
      photorealistic sunglasses with proper frame, lens darkness, and
      light interaction (Phase 6).
    - default -> procedural occluder via ``occluders.render_sunglasses()``
      (Phase 3, fast, looks pasted on).
    """
    if ctx is None:
        return _eye_occlusion_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    import os
    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_sunglasses_ip2p,
            )
            if is_sd_available():
                return render_sunglasses_ip2p(img, ctx, severity, seed)
        except Exception:
            pass

    from ofiq_syngen.occluders import render_sunglasses
    return render_sunglasses(img, ctx, severity, seed)


def _eye_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: band-based eye occlusion."""
    h, w = img.shape[:2]
    band_h = int(h * 0.15 * (1 + severity))
    y_start = int(h * 0.25)
    out = img.copy()
    rng = np.random.RandomState(seed)
    color = rng.randint(0, 60, 3).tolist()
    out[y_start:y_start + band_h, :] = color
    return out


def _mouth_occlusion_polygon(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Render a surgical mask over the mouth region [§7.4.6].

    Method dispatch via env var:

    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p`` -> InstructPix2Pix renders
      a photorealistic surgical mask conforming to face geometry.
    - default -> procedural pleated surgical mask via
      ``occluders.render_surgical_mask()`` (looks pasted on).
    """
    if ctx is None:
        return _mouth_occlusion_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    import os
    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_surgical_mask_ip2p,
            )
            if is_sd_available():
                return render_surgical_mask_ip2p(img, ctx, severity, seed)
        except Exception:
            pass

    from ofiq_syngen.occluders import render_surgical_mask
    return render_surgical_mask(img, ctx, severity, seed)


def _mouth_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: band-based mouth occlusion."""
    h, w = img.shape[:2]
    band_h = int(h * 0.2 * (1 + severity))
    y_start = int(h * 0.55)
    out = img.copy()
    rng = np.random.RandomState(seed)
    color = [rng.randint(180, 220)] * 3
    out[y_start:y_start + band_h, :] = color
    return out


def _face_region_occlusion(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Render an occluder over the face region [§7.4.7].

    Method dispatch via env var:

    - ``OFIQ_SYNGEN_EXPRESSION_METHOD=ip2p`` -> InstructPix2Pix renders
      a photorealistic hand covering the face (Phase 6).
    - default -> procedural skin-tone-matched hand silhouette
      (Phase 3, fast, looks like a brown blob).
    """
    if ctx is None:
        return _rect_occlusion_fallback(img, severity, seed)
    if severity < 0.01:
        return img

    import os
    method = os.environ.get("OFIQ_SYNGEN_EXPRESSION_METHOD", "3dmm").lower()
    if method in ("ip2p", "instructpix2pix", "instruct_pix2pix"):
        try:
            from ofiq_syngen.expression_diffusion import (
                is_sd_available, render_hand_occluder_ip2p,
            )
            if is_sd_available():
                return render_hand_occluder_ip2p(img, ctx, severity, seed)
        except Exception:
            pass

    from ofiq_syngen.occluders import render_hand_occluder
    return render_hand_occluder(img, ctx, severity, seed)


def _rect_occlusion_fallback(
    img: np.ndarray, severity: float, seed: int,
) -> np.ndarray:
    """Fallback: random rectangular occlusion."""
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    frac = severity * 0.6
    oh, ow = int(h * frac), int(w * frac)
    if oh < 2 or ow < 2:
        return img
    y = rng.randint(0, max(h - oh, 1))
    x = rng.randint(0, max(w - ow, 1))
    out = img.copy()
    color = rng.randint(0, 256, 3).tolist()
    out[y:y + oh, x:x + ow] = color
    return out


def _reduce_ied(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shrink face in frame to reduce inter-eye distance [§7.4.8].

    OFIQ measures euclidean distance between eye centers (pixels) / cos(yaw).
    Downscale-and-upscale doesn't change this on aligned crops.
    Instead: shrink image and embed in padded canvas.

    v0.5.1: cap shrink at scale=0.45 (was 0.30) so the face stays
    recognizable, and fill the padding with reflect-extended source
    pixels blended with backdrop color (was flat backdrop) so the
    border doesn't look like a brown rectangle frame around a tiny
    thumbnail.
    """
    h, w = img.shape[:2]
    scale = max(0.45, 1.0 - severity * 0.55)

    new_h, new_w = max(4, int(h * scale)), max(4, int(w * scale))
    pad_top = (h - new_h) // 2
    pad_left = (w - new_w) // 2

    # Build the canvas. v0.5.0 used a flat backdrop color (visible
    # solid frame). v0.5.1a used BORDER_REFLECT (visible kaleidoscope
    # tiles of the shrunken face). v0.5.1b: heavily blur the source
    # image and use it as a soft out-of-focus backdrop. The blur kills
    # any recognizable detail so the eye doesn't perceive the padding
    # as a tile pattern, but the source palette is preserved (no
    # BackgroundUniformity drift, no flat-color sticker look).
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred_bg = cv2.GaussianBlur(img, (0, 0), max(20, min(h, w) // 8))
    bg_color = np.array(_sample_backdrop_color(img, ctx), dtype=np.float32)
    flat = np.full_like(blurred_bg, bg_color, dtype=np.uint8)
    canvas = (blurred_bg.astype(np.float32) * 0.5
              + flat.astype(np.float32) * 0.5).astype(np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = small
    return canvas


def _reduce_head_size(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Zoom-in head-size perturbation [§7.4.9].

    OFIQ measures |t/imageHeight - 0.45| with sigmoid x0=0, w=0.05.
    The optimum is t/imageHeight = 0.45; both higher AND lower raw
    degrade. Most natural face imagery has raw ~0.20 (face takes up
    ~20% of image height), so the v0.4 shrink-only operator drove
    scalar from ~2 to 0 with no real movement (already at the floor
    of the lower-degradation side).

    v0.5 zooms IN instead: severity 0..1 -> zoom 1x..3x, which
    pushes raw past 0.45 and into the upper-degradation regime where
    the OFIQ scalar has ~80 points of headroom. Side-effect: at low
    severity the scalar will pass through 100 (near the optimum)
    before degrading again as zoom continues. This non-monotonic
    behavior is the price of having ANY measurable degradation on
    typical natural-portrait test images. ctx.t_metric is unreliable
    relative to OFIQ's measurement (different alignment basis) so we
    don't try to detect direction.
    """
    if severity < 0.01:
        return img
    return _zoom_in_face(img, severity, ctx)


def _zoom_in_face(
    img: np.ndarray, severity: float, ctx: FaceContext | None,
) -> np.ndarray:
    """Crop a smaller window around the face landmarks and upscale.

    Drives OFIQ HeadSize raw above 0.45 by making the face fill more
    of the image height after the upscale. Center the crop on the
    face landmark centroid (when ctx is present), else on the image
    center.
    """
    h, w = img.shape[:2]
    zoom = 1.0 + float(severity) * 1.5  # 1x..2.5x (3x can break OFIQ face alignment)
    crop_h = max(8, int(h / zoom))
    crop_w = max(8, int(w / zoom))
    if ctx is not None and getattr(ctx, "landmarks_98", None) is not None:
        center = ctx.landmarks_98.mean(axis=0)
        cx, cy = float(center[0]), float(center[1])
    else:
        cx, cy = w / 2, h / 2
    x0 = int(np.clip(cx - crop_w / 2, 0, w - crop_w))
    y0 = int(np.clip(cy - crop_h / 2, 0, h - crop_h))
    cropped = img[y0:y0 + crop_h, x0:x0 + crop_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# =========================================================================
# Section 8 -- Geometric/Pose Components
# =========================================================================

_3D_PIPELINE = None


def _get_3d_pipeline():
    """Singleton 3D pipeline (DECA + FLAME + pyrender). Lazy-loaded on
    first use; ~2.4s cold start, ~0.4s per call after."""
    global _3D_PIPELINE
    if _3D_PIPELINE is None:
        try:
            from ofiq_syngen.three_d.pipeline import DegradationPipeline as P3D
            _3D_PIPELINE = P3D()
        except Exception:
            _3D_PIPELINE = False
    return _3D_PIPELINE if _3D_PIPELINE is not False else None


def _yaw_rotation(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Yaw rotation [§7.4.11 yaw/pitch/roll].

    Tier 1: True 3D rotation via FLAME mesh fit (DECA) + pyrender.
        Source-textured mesh, real anatomical rotation, sev 0..1 -> 0..35°.
    Tier 2: 2D TPS dense BFM warp (with hair-landmark TPS), capped at ±10°.
    Tier 3: 2D perspective squeeze, capped at ±0.5 severity.
    """
    if severity < 0.01:
        return img

    # Tier 1: 3D FLAME rotation
    pipe = _get_3d_pipeline()
    if pipe is not None:
        try:
            out, _ = pipe.degrade_single(
                img, "HeadPoseYaw.scalar", severity=severity, seed=seed,
            )
            return out
        except Exception:
            pass

    if ctx is not None and getattr(ctx, "raw_3ddfa_params", None) is not None:
        # Source-yaw direction: contour asymmetry on ADNet landmarks is
        # the most reliable signal we have — when the face is yawed
        # toward observer-right (OFIQ yaw > 0), the LEFT contour points
        # spread further from the chin than the right (anti-symmetric).
        # ctx.head_pose disagrees with OFIQ's binary on most CelebA
        # crops (sign-inverted on 2 of 3 test images), so use it only
        # as a tie-breaker when contour asymmetry is below threshold.
        direction = 0.0
        try:
            from ofiq_syngen.landmark_utils import CONTOUR
            contour = ctx.landmarks_98[CONTOUR]
            chin_x = contour[16, 0]
            left_extent = abs(chin_x - contour[:16, 0].min())
            right_extent = abs(contour[17:, 0].max() - chin_x)
            asym = (right_extent - left_extent) / max(right_extent + left_extent, 1)
            # Invert: contour asymmetry is anti-correlated with OFIQ yaw.
            if abs(asym) >= 0.10:
                direction = 1.0 if asym <= 0 else -1.0
        except Exception:
            pass
        if direction == 0.0 and getattr(ctx, "head_pose", None) is not None and abs(ctx.head_pose[0]) >= 5.0:
            direction = 1.0 if ctx.head_pose[0] >= 0 else -1.0
        if direction == 0.0:
            rng = np.random.RandomState(seed)
            direction = float(rng.choice([-1, 1]))
        try:
            from ofiq_syngen.face_3dmm_dense import (
                is_dense_available, render_pose_dense,
            )
            if is_dense_available():
                # Cap at 5deg: BFM TPS warp produces non-monotonic OFIQ
                # response past ~5deg (yaw_deg=10 actually degrades less
                # than yaw_deg=5 because the OFIQ pose model misfires on
                # the warped output).
                yaw_deg = float(min(severity, 1.0) * 5.0 * direction)
                return render_pose_dense(img, ctx, yaw_deg=yaw_deg)
        except Exception:
            pass
        try:
            from ofiq_syngen.face_3dmm_nvdiff import (
                is_nvdiff_available, render_pose_nvdiff,
            )
            if is_nvdiff_available():
                yaw_deg = float(min(severity, 1.0) * 5.0 * direction)
                return render_pose_nvdiff(img, ctx, yaw_deg=yaw_deg)
        except Exception:
            pass

    h, w = img.shape[:2]
    # Source yaw direction detection: try ctx.head_pose first (OFIQ-aligned
    # head pose model), fall back to ctx.raw_3ddfa_params, then seed-rng.
    # Empirically the perspective-squeeze direction needs to oppose the
    # squeeze convention: NEGATIVE squeeze adds yaw on a face already
    # turned positive-yaw direction. We explicitly invert here so positive
    # source_yaw -> negative squeeze (which ADDS yaw per our earlier
    # measurement: squeeze=-0.3 on yaw=+21deg face gave yaw=+23deg).
    src_yaw_sign = 0.0
    if ctx is not None:
        if getattr(ctx, "head_pose", None) is not None and abs(ctx.head_pose[0]) >= 5.0:
            src_yaw_sign = 1.0 if ctx.head_pose[0] >= 0 else -1.0
        elif getattr(ctx, "raw_3ddfa_params", None) is not None:
            try:
                from ofiq_syngen.face_3dmm import parse_3ddfa_params
                R, _, _, _ = parse_3ddfa_params(ctx.raw_3ddfa_params)
                src_yaw = float(np.arcsin(np.clip(R[0, 2], -1, 1)))
                if abs(src_yaw) >= 0.1:
                    src_yaw_sign = 1.0 if src_yaw >= 0 else -1.0
            except Exception:
                pass
    if src_yaw_sign != 0.0:
        # Negative squeeze adds yaw on a positive-yaw source.
        direction = -src_yaw_sign
    else:
        rng = np.random.RandomState(seed)
        direction = float(rng.choice([-1, 1]))
    # Push amplitude to 0.40 max (vs old 0.25). At 0.45+ OFIQ face
    # alignment fails (-1 sentinel), so 0.40 is the safe ceiling.
    squeeze = float(min(severity, 1.0)) * 0.50 * direction  # v0.5.1: bumped from 0.40 for visibility
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if squeeze > 0:
        dst = np.float32([[0, int(h * squeeze)], [w, 0], [w, h], [0, int(h * (1 - squeeze))]])
    else:
        dst = np.float32([[0, 0], [w, int(h * (-squeeze))], [w, int(h * (1 + squeeze))], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return _warp_with_inpaint(
        img,
        lambda src_, mode, val: cv2.warpPerspective(
            src_, M, (w, h), borderMode=mode, borderValue=val,
        ),
    )


def _pitch_tilt(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Pitch rotation [§7.4.11 yaw/pitch/roll]. Same tiering as yaw."""
    if severity < 0.01:
        return img

    pipe = _get_3d_pipeline()
    if pipe is not None:
        try:
            out, _ = pipe.degrade_single(
                img, "HeadPosePitch.scalar", severity=severity, seed=seed,
            )
            return out
        except Exception:
            pass

    # Pose-aware direction (same approach as yaw)
    src_pitch_sign = 0.0
    if (
        ctx is not None
        and getattr(ctx, "head_pose", None) is not None
        and abs(ctx.head_pose[1]) >= 5.0
    ):
        src_pitch_sign = 1.0 if ctx.head_pose[1] >= 0 else -1.0

    if ctx is not None and getattr(ctx, "raw_3ddfa_params", None) is not None:
        if src_pitch_sign != 0.0:
            direction = src_pitch_sign
        else:
            rng = np.random.RandomState(seed)
            direction = float(rng.choice([-1, 1]))
        # Prefer dense BFM TPS (hair rotates with head)
        try:
            from ofiq_syngen.face_3dmm_dense import (
                is_dense_available, render_pose_dense,
            )
            if is_dense_available():
                pitch_deg = float(min(severity, 1.0) * 10.0 * direction)
                return render_pose_dense(img, ctx, pitch_deg=pitch_deg)
        except Exception:
            pass
        try:
            from ofiq_syngen.face_3dmm_nvdiff import (
                is_nvdiff_available, render_pose_nvdiff,
            )
            if is_nvdiff_available():
                pitch_deg = float(min(severity, 1.0) * 10.0 * direction)
                return render_pose_nvdiff(img, ctx, pitch_deg=pitch_deg)
        except Exception:
            pass

    h, w = img.shape[:2]
    if src_pitch_sign != 0.0:
        # Negative perspective squeeze adds positive pitch (same convention
        # as yaw operator above); invert the sign to ADD to source pitch.
        direction = -src_pitch_sign
    else:
        rng = np.random.RandomState(seed)
        direction = float(rng.choice([-1, 1]))
    squeeze = float(min(severity, 1.0)) * 0.50 * direction  # v0.5.1: bumped from 0.40 for visibility
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    if squeeze > 0:
        dst = np.float32([[int(w * squeeze), 0], [int(w * (1 - squeeze)), 0], [w, h], [0, h]])
    else:
        dst = np.float32([[0, 0], [w, 0], [int(w * (1 + squeeze)), h], [int(w * (-squeeze)), h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return _warp_with_inpaint(
        img,
        lambda src_, mode, val: cv2.warpPerspective(
            src_, M, (w, h), borderMode=mode, borderValue=val,
        ),
    )


def _roll_rotation(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """In-plane rotation [§7.4.11 yaw/pitch/roll]. Keep as-is (EXCELLENT fidelity)."""
    h, w = img.shape[:2]
    rng = np.random.RandomState(seed)
    angle = severity * 30 * rng.choice([-1, 1])
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return _warp_with_inpaint(
        img,
        lambda src, mode, val: cv2.warpAffine(
            src, M, (w, h), borderMode=mode, borderValue=val,
        ),
    )


def _crop_left(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image LEFT to push face toward left edge (degrades LeftwardCrop) [§7.4.10.1].

    OFIQ measures q_l = rightEyeCenter.x / IED. To DEGRADE this scalar we
    must DECREASE q_l, which means moving the face toward the LEFT edge of
    the image (small X_R). cv2.warpAffine with NEGATIVE shift_x translates
    the source image content LEFT.
    """
    h, w = img.shape[:2]
    shift_x = -int(severity * w * 0.4)  # negative = content moves left = face moves left
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    return _warp_with_flat_backdrop(
        img,
        lambda src, mode, val: cv2.warpAffine(
            src, M, (w, h), borderMode=mode, borderValue=val,
        ),
        ctx,
    )


def _crop_right(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image RIGHT to push face toward right edge (degrades RightwardCrop) [§7.4.10.2].

    OFIQ measures q_r = (imgW - leftEyeCenter.x) / IED. To DEGRADE this scalar
    we must DECREASE q_r, which means moving the face toward the RIGHT edge
    of the image (X_L close to imgW). Positive shift_x translates content RIGHT.
    """
    h, w = img.shape[:2]
    shift_x = int(severity * w * 0.4)  # positive = content moves right = face moves right
    M = np.float32([[1, 0, shift_x], [0, 1, 0]])
    return _warp_with_flat_backdrop(
        img,
        lambda src, mode, val: cv2.warpAffine(
            src, M, (w, h), borderMode=mode, borderValue=val,
        ),
        ctx,
    )


def _margin_above(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image UP to push face toward top edge (degrades MarginAbove) [§7.4.10.3].

    OFIQ measures q_a = eyeMidPoint.y / t. To DEGRADE this scalar we must
    DECREASE q_a, which means moving the face toward the TOP of the image
    (small Y_C). Negative shift_y translates content UP.
    """
    h, w = img.shape[:2]
    shift_y = -int(severity * h * 0.4)  # negative = content moves up = face moves up
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    return _warp_with_flat_backdrop(
        img,
        lambda src, mode, val: cv2.warpAffine(
            src, M, (w, h), borderMode=mode, borderValue=val,
        ),
        ctx,
    )


def _margin_below(
    img: np.ndarray, severity: float, seed: int, ctx: FaceContext | None = None,
) -> np.ndarray:
    """Shift image DOWN to push face toward bottom edge (degrades MarginBelow) [§7.4.10.4].

    OFIQ measures q_b = (imgH - eyeMidPoint.y) / t. To DEGRADE this scalar
    we must DECREASE q_b, which means moving the face toward the BOTTOM of
    the image (Y_C close to imgH). Positive shift_y translates content DOWN.
    """
    h, w = img.shape[:2]
    shift_y = int(severity * h * 0.4)  # positive = content moves down = face moves down
    M = np.float32([[1, 0, 0], [0, 1, shift_y]])
    return _warp_with_flat_backdrop(
        img,
        lambda src, mode, val: cv2.warpAffine(
            src, M, (w, h), borderMode=mode, borderValue=val,
        ),
        ctx,
    )


# =========================================================================
# RBF warp helper (used by EyesOpen, MouthClosed landmark warps)
# =========================================================================

def _apply_rbf_warp(
    img: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Apply smooth image warp using RBF interpolation.

    Warps img so that src_points move to dst_points, with smooth
    interpolation for surrounding pixels.
    """
    h, w = img.shape[:2]

    # Only use points that actually moved
    diffs = dst_points - src_points
    moved_mask = np.abs(diffs).sum(axis=1) > 0.5
    if not moved_mask.any():
        return img

    # Add corner anchors to keep the warp bounded
    corners = np.array([
        [0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1],
        [w // 2, 0], [w // 2, h - 1], [0, h // 2], [w - 1, h // 2],
    ], dtype=np.float64)

    all_src = np.vstack([src_points, corners])
    all_dst = np.vstack([dst_points, corners])  # corners map to themselves

    # Compute displacement field using RBF
    dx = all_dst[:, 0] - all_src[:, 0]
    dy = all_dst[:, 1] - all_src[:, 1]

    rbf_x = RBFInterpolator(all_src, dx, kernel="thin_plate_spline", smoothing=1.0)
    rbf_y = RBFInterpolator(all_src, dy, kernel="thin_plate_spline", smoothing=1.0)

    # Build remap grid (subsample for performance, then upscale)
    step = 4
    grid_y, grid_x = np.mgrid[0:h:step, 0:w:step]
    grid_pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    disp_x = rbf_x(grid_pts).reshape(grid_y.shape)
    disp_y = rbf_y(grid_pts).reshape(grid_y.shape)

    # Upscale displacement to full resolution
    map_x_small = (grid_x + disp_x).astype(np.float32)
    map_y_small = (grid_y + disp_y).astype(np.float32)

    map_x = cv2.resize(map_x_small, (w, h), interpolation=cv2.INTER_LINEAR)
    map_y = cv2.resize(map_y_small, (w, h), interpolation=cv2.INTER_LINEAR)

    return _warp_with_inpaint(
        img,
        lambda src, mode, val: cv2.remap(
            src, map_x, map_y, cv2.INTER_LINEAR, borderMode=mode, borderValue=val,
        ),
    )


# =========================================================================
# Component Registry
# =========================================================================

COMPONENT_REGISTRY: dict[str, list[ComponentDegradation]] = {}


def _register(component: str, fn, desc: str, srange: str, needs_ctx: bool = False):
    if component not in COMPONENT_REGISTRY:
        COMPONENT_REGISTRY[component] = []
    COMPONENT_REGISTRY[component].append(
        ComponentDegradation(
            ofiq_component=component, function=fn,
            description=desc, severity_range=srange,
            requires_context=needs_ctx,
            standard_refs=STANDARDS_REFS.get(component),
        )
    )


# === Capture-related (OFIQ Report Section 6) ===

# S6.1 Background Uniformity
_register("BackgroundUniformity.scalar", _background_clutter_segmented,
          "Segmented background noise [§7.3.2]", "gradient: 0 -> 200", needs_ctx=True)

# S6.2 Illumination Uniformity
_register("IlluminationUniformity.scalar", _uneven_illumination_roi,
          "ROI illumination asymmetry [§7.3.3]", "factor: 1.0 -> 0.2", needs_ctx=True)

# S6.3 Moments of Luminance Distribution
_register("LuminanceMean.scalar", _darken_face,
          "Face-masked darkening [§7.3.4]", "factor: 1.0 -> 0.15", needs_ctx=True)
_register(
    "LuminanceVariance.scalar", _reduce_luminance_variance_face,
    "Bidirectional face variance perturbation [§7.3.4]",
    "factor: 1x -> 5x (expand) or 1x -> 0.02x (compress)",
    needs_ctx=True,
)

# S6.4 Over- and Under-Exposure Prevention
_register("UnderExposurePrevention.scalar", _darken_face_with_occlusion,
          "Face+occlusion masked under-exposure [§7.3.5]", "factor: 1.0 -> 0.15", needs_ctx=True)
_register("OverExposurePrevention.scalar", _brighten_face,
          "Face-masked over-exposure [§7.3.6]", "factor: 1.0 -> 3.5", needs_ctx=True)

# S6.5 Dynamic Range
_register("DynamicRange.scalar", _reduce_dynamic_range,
          "Dynamic range compression [§7.3.7]", "range: 100% -> 10%")

# S6.6 Sharpness
_register("Sharpness.scalar", _blur,
          "Gaussian blur [§7.3.8]", "sigma: 0.5 -> 10.5")
_register("Sharpness.scalar", _motion_blur,
          "Motion blur [§7.3.8]", "kernel: 3 -> 31px")
_register("Sharpness.scalar", _gaussian_noise,
          "Additive Gaussian noise [§7.3.8]", "sigma: 0 -> 80")

# S6.7 No Compression Artefacts
_register("CompressionArtifacts.scalar", _jpeg_compression,
          "JPEG compression [§7.3.9]", "Q: 100 -> 5")

# S6.8 Natural Colour
_register("NaturalColour.scalar", _color_cast_cielab,
          "CIELAB color shift in ROI zones [§7.3.10]", "shift: 0 -> 60", needs_ctx=True)

# S6.9 Radial Distortion Prevention
_register("RadialDistortion.scalar", _radial_distortion,
          "Barrel distortion [Annex D.2.1]", "k: 0 -> 0.5")

# === Subject-related (OFIQ Report Section 7) ===

# S7.1 Single Face Present -- placeholder (generative, Phase 3)
# Registered below after generative imports

# S7.2 Eyes Open
_register("EyesOpen.scalar", _eyes_close_warp,
          "Landmark-warped eye closure [§7.4.3]", "closure: 0% -> 90%", needs_ctx=True)

# S7.3 Mouth Closed
_register("MouthClosed.scalar", _mouth_open_warp,
          "Landmark-warped mouth opening [§7.4.4]", "opening: 0 -> 0.25t", needs_ctx=True)

# S7.4 Eyes Visible
_register("EyesVisible.scalar", _eye_occlusion_evz,
          "EVZ-targeted eye occlusion [§7.4.5]", "coverage: 0% -> 80%", needs_ctx=True)

# S7.5 Mouth Occlusion Prevention
_register("MouthOcclusionPrevention.scalar", _mouth_occlusion_polygon,
          "Polygon-targeted mouth occlusion [§7.4.6]", "coverage: 0% -> 100%", needs_ctx=True)

# S7.6 Face Occlusion Prevention
_register("FaceOcclusionPrevention.scalar", _face_region_occlusion,
          "Face-masked rectangular occlusion [§7.4.7]", "area: 0% -> 60%", needs_ctx=True)

# S7.7 Inter-Eye Distance
_register("InterEyeDistance.scalar", _reduce_ied,
          "Pad-and-shrink to reduce IED [§7.4.8]", "scale: 1.0 -> 0.3")

# S7.8 Head Size
_register("HeadSize.scalar", _reduce_head_size,
          "Pad-and-shrink to reduce head size [§7.4.9]", "scale: 1.0 -> 0.3")

# === Geometric (OFIQ Report Sections 8+) ===

# Head Pose
_register("HeadPoseYaw.scalar", _yaw_rotation,
          "Perspective yaw rotation [§7.4.11 yaw/pitch/roll]", "squeeze: 0% -> 50%")
_register("HeadPosePitch.scalar", _pitch_tilt,
          "Perspective pitch tilt [§7.4.11 yaw/pitch/roll]", "squeeze: 0% -> 40%")
_register("HeadPoseRoll.scalar", _roll_rotation,
          "In-plane rotation [§7.4.11 yaw/pitch/roll]", "angle: 0 -> +/-30 deg")

# Crop / margins (4 separate directional functions)
_register("LeftwardCropOfTheFaceImage.scalar", _crop_left,
          "Leftward image shift (face toward left edge) [§7.4.10.1]", "shift: 0% -> 40%")
_register("RightwardCropOfTheFaceImage.scalar", _crop_right,
          "Rightward image shift (face toward right edge) [§7.4.10.2]", "shift: 0% -> 40%")
_register("MarginAboveOfTheFaceImage.scalar", _margin_above,
          "Upward image shift (face toward top) [§7.4.10.3]", "shift: 0% -> 40%")
_register("MarginBelowOfTheFaceImage.scalar", _margin_below,
          "Downward image shift (face toward bottom) [§7.4.10.4]", "shift: 0% -> 40%")

# === Generative components (Section 7.1, Section 8) ===

from ofiq_syngen.generative.single_face import insert_second_face
from ofiq_syngen.generative.expression import add_expression
from ofiq_syngen.generative.head_covering import add_head_covering

_register("SingleFacePresent.scalar", insert_second_face,
          "Face insertion via Poisson blending [§7.4.2]",
          "area ratio: 0 -> 0.4", needs_ctx=True)
_register("ExpressionNeutrality.scalar", add_expression,
          "Landmark-warped expression [§7.4.11 yaw/pitch/roll]",
          "displacement: 0 -> 0.15t", needs_ctx=True)
_register("NoHeadCoverings.scalar", add_head_covering,
          "Synthetic hat overlay [§7.4.11 yaw/pitch/roll]",
          "coverage: 0% -> 100%", needs_ctx=True)


def list_supported_components() -> list[str]:
    """Return all OFIQ components that have at least one degradation function."""
    return sorted(COMPONENT_REGISTRY.keys())


def list_all_degradations() -> list[tuple[str, str, str]]:
    """Return (component, description, severity_range) for all registered degradations."""
    result = []
    for comp, degs in sorted(COMPONENT_REGISTRY.items()):
        for d in degs:
            result.append((comp, d.description, d.severity_range))
    return result
