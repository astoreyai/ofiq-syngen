"""Photorealistic occluder rendering for ofiq-syngen Phase 3.

Procedural renderings of common face occluders. All routines:
- Operate on the input image without external assets
- Use FaceContext landmarks to position the occluder
- Apply alpha-feathered compositing for natural edges
- Are deterministic given (severity, seed)
- Produce output that OFIQ's segmentation/CNN models will classify as
  the relevant occlusion class

Functions:
    render_sunglasses    - dark elliptical lenses + frame for EyesVisible
    render_surgical_mask - pleated fabric mask for MouthOcclusionPrevention
    render_hand_occluder - skin-tone silhouette for FaceOcclusionPrevention
    render_hat           - fabric hat with brim shadow for NoHeadCoverings
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


# =============================================================================
# Shared helpers
# =============================================================================

def _feather(mask: np.ndarray, sigma: float | None = None) -> np.ndarray:
    h, w = mask.shape[:2]
    if sigma is None:
        sigma = max(1.5, min(h, w) / 80)
    return np.clip(
        cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 0.0, 1.0
    )


def _composite(base: np.ndarray, overlay: np.ndarray,
               soft_mask: np.ndarray) -> np.ndarray:
    a = soft_mask[..., None] if soft_mask.ndim == 2 else soft_mask
    out = base.astype(np.float32) * (1 - a) + overlay.astype(np.float32) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def _drop_shadow(img: np.ndarray, mask: np.ndarray,
                 offset: tuple[int, int] = (0, 4),
                 darken: float = 0.5,
                 blur_sigma: float = 5.0) -> np.ndarray:
    """Cast a soft drop shadow under an occluder."""
    h, w = img.shape[:2]
    shadow = np.zeros_like(mask, dtype=np.float32)
    sx, sy = offset
    y0, y1 = max(0, sy), min(h, h + sy)
    x0, x1 = max(0, sx), min(w, w + sx)
    src_y0, src_y1 = max(0, -sy), min(h, h - sy)
    src_x0, src_x1 = max(0, -sx), min(w, w - sx)
    shadow[y0:y1, x0:x1] = mask[src_y0:src_y1, src_x0:src_x1].astype(np.float32)
    shadow = cv2.GaussianBlur(shadow, (0, 0), blur_sigma)
    shadow = np.clip(shadow * darken, 0, 1)
    out = img.astype(np.float32) * (1 - shadow[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)


# =============================================================================
# Sunglasses (EyesVisible §7.4.5)
# =============================================================================

def render_sunglasses(img: np.ndarray, ctx: FaceContext, severity: float,
                      seed: int) -> np.ndarray:
    """Render realistic sunglasses over the Eye Visibility Zones.

    At severity=0.0 returns the input unchanged (identity).

    Pipeline:
      1. Build elliptical lens shapes matching EVZ aspect ratio
      2. Lens fill: dark RGB with subtle vertical gradient
      3. Specular highlight: small bright Gaussian in upper-left of each lens
      4. Frame: 2-3 px dark border around lens
      5. Bridge: thin line connecting the two lenses
      6. Drop shadow under glasses
      7. Composite with feathered alpha (severity controls opacity)
    """
    if severity < 0.01:
        return img
    h, w = img.shape[:2]
    out = img.copy().astype(np.float32)
    lens_mask = np.zeros((h, w), dtype=np.uint8)
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    centers = []
    for evz in (ctx.left_evz, ctx.right_evz):
        x, y, ew, eh = evz
        cx = x + ew // 2
        cy = y + eh // 2
        # Slightly larger ellipse than EVZ for natural look
        rx = int(ew * 0.65)
        ry = int(eh * 0.85)
        centers.append((cx, cy, rx, ry))

        # Lens fill with vertical gradient (darker top)
        for dy in range(-ry, ry + 1):
            t = (dy + ry) / (2 * ry + 1)  # 0 (top) -> 1 (bottom)
            lens_color = np.array([
                15 + t * 18,   # B: 15-33
                18 + t * 22,   # G: 18-40
                22 + t * 26,   # R: 22-48
            ])
            for dx in range(-rx, rx + 1):
                if (dx / rx) ** 2 + (dy / ry) ** 2 > 1.0:
                    continue
                px, py = cx + dx, cy + dy
                if 0 <= px < w and 0 <= py < h:
                    overlay[py, px] = lens_color
                    lens_mask[py, px] = 1

        # Specular highlight
        hx, hy = cx - int(rx * 0.4), cy - int(ry * 0.5)
        hr = max(2, int(min(rx, ry) * 0.25))
        for dy in range(-hr, hr + 1):
            for dx in range(-hr, hr + 1):
                d2 = (dx / hr) ** 2 + (dy / hr) ** 2
                if d2 > 1.0:
                    continue
                falloff = (1 - d2) ** 2
                px, py = hx + dx, hy + dy
                if 0 <= px < w and 0 <= py < h and lens_mask[py, px]:
                    overlay[py, px] += np.array([180, 200, 220]) * falloff

    overlay = np.clip(overlay, 0, 255)

    # Frame: dilate lens mask, subtract original lens area to get ring
    frame_kernel = np.ones((3, 3), np.uint8)
    frame_dilated = cv2.dilate(lens_mask, frame_kernel, iterations=2)
    frame_mask = (frame_dilated.astype(np.int16) - lens_mask.astype(np.int16)) > 0
    overlay[frame_mask] = np.array([15, 15, 18])  # dark frame
    full_mask = (lens_mask | frame_mask.astype(np.uint8))

    # Bridge between lenses
    if len(centers) == 2:
        (lx, ly, lrx, _), (rx_, ry_, rrx, _) = centers
        # Determine which is left in image
        if lx > rx_:
            lx, ly, rx_, ry_ = rx_, ry_, lx, ly
        bridge_y = (ly + ry_) // 2
        bridge_thick = 2
        for bx in range(lx + lrx, rx_ - rrx + 1):
            for byo in range(-bridge_thick, bridge_thick + 1):
                py = bridge_y + byo
                if 0 <= py < h and 0 <= bx < w:
                    overlay[py, bx] = np.array([15, 15, 18])
                    full_mask[py, bx] = 1

    # Apply drop shadow under glasses
    out = _drop_shadow(out.astype(np.uint8), full_mask, offset=(0, 6),
                       darken=0.35, blur_sigma=6.0).astype(np.float32)

    # Composite with feathered alpha; severity controls opacity (0.6 - 0.97)
    soft = _feather(full_mask, sigma=1.5)
    opacity = 0.6 + severity * 0.37
    soft = soft * opacity
    return _composite(out.astype(np.uint8), overlay.astype(np.uint8), soft)


# =============================================================================
# Surgical mask (MouthOcclusionPrevention §7.4.6)
# =============================================================================

def render_surgical_mask(img: np.ndarray, ctx: FaceContext, severity: float,
                         seed: int) -> np.ndarray:
    """Render realistic surgical mask conforming to face contour.

    At severity=0.0 returns the input unchanged (identity).

    Pipeline:
      1. Determine mask region: top edge follows nose-bridge landmark,
         bottom edge follows chin contour, sides follow jaw landmarks
      2. Procedural fabric color (light blue with subtle variation)
      3. Horizontal pleating (3-4 fold lines via bright/dark gradients)
      4. Drop shadow under mask edge on jaw
      5. Severity controls coverage extent: low = mouth only;
         high = full nose-to-chin
    """
    if severity < 0.01:
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    landmarks = ctx.landmarks_98

    # Mouth corners (76, 82), nose tip (54), chin (16)
    nose_tip = landmarks[54]
    chin = landmarks[16]
    mouth_left = landmarks[76]
    mouth_right = landmarks[82]
    eye_mouth_dist = math.hypot(
        ctx.eye_midpoint[0] - (mouth_left[0] + mouth_right[0]) / 2,
        ctx.eye_midpoint[1] - (mouth_left[1] + mouth_right[1]) / 2,
    )

    # v0.5.1: redesigned mask polygon. v0.5.0 produced a "paper
    # airplane" wedge — narrow at the chin, peaked at the nose-
    # bridge — because the polygon clipped to landmarks within
    # [top_y, bottom_y] and added a sin-peak at the top. Real
    # surgical masks are shaped like a rounded rectangle that
    # extends past the jaw line at the cheeks. Build that shape
    # parametrically from nose / chin / mouth-corner anchors.
    top_y = int(nose_tip[1] - eye_mouth_dist * 0.30 * severity)
    bottom_y = int(chin[1] + eye_mouth_dist * 0.20)
    mid_y = (top_y + bottom_y) // 2
    # Width: extend past mouth corners by 60% of eye-mouth distance
    extend = int(eye_mouth_dist * 0.6)
    left_x = int(mouth_left[0] - extend)
    right_x = int(mouth_right[0] + extend)

    # Build a rounded-rectangle polygon. Use 60 points around the
    # perimeter for a smooth curve.
    pts = []
    # Top edge with subtle nose-bridge indent (a small downward
    # bump at center, NOT the v0.5.0 peak which made the mask look
    # pointed).
    for i in range(20):
        t = i / 19.0
        x = int(left_x + (right_x - left_x) * t)
        # Faint downward bump where the nose-bridge wire would sit
        bump = -math.sin(t * math.pi) * eye_mouth_dist * 0.02
        y = int(top_y - bump)
        pts.append([x, y])
    # Right cheek curve (rounds outward then down)
    for i in range(10):
        t = i / 9.0
        # Bezier-like outward bulge: x extends by 10% of half-width,
        # y descends from top to mid.
        x = int(right_x + math.sin(t * math.pi / 2) * eye_mouth_dist * 0.08)
        y = int(top_y + (mid_y - top_y) * t)
        pts.append([x, y])
    # Right cheek to chin curve
    for i in range(10):
        t = i / 9.0
        x = int(right_x + math.sin(math.pi / 2) * eye_mouth_dist * 0.08
                - math.sin(t * math.pi / 2) * eye_mouth_dist * 0.08
                - (right_x - chin[0]) * t * 0.4)
        y = int(mid_y + (bottom_y - mid_y) * t)
        pts.append([x, y])
    # Bottom curve under chin
    for i in range(20):
        t = i / 19.0
        # Mirror x from right-of-chin to left-of-chin, with a
        # slight downward bow under the chin.
        x_left_anchor = int(left_x + (chin[0] - left_x) * 0.4)
        x_right_anchor = int(right_x - (right_x - chin[0]) * 0.4)
        x = int(x_right_anchor + (x_left_anchor - x_right_anchor) * t)
        bow = math.sin(t * math.pi) * eye_mouth_dist * 0.03
        y = int(bottom_y + bow)
        pts.append([x, y])
    # Left chin to cheek
    for i in range(10):
        t = i / 9.0
        x_left_anchor = int(left_x + (chin[0] - left_x) * 0.4)
        x = int(x_left_anchor - (x_left_anchor - left_x) * t)
        y = int(bottom_y - (bottom_y - mid_y) * t)
        pts.append([x, y])
    # Left cheek up to top
    for i in range(10):
        t = i / 9.0
        x = int(left_x + math.sin((1 - t) * math.pi / 2) * eye_mouth_dist * 0.08)
        y = int(mid_y - (mid_y - top_y) * t)
        pts.append([x, y])

    poly = np.array(pts, dtype=np.int32)
    mask_2d = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask_2d, [poly], 1)

    # Procedural surgical-mask fabric color
    base_color = np.array([225, 220, 210])  # light blue-ish white
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:] = base_color

    # Add fabric texture noise
    fabric_noise = rng.randn(h, w).astype(np.float32) * 6
    for c in range(3):
        overlay[:, :, c] += fabric_noise

    # Add horizontal pleat folds (3 dark lines)
    pleat_band_h = bottom_y - top_y
    if pleat_band_h > 0:
        for i in range(1, 4):
            pleat_y = top_y + int(pleat_band_h * i / 4)
            for dy in range(-2, 3):
                py = pleat_y + dy
                if 0 <= py < h:
                    falloff = max(0, 1 - abs(dy) / 2.5)
                    darken = 18 * falloff
                    overlay[py, :, :] -= darken

    overlay = np.clip(overlay, 0, 255)

    # Drop shadow under mask
    base_shadow = _drop_shadow(img, mask_2d, offset=(0, 8),
                               darken=0.25, blur_sigma=8.0)

    # Composite with feathered edges (severity controls opacity)
    soft = _feather(mask_2d, sigma=2.5)
    soft = soft * (0.7 + severity * 0.3)
    return _composite(base_shadow, overlay.astype(np.uint8), soft)


# =============================================================================
# Hand occluder (FaceOcclusionPrevention §7.4.7)
# =============================================================================

def render_hand_occluder(img: np.ndarray, ctx: FaceContext, severity: float,
                         seed: int) -> np.ndarray:
    """Render skin-toned hand silhouette occluding part of the face.

    At severity=0.0 returns the input unchanged (identity).

    Pipeline:
      1. Sample skin tone from face mask (so hand matches subject)
      2. Build hand silhouette via ellipses (palm + 4-finger blob)
      3. Position to overlap face region; size scales with severity
      4. Drop shadow on face under hand
      5. Composite with feathered edges
    """
    if severity < 0.01:
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]

    # Sample skin tone from face mask (mean BGR)
    if ctx.face_mask is not None and ctx.face_mask.sum() > 0:
        face_pixels = img[ctx.face_mask > 0]
        skin = face_pixels.mean(axis=0)
        # Slightly desaturate (hand often differs from face skin tone)
        skin_hand = skin * 0.92 + 8
    else:
        skin_hand = np.array([180, 200, 220])

    # Hand size relative to face
    face_w = ctx.ied * 4 if ctx.ied > 0 else w * 0.4
    hand_w = int(face_w * (0.4 + severity * 0.5))
    hand_h = int(hand_w * 0.7)

    # Place hand: bottom of face, covering chin/mouth
    chin = ctx.landmarks_98[16]
    cx = int(chin[0] + rng.randint(-10, 10))
    cy = int(chin[1] - hand_h // 3)

    mask = np.zeros((h, w), dtype=np.uint8)
    # Palm: large ellipse
    cv2.ellipse(mask, (cx, cy), (hand_w // 2, hand_h // 2),
                rng.uniform(-15, 15), 0, 360, 1, -1)
    # Fingers: 4 small ellipses on top
    finger_y = cy - hand_h // 2
    for i in range(4):
        fx = cx - hand_w // 3 + i * hand_w // 5
        fr = max(2, hand_w // 12)
        finger_h = int(hand_h * 0.4)
        cv2.ellipse(mask, (fx, finger_y - finger_h // 2),
                    (fr, finger_h // 2), 0, 0, 360, 1, -1)
    # Thumb: smaller ellipse offset
    thumb_x = cx + hand_w // 2 - hand_w // 8
    thumb_y = cy - hand_h // 4
    cv2.ellipse(mask, (thumb_x, thumb_y),
                (hand_w // 10, hand_h // 5), -45, 0, 360, 1, -1)

    # Build overlay: skin tone with subtle texture
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:] = skin_hand
    skin_noise = rng.randn(h, w).astype(np.float32) * 4
    for c in range(3):
        overlay[:, :, c] += skin_noise

    # Subtle palm shading: darker at edges, lighter at center
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
        for c in range(3):
            overlay[:, :, c] += dist_norm * 12

    overlay = np.clip(overlay, 0, 255)

    # Drop shadow on face under hand
    base = _drop_shadow(img, mask, offset=(2, 6), darken=0.3, blur_sigma=5.0)

    # Composite with feathered edges
    soft = _feather(mask, sigma=2.0)
    return _composite(base, overlay.astype(np.uint8), soft)


# =============================================================================
# Hat (NoHeadCoverings §7.4.13)
# =============================================================================

def render_hat(img: np.ndarray, ctx: FaceContext, severity: float,
               seed: int) -> np.ndarray:
    """Render realistic hat (beanie / cap) on top of head.

    At severity=0.0 returns the input unchanged (identity).

    Pipeline:
      1. Determine head-top region from face contour landmarks
      2. Build dome shape conforming to forehead curvature
      3. Pick hat color from BiSeNet-trained palette (dark fabric tones)
      4. Procedural fabric texture (Perlin-like noise)
      5. Add brim/edge shadow on forehead below hat
      6. Composite with feathered edges
      7. Severity controls vertical coverage extent
    """
    if severity < 0.01:
        return img
    rng = np.random.RandomState(seed)
    h, w = img.shape[:2]
    landmarks = ctx.landmarks_98

    # Eyebrow-top y from landmarks 33-46
    brow_y = int(landmarks[33:47, 1].min())
    # Face contour width at eyebrow level
    contour_xs = landmarks[0:33, 0]
    left_x = int(contour_xs.min() - 8)
    right_x = int(contour_xs.max() + 8)
    head_w = right_x - left_x

    # Hat extends UPWARD from above the brows
    hat_top_y = max(0, int(brow_y - ctx.t_metric * (0.3 + severity * 0.5)))
    hat_bottom_y = int(brow_y - 2)  # just above brows

    # Build dome mask (semi-elliptical)
    mask = np.zeros((h, w), dtype=np.uint8)
    cx = (left_x + right_x) // 2
    rx = head_w // 2 + 5
    ry = hat_bottom_y - hat_top_y
    cv2.ellipse(mask, (cx, hat_bottom_y), (rx, ry), 0, 180, 360, 1, -1)
    # Bottom edge: rectangle from hat_bottom_y down a few px
    cv2.rectangle(mask, (left_x, hat_bottom_y - 4), (right_x, hat_bottom_y), 1, -1)

    # Hat colors (BiSeNet-trained recognizable as cloth/hat)
    palette = [
        (35, 35, 38),     # black beanie
        (40, 35, 60),     # navy
        (50, 60, 70),     # dark gray
        (45, 65, 90),     # brown
        (35, 50, 35),     # forest green
        (60, 55, 55),     # charcoal
    ]
    base_color = np.array(palette[rng.randint(0, len(palette))])

    # Build overlay with fabric texture
    overlay = np.zeros((h, w, 3), dtype=np.float32)
    overlay[:] = base_color

    # Procedural fabric noise (multi-scale perlin-ish via filtered random)
    coarse_noise = cv2.GaussianBlur(
        rng.randn(h, w).astype(np.float32) * 12, (0, 0), 4
    )
    fine_noise = rng.randn(h, w).astype(np.float32) * 5
    fabric_tex = coarse_noise + fine_noise
    for c in range(3):
        overlay[:, :, c] += fabric_tex

    # Subtle vertical shading (darker at brim)
    band_h = hat_bottom_y - hat_top_y
    if band_h > 0:
        for y in range(hat_top_y, hat_bottom_y + 1):
            t = (y - hat_top_y) / max(1, band_h)
            shade = -10 + t * 20  # lighter top, darker bottom
            overlay[y, :, :] += shade

    overlay = np.clip(overlay, 0, 255)

    # Brim shadow on forehead
    base_shadow = _drop_shadow(img, mask, offset=(0, 5),
                               darken=0.4, blur_sigma=6.0)

    # Composite with feathered edges
    soft = _feather(mask, sigma=2.5)
    soft = soft * (0.85 + severity * 0.13)
    return _composite(base_shadow, overlay.astype(np.uint8), soft)
