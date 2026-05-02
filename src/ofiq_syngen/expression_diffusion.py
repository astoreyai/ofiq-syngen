"""Stable Diffusion inpainting for photorealistic expression edits (Phase 6).

Inpaints the lower face region with an emotion-conditioned text prompt.
Produces real teeth, dimples, lipstick crinkling, and skin folds that
TPS / 3DMM landmark warps cannot generate.

Heavy: requires torch + diffusers + a Stable Diffusion inpainting
checkpoint (~5GB, downloaded on first use). GPU strongly recommended
(~5s/image on a 24GB card; ~60s/image on CPU).

Usage:
    Set environment variable to opt in:
        OFIQ_SYNGEN_EXPRESSION_METHOD=sd_inpaint

    Optional:
        OFIQ_SYNGEN_SD_MODEL=stabilityai/stable-diffusion-2-inpainting
        OFIQ_SYNGEN_SD_DEVICE=cuda
        OFIQ_SYNGEN_SD_CACHE=~/.cache/ofiq_syngen_sd

Defaults to ``runwayml/stable-diffusion-inpainting`` which is widely
available and well-tuned for portrait edits. ControlNet face conditioning
is documented as a future enhancement; at the moment identity drift is
controlled by (a) restricting the inpaint mask to the lower face only,
(b) using moderate strength (0.4 + severity*0.45), and (c) negative
prompts that ban deformation/identity-shift artifacts.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ofiq_syngen.face_context import FaceContext


_PIPE = None       # cached SD inpaint pipeline
_IP2P_PIPE = None  # cached InstructPix2Pix pipeline
_PIPE_DEVICE: str | None = None


# InstructPix2Pix instructions per emotion. Tuning notes:
# - "Make him/her smile" works well at any strength; SD knows smiling
#   portraits intimately.
# - Frown / surprise instructions push toward theatrical caricature at
#   high strength. Use restrained natural-photography language and a
#   floor on image_guidance to keep the result a real-looking portrait.
# Each entry: (instruction, image_guidance_floor).
# IP2P config per emotion: (instruction, image_guidance_floor at sev=1.0).
# Floors are tuned so sev=1.0 produces a strong-but-natural emotion
# without tipping into caricature. Smile/frown are capped so even at
# severity=1.0 the result looks like a real photograph (no exaggerated
# tears, no clown-mouth smiles). Surprise is handled separately via
# a chained 2-stage edit.
# Per-emotion: (instruction, image_guidance_floor, max_effective_severity).
# max_effective_severity caps the upper end of the user-facing 0..1 scale
# to whatever internal severity produces the best natural-photograph
# result before tipping into caricature. Surprise tips early (open mouth
# + scrunched eyes), so its cap is 0.45 (sev=1.0 user-facing maps to
# IP2P at internal 0.45 strength).
_IP2P_INSTRUCTIONS: dict[str, tuple[str, float, float]] = {
    "smile": (
        "Make her smile naturally with a warm friendly expression",
        1.05,
        0.75,  # natural-smile ceiling; above this -> clown smile
    ),
    "frown": (
        "Make her look sad",
        1.25,
        0.75,  # sad-but-composed ceiling; above this -> sobbing
    ),
    "surprise": (
        # Direct brow-raise prompt without "shock" / "gasp" / "open mouth"
        # words that IP2P interprets as caricature. Empirically tuned via
        # surprise_sweep test: "raise her eyebrows high" reliably lifts
        # brows and slightly parts lips without ever triggering a wide
        # gaping mouth. Floor 1.05 = strong transformation; the prompt
        # itself prevents caricature so cap stays at 1.0.
        "Raise her eyebrows high in surprise",
        1.05,
        1.00,
    ),
}


# Per-emotion config: (positive_prompt, negative_prompt, strength_boost,
# guidance_boost, mask_includes_brows). Strength/guidance boosts let
# stubborn emotions override SD's neutral-portrait prior. The smile
# prompt benefits from SD's strong representation of smiling subjects;
# frown and surprise need more aggressive prompting and parameters.
_EMOTION_PROMPTS: dict[str, tuple[str, str, float, float, bool]] = {
    "smile": (
        "photorealistic close-up portrait of a person with a natural smile, "
        "teeth slightly visible, raised cheek muscles, warm friendly expression, "
        "soft studio lighting, sharp focus, 4k professional photography",
        "blurry, distorted, deformed, ugly, asymmetric, cartoon, illustration, "
        "extra teeth, missing teeth, frown, sad, neutral, closed lips",
        0.0, 0.0, False,
    ),
    "frown": (
        "photorealistic close-up portrait of a person looking deeply unhappy, "
        "mouth corners pulled down sharply, lips pressed together in sadness, "
        "tense jaw, sorrowful expression, on the verge of tears, "
        "natural lighting, sharp focus, 4k professional photography",
        "smile, smiling, happy, joyful, content, pleased, neutral expression, "
        "blurry, distorted, deformed, ugly, asymmetric, cartoon, illustration, "
        "teeth showing, open mouth, calm",
        0.05, 1.0, True,
    ),
    "surprise": (
        "photorealistic close-up portrait of a person reacting in shock, "
        "mouth wide open in O-shape, jaw dropped, eyes wide open, "
        "raised eyebrows, gasping, astonished expression, "
        "natural lighting, sharp focus, 4k professional photography",
        "calm, neutral expression, closed mouth, smile, smiling, happy, "
        "blurry, distorted, deformed, ugly, asymmetric, cartoon, illustration, "
        "relaxed, normal, content",
        0.10, 1.5, True,
    ),
}


def is_sd_available() -> bool:
    """Whether torch + diffusers are importable (model download still needed)."""
    try:
        import torch  # noqa: F401
        import diffusers  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_device() -> str:
    """Resolve compute device: env override, else CUDA if available."""
    env = os.environ.get("OFIQ_SYNGEN_SD_DEVICE")
    if env:
        return env
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_ip2p_pipeline():
    """Lazily load the InstructPix2Pix pipeline (cached singleton)."""
    global _IP2P_PIPE, _PIPE_DEVICE
    if _IP2P_PIPE is not None:
        return _IP2P_PIPE

    import torch
    from diffusers import StableDiffusionInstructPix2PixPipeline

    model_id = os.environ.get(
        "OFIQ_SYNGEN_IP2P_MODEL", "timbrooks/instruct-pix2pix",
    )
    cache = Path(
        os.environ.get(
            "OFIQ_SYNGEN_SD_CACHE",
            str(Path.home() / ".cache" / "ofiq_syngen_sd"),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    _IP2P_PIPE = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=str(cache),
        safety_checker=None, requires_safety_checker=False,
    ).to(device)
    _IP2P_PIPE.set_progress_bar_config(disable=True)
    _PIPE_DEVICE = device
    return _IP2P_PIPE


def render_expression_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    emotion: str,
    severity: float,
    seed: int,
) -> np.ndarray:
    """InstructPix2Pix path: instruction-driven whole-image edit.

    IP2P operates on the entire image (not a mask), so identity drift
    is more pronounced than masked SD inpainting. In exchange, frown and
    surprise produce dramatic, CNN-detectable emotion changes that SD
    inpainting under-delivers on. ``image_guidance_scale`` (lower = more
    transformation) is severity-modulated.
    """
    if emotion not in _IP2P_INSTRUCTIONS:
        raise ValueError(f"Unknown emotion '{emotion}', use one of {list(_IP2P_INSTRUCTIONS)}")

    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)

    instruction, ig_floor, sev_cap = _IP2P_INSTRUCTIONS[emotion]
    # Per-emotion cap: user-facing severity [0, 1] maps to internal
    # [0, sev_cap]. Caps prevent IP2P from tipping into caricature
    # (clown smile, sobbing frown, scream surprise) at high severity.
    sev_eff = min(severity, sev_cap)
    image_guidance = float(np.clip(2.0 - sev_eff * (2.0 - ig_floor), ig_floor, 2.0))
    text_guidance = 7.5 + sev_eff * 1.5

    def _ip2p(pil, instr, ig, tg, s):
        gen = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(s))
        with torch.no_grad():
            return pipe(
                instr, image=pil, num_inference_steps=20,
                image_guidance_scale=ig, guidance_scale=tg,
                generator=gen,
            ).images[0]

    result = _ip2p(img_pil, instruction,
                    image_guidance, text_guidance, seed)

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


def render_side_lighting_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Add directional side lighting via IP2P.

    Real photographic side lighting casts shadows that follow face
    geometry (nose bridge, eye socket, cheekbone). IP2P produces this
    naturally; the linear-gradient fallback can't.

    Empirical sweep showed "Make the left side of her face bright and
    the right side in shadow" at ig=1.30 produces clean directional
    lighting with photographic shadows.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    # "Split lighting" terminology triggers the right photographic
    # interpretation in IP2P -- produces clean half-light/half-shadow
    # with shadows that follow face geometry. The two-side variants
    # randomize left vs right per seed.
    rng = np.random.RandomState(seed)
    side_left = rng.random() < 0.5
    if side_left:
        instruction = (
            "Apply split lighting, the right half of her face in deep shadow"
        )
    else:
        instruction = (
            "Apply split lighting, the left half of her face in deep shadow"
        )

    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)

    # Cap effective severity at 0.6: at higher IP2P strengths the
    # face geometry starts to drift (loses identity) and the shadow
    # tips into "noir scene" territory. User-facing 0..1 maps to
    # internal [0, 0.6] for a clean photographic split lighting range.
    ig_floor = 1.10
    sev_eff = min(severity, 0.6)
    image_guidance = float(np.clip(2.0 - sev_eff * (2.0 - ig_floor), ig_floor, 2.0))
    text_guidance = 7.5 + sev_eff * 1.5

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            instruction, image=img_pil, num_inference_steps=20,
            image_guidance_scale=image_guidance, guidance_scale=text_guidance,
            generator=generator,
        ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # IP2P relights the WHOLE image (hair, background, neck) but OFIQ
    # IlluminationUniformity only measures face cheek ROIs. Composite
    # the IP2P face area over the ORIGINAL image so hair tone, dress,
    # and background stay identical to the source -- only the face
    # luminance gets the directional split.
    if ctx.face_mask is not None:
        # Use full face hull (incl. hair-adjacent) so the lit-vs-shadow
        # transition runs across the natural face boundary; feather wide
        # enough that the composite edge is invisible.
        soft = _ip2p_face_soft_mask(ctx, h, w, sigma_div=18)
        out = (img.astype(np.float32) * (1 - soft)
               + result_bgr.astype(np.float32) * soft)
        return np.clip(out, 0, 255).astype(np.uint8)
    return result_bgr


def _ip2p_face_soft_mask(ctx, h: int, w: int, sigma_div: float = 18.0) -> np.ndarray:
    """Build a feathered face-region alpha mask for compositing IP2P
    output back over the original image."""
    import cv2
    mask = ctx.face_mask.astype(np.float32)
    sigma = max(6.0, ctx.t_metric / sigma_div)
    soft = cv2.GaussianBlur(mask, (0, 0), sigma)
    soft = np.clip(soft / max(soft.max(), 1.0), 0.0, 1.0)
    return soft[..., None]


def render_lip_gap_sd_inpaint(
    img: np.ndarray,
    inner_upper_pts: np.ndarray,
    inner_lower_pts: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Inpaint photographic teeth/dark gap into a TPS-opened mouth.

    The TPS warp opens the lip landmarks geometrically but the gap
    between the upper and lower lips is filled with stretched lip
    texture (no real teeth or dark interior). This helper takes those
    DISPLACED inner-lip landmark positions, builds a tight mask of the
    open mouth region, and SD-inpaints realistic teeth / shadow into
    the gap so the result looks like a real parted-lip photograph.

    Args:
        img: TPS-warped BGR image (mouth already opened geometrically).
        inner_upper_pts: (4, 2) destination positions of ADNet 88-91.
        inner_lower_pts: (4, 2) destination positions of ADNet 92-95.
        seed: deterministic noise seed.

    Returns:
        BGR image with SD-rendered mouth gap composited in.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_pipeline()  # SD inpaint pipeline (not IP2P)
    h, w = img.shape[:2]

    # Build a mask of the open mouth region: convex hull of upper +
    # lower inner-lip landmarks, dilated slightly so SD can render
    # realistic lip-edge transitions to teeth.
    hull_pts = np.vstack([inner_upper_pts, inner_lower_pts]).astype(np.int32)
    if hull_pts.shape[0] < 3:
        return img
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(hull_pts), 255)
    # Modest dilation: too wide blurs the lip edge; too tight gives
    # SD nothing to render. ~8 px lets SD generate teeth detail while
    # leaving the lipstick edge crisp.
    dilate_px = 8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    mask = cv2.dilate(mask, kernel)
    if int(mask.sum()) < 30:
        return img

    target = 512
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(
        (target, target), Image.LANCZOS,
    )
    mask_pil = Image.fromarray(mask).resize((target, target), Image.LANCZOS)

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            prompt=(
                # Focus prompt on what should appear INSIDE the mouth
                # (not on lips -- SD should not render lips in the mask).
                "view of front teeth and dark mouth interior, "
                "individual white teeth visible, gum line, "
                "photorealistic mouth interior, sharp focus, 4k macro"
            ),
            negative_prompt=(
                "closed mouth, sealed lips, lip on lip, smooth pink fill, "
                "distorted, deformed, ugly, extra teeth, missing teeth, "
                "asymmetric, cartoon, stretched, smeared, blurry, "
                "single white blob"
            ),
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=30,
            guidance_scale=9.0,
            strength=0.95,
            generator=generator,
        ).images[0]

    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Feather the composite so the SD output blends invisibly with the
    # surrounding TPS-warped lips.
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 3.0)
    soft = np.clip(soft / 255.0, 0.0, 1.0)[..., None]
    out = img.astype(np.float32) * (1 - soft) + result_bgr.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def render_surgical_mask_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Add a photorealistic surgical face mask via IP2P, blended by severity.

    IP2P generates a real surgical mask covering nose / mouth. We render
    once at full strength and alpha-blend with the original so sev=0 ->
    identity, sev=1.0 -> full mask, with smooth progression that lets
    OFIQ MouthOcclusionPrevention register monotonically.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(
        (target, target), Image.LANCZOS,
    )

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            "Make her wear a surgical face mask",
            image=img_pil,
            num_inference_steps=20,
            image_guidance_scale=1.30,
            guidance_scale=8.0,
            generator=generator,
        ).images[0]

    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Build a soft mask covering nose + mouth + chin so the alpha blend
    # only affects the lower face. Eyes / brows / hair stay original.
    landmarks = ctx.landmarks_98.astype(np.int32)
    lower_face_pts = np.vstack([
        landmarks[4:29],   # jaw contour
        landmarks[51:60],  # nose
    ])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(lower_face_pts), 255)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                              max(4.0, ctx.t_metric / 25))
    soft = np.clip(soft / 255.0, 0.0, 1.0)
    alpha = (soft * severity)[..., None]

    out = (img.astype(np.float32) * (1 - alpha)
           + result_bgr.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_hat_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Add a photorealistic knit beanie hat via IP2P, blended by severity.

    IP2P generates a real beanie covering the top of the head. We
    render once at full strength and alpha-blend over the upper face
    region (forehead and above) so the lower face stays original.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).resize(
        (target, target), Image.LANCZOS,
    )

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            "Make her wear a knit beanie hat",
            image=img_pil,
            num_inference_steps=20,
            image_guidance_scale=1.30,
            guidance_scale=8.0,
            generator=generator,
        ).images[0]

    result_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Build a mask for the upper-head region (above the eye line). This
    # confines the alpha blend to where a hat would actually sit -
    # eyes / nose / mouth / chin stay original.
    landmarks = ctx.landmarks_98.astype(np.int32)
    eye_top_y = int(landmarks[33:51, 1].min())  # top of brow
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:eye_top_y, :] = 255
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0),
                              max(6.0, ctx.t_metric / 18))
    soft = np.clip(soft / 255.0, 0.0, 1.0)
    alpha = (soft * severity)[..., None]

    out = (img.astype(np.float32) * (1 - alpha)
           + result_bgr.astype(np.float32) * alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_second_face_sd(*args, **kwargs):
    """Deprecated: SD inpaint failed to reliably generate a second face
    in side regions of portrait images (ended up extending hair texture
    into the empty area). Use ``render_second_face_library`` instead,
    which inserts a real different-identity face from VGGFace2.
    """
    return render_second_face_library(*args, **kwargs)


# Cached library of "second face" identities (loaded once)
_SECOND_FACE_LIBRARY: list | None = None


def _load_second_face_library() -> list:
    """Load a small set of cropped face images from VGGFace2 to use
    as the inserted second-person face. Each image is a different
    identity. Cached for reuse across calls."""
    global _SECOND_FACE_LIBRARY
    if _SECOND_FACE_LIBRARY is not None:
        return _SECOND_FACE_LIBRARY
    import cv2
    from pathlib import Path
    # The second-face identities used by SingleFacePresent. By default we
    # look up VGGFace2 identities at the path resolved by the env var
    # OFIQ_SYNGEN_SECOND_FACE_DIR. If unset or no faces found, the
    # operator falls back to a flipped crop of the primary face (legacy).
    candidates = [
        "n000001/0001_01.jpg", "n000040/0001_01.jpg", "n000363/0001_01.jpg",
        "n000394/0001_01.jpg", "n000596/0001_01.jpg",
    ]
    import os
    root_env = os.environ.get("OFIQ_SYNGEN_SECOND_FACE_DIR")
    library = []
    if root_env:
        root = Path(root_env)
        for rel in candidates:
            p = root / rel
            if p.exists():
                face_img = cv2.imread(str(p))
                if face_img is not None:
                    library.append(face_img)
    if not library:
        # Fallback: empty list, caller will skip insertion
        library = []
    _SECOND_FACE_LIBRARY = library
    return library


def render_second_face_library(
    img: np.ndarray,
    ctx,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Insert a real different-identity face beside the primary.

    Mode dispatch via env var ``OFIQ_SYNGEN_SINGLEFACE_MODE``:

    - ``clean`` (default): keep original image dimensions, place the
      second face only in available BG region. Preserves primary
      geometry (IED / t_metric / HeadSize untouched). Trade-off: on
      tightly-framed portrait sources the second face may be small.

    - ``visible``: pad the canvas right, render larger second face,
      resize back. The second face is clearly visible but the primary
      gets compressed horizontally (~14% IED change). USE ONLY for
      visualization / demo; contaminates other OFIQ scalars in the
      same sweep.
    """
    import os
    mode = os.environ.get("OFIQ_SYNGEN_SINGLEFACE_MODE", "clean").lower()
    if mode == "visible":
        return _render_second_face_padded(img, ctx, severity, seed)
    return _render_second_face_clean(img, ctx, severity, seed)


def _render_second_face_clean(
    img: np.ndarray,
    ctx,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Clean (geometry-preserving) second-face insertion.

    Finds available BG region (excluding primary face hull + hair),
    inserts a real face crop sized to fit. Primary IED / t_metric /
    HeadSize all preserved within ADNet detection noise.
    """
    import cv2
    from ofiq_syngen.landmark_utils import BISENET_BACKGROUND, BISENET_HAIR

    h, w = img.shape[:2]
    library = _load_second_face_library()
    if not library:
        return img

    rng = np.random.RandomState(seed)
    second_src = library[rng.randint(0, len(library))]

    # Allowed placement = BG ∩ ~(primary face dilated) ∩ ~hair
    if ctx is None or ctx.parsing_map is None or ctx.face_mask is None:
        return img
    bg = cv2.resize(
        (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_NEAREST,
    )
    hair = cv2.resize(
        (ctx.parsing_map == BISENET_HAIR).astype(np.uint8),
        (w, h), interpolation=cv2.INTER_NEAREST,
    )
    forbid = cv2.dilate(
        (ctx.face_mask | hair * 255).astype(np.uint8),
        np.ones((11, 11), np.uint8),
    )
    allowed = bg & (forbid == 0)
    if not allowed.any():
        return img

    # Target second-face size proportional to PRIMARY face area
    primary_area = max(int(ctx.face_mask.sum()), 100)
    target_area = primary_area * severity * 0.4
    target_size = max(20, int(np.sqrt(target_area)))

    # Shrink target until a placement actually fits in the allowed mask
    placement_mask = None
    actual_size = target_size
    for shrink in [1.0, 0.8, 0.6, 0.45, 0.32, 0.22]:
        cand = max(20, int(target_size * shrink))
        eroded = cv2.erode(allowed, np.ones((cand, cand), np.uint8))
        if eroded.any():
            placement_mask = eroded
            actual_size = cand
            break
    if placement_mask is None:
        return img

    # Resize the second-face crop to target size (square)
    second = cv2.resize(
        second_src, (actual_size, actual_size), interpolation=cv2.INTER_AREA,
    )

    # Pick placement: prefer farthest-from-primary point in the allowed
    # mask (stable, deterministic per seed).
    valid = np.argwhere(placement_mask > 0)
    idx = rng.randint(0, len(valid))
    py, px = valid[idx]
    py = max(0, min(py, h - actual_size))
    px = max(0, min(px, w - actual_size))

    # Elliptical mask for seamless clone
    mask = np.zeros((actual_size, actual_size), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (actual_size // 2, actual_size // 2),
        (max(2, actual_size // 2 - 3), max(2, actual_size // 2 - 3)),
        0, 0, 360, 255, -1,
    )
    center = (px + actual_size // 2, py + actual_size // 2)
    try:
        return cv2.seamlessClone(second, img, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        out = img.copy()
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        region = out[py:py + actual_size, px:px + actual_size].astype(np.float32)
        blended = region * (1 - alpha) + second.astype(np.float32) * alpha
        out[py:py + actual_size, px:px + actual_size] = blended.astype(np.uint8)
        return out


def _render_second_face_padded(
    img: np.ndarray,
    ctx,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Padded-canvas second-face insertion (visualization mode).

    Pads the image right by 50% width, places a real face crop in the
    padded region, resizes back to original dimensions. Second face is
    clearly visible but the primary gets compressed horizontally (~14%
    IED change). NOT for calibration sweeps -- contaminates IED /
    t_metric / HeadSize scalars on the primary.
    """
    import cv2
    from ofiq_syngen.landmark_utils import BISENET_BACKGROUND

    h, w = img.shape[:2]
    library = _load_second_face_library()
    if not library:
        return img

    rng = np.random.RandomState(seed)
    second_src = library[rng.randint(0, len(library))]

    # Sample backdrop color
    if ctx is not None and ctx.parsing_map is not None:
        bg_small = (ctx.parsing_map == BISENET_BACKGROUND).astype(np.uint8)
        bg_mask_full = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_NEAREST)
        if bg_mask_full.sum() > 100:
            bg_color = tuple(int(v) for v in img[bg_mask_full > 0].mean(axis=0))
        else:
            bg_color = (180, 180, 180)
    else:
        bg_color = (180, 180, 180)

    pad_w = int(w * 0.50)
    padded_w = w + pad_w
    padded = np.full((h, padded_w, 3), bg_color, dtype=np.uint8)
    padded[:, :w] = img

    s2_h, s2_w = second_src.shape[:2]
    target_w = int(pad_w * (0.40 + severity * 0.55))
    target_w = min(target_w, pad_w - 8)
    target_h = int(target_w * s2_h / s2_w)
    if target_h > h - 16:
        target_h = h - 16
        target_w = int(target_h * s2_w / s2_h)
    second = cv2.resize(second_src, (target_w, target_h), interpolation=cv2.INTER_AREA)

    place_y = max(0, h // 2 - target_h // 2)
    place_x = w + max(0, (pad_w - target_w) // 2)
    place_y_end = min(h, place_y + target_h)
    place_x_end = min(padded_w, place_x + target_w)
    crop_h = place_y_end - place_y
    crop_w = place_x_end - place_x
    if crop_h < 4 or crop_w < 4:
        return img
    second_crop = second[:crop_h, :crop_w]

    mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.ellipse(
        mask, (crop_w // 2, crop_h // 2),
        (max(2, crop_w // 2 - 4), max(2, crop_h // 2 - 4)),
        0, 0, 360, 255, -1,
    )
    center = (place_x + crop_w // 2, place_y + crop_h // 2)
    try:
        composite = cv2.seamlessClone(
            second_crop, padded, mask, center, cv2.NORMAL_CLONE,
        )
    except cv2.error:
        composite = padded.copy()
        alpha = (mask.astype(np.float32) / 255.0)[..., None]
        region = composite[place_y:place_y_end, place_x:place_x_end].astype(np.float32)
        blended = region * (1 - alpha) + second_crop.astype(np.float32) * alpha
        composite[place_y:place_y_end, place_x:place_x_end] = blended.astype(np.uint8)

    return cv2.resize(composite, (w, h), interpolation=cv2.INTER_AREA)


def render_eyes_closed_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Close the subject's eyes via IP2P (photorealistic eyelids).

    Replaces the TPS warp + skin-paint approach which can't generate
    realistic eyelid texture. ``severity`` modulates inpainting strength;
    at sev=1.0 the eyes are clearly closed, OFIQ EyesOpen registers a
    minimum, and identity is preserved.

    Empirical sweep confirmed "Close her eyes" produces reliable closure
    at IG floor 1.10 without face-region distortion.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)

    ig_floor = 1.10
    image_guidance = float(np.clip(2.0 - severity * (2.0 - ig_floor), ig_floor, 2.0))
    text_guidance = 7.5 + severity * 1.5

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            "Close her eyes",
            image=img_pil,
            num_inference_steps=20,
            image_guidance_scale=image_guidance,
            guidance_scale=text_guidance,
            generator=generator,
        ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    return cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)


def render_sunglasses_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Add photorealistic sunglasses via IP2P, blended by severity.

    IP2P produces clean black sunglasses that occlude the eyes. We
    render once at full strength and alpha-blend with the original
    so sev=0 -> identity, sev=1.0 -> full sunglasses, with smooth
    progression that lets OFIQ EyesVisible register monotonically.

    Replaces the procedural sunglasses occluder which produced obvious
    sticker-like ovals.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)

    # Fixed strong inpainting at IG 1.10 -> reliable sunglasses with
    # identity preserved. Severity controls the alpha-blend back to source.
    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            "Make her wear dark sunglasses",
            image=img_pil,
            num_inference_steps=20,
            image_guidance_scale=1.10,
            guidance_scale=8.0,
            generator=generator,
        ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Build a soft mask around the eye region to confine the alpha blend
    # to the eyes only (so the rest of the face stays identical to source).
    landmarks = ctx.landmarks_98.astype(np.int32)
    eye_pts = np.vstack([landmarks[60:76], landmarks[33:51]])  # eyes + brows
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(eye_pts), 255)
    # Dilate so frames have room
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (max(8, int(ctx.t_metric * 0.15)), max(8, int(ctx.t_metric * 0.15))),
    )
    mask = cv2.dilate(mask, kernel)
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(4.0, ctx.t_metric / 25))
    soft = np.clip(soft / 255.0, 0.0, 1.0)
    # Severity modulates the blend strength
    alpha = (soft * severity)[..., None]
    out = img.astype(np.float32) * (1 - alpha) + result_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_hand_occluder_ip2p(
    img: np.ndarray,
    ctx: FaceContext,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Add a photorealistic hand covering part of the face via IP2P.

    Replaces the procedural hand occluder which produced an obvious
    brown blob. IP2P generates a real hand with fingers, nails, and
    natural shadows. Severity alpha-blends back to the source so the
    hand fades in monotonically.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_ip2p_pipeline()
    h, w = img.shape[:2]

    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            "Add her hand covering her face",
            image=img_pil,
            num_inference_steps=20,
            image_guidance_scale=1.10,
            guidance_scale=8.0,
            generator=generator,
        ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Whole-face alpha blend; the hand naturally covers part of the face
    # in IP2P output, so a uniform alpha by severity is sufficient.
    alpha = float(np.clip(severity, 0.0, 1.0))
    out = img.astype(np.float32) * (1 - alpha) + result_bgr.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def render_shrink_sd_background(
    img: np.ndarray,
    new_h: int,
    new_w: int,
    pad_top: int,
    pad_left: int,
    seed: int,
) -> np.ndarray:
    """Embed a shrunken face in a canvas and SD-inpaint the empty border.

    Used by HeadSize / InterEyeDistance to fill the surround with a
    plausible photographic backdrop matching the source image's lighting
    and palette, instead of the stretched Telea inpainting.

    Args:
        img: source BGR image (already at output dimensions).
        new_h, new_w: dimensions of the shrunken face inside the canvas.
        pad_top, pad_left: where the face is embedded.
        seed: deterministic noise seed.

    Returns:
        BGR image with the face at its shrunken position and the
        surrounding region inpainted with photographic background.
    """
    import cv2
    import torch
    from PIL import Image

    pipe = _load_pipeline()  # SD inpaint pipeline
    h, w = img.shape[:2]

    # Build the canvas: source face shrunken and centered. Pre-fill the
    # border with a heavily-blurred version of the source so SD has
    # plausible image content to refine, instead of black (which SD
    # tends to preserve literally).
    small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    blurred_full = cv2.GaussianBlur(img, (0, 0), 30.0)  # heavy blur = bokeh
    canvas = blurred_full.copy()
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = small

    # Mask = the border region (where SD should refine the bokeh)
    mask = np.full((h, w), 255, dtype=np.uint8)
    mask[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = 0
    if int(mask.sum()) == 0:
        return canvas

    target = 512
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    canvas_pil = Image.fromarray(canvas_rgb).resize((target, target), Image.LANCZOS)
    mask_pil = Image.fromarray(mask).resize((target, target), Image.LANCZOS)

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            prompt=(
                "photorealistic blurred indoor portrait background, "
                "soft bokeh, warm ambient lighting, professional photography, "
                "out of focus colorful backdrop"
            ),
            negative_prompt=(
                "person, face, hand, body, multiple people, blank, black, "
                "rectangle, frame, border, sharp focus on background, "
                "distorted, text, watermark"
            ),
            image=canvas_pil,
            mask_image=mask_pil,
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=0.75,  # moderate strength: refine the blurred prefill
            generator=generator,
        ).images[0]

    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    # Composite: keep the shrunken face crisp, use SD output for the border.
    soft_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 8.0)
    soft_mask = np.clip(soft_mask / 255.0, 0.0, 1.0)[..., None]
    # Re-paste the crisp small face at its exact position so feathering
    # from the SD output doesn't blur the face boundary.
    out = result_bgr.astype(np.float32) * soft_mask + canvas.astype(np.float32) * (1 - soft_mask)
    out_u8 = np.clip(out, 0, 255).astype(np.uint8)
    out_u8[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = small
    return out_u8


def _load_pipeline():
    """Lazily load the Stable Diffusion inpainting pipeline (cached singleton)."""
    global _PIPE, _PIPE_DEVICE
    if _PIPE is not None:
        return _PIPE

    import torch
    from diffusers import StableDiffusionInpaintPipeline

    model_id = os.environ.get(
        "OFIQ_SYNGEN_SD_MODEL", "runwayml/stable-diffusion-inpainting",
    )
    cache = Path(
        os.environ.get(
            "OFIQ_SYNGEN_SD_CACHE",
            str(Path.home() / ".cache" / "ofiq_syngen_sd"),
        )
    )
    cache.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    _PIPE = StableDiffusionInpaintPipeline.from_pretrained(
        model_id, torch_dtype=dtype, cache_dir=str(cache),
        safety_checker=None,  # disable for biometric research; not for prod
        requires_safety_checker=False,
    ).to(device)
    _PIPE.set_progress_bar_config(disable=True)
    _PIPE_DEVICE = device
    return _PIPE


def _build_lower_face_mask(
    ctx: FaceContext, h: int, w: int, include_brows: bool = False,
) -> np.ndarray:
    """Build an inpaint mask covering the mouth + lower jaw, optionally brows.

    For smile: lower face only (mouth + jaw). Brows don't change much
    in a smile so we keep them out to preserve identity.

    For frown / surprise: include brow region too (frown furrows brows,
    surprise raises them). Brows are a strong CNN signal that the
    HSEmotion model picks up.

    Mask is always two convex hulls combined (rather than the union
    hull, which would also pull in the bridge of the nose unnecessarily).
    """
    import cv2
    landmarks = ctx.landmarks_98.astype(np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)

    lower_face_pts = np.vstack([
        landmarks[4:29],   # jaw contour
        landmarks[76:88],  # outer mouth
    ])
    cv2.fillConvexPoly(mask, cv2.convexHull(lower_face_pts), 255)

    if include_brows:
        # ADNet brow landmarks: 33-50 (33-41 right, 42-50 left).
        # Extend a bit upward into the forehead so SD can render a
        # furrowed brow / raised brow with skin folds.
        brow_pts = landmarks[33:51].copy()
        brow_pts[:, 1] = np.clip(brow_pts[:, 1] - int(ctx.t_metric * 0.15),
                                  0, h - 1)
        brow_pts = np.vstack([brow_pts, landmarks[33:51]])  # union of orig + raised
        cv2.fillConvexPoly(mask, cv2.convexHull(brow_pts), 255)
    return mask


def render_expression_sd(
    img: np.ndarray,
    ctx: FaceContext,
    emotion: str,
    severity: float,
    seed: int,
) -> np.ndarray:
    """Inpaint the lower face region with the requested emotion.

    Args:
        img: BGR uint8 image.
        ctx: FaceContext (provides landmarks for the inpaint mask).
        emotion: 'smile' | 'frown' | 'surprise'.
        severity: [0, 1] -- maps to inpainting strength (more = more transformation).
        seed: deterministic noise seed.

    Returns:
        BGR uint8 image with the inpainted lower face composited back over
        the original. Forehead, eyes, hair are untouched.
    """
    if emotion not in _EMOTION_PROMPTS:
        raise ValueError(f"Unknown emotion '{emotion}', use one of {list(_EMOTION_PROMPTS)}")

    import cv2
    import torch
    from PIL import Image

    pipe = _load_pipeline()
    h, w = img.shape[:2]

    pos, neg, strength_boost, guidance_boost, mask_includes_brows = (
        _EMOTION_PROMPTS[emotion]
    )

    # Build & feather the inpaint mask (extending into brows for emotions
    # that need that signal: frown, surprise)
    mask = _build_lower_face_mask(ctx, h, w, include_brows=mask_includes_brows)
    feather_sigma = max(4.0, ctx.t_metric / 30.0)
    mask_soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), feather_sigma)
    mask_soft = np.clip(mask_soft, 0.0, 255.0).astype(np.uint8)

    # SD requires 512x512 (or 768x768 for SD2) inputs. Resize and run.
    target = 512
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb).resize((target, target), Image.LANCZOS)
    mask_pil = Image.fromarray(mask_soft).resize((target, target), Image.LANCZOS)

    # Strength: higher severity = more aggressive transformation. Plus a
    # per-emotion boost so stubborn emotions (frown / surprise) clear
    # SD's neutral-portrait prior. Cap at 0.97 so the lower face still
    # resembles the source.
    strength = float(np.clip(0.45 + severity * 0.45 + strength_boost, 0.45, 0.97))
    guidance = 7.5 + guidance_boost

    generator = torch.Generator(device=_PIPE_DEVICE).manual_seed(int(seed))
    with torch.no_grad():
        result = pipe(
            prompt=pos,
            negative_prompt=neg,
            image=img_pil,
            mask_image=mask_pil,
            num_inference_steps=30,
            guidance_scale=guidance,
            strength=strength,
            generator=generator,
        ).images[0]

    # Resize back and feather-composite
    result_rgb = np.array(result)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
    result_bgr = cv2.resize(result_bgr, (w, h), interpolation=cv2.INTER_LANCZOS4)

    soft = mask_soft.astype(np.float32) / 255.0
    soft = soft[..., None]
    out = img.astype(np.float32) * (1 - soft) + result_bgr.astype(np.float32) * soft
    return np.clip(out, 0, 255).astype(np.uint8)
