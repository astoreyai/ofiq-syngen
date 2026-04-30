# ofiq-syngen v0.2 — Usage Guide

ISO/IEC 29794-5 component-aligned synthetic face image quality degradation pipeline.
Each degradation targets the exact region and property that OFIQ measures, using
OFIQ's own ONNX models for face analysis.

---

## Installation

```bash
pip install -e packages/ofiq-syngen           # core (opencv, numpy, onnxruntime, scipy)
pip install -e "packages/ofiq-syngen[gpu]"    # GPU acceleration for ONNX models
pip install -e "packages/ofiq-syngen[generative]"  # diffusion/InsightFace for 3 generative components
```

**OFIQ models required** for context-aware degradations (15 of 28 components):
```
<OFIQ_MODEL_DIR>/
  face_landmark_estimation/ADNet.onnx
  face_parsing/bisenet_400.onnx
  face_occlusion_segmentation/face_occlusion_segmentation_ort.onnx
  head_pose_estimation/mb1_120x120.onnx
```

Model resolution order:
1. Explicit `model_dir` argument to `OFIQModels(model_dir=...)`
2. `OFIQ_MODEL_DIR` environment variable
3. Auto-detect: `~/OFIQ-Project/data/models/`, `/opt/ofiq/models/`, `/usr/local/share/ofiq/models/`

Context-free components (13 of 28) work without any models installed.

---

## Quick Start

```python
from ofiq_syngen import DegradationPipeline, DegradationConfig

pipeline = DegradationPipeline()
image = cv2.imread("face.jpg")  # BGR uint8

# Apply a single degradation
degraded, meta = pipeline.degrade_single(image, "Sharpness.scalar", severity=0.6)

# Sweep severity levels
results = pipeline.degrade_sweep(image, "Sharpness.scalar")  # returns [(img, meta), ...]

# All 28 components at once
all_results = pipeline.degrade_all_components(image, severity=0.5)
```

---

## Architecture

```
image (BGR uint8)
  │
  ├── FaceContext.from_image(image, models)     # runs once per image
  │     ├── ADNet → 98 landmarks
  │     ├── BiSeNet → 19-class parsing map (400×400)
  │     ├── OcclusionSeg → binary mask (1=visible)
  │     ├── HeadPose3DDFAV2 → (yaw, pitch, roll)
  │     └── derived: IED, t-metric, face_mask, ROIs, EVZs, luminance
  │
  └── DegradationPipeline.degrade_single(image, component, severity, ctx=ctx)
        ├── context-free functions: blur, JPEG, rotation, crop shifts
        └── context-requiring functions: face-masked darkening, landmark warps,
            segmentation-guided noise, EVZ occlusion, polygon fills
```

---

## Component Reference

### 28 Registered Components

| # | Component | ISO | Algorithm | Context? |
|---|-----------|-----|-----------|----------|
| 1 | `BackgroundUniformity.scalar` | S6.1 | Structured noise in BiSeNet background mask, eroded 4×4 | Yes |
| 2 | `IlluminationUniformity.scalar` | S6.2 | Darken one of two landmark-derived ROI zones | Yes |
| 3 | `LuminanceMean.scalar` | S6.3 | Multiplicative darkening within face mask | Yes |
| 4 | `LuminanceVariance.scalar` | S6.3 | Per-channel compression toward face-region mean | Yes |
| 5 | `UnderExposurePrevention.scalar` | S6.4 | Darkening within face mask ∩ occlusion mask | Yes |
| 6 | `OverExposurePrevention.scalar` | S6.4 | Brightening within face mask | Yes |
| 7 | `DynamicRange.scalar` | S6.5 | Pixel compression toward mid-gray (128) | No |
| 8 | `Sharpness.scalar` | S6.6 | Gaussian blur / motion blur / Gaussian noise (3 variants) | No |
| 9 | `CompressionArtifacts.scalar` | S6.7 | JPEG re-encoding at quality 5–100 | No |
| 10 | `NaturalColour.scalar` | S6.8 | CIELAB a\*/b\* shift in landmark-derived ROI zones | Yes |
| 11 | `RadialDistortion.scalar` | S6.9 | Barrel distortion via radial coefficient | No |
| 12 | `SingleFacePresent.scalar` | S7.1 | Poisson-blended face insertion in background | Yes |
| 13 | `EyesOpen.scalar` | S7.2 | RBF warp: upper eyelid → lower (pairs 61/67, 62/66, 63/65) | Yes |
| 14 | `MouthClosed.scalar` | S7.3 | RBF warp: inner lip landmarks apart (pairs 89/95, 90/94, 91/93) | Yes |
| 15 | `EyesVisible.scalar` | S7.4 | Dark occlusion within EVZ rectangles (IED/20 expansion) | Yes |
| 16 | `MouthOcclusionPrevention.scalar` | S7.5 | Mask-colored fill within mouth polygon (landmarks 76–87) | Yes |
| 17 | `FaceOcclusionPrevention.scalar` | S7.6 | Random rectangle within face mask convex hull | Yes |
| 18 | `InterEyeDistance.scalar` | S7.7 | Pad-and-shrink (face smaller in frame) | No |
| 19 | `HeadSize.scalar` | S7.8 | Pad-and-shrink (same mechanism as IED) | No |
| 20 | `HeadPoseYaw.scalar` | S8 | Perspective warp (horizontal foreshortening) | No |
| 21 | `HeadPosePitch.scalar` | S8 | Perspective warp (vertical foreshortening) | No |
| 22 | `HeadPoseRoll.scalar` | S8 | In-plane affine rotation ±30° | No |
| 23 | `LeftwardCropOfTheFaceImage.scalar` | S8 | Rightward image shift only | No |
| 24 | `RightwardCropOfTheFaceImage.scalar` | S8 | Leftward image shift only | No |
| 25 | `MarginAboveOfTheFaceImage.scalar` | S8 | Downward image shift only | No |
| 26 | `MarginBelowOfTheFaceImage.scalar` | S8 | Upward image shift only | No |
| 27 | `ExpressionNeutrality.scalar` | S8 | Landmark RBF warp (smile/surprise/frown) | Yes |
| 28 | `NoHeadCoverings.scalar` | S8 | Fabric-textured hat overlay on forehead | Yes |

---

## API Reference

### `DegradationPipeline`

```python
class DegradationPipeline:
    def __init__(
        self,
        config: DegradationConfig | None = None,
        models: OFIQModels | None = None,      # pass for context-aware degradations
    )

    def degrade_single(
        self,
        image: np.ndarray,                      # BGR uint8
        component: str,                         # e.g. "Sharpness.scalar"
        severity: float,                        # [0, 1]
        degradation_index: int = 0,             # which variant (Sharpness has 3)
        seed: int | None = None,                # reproducibility
        ctx: FaceContext | None = None,          # pre-built context (avoids re-inference)
    ) -> tuple[np.ndarray, dict]

    def degrade_sweep(
        self,
        image: np.ndarray,
        component: str,
        seed: int | None = None,
    ) -> list[tuple[np.ndarray, dict]]          # one per severity level

    def degrade_all_components(
        self,
        image: np.ndarray,
        severity: float = 0.5,
        seed: int | None = None,
    ) -> list[tuple[np.ndarray, dict]]          # builds FaceContext once, reuses

    def generate_dataset(
        self,
        image_dir: Path,
        output_dir: Path,
        max_images: int = 500,
        components: list[str] | None = None,    # default: all 28
    ) -> pd.DataFrame                           # manifest with subject_id, severity, etc.

    @staticmethod
    def build_influence_matrix(
        manifest: pd.DataFrame,
        ofiq_scores: pd.DataFrame,
        ofiq_scalar_cols: list[str],
    ) -> pd.DataFrame                           # (n_degradation_types × n_ofiq_components)
```

### `DegradationConfig`

```python
@dataclass
class DegradationConfig:
    severity_levels: list[float] = [0.2, 0.4, 0.6, 0.8, 1.0]
    seed: int = 42
    output_format: str = "jpg"
    jpg_quality: int = 95
```

### `FaceContext`

```python
@dataclass
class FaceContext:
    image: np.ndarray                   # original BGR
    is_aligned: bool                    # True if 616×616 OFIQ-aligned
    landmarks_98: np.ndarray            # (98, 2) int32, ADNet
    parsing_map: np.ndarray             # (400, 400) uint8, BiSeNet classes 0–18
    occlusion_mask: np.ndarray          # image-sized uint8, 1=visible 0=occluded
    head_pose: tuple[float,float,float] # (yaw, pitch, roll) degrees

    # Derived (auto-computed):
    left_eye_center: tuple[float, float]
    right_eye_center: tuple[float, float]
    eye_midpoint: tuple[float, float]
    chin: tuple[float, float]
    ied: float                          # yaw-corrected inter-eye distance
    t_metric: float                     # eye-midpoint to chin distance
    eye_mouth_dist: float
    face_mask: np.ndarray               # convex hull uint8
    luminance: np.ndarray               # rec709 luminance uint8
    left_roi: tuple[int,int,int,int]    # (x, y, w, h)
    right_roi: tuple[int,int,int,int]
    left_evz: tuple[int,int,int,int]    # Eye Visibility Zone
    right_evz: tuple[int,int,int,int]

    @classmethod
    def from_image(cls, image, models=None, is_aligned=None) -> FaceContext
```

### `OFIQModels`

```python
class OFIQModels:
    def __init__(self, model_dir: str | Path | None = None)
    # Resolution: explicit model_dir > OFIQ_MODEL_DIR env > auto-detect (~/OFIQ-Project, /opt/ofiq)

    adnet: ort.InferenceSession       # ADNet 98-pt landmarks
    bisenet: ort.InferenceSession     # BiSeNet face parsing
    occlusion: ort.InferenceSession   # binary occlusion mask
    headpose: ort.InferenceSession    # HeadPose3DDFAV2

# Module-level singleton:
from ofiq_syngen.models import get_models
models = get_models()  # lazy-loaded, cached
```

### `ComponentDegradation`

```python
@dataclass
class ComponentDegradation:
    ofiq_component: str         # e.g. "Sharpness.scalar"
    function: Callable          # fn(img, severity, seed, ctx) -> img
    description: str            # e.g. "Gaussian blur [S6.6]"
    severity_range: str         # e.g. "sigma: 0.5 -> 10.5"
    requires_context: bool      # True if function needs FaceContext
```

### Landmark Utilities

```python
from ofiq_syngen.landmark_utils import (
    # ADNet 98-point index maps
    LEFT_EYE,           # [60..67]
    RIGHT_EYE,          # [68..75]
    LEFT_EYE_CORNERS,   # [60, 64]
    RIGHT_EYE_CORNERS,  # [68, 72]
    MOUTH_OUTER,        # [76..87]
    MOUTH_INNER,        # [88..95]
    CONTOUR,            # [0..32]
    CHIN,               # [16]
    NOSETIP,            # [54]

    # Landmark pairs for openness/closure measurement
    PAIRS_LEFT_EYE,     # [(61,67), (62,66), (63,65)]
    PAIRS_RIGHT_EYE,    # [(69,75), (70,74), (71,73)]
    PAIRS_MOUTH_INNER,  # [(89,95), (90,94), (91,93)]

    # BiSeNet class indices
    BISENET_BACKGROUND, # 0
    BISENET_SKIN,       # 1
    BISENET_CLOTH,      # 16
    BISENET_HAIR,       # 17
    BISENET_HAT,        # 18

    # Geometry functions (exact OFIQ ports)
    get_distance,               # euclidean distance
    get_middle,                 # mean of landmark points
    get_max_pair_distance,      # max distance among landmark pairs
    calculate_eye_centers,      # midpoint of eye corners
    tmetric,                    # eye-midpoint to chin distance
    inter_eye_distance,         # yaw-corrected IED
    calculate_reference_points, # (leftEye, rightEye, IED, eyeMouthDist)
    calculate_roi,              # left/right ROI zones for NaturalColour/IllumUniformity
    compute_evz_rects,          # Eye Visibility Zone rectangles
    get_face_mask,              # convex hull from 98 landmarks
    get_luminance_image,        # rec709 linearized luminance (NOT cv2 grayscale)
    convert_bgr_to_cielab,      # OFIQ's custom sRGB->CIELAB (NOT cv2.cvtColor)
)
```

---

## Usage Patterns

### Pattern 1: Context-free (no OFIQ models)

For components that don't need face analysis (blur, compression, rotation, crops):

```python
from ofiq_syngen import DegradationPipeline

pipeline = DegradationPipeline()
degraded, meta = pipeline.degrade_single(image, "Sharpness.scalar", 0.5)
```

### Pattern 2: Full OFIQ alignment (with models)

For maximum fidelity — every degradation targets the exact OFIQ region:

```python
from ofiq_syngen import DegradationPipeline
from ofiq_syngen.models import get_models

pipeline = DegradationPipeline(models=get_models())

# Pipeline builds FaceContext automatically when needed
degraded, meta = pipeline.degrade_single(image, "EyesOpen.scalar", 0.5)
```

### Pattern 3: Batch processing (shared context)

Build FaceContext once, reuse for all components on the same image:

```python
from ofiq_syngen.face_context import FaceContext
from ofiq_syngen.models import get_models

models = get_models()
ctx = FaceContext.from_image(image, models)

pipeline = DegradationPipeline(models=models)
for comp in pipeline.supported:
    degraded, meta = pipeline.degrade_single(image, comp, 0.5, ctx=ctx)
```

### Pattern 4: Dataset generation

```python
pipeline = DegradationPipeline()
manifest = pipeline.generate_dataset(
    image_dir=Path("data/faces"),
    output_dir=Path("data/degraded"),
    max_images=500,
    components=["Sharpness.scalar", "EyesOpen.scalar"],  # or None for all
)
# manifest is a DataFrame: subject_id, original_image, degraded_image, severity, ...
```

---

## CLI

```bash
# List all components
ofiq-syngen list-components

# Apply single degradation
ofiq-syngen degrade -c Sharpness -s 0.6 -o output.jpg input.jpg

# Sweep severity levels
ofiq-syngen sweep -c Sharpness -n 10 -o ./output/ input.jpg

# Generate batch dataset
ofiq-syngen generate-dataset -i ./faces/ -o ./degraded/ -n 100
```

### Standards presets

The CLI accepts `--preset` to filter components by standards-conformance
profile.

```bash
# List the components in each preset
ofiq-syngen list-components --preset icao-strict     # 22 components
ofiq-syngen list-components --preset iso-19794-5     # 28 components
ofiq-syngen list-components --preset iso-29794-5     # 28 components

# Print the multi-standard cross-reference table
ofiq-syngen show-standards
ofiq-syngen show-standards --preset icao-strict

# Generate a dataset filtered to the ICAO-strict subset
ofiq-syngen generate-dataset -i ./faces -o ./icao_dataset --preset icao-strict
```

`--components` and `--preset` are mutually exclusive on `generate-dataset`;
passing both returns a non-zero exit with a clear error.

Preset definitions live in `src/ofiq_syngen/standards.py` and trace back
to [`docs/standards/MAPPING.csv`](https://github.com/astoreyai/ofiq-syngen/blob/main/docs/standards/MAPPING.csv). The
machine-readable CSV is the source of truth; `standards.py` ships the
same data as a Python dict so the package wheel needs no file-resource
handling. The test suite asserts the two stay in sync.

### Programmatic standards access

```python
from ofiq_syngen.standards import (
    STANDARDS_REFS,
    ICAO_STRICT_COMPONENTS,
    get_refs,
    components_by_alignment,
    components_by_confidence,
)

refs = get_refs("Sharpness.scalar")
print(refs.ofiq_section, refs.iso_29794_5, refs.icao_9303, refs.alignment)
# S6.6  Sharpness (focus)  3.2.3 focus  exact

# All components with exact alignment to ICAO 9303 §3.2.3
print(components_by_alignment("exact"))

# All components flagged 'uncertain' (clause numbers need verification
# against the OFIQ Algorithm Book PDF before publication)
print(components_by_confidence("uncertain"))
```

The `ComponentDegradation` dataclass returned by the registry also carries
the full `standard_refs` field for in-pipeline access.

---

## OFIQ source provenance

Every degradation function cites the OFIQ-Project C++ `Execute()` line
range it was ported from. See [`docs/standards/PROVENANCE.md`](https://github.com/astoreyai/ofiq-syngen/blob/main/docs/standards/PROVENANCE.md)
for the full per-component table plus helper-utility ports
(`landmark_utils`, `image_utils`, `face_context`).

---

## File Structure

```
packages/ofiq-syngen/src/ofiq_syngen/
  __init__.py              # Package entry point, exports
  components.py            # 28 degradation functions + registry
  pipeline.py              # DegradationPipeline orchestration
  face_context.py          # FaceContext (runs OFIQ ONNX models)
  models.py                # OFIQModels singleton loader
  landmark_utils.py        # ADNet index maps, geometry, CIELAB, luminance
  cli.py                   # Command-line interface
  generative/
    __init__.py
    single_face.py         # SingleFacePresent (Poisson blend)
    expression.py          # ExpressionNeutrality (landmark warp)
    head_covering.py       # NoHeadCoverings (fabric overlay)

```

---

## Name Mapping (syngen ↔ OFIQ ↔ world model)

Syngen uses verbose ISO names for 4 crop/margin components. The world model's
`constants.py` provides bidirectional mapping:

| syngen registry key | OFIQ_COMPONENTS name | Mapping function |
|---|---|---|
| `LeftwardCropOfTheFaceImage.scalar` | `CropLeft` | `syngen_to_ofiq()` |
| `RightwardCropOfTheFaceImage.scalar` | `CropRight` | `syngen_to_ofiq()` |
| `MarginAboveOfTheFaceImage.scalar` | `MarginTop` | `syngen_to_ofiq()` |
| `MarginBelowOfTheFaceImage.scalar` | `MarginBottom` | `syngen_to_ofiq()` |
| `RadialDistortion.scalar` | *(not in OFIQ_COMPONENTS)* | In `SYNGEN_EXCLUDED` |
| All other components | Same name (minus `.scalar`) | Identity |

```python
# In world model code (src/constants.py):
from constants import syngen_to_ofiq, ofiq_to_syngen, SYNGEN_EXCLUDED

syngen_to_ofiq("LeftwardCropOfTheFaceImage.scalar")  # → "CropLeft"
ofiq_to_syngen("CropLeft")                           # → "CropLeft.scalar"
```

When iterating syngen components against the world model's 27-slot quality vector:
```python
from ofiq_syngen.components import COMPONENT_REGISTRY
from constants import OFIQ_COMPONENTS, syngen_to_ofiq, SYNGEN_EXCLUDED

for comp in COMPONENT_REGISTRY:
    ofiq_name = syngen_to_ofiq(comp)
    if ofiq_name in SYNGEN_EXCLUDED:
        continue  # RadialDistortion has no OFIQ slot
    idx = OFIQ_COMPONENTS.index(ofiq_name)  # position in 27-d quality vector
```

---

## OFIQ Model Preprocessing (Exact Specs)

For developers extending the pipeline — exact preprocessing to match OFIQ C++:

### ADNet (landmarks)
- Input: face image (616×616 aligned or bbox-cropped)
- Normalize: `2/255 × pixel − 1` (range [−1, 1])
- Layout: HWC → CHW
- Output: `(landmark + 1) / 2 × 255`, scale by `face_height / 256`

### BiSeNet (face parsing)
- Input: aligned face, cropped [0:h−60, 30:w−30]
- Convert: BGR → RGB
- Normalize: `(pixel − mean×255) / (std×255)`, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]
- Resize: 400×400
- Output: argmax across 19 channels → class IDs (0=bg, 1=skin, ..., 16=cloth, 18=hat)

### Occlusion Segmentation
- Input: aligned face, cropped 96px all sides
- Resize: 224×224
- Normalize: `blobFromImage(scale=1/255, swapRB=True)`
- Output: multiply by −1, threshold at 0 → binary mask (1=visible)

### HeadPose3DDFAV2
- Input: face bbox crop (center ± 0.44h/0.51h, square)
- Resize: 120×120
- Normalize: `(pixel − 127.5) / 128`
- Output: 7 params → denormalize with paramMean/paramStd → rotation matrix → Euler angles
