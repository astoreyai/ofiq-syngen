"""Basic usage examples for ofiq-syngen v0.2.

Demonstrates:
1. Listing all 28 registered components
2. Applying a single degradation (no OFIQ models needed)
3. Using FaceContext for OFIQ-aligned degradation (with OFIQ models)
4. Sweeping severity levels
5. Batch dataset generation
"""


import cv2
import numpy as np

from ofiq_syngen import (
    DegradationConfig,
    DegradationPipeline,
    list_all_degradations,
    list_supported_components,
)


def create_synthetic_face(size: int = 224) -> np.ndarray:
    """Create a simple synthetic face image for demonstration.

    In practice, use real aligned face images (e.g., from CelebA, LFW).
    """
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 2 - 20),
                0, 0, 360, (180, 150, 130), -1)
    eye_y = size // 2 - size // 8
    cv2.circle(img, (size // 2 - size // 6, eye_y), 8, (40, 30, 20), -1)
    cv2.circle(img, (size // 2 + size // 6, eye_y), 8, (40, 30, 20), -1)
    mouth_y = size // 2 + size // 6
    cv2.ellipse(img, (size // 2, mouth_y), (20, 8), 0, 0, 180, (100, 60, 60), 2)
    return img


def example_list_components():
    """Example 1: Inspect the component registry."""
    print("=== Example 1: Component Registry ===")

    components = list_supported_components()
    print(f"  Supported components: {len(components)}/28")

    all_degs = list_all_degradations()
    print(f"  Total degradation functions: {len(all_degs)}")

    # Show context requirements
    from ofiq_syngen.components import COMPONENT_REGISTRY
    ctx_count = sum(1 for c in components if COMPONENT_REGISTRY[c][0].requires_context)
    print(f"  Context-requiring: {ctx_count} (use OFIQ models for face analysis)")
    print(f"  Context-free: {len(components) - ctx_count} (work without models)")
    print()

    for comp, desc, srange in all_degs[:8]:
        print(f"  {comp:45s} {desc:45s} {srange}")
    print(f"  ... and {len(all_degs) - 8} more\n")


def example_single_degradation():
    """Example 2: Apply a single degradation (no OFIQ models needed)."""
    print("=== Example 2: Single Degradation (context-free) ===")

    pipeline = DegradationPipeline()
    image = create_synthetic_face()

    # Context-free components work without OFIQ models
    degraded, metadata = pipeline.degrade_single(
        image, component="Sharpness.scalar", severity=0.6
    )

    print(f"  Component: {metadata['target_component']}")
    print(f"  Degradation: {metadata['degradation_type']}")
    print(f"  Severity: {metadata['severity']}")
    print(f"  Shape: {image.shape} -> {degraded.shape}")
    print()


def example_with_face_context():
    """Example 3: OFIQ-aligned degradation using FaceContext.

    This requires OFIQ ONNX models to be available. FaceContext runs
    ADNet landmarks, BiSeNet parsing, occlusion segmentation, and
    HeadPose3DDFAV2 once per image. Each context-requiring degradation
    then targets the exact region OFIQ would analyze.
    """
    print("=== Example 3: OFIQ-Aligned Degradation (with FaceContext) ===")

    try:
        from ofiq_syngen.models import get_models
        from ofiq_syngen.face_context import FaceContext

        models = get_models()
        pipeline = DegradationPipeline(models=models)
        image = create_synthetic_face()

        # Build FaceContext once, reuse for all degradations
        ctx = FaceContext.from_image(image, models)
        print(f"  Landmarks: {ctx.landmarks_98.shape}")
        print(f"  Parsing classes: {np.unique(ctx.parsing_map)}")
        print(f"  Head pose: yaw={ctx.head_pose[0]:.1f}, pitch={ctx.head_pose[1]:.1f}")
        print(f"  IED: {ctx.ied:.1f}, t-metric: {ctx.t_metric:.1f}")

        # Context-requiring components now use exact OFIQ regions
        for comp in ["EyesOpen.scalar", "MouthClosed.scalar", "NaturalColour.scalar"]:
            degraded, meta = pipeline.degrade_single(image, comp, 0.6, ctx=ctx)
            delta = np.abs(degraded.astype(float) - image.astype(float)).mean()
            print(f"  {comp}: delta={delta:.1f}")

    except (ImportError, FileNotFoundError) as e:
        print(f"  Skipped (OFIQ models not available): {e}")
    print()


def example_severity_sweep():
    """Example 4: Sweep severity levels for one component."""
    print("=== Example 4: Severity Sweep ===")

    config = DegradationConfig(
        severity_levels=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        seed=42,
    )
    pipeline = DegradationPipeline(config)
    image = create_synthetic_face()

    results = pipeline.degrade_sweep(image, "CompressionArtifacts.scalar")

    for degraded, meta in results:
        delta = np.abs(degraded.astype(float) - image.astype(float)).mean()
        print(f"  severity={meta['severity']:.1f}  delta={delta:.1f}")

    print(f"  {len(results)} severity levels\n")


def example_all_components():
    """Example 5: Apply one degradation per component."""
    print("=== Example 5: All 28 Components ===")

    pipeline = DegradationPipeline()
    image = create_synthetic_face()

    results = pipeline.degrade_all_components(image, severity=0.5)
    print(f"  Generated {len(results)} degraded images")

    for degraded, meta in results:
        comp = meta["target_component"].replace(".scalar", "")
        delta = np.abs(degraded.astype(float) - image.astype(float)).mean()
        print(f"  {comp:40s} delta={delta:.1f}")
    print()


if __name__ == "__main__":
    example_list_components()
    example_single_degradation()
    example_with_face_context()
    example_severity_sweep()
    example_all_components()
    print("Done.")
