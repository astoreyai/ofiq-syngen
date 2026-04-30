"""ISO-Component-Aligned Synthetic Degradation Pipeline.

Handles the entanglement problem: a single degradation (e.g., blur) affects
multiple OFIQ components simultaneously. The pipeline addresses this by:

1. Measuring the FULL OFIQ delta (all 27 components) for each degradation
2. Building a degradation->component influence matrix from empirical data
3. Using this matrix to weight training signal appropriately

This means we don't pretend blur only affects Sharpness. We measure what it
ACTUALLY changes across ALL components, and use the real multi-component
delta as the training signal.

The influence matrix W[d,c] = mean(|delta_OFIQ_c|) when degradation d is applied
at severity 0.5. This is learned from data, not assumed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ofiq_syngen.components import (
    COMPONENT_REGISTRY,
    list_supported_components,
)


@dataclass
class DegradationConfig:
    """Configuration for the degradation pipeline."""

    severity_levels: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    seed: int = 42
    output_format: str = "jpg"
    jpg_quality: int = 95


@dataclass
class DegradationResult:
    """Result of a single degradation application."""

    original_path: Path
    degraded_path: Path
    degradation_type: str
    target_component: str
    severity: float
    params: dict[str, Any]


class DegradationPipeline:
    """Reproducible, ISO-component-aligned degradation with entanglement tracking.

    When OFIQ models are available, builds a FaceContext once per image and
    passes it to degradation functions that require face analysis (landmarks,
    parsing, occlusion mask, head pose). Functions that don't need context
    (blur, compression, rotation) run without it.
    """

    def __init__(
        self,
        config: DegradationConfig | None = None,
        models: Any | None = None,
    ) -> None:
        """
        Args:
            config: Pipeline configuration.
            models: OFIQModels instance for face analysis. If None, attempts
                lazy loading when context-requiring functions are called.
        """
        self.config = config or DegradationConfig()
        self.supported = list_supported_components()
        self._models = models

    def _get_models(self):
        """Lazy-load OFIQModels if needed."""
        if self._models is None:
            try:
                from ofiq_syngen.models import get_models
                self._models = get_models()
            except (ImportError, FileNotFoundError):
                return None
        return self._models

    def _build_context(self, image: np.ndarray):
        """Build FaceContext for an image (returns None if models unavailable)."""
        models = self._get_models()
        if models is None:
            return None
        try:
            from ofiq_syngen.face_context import FaceContext
            return FaceContext.from_image(image, models)
        except Exception:
            return None

    def degrade_single(
        self,
        image: np.ndarray,
        component: str,
        severity: float,
        degradation_index: int = 0,
        seed: int | None = None,
        ctx: Any | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Apply a single degradation targeting a specific OFIQ component.

        Args:
            image: BGR uint8 face image.
            component: OFIQ component name (e.g., "Sharpness.scalar").
            severity: [0, 1] degradation strength.
            degradation_index: which degradation to use if component has multiple.
            seed: random seed for reproducibility.
            ctx: Pre-built FaceContext (avoids redundant model inference
                when calling degrade_single multiple times on the same image).

        Returns:
            (degraded_image, metadata_dict)
        """
        if component not in COMPONENT_REGISTRY:
            raise ValueError(
                f"No degradation for component '{component}'. "
                f"Supported: {self.supported}"
            )

        degs = COMPONENT_REGISTRY[component]
        deg = degs[degradation_index % len(degs)]
        s = seed if seed is not None else self.config.seed

        # Build context if needed and not provided
        if deg.requires_context and ctx is None:
            ctx = self._build_context(image)

        degraded = deg.function(image, severity, s, ctx)

        metadata = {
            "target_component": component,
            "degradation_type": deg.description,
            "severity": severity,
            "seed": s,
        }

        return degraded, metadata

    def degrade_all_components(
        self,
        image: np.ndarray,
        severity: float = 0.5,
        seed: int | None = None,
    ) -> list[tuple[np.ndarray, dict]]:
        """Apply one degradation per supported component.

        Builds FaceContext ONCE and reuses it for all components.
        """
        results = []
        s = seed if seed is not None else self.config.seed

        # Build context once for the image
        any_needs_ctx = any(
            degs[0].requires_context
            for degs in COMPONENT_REGISTRY.values()
        )
        ctx = self._build_context(image) if any_needs_ctx else None

        for component in self.supported:
            degraded, meta = self.degrade_single(
                image, component, severity, seed=s, ctx=ctx,
            )
            results.append((degraded, meta))

        return results

    def degrade_sweep(
        self,
        image: np.ndarray,
        component: str,
        seed: int | None = None,
    ) -> list[tuple[np.ndarray, dict]]:
        """Sweep all severity levels for one component.

        Builds FaceContext once and reuses across severity levels.
        """
        results = []
        s = seed if seed is not None else self.config.seed

        degs = COMPONENT_REGISTRY.get(component, [])
        needs_ctx = degs[0].requires_context if degs else False
        ctx = self._build_context(image) if needs_ctx else None

        for sev in self.config.severity_levels:
            degraded, meta = self.degrade_single(
                image, component, sev, seed=s, ctx=ctx,
            )
            results.append((degraded, meta))

        return results

    def generate_dataset(
        self,
        image_dir: Path,
        output_dir: Path,
        max_images: int = 500,
        components: list[str] | None = None,
    ):
        """Generate a full degradation dataset.

        For each image x component x severity:
        1. Build FaceContext once per image
        2. Apply degradation
        3. Save degraded image
        4. Record metadata

        The OFIQ extraction is done separately (by ofiq_batch.py) to allow
        parallel processing.

        Returns:
            pandas.DataFrame with columns: subject_id, original_image,
            degraded_image, degradation_type, target_component, severity, seed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for generate_dataset(). "
                "Install it with: pip install ofiq-syngen[pandas]"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        comps = components or self.supported

        images = sorted(image_dir.glob("*.jpg"))[:max_images]
        if not images:
            for subdir in sorted(image_dir.iterdir())[:max_images]:
                if subdir.is_dir():
                    jpgs = list(subdir.glob("*.jpg"))
                    if jpgs:
                        images.append(jpgs[0])

        records = []
        for img_idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            subject_id = img_path.parent.name if img_path.parent != image_dir else img_path.stem

            # Build context once per image
            ctx = self._build_context(img)

            # Save original
            orig_name = f"{subject_id}_{img_path.stem}__original.jpg"
            cv2.imwrite(str(output_dir / orig_name), img,
                        [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpg_quality])
            records.append({
                "subject_id": subject_id,
                "original_image": img_path.name,
                "degraded_image": orig_name,
                "degradation_type": "original",
                "target_component": "none",
                "severity": 0.0,
                "seed": self.config.seed,
            })

            for comp in comps:
                for sev in self.config.severity_levels:
                    degraded, meta = self.degrade_single(
                        img, comp, sev, seed=self.config.seed + img_idx, ctx=ctx,
                    )
                    deg_name = f"{subject_id}_{img_path.stem}__{comp.replace('.scalar', '')}_{sev:.1f}.jpg"
                    cv2.imwrite(str(output_dir / deg_name), degraded,
                                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.jpg_quality])

                    records.append({
                        "subject_id": subject_id,
                        "original_image": img_path.name,
                        "degraded_image": deg_name,
                        "degradation_type": meta["degradation_type"],
                        "target_component": comp,
                        "severity": sev,
                        "seed": meta["seed"],
                    })

        return pd.DataFrame(records)

    @staticmethod
    def build_influence_matrix(manifest, ofiq_scores, ofiq_scalar_cols: list[str]):
        """Build the degradation->component influence matrix from empirical OFIQ data.

        For each degradation type, measures how much EACH OFIQ component changes
        (not just the target).
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for build_influence_matrix(). "
                "Install it with: pip install ofiq-syngen[pandas]"
            )

        ofiq_idx = ofiq_scores.set_index("image_name")

        originals = manifest[manifest["degradation_type"] == "original"]
        orig_scores = {}
        for _, row in originals.iterrows():
            if row["degraded_image"] in ofiq_idx.index:
                orig_scores[row["subject_id"]] = ofiq_idx.loc[
                    row["degraded_image"], ofiq_scalar_cols
                ].values.astype(np.float32)

        degraded = manifest[manifest["degradation_type"] != "original"]
        influence_data = []

        for _, row in degraded.iterrows():
            if row["subject_id"] not in orig_scores:
                continue
            if row["degraded_image"] not in ofiq_idx.index:
                continue

            orig = orig_scores[row["subject_id"]]
            deg = ofiq_idx.loc[row["degraded_image"], ofiq_scalar_cols].values.astype(np.float32)
            delta = deg - orig

            influence_data.append({
                "target_component": row["target_component"],
                "severity": row["severity"],
                **{col: delta[i] for i, col in enumerate(ofiq_scalar_cols)},
            })

        if not influence_data:
            return pd.DataFrame()

        influence_df = pd.DataFrame(influence_data)

        matrix = influence_df.groupby("target_component")[ofiq_scalar_cols].apply(
            lambda x: x.abs().mean()
        )

        return matrix
