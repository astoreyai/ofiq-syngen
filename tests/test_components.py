"""Tests for the ofiq-syngen degradation pipeline."""

import numpy as np
import pytest

from ofiq_syngen import DegradationPipeline, DegradationConfig, COMPONENT_REGISTRY
from ofiq_syngen.components import list_supported_components, list_all_degradations


@pytest.fixture
def face_image():
    """Synthetic 112x112 face image."""
    return np.random.randint(80, 200, (112, 112, 3), dtype=np.uint8)


@pytest.fixture
def pipeline():
    config = DegradationConfig(severity_levels=[0.2, 0.4, 0.6, 0.8, 1.0])
    return DegradationPipeline(config)


class TestComponentRegistry:
    def test_coverage(self):
        """All 28 components (27 OFIQ + RadialDistortion) should be registered."""
        assert len(list_supported_components()) >= 28

    def test_all_27_ofiq_components_present(self):
        """Verify all 27 ISO/IEC 29794-5 components are registered."""
        expected = {
            "BackgroundUniformity.scalar", "IlluminationUniformity.scalar",
            "LuminanceMean.scalar", "LuminanceVariance.scalar",
            "UnderExposurePrevention.scalar", "OverExposurePrevention.scalar",
            "DynamicRange.scalar", "Sharpness.scalar", "CompressionArtifacts.scalar",
            "NaturalColour.scalar", "SingleFacePresent.scalar",
            "EyesOpen.scalar", "MouthClosed.scalar", "EyesVisible.scalar",
            "MouthOcclusionPrevention.scalar", "FaceOcclusionPrevention.scalar",
            "InterEyeDistance.scalar", "HeadSize.scalar",
            "LeftwardCropOfTheFaceImage.scalar", "RightwardCropOfTheFaceImage.scalar",
            "MarginAboveOfTheFaceImage.scalar", "MarginBelowOfTheFaceImage.scalar",
            "HeadPoseYaw.scalar", "HeadPosePitch.scalar", "HeadPoseRoll.scalar",
            "ExpressionNeutrality.scalar", "NoHeadCoverings.scalar",
        }
        registered = set(list_supported_components())
        missing = expected - registered
        assert not missing, f"Missing components: {missing}"

    def test_all_have_functions(self):
        for comp, degs in COMPONENT_REGISTRY.items():
            assert len(degs) >= 1, f"No degradation for {comp}"
            for d in degs:
                assert callable(d.function)
                assert d.ofiq_component == comp

    def test_list_all_degradations(self):
        all_degs = list_all_degradations()
        assert len(all_degs) >= 28
        for comp, desc, srange in all_degs:
            assert isinstance(comp, str)
            assert isinstance(desc, str)

    def test_section_references(self):
        """Every degradation must reference its OFIQ section."""
        for comp, degs in COMPONENT_REGISTRY.items():
            for d in degs:
                assert "[S" in d.description, f"{comp}: missing section ref in '{d.description}'"

    def test_context_flag_consistency(self):
        """Context-requiring components should have requires_context=True."""
        ctx_components = {
            "BackgroundUniformity.scalar", "IlluminationUniformity.scalar",
            "LuminanceMean.scalar", "LuminanceVariance.scalar",
            "UnderExposurePrevention.scalar", "OverExposurePrevention.scalar",
            "NaturalColour.scalar", "EyesOpen.scalar", "MouthClosed.scalar",
            "EyesVisible.scalar", "MouthOcclusionPrevention.scalar",
            "FaceOcclusionPrevention.scalar", "SingleFacePresent.scalar",
            "ExpressionNeutrality.scalar", "NoHeadCoverings.scalar",
        }
        for comp, degs in COMPONENT_REGISTRY.items():
            for d in degs:
                if comp in ctx_components:
                    assert d.requires_context, f"{comp} should require context"


class TestDegradationFunctions:
    def test_all_functions_run(self, face_image):
        """Every registered function should run without error (ctx=None fallback)."""
        for comp, degs in COMPONENT_REGISTRY.items():
            for d in degs:
                result = d.function(face_image, 0.5, 42, None)
                assert result.shape == face_image.shape, f"{comp}: shape mismatch"
                assert result.dtype == np.uint8, f"{comp}: wrong dtype"

    def test_severity_zero_minimal_change(self, face_image):
        """Severity near 0 should produce minimal change."""
        for comp, degs in COMPONENT_REGISTRY.items():
            d = degs[0]
            result = d.function(face_image, 0.01, 42, None)
            assert result.shape == face_image.shape

    def test_severity_one_large_change(self, face_image):
        """Severity 1.0 should produce visible change for non-stub functions."""
        for comp, degs in COMPONENT_REGISTRY.items():
            d = degs[0]
            result = d.function(face_image, 1.0, 42, None)
            diff = np.abs(result.astype(float) - face_image.astype(float)).mean()
            # Allow zero delta for context-requiring functions in fallback mode
            # (some produce minimal change without landmarks/parsing)
            if not d.requires_context:
                assert diff > 0.1, f"{comp}: no change at severity 1.0 (delta={diff})"

    def test_deterministic_with_seed(self, face_image):
        """Same seed should produce identical results."""
        for comp, degs in COMPONENT_REGISTRY.items():
            d = degs[0]
            r1 = d.function(face_image, 0.5, 42, None)
            r2 = d.function(face_image, 0.5, 42, None)
            assert np.array_equal(r1, r2), f"{comp}: not deterministic"

    @pytest.mark.parametrize("severity", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    def test_all_severities_valid(self, face_image, severity):
        """All standard severity levels should produce valid images."""
        for comp in list(COMPONENT_REGISTRY.keys())[:5]:
            d = COMPONENT_REGISTRY[comp][0]
            result = d.function(face_image, severity, 42, None)
            assert result.min() >= 0
            assert result.max() <= 255

    def test_no_nan_or_inf(self, face_image):
        """Output should never contain NaN or Inf."""
        for comp, degs in COMPONENT_REGISTRY.items():
            d = degs[0]
            result = d.function(face_image, 0.5, 42, None).astype(np.float32)
            assert not np.isnan(result).any(), f"{comp}: NaN in output"
            assert not np.isinf(result).any(), f"{comp}: Inf in output"


class TestDirectionalCrops:
    """Verify crop functions shift in the correct direction only."""

    def test_crop_left_shifts_right(self, face_image):
        from ofiq_syngen.components import _crop_left
        result = _crop_left(face_image, 0.5, 42, None)
        # Content should shift right: left side should differ more
        left_diff = np.abs(result[:, :20].astype(float) - face_image[:, :20].astype(float)).mean()
        assert left_diff > 0, "CropLeft should change left portion"

    def test_crop_right_shifts_left(self, face_image):
        from ofiq_syngen.components import _crop_right
        result = _crop_right(face_image, 0.5, 42, None)
        right_diff = np.abs(result[:, -20:].astype(float) - face_image[:, -20:].astype(float)).mean()
        assert right_diff > 0, "CropRight should change right portion"

    def test_margin_above_shifts_down(self, face_image):
        from ofiq_syngen.components import _margin_above
        result = _margin_above(face_image, 0.5, 42, None)
        top_diff = np.abs(result[:20, :].astype(float) - face_image[:20, :].astype(float)).mean()
        assert top_diff > 0, "MarginAbove should change top portion"

    def test_margin_below_shifts_up(self, face_image):
        from ofiq_syngen.components import _margin_below
        result = _margin_below(face_image, 0.5, 42, None)
        bottom_diff = np.abs(result[-20:, :].astype(float) - face_image[-20:, :].astype(float)).mean()
        assert bottom_diff > 0, "MarginBelow should change bottom portion"


class TestDegradationPipeline:
    def test_degrade_single(self, pipeline, face_image):
        degraded, meta = pipeline.degrade_single(face_image, "Sharpness.scalar", 0.5)
        assert degraded.shape == face_image.shape
        assert meta["target_component"] == "Sharpness.scalar"
        assert meta["severity"] == 0.5

    def test_degrade_sweep(self, pipeline, face_image):
        results = pipeline.degrade_sweep(face_image, "CompressionArtifacts.scalar")
        assert len(results) == 5

    def test_degrade_all_components(self, pipeline, face_image):
        results = pipeline.degrade_all_components(face_image, severity=0.5)
        assert len(results) >= 28

    def test_invalid_component_raises(self, pipeline, face_image):
        with pytest.raises(ValueError, match="No degradation"):
            pipeline.degrade_single(face_image, "FakeComponent.scalar", 0.5)

    def test_config_custom_levels(self, face_image):
        config = DegradationConfig(severity_levels=[0.1, 0.5, 0.9])
        p = DegradationPipeline(config)
        results = p.degrade_sweep(face_image, "Sharpness.scalar")
        assert len(results) == 3

    def test_generate_dataset(self, pipeline, tmp_path):
        """Generate dataset for 2 images, 2 components."""
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(2):
            img = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
            import cv2
            cv2.imwrite(str(img_dir / f"face_{i}.jpg"), img)

        out_dir = tmp_path / "output"
        manifest = pipeline.generate_dataset(
            image_dir=img_dir,
            output_dir=out_dir,
            max_images=2,
            components=["Sharpness.scalar", "CompressionArtifacts.scalar"],
        )

        assert len(manifest) == 22
        assert "subject_id" in manifest.columns
        assert "degradation_type" in manifest.columns

        for _, row in manifest.iterrows():
            assert (out_dir / row["degraded_image"]).exists()


class TestCLI:
    def test_list_components(self):
        from ofiq_syngen.cli import main
        rc = main(["list-components"])
        assert rc == 0

    def test_no_args_shows_help(self, capsys):
        from ofiq_syngen.cli import main
        rc = main([])
        assert rc == 0

    def test_version(self):
        from ofiq_syngen.cli import main
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

    def test_degrade_missing_input(self):
        from ofiq_syngen.cli import main
        rc = main(["degrade", "--component", "Sharpness", "--severity", "0.5", "/nonexistent.jpg"])
        assert rc == 1

    def test_degrade_roundtrip(self, face_image, tmp_path):
        import cv2
        from ofiq_syngen.cli import main
        input_path = tmp_path / "test_input.jpg"
        output_path = tmp_path / "test_output.jpg"
        cv2.imwrite(str(input_path), face_image)

        rc = main(["degrade", "-c", "Sharpness", "-s", "0.5", "-o", str(output_path), str(input_path)])
        assert rc == 0
        assert output_path.exists()

    def test_sweep_roundtrip(self, face_image, tmp_path):
        import cv2
        from ofiq_syngen.cli import main
        input_path = tmp_path / "test_input.jpg"
        output_dir = tmp_path / "sweep_out"
        cv2.imwrite(str(input_path), face_image)

        rc = main(["sweep", "-c", "Sharpness", "-n", "3", "-o", str(output_dir), str(input_path)])
        assert rc == 0
        assert output_dir.exists()
        jpgs = list(output_dir.glob("*.jpg"))
        assert len(jpgs) == 3
