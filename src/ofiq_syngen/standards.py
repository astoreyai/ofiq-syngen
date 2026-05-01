"""Multi-standard component cross-reference.

Maps every ofiq-syngen component to its corresponding clauses in:
- ISO/IEC 29794-5 (face image quality measurement)
- ISO/IEC 19794-5 (face image data interchange format)
- ICAO Doc 9303 Part 9 (machine-readable travel documents)

The canonical source of truth is docs/standards/MAPPING.csv. This module
ships the same data as a Python dict so it travels with the wheel and
needs no file-resource handling at runtime.

When MAPPING.csv changes, regenerate STANDARDS_REFS below from it. The
test in tests/test_standards_mapping.py asserts CSV and dict stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StandardRefs:
    """Cross-standard references for a single OFIQ-aligned component.

    Attributes:
        ofiq_section: ISO/IEC FDIS 29794-5:2024 (= IS:2025) clause reference
            (e.g. '§7.3.2', '§7.4.10.1', or 'Annex D.2.1' for unmeasured
            requirements). Updated 2026-04-30 from the legacy 'S6.x/S7.x/S8'
            BSI Public Report numbering to the FDIS clause numbering used
            in the published standard.
        iso_29794_5: Aspect name in ISO/IEC 29794-5:2025 (matches FDIS
            Table 6 algorithm identifier description column).
        iso_19794_5: Clause reference in ISO/IEC 19794-5:2011 (e.g. '8.9 lighting').
        icao_9303: Clause reference in ICAO 9303 Part 9 (e.g. '3.2.3 background').
        alignment: 'exact', 'partial', or 'absent'.
        confidence: 'verified', 'derived', or 'uncertain'.
        ofiq_versions: OFIQ release tags this component aligns to.
            Default ('1.1',) means valid for OFIQ 1.1.x. Bump when a new
            OFIQ release changes the measurement algorithm such that the
            paired degrader needs revision.
    """

    ofiq_section: str
    iso_29794_5: str
    iso_19794_5: str
    icao_9303: str
    alignment: str
    confidence: str
    ofiq_versions: tuple[str, ...] = ("1.1",)


STANDARDS_REFS: dict[str, StandardRefs] = {
    "BackgroundUniformity.scalar": StandardRefs(
        "§7.3.2", "Background uniformity", "8.9 lighting/scene",
        "3.2.3 background", "exact", "verified",
    ),
    "IlluminationUniformity.scalar": StandardRefs(
        "§7.3.3", "Illumination uniformity", "8.9 lighting",
        "3.2.3 lighting", "exact", "verified",
    ),
    "LuminanceMean.scalar": StandardRefs(
        "§7.3.4.2", "Luminance mean", "8.9 exposure",
        "3.2.3 exposure", "exact", "verified",
    ),
    "LuminanceVariance.scalar": StandardRefs(
        "§7.3.4.3", "Luminance variance", "8.9 exposure",
        "3.2.3 contrast", "partial", "verified",
    ),
    "UnderExposurePrevention.scalar": StandardRefs(
        "§7.3.5", "Under-exposure prevention", "8.9 exposure",
        "3.2.3 exposure", "exact", "verified",
    ),
    "OverExposurePrevention.scalar": StandardRefs(
        "§7.3.6", "Over-exposure prevention", "8.9 exposure",
        "3.2.3 exposure", "exact", "verified",
    ),
    "DynamicRange.scalar": StandardRefs(
        "§7.3.7", "Dynamic range", "8.9 lighting",
        "3.2.3 contrast", "partial", "verified",
    ),
    "Sharpness.scalar": StandardRefs(
        "§7.3.8", "Sharpness (focus)", "8.10 focus",
        "3.2.3 focus", "exact", "verified",
    ),
    "CompressionArtifacts.scalar": StandardRefs(
        "§7.3.9", "No compression artefacts", "9.2 JPEG/JPEG2000",
        "3.2.5 compression", "exact", "verified",
    ),
    "NaturalColour.scalar": StandardRefs(
        "§7.3.10", "Natural colour", "9.1 colour space",
        "3.2.4 true colour", "exact", "verified",
    ),
    "RadialDistortion.scalar": StandardRefs(
        "Annex D.2.1", "Radial distortion (no QAA in IS:2025)", "8.10 lens",
        "3.2.3 lens distortion", "partial", "verified",
    ),
    "SingleFacePresent.scalar": StandardRefs(
        "§7.4.2", "Single face present", "8.1 subject",
        "3.2.3 one subject", "exact", "uncertain",
    ),
    "EyesOpen.scalar": StandardRefs(
        "§7.4.3", "Eyes open", "8.5 eye state",
        "3.2.3 eyes open", "exact", "verified",
    ),
    "MouthClosed.scalar": StandardRefs(
        "§7.4.4", "Mouth closed", "8.6 mouth state",
        "3.2.3 mouth closed", "exact", "verified",
    ),
    "EyesVisible.scalar": StandardRefs(
        "§7.4.5", "Eyes visible (no occlusion)", "8.7 eye occlusion",
        "3.2.3 no eye covering", "exact", "verified",
    ),
    "MouthOcclusionPrevention.scalar": StandardRefs(
        "§7.4.6", "Mouth occlusion prevention", "8.7 mouth occlusion",
        "3.2.3 no mouth covering", "exact", "verified",
    ),
    "FaceOcclusionPrevention.scalar": StandardRefs(
        "§7.4.7", "Face occlusion prevention", "8.7 face occlusion",
        "3.2.3 face visible", "exact", "verified",
    ),
    "InterEyeDistance.scalar": StandardRefs(
        "§7.4.8", "Inter-eye distance", "9.3 resolution",
        "3.2.2 eye distance min", "exact", "verified",
    ),
    "HeadSize.scalar": StandardRefs(
        "§7.4.9", "Head size", "7.4 token formats",
        "3.2.2 face size", "exact", "verified",
    ),
    "ExpressionNeutrality.scalar": StandardRefs(
        "§7.4.12", "Expression neutrality", "8.3 expression",
        "3.2.3 neutral expression", "exact", "uncertain",
    ),
    "NoHeadCoverings.scalar": StandardRefs(
        "§7.4.13", "No head covering", "8.8 head coverings",
        "3.2.3 head coverings", "partial", "uncertain",
    ),
    "HeadPoseYaw.scalar": StandardRefs(
        "§7.4.11.2", "Head pose angle yaw frontal alignment", "8.2 pose",
        "3.2.3 frontal", "exact", "verified",
    ),
    "HeadPosePitch.scalar": StandardRefs(
        "§7.4.11.3", "Head pose angle pitch frontal alignment", "8.2 pose",
        "3.2.3 frontal", "exact", "verified",
    ),
    "HeadPoseRoll.scalar": StandardRefs(
        "§7.4.11.4", "Head pose angle roll frontal alignment", "8.2 pose",
        "3.2.3 frontal", "exact", "verified",
    ),
    "LeftwardCropOfTheFaceImage.scalar": StandardRefs(
        "§7.4.10.1", "Leftward crop of face in image", "7.4 token formats",
        "3.2.2 centring", "exact", "verified",
    ),
    "RightwardCropOfTheFaceImage.scalar": StandardRefs(
        "§7.4.10.2", "Rightward crop of face in image", "7.4 token formats",
        "3.2.2 centring", "exact", "verified",
    ),
    "MarginAboveOfTheFaceImage.scalar": StandardRefs(
        "§7.4.10.3", "Margin above face in image", "7.4 token formats",
        "3.2.2 head position", "exact", "verified",
    ),
    "MarginBelowOfTheFaceImage.scalar": StandardRefs(
        "§7.4.10.4", "Margin below face in image", "7.4 token formats",
        "3.2.2 head position", "exact", "verified",
    ),
}


# Component subsets keyed by standard. Useful for CLI presets.
ICAO_STRICT_COMPONENTS: list[str] = sorted(
    name for name, refs in STANDARDS_REFS.items()
    if refs.icao_9303.startswith("3.2") and refs.alignment == "exact"
)

ISO_19794_5_COMPONENTS: list[str] = sorted(STANDARDS_REFS.keys())

ISO_29794_5_COMPONENTS: list[str] = sorted(STANDARDS_REFS.keys())


def get_refs(component: str) -> StandardRefs | None:
    """Return the standards references for a component, or None if missing."""
    return STANDARDS_REFS.get(component)


def components_by_alignment(alignment: str) -> list[str]:
    """Return component names whose alignment matches (exact, partial, absent)."""
    return sorted(
        name for name, refs in STANDARDS_REFS.items() if refs.alignment == alignment
    )


def components_by_confidence(confidence: str) -> list[str]:
    """Return component names whose confidence matches (verified, derived, uncertain)."""
    return sorted(
        name for name, refs in STANDARDS_REFS.items() if refs.confidence == confidence
    )


def components_for_ofiq_version(version: str) -> list[str]:
    """Return component names compatible with a given OFIQ release.

    Matches the version string against each component's ``ofiq_versions``
    tuple. Pass the major.minor portion (e.g. '1.1', not '1.1.0').
    """
    return sorted(
        name for name, refs in STANDARDS_REFS.items() if version in refs.ofiq_versions
    )
