"""ISO/IEC 29794-5 standards references.

Re-exports from ``ofiq_syngen.standards`` so the 3D submodule never
drifts from the canonical 2D standards table. Now that the 3D pipeline
lives inside ``ofiq_syngen.three_d``, the parent package is always
importable -- this module exists for backwards compatibility with the
pre-merge ``three_d_syn`` API.
"""

from __future__ import annotations

try:
    from ofiq_syngen.standards import (
        ICAO_STRICT_COMPONENTS,
        ISO_19794_5_COMPONENTS,
        ISO_29794_5_COMPONENTS,
        STANDARDS_REFS,
        StandardRefs,
        components_by_alignment,
        components_by_confidence,
        components_for_ofiq_version,
        get_refs,
    )

    _OFIQ_SYNGEN_AVAILABLE = True
except ImportError as _exc:
    _OFIQ_SYNGEN_AVAILABLE = False
    _IMPORT_ERROR = _exc

    from dataclasses import dataclass

    @dataclass(frozen=True)
    class StandardRefs:  # type: ignore[no-redef]
        """Stub StandardRefs used when ofiq-syngen is not installed.

        Lookup against STANDARDS_REFS will return None; degradations still
        register but carry no standards metadata. This keeps the scaffold
        importable without ofiq-syngen for plumbing tests.
        """

        ofiq_section: str = ""
        iso_29794_5: str = ""
        iso_19794_5: str = ""
        icao_9303: str = ""
        alignment: str = ""
        confidence: str = ""
        ofiq_versions: tuple[str, ...] = ("1.1",)

    STANDARDS_REFS: dict[str, StandardRefs] = {}  # type: ignore[no-redef]
    ICAO_STRICT_COMPONENTS: list[str] = []  # type: ignore[no-redef]
    ISO_19794_5_COMPONENTS: list[str] = []  # type: ignore[no-redef]
    ISO_29794_5_COMPONENTS: list[str] = []  # type: ignore[no-redef]

    def get_refs(component: str):  # type: ignore[no-redef]
        return None

    def components_by_alignment(alignment: str) -> list[str]:  # type: ignore[no-redef]
        return []

    def components_by_confidence(confidence: str) -> list[str]:  # type: ignore[no-redef]
        return []

    def components_for_ofiq_version(version: str) -> list[str]:  # type: ignore[no-redef]
        return []


def is_ofiq_syngen_available() -> bool:
    """Whether ofiq-syngen is importable and STANDARDS_REFS is populated."""
    return _OFIQ_SYNGEN_AVAILABLE


__all__ = [
    "STANDARDS_REFS",
    "StandardRefs",
    "ICAO_STRICT_COMPONENTS",
    "ISO_19794_5_COMPONENTS",
    "ISO_29794_5_COMPONENTS",
    "components_by_alignment",
    "components_by_confidence",
    "components_for_ofiq_version",
    "get_refs",
    "is_ofiq_syngen_available",
]
