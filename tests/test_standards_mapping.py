"""Validate the multi-standard mapping covers every registered component.

Asserts:
- Every component in COMPONENT_REGISTRY has a non-null standard_refs.
- Every STANDARDS_REFS entry maps to a registered component.
- Clause-ID strings are well-formed (non-empty, no whitespace tokens).
- CSV in docs/standards/MAPPING.csv stays in sync with STANDARDS_REFS.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ofiq_syngen.components import COMPONENT_REGISTRY
from ofiq_syngen.standards import (
    ICAO_STRICT_COMPONENTS,
    STANDARDS_REFS,
    components_by_alignment,
    components_by_confidence,
    get_refs,
)


# -- Coverage -----------------------------------------------------------------


def test_every_registered_component_has_standard_refs():
    missing = [
        comp for comp, degs in COMPONENT_REGISTRY.items()
        if any(d.standard_refs is None for d in degs)
    ]
    assert not missing, (
        f"Components missing standard_refs: {missing}. "
        "Update src/ofiq_syngen/standards.py STANDARDS_REFS."
    )


def test_every_standards_entry_maps_to_a_registered_component():
    registered = set(COMPONENT_REGISTRY.keys())
    extra = [c for c in STANDARDS_REFS if c not in registered]
    assert not extra, (
        f"STANDARDS_REFS contains components not in COMPONENT_REGISTRY: {extra}. "
        "Either remove from standards.py or register the degrader."
    )


def test_get_refs_round_trip():
    for comp in COMPONENT_REGISTRY:
        refs = get_refs(comp)
        assert refs is not None, f"get_refs({comp!r}) returned None"


# -- Field well-formedness ----------------------------------------------------


def test_alignment_values_are_valid():
    valid = {"exact", "partial", "absent"}
    for comp, refs in STANDARDS_REFS.items():
        assert refs.alignment in valid, (
            f"{comp}: alignment={refs.alignment!r} not in {valid}"
        )


def test_confidence_values_are_valid():
    valid = {"verified", "derived", "uncertain"}
    for comp, refs in STANDARDS_REFS.items():
        assert refs.confidence in valid, (
            f"{comp}: confidence={refs.confidence!r} not in {valid}"
        )


def test_clause_strings_non_empty():
    fields = ("ofiq_section", "iso_29794_5", "iso_19794_5", "icao_9303")
    for comp, refs in STANDARDS_REFS.items():
        for field in fields:
            value = getattr(refs, field)
            assert value and value.strip(), (
                f"{comp}: field {field} is empty or whitespace"
            )


# -- Subset accessors ---------------------------------------------------------


def test_icao_strict_subset_non_empty():
    assert len(ICAO_STRICT_COMPONENTS) >= 20, (
        "Expected at least 20 components in the ICAO-strict preset; "
        f"got {len(ICAO_STRICT_COMPONENTS)}"
    )


def test_alignment_partition_covers_everything():
    everything = (
        components_by_alignment("exact")
        + components_by_alignment("partial")
        + components_by_alignment("absent")
    )
    assert sorted(everything) == sorted(STANDARDS_REFS.keys())


def test_confidence_partition_covers_everything():
    everything = (
        components_by_confidence("verified")
        + components_by_confidence("derived")
        + components_by_confidence("uncertain")
    )
    assert sorted(everything) == sorted(STANDARDS_REFS.keys())


# -- CSV sync -----------------------------------------------------------------


def _csv_path() -> Path:
    here = Path(__file__).resolve()
    candidate = here.parent.parent / "docs" / "standards" / "MAPPING.csv"
    return candidate


@pytest.mark.skipif(
    not _csv_path().exists(),
    reason="MAPPING.csv not present (running from sdist or installed wheel)",
)
def test_csv_matches_standards_refs():
    """The CSV is the human-edited source of truth; standards.py must match."""
    rows = list(csv.DictReader(_csv_path().open()))
    csv_components = {row["ofiq_syngen_component"] for row in rows}
    py_components = set(STANDARDS_REFS.keys())

    only_in_csv = csv_components - py_components
    only_in_py = py_components - csv_components

    assert not only_in_csv, (
        f"In MAPPING.csv but not standards.py: {only_in_csv}. "
        "Run scripts/regen_standards.py (or add manually to STANDARDS_REFS)."
    )
    assert not only_in_py, (
        f"In standards.py but not MAPPING.csv: {only_in_py}. "
        "Add the row to MAPPING.csv to keep them in sync."
    )

    for row in rows:
        comp = row["ofiq_syngen_component"]
        refs = STANDARDS_REFS[comp]
        assert refs.ofiq_section == row["ofiq_section"], (
            f"{comp}: ofiq_section mismatch CSV={row['ofiq_section']!r} "
            f"py={refs.ofiq_section!r}"
        )
        assert refs.alignment == row["alignment"], (
            f"{comp}: alignment mismatch CSV={row['alignment']!r} "
            f"py={refs.alignment!r}"
        )
        assert refs.confidence == row["confidence"], (
            f"{comp}: confidence mismatch CSV={row['confidence']!r} "
            f"py={refs.confidence!r}"
        )
