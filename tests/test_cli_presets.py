"""Validate CLI standards presets."""

from __future__ import annotations

import io
import sys

import pytest

from ofiq_syngen.cli import PRESETS, _resolve_preset, main
from ofiq_syngen.standards import (
    ICAO_STRICT_COMPONENTS,
    ISO_19794_5_COMPONENTS,
    ISO_29794_5_COMPONENTS,
)


# -- Preset registry ----------------------------------------------------------


def test_presets_defined():
    assert set(PRESETS.keys()) == {"icao-strict", "iso-19794-5", "iso-29794-5"}


def test_preset_resolves_to_expected_lists():
    assert _resolve_preset("icao-strict") == ICAO_STRICT_COMPONENTS
    assert _resolve_preset("iso-19794-5") == ISO_19794_5_COMPONENTS
    assert _resolve_preset("iso-29794-5") == ISO_29794_5_COMPONENTS


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="Unknown preset"):
        _resolve_preset("nonsense")


def test_icao_strict_is_subset_of_full():
    assert set(ICAO_STRICT_COMPONENTS).issubset(set(ISO_29794_5_COMPONENTS))


def test_icao_strict_size_meaningful():
    assert 20 <= len(ICAO_STRICT_COMPONENTS) <= 28


# -- show-standards command ---------------------------------------------------


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    """Run main(argv) capturing stdout/stderr. Treats argparse SystemExit as a return code."""
    out_buf, err_buf = io.StringIO(), io.StringIO()
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = out_buf, err_buf
    rc: int
    try:
        rc = main(argv)
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
    finally:
        sys.stdout, sys.stderr = real_out, real_err
    return rc, out_buf.getvalue(), err_buf.getvalue()


def test_show_standards_no_preset_lists_all():
    rc, out, _ = _run_cli(["show-standards"])
    assert rc == 0
    for comp in ISO_29794_5_COMPONENTS:
        assert comp in out, f"{comp} not in show-standards output"


def test_show_standards_preset_filters():
    rc, out, _ = _run_cli(["show-standards", "--preset", "icao-strict"])
    assert rc == 0
    assert "Preset: icao-strict" in out
    for comp in ICAO_STRICT_COMPONENTS:
        assert comp in out
    excluded = set(ISO_29794_5_COMPONENTS) - set(ICAO_STRICT_COMPONENTS)
    for comp in excluded:
        assert comp not in out, f"{comp} leaked into icao-strict output"


def test_show_standards_includes_three_clause_columns():
    rc, out, _ = _run_cli(["show-standards"])
    assert rc == 0
    assert "ISO 29794-5" in out
    assert "ISO 19794-5" in out
    assert "ICAO 9303 P9" in out


# -- list-components --preset -------------------------------------------------


def test_list_components_preset_filters():
    rc, out, _ = _run_cli(["list-components", "--preset", "icao-strict"])
    assert rc == 0
    assert "Preset: icao-strict" in out
    for comp in ICAO_STRICT_COMPONENTS:
        assert comp in out


# -- generate-dataset --preset (validation only, no GPU run) -----------------


def test_generate_dataset_rejects_both_components_and_preset():
    rc, _, err = _run_cli([
        "generate-dataset",
        "--images-dir", "/nonexistent",
        "--components", "Sharpness",
        "--preset", "icao-strict",
    ])
    assert rc != 0
    assert "either --components or --preset" in err


def test_generate_dataset_rejects_invalid_preset():
    rc, _, err = _run_cli([
        "generate-dataset",
        "--images-dir", "/nonexistent",
        "--preset", "made-up",
    ])
    assert rc != 0
    assert "invalid choice" in err.lower() or "made-up" in err
