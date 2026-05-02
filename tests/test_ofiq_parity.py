"""OFIQ binary score-parity tests.

Reads ``tests/fixtures/ofiq_parity/manifest.json`` and asserts that for
every recorded (image, component, severity) tuple, regenerating the
degraded image and rescoring it with the OFIQ binary reproduces the
expected score within tolerance.

The manifest is empty by default. See ``tests/fixtures/ofiq_parity/REGENERATE.md``
for how to populate it. When the manifest is empty, every test in this
file is skipped.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import pytest

from ofiq_syngen import DegradationPipeline


MANIFEST_PATH = Path(__file__).resolve().parent / "fixtures" / "ofiq_parity" / "manifest.json"
OFIQ_BINARY_ENV = "OFIQ_BINARY"


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {"vectors": [], "image_set": [], "tolerance": 5.0}
    return json.loads(MANIFEST_PATH.read_text())


_MANIFEST = _load_manifest()
_VECTORS = _MANIFEST.get("vectors", [])
_IMAGE_SET = {entry["id"]: entry for entry in _MANIFEST.get("image_set", [])}
_TOLERANCE = float(_MANIFEST.get("tolerance", 5.0))


def _ofiq_binary_available() -> bool:
    binary_path = os.environ.get(OFIQ_BINARY_ENV)
    if not binary_path:
        return False
    return Path(binary_path).exists()


def _resolve_image_path(image_id: str) -> Path | None:
    """Resolve an image_id to its on-disk path.

    Manifest stores paths relative to its own directory
    (``tests/fixtures/ofiq_parity/``), so we anchor on
    ``MANIFEST_PATH.parent``. The fallback up one level preserves the
    pre-v0.5 layout where images lived under ``tests/fixtures/images/``.
    """
    entry = _IMAGE_SET.get(image_id)
    if entry is None:
        return None
    for base in (MANIFEST_PATH.parent, MANIFEST_PATH.parent.parent):
        candidate = base / entry["path"]
        if candidate.exists():
            return candidate
    return None


def _run_ofiq(image_path: Path) -> dict[str, float]:
    """Invoke the OFIQ 1.1.0 SampleApp, return per-component scores.

    OFIQ 1.1.0 CLI: ``-c <configDir> -i <inputFile> -o <outputFile>``
    (no ``-l`` listing flag). Output is semicolon-delimited CSV.
    """
    binary = Path(os.environ[OFIQ_BINARY_ENV])
    env_data = os.environ.get("OFIQ_DATA_DIR")
    if env_data:
        config_dir = Path(env_data)
    else:
        candidate = binary.resolve().parent
        for _ in range(6):
            if (candidate / "data").is_dir() and (candidate / "data" / "models").is_dir():
                config_dir = candidate / "data"
                break
            candidate = candidate.parent
        else:
            raise FileNotFoundError(
                f"OFIQ data directory not found relative to {binary}. "
                "Set OFIQ_DATA_DIR environment variable."
            )
    with tempfile.TemporaryDirectory() as td:
        out_csv = Path(td) / "scores.csv"
        subprocess.run(
            [
                str(binary),
                "-c", str(config_dir),
                "-i", str(image_path),
                "-o", str(out_csv),
            ],
            check=True, capture_output=True,
        )
        import csv
        rows = list(csv.DictReader(out_csv.open(), delimiter=";"))
        out = {}
        for k, v in rows[0].items():
            try:
                out[k] = float(v)
            except (ValueError, TypeError):
                continue
        return out


@pytest.mark.skipif(
    not _VECTORS,
    reason="No parity vectors recorded; see tests/fixtures/ofiq_parity/REGENERATE.md",
)
@pytest.mark.skipif(
    not _ofiq_binary_available(),
    reason=f"OFIQ binary not available; set {OFIQ_BINARY_ENV} env var",
)
@pytest.mark.parametrize(
    "vector",
    _VECTORS,
    ids=lambda v: f"{v['image_id']}__{v['component'].replace('.scalar', '')}__sev{v['severity']}",
)
def test_parity_for_vector(vector: dict):
    """Regenerate the degraded image and check OFIQ score within tolerance."""
    image_path = _resolve_image_path(vector["image_id"])
    if image_path is None:
        pytest.skip(f"Source image {vector['image_id']} not found on disk")

    src = cv2.imread(str(image_path))
    pipeline = DegradationPipeline()
    degraded, _ = pipeline.degrade_single(
        src, vector["component"], vector["severity"], seed=vector["seed"],
    )

    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        cv2.imwrite(tf.name, degraded)
        scores = _run_ofiq(Path(tf.name))

    actual = scores.get(vector["component"])
    assert actual is not None, (
        f"OFIQ produced no score for {vector['component']}"
    )
    expected = float(vector["expected_score"])
    delta = abs(actual - expected)
    assert delta <= _TOLERANCE, (
        f"{vector['component']} sev={vector['severity']} on "
        f"{vector['image_id']}: expected {expected:.2f}, got {actual:.2f}, "
        f"delta {delta:.2f} exceeds tolerance {_TOLERANCE}"
    )


def test_manifest_well_formed():
    """The manifest file (empty or populated) parses and has the expected schema."""
    m = _load_manifest()
    assert m.get("schema_version") == "1"
    assert isinstance(m.get("vectors", []), list)
    assert isinstance(m.get("image_set", []), list)
    assert isinstance(m.get("tolerance", 0), (int, float))


def test_vectors_reference_known_images():
    """If vectors are present, every image_id must appear in image_set."""
    if not _VECTORS:
        pytest.skip("No vectors recorded")
    image_ids = {entry["id"] for entry in _IMAGE_SET.values()}
    for v in _VECTORS:
        assert v["image_id"] in image_ids, (
            f"Vector references unknown image_id {v['image_id']}"
        )
