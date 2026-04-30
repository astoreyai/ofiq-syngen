"""Command-line interface for ofiq-syngen.

Usage:
    ofiq-syngen degrade --component Sharpness.scalar --severity 0.6 --output degraded.jpg input.jpg
    ofiq-syngen sweep --component Sharpness.scalar --levels 10 --output-dir ./output input.jpg
    ofiq-syngen list-components
    ofiq-syngen list-components --preset icao-strict
    ofiq-syngen generate-dataset --images-dir ./faces --output-dir ./degraded --max-images 100
    ofiq-syngen generate-dataset --images-dir ./faces --preset icao-strict --output-dir ./icao
    ofiq-syngen show-standards
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from ofiq_syngen import __version__
from ofiq_syngen.components import (
    COMPONENT_REGISTRY,
    list_all_degradations,
    list_supported_components,
)
from ofiq_syngen.pipeline import DegradationConfig, DegradationPipeline
from ofiq_syngen.standards import (
    ICAO_STRICT_COMPONENTS,
    ISO_19794_5_COMPONENTS,
    ISO_29794_5_COMPONENTS,
    STANDARDS_REFS,
    get_refs,
)


PRESETS: dict[str, list[str]] = {
    "icao-strict": ICAO_STRICT_COMPONENTS,
    "iso-19794-5": ISO_19794_5_COMPONENTS,
    "iso-29794-5": ISO_29794_5_COMPONENTS,
}


def _resolve_preset(preset: str) -> list[str]:
    """Resolve a preset name to its component list. Raises on unknown."""
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {preset!r}. Valid: {valid}")
    return PRESETS[preset]


def _resolve_component(name: str) -> str:
    """Resolve a component name, adding .scalar suffix if missing."""
    if name in COMPONENT_REGISTRY:
        return name
    candidate = name + ".scalar"
    if candidate in COMPONENT_REGISTRY:
        return candidate
    return name  # Let the pipeline raise a clear error


def cmd_degrade(args: argparse.Namespace) -> int:
    """Apply a single degradation to an image."""
    img = cv2.imread(str(args.input))
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        return 1

    component = _resolve_component(args.component)
    pipeline = DegradationPipeline(DegradationConfig(seed=args.seed))

    degraded, meta = pipeline.degrade_single(
        img, component, args.severity, seed=args.seed
    )

    output = args.output or f"degraded_{Path(args.input).name}"
    cv2.imwrite(str(output), degraded)
    print(f"Saved: {output}")
    print(f"  Component: {meta['target_component']}")
    print(f"  Degradation: {meta['degradation_type']}")
    print(f"  Severity: {meta['severity']}")
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    """Sweep severity levels for a component."""
    img = cv2.imread(str(args.input))
    if img is None:
        print(f"Error: cannot read image '{args.input}'", file=sys.stderr)
        return 1

    component = _resolve_component(args.component)
    levels = [i / (args.levels - 1) for i in range(args.levels)] if args.levels > 1 else [0.5]
    config = DegradationConfig(severity_levels=levels, seed=args.seed)
    pipeline = DegradationPipeline(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = pipeline.degrade_sweep(img, component, seed=args.seed)
    stem = Path(args.input).stem
    comp_short = component.replace(".scalar", "")

    for degraded, meta in results:
        sev = meta["severity"]
        out_path = output_dir / f"{stem}__{comp_short}_{sev:.2f}.jpg"
        cv2.imwrite(str(out_path), degraded)
        print(f"  {out_path.name}  severity={sev:.2f}")

    print(f"\nSaved {len(results)} images to {output_dir}/")
    return 0


def cmd_list_components(args: argparse.Namespace) -> int:
    """List all supported OFIQ components and their degradation functions."""
    if args.preset:
        try:
            allowed = set(_resolve_preset(args.preset))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    else:
        allowed = None

    all_degs = list_all_degradations()
    if allowed is not None:
        all_degs = [d for d in all_degs if d[0] in allowed]
    if not all_degs:
        print(f"No components match preset {args.preset!r}", file=sys.stderr)
        return 1

    max_comp = max(len(c) for c, _, _ in all_degs)
    max_desc = max(len(d) for _, d, _ in all_degs)

    if args.preset:
        print(f"Preset: {args.preset}")
    print(f"{'Component':<{max_comp}}  {'Degradation':<{max_desc}}  {'Severity Range':<25}  Ctx")
    print(f"{'-' * max_comp}  {'-' * max_desc}  {'-' * 25}  ---")
    for comp, desc, srange in all_degs:
        d = COMPONENT_REGISTRY[comp][0]
        ctx = "yes" if d.requires_context else ""
        print(f"{comp:<{max_comp}}  {desc:<{max_desc}}  {srange:<25}  {ctx}")

    components = sorted({c for c, _, _ in all_degs})
    total_degs = len(all_degs)
    ctx_count = sum(1 for c in components if COMPONENT_REGISTRY[c][0].requires_context)
    print(f"\n{len(components)} components, {total_degs} degradation functions ({ctx_count} require FaceContext)")
    return 0


def cmd_export_conformance(args: argparse.Namespace) -> int:
    """Export a conformance bundle as a single ZIP.

    The bundle contains the parity manifest, multi-standard mapping CSV,
    standards.py source, and SOURCES.md so any OFIQ-aligned implementation
    can run the same conformance assertions against its own measurement
    chain.
    """
    import json
    import zipfile
    from datetime import datetime

    from ofiq_syngen import __version__

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pkg_root = Path(__file__).resolve().parent
    repo_root = pkg_root.parent.parent
    items = {
        "STANDARDS_REFS.json": json.dumps(
            {
                name: {
                    "ofiq_section": refs.ofiq_section,
                    "iso_29794_5": refs.iso_29794_5,
                    "iso_19794_5": refs.iso_19794_5,
                    "icao_9303": refs.icao_9303,
                    "alignment": refs.alignment,
                    "confidence": refs.confidence,
                    "ofiq_versions": list(refs.ofiq_versions),
                }
                for name, refs in STANDARDS_REFS.items()
            },
            indent=2, sort_keys=True,
        ),
        "MAPPING.csv": (repo_root / "docs" / "standards" / "MAPPING.csv").read_text()
            if (repo_root / "docs" / "standards" / "MAPPING.csv").exists() else "",
        "SOURCES.md": (repo_root / "docs" / "standards" / "SOURCES.md").read_text()
            if (repo_root / "docs" / "standards" / "SOURCES.md").exists() else "",
        "PROVENANCE.md": (repo_root / "docs" / "standards" / "PROVENANCE.md").read_text()
            if (repo_root / "docs" / "standards" / "PROVENANCE.md").exists() else "",
    }
    parity_manifest_path = (
        repo_root / "tests" / "fixtures" / "ofiq_parity" / "manifest.json"
    )
    if parity_manifest_path.exists():
        items["parity_manifest.json"] = parity_manifest_path.read_text()
    items["BUNDLE_INFO.json"] = json.dumps({
        "syngen_version": __version__,
        "exported_at": datetime.utcnow().isoformat() + "Z",
        "purpose": "OFIQ conformance bundle for alternative implementations",
        "consumers": [
            "alternative OFIQ-aligned measurement implementations",
            "regression tests in OFIQ-Project",
            "third-party FIQA tools comparing to OFIQ baseline",
        ],
        "schema": {
            "STANDARDS_REFS.json": "Per-component cross-reference (ISO 29794-5, ISO 19794-5, ICAO 9303)",
            "MAPPING.csv": "Same data, machine-readable CSV",
            "SOURCES.md": "Standards editions and version pinning",
            "PROVENANCE.md": "OFIQ C++ source line refs per component",
            "parity_manifest.json": "Expected OFIQ scores per (image, component, severity); empty if not regenerated",
        },
    }, indent=2, sort_keys=True)

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in items.items():
            zf.writestr(name, content)

    print(f"Wrote conformance bundle to {out_path} ({len(items)} entries)")
    return 0


def cmd_show_standards(args: argparse.Namespace) -> int:
    """Print the multi-standard cross-reference for every component."""
    if args.preset:
        try:
            allowed = set(_resolve_preset(args.preset))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        rows = [(c, get_refs(c)) for c in sorted(allowed) if get_refs(c) is not None]
    else:
        rows = sorted(STANDARDS_REFS.items())

    if not rows:
        print(f"No components match preset {args.preset!r}", file=sys.stderr)
        return 1

    max_comp = max(len(c) for c, _ in rows)
    max_29 = max(len(r.iso_29794_5) for _, r in rows)
    max_19 = max(len(r.iso_19794_5) for _, r in rows)
    max_icao = max(len(r.icao_9303) for _, r in rows)

    if args.preset:
        print(f"Preset: {args.preset}")
    header = (
        f"{'Component':<{max_comp}}  {'OFIQ':<5}  "
        f"{'ISO 29794-5':<{max_29}}  "
        f"{'ISO 19794-5':<{max_19}}  "
        f"{'ICAO 9303 P9':<{max_icao}}  "
        f"{'Align':<7}  Conf"
    )
    print(header)
    print("-" * len(header))
    for comp, refs in rows:
        print(
            f"{comp:<{max_comp}}  {refs.ofiq_section:<5}  "
            f"{refs.iso_29794_5:<{max_29}}  "
            f"{refs.iso_19794_5:<{max_19}}  "
            f"{refs.icao_9303:<{max_icao}}  "
            f"{refs.alignment:<7}  {refs.confidence}"
        )
    print(f"\n{len(rows)} components")
    return 0


def cmd_generate_dataset(args: argparse.Namespace) -> int:
    """Generate a full degradation dataset."""
    if args.components and args.preset:
        print("Error: pass either --components or --preset, not both", file=sys.stderr)
        return 1

    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        print(f"Error: '{images_dir}' is not a directory", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    config = DegradationConfig(seed=args.seed)
    pipeline = DegradationPipeline(config)

    components: list[str] | None
    if args.preset:
        try:
            components = list(_resolve_preset(args.preset))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
    elif args.components:
        components = [_resolve_component(c.strip()) for c in args.components.split(",")]
    else:
        components = None

    print(f"Generating dataset from {images_dir} -> {output_dir}")
    print(f"  Max images: {args.max_images}")
    if args.preset:
        print(f"  Preset: {args.preset} ({len(components)} components)")
    else:
        print(f"  Components: {len(components) if components else 'all'}")

    manifest = pipeline.generate_dataset(
        image_dir=images_dir,
        output_dir=output_dir,
        max_images=args.max_images,
        components=components,
    )

    manifest_path = output_dir / "manifest.csv"
    manifest.to_csv(str(manifest_path), index=False)
    print(f"\nGenerated {len(manifest)} entries")
    print(f"Manifest saved to {manifest_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the ofiq-syngen CLI."""
    parser = argparse.ArgumentParser(
        prog="ofiq-syngen",
        description="ISO/IEC 29794-5 component-aligned synthetic face image quality degradation pipeline",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # degrade
    p_degrade = subparsers.add_parser("degrade", help="Apply a single degradation to an image")
    p_degrade.add_argument("input", help="Input image path")
    p_degrade.add_argument("--component", "-c", required=True, help="OFIQ component name (e.g., Sharpness or Sharpness.scalar)")
    p_degrade.add_argument("--severity", "-s", type=float, default=0.5, help="Severity level 0.0-1.0 (default: 0.5)")
    p_degrade.add_argument("--output", "-o", help="Output image path (default: degraded_<input>)")
    p_degrade.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # sweep
    p_sweep = subparsers.add_parser("sweep", help="Sweep severity levels for a component")
    p_sweep.add_argument("input", help="Input image path")
    p_sweep.add_argument("--component", "-c", required=True, help="OFIQ component name")
    p_sweep.add_argument("--levels", "-n", type=int, default=10, help="Number of severity levels (default: 10)")
    p_sweep.add_argument("--output-dir", "-o", default="./sweep_output", help="Output directory (default: ./sweep_output)")
    p_sweep.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # list-components
    p_list = subparsers.add_parser("list-components", help="List all supported OFIQ components")
    p_list.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Filter to a standards preset (icao-strict, iso-19794-5, iso-29794-5)",
    )

    # show-standards
    p_std = subparsers.add_parser(
        "show-standards",
        help="Print the multi-standard cross-reference for every component",
    )
    p_std.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Filter to a standards preset (icao-strict, iso-19794-5, iso-29794-5)",
    )

    # export-conformance
    p_exp = subparsers.add_parser(
        "export-conformance",
        help="Export a conformance bundle (parity manifest + standards mapping)",
    )
    p_exp.add_argument(
        "--output", "-o", required=True,
        help="Output path for the bundle (.zip)",
    )

    # generate-dataset
    p_gen = subparsers.add_parser("generate-dataset", help="Generate a full degradation dataset")
    p_gen.add_argument("--images-dir", "-i", required=True, help="Directory of source face images")
    p_gen.add_argument("--output-dir", "-o", default="./degraded_dataset", help="Output directory (default: ./degraded_dataset)")
    p_gen.add_argument("--max-images", "-n", type=int, default=100, help="Max source images to process (default: 100)")
    p_gen.add_argument("--components", help="Comma-separated component list (default: all). Mutually exclusive with --preset.")
    p_gen.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default=None,
        help="Standards preset to filter components (icao-strict, iso-19794-5, iso-29794-5). Mutually exclusive with --components.",
    )
    p_gen.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    dispatch = {
        "degrade": cmd_degrade,
        "sweep": cmd_sweep,
        "list-components": cmd_list_components,
        "show-standards": cmd_show_standards,
        "generate-dataset": cmd_generate_dataset,
        "export-conformance": cmd_export_conformance,
    }

    return dispatch[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
