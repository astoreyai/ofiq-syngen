"""Render a multi-page PDF of ofiq-syngen perturbations on VGGFace2 faces,
grouped by ICAO 9303 Part 9 §3.2 sub-clause.

For each ICAO category, picks a random VGGFace2 identity, runs every
ofiq-syngen operator that maps to that category at sev = {0.0, 0.5, 1.0},
and renders one row per operator. Category headers separate the groups.

Usage:
    OFIQ_MODEL_DIR=/path/to/OFIQ/data/models \
    python scripts/icao_perturbation_pdf.py \
        --vggface-dir /path/to/vggface2/test \
        --mapping-csv docs/standards/MAPPING.csv \
        --output notebooks/icao_perturbation_gallery.pdf \
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402  — must precede pyplot import
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from ofiq_syngen.pipeline import DegradationPipeline


SEVERITIES = [0.0, 0.5, 1.0]


def load_icao_groups(mapping_csv: Path) -> dict[str, list[tuple[str, str]]]:
    """Group components by ICAO §3.2.x sub-clause label."""
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    with open(mapping_csv) as f:
        for row in csv.DictReader(f):
            clause = row["icao_9303_part9_clause"]
            m = re.match(r"§(3\.2\.\d+|3\.2)\s*(.*)", clause)
            if m:
                label = m.group(2).strip("()").split("/")[0].strip()
                key = f"§{m.group(1)} — {label}" if label else f"§{m.group(1)}"
            else:
                key = clause
            groups[key].append((row["ofiq_syngen_component"], row["alignment"]))
    return dict(sorted(groups.items()))


def pick_random_face(vggface_dir: Path, rng: np.random.Generator) -> tuple[Path, np.ndarray]:
    """Pick a random VGGFace2 image from a random identity directory."""
    ids = sorted(p for p in vggface_dir.iterdir() if p.is_dir())
    for _ in range(10):
        ident_dir = ids[rng.integers(0, len(ids))]
        imgs = sorted(ident_dir.glob("*.jpg"))
        if not imgs:
            continue
        img_path = imgs[rng.integers(0, len(imgs))]
        img = cv2.imread(str(img_path))
        if img is not None and img.shape[0] >= 100 and img.shape[1] >= 100:
            return img_path, img
    raise RuntimeError(f"Could not find a usable VGGFace2 image under {vggface_dir}")


def render_strip(
    pipeline: DegradationPipeline,
    img: np.ndarray,
    component: str,
    seed: int,
) -> list[np.ndarray]:
    """Run the operator at each severity, return BGR images."""
    strip = []
    for sev in SEVERITIES:
        try:
            deg, _ = pipeline.degrade_single(img, component, sev, seed=seed)
        except Exception as exc:  # noqa: BLE001
            print(f"  warn: {component} sev={sev} failed: {exc}", file=sys.stderr)
            deg = img.copy()
        strip.append(deg)
    return strip


def render_pdf(
    vggface_dir: Path,
    mapping_csv: Path,
    output: Path,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    groups = load_icao_groups(mapping_csv)
    pipeline = DegradationPipeline()

    # Pre-pick one VGGFace2 image per ICAO category for variety
    category_imgs: dict[str, tuple[Path, np.ndarray]] = {}
    for cat in groups:
        category_imgs[cat] = pick_random_face(vggface_dir, rng)
        print(f"{cat}: using {category_imgs[cat][0].relative_to(vggface_dir.parent)}",
              file=sys.stderr)

    with PdfPages(output) as pdf:
        # ---- Cover page ----
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            "ofiq-syngen 0.5.1 — ICAO 9303 P9 §3.2 perturbation gallery",
            fontsize=14, y=0.95, fontweight="bold",
        )
        cover_text = (
            "Per-category random VGGFace2 perturbations.\n\n"
            f"Categories: {len(groups)} ICAO §3.2.x sub-clauses\n"
            f"Components: {sum(len(v) for v in groups.values())} ofiq-syngen operators\n"
            f"Severities: {SEVERITIES}\n"
            f"Random seed: {seed}\n\n"
            "Each row shows one operator at three severity levels.\n"
            "Source images: VGGFace2 test split, license-clean for academic use.\n\n"
            "Standards mapping: docs/standards/MAPPING.csv\n"
            "Operator details: ofiq-syngen ANALYSIS.md\n"
            "Parity vectors: tests/fixtures/ofiq_parity/manifest.json (245 vs OFIQ 1.1.0)"
        )
        fig.text(0.1, 0.7, cover_text, fontsize=11, va="top", family="monospace")

        # Table of contents
        toc = "TABLE OF CONTENTS\n\n"
        for i, cat in enumerate(groups, 1):
            ops = ", ".join(c.replace(".scalar", "") for c, _ in groups[cat])
            toc += f"  {i:>2}. {cat}\n       {ops}\n\n"
        fig.text(0.1, 0.42, toc, fontsize=8, va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # ---- One page per ICAO category ----
        for cat, comps in groups.items():
            img_path, img = category_imgs[cat]
            n = len(comps)
            fig, axes = plt.subplots(
                n, len(SEVERITIES), figsize=(8.5, max(2.4, 2.4 * n)),
                squeeze=False,
            )
            fig.suptitle(
                f"{cat}\n{img_path.parent.name}/{img_path.name}",
                fontsize=11, y=0.99, fontweight="bold",
            )
            for row, (component, alignment) in enumerate(comps):
                strip = render_strip(pipeline, img, component, seed)
                for col, (sev, deg) in enumerate(zip(SEVERITIES, strip, strict=True)):
                    ax = axes[row, col]
                    ax.imshow(cv2.cvtColor(deg, cv2.COLOR_BGR2RGB))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    if col == 0:
                        ax.set_ylabel(
                            f"{component.replace('.scalar', '')}\n[{alignment}]",
                            fontsize=8, rotation=0, ha="right", va="center",
                        )
                    if row == 0:
                        ax.set_title(f"sev={sev}", fontsize=9)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            print(f"  rendered category: {cat} ({n} ops)", file=sys.stderr)

    print(f"\nWrote {output}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--vggface-dir", type=Path, required=True,
                        help="Path to a VGGFace2 split directory (contains nXXXXXX/ identity dirs)")
    parser.add_argument("--mapping-csv", type=Path,
                        default=Path(__file__).resolve().parent.parent
                        / "docs" / "standards" / "MAPPING.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    render_pdf(args.vggface_dir, args.mapping_csv, args.output, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
