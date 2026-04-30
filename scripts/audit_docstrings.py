#!/usr/bin/env python3
"""Docstring audit.

Walks every public symbol in src/ofiq_syngen/ and reports which ones
lack a docstring. Used as a pre-release gate.

Usage:
    python scripts/audit_docstrings.py
    python scripts/audit_docstrings.py --strict     # exit 1 if any missing
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _walk_module(path: Path) -> list[tuple[str, str, int]]:
    """Return (kind, qualname, lineno) for every public symbol missing a docstring."""
    src = path.read_text()
    tree = ast.parse(src)
    module_name = path.stem
    missing = []
    if not ast.get_docstring(tree):
        missing.append(("module", module_name, 1))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                missing.append(("function", f"{module_name}.{node.name}", node.lineno))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_"):
                continue
            if not ast.get_docstring(node):
                missing.append(("class", f"{module_name}.{node.name}", node.lineno))
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name.startswith("_") and item.name != "__init__":
                        continue
                    if not ast.get_docstring(item):
                        missing.append((
                            "method",
                            f"{module_name}.{node.name}.{item.name}",
                            item.lineno,
                        ))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any public symbol lacks a docstring.")
    parser.add_argument(
        "--root", type=Path,
        default=Path(__file__).resolve().parent.parent / "src" / "ofiq_syngen",
    )
    args = parser.parse_args()

    files = sorted(args.root.rglob("*.py"))
    if not files:
        print(f"No .py files under {args.root}", file=sys.stderr)
        return 1

    total_missing = 0
    for path in files:
        if path.name == "__init__.py":
            continue
        missing = _walk_module(path)
        if missing:
            print(f"\n{path.relative_to(args.root.parent.parent)}:")
            for kind, qual, line in missing:
                print(f"  {kind:8s}  {qual:50s}  line {line}")
            total_missing += len(missing)

    print(f"\n{total_missing} public symbol(s) without docstrings")
    if args.strict and total_missing:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
