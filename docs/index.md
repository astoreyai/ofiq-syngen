---
hide:
  - navigation
---

# ofiq-syngen

[![PyPI version](https://img.shields.io/pypi/v/ofiq-syngen.svg)](https://pypi.org/project/ofiq-syngen/)
[![CI](https://github.com/astoreyai/ofiq-syngen/actions/workflows/ci.yml/badge.svg)](https://github.com/astoreyai/ofiq-syngen/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ISO/IEC 29794-5 component-aligned synthetic face image quality degradation pipeline.**

Companion synthetic-degradation tool for the [BSI OFIQ](https://github.com/BSI-OFIQ/OFIQ-Project)
reference implementation. Generates controlled quality perturbations
that target the same image regions OFIQ measures, with multi-standard
cross-references to ISO/IEC 19794-5 and ICAO Doc 9303 Part 9.

```bash
pip install ofiq-syngen
```

## Where to start

| If you want to | Read |
|---|---|
| See the package's README and at-a-glance usage | [README on GitHub](https://github.com/astoreyai/ofiq-syngen#readme) |
| Walk through every public surface | [Usage](USAGE.md) |
| Understand the standards mapping | [Multi-standard mapping](standards/MAPPING.md) |
| See per-component status (test + parity coverage) | [Component status](theory/COMPONENT_STATUS.md) |
| Trace each degrader to its OFIQ source line | [OFIQ source provenance](standards/PROVENANCE.md) |
| Cite the package in published work | [Citation guide](CITING.md) |
| Compare to alternative degradation libraries | [Competitors](https://github.com/astoreyai/ofiq-syngen/blob/main/benchmarks/COMPETITORS.md) |
| Build a webcam capture demo | [Real-time capture example](https://github.com/astoreyai/ofiq-syngen/blob/main/examples/README_realtime.md) |
| Contribute | [Contributing](CONTRIBUTING.md) |
| Understand the OFIQ upstream relationship | [OFIQ upstream policy](OFIQ_UPSTREAM.md) |

## What's in this site

- **Usage**: full programmatic + CLI walkthrough.
- **Standards**: ISO/IEC 29794-5 + ISO/IEC 19794-5 + ICAO 9303 Part 9 cross-reference, source pinning, OFIQ provenance.
- **Theory**: per-component status table and pointers to the algorithmic analysis.
- **Project**: contributing guide, code of conduct, OFIQ upstream policy, changelog.

## What's not in this site (yet)

- Auto-generated API reference. The `mkdocstrings` plugin is configured in `mkdocs.yml` but the rendered API page is not currently included in the navigation. Run `mkdocs build` locally to generate it.
- Per-component visual gallery. Generated locally from a user-supplied face image; see [`docs/gallery/README.md`](https://github.com/astoreyai/ofiq-syngen/blob/main/docs/gallery/README.md).

## License

[MIT](https://github.com/astoreyai/ofiq-syngen/blob/main/LICENSE).
