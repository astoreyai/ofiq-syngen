# Relationship to the BSI OFIQ-Project

`ofiq-syngen` is a **companion** to the BSI OFIQ reference implementation,
not a fork or replacement. This document spells out the relationship,
what would be candidates to upstream, and how to coordinate.

## What we are

A Python package that generates synthetic face image degradations
**aligned to the OFIQ measurement algorithms**. Each degradation
operates on the exact image regions OFIQ measures, so `ofiq-syngen`
output, scored by OFIQ, produces predictable per-component score
movements.

Targets the same component set as OFIQ (28 components in v0.3.0,
including the forward-looking `RadialDistortion` from ISO/IEC 29794-5
not yet measured by OFIQ 1.1.0).

## What we are not

- Not a reimplementation of OFIQ's measurement algorithms.
- Not a fork of `BSI-OFIQ/OFIQ-Project`.
- Not a Python port of OFIQ. The C++ reference implementation remains
  the authoritative measurement code; we ship perturbation code that
  pairs with it.
- Not a competing standard. Where ISO/IEC 29794-5 specifies a
  measurement, OFIQ implements it; we degrade against it.

## Coordination

### Upstream first

Issues with the OFIQ measurement algorithms themselves (bugs, edge
cases, unclear specifications) belong on `BSI-OFIQ/OFIQ-Project`. We do
not patch around upstream issues silently; we file the issue upstream
and document the workaround in `docs/standards/SOURCES.md` until it is
resolved.

### Discrepancy tracking

When `ofiq-syngen` and OFIQ disagree (e.g., because we discover a
mismatch between our perturbation and OFIQ's measurement region), we
log the discrepancy in two places:

1. `CHANGELOG.md` under the relevant version, "Standards alignment"
   subsection.
2. `docs/standards/SOURCES.md` "Discrepancy flag" subsection.

The current open discrepancy is the OFIQ version pin: `OFIQ-Project/Version.txt`
reports `1.1.0` while the v0.2.0 CHANGELOG entry says "BSI V1.2." This
must be resolved before tagging v0.4.0.

## Candidates to upstream

Items in `ofiq-syngen` that would be useful to OFIQ itself, ordered by
maturity:

### Likely good candidates

- **`docs/standards/PROVENANCE.md`** — the per-component citation of
  OFIQ C++ source line ranges is a documentation product that would
  also be useful in the OFIQ repository as a doc-block at the top of
  each measure file. We are happy to submit a PR.
- **Per-component perturbation reference image** — once
  `tests/fixtures/ofiq_parity/` (Phase 1.Z.3) lands, the (input image,
  expected score) pairs could form the basis of an OFIQ regression test
  suite. We are happy to contribute the dataset + Python harness; OFIQ
  maintainers would write the C++ side.

### Not yet ready to upstream

- **`src/ofiq_syngen/standards.py` STANDARDS_REFS dict** — useful for
  cross-standards work but Python-specific; the equivalent in OFIQ
  would be a `static const` map in C++, which we have not drafted.
- **The 28-component to 5-panel aggregation in `examples/realtime_capture.py`**
  — opinionated UX choice, probably belongs in a downstream
  application-layer tool rather than OFIQ core.

### Not candidates

- **Degradation functions themselves** — these belong here. OFIQ is a
  measurement library; perturbation tools should remain separate.
- **The Python wheel build** — OFIQ ships C++ + bindings; a Python
  port would be redundant.

## BSI patch acceptance process

To submit upstream contributions:

1. File an issue on `BSI-OFIQ/OFIQ-Project` describing the change. Wait
   for maintainer acknowledgment before writing code. BSI's review
   process is more deliberate than typical OSS turnaround; budget
   weeks, not days.
2. Sign the relevant contributor agreements. BSI is a German federal
   agency and may require a CLA depending on the contribution scope.
3. Submit a PR referencing the issue. Include a test plan and
   conformance evidence (ideally using `ofiq-syngen`'s parity test
   vectors once those exist).

The OFIQ project is conservative about behavior changes because OFIQ
output enters operational identity-verification pipelines. Bug fixes
are easier to land than new features.

## Citation reciprocity

Any reciprocity is one-way until OFIQ ships new functionality that
references `ofiq-syngen`. We cite OFIQ unconditionally (see
[`docs/CITING.md`](docs/CITING.md)). If you publish work using both
projects, citing both is the right thing to do.

## Contact

For coordination on standards alignment, conformance testing, or
upstream submissions, open an issue on this repository's tracker. Do
not ping BSI maintainers directly about `ofiq-syngen` issues; use the
upstream-first principle above.
