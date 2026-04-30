# Theory and per-component reference

| Doc | Purpose |
|---|---|
| [`COMPONENT_STATUS.md`](COMPONENT_STATUS.md) | Single-page rollup: standards mapping + OFIQ source provenance + test coverage + parity-vector status, per component |
| [`ANALYSIS.md`](../ANALYSIS.md) | Component-by-component algorithmic analysis: OFIQ measurement algorithm vs syngen perturbation algorithm with alignment grading |
| [`../standards/MAPPING.md`](../standards/MAPPING.md) | Multi-standard cross-reference (ISO/IEC 29794-5, ISO/IEC 19794-5, ICAO Doc 9303 Part 9) |
| [`../standards/PROVENANCE.md`](../standards/PROVENANCE.md) | OFIQ C++ `Execute()` line ranges per component plus helper-utility ports |
| [`../standards/SOURCES.md`](../standards/SOURCES.md) | Standards editions, OFIQ version pin, mapping confidence convention |
| [`../CITING.md`](../CITING.md) | Four-citation block for using ofiq-syngen in published work |

## Reading order

For a quick orient, read in this order:

1. **`COMPONENT_STATUS.md`** for the bird's-eye-view table.
2. **`MAPPING.md`** for the standards cross-reference.
3. **`ANALYSIS.md`** for any specific component you care about.
4. **`PROVENANCE.md`** when you need to trace a behavior back to the OFIQ C++ source.

For a rigorous traversal (e.g. drafting a methods section), include
**`SOURCES.md`** for version pinning and **`CITING.md`** for the
citation chain.
