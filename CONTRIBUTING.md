# Contributing to ofiq-syngen

Thanks for considering a contribution. This document describes how to
report issues, propose changes, and run the test suite.

## Reporting issues

Use the GitHub issue tracker. Choose the template that fits:

- **Bug**: a degrader produces wrong output, a test fails, or
  documentation contradicts behavior.
- **Standards alignment**: a clause reference, OFIQ source citation,
  or alignment classification looks wrong.
- **Feature request**: a new degrader, CLI flag, or workflow.

Before filing, search existing issues. Standards-alignment questions
should reference the relevant ISO/IEC 29794-5, ISO/IEC 19794-5, or
ICAO 9303 Part 9 clause.

## Development setup

```bash
git clone https://github.com/astoreyai/ofiq-syngen
cd ofiq-syngen
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,pandas]"
pytest tests/
```

The test suite must pass on every commit. CI runs on Python 3.9 - 3.13.

## Running tests

```bash
pytest tests/                           # full suite
pytest tests/test_components_per_component.py -v   # parametrized per-component
pytest tests/test_standards_mapping.py             # CSV / Python sync check
pytest tests/test_cli_presets.py                   # CLI preset behavior
pytest tests/test_ofiq_parity.py                   # OFIQ binary parity (skipped without binary)
```

To enable the OFIQ-binary parity tests, set `OFIQ_BINARY=/path/to/OFIQSampleApp`
and populate the manifest per `tests/fixtures/ofiq_parity/REGENERATE.md`.

## Adding a new component

1. Add the component to `STANDARDS_REFS` in `src/ofiq_syngen/standards.py`
   with all six clause/alignment fields populated.
2. Add the same row to `docs/standards/MAPPING.csv` (the test suite
   asserts the two stay in sync).
3. Implement the degrader in `src/ofiq_syngen/components.py` and call
   `_register(...)`. The parametrized test suite picks it up
   automatically.
4. Cite the OFIQ C++ source line range in `docs/standards/PROVENANCE.md`.
5. Add a row to the appropriate section of `docs/theory/COMPONENT_STATUS.md`.
6. Run `pytest tests/` and confirm all per-component test rows pass for
   the new component.

If the component is context-noop without a `FaceContext` (i.e., requires
landmarks/parsing to vary with severity), add it to `CTX_NOOP_COMPONENTS`
in `tests/test_components_per_component.py`. If it is deterministic
regardless of seed, add it to `SEED_INSENSITIVE_COMPONENTS`. If its
image-delta is non-monotonic by design (e.g., geometric translation),
add it to `MONOTONICITY_EXEMPT` with a comment.

## Modifying an existing degrader

Run the full suite before and after your change. If the change moves
the OFIQ score envelope for any (image, component, severity) triple in
the parity manifest, regenerate the manifest with the OFIQ binary and
update the CHANGELOG under "Standards alignment."

## Coding style

- Type hints on all public functions.
- Docstrings on all public symbols (one-sentence summary, then
  parameters and returns).
- Avoid em-dashes and en-dashes in user-facing text.
- Tests use `pytest`; new test files belong in `tests/` and follow the
  existing naming convention.

## Commits and PRs

- Conventional commit messages preferred but not required. Be specific
  about what changed and why.
- One concern per PR. Mixing unrelated changes makes review harder.
- Update `CHANGELOG.md` under the next version's `Added` /
  `Changed` / `Fixed` section.
- The CI workflow runs `pytest` and `ruff`. Both must pass.

## OFIQ upstream coordination

If your change touches OFIQ-aligned behavior, read [`OFIQ_UPSTREAM.md`](OFIQ_UPSTREAM.md)
first. Some changes are better proposed upstream to BSI OFIQ-Project.

## License

By contributing you agree your contribution is licensed under MIT, the
same license as the rest of the project (see `LICENSE`).
