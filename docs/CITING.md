# Citing ofiq-syngen

When using `ofiq-syngen` in published work, cite **all four** of the
references below together. The package is a companion to the OFIQ
reference implementation; citing only the package omits the standards
and reference-implementation chain that make the perturbations
meaningful.

## The package

```bibtex
@software{ofiq_syngen,
  author       = {Storey, Aaron W.},
  title        = {{ofiq-syngen: ISO/IEC 29794-5 component-aligned
                   synthetic face image quality degradation}},
  version      = {0.3.0},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.PENDING},
  url          = {https://github.com/astoreyai/ofiq-syngen}
}
```

> The Zenodo DOI is minted by the GitHub Actions release workflow once
> `astoreyai/ofiq-syngen` is published. Until then, cite the GitHub
> repository URL.

## The OFIQ reference implementation

```bibtex
@inproceedings{schlett2024ofiq,
  author       = {Schlett, Torsten and Damer, Naser},
  title        = {{OFIQ: an open-source software library for face image
                   quality assessment}},
  booktitle    = {IEEE International Joint Conference on Biometrics (IJCB)},
  year         = {2024}
}
```

> Verify the exact title, page numbers, and author list against the
> published version before submission. The OFIQ project page at
> <https://www.bsi.bund.de/OFIQ> hosts the canonical citation.

## The quality measurement standard

```bibtex
@standard{iso29794_5_2024,
  title        = {{Information technology -- Biometric sample quality --
                   Part 5: Face image data}},
  author       = {{ISO/IEC JTC 1/SC 37}},
  year         = {2024},
  organization = {International Organization for Standardization},
  number       = {ISO/IEC 29794-5:2024}
}
```

## The data interchange format / ICAO chain

If your work targets travel-document scenarios or ICAO conformance,
also cite the data interchange standard and the ICAO specification it
underpins:

```bibtex
@standard{iso19794_5_2011,
  title        = {{Information technology -- Biometric data interchange
                   formats -- Part 5: Face image data}},
  author       = {{ISO/IEC JTC 1/SC 37}},
  year         = {2011},
  organization = {International Organization for Standardization},
  number       = {ISO/IEC 19794-5:2011}
}

@manual{icao9303_part9,
  title        = {{Machine Readable Travel Documents -- Part 9:
                   Deployment of Biometric Identification and Electronic
                   Storage of Data in MRTDs}},
  author       = {{International Civil Aviation Organization}},
  year         = {2021},
  edition      = {8th},
  organization = {ICAO},
  number       = {Doc 9303 Part 9}
}
```

## Why all four

Each citation answers a different reviewer question:

| Citation | Answers |
|---|---|
| `ofiq-syngen` | "What software was used to generate the synthetic data?" |
| OFIQ paper | "What measurement implementation defines the alignment target?" |
| ISO/IEC 29794-5:2024 | "What standard defines the components being targeted?" |
| ISO/IEC 19794-5:2011 + ICAO 9303 | "What operational standards do these components serve?" |

A paper that cites only `ofiq-syngen` cannot defend the claim that the
perturbations are standards-aligned. A paper that cites only the standard
cannot defend the claim that the targets match a specific measurement
implementation. The four-citation block establishes the full provenance
chain: standard -> implementation -> companion tool -> operational
deployment context.

## Citing per-component algorithm provenance

For methods sections that need to cite the OFIQ source location of each
measurement algorithm, see [`docs/standards/PROVENANCE.md`](standards/PROVENANCE.md).
That document gives per-component file:line citations against the
pinned OFIQ release listed in [`docs/standards/SOURCES.md`](standards/SOURCES.md).

## Citing the multi-standard mapping

The cross-reference between OFIQ components, ISO/IEC 29794-5 aspects,
ISO/IEC 19794-5 clauses, and ICAO 9303 Part 9 clauses is shipped in
[`docs/standards/MAPPING.csv`](standards/MAPPING.csv) (machine-readable)
and [`docs/standards/MAPPING.md`](standards/MAPPING.md) (rendered). Cite
the package version that ships them; the mapping is part of the package's
content contribution.

## Reciprocity

If you build on `ofiq-syngen` (forks, derivatives, integration into a
larger tool), please cite this package and consider opening a pull
request to add your project to a "Used by" list in the README. See
[`OFIQ_UPSTREAM.md`](OFIQ_UPSTREAM.md) for the relationship policy
with the upstream OFIQ project.
