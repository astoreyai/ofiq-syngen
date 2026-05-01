# Standards Sources

Reference material the multi-standard mapping (`MAPPING.md`, `MAPPING.csv`) is
built against. All clause numbers in those documents trace back to the editions
listed here.

## OFIQ reference implementation

| Field | Value |
|---|---|
| Project | BSI OFIQ-Project |
| Repository | <https://github.com/BSI-OFIQ/OFIQ-Project> |
| Version pinned for v0.2.0 of `ofiq-syngen` | **1.1.0** (per `OFIQ-Project/Version.txt`) |
| Algorithm Book | "OFIQ — Open Source Face Image Quality" (BSI, latest at time of pin) |
| C++ source root | `OFIQlib/modules/measures/src/` |
| Citation | Schlett, T., et al. *OFIQ: An open-source library for face image quality assessment.* IJCB 2024. |

> **Discrepancy flag**: `CHANGELOG.md` for `ofiq-syngen` v0.2.0 references "BSI
> V1.2"; the OFIQ-Project source pinned in this workspace reports `1.1.0`.
> Reconcile before tagging v0.3.0 — either update the pinned OFIQ checkout or
> correct the CHANGELOG.

## ISO/IEC 29794-5 (quality measurement)

| Field | Value |
|---|---|
| Title | Information technology — Biometric sample quality — Part 5: Face image data |
| Edition referenced | **ISO/IEC 29794-5:2024** (current published edition) |
| OFIQ Algorithm Book section convention | `§7.3.x` = capture-related; `§7.4.2`–`§7.4.13` = subject-related; `§7.4.10`–`§7.4.11` = geometric/pose; FDIS Annex D = quality requirements without a QAA |
| Note | OFIQ's clause numbering follows the Algorithm Book, which tracks the standard's clause structure. The `S<n>.<m>` shorthand used in `README.md`, `ANALYSIS.md`, and the rest of this docs tree refers to those Algorithm Book sections. |

## ISO/IEC 19794-5 (face image data interchange format)

| Field | Value |
|---|---|
| Title | Information technology — Biometric data interchange formats — Part 5: Face image data |
| Edition referenced | **ISO/IEC 19794-5:2011** + Amd 1:2014, Amd 2:2014 |
| Quality clauses | §8 (scene/photographic), §9 (digital), Annex A (informative quality requirements) |
| Status | Predates 29794-5. ICAO Doc 9303 references this standard for travel-document face-photo specifications. |

## ICAO Doc 9303 (travel documents)

| Field | Value |
|---|---|
| Title | Machine Readable Travel Documents — Part 9: Deployment of Biometric Identification and Electronic Storage of Data in MRTDs |
| Edition referenced | **8th edition (2021)** §3.2 *Quality of the portrait* |
| Underlying spec | References ISO/IEC 19794-5 for portrait technical specifications |
| Practical scope | Passports, visas, eMRTDs, eIDs |

## Mapping confidence

Each row in `MAPPING.csv` carries a `confidence` flag:

- `verified`: clause text in hand or referenced directly in OFIQ Algorithm Book.
- `derived`: clause inferred from cross-references (e.g., 19794-5 → 29794-5 carry-over).
- `uncertain`: best inference; user should verify against the standard before
  citing in publications. **Flagged with `?` in the rendered MAPPING.md.**

## Update protocol

When OFIQ ships a new release, or when ISO/ICAO publish revised editions:

1. Update the pin in this document.
2. Re-run any tests in `tests/test_standards_mapping.py` that depend on clause IDs.
3. Diff `docs/standards/MAPPING.csv` against the new edition; bump `ofiq-syngen`
   minor version if any clause ID moves.
4. Note the change in `CHANGELOG.md` under "Standards alignment."
