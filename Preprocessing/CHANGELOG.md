# Changelog

## 0.2.1 — 2026-11-01

- Add explicit `metadata_policy="preserve_sources"` to multi-file reading,
  preparation and CLI configuration. File-specific history/dates and descriptive
  ranges are recorded per source without being mistaken for scientific conflicts.
- Retain strict checks on units, calendars, grids, CF references and unknown
  attributes. The default policy remains `strict`.
- Remove file-specific ranges from merged output without computing new extrema;
  preserve semantic variable attributes and CF references through concatenation.
- Add regression tests and bounded real-ERA5 validation for this policy.
- Verify 147 tests on the local source, the installed wheel outside the source
  tree, and the Linux HPC installation. Original 2003/2004 ERA5 V files form a
  lazy 24-month dataset; two small monthly samples match reference values
  exactly, and a NetCDF roundtrip retains semantic attributes and provenance.
  This does not validate full-field processing or claim a measured speedup.
- Update bilingual usage, configuration examples, validation scope and code
  management notes. No new statistical methods or optional data backends are
  introduced.
- Provide the CI workflow as an inactive template under
  `docs/preprocessing/preprocessing-ci.yml.example`; GitHub Actions automation
  is not enabled by this update.

## 0.2.0 — 2026-11-01

### Added

- Lazy xarray/Dask Dataset I/O, strict multi-file concatenation, CF metadata/time checks, rectilinear normalization, geographic subset, nonfinite masking and provenance.
- `prepare_dataset`, `inspect/prepare` CLI with a versioned configuration and an offline synthetic example.
- `MonthlyArchive.read_lazy` and `convert_dataarray_units` for workflows that compose before computing.
- Module-level references/improvements, a package-scoped CI template and lint, and installed-wheel verification.

### Changed

- Source layout is now `src/climate_preprocessing`, with a single version source.
- xarray and Dask are required. Existing monthly/seasonal field operations use labelled chunked execution, with original return types and explicit compute boundaries.
- General output reconstructs encoding after transformations. Unsupported metadata/coordinate conflicts fail explicitly.

### Scope

This implements the first NetCDF preprocessing workflow, not all planned adapters. No new statistical computation is introduced. GRIB/Zarr, regridding, hybrid levels and provider/ station adapters are not implemented in this release. Nothing is published automatically.

## 0.1.0 — baseline

Strict monthly CF readers, complete-season reduction and non-overwriting seasonal output. See the original [source notes](SOURCE_NOTES.md).
