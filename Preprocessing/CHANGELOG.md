# Changelog

## 0.7.0 — repository update — 2026-11-05

Includes the P4–P7 development milestones validated on 2026-11-01. This repository
update changes publication documentation; runtime code and tests match the
validated 0.7.0 candidate. No PyPI release or GitHub Actions result is claimed.

- Add explicitly declared monthly pressure-level product adaptation for ERA5,
  MERRA-2 M2IMNPASM 5.12.4 and JRA-3Q GDEX monthly isobaric analyses. Require
  explicit physical-variable mappings and source coordinate/unit evidence;
  retain original units, values, calendars and temporal-statistic metadata.
- Require exact label selection of expver/member streams and other additional
  mapped-field dimensions, including singleton streams. Preserve selected
  scalar coordinates and provenance; never merge complementary versions,
  choose a member implicitly, infer monthly averaging windows or fill gaps.
- Adapt each NetCDF source before strict multi-file comparison. Preserve the
  original compatibility checks, exact grids/calendars and separate source
  provenance; include the adapter in preparation, schema-v1 JSON and CLI.
- Rewrite CF relationships and actual cell-method axis labels while retaining
  parenthesized comments verbatim. Include a runnable three-product schema
  example; provider identity remains a declaration checked against available
  metadata, not an authenticity certificate.
- Remove WRF adapters and associated future dependencies from the project plan
  at the user's request. Historical WRF-Python design references remain history.
- Require an existing explicit geopotential_height standard_name for that
  MERRA-2 role; a length unit alone does not identify ASM layer heights as
  geopotential height. Preserve raw per-source ranges before member selection.
- Verify Windows source and installed wheel at 714 passes/26 skips, core-only
  at 602 passes/138 optional skips, and the HPC installed candidate at 740 passes
  without skips. ERA5 P7 checks record 10 passes and four expected rejections,
  unchanged originals and exact source/API/CLI agreement. All 38 runtime modules
  match byte-for-byte. MERRA-2/JRA-3Q original field values, full-field processing
  and performance speedup remain unvalidated; their synthetic contracts pass.
- Record the authorized repository update and the unchanged stage-specific
  validation evidence in validation.json.

## 0.6.0 — development milestone, included in 0.7.0 — 2026-11-01

- Add physical pressure construction for exactly the CF a*p0+b*ps and ap+b*ps
  hybrid formulas, with explicit units and matched cell-center coefficients.
  Fixed pressure axes retain order; interfaces/bounds are not inferred as centers.
- Add explicit linear/log pressure interpolation using MetPy public kernels
  inside scheduled Dask tasks. GeoCAT supplies the upstream design and optional
  parity comparison, rather than an unused mandatory runtime dependency.
- Preserve exact source hits, invalidate only original adjacent missing-value
  intervals, return NaN outside support, and prohibit extrapolation. Hybrid ps
  and optional fixed-level ps exclude below-ground source and target centers.
- Validate lazy surface/column pressures when executed; combine only vertical
  core chunks while retaining nonvertical field partitions. Keep field units,
  calendar/time and horizontal geometry, audit obsolete vertical CF/QC/ranges,
  and preserve remaining nonvertical dependencies and source close callbacks.
- Add vertical preparation/JSON/CLI options before generic nonfinite masking and
  horizontal remapping, so invalid pressure geometry cannot be silently erased.
  Include a runnable synthetic hybrid example and extend the inactive CI template.
- Verify 593 passes/26 skips on Windows source and installed wheel, 483 passes/
  136 optional skips in core-only, and 619 passes/no skips on the HPC installed
  wheel, including four actual GeoCAT parity cases. ERA5 P6 acceptance records
  12 passes and two expected rejections; linear/log results match independent
  references exactly and originals remain unchanged. All 35 runtime modules
  match byte-for-byte. Hybrid coefficients are synthetic fixtures; original
  model-level products, full-field processing and speedup remain untested.
- At this milestone, 0.4.0–0.6.0 work remained local, uncommitted and unpublished.
  Validation evidence is versioned in validation.json; old phase counts are not
  current acceptance.

## 0.5.0 — development milestone, included in 0.7.0 — 2026-11-01

- Add explicit scalar rectilinear horizontal remapping through optional xESMF/
  ESMPy: bilinear, nearest_s2d, nearest_d2s, conservative and conservative_normed.
  Method choice, degree geometry, conservative cell bounds and static masks are
  validated; no bounds guessing, vector rotation or extensive-total conversion.
- Keep meteorological fields Dask-backed and nonhorizontal chunks unchanged;
  geometry/weight generation is eager. Propagate participating source NaNs,
  mark unmapped destinations missing, and disable dynamic skipna normalization.
- Add local content-addressed weight caches keyed by real coordinates, bounds,
  masks, method, periodicity, fixed options and backend versions. Validate
  checksums on reuse; reject incomplete/corrupt entries and concurrent writers.
- Rebuild target CF geometry, preserve units/time/calendar and valid nonspatial
  links, audit removed spatial fields/QC/area/CRS/ranges, and clear disk packing.
- Compose remapping via prepare_dataset and schema-v1 JSON; inspect builds only
  in-memory geometry weights. Add a runnable synthetic example and explicit
  Linux native-backend CI template (not enabled or claimed as executed).
- Verify 473 passes/25 explicit skips on Windows source and installed wheel,
  406 passes/92 optional skips in core-only, and 498 passes/no skips on the Linux
  HPC installed wheel. Real ERA5 P5 checks: 8 passes, 2 expected rejections,
  unchanged originals; 31 runtime modules match byte-for-byte. Native Windows
  ESMF, complete fields and speedup remain untested.
- This milestone preserved 0.4.0 work locally without a commit or GitHub upload.
  Its validation evidence and platform limits are retained in validation.json.

## 0.4.0 — development milestone, included in 0.7.0 — 2026-11-01

- Add optional cfgrib/ecCodes GRIB group inspection and strict lazy reading,
  preserving forecast axes/statistical windows and disabling sidecar indexes.
- Reject unsupported message layouts that could silently lose fields or time
  precision; preserve float64 decoded values and record selected source groups.
- Add local consolidated Zarr v2 reading/writing with a completion manifest,
  metadata and accessed-chunk guards, non-overwriting publication and cleanup
  that retains foreign/replaced destinations.
- Add explicit input/output format policies to preparation and CLI/configuration;
  existing NetCDF defaults remain. Add a synthetic optional-format example.
- Support eager cftime bounds in the shared lazy wrapper with bounded object
  element counts, preserving existing Dask blocks and the NetCDF path.
- Share paired date units/dtype and compatible calendars across time and its CF
  bounds/climatology in NetCDF and Zarr output; reject conflicting overrides.
  Default to integer encoding at the required precision, using microseconds for
  whole-microsecond datetime64 labels and nanoseconds only when needed.
- Before publication, reread date variables and verify exact labels, missing
  positions and calendar semantics, with at most 1,000,000 labels per variable.
  Reject lossy backend roundtrips, including affected nanosecond/NaT and
  distant-year cftime-microsecond paths, without rereading numeric fields.
- Verify 393 tests plus one platform skip on Windows source and installed wheel,
  326 passes/68 optional skips without extras, and 394 passes on HPC. Real ERA5
  checks record 18 passes and 8 expected rejections; GRIB is a derived fixture.
- This milestone kept changes local for the later unified repository update.
  CI remains an inactive template. Verification details are in validation.json.

## 0.3.0 — 2026-11-01

- Add generic CF pressure normalization with Pa/hPa conversion, canonical
  pressure coordinates, synchronized bounds/reordering and lazy fields.
- Add exact expected-time checks and inclusive time selection for datetime64
  and cftime calendars. Complete ISO strings are interpreted in the source
  calendar; no frequency inference or calendar conversion is introduced.
- Add explicit boolean masking with exact label alignment, protected QC/CF
  structure, metadata cleanup and recorded reasons. No imputation is performed.
- Expose pressure and time policies in preparation/JSON configuration; preserve
  existing defaults and monthly/seasonal APIs. Add a synthetic composed example.
- Pass 283 tests each on source, installed wheel and HPC. Real ERA5 tests
  record 12 passes and 5 expected rejections across the merge regression and
  pressure/time/mask suite, with zero failures. See validation.json for scope.
- Keep CI as an inactive template and optional P4–P8 backends unimplemented.

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
