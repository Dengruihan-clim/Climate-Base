# Developing climate-preprocessing

Repository update: 0.7.0, dated 2026-11-05; validation was performed on 2026-11-01. Version 0.3.0 is historical, and development stages 0.4.0–0.6.0 were not separately published.

Use the `src/` package through an installed editable package or wheel. Tests do not modify `sys.path`. On the current Windows workstation use the user-selected Anaconda interpreter for Python and pip; use an isolated dependency location when additional packages are needed.

```text
python -m pip install -e ".[dev]"
python -m ruff check src tests examples tools
python -m pytest tests -q
python -m build
python tools/check_wheel.py dist
python tools/update_manifest.py
```

`check_wheel.py` installs the built wheel into a temporary directory without changing the base environment, checks its import path, and reruns the complete tests from outside the source tree using the same interpreter. Dependencies must already be available. The [CI template](../docs/preprocessing/preprocessing-ci.yml.example) defines Linux/Windows Python 3.11/3.12 jobs and a conservative dependency combination. It is inactive: this update does not enable GitHub Actions, and local/HPC passes are not remote CI results. To enable automation, copy the template to the repository's `.github/workflows/preprocessing.yml` and commit it.

Package text uses LF line endings, enforced by `.gitattributes`, so Git source
archives and local builds have matching bytes. Regenerate `SHA256SUMS.txt` only
after documentation and validation records are final. Before publishing, verify
the manifest against the staged Git objects as well as the sdist; checking only
working-tree bytes can miss Git's line-ending normalization.

Field operations must use xarray/Dask without implicit `.values`, `.load()` or `.compute()`. Small coordinate validation and NumPy inside scheduled array kernels are allowed. Preserve existing chunks when adding lazy fields. Compute only at documented output/legacy-return boundaries. Keep source Dataset resources alive through execution and close them on failure.

Each changed numerical module must include `References` and `Climate-Base improvements`, distinguish original author code from independently written adapters, and state the input units/dimensions/calendar/missing-value contract. Update [THIRD_PARTY.md](THIRD_PARTY.md), examples and both READMEs when capabilities change. Do not label metadata recognition as complete CF certification or unmeasured Dask execution as a guaranteed speedup.

Tests should prove meaningful behavior: analytical values, CF relationships, calendar semantics, rejection of ambiguous inputs, zero Dask field tasks before explicit execution, resource closure, and failure-safe output. Default tests are offline synthetic data; remote providers and heavy backends need separate future integration suites. Preserve the legacy 42-case baseline and add tests for new behavior.

The package does not select a new software license in this refactor. Preserve existing provenance and all third-party licensing; the maintainer decides the license of author-owned code before distribution.

For the optional-format suite, install `.[dev,grib,zarr]` in an isolated
environment and run the same tests. Real ecCodes-generated fixtures and real
Zarr stores are generated only under pytest temporary directories. Core-only
runs skip unavailable backend tests; skips are not evidence of backend support.
Report native ecCodes as well as Python distribution versions. A NetCDF-derived
real-value GRIB fixture is not evidence that an original GRIB product was tested.
Keep unfinished Zarr-store and publication-failure cases in the test matrix.

Horizontal regridding requires an independently installed native ESMPy/ESMF
environment plus `.[dev,regrid]`; the Python extra alone cannot supply ESMF.
Follow the [official installation guide](https://xesmf.readthedocs.io/en/stable/installation.html).
Do not alter the workstation's selected Anaconda environment to force a native
backend installation. Missing backend tests skip and are not integration passes.
The grid/cache/metadata contract tests remain runnable without xESMF.

Keep analytical fields, constant fields where the method preserves constants,
native nearest-s2d/d2s distinctions, static masks, partial-coverage normalization,
unmapped/NaN behavior and periodic seams in the real-backend suite. Conservative
integrals must use matching ESMF great-circle polygon geometry and coverage;
point means or arbitrary cosine-latitude weights are not conservation evidence.
Weight construction may eagerly read coordinate/bound geometry, but not the
meteorological field graph. Test cache corruption, same-shape/different-grid
identity, CF/units/calendar retention, and `inspect` without file writes.
Current version-specific source/wheel/HPC results belong in `validation.json`;
do not substitute the historical 0.4.0 suite counts for current validation.

The 0.5.0 installed candidate was tested on Linux Python 3.12.16 with xESMF
0.9.2 and ESMPy/ESMF 8.9.1: 498 passed with no skips. Windows source and the
tree-external candidate wheel each recorded 473 passed and 25 skips (24 native
backend cases plus one symbolic-link privilege case); core-only recorded 406
passed and 92 optional-backend skips. The 31 runtime modules were byte-matched
between local source, candidate package and HPC. That stage's final
documentation-only package preserved identical runtime, tests, examples and
tools and was retained locally without redeploying to HPC. Those are historical P5 results and do not
replace current validation. Curvilinear grids and rotation remain unsupported.

The 0.6.0 `vertical` extra installs MetPy for the actual public linear/log kernels.
GeoCAT uses those same kernels and remains an optional comparison dependency,
not an unused runtime requirement. Pressure/formula/CF tests need no MetPy.
Cover both hybrid formulas, unit conversions, center/interface ambiguity, exact
labels, ascending/descending targets, original adjacent gaps, exact source hits,
terrain and missing ps, plus delayed invalid-column failures. Source fields and
ps must remain lazy; only the full vertical core is rechunked. Coordinate/time
bounds decoding may execute geometry tasks and must be counted separately from
meteorological fields. Pipeline order is units → vertical → nonfinite mask →
horizontal remapping, preserving invalid-ps errors.

The first P6 stage is complete: 0.6.0 Windows source and tree-external installed
wheel each recorded 593 passed and 26 skipped; core-only recorded 483 passed
and 136 skipped; the installed HPC candidate passed all 619 tests with no skips.
The wheel run completed in 123.27 seconds with seven warnings; this is test-suite
duration, not a field-performance benchmark. All 35 runtime modules match
byte-for-byte, and four actual GeoCAT comparisons passed. Real ERA5 checks
recorded 12 passes and two expected rejections, with zero linear/log comparison
error. Hybrid fixtures use synthetic CF coefficients, not an original model-level
product. These are historical P6 results. Current P7 validation belongs in
`validation.json`; public records must omit private accounts and absolute source-data
paths. WRF adaptation is outside the implementation scope.

The 0.7.0 P7 adapter tests must distinguish product identity, physical roles and
statistics. Explicit mappings do not override conflicting units or metadata.
Test renamed source time axes, exact selectors (including singleton expver/member
axes), nonmutation, retained scalar selection/provenance, CF-reference remapping,
Dask field laziness, strict multi-file combination and installed CLI behavior.
Do not manufacture time bounds or `time: mean`. Synthetic MERRA-2/JRA-3Q tests
verify contracts; actual ERA5 checks are separately scoped. Record missing
metadata as an unverified caller declaration rather than verified product identity.
The supported profiles and exclusions are listed in both READMEs. No WRF or new
runtime dependency is added. MERRA-2 geopotential height requires an existing
`standard_name="geopotential_height"`; matching length units do not establish
that interpretation. Test missing and ad hoc declarations as rejections. For
`preserve_sources`, capture allowlisted range attributes before adapter selection
and retain them per source even when selected outputs discard stale ranges.

The first 0.7.0 P7 stage is validated: Windows source and the tree-external installed candidate wheel each recorded **714 passed, 26 skipped**; core-only recorded **602 passed, 138 skipped**; the installed HPC candidate recorded **740 passed, no skips** (105 warnings). All 38 runtime modules and 77 Python files were byte-matched. Real ERA5 P7 checks recorded **10 passes and 4 expected rejections**, with zero original-value/API/CLI comparison error and unchanged source sizes/mtime. All three synthetic profile examples and their CLI runs passed. Original MERRA-2/JRA-3Q samples remain unvalidated. P8 station support is next; WRF is excluded, no runtime dependency was added, and these results remain dated 2026-11-01 for the 0.7.0 repository update.
