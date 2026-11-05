# Climate preprocessing with xarray and Dask

[中文](README.md)

**Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).**

## 0.7.0 repository version: named monthly pressure-level products

The first P7 stage is implemented and validated. It adds `adapt_product`: callers explicitly choose a product, variable roles, four coordinate axes and extra-dimension labels. The result is an xarray/Dask Dataset with canonical names. This **0.7.0** GitHub repository update is dated **2026-11-05**, with validation performed on **2026-11-01**. Version 0.3.0 is historical; development stages 0.4.0–0.6.0 were not separately published. WRF is outside the current implementation scope; P8 station support is next.

| `product` | Supported product representation |
| --- | --- |
| `era5-monthly-pressure-levels` | ERA5 whole-month pressure-level reanalysis means, `monthly_averaged_reanalysis` / `moda`, decoded onto one-dimensional latitude/longitude axes |
| `merra2-m2imnpasm-5.12.4` | MERRA-2 `M2IMNPASM.5.12.4` / `instM_3d_asm_Np`, monthly means of instantaneous assimilated fields |
| `jra3q-monthly-isobaric-analysis` | GDEX `d640002` `anl_p` / `anl_p125` monthly-mean isobaric analyses in CF NetCDF |

JRA-3Q `anl_p` is GDEX's regular Gaussian representation converted from the native reduced grid, with nonuniform latitude spacing; `anl_p125` is the 1.25-degree grid. Cropped subsets are permitted without imposing global dimensions. Hourly data, monthly diurnal means, standard deviations, forecasts, surface products and original hybrid model levels are outside these profiles. See the directly checked [official product evidence](../docs/preprocessing/upstream-references.md#p7-具名产品证据).

`variables={"northward_wind": "v"}` explicitly maps a physical role to a source variable; provider names and short names never establish physical meaning. Roles are limited to `air_temperature`, `eastward_wind`, `northward_wind`, `specific_humidity`, `geopotential`, `geopotential_height` and `lagrangian_tendency_of_air_pressure`. `coordinates` can override profile defaults for `time/pressure/latitude/longitude`: ERA5 uses `time/level/latitude/longitude`, MERRA-2 uses `time/lev/lat/lon`, and JRA-3Q uses `time/pressure_level/lat/lon`. Other representations need explicit mappings.

Decoded time must be ordered with at most one record per calendar year/month; fixed pressure must retain its original strictly monotonic order. Selected physical fields must have exactly the four axes. Every additional mapped-field dimension and known `expver/member/number/realization` coordinate requires an exact label in `selectors`, including singleton dimensions and known scalar coordinates; selected coordinates and provenance remain. There is no expver merging, first-nonmissing selection or ensemble averaging. Only mapped fields are claimed as adapted; other fields remain with shared selections and coordinate-reference updates.

Adaptation validates units and existing metadata, renames variables and updates CF references. Missing `standard_name` values are generally filled from explicit roles and audited. MERRA-2 `geopotential_height` is an exception: the source must already declare `standard_name="geopotential_height"`; compatible `m` or `gpm` units alone are insufficient. It does not sort coordinates, silently convert units, calculate statistics, insert `time: mean` or invent time bounds. ERA5 geopotential in `m2 s-2` and JRA geopotential height in `gpm` are distinct; MERRA-2 `H` ad hoc height declarations do not establish an automatic geopotential-height conversion. Unverified ad hoc `standard_name` values are rejected and require prior explicit scientific interpretation by the caller. Conflicting product/statistical/physical metadata is rejected. Missing evidence is recorded as a caller declaration, not complete product certification or proof of full monthly sampling. Unit conversion and vertical/horizontal interpolation remain separate explicit operations.

```python
from climate_preprocessing import adapt_product, open_dataset, write_dataset

with open_dataset("monthly.nc", chunks={"valid_time": 1}) as source:
    result = adapt_product(
        source,
        product="era5-monthly-pressure-levels",
        variables={"northward_wind": "v"},
        coordinates={"time": "valid_time", "pressure": "pressure_level",
                     "latitude": "latitude", "longitude": "longitude"},
        selectors={"expver": 1},  # only if this dimension exists; use its actual label
    )
    write_dataset(result, "adapted.nc")
```

`prepare_dataset(..., adapter={...})` and the schema-v1 JSON `adapter` accept the same four options for NetCDF or derived local Zarr representations; combining them with `input_format="grib"` is rejected. Standalone `adapt_product` also requires decoded monthly fixed-pressure data and does not certify original GRIB time/step semantics. Each input is adapted before strict multi-file combination. Explicit `preserve_sources` captures each source file's allowlisted range attributes before selection and removes ranges invalidated by selection, retaining per-source provenance. Source expver lengths may differ, but selected scalars, grids, units and calendars must still agree. `chunks` and `adapter.coordinates` use source names; top-level `time` remains `time`, and subsequent selection, units and vertical/horizontal operations use canonical output names. See the synthetic [product_adapters.py](examples/product_adapters.py) example; it does not certify original provider fields.

For 0.7.0, Windows source and the actually installed tree-external candidate wheel each recorded **714 passed, 26 skipped**; core-only recorded **602 passed, 138 skipped**. The installed HPC candidate recorded **740 passed, no skips** (105 warnings). All 38 runtime modules and 77 Python files were byte-matched; see [validation.json](validation.json).

Real ERA5 P7 sample checks recorded **10 passes and 4 expected rejections**, with zero maximum error against original values and between API/CLI outputs; source sizes and modification times remained unchanged. Synthetic examples for all three profiles and their CLI runs passed. Original MERRA-2/JRA-3Q samples remain unvalidated; their synthetic checks establish contracts only. Historical P4–P6 counts remain separate. Full fields and speedup are untested. Public records omit private accounts and absolute source-data paths.

## 0.6.0 historical stage: fixed/hybrid pressure interpolation

The first fixed/hybrid-center P6 stage is implemented and validated, with `hybrid_pressure` and `interpolate_pressure` retaining xarray/Dask field operations. The following results belong to historical 0.6.0 and do not replace current P7 validation.

For 0.6.0, Windows source and the actually installed tree-external wheel each recorded **593 passed, 26 skipped**; core-only recorded **483 passed, 136 optional-backend skips**. The installed HPC candidate recorded **619 passed, no skips**. All 35 runtime modules match byte-for-byte, and four actual GeoCAT comparison cases passed. Skips do not establish backend validation; see [validation.json](validation.json) for environments and scope.

Real ERA5 sample checks recorded **12 passes and 2 expected rejections**, with zero maximum error for both linear and log comparisons. Hybrid cases use synthetic CF coefficients and do not validate an original model-level product. Full fields and performance speedup remain untested; public records omit private accounts and absolute source-data paths.

`hybrid_pressure` needs no optional numerical backend. Install `.[vertical]` (MetPy 1.7.1 series) for interpolation. GeoCAT uses the same public MetPy one-dimensional kernels, so this package calls those directly; GeoCAT remains a reference and optional comparison dependency. Keep workstation installation/testing on the selected Anaconda Python and isolated dependency locations.

```python
from climate_preprocessing import open_dataset, hybrid_pressure, interpolate_pressure, write_dataset

with open_dataset("hybrid.nc", chunks={"time": 1, "lev": 16}) as source:
    pressure_pa = hybrid_pressure(source, coordinate="lev")  # lazy Pa pressure field
    result = interpolate_pressure(
        source, [850., 500., 200.], variables=["temperature"],
        method="log", coordinate="lev", target_units="hPa",
    )
    write_dataset(result, "new_pressure.nc")
```

Sources are fixed physical pressure coordinates with explicit units, or CF `atmosphere_hybrid_sigma_pressure_coordinate` using exactly one complete formula:

| Formula | Required units and layout |
| --- | --- |
| `p = a*p0 + b*ps` | One-dimensional cell-center `a`/`b` coefficients with units `1`; positive scalar reference pressure `p0`; surface pressure `ps` |
| `p = ap + b*ps` | Cell-center pressure coefficient `ap`, dimensionless `b` with units `1`, and surface pressure `ps` |

`p0/ap/ps` require explicit pressure units accepted by the existing Pa/hPa/millibar conversion table. The source needs at least two centers, and coefficient dimensions/labels must match exactly; n+1 interfaces and bounds are not reinterpreted as centers. Coordinate/coefficient/reference geometry may be read eagerly, while ps and meteorological fields remain lazy. Standalone derived pressure carries generic source-level labels, removes dangling hybrid-formula declarations, and records formula/source names in non-CF JSON metadata without changing the source Dataset.

Choose `method="linear"` for pressure or `method="log"` for log-pressure explicitly. Targets use `target_units` (Pa/hPa, default hPa), must be positive, finite and unique, and preserve their requested strictly increasing/decreasing order. The output dimension is `pressure`. Only explicitly selected floating-point physical fields are accepted. Results are center point values, not layer means or conservative vertical remapping; target bounds are not invented.

There is no extrapolation or interpolation across missing layers. Source NaN/inf invalidates neighboring intervals, while an exact finite source-level hit remains valid beside a missing level. Every valid pressure column must be strictly monotonic; nonmonotonic hybrid levels are not sorted. Fixed levels may use `surface_pressure="ps"` to exclude below-ground sources/targets; without it terrain is unknown and only source-center support is enforced. Hybrid formulas always use their CF ps term. Missing ps makes the entire column NaN. Nonpositive/infinite ps, invalid derived pressure and nonmonotonic columns fail when the corresponding Dask task executes. An above-ground target below the lowest valid source center still returns NaN.

The numerical kernel rechunks each full source column into one vertical block while retaining nonvertical chunks. Choose time/horizontal chunks to fit memory. Outputs retain units, time/calendars, horizontal metadata, surface pressure and independent variables; obsolete coefficients, source-level QC, bounds and invalid links are removed and audited. Keep source resources open through compute/write; the result shares its source close callback.

`prepare_dataset` and schema-v1 JSON accept `vertical`, for example `"vertical": {"target_levels": [850, 500], "variables": ["temperature"], "method": "log", "coordinate": "lev", "target_units": "hPa"}`. Execution order is normalization/selection and units → vertical interpolation → nonfinite masking → optional horizontal regridding, so invalid infinite ps is not silently converted into missing data first. `inspect` builds graphs without writing outputs or executing meteorological fields; readers may decode time/bounds labels, which are coordinate tasks rather than field calculations. See the runnable synthetic [vertical_pressure.py](examples/vertical_pressure.py) example.

## 0.5.0 historical stage: horizontal regridding and weight caching

Version 0.5.0 completed the first rectilinear P5 stage. Its changes and the earlier 0.4.0 work were retained locally without separate publication and are included in the 0.7.0 repository version. xESMF/ESMPy/ESMF supplies its numerical kernel, while meteorological fields remain lazy xarray/Dask arrays without new statistical computation. Curvilinear grids, vector rotation and other P5 extensions require separate validation. The following test counts belong to the historical 0.5.0 stage.

For 0.5.0, Windows source and the tree-external candidate wheel each recorded **473 passes, 25 skips**: 24 native xESMF/ESMPy cases and one symbolic-link privilege case. The core-only environment recorded **406 passes, 92 optional-backend skips**. The installed candidate wheel on Linux HPC recorded **498 passes, no skips or failures**, using isolated Python 3.12.16 with **xESMF 0.9.2 and ESMPy/ESMF 8.9.1**. All 31 runtime modules match byte-for-byte between local source, the candidate package and HPC.

Repeated installed-wheel checks on real ERA5 P5 samples recorded **8 passes, 2 expected rejections, no failures**: zero field tasks during graph construction, exact same-grid identities, known-index nearest-neighbor values and NetCDF roundtrip, with original file sizes and modification times unchanged. The older P4 real-data suites were not rerun in either 0.5.0 or 0.6.0 and are not added to those stages’ results. Full fields and performance speedup remain untested. That historical stage's final documentation package was retained locally after byte-level verification of runtime/test/example/tool files, without redeploying to HPC. See [validation.json](validation.json).

Core imports do not load the optional backend. The `.[regrid]` extra supplies xESMF's Python dependencies, **not a working native ESMF installation**. Install ESMPy/ESMF first in a separate supported environment; see the [official xESMF installation guide](https://xesmf.readthedocs.io/en/stable/installation.html) for Linux/conda-forge. Local scripts continue to use the selected Anaconda Python without modifying the existing scientific environment.

```python
from climate_preprocessing import prepare_dataset, rectilinear_grid, regrid_dataset, write_dataset

target = rectilinear_grid(
    latitude=[-10, 10], longitude=[170, 190],
    latitude_bounds=[-20, 0, 20], longitude_bounds=[160, 180, 200],
)
with prepare_dataset("input.nc", chunks={"time": 1}) as source:
    result = regrid_dataset(
        source, target, variables=["temperature"], method="bilinear",
        weights_dir="weights",  # explicit local cache; omit to avoid weight files
    )
    write_dataset(result, "new_regridded.nc")
```

Both grids need ascending one-dimensional `latitude` / `longitude` dimension coordinates, explicit degree units and at least two centers per axis. Unwrapped longitudes such as 170, 180, 190 can cross the dateline. Duplicate cyclic endpoints, curvilinear/projected grids and ambiguous wrapped bounds are rejected. Conservative methods require explicit contiguous CF bounds on both axes; no cell edges are inferred. Select floating-point scalar fields explicitly. Integer categories, extensive cell totals and grid-relative vector rotation are outside this initial scope.

| Explicit method | Meaning |
| --- | --- |
| `bilinear` | Bilinear interpolation; unmapped destinations are NaN, without extrapolation |
| `nearest_s2d` | One nearest source value per destination, using the backend's global nearest-point search domain |
| `nearest_d2s` | Each source selects a destination and collocated contributions are summed; constants and intensive magnitudes are not generally preserved |
| `conservative` | Normalizes by full destination area; integral conservation requires matching coverage and ESMF spherical-polygon geometry |
| `conservative_normed` | Normalizes by covered destination area; differs under partial coverage and does not promise full-area integral conservation |

Areas use ESMF's spherical polygons with great-circle edges. Ordinary point means or arbitrary cosine-latitude weights are not conservation evidence. `periodic=True` is for nonconservative methods on global source grids whose explicit longitude bounds prove a 360-degree span; conservative geometry handles closure and rejects this flag.

Python `source_mask` / `target_mask` arguments require eager static boolean DataArrays with exactly matching ordered grid labels; True means valid. No masks are inferred from QC or missing fields. Coordinates, declared bounds, masks and weight generation are explicit eager geometry operations; the meteorological field graph stays lazy. Source NaNs propagate through participating weights, with no `skipna` renormalization or filling. Unmapped/masked destinations are NaN. Nonhorizontal chunks are retained, while horizontal output chunks are bounded by source chunk lengths and target sizes. Performance and memory usage still depend on the selected chunks and data size.

Cache keys include coordinates, bounds, masks, method, periodicity, fixed numerical options and backend versions; matching shapes are insufficient. Local caching uses exclusive locks, staging, non-overwriting publication and SHA256 manifests. Corrupt, incomplete or incompatible entries fail instead of being silently repaired; stale locks are not automatically deleted. This protocol requires a local filesystem supporting hard links. Provenance records digests rather than private absolute paths.

Outputs retain selected fields and grid-independent source variables. Obsolete spatial fields, QC/area/CRS references, ranges and packing are removed and audited. Units, time/calendar labels and surviving CF links remain intact; unsupported invalidated hybrid-formula dependencies fail. Keep the source Dataset open until computation completes; the result shares its close callback.

`prepare_dataset(..., regrid={...})` and schema-v1 JSON accept `regrid`. Source normalization, selection, units and nonfinite masking run before regridding. JSON does not execute mask expressions, and `weights_dir` resolves relative to the configuration file:

```json
{
  "schema_version": 1,
  "input": "input.nc",
  "chunks": {"time": 1},
  "regrid": {
    "target_grid": {"latitude": [-10, 10], "longitude": [170, 190]},
    "variables": ["temperature"],
    "method": "bilinear",
    "weights_dir": "weights"
  },
  "output": "new_regridded.nc"
}
```

`inspect` also needs a working backend and may generate geometry weights in memory, but writes neither cache nor output and executes no meteorological fields. `prepare` persists the requested cache and output. The runnable [horizontal_regrid.py](examples/horizontal_regrid.py) example and CLI inspect/prepare with its generated configuration passed on HPC. See [THIRD_PARTY.md](THIRD_PARTY.md) for references and independent changes.

## 0.4.0 historical stage: optional GRIB / Zarr

Version 0.4.0 implemented P4 and was retained locally with its test deployment, without separate publication; its changes are included in the 0.7.0 repository version. Its adapters independently compose official cfgrib/ecCodes and xarray/Zarr APIs; no decoder is vendored and no statistical calculation is added.

Install only the optional backends needed; importing the core package does not load them:

```bash
python -m pip install -e ".[grib,zarr]"
```

Historical 0.4.0 validation recorded **393 passed, 1 platform-limited skip** for both Windows source and installed-wheel suites; **326 passed, 68 optional-backend skips** in the core-only environment; and **394 passed** on Linux HPC. At that stage, three real ERA5 suites totaled **18 passes, 8 expected rejections, no failures**, with original files unchanged. The GRIB fixture was derived from actual NetCDF sample values, not an original provider GRIB archive. Those three real-data suites were not rerun in either 0.5.0 or 0.6.0 and are not current validation evidence. Full fields and speedup were also untested at that stage.

Historical 0.4.0 configurations used cfgrib 0.9.15.1, Zarr 2.18.7 and numcodecs 0.15.1. Windows used ecCodes Python bindings 2.50.0 / native library 2.50.0; Linux HPC used bindings 2.50.0 / native library 2.44.0, both passing that stage's format tests. Binding and loaded native-library versions were recorded separately. The workstation retained the selected Anaconda interpreter and workspace dependencies, while the historical HPC versioned environment reused its existing scientific stack. Later isolated Linux environments are recorded separately in their stage-specific validation.

```python
from climate_preprocessing import inspect_grib_groups, open_grib, open_zarr, write_zarr

print(inspect_grib_groups("input.grib2"))  # group headers, without field decoding
with open_grib("input.grib2", filter_by_keys={"typeOfLevel": "isobaricInhPa"}) as ds:
    # time is the forecast reference time, step is lead time; valid_time may be 2-D.
    write_zarr(ds, "new_output.zarr")  # explicit execution; existing targets refused
with open_zarr("new_output.zarr") as ds:
    print(ds.sizes)  # fields remain Dask-backed
```

GRIB accepts local pure concatenations of single-field physical messages. Heterogeneous level/grid/step-type groups need explicit filtering; duplicate slots, sparse message products, multi-field GRIB2 and nonzero reference seconds are rejected to avoid silent field loss or time truncation. Statistical windows are retained in provenance without converting averages/accumulations to instantaneous fields. Header inspection may read complete encoded message bytes but does not request or decode meteorological field values. No `.idx` is written, the geographic cache is disabled, and float64 preserves ecCodes decoded precision.

Zarr initially supports local consolidated **v2 root groups** on filesystems supporting hard links. A sibling staging store is completed first; a new target is reserved exclusively, files are published, and a versioned completion marker is published last. This is not a cross-platform atomic directory replacement or a power-loss durability guarantee. Readers require the marker by default, verify metadata hashes/file inventory and check source state during block reads. Missing chunks, metadata changes and target conflicts fail explicitly. Preserve modification times when migrating managed stores, or rewrite from a validated source. Use `require_complete=False` only for independently validated external v2 stores: these have no package completion guarantee, and initially absent chunks retain standard Zarr fill semantics.

Zarr v3, cloud/object stores, nested groups, append/region overwrite and GRIB writing remain unsupported. Packing floating meteorological fields as integers requires a valid explicit `_FillValue`; field packing and precision are caller choices. Original data stay read-only; keep sources available until execution and close Dataset resources afterwards.

NetCDF and Zarr writers share a date-encoding policy. A time variable and its CF `bounds` / `climatology` variables use consistent `units`, `dtype` and compatible calendars; explicit conflicts are rejected. The default storage dtype is `int64`, with units chosen for the precision needed by the linked date group. Nanosecond arrays whose labels are all whole microseconds use microseconds for cftime/legacy-reader compatibility; nanoseconds are used only when actual labels need sub-microsecond precision. Override date encoding with a paired `units` and `dtype`; integer storage alone does not prove precision was retained.

Before publishing the final file or Zarr completion marker, writers reread identified decoded date variables, including `valid_time` and time bounds/climatology, and compare every label's precision, missing-date position and calendar semantics. The limit is **1,000,000 labels per date variable**, counting each endpoint in multidimensional bounds; larger date arrays are refused. If an older xarray path rounds mixed nanosecond/NaT dates or distant-year cftime microseconds, publication is rejected instead of accepting altered times. Date encoding selection and verification occur at the explicit write boundary. They do not add a complete reread of meteorological numeric fields, convert calendars or perform statistical calculations.

`prepare_dataset` / schema-v1 config add explicit `input_format` (default `netcdf`) and `input_options`. GRIB accepts `filter_by_keys`; Zarr accepts `require_complete`. Both accept one selected input, without suffix guessing. CLI `output_format` is `netcdf` or `zarr`, defaulting to the existing NetCDF behavior. GRIB `expected_time` normally checks the reference-time axis; valid times are never flattened into an analysis series.

```json
{
  "schema_version": 1,
  "input": "input.grib2",
  "input_format": "grib",
  "input_options": {"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
  "pressure": {"pressure": "isobaricInhPa", "target_units": "hPa"},
  "output": "new_prepared.zarr",
  "output_format": "zarr"
}
```

Run `python examples/optional_formats.py new_demo` for synthetic NetCDF→Zarr data and an inspect config. See [THIRD_PARTY.md](THIRD_PARTY.md) for references/improvements and [validation.json](validation.json) for current verification scope.

## 0.3.0 historical delivery: pressure, expected time indexes and user masks

Version 0.3.0 passed **283 tests each** on the local source, a wheel installed outside the source tree, and Linux HPC. The original merge regression recorded 5 passes and 1 expected rejection; the new pressure/time/mask validation recorded 7 passes and 4 expected rejections, with no failures or unrun cases. A January 2004 sample of two levels and 16×32 points matched prior reference values exactly; construction ran zero field tasks, masking and output roundtrips agreed, and original sizes/mtimes were unchanged. Expected rejections cover incomplete indexes, empty selections, misaligned masks, output overwrite and the original strict attribute policy. Full fields and speedup were not benchmarked. See the [0.3.0 validation record](https://github.com/Dengruihan-clim/Climate-Base/blob/1e48f285ee5760ac42253a6c6ab01d91b29f880f/Preprocessing/validation.json).

The next P2/P3 operations keep field transformations lazy through xarray/Dask:

- `normalize_pressure(ds, pressure="level", target_units="hPa", order="ascending")` identifies a one-dimensional physical pressure axis from explicit units and CF evidence, converts Pa/hPa and reorders fields/bounds together. The canonical coordinate is `pressure`. It does not reinterpret arbitrary model levels or interpolate vertically.
- `validate_time(ds, expected=...)` compares the entire input against an explicit expected index, including internal and endpoint gaps. Decoded dates or complete ISO strings are accepted in the source calendar; no frequency inference, filling or calendar conversion occurs.
- `subset_time(ds, start=..., end=...)` selects an inclusive interval of exact instants, retaining time bounds and auxiliary variables. `YYYY-MM-DD` means midnight; partial year/month strings and empty selections are rejected.
- `apply_mask(ds, condition, variables=[...], reason="...")` retains True and masks False. Target fields and a reason are required. Shared dimension labels must match exactly in the same order; dimension broadcasting is allowed, implicit intersection or realignment is not. QC flags and CF structural variables are protected. No product-specific QC meaning is guessed.

`prepare_dataset` and schema-v1 JSON accept optional `pressure`, `expected_time` and `time_range`. The complete input index is checked before selection. Surface datasets do not need pressure unless requested. Compose arbitrary masks afterwards with the Python API; JSON does not execute expressions.

```python
from climate_preprocessing import apply_mask, prepare_dataset, write_dataset

with prepare_dataset(
    "monthly.nc", chunks={"time": 1},
    pressure={"pressure": "level", "target_units": "hPa", "order": "ascending"},
    expected_time=["2000-01-01", "2000-02-01", "2000-03-01"],
    time_range={"start": "2000-02-01", "end": "2000-03-01"},
) as ds:
    selected = apply_mask(ds, ds.quality == 0, variables=["temperature"],
                          reason="Retain quality == 0 according to this product's definition")
    write_dataset(selected, "new_masked.nc")
```

Variable names and QC meanings must match the actual product. Keep independently opened mask files alive until execution ends; closing the result closes its shared source Dataset, but does not own an external mask file. Run `python examples/pressure_time_mask.py new_demo` for a fully synthetic pressure/time/QC roundtrip with no downloads.

Pressure centers and bounds endpoints must currently be strictly positive; zero-pressure top interfaces need a later interface-coordinate workflow. Masked integer fields use floating point to represent NaN. Retained integers that cannot be represented exactly raise when the relevant Dask block executes, preventing silent value changes.

Module references and independently implemented improvements are listed in [THIRD_PARTY.md](THIRD_PARTY.md). Initial GRIB/Zarr v2 support dates from local 0.4.0; 0.5.0 adds horizontal regridding under the contracts above. Version 0.6.0 adds the fixed/hybrid-center interpolation described above. Version 0.7.0 adds three named monthly pressure-level profiles. Calendar conversion and the Meteostat provider remain planned.

## NetCDF preprocessing workflow (available since 0.2.1)

New public functions use **xarray Dataset/DataArray and Dask-backed fields**. Reading, coordinate normalization, geographic selection, masking and unit conversion stay lazy until `write_dataset` or an explicit `.compute()`. Small coordinate/time/bounds arrays may be inspected eagerly; NumPy is used inside numerical blocks. Tests verify that constructing transformations does not run field tasks. This is not a measured speedup claim; throughput depends on chunks, data size and I/O.

Implemented: local NetCDF single/multiple-file I/O, strict CF references and decoded time checks, rectilinear coordinates, dateline-aware subsetting, nonfinite masking, explicit affine units, provenance, non-overwriting output, and `inspect/prepare` commands. No new statistical aggregation is added.

```python
from climate_preprocessing import prepare_dataset, write_dataset

with prepare_dataset(
    ["input_2000.nc", "input_2001.nc"],
    coordinates={"latitude": "lat", "longitude": "lon"},
    chunks={"time": 12, "lat": 90, "lon": 180},
    longitude_convention="0_360",
    region={"latitude_bounds": [-30, 60], "longitude_bounds": [270, 90]},
    units={"temperature": "K"},
) as ds:
    write_dataset(ds, "new_prepared.nc")  # explicit execution boundary
```

Run `python examples/netcdf_prepare.py new_demo` for a fully synthetic example. Its schema-v1 JSON config can be passed to `python -m climate_preprocessing inspect CONFIG` and `python -m climate_preprocessing prepare CONFIG`. The example already writes `prepared.nc`; choose a new output filename before preparing again. Relative paths resolve against the configuration. Explicit input lists are required. Chunks use native input dimension names; unknown dimensions and `None` are rejected. Keep inputs unchanged and available until computation/output finishes, then close the Dataset.

Current limits: one-dimensional rectilinear latitude/longitude; longitude bounds must enclose centers in unwrapped intervals narrower than 180 degrees; explicit existing unit table rather than a full Pint vocabulary. Time supports order/calendar validation, explicit expected-index checks and exact inclusive selection, without frequency inference or calendar conversion. Optional GRIB/Zarr v2, horizontal regridding and fixed/hybrid-center interpolation follow the contracts above. The three product profiles follow the explicit 0.7.0 contracts; Meteostat fetching remains future work.

Multiple files must agree on non-time coordinates, field names, units and decoded dtypes. Duplicate times or incompatible calendars fail. Repeated static data variables must currently be read separately; the reader does not execute large fields to prove equality.

The default `metadata_policy="strict"` also requires all descriptive attributes to agree. Since 0.2.1, `open_mfdataset`, `prepare_dataset` and the JSON configuration accept explicit `metadata_policy="preserve_sources"`. It permits differences only in global `history`, `CONVERSION_DATE`, `date_created`, `date_modified`, `creation_date` and variable `actual_range`, `minimum_value`, `maximum_value`. Their original values are recorded under each provenance input's `source_metadata`, and these file-scoped attributes are removed from multi-file output without recomputing extrema. Single-file attributes remain unchanged. Units, calendars, grids/pressure levels, CF links, `valid_*` limits and every unlisted attribute remain strict. Confirm that inputs represent the same compatible product before selecting this policy.

```python
from climate_preprocessing import open_mfdataset

with open_mfdataset(
    ["V_2003.nc", "V_2004.nc"],
    chunks={"time": 1},
    metadata_policy="preserve_sources",
) as ds:
    print(ds.sizes)  # sorted by actual time; fields remain Dask-backed
```

The same policy is available in a schema-v1 JSON configuration. Paths below resolve relative to that file:

```json
{
  "schema_version": 1,
  "input": ["V_2003.nc", "V_2004.nc"],
  "metadata_policy": "preserve_sources",
  "chunks": {"time": 1},
  "coordinates": {"latitude": "latitude", "longitude": "longitude"}
}
```

Save it as `inspect.json` and run `python -m climate_preprocessing inspect inspect.json`. For `prepare`, also specify a new `output` filename and any required `region` or `units` options.

Output reconstructs encoding to avoid double packing while preserving CF references. Unit/longitude transformations remove and audit stale range attributes rather than scanning fields to recompute them. JSON provenance records versions, operations, options and input basename/size/mtime. NetCDF/GRIB readers check source size/mtime at graph construction and field-block execution. Zarr additionally checks JSON metadata SHA256 and its file inventory, with source-state checks during block access. None of these operations hashes entire meteorological fields or establishes a cryptographic snapshot of their values. Existing source metadata are retained and may need review before publication.

See [source](src/climate_preprocessing/), [references and improvements](THIRD_PARTY.md), [contributing](CONTRIBUTING.md), and [changes](CHANGELOG.md). Modules explicitly label `References` and `Climate-Base improvements`.

## 0.2.1 historical validation record

On 2026-11-01, the Windows source checkout, an installed wheel tested outside the source tree, and a separate Linux HPC installation each passed **147 tests**. That version's dependencies and scope are recorded in the [0.2.1 validation record](https://github.com/Dengruihan-clim/Climate-Base/blob/14378910e5ef88eb025b83a3c04429f1c48ce5e5/Preprocessing/validation.json). That update provided a [CI template](../docs/preprocessing/preprocessing-ci.yml.example) only: GitHub Actions is not enabled and there is no remote CI execution result. To enable it, copy the template to the repository's `.github/workflows/preprocessing.yml` and commit it.

Real-data validation used the original 2003 and 2004 ERA5 monthly mean pressure-level V wind files. Explicit `preserve_sources` produced a lazy dataset with `time=24, level=37, latitude=721, longitude=1440`, executing zero field tasks during construction. Two times, two levels and a 16×32 spatial subset were then computed from that merged Dask graph. Values, coordinates and time identifiers matched previously validated real samples exactly. A small NetCDF roundtrip retained units, calendar and per-source provenance; original sizes and modification stamps were unchanged.

The focused run recorded **5 passes, 1 expected rejection and 0 failures**. The expected rejection confirms that the default `strict` policy still rejects differing descriptive attributes; it is not a failed merge with the new policy. The full 24-month field was not computed, and neither complete archive quality nor Dask speedup was assessed. Real data, account paths and server logs are not distributed with the public source.

## Preserved monthly/seasonal interfaces

The remainder documents the existing monthly contract. `MonthlyArchive.read()` and seasonal interfaces retain their original return types; field calculations now use xarray/Dask and compute at these compatibility return boundaries. `MonthlyArchive.read_lazy(..., chunks="auto")` returns a DataArray that callers must close after evaluation. It is a normalized monthly view, not a full CF Dataset round trip.

The legacy interfaces turn the authors' monthly-input, complete-record and seasonal-reduction methods into an independent Python package. Supply local NetCDF data and explicit configuration to produce complete seasonal series for downstream analysis.

It supports ERA5, MERRA-2, JRA-3Q and similar products **already converted to monthly CF NetCDF**. Coordinate/variable names must match the actual file metadata; no common naming scheme is assumed for all providers or JRA-3Q products. Original GRIB and automatic hybrid-model-level adaptation are outside the supported input contract.

## Install and try

Python ≥3.11 is required, with NumPy, xarray, Dask, netCDF4 and cftime. Tests additionally require pytest. On this project's Windows workstation use the user-selected Anaconda interpreter; `python` below denotes the user's selected interpreter:

```bash
python -m pip install -e ".[test]"
python examples/netcdf_prepare.py new_demo
python -m climate_preprocessing inspect new_demo/prepare.json
python -m pytest tests -q
```

Run these commands from `Preprocessing`. The generator creates wholly synthetic data in a new directory and demonstrates writing; choose a new filename for another write. The source distribution contains code, example configuration and tests, with no real data. Replace every `/path/to/...` placeholder. The legacy monthly/seasonal example is still generated by `examples/generate_synthetic.py`; its `config.json` is used by `check` and `seasonal`.

## Input contract and configuration

See [the configuration example](examples/config.example.json). `files` accepts a filename, glob or list of strings, relative to `data_root` when not absolute. Templates support `{data_root}`, `{year}`, `{month:02d}` and `{yyyymm}`. Both monthly files and multi-record annual files are supported.

- Every file needs a finite, nonempty, strictly increasing one-dimensional CF time coordinate with units. Decode using its original calendar; never infer dates from filenames. At most one record per calendar month is allowed within a file, and each requested month must occur exactly once across candidate files.
- Supply actual monthly fields of the intended product. A single monthly record does not prove that all daily/hourly samples were averaged; verify the product definition. Daily/hourly inputs are not automatically aggregated.
- Latitude, longitude and optional pressure must be one-dimensional. Coordinate units explicitly indicate northward/eastward degrees; common CF aliases are accepted, while `degrees_south`/`degrees_west` are rejected. Curvilinear grids, reduced longitude grids, hybrid model levels and extra non-singleton ensemble/expver dimensions require explicit preprocessing/selection first.
- Variable dimensions are transposed as configured; extra singleton dimensions may be squeezed. Defaults are ascending latitude, ascending `[0,360)` longitude and ascending hPa pressure. Descending latitude/pressure, `[-180,180)` longitude and Pa pressure are configurable. Duplicate coordinates/cyclic endpoints are errors, not silently removed. Regional grids are allowed by default; `require_global_longitude=true` checks cyclic longitude coverage without interpolation or filling.
- A field's normalized grid, pressure levels and output units must remain identical across months. Fields may retain separate native grids by default. `shared_grid=true` additionally enforces a common horizontal grid and a common pressure axis for all three-dimensional fields; two-dimensional surface fields on that grid are allowed. There is no automatic level intersection or regridding.
- `hash_inputs`, `shared_grid` and `require_global_longitude` must be actual booleans; the string `"false"` is rejected. Dates require Python/NumPy integers, excluding booleans and implicit float truncation.

## Python API

```python
from climate_preprocessing import Coordinates, FieldSpec, MonthlyArchive, iter_seasons
from climate_preprocessing import write_seasonal_stream

archive = MonthlyArchive(
    {"temperature": FieldSpec(
        files="temperature_{year}.nc",
        variable="temperature",
        coordinates=Coordinates(time="time", latitude="latitude",
                                longitude="longitude", pressure="pressure_level"),
        expected_units=("K", "degree_Celsius"),
        target_units="K",
    )},
    data_root="/path/to/data",
    calendar_policy="modern_gregorian",
    pressure_units="hPa",
)
month = archive.read("temperature", 2000, 1)
# month.values: (pressure, latitude, longitude); surface fields omit pressure.
metadata = archive.inspect("temperature", 2000, 2)

records = iter_seasons(archive, "temperature", 2000, 2020,
                       months=(12, 1, 2), weighting="days")
write_seasonal_stream("/path/to/output/seasonal.nc", records,
    pressure_units=archive.pressure_units,
    inventory=lambda: archive.inventory, audit=lambda: archive.audit)
```

`MonthlyField` exposes values, latitude, longitude, optional pressure, units, source/effective calendar and conversion metadata. `seasonal_mean(...)` returns an in-memory `SeasonalField`; use `iter_seasons(...)` and streaming export for large fields. The `seasonal` CLI writes seasons incrementally into one temporary NetCDF instead of retaining the full multi-year output. Each yielded season is a materialized array; execution memory also depends on Dask chunks, concurrency and reduction buffers.

After separately installing the statistics package, align your predictor to the same season years and pass the arrays directly to regression. Preprocessing does not alter the predictor's scale:

```python
from climate_preprocessing import seasonal_mean
from climate_statistics import linear_regression
seasons = seasonal_mean(archive, "temperature", 2000, 2020, months=(12, 1, 2))
# predictor is aligned to seasons.years; controls can use the same years.
fit = linear_regression(predictor, seasons.values, seasons.years, controls=None)
```

## Calendars and complete seasons

Season months must be a contiguous, nonrepeated sequence of 1–12 months, labelled by the **year of the final month**. DJF 2000 uses 1999-12, 2000-01 and 2000-02. The configured first year is the first season label; the first winter is not silently dropped, and missing previous December raises an error.

`weighting="days"` uses actual CF calendar-month durations, including Gregorian leap years, `noleap`, `all_leap` and `360_day`. `weighting="equal"` explicitly chooses equal-month means, for example an analysis convention for month-end states. The package never guesses that a field should be accumulated; seasonal totals from monthly accumulated quantities require a separately specified method.

Every requested month must exist. A seasonal cell is finite only if all its months are finite. Masks and original nonfinite values remain NaN: no zero filling, shortened seasons or renormalization over surviving months. Real zeros and signs are retained.

The default `calendar_policy="strict"` merges only CF synonyms, such as standard/gregorian and noleap/365_day. With explicit `modern_gregorian`, time is first decoded using the source calendar; standard/gregorian/proleptic_gregorian are then unified to proleptic_gregorian **only for selected years ≥1583**. Ancient mixed calendars and other calendar families remain incompatible. Use this policy for entirely modern analysis periods. Source calendar labels remain in the audit.

## Units and physical meaning

Without `target_units`, input units are retained; variable names never determine a conversion. `expected_units` explicitly restricts allowed source units. `units_override` is a user assertion correcting the source attribute; the original units, override and every transformation remain recorded.

Known compatible conversions include K/Celsius, Pa/hPa, kg/kg and g/kg, m/mm, Pa/s and hPa/s, and K/s and K/day, with common equivalent spellings. Geometric m/s vertical velocity cannot become Pa/s. Geopotential-to-height, energy-to-temperature tendency and accumulation-to-rate transformations need physical assumptions and are not inferred. When justified, declare `scale`, `offset` and `target_units` explicitly; for example, the user must choose the gravity for geopotential height:

```python
from climate_preprocessing import convert_units
height = convert_units(geopotential, "m2 s-2", "m", scale=1/9.80665)
```

Unknown units can be retained verbatim, but cross-dimension conversion needs an explicit declaration. Complex values are rejected. Finite inputs that overflow to Inf/NaN during conversion raise instead of becoming apparent source missing values. The legacy MonthlyArchive does not validate field-specific physical ranges, clip values, apply terrain/below-ground masks, restandardize or interpolate horizontally/vertically. Lazy unit overflow checks execute during compute/write.

## Output, checks and validation boundary

NetCDF `seasonal_value` uses `(season_year, [pressure], latitude, longitude)` and records calendar, units, season months, weighting and audit. Provenance contains paths relative to `data_root`, sizes, optional SHA256 and selected monthly records; external inputs receive anonymous directory labels. Inputs are unchanged. Existing output files, including symlinks, are never replaced. A completed sibling temporary file is atomically published by hard link, requiring filesystem support; failures remove the temporary file and do not publish partial output.

`check` validates requested time, dimensions, grid and units for every configured field/month; it **does not decode field-variable values**. With default `hash_inputs=true`, SHA256 still reads every source-file byte, so this is not an entirely content-free metadata operation. Disable hashing to avoid that I/O. `check` is not a full field-value/scientific quality assessment. Input changes after indexing are rejected.

Default automated tests create synthetic CF files and cover dates/calendars, cross-year complete seasons, coordinate/dimension ordering, strict units, missing/duplicate months, grid/pressure changes, nonfinite values, privacy-safe provenance, streaming export and overwrite protection. No real data or remote connection code are distributed, and existing HPC scientific workflows are not invoked. This module provides preprocessing, not downstream statistical inference.

### Bounded validation with your real NetCDF files

The standalone source tool can check existing local data in an environment with the declared dependencies. The output directory must not exist:

```bash
python tools/validate_real_data.py \
  --input /path/to/year_a.nc --input /path/to/year_b.nc \
  --output-dir /path/to/new_validation_report \
  --variable V --latitude latitude --longitude longitude --time time --pressure level \
  --monthly-product
```

Use `--monthly-product` only after confirming the product definition; omit it for hourly/daily data. The tool does not calculate monthly means. It accepts at most two inputs and reads one time, up to two pressure levels and a contiguous 16×32 spatial subset per source. Extra non-singleton dimensions need explicit `--select DIM=INDEX`. Field transformations use xarray/Dask with at most two Dask threads and one BLAS/OpenMP thread, without statistical reductions or full-input hashing.

A tiny slice may still decode an entire native compressed chunk. The default decoded chunk budget is 192 MiB, adjustable with `--allow-native-chunk-mib`. Originals are used for metadata, strict multi-file contracts and lazy graphs; core transforms, write/read checks and legacy computations use small derived real subsets. Derived shards have explicitly harmonized test attributes. Reports distinguish `original_source` from `derived_real_subset`; this tool alone does not demonstrate full original-file reconciliation or full-field quality. The separate 0.2.1 original-file reconciliation run is summarized above.

Each case is immediately written to `results.jsonl`, followed by `report.json`. Statuses are `PASS`, `FAIL`, `NOT_APPLICABLE` and `EXPECTED_REJECTION`; any failure gives a nonzero exit code. Independently verified strict metadata conflicts and output overwrite rejection are expected rejections. Resource limits and unconfirmed monthly products are reported explicitly. Inputs remain unchanged.
