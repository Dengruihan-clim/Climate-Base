# Climate preprocessing with xarray and Dask

[中文](README.md)

**Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).**

## 0.3.0: pressure, expected time indexes and user masks

Version 0.3.0 passed **283 tests each** on the local source, a wheel installed outside the source tree, and Linux HPC. The original merge regression recorded 5 passes and 1 expected rejection; the new pressure/time/mask validation recorded 7 passes and 4 expected rejections, with no failures or unrun cases. A January 2004 sample of two levels and 16×32 points matched prior reference values exactly; construction ran zero field tasks, masking and output roundtrips agreed, and original sizes/mtimes were unchanged. Expected rejections cover incomplete indexes, empty selections, misaligned masks, output overwrite and the original strict attribute policy. Full fields and speedup were not benchmarked. See [validation.json](validation.json).

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

Variable names and QC meanings must match the actual product. Keep independently opened mask files alive until execution ends; closing the result closes its shared source Dataset, but does not own an external mask file. Run `python examples/pressure_time_mask.py /path/to/new_demo` for a fully synthetic pressure/time/QC roundtrip with no downloads.

Pressure centers and bounds endpoints must currently be strictly positive; zero-pressure top interfaces need a later interface-coordinate workflow. Masked integer fields use floating point to represent NaN. Retained integers that cannot be represented exactly raise when the relevant Dask block executes, preventing silent value changes.

Module references and independently implemented improvements are listed in [THIRD_PARTY.md](THIRD_PARTY.md). Calendar conversion, GRIB/Zarr, regridding, hybrid levels, product adapters and the Meteostat provider remain planned.

## 0.2.1: implemented lazy preprocessing workflow

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

Run `python examples/netcdf_prepare.py /path/to/new_demo` for a fully synthetic example. Its schema-v1 JSON config can be passed to `python -m climate_preprocessing inspect CONFIG` and `python -m climate_preprocessing prepare CONFIG`. The example already writes `prepared.nc`; choose a new output filename before preparing again. Relative paths resolve against the configuration. Explicit input lists are required. Chunks use native input dimension names; unknown dimensions and `None` are rejected. Keep inputs unchanged and available until computation/output finishes, then close the Dataset.

Current limits: one-dimensional rectilinear latitude/longitude; longitude bounds must enclose centers in unwrapped intervals narrower than 180 degrees; explicit existing unit table rather than a full Pint vocabulary. Time supports order/calendar validation, explicit expected-index checks and exact inclusive selection, without frequency inference or calendar conversion. GRIB/Zarr, regridding, hybrid levels, product-specific adapters and Meteostat fetching remain future work.

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

Output reconstructs encoding to avoid double packing while preserving CF references. Unit/longitude transformations remove and audit stale range attributes rather than scanning fields to recompute them. JSON provenance records versions, operations, options and input basename/size/mtime. Generic readers check size/mtime at graph construction and for executed field blocks; this does not hash input content. Existing source metadata are retained and may need review before publication.

See [source](src/climate_preprocessing/), [references and improvements](THIRD_PARTY.md), [contributing](CONTRIBUTING.md), and [changes](CHANGELOG.md). Modules explicitly label `References` and `Climate-Base improvements`.

## 0.2.1 validation record

On 2026-11-01, the Windows source checkout, an installed wheel tested outside the source tree, and a separate Linux HPC installation each passed **147 tests**. Exact dependencies and scope are recorded in [validation.json](validation.json). This update provides a [CI template](../docs/preprocessing/preprocessing-ci.yml.example) only: GitHub Actions is not enabled and there is no remote CI execution result. To enable it, copy the template to the repository's `.github/workflows/preprocessing.yml` and commit it.

Real-data validation used the original 2003 and 2004 ERA5 monthly mean pressure-level V wind files. Explicit `preserve_sources` produced a lazy dataset with `time=24, level=37, latitude=721, longitude=1440`, executing zero field tasks during construction. Two times, two levels and a 16×32 spatial subset were then computed from that merged Dask graph. Values, coordinates and time identifiers matched previously validated real samples exactly. A small NetCDF roundtrip retained units, calendar and per-source provenance; original sizes and modification stamps were unchanged.

The focused run recorded **5 passes, 1 expected rejection and 0 failures**. The expected rejection confirms that the default `strict` policy still rejects differing descriptive attributes; it is not a failed merge with the new policy. The full 24-month field was not computed, and neither complete archive quality nor Dask speedup was assessed. Real data, account paths and server logs are not distributed with the public source.

## Preserved monthly/seasonal interfaces

The remainder documents the existing monthly contract. `MonthlyArchive.read()` and seasonal interfaces retain their original return types; field calculations now use xarray/Dask and compute at these compatibility return boundaries. `MonthlyArchive.read_lazy(..., chunks="auto")` returns a DataArray that callers must close after evaluation. It is a normalized monthly view, not a full CF Dataset round trip.

The legacy interfaces turn the authors' monthly-input, complete-record and seasonal-reduction methods into an independent Python package. Supply local NetCDF data and explicit configuration to produce complete seasonal series for downstream analysis.

It supports ERA5, MERRA-2, JRA-3Q and similar products **already converted to monthly CF NetCDF**. Coordinate/variable names must match the actual file metadata; no common naming scheme is assumed for all providers or JRA-3Q products. Original GRIB and automatic hybrid-model-level adaptation are outside the supported input contract.

## Install and try

Python ≥3.11 is required, with NumPy, xarray, Dask, netCDF4 and cftime. Tests additionally require pytest. On this project's Windows workstation the selected interpreter is `C:\ProgramData\anaconda3\python.exe`; `python` below denotes the user's selected interpreter:

```bash
python -m pip install -e ".[test]"
python examples/netcdf_prepare.py /path/to/new_demo
python -m climate_preprocessing inspect /path/to/new_demo/prepare.json
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
