# Monthly climate preprocessing

[中文](README.md)

**Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).**

`climate_preprocessing` turns the authors' existing monthly-input, complete-record and seasonal-reduction methods into an independent Python package. Supply local NetCDF data and explicit variable/coordinate configuration to read normalized monthly fields and produce complete seasonal series for later regression, trend analysis and inference.

It supports ERA5, MERRA-2, JRA-3Q and similar products **already converted to monthly CF NetCDF**. Coordinate/variable names must match the actual file metadata; no common naming scheme is assumed for all providers or JRA-3Q products. Original GRIB and automatic hybrid-model-level adaptation are outside the supported input contract.

## Install and try

Python ≥3.11 is required, with NumPy, netCDF4 and cftime. Tests additionally require pytest. Run these commands in your chosen environment:

```bash
python -m pip install -e ".[test]"
python examples/generate_synthetic.py /path/to/new_demo
python -m climate_preprocessing check /path/to/new_demo/config.json
python -m climate_preprocessing seasonal /path/to/new_demo/config.json
python -m pytest tests -q
```

The generator creates wholly synthetic data in a new directory. The distribution contains only source, example configuration and tests; no real data are included. Replace every `/path/to/...` placeholder.

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

`MonthlyField` exposes values, latitude, longitude, optional pressure, units, source/effective calendar and conversion metadata. `seasonal_mean(...)` returns an in-memory `SeasonalField`; use `iter_seasons(...)` and streaming export for large fields. The `seasonal` CLI writes seasons incrementally into one temporary NetCDF instead of retaining the full multi-year output. Memory still needs to accommodate one monthly field plus reduction buffers.

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

Unknown units can be retained verbatim, but cross-dimension conversion needs an explicit declaration. Complex values are rejected. Finite inputs that overflow to Inf/NaN during conversion raise instead of becoming apparent source missing values. The library does not validate field-specific physical ranges, clip values, apply terrain/below-ground masks, restandardize or interpolate horizontally/vertically.

## Output, checks and validation boundary

NetCDF `seasonal_value` uses `(season_year, [pressure], latitude, longitude)` and records calendar, units, season months, weighting and audit. Provenance contains paths relative to `data_root`, sizes, optional SHA256 and selected monthly records; external inputs receive anonymous directory labels. Inputs are unchanged. Existing output files, including symlinks, are never replaced. A completed sibling temporary file is atomically published by hard link, requiring filesystem support; failures remove the temporary file and do not publish partial output.

`check` validates requested time, dimensions, grid and units for every configured field/month; it **does not decode field-variable values**. With default `hash_inputs=true`, SHA256 still reads every source-file byte, so this is not an entirely content-free metadata operation. Disable hashing to avoid that I/O. `check` is not a full field-value/scientific quality assessment. Input changes after indexing are rejected.

Tests create synthetic CF files and cover dates/calendars, cross-year complete seasons, coordinate/dimension ordering, strict units, missing/duplicate months, grid/pressure changes, nonfinite values, privacy-safe provenance, streaming export and overwrite protection. No real data, remote connection code or model entry points are distributed or invoked. This module provides preprocessing, not downstream statistical inference.
