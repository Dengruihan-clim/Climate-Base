# Diagnose Reanalysis Heating and Build LBM Thermal Forcing

[简体中文](README.md)

**Tool organization and refactoring: RH_Deng; Codex (OpenAI, AI-assisted refactoring)**
Prepared: 2026-10-25

This tool provides two independent entry points. `diagnose` derives pressure-resolved apparent heating Q1 from monthly reanalysis and computes either a seasonal climatology or an optional predictor regression. `convert` maps an existing pressure-level heating field to **LBM T42L20** and writes thermal-forcing inputs. Use `convert` directly when a suitable heating field is already available.

The package prepares inputs without running LBM or submitting Slurm jobs. It is self-contained and does not require installation of the adjacent basic-state package. Conversion still requires the actual model background and sigma axis intended for the experiment. See the [references and credits](../REFERENCES.md) for the original model and community documentation, and the [source notes](docs/SOURCE_PROVENANCE.md) for the extracted code. The LBM model itself is not original work by this package's authors.

## 1. Environment and Synthetic Example

Use Python 3.11+ with `numpy`, `scipy`, and `netCDF4`. From this directory, use your own interpreter:

```bash
cd LBM/Forcing
python -m pip install -r requirements.txt
python -m lbm_forcing --help
```

No model executable, Fortran compiler, NCL, CDO, or scheduler is required. Replace `python` with your environment's interpreter and follow any local interpreter requirements.

Generate an entirely synthetic example, then diagnose before converting:

```bash
python tools/make_demo.py --directory /path/to/new_demo
python -m lbm_forcing diagnose --config /path/to/new_demo/diagnose.json --check
python -m lbm_forcing diagnose --config /path/to/new_demo/diagnose.json
python -m lbm_forcing convert --config /path/to/new_demo/convert.json --check
python -m lbm_forcing convert --config /path/to/new_demo/convert.json
```

The demo directory and all build output directories must be new. **Synthetic data, synthetic model references, and their generated forcing are for software testing only, not scientific or production model experiments.**

## 2. Diagnose Heating from Monthly Data

Copy the [MERRA-2 diagnosis configuration](examples/diagnose_merra2.json) and replace the `/path/to/...` paths and source variable names:

```bash
python -m lbm_forcing diagnose --config examples/diagnose_merra2.json --check
python -m lbm_forcing diagnose --config examples/diagnose_merra2.json
```

### Inputs and configuration

| Option | Meaning |
| --- | --- |
| `name`, `source_product` | Safe filename prefix and actual data-product identifier |
| `data_root` | Data root used by the `{data_root}` filename placeholder |
| `output` | A new output directory; relative paths are resolved from the configuration file's directory |
| `period` | `start_year`, `end_year`, and consecutive `months` in seasonal order |
| `grid.step_degrees` | Global regular diagnostic-grid spacing; 2.5° in the example |
| `coordinates` | Source coordinate names for `time`, `latitude`, `longitude`, and `pressure` |
| `fields` | The `files` template and `variable` for each of T, U, V, OMEGA, and PS |
| `integrity` | Hash inputs by default; `require_receipts: false` makes downloader receipts optional, while existing receipts are still checked |
| `regression` | `null` selects seasonal Q1 climatology; an object enables a CSV predictor regression |

T must use K or recognized Celsius units; U/V use m s⁻¹. OMEGA is **dp/dt**, in Pa s⁻¹ or hPa s⁻¹, positive downward. Geometric vertical velocity in m s⁻¹ is not a substitute. PS and the pressure axis may use Pa or hPa. The diagnostic reader converts to K, m s⁻¹, Pa s⁻¹, and Pa. PS must be contemporaneous atmospheric surface pressure from the same reanalysis product, not a substitute land-surface product.

Inputs are global pressure-level CF NetCDF data with one-dimensional latitude and longitude axes, stored monthly or in files with multiple monthly records. Actual CF time determines the selected month, rather than the filename. Missing months, duplicate target records, incompatible coordinates, or unsupported units cause errors. Atmospheric fields must share the same physical pressure levels, with at least three levels for differentiation. Hybrid/model-level data, curvilinear grids, and direct GRIB reading are unsupported. Coordinate units must use recognized equivalents of degrees_north/degrees_east.

Times are first decoded with each file's original CF calendar. For selected records in 1583 or later, `standard`, `gregorian`, and `proleptic_gregorian` are treated as equivalent modern Gregorian calendars. Earlier records do not mix distinct calendars. Calendars such as `360_day`, `noleap`, and `all_leap` retain their own definitions; incompatible mixtures are rejected. The quality report preserves the source calendar labels.

Filename templates support `{data_root}`, `{year}`, `{month}`, and `{yyyymm}`; `{month:02d}` formats a two-digit month. The MERRA-2 example places OMEGA in `Omega` and PS in `SP`; adapt templates to your archive.

### Season years and neighboring months

Seasons are labeled by the **year of their final month**. DJF 1991 consists of December 1990 and January–February 1991. Each diagnosed month also needs previous/next-month T, masked by PS from that same neighboring month. Consequently each DJF typically needs **November–March T and PS**, while U/V/OMEGA are needed for the selected December–February months. `--check` lists required months.

Diagnose Q1 from each complete monthly state first. Require the selected months to be valid, average them with **equal monthly weights**, and then form a climatology across seasons or fit a regression. Do not substitute a calculation from already averaged multiyear climatologies for the monthly nonlinear terms. Averaging is not day-weighted. Time derivatives use the actual CF-calendar month-midpoint intervals in seconds, including leap years or the month lengths of the applicable calendar.

### Thermodynamics and masks

```text
Q1/Cp = dT/dt + u/(a cosφ) dT/dλ + v/a dT/dφ
        + omega [dT/dp - (Rd/Cp) T/p]
Cp = 1004 J kg-1 K-1; Rd = 287.05 J kg-1 K-1; a = 6371000 m
```

Q1 is in W kg⁻¹, positive for heating. Multiply by `86400/Cp` to obtain K day⁻¹. Pressure resolution is retained, with **no vertical integration or layer averaging**.

For every month, its own PS is first mapped to the native atmospheric locations and used to exclude underground values. Strict bilinear interpolation then maps to the diagnostic grid: any missing positive-weight corner makes the result missing, with no missing-corner weight renormalization. Neighboring T records use their own month's PS. Spatial and pressure derivatives use contiguous three-point stencils without jumping over missing neighbors. Longitude wraps cyclically, poles remain missing, and all heating terms share a common valid mask.

Q1 diagnosed from monthly means does not recover unresolved submonthly transient heat-transport covariances. Output `ps_mean` is the mean PS over the selected seasonal months; `ps_min` includes the neighboring-temperature months and supports underground exclusion and provenance. Neither **replaces model-background PS during conversion**.

## 3. Optional Predictor Regression

Replace `regression: null` with:

```json
{
  "file": "/path/to/predictor.csv",
  "year_column": "year",
  "predictor_column": "predictor",
  "control_column": "control",
  "predictor_units": "original supplied unit",
  "min_years": 10,
  "max_lag": 3
}
```

Provide one CSV row for every configured season year, with an integer year and the predictor in its original scale. Include the corresponding control column when used; set `control_column: null` to omit it. The CSV year convention must match the season labels. Supply complete, finite predictor/control records as required by the reader.

At each grid cell, use its own paired finite years to fit:

```text
heating = intercept + actual_year [+ control] + beta × predictor
```

The default threshold is 10 valid seasons. Predictor residuals are **not standardized again**: beta is the response per **one input predictor unit**. For a per-standard-deviation response, supply an explicitly standardized predictor and label its units accurately. Amplitude is not normalized automatically.

Statistics include nominal and effective sample sizes. Bartlett autocorrelations pair observations at actual year distances `1..max_lag`, defaulting to 1–3 years. Nominal/effective degrees of freedom subtract 3 without a control or 4 with one. True zero coefficients for constant responses and arbitrarily small coefficients are retained even when correlation or p values are undefined. P values have no multiple-comparison adjustment and do not filter or zero the forcing.

## 4. Convert to LBM Thermal Forcing

Copy the [pressure-level conversion configuration](examples/convert_pressure.json):

```bash
python -m lbm_forcing convert --config examples/convert_pressure.json --check
python -m lbm_forcing convert --config examples/convert_pressure.json
```

| Option | Meaning |
| --- | --- |
| `name`, `output` | Safe filename prefix and a new output directory |
| `input.file`, `variable` | File and explicit variable name for one static three-dimensional pressure-level heating field |
| `input.coordinates` | Pressure, latitude, and longitude coordinate names; pressure_hpa/lat/lon in the example |
| `input.units_override` | Optional explicit unit specification; use only when the true source units are known |
| `model.background` | Native T42L20 basic state intended for the actual experiment; supplies model PS |
| `model.sigma` | Exact `GTAXLOC.CSIG20` axis from the same model |
| `region` | south/north/west/east bounds and an inward transition width, in degrees |
| `amplitude_multiplier` | Explicit amplitude factor; defaults to 1, without additional peak normalization |

Supply a static `pressure × latitude × longitude` field. Pressure must be positive in Pa/hPa; ascending or descending axes are normalized. Horizontal coordinates must be one-dimensional and global. Accepted heating units are K day⁻¹, K s⁻¹, or W kg⁻¹, using Cp=1004 for conversion. Pressure-integrated metadata, column-integrated fluxes, and incompatible units are rejected: W m⁻² is not a pressure-resolved heating rate. Predictor scale is retained as separate metadata; a variable name does not determine physical units.

`heating_k_day` in `diagnosed_heating.nc` is the selected seasonal climatology or regression beta and can be converted directly. The variable is not guessed. If the source has an additional time dimension, explicitly select a static field first.

Use **the same basic state as the intended model integration**. Target pressure is `sigma × model_background_PS`. A basic-state builder's output may be used; an encoding template is not a newly prepared scientific background and must not silently replace it. Conversion needs neither model topography nor a real forcing-file template, and does not modify the background.

The conversion first performs missing-aware periodic bilinear mapping on the source pressure levels, allowing valid-corner weight renormalization. Each column then excludes missing levels and uses a natural cubic spline in `ln(p)`, with constant extension of the nearest valid endpoint outside the available range. This differs from the strict missing-corner policy during diagnosis.

Finally multiply on the sigma grid by the geographic taper and convert to K s⁻¹. The default region is 45°S–15°N, 30°E–120°E, with a 5° inward cosine transition. Boundaries and the exterior are exactly zero. Heating and cooling signs are preserved, without vertical integration, significance filtering, or default amplitude normalization. **No global spectral filter follows the taper**, preserving exact exterior zeros.

## 5. The Two Check Modes

| Command | Reads | Writes output |
| --- | --- | --- |
| `diagnose --check` | Configuration and required-month calculations; no original reanalysis fields | No |
| `convert --check` | Input heating field, model background, sigma axis, and region compatibility | No |

A successful check does not establish scientific validation with real data, model read-in, or integration stability. Builds do not overwrite existing output directories. Use reports to resolve input or numerical-check failures.

## 6. Outputs and Validation Limits

`diagnose` writes `seasonal/<year>.nc`, `diagnosed_heating.nc`, quality/input/output manifests, and status records. The pressure-level result stores the selected `heating_k_day`, Q1 in K/day and energy units, climatologies, and available term statistics. Regression configurations also save the corresponding betas and inference results. Use variable metadata to distinguish climatology from responses per input predictor unit.

`convert` writes `<name>.grd`, `<name>.ctl`, `forcing.nc`, quality reports, input/output manifests, and status records. Its big-endian float32 binary stores longitude fastest, latitude north to south, and sigma from the surface to model top. There are **61 Fortran sequential records**: 20 vorticity planes, 20 divergence planes, 20 temperature planes, and one surface-pressure plane. Only temperature records carry thermal forcing; the other records are zero. Temperature forcing is in **K s⁻¹**.

Run offline tests with:

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

Tests and synthetic examples check software, units, masks, interpolation, and file formats. Consult the distributed validation report for actual results. Real model references, historical production validation, or synthetic-field comparisons do not establish a new scientific validation with real reanalysis. This tool does not run the model. Before scientific use, assess the input physics, endpoint extension, chosen model PS, model read-in, and integration results.

## 7. Sharing and Privacy

Examples use placeholder paths. The code package contains no real data, model backgrounds, internal project names, accounts, SSH settings, or historical job logs. Generated reports may identify supplied data and locations; review them before sharing actual results. The author credits describe this source organization and refactoring, and do not replace attribution or terms of use for the original LBM and community documentation.
