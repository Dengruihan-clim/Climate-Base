# Build LBM T42L20 Basic States from Reanalysis Data

[简体中文](README.md)

**Repository tool organization and refactoring: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring)**  
Prepared: 2026-10-25

This tool constructs **three-dimensional basic states for the dry T42L20 Linear Baroclinic Model (LBM)** from global monthly pressure-level reanalysis data. Provide the reanalysis data and three model reference files, then edit a JSON configuration to run the workflow. It is independent of the original server account, directory layout, and job scheduler.

The numerical workflow was refactored from verified HPC basic-state builders, including an original DJF implementation and extensions for other seasons. Original entry points and source SHA256 fingerprints are documented in the [source provenance notes](docs/SOURCE_PROVENANCE_EN.md). Outputs contain complete U, V, T, PS, and Q fields. V retains its nonzero structure rather than being forced to zero. Q is retained for the virtual-temperature terrain correction and file compatibility; this does not configure a moist-model experiment.

The current target is fixed at a **128 × 64 Gaussian grid, scalar T42 spectral projection, and 20 sigma levels**. The input interface can read MERRA-2, ERA5, or other reanalysis NetCDF data that meet the requirements below. When changing products, verify the variables, units, pressure levels, and scientific suitability. The package supplies a configuration example for actual MERRA-2 data; it does not claim scientific validation with every reanalysis product.

## Model Attribution and Research Citation

Credit the original LBM development and linear-equation formulation described by **M. Watanabe and M. Kimoto (2000, 2001)**. The [official development page](https://ccsr.aori.u-tokyo.ac.jp/~lbm/sub/lbm_2.html) states that LBM is based on the dynamical core of **CCSR/NIES AGCM version 5.4g**, jointly developed by CCSR at the University of Tokyo and NIES, with sigma-coordinate primitive equations explicitly linearized about a basic state. Research using LBM should cite the 2000 methods paper together with its 2001 corrigendum; full references and DOIs are in the [shared LBM references and contribution notes](../REFERENCES.md).

The basic-state preparation workflow also refers to the community guide in **HanXiang0721/Note-Numerical-Methods**, credited by the GitHub handle **HanXiang0721**: [guide content](https://github.com/HanXiang0721/Note-Numerical-Methods/blob/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md) · [user-specified commit history](https://github.com/HanXiang0721/Note-Numerical-Methods/commits/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md). Credit this community workflow resource separately from the original LBM developers.

RH_Deng and Codex are credited for organizing and refactoring this repository's Python tools, tests, interfaces, and documentation. That credit does not establish authorship of the original LBM, the community guide, or the reanalysis products. This tool prepares dry-model basic states; retaining Q does not implement a moist model. Cite the datasets actually used and follow the original model's and other providers' conditions.

## 1. Environment

From the repository root, enter this tool's directory:

```bash
cd LBM/Basic
```

Use Python 3.11 or newer with `numpy`, `scipy`, and `netCDF4`. Plotting is optional and additionally requires `matplotlib`. Install the dependencies in your own Python environment:

```bash
python -m pip install -r requirements.txt
```

Run `python -m lbm_background` from the code directory. Alternatively, install the package in editable mode; select the `plots` extra if needed:

```bash
python -m pip install -e .
python -m pip install -e ".[plots]"
```

Replace `python` with the interpreter for your environment, following any local interpreter requirements. The program does not invoke the LBM executable, a Fortran compiler, NCL, or CDO, and does not submit scheduler jobs.

## 2. Required Inputs

### Reanalysis data

The JSON field keys and internal units are listed below. Field keys are fixed; configure each source NetCDF variable name with `variable`.

| Field key | Quantity | Supported input units | Internal units |
| --- | --- | --- | --- |
| `U`, `V` | Zonal and meridional wind | Common CF forms of m s⁻¹ | m s⁻¹ |
| `T` | Pressure-level air temperature | K or recognized CF forms of degrees Celsius | K |
| `Q` | Pressure-level specific humidity | kg/kg or g/kg | kg kg⁻¹ |
| `H` | Pressure-level height or geopotential | m, gpm, or m² s⁻² | m |
| `SLP` | Sea-level pressure | Pa or hPa | hPa |
| `PS_SOURCE` | Original reanalysis surface pressure; optional | Pa or hPa | hPa |

The original ERA5 `z` variable is geopotential. Set `H.variable` to `z`; the input reader recognizes the units and divides by **9.80665**. Fields already expressed as height are not divided again. Unit attributes must accurately describe the data; renaming a variable does not replace a unit conversion. `PS_SOURCE` is used only for comparison and does not replace the final PS consistent with LBM terrain. Remove its configuration entry if it is unavailable.

Inputs must meet these requirements:

- Three-dimensional fields use one-dimensional pressure, latitude, and longitude coordinates. SLP and PS_SOURCE use the same type of horizontal coordinates.
- Latitude and longitude units must be `degrees_north` and `degrees_east`, or recognized CF equivalents. Convert radians and west-positive or south-positive axes before use.
- The horizontal grid must be global and rectilinear, with latitude coverage extending across the target Gaussian latitudes and complete cyclic longitude coverage. Regular latitude–longitude and full Gaussian grids are supported. Curvilinear grids, two-dimensional coordinate arrays, and regional subsets are not accepted.
- Pressure coordinates use Pa or hPa. After conversion, all pressure-level fields must share the same pressure levels. Different fields may have different horizontal grids, but each field must pass the grid-consistency checks across months.
- Time must have decodable CF `time.units` and a calendar. The reader selects records using the **actual year and month in the time coordinate**, rather than inferring dates from filenames. Missing or duplicate target months cause an error.
- Inputs may contain one month per file or multiple monthly records in annual or multiyear files. A multirecord filename template need not contain a month. Each target month must correspond to one monthly-mean record; hourly records cannot be supplied as if they were monthly means.
- Missing values should be represented by NetCDF masks, `_FillValue`, `missing_value`, or nonfinite values. Missing-data handling is retained and reported; missing profiles are not filled with a global mean.

Direct reading of hybrid/model-level data or GRIB is not supported. Prepare those inputs as physical pressure-level NetCDF data first.

### Three model reference files

Provide these files from the **same LBM T42L20 environment** that you intend to use:

| Configuration key | Typical filename | Purpose |
| --- | --- | --- |
| `model.topography` | `grz.t42` | Model terrain for terrain-consistent surface-pressure correction |
| `model.sigma` | `GTAXLOC.CSIG20` | Exact centers of the 20 sigma levels |
| `model.template` | `erawin.t42l20` | Gtool3 background record, axis, and format template |

These are data and format references; the model executable is not required. Real model references and reanalysis data are not distributed with the code. Do not substitute files for another grid or vertical resolution. The template supplies encoding metadata; its climatological field values are not reused as the new reanalysis basic state.

## 3. Configuration and Execution

Copy the [MERRA-2 DJF example](examples/merra2_djf.json), then edit `data_root`, the three `model` paths, the output directory, and the input filename templates. Every `/path/to/...` path in the example is a placeholder.

```bash
python -m lbm_background --config examples/merra2_djf.json --check
python -m lbm_background --config examples/merra2_djf.json
```

`--check` **checks only the configuration and the three model reference files**. It does not read or validate the reanalysis fields or create outputs. A successful check does not prevent a subsequent build from stopping because of missing months, duplicate records, units, grids, missing profiles, or final quality checks. The build command reads the inputs and writes one seasonal basic state.

Main configuration options:

| Option | Meaning |
| --- | --- |
| `name` | Output filename prefix using safe ASCII filename characters; for example, `merra2_djf_1991_2020` |
| `source_product` | Actual data product identifier, written to provenance metadata |
| `data_root` | Data root available as `{data_root}` in filename templates |
| `output` | A new output directory that must not already exist; relative paths are resolved from the configuration file's directory |
| `period.start_year`, `end_year` | Inclusive range of season-year labels |
| `period.months` | Consecutive months in seasonal order, including a possible year boundary; for example, `[12,1,2]` |
| `period.minimum_valid_years` | Minimum valid-year count for each calendar month and source grid cell; the example uses 1, allowing partially sampled monthly climatologies |
| `coordinates` | Source coordinate variable names for `time`, `latitude`, `longitude`, and `pressure` |
| `fields` | The `files` template and source `variable` for each field |
| `integrity.hash_inputs` | Compute input SHA256 hashes; defaults to `true` |
| `integrity.require_receipts` | Require existing `.verified.json` receipts; defaults to `false` |
| `diagnostics.plot` | Generate additional diagnostic plots; defaults to `false` and requires `matplotlib` when enabled |

Filename templates support `{data_root}`, `{year}`, `{month}`, and `{yyyymm}`. Use `{month:02d}` for a two-digit integer month. For an annual file with multiple monthly records:

```json
"U": {
  "files": "{data_root}/U/U_{year}.nc",
  "variable": "u"
}
```

The supplied example follows the original MERRA-2 archive naming: `QV` in the Q directory, `H` in GPH, and `PS` in SP. If your archive uses other names, update the templates rather than changing the scientific variables to match the example.

### Season-year convention and averaging

Seasons crossing a year boundary are labeled by their **ending year**. DJF 1991–2020 therefore uses Decembers in 1990–2019 and Januaries and Februaries in 1991–2020, for 90 target months. MAM `[3,4,5]` uses the corresponding months within 1991–2020. Each configuration generates one seasonal basic state. Copy the configuration and choose a separate new output directory for each additional season.

For each calendar month, the program first computes the gridpoint climatology over valid years. It then averages the selected monthly climatologies with equal weights. Every selected month must be valid at a grid point for the seasonal mean to be valid. This is **not weighting by the number of days in each month, or directly averaging all valid daily or hourly samples**. `minimum_valid_years` controls the minimum sampling of each monthly climatology. Valid-year counts may differ among grid points; review the quality report before deciding whether to require the full year count.

### Input integrity

Ordinary CF NetCDF inputs do not need receipts from a particular downloader. The program computes input SHA256 hashes by default. If `filename.verified.json` exists beside an input, its byte count and hash are checked. Set `require_receipts: true` to require a compatible receipt for every input. A hash identifies the file and supports integrity checks; it does not establish the physical correctness of the data product.

## 4. Preserved Numerical Workflow

1. Read each target month, check its CF time, units, and coordinates, and compute calendar-month climatologies over valid years.
2. Use periodic bilinear interpolation to the exact 128 × 64 Gaussian grid. Longitude wraps cyclically, and valid corner weights are renormalized when other corners are missing. Target Gaussian latitudes run north to south, and longitudes start at 0°.
3. Average the selected months with equal weights. Estimate temporary PS from H–ln(p) profiles and model terrain, then obtain near-surface T/Q. Use the original `ecmsbs` virtual-temperature/lapse-rate correction to convert SLP to final PS consistent with LBM terrain. Temporary and final pressures receive scalar T42 projection as in the original workflow.
4. Interpolate to the exact model sigma levels with a natural cubic spline in `ln(p)`. Missing levels are excluded from each profile. Outside the valid pressure range, extend the nearest valid endpoint as a constant. The H–ln(p) pressure estimate instead extrapolates using the nearest two valid levels; these are distinct boundary rules.
5. Apply **scalar spherical-harmonic projection** through total degree 42 independently to every sigma level of U, V, T, and Q. This preserves the original scalar treatment rather than using a vector transform or zonal-mean U/V.
6. Set negative Q values after projection to zero and record the adjustment. Check finite values, physical ranges, spectral reprojection errors, endpoint extension, and adjacent-level potential-temperature differences. Temperature is not artificially adjusted for stability.
7. Write big-endian float32 Gtool3 basic-state data and GrADS data with one Fortran sequential record per horizontal plane. Field order is **U, V, T, PS, Q**, with PS in hPa. Read back Gtool3 and NetCDF separately and compare their float32 values exactly; also save provenance and quality reports.

Source-unit normalization and configurable file discovery are part of the reusable interface. Horizontal, vertical, and binary algorithms follow the verified builders. Fixed experiment dates, private paths, task state, and installation actions are not bound to the new entry point.

## 5. Outputs and LBM Integration

Default outputs are:

| File | Contents |
| --- | --- |
| `<name>.t42l20` | Native Gtool3 three-dimensional basic state read by LBM |
| `<name>.t42l20.grd`, `<name>.t42l20.ctl` | Matching GrADS data and descriptor |
| `background_sigma.nc` | Background on 20 sigma levels, with pressure, terrain, and diagnostic fields |
| `pressure_climatology_t42.nc` | Pressure-level climatology before sigma conversion |
| `quality_report.json` | Ranges, spectral checks, missing data, endpoint extension, and static-stability diagnostics |
| `input_manifest.json` | Inputs actually read, selected months, and integrity records |
| `output_manifest.json` | Output byte counts and SHA256 hashes |
| `build_status.json` | Build status or failure information |

Enabling plotting adds diagnostic figures. Model files use a conventional model-time label; the actual climatological period is recorded separately in metadata and configuration. Do not infer the climatology years from the model date `0001-01-01`.

In your LBM experiment, point `BSFILE` and `cbs0` to the same generated Gtool3 file, and point `cbs` to its matching `.grd`. Adapt the syntax to your model version:

```text
BSFILE = /path/to/output/merra2_djf_1991_2020.t42l20
cbs0   = /path/to/output/merra2_djf_1991_2020.t42l20
cbs    = /path/to/output/merra2_djf_1991_2020.t42l20.grd
```

These lines illustrate file relationships; they are not a complete namelist for every model version. The program creates files without changing existing experiments, installing the basic state in the model directory, or starting an integration. Subsequently verify model read-in, a short integration, stability, and the intended experiment.

## 6. Software Validation and Scientific Limits

Use the supplied tool to generate entirely synthetic reanalysis data, model templates, and a configuration for an initial environment and end-to-end software check:

```bash
python tools/make_demo.py --directory /path/to/new_demo
python -m lbm_background --config /path/to/new_demo/config.json --check
python -m lbm_background --config /path/to/new_demo/config.json
```

The demo directory must be new. **Synthetic demo data are for software testing only and must not be used as a scientific basic state or as input for a production model experiment.**

Validation included offline numerical and input tests, a complete build with synthetic data and synthetic templates, and a complete build with real model reference files and synthetic reanalysis. See the included [validation report](validation_report.json). No real reanalysis climatology was rebuilt, and no LBM integration was run. Earlier production-data checks, short integrations, and experiment results from the original programs are not acceptance results for this package.

Run the offline tests with:

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

All 42 tests passed. Using the same synthetic reanalysis and real model reference files, the refactored program and the original DJF builder produced exactly equal final float32 U, V, T, PS, and Q fields. This checks computational equivalence after refactoring; it does not establish scientific validation with real reanalysis data.

When using the results:

- Terrain correction changes PS. Final sigma-level pressure is `σ × PS`; original reanalysis PS is not a substitute for final PS.
- Insufficient source-pressure coverage leads to constant endpoint extension, potentially affecting near-surface and upper levels. Review the extension counts. Each profile needs enough valid levels, and excessive missing data stop the build.
- Scalar T42 projection preserves the original treatment and is not equivalent to a vector-wind transform that maintains vorticity/divergence relationships.
- Clipping Q to nonnegative values can reintroduce power above T42, so Q is not claimed to remain strictly band-limited. This adjustment does not validate moist physics.
- Weak negative adjacent-level potential-temperature differences are reported without adjusting temperature. Passing the quality thresholds does not establish stability during long model integrations.
- Products may differ in the meaning of similarly named variables, monthly-mean definitions, humidity treatment, and height definitions. Reassess the results when changing products, periods, missing-data thresholds, or model references.

## 7. Code and Sharing

```text
lbm_background/
  config.py      Configuration and seasonal settings
  inputs.py      Files, CF time, coordinates, units, and integrity
  pipeline.py    Climatology, terrain correction, construction, and diagnostics
  horizontal.py  Gaussian grid, periodic interpolation, and scalar T42
  vertical.py    H-to-PS, SLP-to-PS, and pressure-to-sigma conversion
  binary.py      Gtool3 and GrADS input/output
examples/merra2_djf.json
tools/make_demo.py
docs/SOURCE_PROVENANCE_EN.md
```

Examples and documentation use placeholder paths. The code package contains no real reanalysis data, model reference data, SSH connection settings, or historical task logs. Newly generated manifests record input filenames and relative locations. Review results and logs for private directory information before sharing them.

Conditions governing the original model, datasets, and reference documentation remain with their providers. The authorship above describes this organization and refactoring work and does not replace third-party attribution or licensing.
