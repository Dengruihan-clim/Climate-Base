# Hadley-Ma2025: regional Hadley circulation and index responses

[中文](README.md) · [References](REFERENCES.bib) · [Input and command examples](examples/README.md)

A standalone Python workflow informed by the regional Hadley circulation diagnosis of Ma et al. (2025). It computes a local three-dimensional mass streamfunction, horizontal circulation intensity, regional latitude–pressure structure, and linear responses to a supplied climate index.

**Organization and implementation: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).** Original scientific methods and upstream software remain credited to their authors in [REFERENCES.bib](REFERENCES.bib).

Supply your own wind fields and index. Observational data, private results, and server connection settings are not distributed with this directory. Cite the datasets actually used.

## Installation

Use Python 3.10 or later and install the scientific and file I/O dependencies in a compatible environment:

```bash
python -m pip install -r requirements.txt
```

These dependencies are sufficient when the input is an independently validated divergent meridional wind `vd`. Total winds `u/v` additionally require a real global Helmholtz decomposition backend:

```bash
python -m pip install -r requirements-spectral.txt
```

The optional backend is `windspharm` or compatible `skyborn.windspharm`. `windspharm` uses a compiled spectral transform library; some platforms require a Fortran toolchain, and prebuilt wheels are not guaranteed for every operating system and Python version. An environment providing a compatible backend can be used instead; install the base dependencies into that same environment. Total meridional wind must not be substituted for divergent wind when a backend is unavailable.

Alternatively, run `python -m pip install skyborn`; the real-backend validation here used skyborn 0.4.5. See [VALIDATION.md](VALIDATION.md) for the validation record.

After installation, verify the actual backend with analytic synthetic winds:

```bash
python tools/validate_spectral_backend.py
```

The command fails explicitly if no backend is available or numerical errors exceed its thresholds. Skipped, substituted, or mocked transforms do not count as actual backend validation.

## Run

Compute the global Helmholtz decomposition from total winds:

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/winds.nc" --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley"
```

Alternatively, supply precomputed divergent wind:

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/divergent_winds.nc" --input-kind divergent --v-var vd --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley"
```

Use `--frequency seasonal` when each year already contains one target-season sample. Add `--no-plots` to save numerical outputs without figures. For all options:

```bash
python hadley_ma2025.py --help
```

Replace `<DATA_ROOT>` and `<OUTPUT_ROOT>` with your own directories. Input requirements are described in [examples/README.md](examples/README.md).

The selected wind variables and pressure levels are loaded into memory over the input's full time span. Large domains and long records require sufficient memory; prepare input files for the desired years when necessary.

Frequently used options:

| Option | Purpose / default |
| --- | --- |
| `--winds` | One or more NetCDF wind files, combined by coordinates |
| `--index-column` | Numerical index column in the CSV |
| `--input-kind total\|divergent` | Defaults to `total`; `divergent` reads divergent meridional wind |
| `--u-var`, `--v-var` | Default `u`, `v`; set `--v-var vd` for a variable named `vd` |
| `--frequency monthly\|seasonal` | Default `monthly`; monthly aggregation is day-weighted and requires all three months |
| `--season` | Default `SON` |
| `--start-year`, `--end-year` | Analysis-year limits |
| `--baseline-start`, `--baseline-end` | Default `1981`, `2010` |
| `--pressure-levels` | Requested pressure levels in hPa |
| `--pressure-units` | Explicit input pressure unit override; otherwise read from coordinate attributes |
| `--wind-units` | Explicit wind unit declaration, e.g. `"m s-1"`; otherwise read from variable attributes |
| `--regional-reduction mean\|sum` | Default `mean` |
| `--main-lon-min`, `--main-lon-max` | Default `60`, `120` |
| `--sensitivity-lon-min`, `--sensitivity-lon-max` | Default `70`, `110` |
| `--no-detrend-fields`, `--no-detrend-index` | Disable field or index linear detrending |
| `--alpha` | Significance threshold, default `0.1` |
| `--truncation` | Optional triangular spherical-harmonic truncation |
| `--edge-target-level` | Edge diagnostic pressure, default `500` hPa |
| `--edge-core-lat-min`, `--edge-core-lat-max` | Core search interval, default `0`, `25`°N |
| `--edge-search-lat-min`, `--edge-search-lat-max` | Edge search window north of the core, default `20`, `40`°N |
| `--no-smooth-edge` | Disable the default three-point latitude smoothing before edge diagnosis |
| `--no-plots` | Save numerical outputs without figures |
| `--overwrite` | Explicitly allow replacement of existing outputs |

## Synthetic example

Generate analytic winds and an index, then run the workflow using the known divergent wind:

```bash
python examples/generate_synthetic.py --output-dir demo_input
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind divergent --v-var vd --index demo_input/index.csv --index-column nino34 --output-dir demo_output/divergent
```

The generator requires a directory that does not yet exist. `winds.nc` contains total `u/v` and the analytic divergent meridional component `vd`; `index.csv` contains annual index values. With a real spectral backend installed, run the total-wind path on the same fields:

```bash
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind total --index demo_input/index.csv --index-column nino34 --output-dir demo_output/total
```

## Workflow

1. Read winds and the chosen index, aggregate the target season, and align common years.
2. Diagnose divergent meridional wind from global total winds, or use supplied `vd`.
3. Integrate divergent meridional wind in pressure to obtain the local mass streamfunction.
4. Compute each year's horizontal intensity from the original streamfunction, then calculate intensity anomalies and their index regression.
5. Reduce the original streamfunction over each longitude sector to obtain its latitude–pressure climatology, anomalies, and regression.
6. Diagnose a circulation edge from zero crossings at the configured level and latitude search window, recording valid sample counts.

Defaults are SON; 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, and 1000 hPa; a main sector of 60°–120°E; a sensitivity sector of 70°–110°E; longitude mean reduction; and a 1981–2010 climatological baseline. Regression uses the common sample period. Fields and the index are detrended according to the settings, and the index is standardized again using the common sample's standard deviation (`ddof=1`). Coefficients therefore represent an increase of one sample standard deviation in the index.

## Definitions and interpretation

The ideal upper boundary in Eq. (3) of Ma et al. (2025) is pressure zero. On a uniform global longitude grid, this implementation uses

\[
\psi(t,p,\phi,\lambda)
=\frac{R\cos\phi\,\Delta\lambda}{g}
\int_{p_{\mathrm{top}}}^{p}v_d(t,p',\phi,\lambda)\,\mathrm{d}p',
\qquad \Delta\lambda=\frac{2\pi}{n_x}.
\]

Pressure is integrated in Pa with adjacent trapezoids; output units are kg s⁻¹. The first supplied pressure is the integration reference, defaulting to 100 hPa, where the streamfunction is set to zero. **This is a finite-layer integral relative to the supplied top, without recovering unknown transport above it.** No bottom residual correction is applied.

- **Horizontal structure:** `HC_intensity = max_level(abs(psi))` is computed at each horizontal grid point for each year before anomaly calculation and regression. A positive coefficient denotes strengthening as the index increases; a negative coefficient denotes weakening. Because absolute value and maximum are nonlinear, this differs from taking the maximum absolute value of a streamfunction regression.
- **Vertical structure:** longitude reduction leaves latitude and pressure dimensions. Shading represents the index regression; background contours show the climatology. The shifted zero contour shows the climatology plus the response to a one-standard-deviation index increase.
- **Regional `mean` and `sum`:** local `psi` already includes one longitude-cell width. `sum` adds cell transports across the sector and is closer to the zonally integrated vertical-structure procedure used in Figure 2 of the paper. `mean` is a retained option averaging single-cell transports, rather than a claim about the paper's default. Both retain kg s⁻¹, but have different scales and meanings. The mean also changes with longitude resolution. Amplitude comparisons with the paper additionally require matching sectors, datasets, periods, and pressure layers. Neither operation automatically produces a closed two-dimensional regional overturning cell or the conventional full-circumference zonal-mean streamfunction.
- **Circulation edge:** the diagnostic finds the first zero crossing north of the circulation core within a specified latitude window at a specified pressure. Missing crossings remain missing. A valid count of zero is possible and does not demonstrate an absent edge response. Edge statistics use only finite edge–index pairs, apply the configured detrending to that subset, and standardize the index again there. Their sample and predictor scale may therefore differ from grid-point regressions over the full common period.
- **Significance:** ordinary two-sided correlation/regression t tests use `n-2` degrees of freedom, with default `p < 0.1`. No temporal autocorrelation or spatial multiple-testing correction is applied. Dots mark pointwise significance, rather than significance of a region as a whole.

## Relation to the paper

Ma, W., Wang, Y., Miao, Y., and Li, X. (2025). *Effect of Regional Hadley Circulation–Subtropical Jet Interaction: The Amplification of Central-Pacific ENSO-Induced Teleconnection Pattern*. *Geophysical Research Letters*, 52(9), e2025GL115344. DOI: [10.1029/2025GL115344](https://doi.org/10.1029/2025GL115344).

This tool adopts the regional circulation diagnosis and adds a generic index-response workflow. Its default season, period, pressure reference, sectors, and index settings are configuration choices, rather than a reproduction of every paper setting. It does not reproduce the complete central/eastern-Pacific ENSO classification, subtropical-jet interaction, or PNA mechanism analysis.

## Outputs and reuse

Main output files:

| File | Contents |
| --- | --- |
| `mass_streamfunction.nc` | Local mass streamfunction and associated metadata |
| `horizontal_intensity.nc` | Horizontal intensity, climatology, anomalies, and index regression |
| `regional_sections.nc` | Latitude–pressure structures and regressions for the two sectors |
| `predictor.csv` | Common years and the predictor actually used in regression |
| `summary.csv` | Run settings and regional summary |
| `edge_timeseries.csv` | Annual regional circulation-core and edge latitudes |
| `edge_statistics.csv` | Edge–index statistics and valid sample counts |
| `horizontal_response.png` | Horizontal intensity response, when plotting is enabled |
| `vertical_response.png` | Regional vertical structure response, when plotting is enabled |

Record the code revision, input datasets, common sample period, integration top, regional reduction, index definition, and detrending choices when reusing the workflow. Consult the saved metadata and current `--help` for exact fields and options.

Synthetic examples check the data interface and numerical workflow. They do not replace quality assessment of real inputs or verify all scientific conclusions of the reference paper.
