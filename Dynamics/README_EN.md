# Dynamics: atmospheric circulation diagnostics

[中文](README.md) · [Original methods and software references](REFERENCES.md)

A standalone Python module for Takaya–Nakamura stationary horizontal wave activity flux, full and linear Rossby wave sources, spherical wind/Helmholtz diagnostics, and meridional mass streamfunction. Supply your own physical fields. Observed data, study results, and server connection code are not bundled.

**Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).** Original method and software authors are credited separately in [REFERENCES.md](REFERENCES.md), kernel docstrings, and the [machine-readable citations](climate_dynamics/citations.py).

## Installation

Python 3.11+ is required. Flux, source assembly, and mass-streamfunction calculations require only NumPy:

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
```

**Spherical wind diagnostics additionally require windspharm and its compiled pyspharm/SPHEREPACK backend.** With a compatible build environment, use `python -m pip install ".[spectral]"`. This can require a Fortran toolchain; prebuilt wheels are not guaranteed for all Windows/Python combinations. A platform-supported conda-forge `windspharm` environment is another option; install this package into that environment. Core NumPy methods remain usable without the optional backend.

Verify the actual backend after installation:

```bash
python tools/validate_winds_backend.py
```

The runner uses analytic synthetic fields and fails explicitly when dependencies are unavailable. Skipped tests or mocked transforms do not count as numerical backend validation.

## Synthetic examples and configuration

```bash
python examples/generate_synthetic.py --output-dir ./demo_input
python -m climate_dynamics --config ./demo_input/tn_waf.json
python -m climate_dynamics --config ./demo_input/rossby_wave_source.json
python -m climate_dynamics --config ./demo_input/linear_rws.json
python -m climate_dynamics --config ./demo_input/mass_streamfunction.json
# Requires the real spectral backend:
python -m climate_dynamics --config ./demo_input/wind.json
```

The generator requires a new directory and creates numeric NPZ inputs plus five configurations. For your data, adapt the [template](examples/config.example.json) and replace `<DATA_ROOT>`/`<OUTPUT_ROOT>`. The CLI uses `allow_pickle=False`, accepts real numeric arrays, and requires explicit physical-unit declarations. **It performs no implicit unit conversion.** Inputs must already have the intended time selection, pressure levels, grids, and physical definitions. Original GRIB reading, seasonal aggregation, and regression are separate steps; see [Preprocessing](../Preprocessing/README_EN.md) and [Statistical](../Statistical/README_EN.md).

| `method` | Required NPZ arrays | Outputs and units |
| --- | --- | --- |
| `tn_waf` | `lat, lon, psi, u_background, v_background` | `waf_x, waf_y`, m² s⁻²; one configured pressure level |
| `rossby_wave_source` | `lat, lon` and the six components below | `stretch, advection, rws`, s⁻² |
| `linear_rws` | `lat, lon`, six `background_*` and six `perturbation_*` arrays | Four first-order terms and `rws`, preserving the supplied perturbation/coefficient scale |
| `wind` | `lat, lon, u, v` | Streamfunction, velocity potential, vorticity, divergence, component winds, and absolute-vorticity gradients |
| `mass_streamfunction` | `lat, lon, pressure, v`; CLI `v=(sample,pressure,lat,lon)` | Fixed-support longitude mean, support diagnostics, relative mass streamfunction in kg s⁻¹ |

The six RWS components have final latitude/longitude axes: `eta` is absolute vorticity (s⁻¹), `div` divergence (s⁻¹), `udiv/vdiv` divergent winds (m s⁻¹), and `eta_x/eta_y` physical gradients (s⁻¹ m⁻¹). Coordinates use degrees. Pressure integration uses **Pa**; the TN option `pressure_hpa` uses **hPa**. Use the exact unit strings shown in the examples, after physically converting your inputs rather than relabelling them.

For `linear_rws`, `units` declares physical base units. Optional `predictor_unit` specifies the denominator unit of all perturbation coefficients (default `"1"`, a dimensionless amplitude). For example, `"K"` means each perturbation component is a coefficient per kelvin of the predictor, and output metadata records s⁻²/K. Values are not restandardized; background components retain physical base units.

Output `_metadata_json` is a text scalar read with `json.loads(str(archive["_metadata_json"]))`. It records references, settings, units, input SHA256, and method limitations without storing private input absolute paths. Results are completed in the destination directory and published atomically using a hard link, which the filesystem must support. Existing files/symlinks are refused; failed computations do not publish partial outputs.

## Python API and connections

```python
from climate_dynamics import diagnose_wind, tn_waf, rossby_wave_source, linear_rws

# Complete global arrays (..., lat, lon); outputs always use ascending latitude.
background = diagnose_wind(u_mean, v_mean, lat, lon, truncation=42)
# A separately diagnosed perturbation must exclude fixed planetary terms:
perturbation = diagnose_wind(u_pert, v_pert, lat, lon, truncation=42, omega=0.)
source_mean = rossby_wave_source(background)
source_linear = linear_rws(background, perturbation)
flux = tn_waf(perturbation["psi"], background["u_total"], background["v_total"],
              background["lat"], background["lon"], pressure_hpa=200.)
```

Here the background is also spectrally truncated. `u_total/v_total` sum retained spectral components and may differ from the input winds. Use matching grids, coordinates, radius, and truncation for both diagnoses. Do not combine ascending diagnostic fields with unreordered descending input winds. TN also accepts explicitly prepared background total winds consistent with your scientific definition.

```python
from climate_dynamics import fixed_longitude_mean, mass_streamfunction

means, fraction, mask, averaging = fixed_longitude_mean(
    {"v": v, "omega": omega}, longitude=lon,
    longitude_bounds=(170., -170.), cyclic=True, min_fraction=0.9,
)
psi_mass, integration = mass_streamfunction(
    means["v"], pressure_pa, lat, reference_pressure_pa=pressure_pa[0],
)
```

The averaging API supports multiple leading sample dimensions through `sample_axes`; the default is axis 0, with longitude last. Mass integration defaults to `(..., pressure, latitude)` and permits explicit `pressure_axis/latitude_axis`. Align arrays and coordinates first. `longitude_cell_weights` also accepts actual `cell_bounds` to avoid inferring edges from centres.

## Scientific conventions

- **TN:** the stationary horizontal form of [Takaya and Nakamura (2001)](REFERENCES.md). Derivatives use radians, with a complete uniform global `[0,360)` longitude grid and ascending, possibly nonuniform latitude. Defaults exclude poles, `|lat|<=10°`, total wind speed below 5 m/s, and nonwesterly flow. Contiguous derivative stencils do not bridge missing cells. Supply a rotational streamfunction perturbation; geopotential is not divided by Coriolis internally. Flux is **quadratic** in amplitude. Explicitly loop over pressure levels rather than applying one pressure setting to different levels.
- **RWS:** [Sardeshmukh and Hoskins (1988)](REFERENCES.md), `S=-eta*div-Vchi·grad(eta)`. The full source is nonlinear; the first variation excludes perturbation-by-perturbation products. Background `eta` includes planetary vorticity, while its perturbation and perturbation gradients exclude fixed planetary terms. Component regression coefficients must share the same predictor scale. Terms and totals use one common finite mask, without significance filtering.
- **Spherical winds:** the [Dawson (2016) and SPHEREPACK software methods](REFERENCES.md). Complete finite global winds are required; missing or underground winds are not filled. Regular latitude includes the true poles; Gaussian latitude matches complete Legendre nodes. Longitude is uniform and increasing, without a cyclic duplicate. All diagnostics share one truncation, preserve leading axes and longitude, and return ascending latitude. The backend's degree-zero potential gauge and eastward/northward signs are retained.
- **Mass streamfunction:** [Oort and Yienger (1996) and Adam et al. (2018)](REFERENCES.md) provide the circulation context and continuity-based definition. This module calculates `2*pi*a*cos(phi)/g * integral[p_ref,p](v_mean dp)`, setting zero at the first supplied pressure only where its wind is finite. Pressure increases downward. Adjacent trapezoids propagate a missing layer to every deeper level. Unknown upper-layer flow is not assumed zero or extrapolated, and underground layers are not added.
- **Longitude support:** cell-intersection widths define fixed common support across specified fields and samples. Insufficient coverage produces missing rows; genuine zeros remain zero. A finite-band or incomplete-support mean yields a **full-circumference-equivalent relative quantity**, rather than actual regional total mass transport or an automatically closed two-dimensional overturning circulation.

Defaults are radius 6,371,000 m, rotation rate 7.292115×10⁻⁵ s⁻¹, and gravity 9.80665 m s⁻², all explicitly configurable. Original-kernel comparisons, analytic tests, and actual spectral-backend checks are recorded in [validation.json](validation.json). Synthetic verification does not replace scientific quality assessment of a particular dataset.

## Attribution and citation

Cite the original methods, upstream software, and reanalysis datasets actually used, together with the repository revision. [REFERENCES.md](REFERENCES.md) contains complete entries and explains implementation boundaries. Scientific authors, software developers, and repository organizers are credited separately; third-party licensing conditions are not replaced.
