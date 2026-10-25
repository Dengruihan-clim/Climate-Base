# Statistical: regression, trends, and inference

[中文](README.md)

A standalone Python module for annual series and seasonal fields: explicit year alignment, gridpoint regression with controls, trend estimation, circular residual-block bootstrap, lag-1 effective-sample-size diagnostics, and Benjamini–Hochberg FDR. Only NumPy and SciPy are required. Supply your own data, domain, and predictors.

**Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).**

## Installation and example

Requires Python 3.11+. From this directory:

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
python examples/generate_synthetic.py --output-dir ./demo_input
python examples/regression_csv.py --config ./demo_input/config.json
```

The generator creates synthetic CSV inputs and a runnable configuration. The regression runner writes JSON. Existing target directories and output files are refused. For your own data, use the [configuration template](examples/regression.json) and replace `<DATA_ROOT>` and `<OUTPUT_ROOT>`. CSV headers identify the columns. Years must be unique, strictly increasing integers; blank cells, `NA`, `NaN`, and `null` denote missing values. The FDR testing family comprises all configured response columns.

## Python API

```python
from climate_statistics import align_years, linear_regression, benjamini_hochberg

years, (x, y, z) = align_years(
    (predictor_years, predictor),
    (field_years, field),
    (control_years, control),
)
result = linear_regression(x, y, years, controls=z,
                           include_trend=True, inference="ols", min_samples=15)
fdr = benjamini_hochberg(result["p"], alpha=0.05)
beta = result["beta"]
q_values = fdr["q_values"]
```

`field` has shape `(time, *space)` or `(time,)`; `predictor` is `(time,)`. Controls may have shape `(time,)` or `(time, n_controls)`. First aggregate inputs to scientifically comparable annual or seasonal series. [Preprocessing](../Preprocessing/README_EN.md) can prepare complete seasonal fields.

| Function | Purpose and main outputs |
| --- | --- |
| `align_years(*series)` | Each input is `(years, values)`; return common years and aligned arrays without filling gaps |
| `linear_regression(x, field, years, controls=None, ...)` | OLS with intercept, optional calendar-year trend and controls; `beta`, `control_beta`, `trend_beta`, partial `r`, `p`, `p_nominal`, `n`, `n_eff`, `df` |
| `lag1_bartlett(x, y, years, ...)` | Actual-adjacent-year lag-1 correlations, approximate effective sample size, and t tests for two supplied series |
| `benjamini_hochberg(p_values, alpha=.05)` | Adjust all finite supplied p values; return `q_values`, `reject`, `n_tests` |
| `ols_trend(field, years, ...)` | Gridpoint OLS slopes and nominal standard errors/confidence intervals |
| `sen_trend(field, years, ...)` | Theil–Sen slopes, SciPy slope intervals, and nominal Kendall tests |
| `circular_block_bootstrap(field, years, ...)` | Paired circular resampling of trend residuals on fixed complete spatial support; slope distributions and percentile intervals |

```python
from climate_statistics import ols_trend, sen_trend, circular_block_bootstrap

ols = ols_trend(field, years, min_samples=15)
sen = sen_trend(field, years, min_samples=15, confidence=0.95)
boot = circular_block_bootstrap(
    field, years, method="sen", n_resamples=1000,
    block_length=3, seed=42, confidence=0.90,
)
```

## Scientific conventions

- Each regression uses the locally joint finite sample of predictor, response, and controls. Insufficient or rank-deficient samples retain counts and yield NaN estimates. Year intersection does not replace local missing-data selection.
- Coefficients preserve the **supplied predictor scale**. No standardization is repeated after subsetting. For an index standardized over a predefined full reference period, coefficients remain responses per one standard deviation of that reference period.
- Trends use actual year distances. `slope`, confidence limits, and bootstrap slopes are in input units per year; `slope_per_decade` is ten times the annual slope. Nominal OLS intervals, Sen intervals, and Kendall p values do not automatically account for serial dependence. Sen confidence levels are restricted to `[0.5, 1)` because SciPy interprets lower `alpha` values symmetrically; OLS and bootstrap confidence levels may be chosen within `(0, 1)`.
- `inference="ols"` provides nominal linear-model inference. Regression also accepts `"lag1_bartlett"`: remove intercept, controls, and optional trend, then use residual lag-1 correlations to approximate effective degrees of freedom. Only samples **one calendar year apart** form lag pairs; gaps are never collapsed. Unavailable diagnostics return NaN. This approximation does not implement HAC standard errors.
- `stderr_nominal` is the nominal OLS standard error. With the lag-1 option, `stderr` is approximately rescaled using effective degrees of freedom; unavailable diagnostics do not fall back to nominal values. Trend functions accept only `inference="ols"`; a deterministic year predictor is not passed through a stationary-series lag-1 correction.
- Circular block bootstrap requires a complete consecutive integer-year axis and refuses gaps or missing input values. Explicitly select fixed spatial support with complete records before calling it; all cells share the same residual resampling indices. Choose `method="sen"` or `"ols"`; report block length, seed, replicate count, and confidence level.
- FDR acts on the finite p values passed to that call and cannot repair incorrect pointwise inference. Standard BH control relies on independence or suitable positive dependence. Define the testing family before analysis.

## Validation and scope

Tests cover independent OLS/SciPy comparisons, partial-regression equivalence, missing years and local missing values, constants and rank deficiency, calendar-year lag pairing, FDR, and block resampling. Synthetic examples check the interface; no observed data, study results, or validation of a particular climate mechanism are included.

The module extracts reusable statistical kernels from existing research workflows and removes fixed study variables, paths, and scheduling. Model preparation is available under [LBM](../LBM/README.md). Cite [Climate-Base](https://github.com/Dengruihan-clim/Climate-Base) and the revision used, together with the datasets and methods actually employed. This guide does not create or modify third-party licensing conditions.

## Method references

- Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, **57**, 289–300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).
- Sen, P. K., 1968: Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, **63**, 1379–1389. [DOI](https://doi.org/10.1080/01621459.1968.10480934).
- SciPy documentation for [`theilslopes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html) and [`kendalltau`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html) describes implementation details and nominal intervals.
