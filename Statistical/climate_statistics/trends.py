"""Nominal OLS/Sen trends and paired circular residual-block uncertainty.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
All slopes use input units per actual year, with an additional per-decade
view. Neither a nominal trend p value nor a bootstrap interval proves
scientific causation or spatial field significance.
"""
from numbers import Real

import numpy as np
from scipy import stats

from .inference import _field, _integer, _years
from .regression import linear_regression


def _confidence(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or not 0 < value < 1):
        raise ValueError('confidence must be finite and strictly between zero and one')
    return float(value)


def ols_trend(field, years, *, min_samples=None, inference='ols',
              min_lag_pairs=3, confidence=.95):
    """OLS trend against actual years, using finite observations per cell.

    Its t p values and confidence limits are nominal independent-error OLS.
    Lag-one stationarity formulas are not applied to the deterministic year
    predictor. Use the residual-block bootstrap below for a separate,
    explicitly chosen time-dependence sensitivity calculation.
    """
    if inference != 'ols':
        raise ValueError("ols_trend only supports nominal inference='ols'; use a block bootstrap for trend uncertainty")
    level = _confidence(confidence)
    result = linear_regression(years, field, years, include_trend=False,
                               min_samples=min_samples, inference='ols',
                               min_lag_pairs=min_lag_pairs)
    slope = result['beta']
    critical = stats.t.isf((1-level)/2, result['df_nominal'])
    half_width = critical * result['stderr_nominal']
    return result | {'slope': slope, 'slope_per_decade': 10*slope,
                     'slope_ci_low': slope-half_width,
                     'slope_ci_high': slope+half_width,
                     'confidence': level, 'method': 'ols_trend',
                     'interval_method': 'nominal OLS Student t'}


def sen_trend(field, years, *, min_samples=None, confidence=.95):
    """Theil-Sen slopes on finite actual-year pairs, with nominal intervals.

    Each cell uses its own finite observations and actual year differences.
    Intercept is median(y)-slope*median(year), matching SciPy's separate
    intercept convention. The p value is the nominal two-sided Kendall test;
    no autocorrelation or multiplicity correction is implied. Constant data
    have slope=0 and zero-width slope bounds but undefined tau/p.
    Confidence must be in [0.5, 1), because SciPy treats smaller alpha values
    as the complementary confidence level rather than that requested level.
    """
    values = _field(field)
    year = _years(years, values.shape[0])
    minimum = 3 if min_samples is None else _integer(min_samples, 'min_samples', 3)
    if len(year) < minimum:
        raise ValueError(f'At least {minimum} time samples are required')
    level = _confidence(confidence)
    if level < .5:
        raise ValueError('Sen confidence must be in [0.5, 1); smaller SciPy alpha values represent the complementary level')
    shape = values.shape[1:]
    flat = values.reshape(len(year), -1)
    counts = np.isfinite(flat).sum(axis=0)
    arrays = {key: np.full(flat.shape[1], np.nan) for key in
              ('slope', 'intercept', 'slope_ci_low', 'slope_ci_high', 'tau', 'p_nominal')}
    for column in np.flatnonzero(counts >= minimum):
        valid = np.isfinite(flat[:, column])
        y, x = flat[valid, column], year[valid]
        fit = stats.theilslopes(y, x, alpha=level, method='separate')
        if not np.isfinite((fit.slope, fit.intercept, fit.low_slope, fit.high_slope)).all():
            raise ValueError('Trend range is not representable as finite float64')
        arrays['slope'][column], arrays['intercept'][column] = fit.slope, fit.intercept
        arrays['slope_ci_low'][column], arrays['slope_ci_high'][column] = fit.low_slope, fit.high_slope
        if np.ptp(y) > 0:
            test = stats.kendalltau(x, y, alternative='two-sided')
            arrays['tau'][column], arrays['p_nominal'][column] = test.statistic, test.pvalue
    result = {key: value.reshape(shape) for key, value in arrays.items()}
    return result | {'slope_per_decade': 10*result['slope'], 'n': counts.reshape(shape),
                     'confidence': level, 'method': 'theil_sen',
                     'interval_method': 'nominal Theil-Sen rank interval',
                     'p_method': 'nominal two-sided Kendall tau'}


def circular_block_bootstrap(field, years, *, n_resamples=1000, block_length=3,
                             seed=0, method='sen', confidence=.90, batch_size=64):
    """Paired circular residual blocks for complete consecutive annual data.

    Fit the chosen Sen/OLS line, center its residuals, draw circular residual
    blocks, and add that same line back at the original year coordinates.
    All cells share the same index draws, preserving spatial pairing.
    A final partial block is truncated. Missing data, calendar-year gaps,
    and nonconsecutive labels are rejected rather than interpolated or
    silently treated as adjacent. This is a conditional percentile interval
    whose usefulness depends on the block length and residual assumptions.

    ``bootstrap_slopes`` has shape (resample, *space); all slopes are per
    year. Sampling indices are returned for an auditable, reproducible draw.
    """
    values = _field(field)
    year = _years(years, values.shape[0])
    if len(year) < 3 or not np.all(np.diff(year) == 1):
        raise ValueError('Block bootstrap requires at least three consecutive integer years; gaps are unsupported')
    if not np.isfinite(values).all():
        raise ValueError('Block bootstrap requires complete finite data; no missing-year imputation is performed')
    replicates = _integer(n_resamples, 'n_resamples', 2)
    block = _integer(block_length, 'block_length', 1)
    batch = _integer(batch_size, 'batch_size', 1)
    if block > len(year):
        raise ValueError('block_length must not exceed the time-series length')
    if seed is not None:
        seed = _integer(seed, 'seed', 0)
    if method not in ('sen', 'ols'):
        raise ValueError("method must be 'sen' or 'ols'")
    level = _confidence(confidence)
    shape = values.shape[1:]
    flat = values.reshape(len(year), -1)
    fit = sen_trend(values, year) if method == 'sen' else ols_trend(values, year)
    slope = np.asarray(fit['slope']).reshape(-1)
    if method == 'sen':
        line = np.median(flat, axis=0) + (year-np.median(year))[:, None] * slope
    else:
        line = flat.mean(axis=0) + (year-year.mean())[:, None] * slope
    residual = flat-line
    residual -= residual.mean(axis=0)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(year)/block))
    starts = rng.integers(0, len(year), size=(replicates, n_blocks))
    indices = ((starts[..., None] + np.arange(block)) % len(year)).reshape(replicates, -1)[:, :len(year)]
    output = np.empty((replicates, flat.shape[1]))
    if method == 'sen':
        i, j = np.triu_indices(len(year), 1)
        elapsed = (year[j]-year[i])[None, :, None]
        batch = min(batch, max(1, 2_000_000 // len(i)))
    else:
        centered_year = year-year.mean()
        year_ss = np.sum(centered_year * centered_year)
    for begin in range(0, replicates, batch):
        end = min(begin+batch, replicates)
        # Bound temporary pair-slope arrays independently of spatial size.
        space_chunk = max(1, 2_000_000 // ((end-begin) * (len(i) if method == 'sen' else len(year))))
        for left in range(0, flat.shape[1], space_chunk):
            right = min(left+space_chunk, flat.shape[1])
            samples = line[None, :, left:right] + residual[indices[begin:end], left:right]
            if method == 'sen':
                output[begin:end, left:right] = np.median((samples[:, j]-samples[:, i])/elapsed, axis=1)
            else:
                samples = samples-samples[:, :1]
                output[begin:end, left:right] = np.einsum('t,btp->bp', centered_year, samples)/year_ss
    samples = output.reshape((replicates,) + shape)
    tail = (1-level)/2
    interval = np.quantile(samples, (tail, 1-tail), axis=0)
    return {'slope': fit['slope'], 'slope_per_decade': fit['slope_per_decade'],
            'bootstrap_slopes': samples, 'ci_low': interval[0], 'ci_high': interval[1],
            'confidence': level, 'block_length': block, 'seed': seed,
            'n_resamples': replicates, 'resample_indices': indices,
            'n': len(year), 'method': f'{method}_circular_residual_block_bootstrap'}
