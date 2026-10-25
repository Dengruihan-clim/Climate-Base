"""Explicit lag-one effective-sample-size and multiple-test diagnostics.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
The lag-one calculation is a residual-correlation approximation, not HAC,
GLS, an estimated AR model, or a guarantee of spatial field significance.
"""
from numbers import Integral, Real

import numpy as np
from scipy import stats


def _real_array(value, name):
    raw = np.ma.asarray(value)
    if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f'{name} must contain real numerical values')
    result = np.asarray(np.ma.asarray(raw, dtype=np.float64).filled(np.nan))
    if np.isinf(result).any():
        raise ValueError(f'{name} contains infinity; use NaN or a mask for missing data')
    return result


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _years(years, size):
    result = _real_array(years, 'years')
    if (result.ndim != 1 or result.size != size or not np.isfinite(result).all()
            or not np.equal(result, np.rint(result)).all()
            or np.any(np.diff(result) <= 0)):
        raise ValueError('years must be finite unique increasing integer labels matching time')
    return result


def _field(value, name='field'):
    result = _real_array(value, name)
    if result.ndim == 0 or result.shape[0] < 2 or not result.size or any(n == 0 for n in result.shape):
        raise ValueError(f'{name} must be nonempty with at least two time samples on axis zero')
    return result


def _pearson_columns(x, y, valid):
    """Stable paired Pearson r; offset and scale each endpoint separately."""
    counts = valid.sum(axis=0)
    columns = np.arange(valid.shape[1])
    first = np.argmax(valid, axis=0)
    normalized = []
    nonconstant = np.ones(len(columns), dtype=bool)
    for values in (x, y):
        values = np.broadcast_to(values, valid.shape)
        reference = np.where(counts > 0, values[first, columns], 0.)
        delta = np.zeros(values.shape)
        with np.errstate(over='ignore', invalid='ignore'):
            np.subtract(values, reference, out=delta, where=valid)
        if not np.isfinite(delta).all():
            raise ValueError('Numerical range is too large for stable centering')
        with np.errstate(over='ignore', invalid='ignore'):
            mean = delta.sum(axis=0) / np.maximum(counts, 1)
            delta = np.where(valid, delta - mean, 0.)
        if not np.isfinite(delta).all():
            raise ValueError('Numerical range is too large for stable mean removal')
        scale = np.max(np.abs(delta), axis=0)
        nonconstant &= scale > 0
        normalized.append(delta / np.where(scale > 0, scale, 1.))
    xx, yy = normalized
    denominator = np.sqrt(np.sum(xx * xx, axis=0) * np.sum(yy * yy, axis=0))
    r = np.full(counts.shape, np.nan)
    np.divide(np.sum(xx * yy, axis=0), denominator, out=r,
              where=(counts >= 3) & nonconstant & (denominator > 0))
    return np.clip(r, -1., 1.), counts


def _p_from_r(r, df):
    r, df = np.broadcast_arrays(r, df)
    output = np.full(r.shape, np.nan)
    valid = np.isfinite(r) & np.isfinite(df) & (df > 0)
    perfect = valid & (np.abs(r) == 1.)
    output[perfect] = 0.
    ordinary = valid & ~perfect
    statistic = np.zeros(r.shape)
    statistic[ordinary] = (np.abs(r[ordinary])
                           * np.sqrt(df[ordinary] / ((1-r[ordinary]) * (1+r[ordinary]))))
    output[ordinary] = 2. * stats.t.sf(statistic[ordinary], df[ordinary])
    return output


def lag1_bartlett(x, y, years, *, n_parameters=2, min_lag_pairs=3):
    """Lag-one effective N from already prepared paired residual series.

    ``y`` is (time, *space); ``x`` is either the same shape or a common
    one-dimensional series. NaN/masked pairs are excluded. Only pairs whose
    actual integer labels differ by one year enter either lag correlation.
    Separate endpoint means are used for each autocorrelation.

    Neff_raw = n*(1-rho_x*rho_y)/(1+rho_x*rho_y), capped above at actual n.
    df = Neff-n_parameters; callers must supply the number of fitted joint
    coefficients, including intercept and nuisance terms. No detrending or
    residualization occurs here. Undefined/short lag series yield missing
    adjusted p, with no nominal fallback. The raw estimate can be +infinity
    for exactly opposite unit lag correlations; the capped estimate is finite.
    """
    parameters = _integer(n_parameters, 'n_parameters', 2)
    minimum = _integer(min_lag_pairs, 'min_lag_pairs', 3)
    yy = _field(y, 'y')
    year = _years(years, yy.shape[0])
    xx = _real_array(x, 'x')
    if xx.ndim == 1 and xx.shape == (len(year),):
        xx = np.broadcast_to(xx.reshape((len(year),) + (1,) * (yy.ndim-1)), yy.shape)
    elif xx.shape != yy.shape:
        raise ValueError('x must be a shared time series or have exactly the shape of y')
    shape = yy.shape[1:]
    xx, yy = xx.reshape(len(year), -1), yy.reshape(len(year), -1)
    valid = np.isfinite(xx) & np.isfinite(yy)
    r, n = _pearson_columns(xx, yy, valid)
    lag_valid = valid[:-1] & valid[1:] & (np.diff(year) == 1)[:, None]
    rho_x, lag_count = _pearson_columns(xx[:-1], xx[1:], lag_valid)
    rho_y, _ = _pearson_columns(yy[:-1], yy[1:], lag_valid)
    rho_x[lag_count < minimum] = np.nan
    rho_y[lag_count < minimum] = np.nan
    product = rho_x * rho_y
    raw = np.full(n.shape, np.nan)
    defined = np.isfinite(product)
    np.divide(n * (1-product), 1+product, out=raw, where=defined & (product > -1))
    raw[defined & (product == -1.)] = np.inf
    effective = np.minimum(raw, n)
    df = effective - parameters
    nominal_df = n.astype(float) - parameters
    arrays = {'r': r, 'p': _p_from_r(r, df), 'p_nominal': _p_from_r(r, nominal_df),
              'n': n, 'n_eff': effective, 'n_eff_raw': raw, 'df': df,
              'df_nominal': nominal_df, 'n_lag_pairs': lag_count,
              'rho1_x': rho_x, 'rho1_y': rho_y}
    return {key: value.reshape(shape) for key, value in arrays.items()} | {
        'method': 'lag1_bartlett', 'n_parameters': parameters}


def benjamini_hochberg(p_values, alpha=.05):
    """BH adjusted p values across the finite entries of one explicit family.

    Shape is retained; missing p values have q=NaN and reject=False. The
    family is all finite supplied entries, never selected only by their p.
    Classical FDR control assumes independent or suitable positively
    dependent tests; arbitrary spatial dependence is not covered here.
    """
    if (isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real)
            or not np.isfinite(alpha) or not 0 < alpha <= 1):
        raise ValueError('alpha must be finite and in (0, 1]')
    values = _real_array(p_values, 'p_values')
    valid = np.isfinite(values)
    if np.any(valid & ((values < 0) | (values > 1))):
        raise ValueError('Finite p values must be in [0, 1]')
    q = np.full(values.shape, np.nan)
    finite = values[valid]
    count = finite.size
    if count:
        order = np.argsort(finite, kind='stable')
        ranked = finite[order] * count / np.arange(1, count+1)
        adjusted = np.clip(np.minimum.accumulate(ranked[::-1])[::-1], 0., 1.)
        restored = np.empty(count)
        restored[order] = adjusted
        q[valid] = restored
    return {'q_values': q, 'reject': valid & (q <= alpha), 'n_tests': int(count),
            'alpha': float(alpha), 'method': 'benjamini_hochberg'}
