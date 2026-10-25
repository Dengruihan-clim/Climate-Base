"""Paired joint OLS maps with preserved predictor units and explicit inference.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
import numpy as np

from .inference import (_field, _integer, _p_from_r, _real_array, _years,
                        lag1_bartlett)


def linear_regression(predictor, field, years, controls=None, *,
                      include_trend=True, min_samples=None, inference='ols',
                      min_lag_pairs=3):
    """Fit field = intercept [+ year trend] [+ controls] + beta*predictor.

    Time is axis zero. A common predictor is one-dimensional; controls are
    either (time,), (time, k), or None. Every grid cell fits all coefficients
    on the same finite predictor/control/response years. Years must be unique
    increasing integer labels. Masked values and NaNs are missing; infinities
    and complex data are rejected. The predictor is never restandardized.

    Intercept is at year=0, predictor=0, controls=0. Slopes preserve each input
    unit. Centering/scaling inside the solver is numerical conditioning only.
    Constant/near-collinear designs and too-short local samples have missing
    estimates. A constant response has beta=0 but undefined r and p.

    ``ols`` uses df=n-p for the p joint coefficients. ``lag1_bartlett`` uses
    lag-one correlations of predictor and response after nuisance removal,
    pairing actual adjacent calendar years. Its adjusted df is Neff-p, its
    standard error rescales the nominal OLS error by sqrt((n-p)/(Neff-p)),
    and undefined lag diagnostics yield missing adjusted p/error. This is a
    stationary-residual approximation, not HAC or a fitted AR model.
    """
    yfield = _field(field)
    year = _years(years, yfield.shape[0])
    x = _real_array(predictor, 'predictor')
    if x.ndim != 1 or x.shape != year.shape:
        raise ValueError('predictor must be one-dimensional and match the time axis')
    if not isinstance(include_trend, (bool, np.bool_)):
        raise ValueError('include_trend must be boolean')
    if inference not in ('ols', 'lag1_bartlett'):
        raise ValueError("inference must be 'ols' or 'lag1_bartlett'")
    _integer(min_lag_pairs, 'min_lag_pairs', 3)
    if controls is None:
        control = np.empty((len(year), 0))
    else:
        control = _real_array(controls, 'controls')
        if control.ndim == 1:
            control = control[:, None]
        if control.ndim != 2 or control.shape[0] != len(year) or control.shape[1] == 0:
            raise ValueError('controls must be a time series or a nonempty (time, k) array')
    names = ['intercept'] + (['year'] if include_trend else [])
    names += [f'control_{index}' for index in range(control.shape[1])] + ['predictor']
    parameters = len(names)
    minimum = parameters + 1 if min_samples is None else _integer(min_samples, 'min_samples', parameters + 1)
    if len(year) < minimum:
        raise ValueError(f'At least {minimum} time samples are required for this joint model')
    shape = yfield.shape[1:]
    y = yfield.reshape(len(year), -1)
    design_valid = np.isfinite(x) & np.isfinite(control).all(axis=1)
    valid = np.isfinite(y) & design_valid[:, None]
    counts = valid.sum(axis=0)
    columns = y.shape[1]
    coefficients = np.full((parameters, columns), np.nan)
    arrays = {key: np.full(columns, np.nan) for key in
              ('r', 'p', 'p_nominal', 'stderr', 'stderr_nominal', 'n_eff',
               'df', 'rho1_predictor', 'rho1_field', 'model_r_squared')}
    ranks = np.zeros(columns, dtype=np.int32)
    fitted = np.zeros(columns, dtype=bool)
    lag_counts = np.zeros(columns, dtype=np.int32)

    def fit(mask, selected):
        n = int(mask.sum())
        if n < minimum or not selected.size:
            return
        raw = np.column_stack(([year[mask]] if include_trend else [])
                              + [control[mask, index] for index in range(control.shape[1])]
                              + [x[mask]])
        with np.errstate(over='ignore', invalid='ignore'):
            means = raw.mean(axis=0)
            centered = raw - means
        if not np.isfinite(means).all() or not np.isfinite(centered).all():
            raise ValueError('Numerical range is too large for finite predictor centering')
        scales = np.max(np.abs(centered), axis=0)
        safe_scales = np.where(scales > 0, scales, 1.)
        design = np.column_stack((np.ones(n), centered / safe_scales))
        singular = np.linalg.svd(design, compute_uv=False)
        tolerance = 64 * np.finfo(float).eps * max(design.shape) * singular[0]
        rank = int(np.sum(singular > tolerance))
        ranks[selected] = rank
        if rank != parameters:
            return
        response = y[np.ix_(mask, selected)]
        # An anchored mean preserves small variations on a large offset.
        with np.errstate(over='ignore', invalid='ignore'):
            shifted = response - response[:1]
            mean_delta = shifted.mean(axis=0)
            response_mean = response[0] + mean_delta
            centered_response = shifted - mean_delta
        if not np.isfinite(response_mean).all() or not np.isfinite(centered_response).all():
            raise ValueError('Numerical range is too large for finite response centering')
        response_scale = np.max(np.abs(centered_response), axis=0)
        safe_response_scale = np.where(response_scale > 0, response_scale, 1.)
        normalized_response = centered_response / safe_response_scale
        normalized_coef = np.linalg.lstsq(design, normalized_response, rcond=None)[0]
        with np.errstate(over='ignore', invalid='ignore'):
            raw_slopes = normalized_coef[1:] * safe_response_scale / safe_scales[:, None]
            intercept = response_mean - np.sum(raw_slopes * means[:, None], axis=0)
        if not np.isfinite(raw_slopes).all() or not np.isfinite(intercept).all():
            raise ValueError('Coefficient range is not representable as finite float64')
        coefficients[:, selected] = np.vstack((intercept, raw_slopes))
        residual = normalized_response - design @ normalized_coef
        sse = np.sum(residual * residual, axis=0)
        df_nominal = n - parameters
        _, triangle = np.linalg.qr(design, mode='reduced')
        inverse = np.linalg.inv(triangle)
        predictor_variance_factor = np.sum(inverse[-1] ** 2)
        nominal_error = (np.sqrt(sse / df_nominal * predictor_variance_factor)
                         * safe_response_scale / safe_scales[-1])
        if not np.isfinite(nominal_error).all():
            raise ValueError('Standard-error range is not representable as finite float64')
        nuisance_q = np.linalg.qr(design[:, :-1], mode='reduced')[0]
        xr = design[:, -1] - nuisance_q @ (nuisance_q.T @ design[:, -1])
        yr = normalized_response - nuisance_q @ (nuisance_q.T @ normalized_response)
        numerator = np.sum(xr[:, None] * yr, axis=0)
        denominator = np.sqrt(np.sum(xr * xr) * np.sum(yr * yr, axis=0))
        meaningful = (np.linalg.norm(yr, axis=0)
                       > 64 * np.finfo(float).eps * max(design.shape)
                       * np.linalg.norm(normalized_response, axis=0))
        meaningful &= response_scale > 0
        r = np.full(len(selected), np.nan)
        np.divide(numerator, denominator, out=r, where=(denominator > 0) & meaningful)
        r = np.clip(r, -1., 1.)
        arrays['r'][selected] = r
        arrays['p_nominal'][selected] = _p_from_r(r, df_nominal)
        arrays['stderr_nominal'][selected] = nominal_error
        total = np.sum(normalized_response * normalized_response, axis=0)
        arrays['model_r_squared'][selected] = np.divide(
            total-sse, total, out=np.full(len(selected), np.nan), where=total > 0)
        fitted[selected] = True
        if inference == 'ols':
            arrays['n_eff'][selected] = n
            arrays['df'][selected] = df_nominal
            arrays['p'][selected] = arrays['p_nominal'][selected]
            arrays['stderr'][selected] = nominal_error
        else:
            correction = lag1_bartlett(xr, yr, year[mask],
                                       n_parameters=parameters, min_lag_pairs=min_lag_pairs)
            for output_key, correction_key in (
                    ('n_eff', 'n_eff'), ('df', 'df'),
                    ('rho1_predictor', 'rho1_x'), ('rho1_field', 'rho1_y')):
                arrays[output_key][selected] = np.where(meaningful, correction[correction_key], np.nan)
            lag_counts[selected] = correction['n_lag_pairs']
            arrays['p'][selected] = _p_from_r(r, arrays['df'][selected])
            eligible = np.isfinite(arrays['df'][selected]) & (arrays['df'][selected] > 0)
            error = np.full(len(selected), np.nan)
            error[eligible] = nominal_error[eligible] * np.sqrt(df_nominal / arrays['df'][selected][eligible])
            arrays['stderr'][selected] = error

    complete = valid.all(axis=0)
    if np.any(complete):
        fit(np.ones(len(year), dtype=bool), np.flatnonzero(complete))
    incomplete = np.flatnonzero(~complete)
    if incomplete.size:
        patterns, inverse = np.unique(valid[:, incomplete].T, axis=0, return_inverse=True)
        for group, mask in enumerate(patterns):
            fit(mask, incomplete[inverse == group])
    result = {key: value.reshape(shape) for key, value in arrays.items()}
    coefficients = coefficients.reshape((parameters,) + shape)
    control_start = 2 if include_trend else 1
    return result | {
        'beta': coefficients[-1], 'intercept': coefficients[0],
        'trend_beta': coefficients[1] if include_trend else np.full(shape, np.nan),
        'control_beta': coefficients[control_start:control_start+control.shape[1]],
        'coefficients': coefficients, 'coefficient_names': tuple(names),
        'n': counts.reshape(shape), 'df_nominal': (counts-parameters).reshape(shape),
        'rank': ranks.reshape(shape), 'fit_valid': fitted.reshape(shape),
        'n_lag_pairs': lag_counts.reshape(shape), 'n_parameters': parameters,
        'method': inference, 'include_trend': bool(include_trend),
        'predictor_scale': 'original supplied values; no restandardization'}
