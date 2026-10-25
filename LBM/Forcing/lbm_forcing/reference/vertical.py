"""Missing-aware LBM Z2PS, SLP-to-PS, and natural log-pressure P2S.

Pressure is in hPa, height in m, temperature in K, and specific humidity in
kg/kg.  Field arrays use (pressure, latitude, longitude); surface arrays use
(latitude, longitude).  Missing pressure-level values remain NaN until each
column's interpolation, which discards them and reports endpoint extension.
"""
import numpy as np
from scipy.interpolate import CubicSpline

RAIR = 287.04
GRAV = 9.8
EPSV = 0.622


def _float_array(value):
    return np.asarray(np.ma.asarray(value, dtype=float).filled(np.nan))


def _pressure_axis(pressure):
    pressure = _float_array(pressure)
    if (pressure.ndim != 1 or pressure.size < 2
            or not np.isfinite(pressure).all() or np.any(pressure <= 0)
            or np.any(np.diff(pressure) >= 0)):
        raise ValueError('Pressure must be finite, positive, one-dimensional, '
                         'and strictly descend from surface to top')
    return pressure


def _sigma_axis(sigma):
    sigma = _float_array(sigma)
    if (sigma.ndim != 1 or sigma.size == 0
            or not np.isfinite(sigma).all()
            or np.any((sigma <= 0) | (sigma >= 1))
            or np.any(np.diff(sigma) >= 0)):
        raise ValueError('Sigma must be a finite one-dimensional strictly '
                         'decreasing axis within (0, 1)')
    return sigma


def height_to_surface_pressure(height, pressure, topography):
    """Interpolate ln(p) in height; linearly extend the nearest two levels."""
    height = _float_array(height)
    pressure = _pressure_axis(pressure)
    topography = _float_array(topography)
    if (height.ndim < 2 or height.shape[0] != pressure.size
            or height.shape[1:] != topography.shape or not topography.size):
        raise ValueError('Height, pressure and topography dimensions do not match')
    if not np.isfinite(topography).all():
        raise ValueError('Topography must be finite')
    result = np.empty(topography.size)
    below = 0
    above = 0
    minimum_layers = len(pressure)
    maximum_extension = 0.
    for col, (h, zs) in enumerate(zip(
            height.reshape(len(pressure), -1).T, topography.ravel())):
        valid = np.isfinite(h)
        hv = h[valid]
        logp = np.log(pressure[valid])
        minimum_layers = min(minimum_layers, len(hv))
        if len(hv) < 2 or np.any(np.diff(hv) <= 0):
            raise ValueError(f'Insufficient or nonmonotonic height column {col}')
        below += int(zs < hv[0])
        above += int(zs > hv[-1])
        maximum_extension = max(maximum_extension,
                                float(max(hv[0] - zs, zs - hv[-1], 0)))
        upper = np.clip(np.searchsorted(hv, zs), 1, len(hv) - 1)
        lower = upper - 1
        fraction = (zs - hv[lower]) / (hv[upper] - hv[lower])
        result[col] = np.exp(logp[lower] + fraction * (logp[upper] - logp[lower]))
    return result.reshape(topography.shape), {
        'columns': topography.size,
        'below_lowest_valid_height_columns': below,
        'above_highest_valid_height_columns': above,
        'maximum_height_extrapolation_m': maximum_extension,
        'minimum_valid_height_levels': minimum_layers}


def pressure_to_sigma(field, pressure, surface_pressure, sigma):
    """Natural cubic spline in ln(p); discard missing layers, clamp endpoints."""
    field = _float_array(field)
    pressure = _pressure_axis(pressure)
    surface_pressure = _float_array(surface_pressure)
    sigma = _sigma_axis(sigma)
    if (field.ndim < 2 or field.shape[0] != pressure.size
            or field.shape[1:] != surface_pressure.shape
            or not surface_pressure.size):
        raise ValueError('Pressure-to-sigma dimension mismatch')
    if not np.isfinite(surface_pressure).all() or np.any(surface_pressure <= 0):
        raise ValueError('Surface pressure must be finite and positive')
    x = np.log(pressure[::-1])
    target = np.log(sigma[:, None] * surface_pressure.ravel()[None, :])
    output = np.empty(target.shape)
    low = np.zeros(len(sigma), dtype=int)
    high = low.copy()
    discarded = 0
    for col, profile in enumerate(field[::-1].reshape(len(pressure), -1).T):
        valid = np.isfinite(profile)
        xv = x[valid]
        yv = profile[valid]
        if len(xv) < 2:
            raise ValueError(f'Insufficient valid pressure levels in column {col}')
        discarded += int((~valid).sum())
        low += target[:, col] > xv[-1]
        high += target[:, col] < xv[0]
        clipped = np.clip(target[:, col], xv[0], xv[-1])
        output[:, col] = CubicSpline(
            xv, yv, bc_type='natural', extrapolate=False)(clipped)
    return output.reshape((len(sigma),) + surface_pressure.shape), {
        'missing_pressure_values_discarded': discarded,
        'lower_endpoint_extension_by_sigma': low.tolist(),
        'upper_endpoint_extension_by_sigma': high.tolist(),
        'boundary_rule': 'constant nearest valid value; natural cubic spline '
                         'in log(p) inside bounds'}


def sea_level_to_surface_pressure(slp, topography, lowest_temperature,
                                  lowest_humidity, lowest_sigma):
    """Port ecmsbs.f's virtual-temperature/lapse-rate SLP reduction exactly."""
    sigma = _float_array(lowest_sigma)
    if (sigma.ndim != 0 or not np.isfinite(sigma)
            or not 0 < float(sigma) < 1):
        raise ValueError('Lowest sigma must be a finite scalar within (0, 1)')
    try:
        slp, z, t, q = np.broadcast_arrays(*map(_float_array, (
            slp, topography, lowest_temperature, lowest_humidity)))
    except ValueError as exc:
        raise ValueError('SLP reduction field shapes must be broadcast-compatible') from exc
    if not slp.size or not all(np.isfinite(x).all() for x in (slp, z, t, q)):
        raise ValueError('SLP reduction requires nonempty finite fields')
    if np.any(slp <= 0) or np.any(t <= 0) or np.any(q < 0):
        raise ValueError('SLP and temperature must be positive; humidity must be nonnegative')
    lapse = np.full(t.shape, 0.0065)
    tv = t * (1 + (1 / EPSV - 1) * q)
    tvs = np.array(tv * (1 + lapse * RAIR / GRAV * (1 / float(sigma) - 1)),
                   dtype=float, copy=True)
    tv0 = tvs + z * lapse
    a = (tv0 > 290.5) & (tvs <= 290.5)
    b = (tv0 > 290.5) & (tvs > 290.5)
    c = (~a) & (~b) & (tvs < 255.)
    lapse[a] = (290.5 - tvs[a]) / z[a]
    lapse[b] = 0.
    tvs[b] = 0.5 * (290.5 + tvs[b])
    tvs[c] = 0.5 * (255. + tvs[c])
    x = lapse * z / tvs
    factor = np.exp(GRAV * z / (RAIR * tvs) * (1 - x / 2 + x * x / 3))
    return slp / factor
