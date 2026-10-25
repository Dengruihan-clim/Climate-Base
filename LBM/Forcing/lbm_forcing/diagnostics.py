"""Pressure-resolved apparent heating from monthly reanalysis fields.

Returned fields are Q1/Cp in K/day. No column integration, pressure-level
averaging, or submonthly transient-flux reconstruction is performed.
"""
from __future__ import annotations
import numpy as np

CP = 1004.0
RD = 287.05
EARTH_RADIUS_M = 6_371_000.0
SECONDS_PER_DAY = 86400.0


def _three_point_derivative(values, coordinates, axis):
    """Nonuniform derivative with contiguous central or same-side stencils.

    An invalid center stays invalid. No stencil jumps across missing levels
    or adjacent missing spatial cells. Three finite adjacent points are
    needed, including for a one-sided derivative at a physical boundary.
    """
    coordinates = np.asarray(coordinates, float)
    if (coordinates.ndim != 1 or len(coordinates) < 3 or
            not np.isfinite(coordinates).all() or np.any(np.diff(coordinates) <= 0)):
        raise ValueError('Derivative coordinates must contain >=3 finite ascending values')
    field = np.moveaxis(np.asarray(values, float), axis, 0)
    if field.shape[0] != len(coordinates):
        raise ValueError('Derivative coordinate length differs from its field axis')
    output = np.full(field.shape, np.nan)

    def stencil(i, j, k):
        x0, x1, x2 = coordinates[[i, j, k]]
        c0 = (2*x0-x1-x2)/((x0-x1)*(x0-x2))
        c1 = (x0-x2)/((x1-x0)*(x1-x2))
        c2 = (x0-x1)/((x2-x0)*(x2-x1))
        valid = np.isfinite(field[i]) & np.isfinite(field[j]) & np.isfinite(field[k])
        return np.where(valid, c0*field[i]+c1*field[j]+c2*field[k], np.nan)

    for i in range(len(coordinates)):
        if 0 < i < len(coordinates)-1:
            output[i] = stencil(i, i-1, i+1)
        if i+2 < len(coordinates):
            forward = stencil(i, i+1, i+2)
            output[i] = np.where(np.isfinite(output[i]), output[i], forward)
        if i-2 >= 0:
            backward = stencil(i, i-1, i-2)
            output[i] = np.where(np.isfinite(output[i]), output[i], backward)
    return np.moveaxis(output, 0, axis)


def _periodic_longitude_derivative(values, longitude_radians):
    longitude_radians = np.asarray(longitude_radians, float)
    if longitude_radians.ndim != 1 or len(longitude_radians) < 4:
        raise ValueError('A complete cyclic longitude grid needs at least four cells')
    spacing = np.diff(np.r_[longitude_radians, longitude_radians[0]+2*np.pi])
    step = 2*np.pi/len(longitude_radians)
    if (not np.isfinite(longitude_radians).all() or
            not np.allclose(spacing, step, atol=1e-10, rtol=1e-8)):
        raise ValueError('Longitude must be a complete increasing uniform global grid without a cyclic duplicate')
    values = np.asarray(values, float)
    before, after = np.roll(values, 1, axis=-1), np.roll(values, -1, axis=-1)
    central = (after-before)/(2*step)
    valid = np.isfinite(values) & np.isfinite(before) & np.isfinite(after)
    output = np.where(valid, central, np.nan)
    after2 = np.roll(values, -2, axis=-1)
    forward = (-3*values+4*after-after2)/(2*step)
    valid_forward = np.isfinite(values) & np.isfinite(after) & np.isfinite(after2)
    output = np.where(~np.isfinite(output) & valid_forward, forward, output)
    before2 = np.roll(values, 2, axis=-1)
    backward = (3*values-4*before+before2)/(2*step)
    valid_backward = np.isfinite(values) & np.isfinite(before) & np.isfinite(before2)
    return np.where(~np.isfinite(output) & valid_backward, backward, output)


def _as_float(value):
    return np.ma.asarray(value, dtype=float).filled(np.nan)


def _air_mask(surface_pressure, pressure, horizontal_shape, name):
    if surface_pressure is None:
        return np.ones((len(pressure), *horizontal_shape), dtype=bool)
    ps = _as_float(surface_pressure)
    if ps.shape != horizontal_shape:
        raise ValueError(f'{name} must match the latitude/longitude grid')
    if np.any(np.isfinite(ps) & (ps <= 0)):
        raise ValueError(f'{name} must contain positive surface pressure in Pa')
    # Typical hPa input is an unsafe silent unit error; near-vacuum cells
    # should be missing rather than assigned an atmospheric surface pressure.
    if np.any(np.isfinite(ps)) and np.nanmax(ps) < 2000:
        raise ValueError(f'{name} appears to be hPa: supply surface pressure in Pa')
    return np.isfinite(ps)[None] & (pressure[:, None, None] <= ps[None])


def q1_layers(t_prev, t, t_next, u, v, omega, pressure_pa, lat, lon,
              dt_prev_seconds, dt_next_seconds, surface_pressure=None,
              surface_pressure_prev=None, surface_pressure_next=None):
    """Return Q1/cp and its three terms in K day-1, without integration.

    All six fields have shape (pressure, latitude, longitude), using K,
    m s-1 and omega=dp/dt in Pa s-1 (positive downward). Pressure is positive
    increasing Pa with at least three levels; latitude increases; longitude
    is a full uniform 0<=lon<360 cyclic grid without a duplicate endpoint.
    Every supplied native pressure level is returned, including the ends,
    where three-point one-sided derivatives are used.

    The time derivative uses actual neighboring month-midpoint intervals
    and the unequal three-point Lagrange stencil at the current month.
    Current PS masks current T/u/v/omega before any spatial derivative;
    previous/next PS independently mask their own T before the time
    derivative. Each optional PS has (latitude, longitude) shape in Pa;
    omitted PS means no PS mask for that month. No stencil crosses a missing
    neighboring cell. Poles and any cell lacking any term are missing in
    all four outputs, preserving exact term closure and shared sampling.

    Multiply any returned field by CP/86400 for W kg-1.
    """
    pressure = _as_float(pressure_pa)
    latitude = _as_float(lat)
    longitude = _as_float(lon)
    if (pressure.ndim != 1 or len(pressure) < 3 or
            not np.isfinite(pressure).all() or np.any(pressure <= 0) or
            np.any(np.diff(pressure) <= 0)):
        raise ValueError('Pressure must contain >=3 finite strictly increasing positive Pa levels')
    if (latitude.ndim != 1 or len(latitude) < 3 or
            not np.isfinite(latitude).all() or np.any(np.abs(latitude) > 90) or
            np.any(np.diff(latitude) <= 0)):
        raise ValueError('Latitude must contain >=3 finite increasing values within -90..90')
    if (longitude.ndim != 1 or not np.isfinite(longitude).all() or
            np.any(longitude < 0) or np.any(longitude >= 360)):
        raise ValueError('Longitude must use increasing 0<=lon<360 native centers')
    h0, h1 = float(dt_prev_seconds), float(dt_next_seconds)
    if not np.isfinite(h0) or not np.isfinite(h1) or h0 <= 0 or h1 <= 0:
        raise ValueError('Both neighboring month-midpoint time intervals must be finite positive seconds')
    shape = (len(pressure), len(latitude), len(longitude))
    fields = [_as_float(value) for value in (t_prev, t, t_next, u, v, omega)]
    if any(value.shape != shape for value in fields):
        raise ValueError('Temperature, wind and omega arrays must share (pressure, latitude, longitude) shape')
    prev, current, nxt, uwind, vwind, pressure_velocity = fields
    air = _air_mask(surface_pressure, pressure, shape[1:], 'surface_pressure')
    prev_air = _air_mask(surface_pressure_prev, pressure, shape[1:], 'surface_pressure_prev')
    next_air = _air_mask(surface_pressure_next, pressure, shape[1:], 'surface_pressure_next')
    prev = np.where(prev_air, prev, np.nan)
    nxt = np.where(next_air, nxt, np.nan)
    current, uwind, vwind, pressure_velocity = [
        np.where(air, value, np.nan)
        for value in (current, uwind, vwind, pressure_velocity)]
    time_rate = (-h1/(h0*(h0+h1))*prev + (h1-h0)/(h0*h1)*current
                 + h0/(h1*(h0+h1))*nxt)
    phi, lam = np.deg2rad(latitude), np.deg2rad(longitude)
    dtdp = _three_point_derivative(current, pressure, axis=0)
    dtdphi = _three_point_derivative(current, phi, axis=1)
    dtdlambda = _periodic_longitude_derivative(current, lam)
    cosphi = np.cos(phi)
    safe_cos = np.where(np.abs(cosphi) > 1e-10, cosphi, np.nan)
    horizontal = (uwind*dtdlambda/(EARTH_RADIUS_M*safe_cos[None, :, None])
                  + vwind*dtdphi/EARTH_RADIUS_M)
    vertical = pressure_velocity*(dtdp-(RD/CP)*current/pressure[:, None, None])
    terms = {'tendency': time_rate, 'horizontal': horizontal, 'vertical': vertical}
    common = np.logical_and.reduce([np.isfinite(value) for value in terms.values()])
    common &= np.isfinite(safe_cos)[None, :, None]
    result = {name: np.where(common, value*SECONDS_PER_DAY, np.nan)
              for name, value in terms.items()}
    result['q1'] = result['tendency']+result['horizontal']+result['vertical']
    return result


