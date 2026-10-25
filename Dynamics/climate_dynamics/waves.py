"""Horizontal stationary wave activity flux and Rossby wave sources.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).

Arrays use (..., latitude, longitude), with any leading axes preserved.
Coordinates are degrees; derivatives inside TN flux are with respect to
radians. No file access, missing-value filling, or predictor rescaling occurs.

Physical references:
Takaya & Nakamura (2001), DOI 10.1175/1520-0469(2001)058<0608:AFOAPI>2.0.CO;2
(the stationary horizontal flux only; no vertical or migratory component).
Sardeshmukh & Hoskins (1988), DOI
10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2 (Rossby wave source).
"""
from collections.abc import Mapping

import numpy as np


EARTH_RADIUS_M = 6_371_000.0
RWS_KEYS = ('eta', 'div', 'udiv', 'vdiv', 'eta_x', 'eta_y')


def _real_array(values, name):
    original = np.ma.asarray(values)
    if original.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numeric values, excluding booleans')
    return np.ma.asarray(original, dtype=float).filled(np.nan)


def _scalar(value, name):
    array = _real_array(value, name)
    if array.ndim != 0 or not np.isfinite(array):
        raise ValueError(f'{name} must be a finite scalar')
    return float(array)


def _broadcast_fields(arrays, names):
    fields = [_real_array(value, name) for value, name in zip(arrays, names)]
    if any(value.ndim < 2 or 0 in value.shape for value in fields):
        raise ValueError('Fields must be nonempty arrays with final latitude/longitude axes')
    spatial = fields[0].shape[-2:]
    if any(value.shape[-2:] != spatial for value in fields):
        raise ValueError('All fields must share the same final latitude/longitude shape')
    try:
        return np.broadcast_arrays(*fields)
    except ValueError as error:
        raise ValueError('Field leading dimensions must be broadcast compatible') from error


def _components(components, label):
    if not isinstance(components, Mapping):
        raise ValueError(f'{label} must be a mapping with the six RWS component keys')
    missing = [name for name in RWS_KEYS if name not in components]
    if missing:
        raise ValueError(f'Missing {label} RWS components: {missing}')
    return [components[name] for name in RWS_KEYS]


def _common_output(terms, valid):
    """Give each component and its sum exactly the same finite-cell mask."""
    with np.errstate(over='ignore', invalid='ignore'):
        total = sum(terms.values())
    for value in (*terms.values(), total):
        valid = valid & np.isfinite(value)
    return {name: np.where(valid, value, np.nan)
            for name, value in dict(terms, rws=total).items()}


def rossby_wave_source(components):
    """Return full S = -eta*div - Vchi dot grad(eta), in s**-2.

    ``components`` is a mapping containing six arrays: absolute vorticity
    ``eta`` (relative vorticity + Coriolis, s**-1), divergence ``div``
    (s**-1), divergent eastward/northward wind ``udiv``/``vdiv`` (m/s), and
    physical eastward/northward gradients ``eta_x``/``eta_y`` (s**-1 m**-1).
    Supply physical gradients, not derivatives per degree or per radian.

    Returns ``stretch``, ``advection``, and ``rws`` arrays. All six inputs,
    both terms and the total share one finite mask; masked/nonfinite input
    cells remain missing. Final latitude/longitude shapes must be identical;
    leading dimensions may broadcast. Inputs are not modified. This is a
    nonlinear diagnostic, so applying it to regressed components does not
    give the regression of the instantaneous source.

    Reference: Sardeshmukh & Hoskins (1988), DOI
    10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2.
    """
    values = _components(components, 'source')
    fields = _broadcast_fields(values, RWS_KEYS)
    eta, divergence, udiv, vdiv, eta_x, eta_y = fields
    valid = np.logical_and.reduce([np.isfinite(value) for value in fields])
    with np.errstate(over='ignore', invalid='ignore'):
        terms = {'stretch': -eta*divergence,
                 'advection': -udiv*eta_x-vdiv*eta_y}
    return _common_output(terms, valid)


def linear_rws(background, perturbation):
    """Return the first variation of the Rossby wave source in four terms.

    Both mappings use the six component keys and SI units documented in
    :func:`rossby_wave_source`. A perturbation may be a pattern, an anomaly,
    or component regression coefficients on the same original predictor
    scale. No component or predictor is standardized here.

    S' = -eta_bar*div' - eta'*div_bar
         -Vchi' dot grad(eta_bar) - Vchi_bar dot grad(eta').

    Returns ``stretch_div``, ``stretch_vort``, ``adv_pert_wind``,
    ``adv_pert_vort``, and their sum ``rws``, with one complete finite mask
    across all twelve inputs, all terms and the total. Output units are
    s**-2 (or s**-2 per predictor unit for coefficient inputs). The
    perturbation-by-perturbation products are omitted: this is neither a
    finite nonlinear difference nor a regression of instantaneous sources.

    Background eta includes planetary f. Perturbation eta is the change in
    absolute vorticity at fixed coordinates, hence the relative-vorticity
    perturbation: do not add planetary f again. Likewise perturbation eta_x
    and eta_y must exclude the fixed planetary-vorticity gradients. When
    diagnosing perturbation wind separately, explicitly use zero planetary
    rotation in that diagnostic.

    All arrays have identical final latitude/longitude shapes; leading axes
    may broadcast, including a two-dimensional background with a sequence
    of perturbations. No averaging or filling of missing cells occurs.

    Reference: the source follows Sardeshmukh & Hoskins (1988), DOI
    10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2. The four-term expression
    here is its direct algebraic first variation, not a separate method.
    """
    values = _components(background, 'background') + _components(perturbation, 'perturbation')
    names = [f'{label}.{name}' for label in ('background', 'perturbation') for name in RWS_KEYS]
    fields = _broadcast_fields(values, names)
    mean = dict(zip(RWS_KEYS, fields[:6]))
    delta = dict(zip(RWS_KEYS, fields[6:]))
    valid = np.logical_and.reduce([np.isfinite(value) for value in fields])
    with np.errstate(over='ignore', invalid='ignore'):
        terms = {
            'stretch_div': -mean['eta']*delta['div'],
            'stretch_vort': -delta['eta']*mean['div'],
            'adv_pert_wind': -delta['udiv']*mean['eta_x']-delta['vdiv']*mean['eta_y'],
            'adv_pert_vort': -mean['udiv']*delta['eta_x']-mean['vdiv']*delta['eta_y'],
        }
    return _common_output(terms, valid)


def _three_point_derivative(values, coordinates, axis):
    """Lagrange derivative on three contiguous finite points, no gap jumps."""
    field = np.moveaxis(values, axis, 0)
    output = np.full(field.shape, np.nan)

    def stencil(i, j, k):
        x0, x1, x2 = coordinates[[i, j, k]]
        c0 = (2*x0-x1-x2)/((x0-x1)*(x0-x2))
        c1 = (x0-x2)/((x1-x0)*(x1-x2))
        c2 = (x0-x1)/((x2-x0)*(x2-x1))
        valid = np.isfinite(field[i]) & np.isfinite(field[j]) & np.isfinite(field[k])
        return np.where(valid, c0*field[i]+c1*field[j]+c2*field[k], np.nan)

    with np.errstate(over='ignore', invalid='ignore'):
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


def _periodic_longitude_derivative(values, step):
    """Periodic central differences with contiguous same-side fallbacks."""
    before, after = np.roll(values, 1, axis=-1), np.roll(values, -1, axis=-1)
    with np.errstate(over='ignore', invalid='ignore'):
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


def tn_waf(psi, u_background, v_background, lat, lon, *, pressure_hpa=200.,
           min_speed=5., min_abs_lat=10., require_westerly=True,
           radius=EARTH_RADIUS_M):
    """Takaya--Nakamura stationary horizontal wave activity flux, m**2/s**2.

    ``psi`` is the perturbation rotational streamfunction pattern (m**2/s),
    not geopotential divided by f. Background total winds are in m/s.
    Fields have shape (..., latitude, longitude); leading dimensions may
    broadcast but the spatial shapes must match. ``lat`` is finite strictly
    increasing in [-90,90], with >=3 cells, possibly nonuniform. ``lon`` is
    a complete uniform strictly increasing 0<=lon<360 grid with >=4 cells
    and no cyclic duplicate. Coordinates are degrees; all derivatives are
    evaluated in radians. ``pressure_hpa`` is one finite positive scalar;
    loop explicitly for multiple pressure levels. ``radius`` is in meters.

    With lambda/phi in radians, p = pressure_hpa/1000 and speed = |U|,
    A = psi_lambda**2 - psi*psi_lambdalambda,
    B = psi_lambda*psi_phi - psi*psi_lambdaphi,
    C = psi_phi**2 - psi*psi_phiphi,

    Fx = p*cos(phi)/(2*speed*radius**2)
         * [U*A/cos(phi)**2 + V*B/cos(phi)],
    Fy = p*cos(phi)/(2*speed*radius**2)
         * [U*B/cos(phi) + V*C].

    Returns ``waf_x``/``waf_y``, eastward/northward flux. This diagnostic is
    quadratic in psi: doubling psi quadruples flux and reversing psi leaves
    it unchanged. A flux calculated from a regression pattern is NOT a
    linear regression coefficient or the regression of instantaneous flux.

    Latitude uses contiguous three-point Lagrange differences; longitude
    uses periodic central differences, both with contiguous same-side
    fallbacks beside missing cells. Second derivatives apply the operators
    twice; mixed derivatives average both orders only where both are
    finite. No stencil crosses a missing cell. Poles are removed before
    differentiation. Both outputs share the final finite mask, excluding
    |lat| <= min_abs_lat, zero/weak total wind speed < min_speed, and by
    default U <= 0. Masked/nonfinite inputs remain missing; no filling occurs.

    Reference: Takaya & Nakamura (2001), DOI
    10.1175/1520-0469(2001)058<0608:AFOAPI>2.0.CO;2. This implementation is
    restricted to stationary horizontal flux; it does not provide the full
    three-dimensional or migratory-wave formulation.
    """
    latitude, longitude = _real_array(lat, 'lat'), _real_array(lon, 'lon')
    if (latitude.ndim != 1 or latitude.size < 3 or not np.isfinite(latitude).all()
            or np.any(np.diff(latitude) <= 0) or np.any(np.abs(latitude) > 90)):
        raise ValueError('lat must contain >=3 finite ascending values in [-90,90] degrees')
    if (longitude.ndim != 1 or longitude.size < 4 or not np.isfinite(longitude).all()
            or np.any(longitude < 0) or np.any(longitude >= 360) or np.any(np.diff(longitude) <= 0)):
        raise ValueError('lon must be a complete increasing 0<=lon<360 global grid')
    phi, lam = np.deg2rad(latitude), np.deg2rad(longitude)
    step = 2*np.pi/longitude.size
    if not np.allclose(np.diff(np.r_[lam, lam[0]+2*np.pi]), step, atol=1e-10, rtol=1e-8):
        raise ValueError('lon must be a complete uniform global grid without a cyclic duplicate')
    pressure = _scalar(pressure_hpa, 'pressure_hpa')
    speed_limit, latitude_limit = _scalar(min_speed, 'min_speed'), _scalar(min_abs_lat, 'min_abs_lat')
    sphere_radius = _scalar(radius, 'radius')
    if pressure <= 0 or sphere_radius <= 0:
        raise ValueError('pressure_hpa and radius must be positive')
    radius_squared = sphere_radius*sphere_radius
    if not np.isfinite(radius_squared) or radius_squared == 0:
        raise ValueError('radius squared must be representable as a positive finite float')
    if speed_limit < 0 or not 0 <= latitude_limit < 90:
        raise ValueError('min_speed must be nonnegative and 0<=min_abs_lat<90')
    if not isinstance(require_westerly, (bool, np.bool_)):
        raise ValueError('require_westerly must be boolean')
    streamfunction, u, v = _broadcast_fields((psi, u_background, v_background),
                                            ('psi', 'u_background', 'v_background'))
    if streamfunction.shape[-2:] != (latitude.size, longitude.size):
        raise ValueError('Field spatial shape must match the latitude/longitude coordinate lengths')
    cosphi = np.cos(phi)[:, None]
    nonpolar = np.abs(cosphi) > 1e-10
    streamfunction = np.where(nonpolar & np.isfinite(streamfunction), streamfunction, np.nan)
    psi_lambda = _periodic_longitude_derivative(streamfunction, step)
    psi_phi = _three_point_derivative(streamfunction, phi, axis=-2)
    psi_lambdalambda = _periodic_longitude_derivative(psi_lambda, step)
    psi_phiphi = _three_point_derivative(psi_phi, phi, axis=-2)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        psi_lambdaphi = .5*(_three_point_derivative(psi_lambda, phi, axis=-2)
                           + _periodic_longitude_derivative(psi_phi, step))
        term_a = psi_lambda**2-streamfunction*psi_lambdalambda
        term_b = psi_lambda*psi_phi-streamfunction*psi_lambdaphi
        term_c = psi_phi**2-streamfunction*psi_phiphi
        speed = np.hypot(u, v)
        eligible = (nonpolar & (np.abs(latitude)[:, None] > latitude_limit)
                    & np.isfinite(u) & np.isfinite(v) & np.isfinite(speed)
                    & (speed > 0) & (speed >= speed_limit))
        if require_westerly:
            eligible &= u > 0
        safe_cos = np.where(nonpolar, cosphi, np.nan)
        safe_speed = np.where(eligible, speed, np.nan)
        coefficient = (pressure/1000.)*safe_cos/(2*safe_speed*radius_squared)
        flux_x = coefficient*(u*term_a/safe_cos**2+v*term_b/safe_cos)
        flux_y = coefficient*(u*term_b/safe_cos+v*term_c)
    valid = eligible & np.isfinite(flux_x) & np.isfinite(flux_y)
    return {'waf_x': np.where(valid, flux_x, np.nan),
            'waf_y': np.where(valid, flux_y, np.nan)}
