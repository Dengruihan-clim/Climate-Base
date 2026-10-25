"""Global spherical wind and Helmholtz diagnostics with an optional backend.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).

This interface uses windspharm.standard.VectorWind; it does not implement a
replacement spherical-harmonic transform. Cite Dawson (2016), "Windspharm:
A High-Level Library for Global Wind Field Computations Using Spherical
Harmonics", https://doi.org/10.5334/jors.129. The pyspharm/SPHEREPACK backend
has its own authorship and terms. No upstream implementation is distributed.

Only complete global, finite eastward/northward winds in m/s are accepted.
No missing-value filling, regional inversion, finite-difference fallback,
or sign reversal of northward wind is performed. Returned fields always use
ascending latitude, with leading dimensions preserved and longitude unchanged.
"""
from numbers import Real

import numpy as np


EARTH_RADIUS_M = 6_371_000.0
EARTH_OMEGA_S = 7.292115e-5
COORDINATE_TOLERANCE_DEGREES = 1e-5

FIELD_UNITS = {
    "psi": "m2 s-1", "chi": "m2 s-1",
    "vort": "s-1", "div": "s-1", "eta": "s-1",
    "eta_x": "s-1 m-1", "eta_y": "s-1 m-1",
    "urot": "m s-1", "vrot": "m s-1",
    "udiv": "m s-1", "vdiv": "m s-1",
    "u_total": "m s-1", "v_total": "m s-1",
    "uchi": "m s-1", "vchi": "m s-1",
}


def _real_array(value, name):
    raw = np.ma.asarray(value)
    if (not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw)
            or np.issubdtype(raw.dtype, np.bool_)):
        raise ValueError(f"{name} must contain real numerical values")
    array = np.asarray(np.ma.asarray(raw, dtype=np.float64).filled(np.nan))
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be entirely finite; masked/missing values cannot be filled")
    return array


def _parameter(value, name, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or (positive and value <= 0)):
        suffix = "positive and " if positive else ""
        raise ValueError(f"{name} must be a {suffix}finite real scalar")
    return float(value)


def _coordinates(lat, lon, gridtype):
    latitude = _real_array(lat, "latitude")
    longitude = _real_array(lon, "longitude")
    if (latitude.ndim != 1 or latitude.size < 3 or np.any(np.abs(latitude) > 90.)
            or not (np.all(np.diff(latitude) > 0) or np.all(np.diff(latitude) < 0))):
        raise ValueError("Latitude must be a strictly monotonic global one-dimensional axis with >=3 points")
    if gridtype not in ("regular", "gaussian"):
        raise ValueError("gridtype must explicitly be 'regular' or 'gaussian'")
    ascending = np.sort(latitude)
    if gridtype == "regular":
        if not np.array_equal(ascending[[0, -1]], [-90., 90.]):
            raise ValueError("Regular global latitude must include both poles exactly at -90 and +90 degrees")
        expected = np.linspace(-90., 90., latitude.size)
        description = "equally spaced and include both poles"
    else:
        expected = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(latitude.size)[0]))
        description = "match the complete Gaussian Legendre-node grid"
    if not np.allclose(ascending, expected, rtol=0, atol=COORDINATE_TOLERANCE_DEGREES):
        raise ValueError(f"Global {gridtype} latitude must {description}")
    if (longitude.ndim != 1 or longitude.size < 4
            or np.any(longitude < 0) or np.any(longitude >= 360)
            or np.any(np.diff(longitude) <= 0)):
        raise ValueError("Longitude must be increasing 0<=lon<360 without a duplicated endpoint")
    spacing = np.diff(np.r_[longitude, longitude[0] + 360.])
    if not np.allclose(spacing, 360. / longitude.size, rtol=0,
                       atol=COORDINATE_TOLERANCE_DEGREES):
        raise ValueError("Longitude must be a complete equally spaced cyclic global grid")
    return latitude, longitude


def _vector_wind_class():
    try:
        from windspharm.standard import VectorWind
    except ImportError as error:
        raise ImportError(
            "Global spherical diagnostics require optional windspharm and its "
            "pyspharm/SPHEREPACK backend; no numerical fallback is used"
        ) from error
    return VectorWind


def _backend_input(values, north_first, leading):
    ordered = np.take(values, north_first, axis=-2)
    nlat, nlon = values.shape[-2:]
    if not leading:
        return np.ascontiguousarray(ordered, dtype=np.float64)
    fields = ordered.reshape((-1, nlat, nlon)).transpose(1, 2, 0)
    return np.ascontiguousarray(fields, dtype=np.float64)


def _restore(values, name, nlat, nlon, leading):
    raw = np.ma.asarray(values)
    if not np.issubdtype(raw.dtype, np.number) or np.iscomplexobj(raw):
        raise RuntimeError(f"windspharm returned a non-real {name} field")
    array = np.asarray(np.ma.asarray(raw, dtype=np.float64).filled(np.nan))
    count = int(np.prod(leading, dtype=np.intp)) if leading else 1
    expected = (nlat, nlon, count) if leading else (nlat, nlon)
    if leading and count == 1 and array.shape == (nlat, nlon):
        array = array[..., None]
    if array.shape != expected or not np.isfinite(array).all():
        raise RuntimeError(f"windspharm returned an invalid global {name} shape or missing values")
    if leading:
        array = array[::-1].transpose(2, 0, 1).reshape((*leading, nlat, nlon))
    else:
        array = array[::-1]
    return np.ascontiguousarray(array, dtype=np.float64)


def diagnose(u, v, lat, lon, *, truncation=42, gridtype="regular",
             radius=EARTH_RADIUS_M, omega=EARTH_OMEGA_S, legfunc="stored"):
    """Diagnose a complete global wind using real spherical harmonics.

    Inputs have matching shape ``(*leading_dimensions, latitude, longitude)``
    and units m/s. Coordinates are degrees; latitude may ascend or descend,
    while longitude must increase over a full uniform 0..360 cycle. ``regular``
    requires endpoints exactly -90/+90 degrees; float32 rounding within 1e-5
    degrees is permitted only for its interior nodes. ``gaussian`` must match
    Gaussian Legendre nodes within the same 1e-5-degree rounding tolerance.

    Every output field uses ASCENDING latitude, preserved leading dimensions,
    and the supplied increasing longitude. Returned ``lat``/``lon`` make this
    ordering explicit. psi/chi are streamfunction/velocity potential (m2/s),
    vort/div are relative vorticity/divergence (s-1), eta is absolute vorticity,
    and eta_x/eta_y are physical eastward/northward gradients (s-1/m).
    urot/vrot and udiv/vdiv are rotational/divergent winds. uchi/vchi are aliases
    of udiv/vdiv; u_total/v_total sum the retained spectral components and may
    differ from the original wind after truncation. The backend's degree-zero
    potential gauge and eastward/northward signs are retained.

    ``radius`` is a positive sphere radius in metres; ``omega`` is rotation
    rate in s-1 and may be zero. Background diagnostics normally use Earth's
    rotation. Diagnose perturbation winds with ``omega=0`` when eta is to be
    passed as delta-eta to a linear Rossby-wave-source calculation: otherwise
    the constant planetary vorticity and its meridional gradient would be
    incorrectly included in the perturbation. A regression slope of full eta
    fitted with an intercept already removes a constant planetary term.

    A positive integer truncation applies to every diagnostic; it must fit
    both grids and lie below the longitudinal Nyquist mode. ``None`` retains
    the backend's native truncation. No detrending, interpolation, masking,
    missing-cell filling, or finite-difference substitute occurs here.

    Cite Dawson (2016), doi:10.5334/jors.129. windspharm and its underlying
    pyspharm/SPHEREPACK implementation retain their separate attribution.
    """
    latitude, longitude = _coordinates(lat, lon, gridtype)
    radius = _parameter(radius, "radius", positive=True)
    omega = _parameter(omega, "omega")
    if legfunc not in ("stored", "computed"):
        raise ValueError("legfunc must be 'stored' or 'computed'")
    maximum = min(latitude.size - 1, (longitude.size - 1) // 2)
    if truncation is not None and (
            isinstance(truncation, (bool, np.bool_))
            or not isinstance(truncation, (int, np.integer))
            or not 1 <= truncation <= maximum):
        raise ValueError(f"Triangular truncation must be an integer in 1..{maximum}, or None")
    if truncation is not None:
        truncation = int(truncation)
    uwind = _real_array(u, "U wind")
    vwind = _real_array(v, "V wind")
    expected = (latitude.size, longitude.size)
    if (uwind.ndim < 2 or uwind.shape != vwind.shape
            or uwind.shape[-2:] != expected or any(size == 0 for size in uwind.shape)):
        raise ValueError("U/V must have identical shape (*leading_dimensions, latitude, longitude)")
    leading = uwind.shape[:-2]
    north_first = np.argsort(latitude)[::-1]
    vector_wind = _vector_wind_class()(
        _backend_input(uwind, north_first, leading),
        _backend_input(vwind, north_first, leading),
        gridtype=gridtype, rsphere=radius, legfunc=legfunc,
    )
    psi, chi = vector_wind.sfvp(truncation=truncation)
    vort, div = vector_wind.vrtdiv(truncation=truncation)
    udiv, vdiv, urot, vrot = vector_wind.helmholtz(truncation=truncation)
    eta = vector_wind.absolutevorticity(omega=omega, truncation=truncation)
    eta_x, eta_y = vector_wind.gradient(eta, truncation=truncation)
    raw = {"psi": psi, "chi": chi, "vort": vort, "div": div, "eta": eta,
           "eta_x": eta_x, "eta_y": eta_y, "udiv": udiv, "vdiv": vdiv,
           "urot": urot, "vrot": vrot}
    result = {name: _restore(value, name, *expected, leading) for name, value in raw.items()}
    result["u_total"] = result["udiv"] + result["urot"]
    result["v_total"] = result["vdiv"] + result["vrot"]
    result["uchi"], result["vchi"] = result["udiv"], result["vdiv"]
    result.update({"lat": np.sort(latitude).copy(), "lon": longitude.copy(),
                   "gridtype": gridtype, "truncation": truncation,
                   "radius": radius, "omega": omega, "legfunc": legfunc,
                   "backend": "windspharm.standard.VectorWind",
                   "units": FIELD_UNITS.copy(), "latitude_order": "ascending"})
    return result
