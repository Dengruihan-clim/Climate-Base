"""Horizontal interpolation and scalar T42 filtering for LBM basic states.

All fields use the last two dimensions for latitude and longitude.  Scalar
spectral filtering deliberately treats each supplied field independently,
including U and V, matching the scalar SPECSM step in the reference workflow.
"""

from functools import lru_cache

import numpy as np
from scipy.special import gammaln, lpmv


def gaussian_grid(nlat=64, nlon=128):
    """Return Gaussian latitudes north-to-south and longitudes in [0, 360).

    The quadrature nodes are sin(latitude), as required for a spherical-area
    integral.  T42L20 uses nlat=64 and nlon=128.
    """
    if int(nlat) != nlat or int(nlon) != nlon or nlat < 2 or nlon < 2:
        raise ValueError("nlat and nlon must be integers of at least 2")
    nodes, _ = np.polynomial.legendre.leggauss(int(nlat))
    lat = np.rad2deg(np.arcsin(nodes[::-1]))
    lon = np.arange(int(nlon), dtype=np.float64) * (360.0 / nlon)
    return lat, lon


def _as_missing_float(data):
    """Represent masked input and NaN/Inf source values as missing data."""
    return np.asarray(np.ma.asarray(data, dtype=np.float64).filled(np.nan))


def periodic_bilinear(data, lat, lon, target_lat, target_lon):
    """Interpolate a rectilinear field with cyclic longitude.

    ``data`` has shape (..., latitude, longitude).  Source latitude and
    longitude coordinates must be strictly increasing; the longitude span
    must be less than 360 degrees (no duplicated cyclic endpoint).  Target
    coordinates are independent 1-D axes and may be in either order.  Missing
    source corners are excluded and the remaining bilinear weights are
    normalized.  A point with no valid positive-weight corner remains NaN.
    Latitude extrapolation is rejected.
    """
    field = _as_missing_float(data)
    lat, lon, target_lat, target_lon = [
        np.asarray(axis, dtype=np.float64)
        for axis in (lat, lon, target_lat, target_lon)
    ]
    for name, axis in (("lat", lat), ("lon", lon),
                       ("target_lat", target_lat), ("target_lon", target_lon)):
        if axis.ndim != 1 or not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} must be a finite one-dimensional axis")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("at least two source latitudes and longitudes are required")
    if field.ndim < 2 or field.shape[-2:] != (lat.size, lon.size):
        raise ValueError("data's last dimensions must match latitude and longitude")
    if np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("source latitude and longitude must be strictly increasing")
    if lon[-1] - lon[0] >= 360.0:
        raise ValueError("source longitude must omit the duplicate cyclic endpoint")
    if np.any(target_lat < lat[0]) or np.any(target_lat > lat[-1]):
        raise ValueError("target latitude lies outside the source latitude range")

    # The final longitude cell joins the last sample to first+360 degrees.
    wrapped_lon = (target_lon - lon[0]) % 360.0 + lon[0]
    ilon = np.searchsorted(lon, wrapped_lon, side="right") - 1
    jlon = (ilon + 1) % lon.size
    right_lon = np.where(jlon == 0, lon[0] + 360.0, lon[jlon])
    wx = (wrapped_lon - lon[ilon]) / (right_lon - lon[ilon])
    ilat = np.clip(np.searchsorted(lat, target_lat, side="right") - 1,
                   0, lat.size - 2)
    jlat = ilat + 1
    wy = (target_lat - lat[ilat]) / (lat[jlat] - lat[ilat])

    shape = field.shape[:-2] + (target_lat.size, target_lon.size)
    numerator = np.zeros(shape, dtype=np.float64)
    denominator = np.zeros(shape, dtype=np.float64)
    for iy, yweight in ((ilat, 1.0 - wy), (jlat, wy)):
        for ix, xweight in ((ilon, 1.0 - wx), (jlon, wx)):
            corner = field[..., iy[:, None], ix[None, :]]
            weight = yweight[:, None] * xweight[None, :]
            valid = np.isfinite(corner)
            numerator += np.where(valid, corner, 0.0) * weight
            denominator += valid * weight
    result = np.full(shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 0.0)
    return result


@lru_cache(maxsize=8)
def _gaussian_bases(nlat, truncation):
    """Cache normalized associated-Legendre analysis/synthesis matrices."""
    nodes, weights = np.polynomial.legendre.leggauss(nlat)
    nodes, weights = nodes[::-1], weights[::-1]
    matrices = []
    for m in range(truncation + 1):
        degrees = np.arange(m, truncation + 1)
        normalization = np.exp(0.5 * (
            np.log((2.0 * degrees + 1.0) / 2.0)
            + gammaln(degrees - m + 1.0)
            - gammaln(degrees + m + 1.0)
        ))
        basis = lpmv(m, degrees[None, :], nodes[:, None]) * normalization
        analysis = weights[:, None] * basis
        basis.setflags(write=False)
        analysis.setflags(write=False)
        matrices.append((basis, analysis))
    return tuple(matrices)


def scalar_triangular_truncate(data, truncation=42):
    """Project a finite real Gaussian-grid scalar field onto l <= truncation.

    ``data`` is (..., nlat, nlon), using ``gaussian_grid(nlat, nlon)`` exactly:
    north-to-south Gaussian latitudes and equally spaced longitudes starting
    at zero.  For the LBM T42 grid the final shape is (64, 128).  The function
    also supports other adequately resolved Gaussian grids for verification.

    Longitude coefficients use NumPy's FFT convention.  Each zonal order m
    is analyzed with Gauss-Legendre weights and orthonormal associated
    Legendre functions, then synthesized using degrees m <= l <= truncation.
    No vector spherical-harmonic transformation is applied.
    """
    if np.iscomplexobj(data):
        raise ValueError("data must be real")
    field = _as_missing_float(data)
    if field.ndim < 2 or not np.all(np.isfinite(field)):
        raise ValueError("data must have latitude/longitude dimensions and be finite")
    if int(truncation) != truncation or truncation < 0:
        raise ValueError("truncation must be a nonnegative integer")
    truncation = int(truncation)
    nlat, nlon = field.shape[-2:]
    if truncation >= nlat or 2 * truncation >= nlon:
        raise ValueError("grid is too small for the requested spectral truncation")
    fourier = np.fft.rfft(field, axis=-1) / nlon
    filtered = np.zeros_like(fourier)
    for m, (basis, analysis) in enumerate(_gaussian_bases(nlat, truncation)):
        coefficients = fourier[..., :, m] @ analysis
        filtered[..., :, m] = coefficients @ basis.T
    return np.fft.irfft(filtered * nlon, n=nlon, axis=-1)
