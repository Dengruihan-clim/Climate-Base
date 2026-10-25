"""Longitude support and pressure-latitude mass-circulation diagnostics.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).

The conventional 2*pi*a*cos(latitude)/g streamfunction definition follows the
mass-continuity formulation discussed by Adam et al. (2018), The TropD
software package (v1): standardized methods for calculating tropical-width
diagnostics, GMD 11, 4339--4357, doi:10.5194/gmd-11-4339-2018.
That article is a formula reference, not a claim to reproduce its integration
algorithm or to redistribute TropD code.  Here the first supplied pressure
level is an explicit zero reference; no unknown atmosphere above it is added.
For an earlier observational application of zonal-mean meridional mass
streamfunctions, see Oort & Yienger (1996), Journal of Climate 9, 2751--2767,
doi:10.1175/1520-0442(1996)009<2751:OIVITH>2.0.CO;2. Neither paper is claimed
to use this module's regional support mask or its supplied-level boundary.

For a finite-longitude-band mean meridional wind, the result is a full-
circumference (2*pi-equivalent) relative streamfunction, not actual mass
transport integrated over that band.  Only complete global zonal support
permits the usual global-zonal-mean interpretation.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np


EARTH_RADIUS_M = 6_371_000.0
GRAVITY = 9.80665
FORMULA_REFERENCE = {
    "citation": "Adam et al. (2018), The TropD software package (v1): standardized methods for calculating tropical-width diagnostics, Geoscientific Model Development 11, 4339-4357",
    "doi": "10.5194/gmd-11-4339-2018",
    "scope": "Standard mass-continuity/streamfunction definition; no TropD implementation copied",
    "historical_application": "Oort & Yienger (1996), doi:10.1175/1520-0442(1996)009<2751:OIVITH>2.0.CO;2",
}


def _real(values, name):
    raw = np.ma.asarray(values)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numeric values, excluding booleans and strings")
    result = np.ma.asarray(raw, dtype=np.float64).filled(np.nan)
    return np.where(np.isfinite(result), result, np.nan)


def _axis(axis, ndim, name):
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError(f"{name} must be an integer axis")
    axis = int(axis)
    if not -ndim <= axis < ndim:
        raise ValueError(f"{name} is outside the array dimensions")
    return axis % ndim


def _positive(value, name):
    raw = np.ma.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu" or np.ma.getmaskarray(raw).any():
        raise ValueError(f"{name} must be a finite positive scalar")
    result = float(raw)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return result


def mass_streamfunction(v, pressure_pa, latitude, *, pressure_axis=-2,
                        latitude_axis=-1, reference_pressure_pa=None,
                        radius=EARTH_RADIUS_M, gravity=GRAVITY):
    """Return ``(psi, metadata)`` using a supplied-pressure zero reference.

    ``v`` is northward wind in m/s, normally ``(..., pressure, latitude)``.
    Pressure is finite, nonnegative, strictly increasing Pa.  Latitude may
    increase or decrease.  Output has exactly the input shape and units kg/s:

        psi(p,phi) = 2*pi*radius*cos(phi)/gravity * integral[p_ref,p] v dp.

    ``reference_pressure_pa=None`` means the first input pressure; an explicit
    reference must equal that first level exactly.  The value is zero there
    only if its wind is finite.  No extrapolation or assumed zero wind above
    the first level is introduced.  Every interval is trapezoidal; a missing
    level invalidates that level and every deeper level without restarting.

    A regional or incomplete-support longitude mean produces a 2*pi-equivalent
    circulation, not an actual band-integrated transport.  No longitude-width
    scaling is applied by this function.
    """
    values = _real(v, "v")
    if values.ndim < 2 or any(size == 0 for size in values.shape):
        raise ValueError("v must have nonempty pressure and latitude axes")
    p_axis = _axis(pressure_axis, values.ndim, "pressure_axis")
    y_axis = _axis(latitude_axis, values.ndim, "latitude_axis")
    if p_axis == y_axis:
        raise ValueError("Pressure and latitude axes must be distinct")
    pressure = _real(pressure_pa, "pressure_pa")
    lat = _real(latitude, "latitude")
    if (pressure.ndim != 1 or pressure.size < 2 or not np.isfinite(pressure).all()
            or np.any(pressure < 0) or np.any(np.diff(pressure) <= 0)):
        raise ValueError("Pressure must contain at least two finite, nonnegative, strictly increasing Pa levels")
    if (lat.ndim != 1 or not lat.size or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90)
            or (lat.size > 1 and not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)))):
        raise ValueError("Latitude must be finite, distinct and monotonic within -90..90")
    if values.shape[p_axis] != pressure.size or values.shape[y_axis] != lat.size:
        raise ValueError("Wind pressure/latitude dimensions do not match their coordinates")
    if reference_pressure_pa is None:
        reference = float(pressure[0])
    else:
        raw_reference = np.ma.asarray(reference_pressure_pa)
        if raw_reference.ndim != 0 or raw_reference.dtype.kind not in "fiu" or np.ma.getmaskarray(raw_reference).any():
            raise ValueError("reference_pressure_pa must equal the first input pressure")
        reference = float(raw_reference)
        if not np.isfinite(reference) or reference != pressure[0]:
            raise ValueError("reference_pressure_pa must equal the first input pressure; no upper-layer extrapolation is performed")
    radius, gravity = _positive(radius, "radius"), _positive(gravity, "gravity")
    canonical = np.moveaxis(values, (p_axis, y_axis), (-2, -1))
    finite_pairs = np.isfinite(canonical[..., 1:, :]) & np.isfinite(canonical[..., :-1, :])
    with np.errstate(over="ignore", invalid="ignore"):
        intervals = (.5 * canonical[..., 1:, :] + .5 * canonical[..., :-1, :]) * np.diff(pressure)[:, None]
        top = np.where(np.isfinite(canonical[..., :1, :]), 0., np.nan)
        integral = np.concatenate((top, np.cumsum(intervals, axis=-2)), axis=-2)
    if np.any(finite_pairs & ~np.isfinite(intervals)):
        raise ValueError("Pressure integration overflowed for finite input")
    expected_finite = np.logical_and.accumulate(np.isfinite(canonical), axis=-2)
    if np.any(expected_finite & ~np.isfinite(integral)):
        raise ValueError("Accumulated pressure integration overflowed for finite input")
    cosine = np.cos(np.deg2rad(lat))
    cosine[np.abs(lat) == 90.] = 0.
    with np.errstate(over="ignore", invalid="ignore"):
        psi = integral * (2 * np.pi * radius * cosine / gravity)
    if np.any(expected_finite & ~np.isfinite(psi)):
        raise ValueError("Mass streamfunction overflowed for finite input")
    psi = np.moveaxis(psi, (-2, -1), (p_axis, y_axis))
    metadata = {
        "units": "kg s-1", "wind_units": "m s-1", "pressure_units": "Pa",
        "pressure_axis": p_axis, "latitude_axis": y_axis,
        "reference_pressure_pa": reference, "radius_m": radius, "gravity_m_s2": gravity,
        "reference_value": "zero only where wind at the reference level is finite",
        "upper_boundary_assumption": "relative to first supplied pressure; unknown atmosphere above reference is not integrated or extrapolated",
        "integration": "trapezoidal over adjacent supplied pressure levels",
        "restart_below_missing": False, "missing_output_values": int(np.sum(~np.isfinite(psi))),
        "longitude_geometry": "full-circumference 2pi-equivalent of the supplied longitude-mean wind",
        "regional_interpretation": "A finite-band or incomplete-support mean does not give actual band-integrated mass transport; complete global support is required for a global-zonal interpretation",
        "formula_reference": dict(FORMULA_REFERENCE),
    }
    return psi, metadata


def _arc(bounds):
    if bounds is None:
        return 0., 360., [0., 360.]
    values = _real(bounds, "longitude_bounds")
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError("longitude_bounds must contain two finite degree values")
    west, east = map(float, values)
    difference = east - west
    if abs(difference) > 360. + 1e-10:
        raise ValueError("Longitude arc cannot exceed a full circumference")
    if np.isclose(abs(difference), 360., rtol=0, atol=1e-10):
        span = 360.
    else:
        span = difference % 360.
        if span <= 1e-12:
            raise ValueError("Identical longitude bounds are an empty arc; use a 360-degree difference for the globe")
    return west % 360., span, [west, east]


def _segments(left, width):
    left = float(left) % 360.
    end = left + float(width)
    if end <= 360.:
        return [(left, end)]
    return [(left, 360.), (0., end - 360.)]


def _width_total(weights):
    with np.errstate(over="ignore", invalid="ignore"):
        total = float(np.sum(weights))
    if not np.isfinite(total) or not 0. < total <= 360. + 1e-8:
        raise ValueError("Longitude widths must sum to a finite positive width no greater than 360 degrees")
    return total


def longitude_cell_weights(longitude, longitude_bounds=None, *, cyclic=False, cell_bounds=None):
    """Return ``(intersection_widths_degrees, metadata)`` in input order.

    Regions are eastward circular arcs: (170,-170) crosses the date line;
    (0,360), (-180,180), or None request the globe.  Noncyclic native centers
    may increase/decrease across one wrap; exterior edges use half the nearest
    spacing, matching regional center-grid conventions.  ``cyclic=True``
    explicitly declares a periodic global center grid and uses cyclic midpoint
    edges.  Optional ``cell_bounds`` supplies actual per-cell west/east bounds
    in original order and avoids center-edge inference altogether.

    Declared cells may not overlap, and must completely cover the requested
    arc.  Widths are longitude intersections at a fixed latitude, not cell
    areas or meridional weights.  No Gaussian-latitude assumptions are needed.
    """
    lon = _real(longitude, "longitude")
    if lon.ndim != 1 or not lon.size or not np.isfinite(lon).all():
        raise ValueError("Longitude must be a nonempty finite one-dimensional vector")
    if not isinstance(cyclic, (bool, np.bool_)):
        raise ValueError("cyclic must be boolean")
    normalized = lon % 360.
    if normalized.size > 1 and np.any(np.diff(np.sort(normalized)) <= 1e-10):
        raise ValueError("Longitude contains duplicate cyclic centers")
    west, span, declared_bounds = _arc(longitude_bounds)
    if cell_bounds is not None:
        supplied = _real(cell_bounds, "cell_bounds")
        if supplied.shape != (len(lon), 2) or not np.isfinite(supplied).all():
            raise ValueError("cell_bounds must be finite (longitude,2) west/east bounds")
        left, widths = [], []
        for center, (a, b) in zip(normalized, supplied):
            start, width, _ = _arc((a, b))
            if (center - start) % 360. > width + 1e-10:
                raise ValueError("A longitude center lies outside its declared cell")
            left.append(start)
            widths.append(width)
        left, widths = np.asarray(left), np.asarray(widths)
        method = "explicit cell west/east bounds"
    elif cyclic:
        if lon.size < 2:
            raise ValueError("At least two centers are required without explicit cell bounds")
        order = np.argsort(normalized)
        sorted_lon = normalized[order]
        gaps = np.diff(np.r_[sorted_lon, sorted_lon[0] + 360.])
        left_sorted = sorted_lon - .5 * np.roll(gaps, 1)
        width_sorted = .5 * (gaps + np.roll(gaps, 1))
        left, widths = np.empty_like(lon), np.empty_like(lon)
        left[order], widths[order] = left_sorted, width_sorted
        method = "periodic midpoint cell edges from an explicitly global center grid"
    else:
        if lon.size < 2:
            raise ValueError("At least two centers are required without explicit cell bounds")
        unwrapped = np.rad2deg(np.unwrap(np.deg2rad(lon)))
        difference = np.diff(unwrapped)
        if np.all(difference > 0):
            order = np.arange(lon.size)
        elif np.all(difference < 0):
            order = np.arange(lon.size - 1, -1, -1)
        else:
            raise ValueError("Noncyclic centers must be monotonic after longitude unwrapping")
        sorted_lon = unwrapped[order]
        edges = np.r_[sorted_lon[0] - .5 * (sorted_lon[1] - sorted_lon[0]),
                      .5 * (sorted_lon[:-1] + sorted_lon[1:]),
                      sorted_lon[-1] + .5 * (sorted_lon[-1] - sorted_lon[-2])]
        if edges[-1] - edges[0] > 360. + 1e-9:
            raise ValueError("Inferred noncyclic cells span more than one circumference; use explicit bounds or cyclic=True")
        left, widths = np.empty_like(lon), np.empty_like(lon)
        left[order], widths[order] = edges[:-1], np.diff(edges)
        method = "interior midpoint edges with exterior nearest-half-spacing extension"
    native_segments = []
    for start, width in zip(left, widths):
        native_segments.extend(_segments(start, width))
    ordered_segments = sorted(native_segments)
    tolerance = 1e-9
    previous_end = -np.inf
    for start, end in ordered_segments:
        if start < previous_end - tolerance:
            raise ValueError("Longitude cells overlap; actual intersection widths would double count")
        previous_end = max(previous_end, end)
    requested = _segments(west, span)
    weights = np.zeros(len(lon), dtype=np.float64)
    for index, (start, width) in enumerate(zip(left, widths)):
        for a, b in _segments(start, width):
            for c, d in requested:
                weights[index] += max(0., min(b, d) - max(a, c))
    represented = float(weights.sum())
    if not np.isclose(represented, span, rtol=0, atol=1e-8):
        raise ValueError("Native longitude cells do not cover the complete requested arc")
    return weights, {
        "units": "degrees longitude", "longitude_bounds": declared_bounds,
        "arc_direction": "eastward", "requested_width_degrees": span,
        "represented_width_degrees": represented, "cell_edge_method": method,
        "cyclic": bool(cyclic), "weights_match_input_order": True,
        "coverage_complete": True,
        "physical_scope": "Longitude cell intersections at fixed latitude; not spherical cell areas",
    }


def fixed_longitude_mean(fields, longitude=None, *, weights=None, min_fraction=.8,
                         longitude_bounds=None, cyclic=False, cell_bounds=None,
                         longitude_axis=-1, sample_axes=(0,)):
    """Return means, support fraction, fixed mask and geometry metadata.

    Every field must share one shape.  The positive-width longitude cells in
    a row are retained only if finite in every named variable and every sample
    along ``sample_axes``.  Other axes identify independent rows.  The same
    retained widths are normalized once per row for all supplied samples and
    fields.  Rejected rows return NaN even when their surviving values are zero.

    Defaults match (sample, ..., longitude).  For (ensemble,time,p,lat,lon),
    use sample_axes=(0,1) when support must be common across both dimensions.
    Means retain original axis order with longitude removed.  The support
    fraction and mask use remaining row axes in original order, with longitude
    last on the mask.  Geometry metadata identifies all these axis positions.
    """
    if not isinstance(fields, Mapping) or not fields:
        raise ValueError("At least one named field is required")
    arrays = {name: _real(value, f"field {name}") for name, value in fields.items()}
    shape = next(iter(arrays.values())).shape
    if len(shape) < 1 or any(size == 0 for size in shape) or any(array.shape != shape for array in arrays.values()):
        raise ValueError("Fields must share the same nonempty shape")
    lon_axis = _axis(longitude_axis, len(shape), "longitude_axis")
    if isinstance(sample_axes, (int, np.integer)) and not isinstance(sample_axes, (bool, np.bool_)):
        sample_axes = (sample_axes,)
    try:
        samples = tuple(_axis(axis, len(shape), "sample axis") for axis in sample_axes)
    except TypeError as error:
        raise ValueError("sample_axes must be an axis or an iterable of axes") from error
    if len(set(samples)) != len(samples) or lon_axis in samples:
        raise ValueError("Sample axes must be distinct and must not contain longitude")
    rows = tuple(axis for axis in range(len(shape)) if axis not in samples and axis != lon_axis)
    permutation = samples + rows + (lon_axis,)
    canonical = {name: np.transpose(array, permutation) for name, array in arrays.items()}
    fraction_limit = _positive(min_fraction, "min_fraction")
    if fraction_limit > 1:
        raise ValueError("min_fraction must lie in (0,1]")
    if longitude is not None and weights is not None:
        raise ValueError("Supply longitude centers or explicit weights, not both")
    if weights is None:
        if longitude is None:
            raise ValueError("Longitude centers or explicit cell-intersection weights are required")
        cell_weights, geometry = longitude_cell_weights(longitude, longitude_bounds,
                                                        cyclic=cyclic, cell_bounds=cell_bounds)
    else:
        if cell_bounds is not None:
            raise ValueError("cell_bounds cannot accompany explicit weights")
        cell_weights = _real(weights, "weights")
        supplied_width = _width_total(cell_weights)
        geometry = {"units": "degrees longitude", "cell_edge_method": "explicit user-supplied intersection widths",
                    "weights_match_input_order": True, "coverage_complete": "asserted by supplied complete-arc widths"}
        if longitude_bounds is not None:
            _, span, declared = _arc(longitude_bounds)
            if not np.isclose(supplied_width, span, rtol=0, atol=1e-8):
                raise ValueError("Explicit cell weights must sum to the requested longitude arc width")
            geometry.update(longitude_bounds=declared, requested_width_degrees=span)
    if (cell_weights.shape != (shape[lon_axis],) or not np.isfinite(cell_weights).all()
            or np.any(cell_weights < 0)):
        raise ValueError("Cell weights must be finite, nonnegative, match longitude and sum positive")
    full_width = _width_total(cell_weights)
    geometry.setdefault("requested_width_degrees", full_width)
    sample_shape = tuple(shape[axis] for axis in samples)
    row_shape = tuple(shape[axis] for axis in rows)
    fixed_mask = np.ones(row_shape + (shape[lon_axis],), dtype=bool)
    for array in canonical.values():
        finite = np.isfinite(array)
        common = finite.all(axis=tuple(range(len(samples)))) if samples else finite
        fixed_mask &= common
    fixed_mask &= cell_weights > 0
    retained = np.sum(fixed_mask * cell_weights, axis=-1)
    support_fraction = retained / full_width
    accepted = (retained > 0) & (support_fraction + 16 * np.finfo(float).eps >= fraction_limit)
    expanded_mask = fixed_mask.reshape((1,) * len(samples) + fixed_mask.shape)
    expanded_retained = retained.reshape((1,) * len(samples) + retained.shape)
    expanded_accepted = accepted.reshape((1,) * len(samples) + accepted.shape)
    kept_axes = tuple(axis for axis in range(len(shape)) if axis != lon_axis)
    canonical_mean_axes = samples + rows
    inverse_mean = tuple(canonical_mean_axes.index(axis) for axis in kept_axes)
    means = {}
    for name, array in canonical.items():
        with np.errstate(over="ignore", invalid="ignore"):
            total = np.sum(np.where(expanded_mask, array, 0.) * cell_weights, axis=-1)
            mean = np.divide(total, expanded_retained, out=np.full(sample_shape + row_shape, np.nan), where=expanded_retained > 0)
        if np.any(np.broadcast_to(expanded_accepted, mean.shape) & ~np.isfinite(mean)):
            raise ValueError("Longitude averaging overflowed on finite common support")
        mean = np.where(expanded_accepted, mean, np.nan)
        means[name] = np.transpose(mean, inverse_mean) if inverse_mean else mean
    metadata = {
        "longitude_geometry": geometry, "longitude_axis": lon_axis, "sample_axes": list(samples),
        "row_axes": list(rows), "fixed_mask_axes": list(rows) + [lon_axis], "mean_axes": list(kept_axes),
        "min_fraction": fraction_limit, "common_support_across_fields": list(arrays),
        "common_support_across_samples": True,
        "normalization": "retained longitude width, fixed once per row for all fields and samples",
        "regional_streamfunction_interpretation": "A mass streamfunction made from this finite-band or incomplete-support mean is 2pi-equivalent, not actual band-integrated transport",
    }
    return means, support_fraction, fixed_mask, metadata
