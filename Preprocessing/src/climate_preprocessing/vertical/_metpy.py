"""Lazy pressure interpolation using the public MetPy numerical kernels.

References
----------
GeoCAT-comp v2026.04.0 delegates vertical remapping to MetPy's public linear
and log-pressure interpolation functions. GeoCAT supplies the organizational
reference, while this optional adapter uses the same upstream kernels:
https://github.com/NCAR/geocat-comp/blob/v2026.04.0/geocat/comp/interpolation.py
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/interpolate/one_dimension.py

Climate-Base improvements
-------------------------
Independently written validation and eligibility gates enforce strictly ordered
pressure columns, original adjacent brackets, exact source hits, missing-value
and below-ground handling. They do not implement interpolation weights.
MetPy performs the actual linear/log interpolation inside scheduled NumPy
column kernels. Fields remain Dask-backed, only the vertical core dimension is
combined, and requested level order and nonvertical chunks are preserved.
Inputs and targets are Pa; physical field units and calendars are unchanged.
No extrapolation, sorting of nonmonotonic columns, or gap filling is performed.
"""

from importlib import import_module, metadata

import dask.array as da
import numpy as np
import xarray as xr


def _load_backend():
    try:
        package = import_module("metpy")
        interpolation = import_module("metpy.interpolate")
    except (ImportError, OSError) as error:
        raise ImportError(
            "Pressure interpolation requires the optional MetPy backend. Install "
            "climate-preprocessing[vertical] in an isolated environment; the core "
            "package does not require MetPy or GeoCAT."
        ) from error
    return package, interpolation


def backend_versions():
    """Identify the imported numerical backend without guessing availability."""
    package, _ = _load_backend()
    versions = {"metpy": str(package.__version__)}
    for name in ("numpy", "xarray", "dask", "scipy", "pint"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "distribution-not-installed"
    return versions


def _column(values, pressure, surface, *, targets, interpolator, use_surface):
    """Gate one original column, then delegate numerical interpolation upstream."""
    values = np.asarray(values, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    result = np.full(targets.shape, np.nan, dtype=np.float64)
    if use_surface:
        surface = float(surface)
        if np.isnan(surface):
            return result
        if not np.isfinite(surface) or surface <= 0:
            raise ValueError("Surface pressure must be positive finite Pa or NaN for a missing column")
    if pressure.ndim != 1 or values.ndim != 1 or pressure.shape != values.shape or pressure.size < 2:
        raise ValueError("Pressure interpolation requires at least two matching vertical centers per column")
    if np.any(~np.isfinite(pressure)) or np.any(pressure <= 0):
        raise ValueError("Pressure must be positive finite Pa in every nonmissing column")
    differences = np.diff(pressure)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError("Pressure columns must be strictly monotonic without duplicate levels")
    # Reversing a validated descending column is only an index view for bracket
    # eligibility. MetPy receives these ordered values and computes all weights.
    if differences[0] < 0:
        pressure, values = pressure[::-1], values[::-1]
    valid = np.isfinite(values)
    allowed = (targets >= pressure[0]) & (targets <= pressure[-1])
    if use_surface:
        valid &= pressure <= surface
        allowed &= targets <= surface
    positions = np.searchsorted(pressure, targets)
    upper = np.minimum(positions, pressure.size - 1)
    exact = allowed & (positions < pressure.size) & (pressure[upper] == targets)
    exact_valid = exact & valid[upper]
    result[exact_valid] = values[upper[exact_valid]]
    # A gap is never removed from xp: only a pair that was adjacent in the
    # original profile and has two valid values may contribute. Exact valid
    # matches above survive even when either neighboring level is missing.
    lower = np.maximum(positions - 1, 0)
    interior = allowed & ~exact & (positions > 0) & (positions < pressure.size)
    eligible = interior & valid[lower] & valid[upper]
    if np.any(eligible):
        safe_values = np.where(valid, values, np.nan)
        result[eligible] = np.asarray(
            interpolator(targets[eligible], pressure, safe_values, axis=0, fill_value=np.nan),
            dtype=np.float64,
        )
    return result


def interpolate_field(field, pressure, target_pa, *, vertical_dim, method, surface_pressure=None):
    """Return a lazy field on requested pressure centers, preserving axis order.

    ``pressure`` is a one-dimensional fixed Pa axis or labelled hybrid pressure
    whose dimensions are a subset of the field dimensions. The caller validates
    physical coordinate metadata; this boundary still uses exact index alignment.
    Hybrid pressure monotonicity and source/target below-ground decisions run
    when each Dask column executes. Missing surface pressure gives an all-NaN
    column; other invalid surface pressure and nonpositive/nonfinite pressure in
    a valid column raise. Source NaN/inf values invalidate only adjacent brackets.

    Existing nonvertical field chunk partitions are retained. The vertical
    dimension is explicitly rechunked into one core block; pressure/surface
    arguments use identical chunks on their shared nonvertical dimensions.
    ``allow_rechunk=False`` prevents gufunc from silently repartitioning the
    field's nonvertical dimensions. No field values are read during construction.
    """
    if not isinstance(field, xr.DataArray) or not isinstance(pressure, xr.DataArray):
        raise TypeError("field and pressure must be labelled xarray.DataArray objects")
    if method not in {"linear", "log"}:
        raise ValueError("Pressure interpolation method must be linear or log")
    if field.dtype.kind != "f":
        raise ValueError("Pressure interpolation requires floating-point scalar fields")
    if vertical_dim not in field.dims or vertical_dim not in pressure.dims:
        raise ValueError("The declared vertical dimension must belong to the field and pressure")
    if set(pressure.dims) - set(field.dims):
        raise ValueError("Pressure dimensions must be a subset of the field dimensions")
    if vertical_dim != "pressure" and "pressure" in field.dims:
        raise ValueError("The output pressure dimension conflicts with another field dimension")
    targets = np.asarray(target_pa, dtype=np.float64)
    if targets.ndim != 1 or targets.size == 0 or np.any(~np.isfinite(targets)) or np.any(targets <= 0):
        raise ValueError("Target pressure must be a nonempty one-dimensional positive finite Pa array")
    if targets.size > 1 and not (np.all(np.diff(targets) > 0) or np.all(np.diff(targets) < 0)):
        raise ValueError("Target pressure must be strictly ascending or descending without duplicates")
    targets = targets.copy()
    if surface_pressure is not None:
        if not isinstance(surface_pressure, xr.DataArray):
            raise TypeError("Surface pressure must be a labelled xarray.DataArray")
        if vertical_dim in surface_pressure.dims or set(surface_pressure.dims) - set(field.dims):
            raise ValueError("Surface pressure dimensions must be nonvertical field dimensions")
        field, pressure, surface = xr.align(field, pressure, surface_pressure, join="exact", copy=False)
    else:
        field, pressure = xr.align(field, pressure, join="exact", copy=False)
        surface = xr.DataArray(np.nan)
    _, interpolation = _load_backend()
    interpolator = interpolation.interpolate_1d if method == "linear" else interpolation.log_interpolate_1d
    source = field if isinstance(field.data, da.Array) else field.chunk("auto")
    source = source.chunk({vertical_dim: -1})

    def matching_chunks(array):
        chunks = {dimension: (-1 if dimension == vertical_dim else source.chunksizes[dimension])
                  for dimension in array.dims}
        # Strip ancillary coordinates on geometry arguments: they must not leak
        # into the output or compete with field coordinates in apply_ufunc.
        return xr.DataArray(array.data, dims=array.dims,
                            coords={dim: array[dim] for dim in array.dims if dim in array.coords}).chunk(chunks)

    pressure = matching_chunks(pressure)
    surface = matching_chunks(surface)
    output_dim = "_climate_pressure_target"
    while output_dim in field.dims or output_dim in field.coords:
        output_dim += "_"
    result = xr.apply_ufunc(
        _column, source, pressure, surface,
        input_core_dims=[[vertical_dim], [vertical_dim], []], output_core_dims=[[output_dim]],
        kwargs={"targets": targets, "interpolator": interpolator, "use_surface": surface_pressure is not None},
        vectorize=True, dask="parallelized", output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {output_dim: targets.size}, "allow_rechunk": False},
        keep_attrs=False, join="exact",
    )
    # Rebuild rather than retain source vertical-dependent auxiliary coordinates.
    coords = {name: coordinate for name, coordinate in field.coords.items()
              if vertical_dim not in coordinate.dims and name != "pressure"}
    coords["pressure"] = ("pressure", targets,
                          {"standard_name": "air_pressure", "units": "Pa", "positive": "down", "axis": "Z"})
    dims = tuple("pressure" if dim == output_dim else dim for dim in result.dims)
    result = xr.DataArray(result.data, dims=dims, coords=coords, name=field.name, attrs=dict(field.attrs))
    result.encoding = {}
    order = tuple("pressure" if dim == vertical_dim else dim for dim in field.dims)
    return result.transpose(*order)
