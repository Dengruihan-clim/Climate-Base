"""Strict rectilinear coordinate normalization using labelled lazy operations.

References
----------
CF 1.12 latitude, longitude, dimensional vertical coordinates and cell boundaries:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
MetPy's axis identification is a design reference:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/xarray.py

Climate-Base improvements
-------------------------
Ambiguous axes and repeated cyclic endpoints fail explicitly. Coordinates and
bounds move together, and longitude bounds shift per cell instead of wrapping
their endpoints independently. Only small coordinate/bounds arrays are loaded;
meteorological fields remain Dask-backed. Independently implemented here.
Pressure normalization uses one shared small-coordinate conversion/order kernel
with the legacy monthly reader. Only physical one-dimensional pressure levels
are accepted; model/hybrid or height axes are not reinterpreted as pressure.
"""

from datetime import date
import re

import cftime
import numpy as np
import xarray as xr

from .exceptions import CoordinateError
from .metadata import _attribute, rename_cf_references, validate_metadata
from .provenance import record_operation
from .units import conversion_plan, _apply_affine
from .validation import ensure_lazy, require_dataset

_UNITS = {
    "latitude": {"degreesnorth", "degreenorth", "degreesn", "degreen"},
    "longitude": {"degreeseast", "degreeeast", "degreese", "degreee"},
}


def _unit_key(value):
    return "".join(str(value).lower().split()).replace("_", "")


def identify_coordinate(dataset, role, explicit=None):
    """Return a unique one-dimensional angular coordinate name.

    Explicit names take precedence. Otherwise standard_name or the CF X/Y axis
    attribute identifies candidates. Names alone never imply a physical unit.
    Projected X/Y coordinates fail the angular-units check.
    """
    require_dataset(dataset)
    if role not in _UNITS:
        raise ValueError("role must be latitude or longitude")
    if explicit is not None:
        candidates = [explicit]
    else:
        axis = "Y" if role == "latitude" else "X"
        candidates = [
            name for name, variable in dataset.variables.items()
            if variable.attrs.get("standard_name") == role
            or str(variable.attrs.get("axis", "")).upper() == axis
        ]
    if len(candidates) != 1:
        raise CoordinateError(
            f"Expected one {role} coordinate; found {candidates}. Supply an explicit mapping."
        )
    name = candidates[0]
    if name not in dataset.variables:
        raise CoordinateError(f"Missing {role} coordinate {name!r}")
    variable = dataset[name]
    if variable.ndim != 1 or variable.size == 0:
        raise CoordinateError(f"{role} must be a nonempty one-dimensional coordinate")
    if _unit_key(variable.attrs.get("units", "")) not in _UNITS[role]:
        expected = "degrees_north" if role == "latitude" else "degrees_east"
        raise CoordinateError(f"{role} units must explicitly indicate {expected}")
    return name


def coordinate_values(dataset, name):
    """Inspect a small numeric coordinate, never a meteorological data field."""
    variable = dataset[name]
    if variable.dtype.kind not in "iuf":
        raise CoordinateError(f"Coordinate {name!r} must be real numeric")
    values = np.asarray(variable.data, dtype=np.float64)
    if not np.isfinite(values).all():
        raise CoordinateError(f"Coordinate {name!r} contains nonfinite values")
    return values


def _checked_bounds(dataset, coordinate, values, role):
    name = _attribute(dataset[coordinate], "bounds")
    if name is None:
        return None
    bounds = dataset[name]
    dimension = dataset[coordinate].dims[0]
    other = [dim for dim in bounds.dims if dim != dimension]
    if bounds.ndim != 2 or dimension not in bounds.dims or len(other) != 1 or bounds.sizes[other[0]] != 2:
        raise CoordinateError(f"{role} bounds must have the coordinate dimension and a size-2 vertex dimension")
    matrix = coordinate_values(dataset, name)
    if bounds.dims[0] != dimension:
        matrix = matrix.T
    lower, upper = matrix.min(axis=1), matrix.max(axis=1)
    if np.any(lower > values) or np.any(upper < values) or np.any(upper <= lower):
        raise CoordinateError(f"{role} bounds must enclose each center without a wrapped discontinuity")
    if role == "latitude" and (np.any(lower < -90) or np.any(upper > 90)):
        raise CoordinateError("Latitude bounds must lie in [-90, 90]")
    if role == "longitude" and np.any(upper - lower >= 180):
        raise CoordinateError("Longitude cells spanning 180 degrees or more are not supported")
    return name, matrix, other[0]


def _clear_coordinate_packing(variable):
    # Coordinate values have changed; retaining old packed encoding could apply
    # scale/offset twice or force an unsuitable integer representation on write.
    variable.encoding = {
        key: value for key, value in variable.encoding.items()
        if key not in {"scale_factor", "add_offset", "dtype", "_FillValue", "missing_value"}
    }


def _remove_range_metadata(variable):
    """Drop source-domain ranges after wrapping and report invalidated keys."""
    removed = []
    for key in ("valid_min", "valid_max", "valid_range", "actual_range"):
        for container_name, container in (("attrs", variable.attrs), ("encoding", variable.encoding)):
            if key in container:
                del container[key]
                removed.append(f"{container_name}.{key}")
    return removed


def normalize_coordinates(
    dataset, *, latitude=None, longitude=None,
    longitude_convention="0_360", latitude_order="ascending",
):
    """Normalize a rectilinear Dataset, leaving field computations lazy.

    Returns canonical latitude/longitude dimension coordinates. Latitude may be
    ascending or descending; longitude is ascending in the selected domain.
    All dependent variables are reindexed together. Source arrays are unchanged.
    Curvilinear grids, ambiguous mappings and duplicate cyclic points raise.
    Longitude bounds must enclose their centers in an unwrapped local interval
    narrower than 180 degrees; unsupported forms require explicit preprocessing.
    """
    require_dataset(dataset)
    validate_metadata(dataset)
    if longitude_convention not in {"0_360", "-180_180"}:
        raise ValueError("longitude_convention must be '0_360' or '-180_180'")
    if latitude_order not in {"ascending", "descending"}:
        raise ValueError("latitude_order must be ascending or descending")
    lat_name = identify_coordinate(dataset, "latitude", latitude)
    lon_name = identify_coordinate(dataset, "longitude", longitude)
    if dataset[lat_name].dims[0] == dataset[lon_name].dims[0]:
        raise CoordinateError("Rectilinear latitude and longitude require distinct dimensions")
    lat = coordinate_values(dataset, lat_name)
    lon = coordinate_values(dataset, lon_name)
    if np.any(np.abs(lat) > 90) or np.any(np.diff(np.sort(lat)) <= 0):
        raise CoordinateError("Latitude coordinates must be distinct and within [-90, 90]")
    normalized_lon = lon % 360 if longitude_convention == "0_360" else (lon + 180) % 360 - 180
    if np.any(np.diff(np.sort(normalized_lon)) <= 1e-9):
        raise CoordinateError("Longitude coordinates contain duplicate or repeated cyclic endpoints")
    _checked_bounds(dataset, lat_name, lat, "latitude")
    lon_bounds = _checked_bounds(dataset, lon_name, lon, "longitude")
    mapping = {lat_name: "latitude", lon_name: "longitude"}
    for old, new in mapping.items():
        if old != new and (new in dataset.variables or new in dataset.dims):
            raise CoordinateError(f"Cannot rename {old!r}: target {new!r} already exists")
    result = dataset.set_coords([lat_name, lon_name]).copy(deep=False)
    lon_variable = xr.DataArray(normalized_lon, dims=result[lon_name].dims, attrs=dict(result[lon_name].attrs))
    lon_variable.encoding = dict(result[lon_name].encoding)
    _clear_coordinate_packing(lon_variable)
    invalidated_ranges = {}
    longitude_changed = not np.array_equal(normalized_lon, lon)
    if longitude_changed:
        removed = _remove_range_metadata(lon_variable)
        if removed:
            invalidated_ranges["longitude"] = removed
    result = result.assign_coords({lon_name: lon_variable})
    if lon_bounds is not None:
        bounds_name, matrix, vertex_dimension = lon_bounds
        original = result[bounds_name]
        shifted = xr.DataArray(
            matrix + (normalized_lon - lon)[:, None],
            dims=(result[lon_name].dims[0], vertex_dimension), attrs=dict(original.attrs),
        ).transpose(*original.dims)
        shifted.encoding = dict(original.encoding)
        _clear_coordinate_packing(shifted)
        if longitude_changed:
            removed = _remove_range_metadata(shifted)
            if removed:
                invalidated_ranges[bounds_name] = removed
        result[bounds_name] = shifted
    result = result.rename({old: new for old, new in mapping.items() if old != new})
    dimension_mapping = {
        result[name].dims[0]: name for name in ("latitude", "longitude")
        if result[name].dims[0] != name
    }
    if dimension_mapping:
        result = result.swap_dims(dimension_mapping)
    result = rename_cf_references(result, {**mapping, **dimension_mapping})
    result["latitude"].attrs.update(standard_name="latitude", axis="Y")
    result["longitude"].attrs.update(standard_name="longitude", axis="X")
    result = result.isel(
        latitude=np.argsort(lat)[::1 if latitude_order == "ascending" else -1],
        longitude=np.argsort(normalized_lon),
    )
    result = ensure_lazy(result)
    result.set_close(dataset.close)
    return record_operation(result, "normalize_coordinates", {
        "latitude": lat_name, "longitude": lon_name,
        "longitude_convention": longitude_convention, "latitude_order": latitude_order,
        "removed_range_attributes": invalidated_ranges,
    })


def _pressure_factor(source_units, target_units):
    if not isinstance(target_units, str) or target_units not in {"Pa", "hPa"}:
        raise ValueError("target_units must be 'Pa' or 'hPa'")
    if not isinstance(source_units, str) or not source_units.strip():
        raise CoordinateError("Pressure coordinate units must explicitly indicate Pa or hPa")
    try:
        factor, shift, _ = conversion_plan(source_units, target_units)
    except ValueError as error:
        raise CoordinateError("Pressure coordinate units must be Pa or hPa (including millibar aliases)") from error
    return factor, shift


def pressure_order(values, source_units, target_units="hPa", order="ascending"):
    """Normalize a small physical pressure vector and return values/indexer.

    This pure coordinate kernel is shared by the general Dataset interface and
    the existing monthly reader. Its caller owns dimensionality/CF selection;
    only already-materialized coordinate values belong here, never a field.
    The existing explicit pressure-unit table is reused rather than duplicated.
    """
    if not isinstance(order, str) or order not in {"ascending", "descending"}:
        raise ValueError("order must be ascending or descending")
    raw = np.asanyarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise CoordinateError("Pressure must be a nonempty one-dimensional real numeric coordinate")
    numeric = np.ma.asarray(raw, dtype=np.float64).filled(np.nan)
    if not np.isfinite(numeric).all():
        raise CoordinateError("Pressure coordinate contains nonfinite values")
    factor, shift = _pressure_factor(source_units, target_units)
    normalized = _apply_affine(numeric, factor, shift)
    if np.any(normalized <= 0) or np.any(np.diff(np.sort(normalized)) <= 0):
        raise CoordinateError("Pressure coordinates must be distinct and positive")
    indexer = np.argsort(normalized)
    if order == "descending":
        indexer = indexer[::-1]
    return normalized[indexer], indexer


def _pressure_units(variable, fallback=None):
    source = _attribute(variable, "units")
    source = fallback if source is None else source
    _pressure_factor(source, "Pa")
    return source


def _check_pressure_semantics(variable):
    if _attribute(variable, "formula_terms") is not None:
        raise CoordinateError("Hybrid/model pressure formulas require an explicit physical-pressure calculation")
    standard = _attribute(variable, "standard_name")
    if standard is not None and standard.strip() != "air_pressure":
        raise CoordinateError(f"Unsupported pressure standard_name {standard!r}; model/hybrid/height axes are not pressure")
    axis = _attribute(variable, "axis")
    if axis is not None and axis.strip().upper() != "Z":
        raise CoordinateError("Pressure coordinate axis must be Z")
    positive = _attribute(variable, "positive")
    if positive is not None and positive.strip().lower() != "down":
        raise CoordinateError("Pressure positive direction must be down")


def _nonvertical_dimension(variable):
    """Recognize dimensional time/angular evidence even without axis labels."""
    units = variable.attrs.get("units", variable.encoding.get("units", ""))
    if (variable.attrs.get("standard_name") in {"time", "latitude", "longitude"}
            or str(variable.attrs.get("axis", "")).upper() in {"T", "X", "Y"}
            or variable.dtype.kind == "M" or "since" in str(units).lower()
            or _unit_key(units) in _UNITS["latitude"] | _UNITS["longitude"]):
        return True
    if variable.dtype.kind == "O":
        # A dimension coordinate is a small index, not a meteorological field.
        return any(isinstance(value, (date, cftime.datetime, np.datetime64))
                   for value in np.asarray(variable.data).flat)
    return False


def _identify_pressure(dataset, explicit):
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise CoordinateError("pressure must name an existing coordinate")
        candidates = [explicit]
    else:
        bounds_names = {_attribute(variable, "bounds") for variable in dataset.variables.values()}
        candidates = []
        for name, variable in dataset.variables.items():
            if name in bounds_names:
                continue
            if variable.attrs.get("standard_name") == "air_pressure":
                candidates.append(name)
            elif ((name in dataset.coords and variable.ndim == 1)
                  or str(variable.attrs.get("axis", "")).upper() == "Z"):
                try:
                    _pressure_units(variable)
                except CoordinateError:
                    continue
                candidates.append(name)
    if len(candidates) != 1:
        raise CoordinateError(f"Expected one pressure coordinate; found {candidates}. Supply pressure explicitly.")
    name = candidates[0]
    if name not in dataset.variables:
        raise CoordinateError(f"Missing pressure coordinate {name!r}")
    variable = dataset[name]
    if variable.ndim != 1 or not variable.size:
        raise CoordinateError("Pressure must be a nonempty one-dimensional coordinate")
    if variable.size > 1_000_000:
        raise CoordinateError("Pressure coordinate exceeds the bounded comparison size")
    _check_pressure_semantics(variable)
    _pressure_units(variable)
    dimension = variable.dims[0]
    if dimension != name and dimension in dataset.coords:
        dimension_variable = dataset[dimension]
        if _nonvertical_dimension(dimension_variable):
            raise CoordinateError("Pressure coordinate cannot replace a time or horizontal dimension")
    return name


def _pressure_values(dataset, name):
    variable = dataset[name]
    if "scale_factor" in variable.attrs or "add_offset" in variable.attrs:
        raise CoordinateError("CF-decode pressure coordinates and bounds before normalization")
    values = coordinate_values(dataset, name)
    for key in ("_FillValue", "missing_value"):
        if key in variable.attrs and np.isin(values, np.asarray(variable.attrs[key])).any():
            raise CoordinateError(f"Pressure coordinate/bounds {name!r} contains declared missing values")
    return values


def _pressure_bounds(dataset, coordinate, source_units, target_units, normalized_centers):
    name = _attribute(dataset[coordinate], "bounds")
    if name is None:
        return None
    bounds = dataset[name]
    dimension = dataset[coordinate].dims[0]
    other = [dim for dim in bounds.dims if dim != dimension]
    if bounds.ndim != 2 or dimension not in bounds.dims or len(other) != 1 or bounds.sizes[other[0]] != 2:
        raise CoordinateError("Pressure bounds require the pressure dimension and a size-2 vertex dimension")
    _check_pressure_semantics(bounds)
    units = _pressure_units(bounds, fallback=source_units)
    factor, shift = _pressure_factor(units, target_units)
    matrix = _apply_affine(_pressure_values(dataset, name), factor, shift)
    if bounds.dims[0] != dimension:
        matrix = matrix.T
    lower, upper = matrix.min(axis=1), matrix.max(axis=1)
    if (np.any(matrix <= 0) or np.any(upper <= lower)
            or np.any(lower > normalized_centers) or np.any(upper < normalized_centers)):
        raise CoordinateError("Pressure bounds must be positive, have nonzero width and enclose each center")
    return name, matrix, other[0], units, factor


def _converted_pressure_variable(original, values, target_units, factor, *, dims=None):
    """Create transformed coordinate metadata without stale packing/ranges."""
    result = xr.DataArray(values, dims=original.dims if dims is None else dims, attrs=dict(original.attrs))
    result.encoding = dict(original.encoding)
    invalidate = factor != 1 or any(key in original.encoding for key in ("scale_factor", "add_offset"))
    removed = _remove_range_metadata(result) if invalidate else []
    if invalidate:
        for key in ("minimum_value", "maximum_value"):
            for label, container in (("attrs", result.attrs), ("encoding", result.encoding)):
                if key in container:
                    container.pop(key)
                    removed.append(f"{label}.{key}")
    _clear_coordinate_packing(result)
    for key in ("units", "source", "original_shape", "preferred_chunks", "chunksizes", "contiguous"):
        result.encoding.pop(key, None)
    for key in ("_FillValue", "missing_value"):
        result.attrs.pop(key, None)
    result.attrs["units"] = target_units
    return result, removed


def normalize_pressure(dataset, *, pressure=None, target_units="hPa", order="ascending"):
    """Normalize fixed one-dimensional physical pressure levels and bounds.

    Explicit names still require pressure units. Automatic recognition uses CF
    ``air_pressure``/Z evidence or a coordinate's explicit pressure units, never
    its name alone. Height/model/hybrid coordinates, formulas, multiple possible
    axes, nonpositive/duplicate levels and incompatible metadata are rejected.
    Only Pa/hPa targets are supported; source millibar aliases use the existing
    conversion table. Pressure bounds inherit absent units from their coordinate
    or are converted from explicitly declared compatible units. Bounds must have
    two positive endpoints enclosing the center; zero-pressure top interfaces
    require a separate, explicitly defined interface-coordinate workflow.

    Returns canonical ``pressure`` dimension/coordinate in requested order. All
    fields are reindexed lazily, CF links follow renamed axes, stale coordinate
    packing/ranges are removed and audited. Small coordinate/bounds values are
    inspected; no field computation, interpolation or vertical regridding occurs.
    The result retains the source close callback; the input is unchanged.
    """
    require_dataset(dataset)
    validate_metadata(dataset)
    if not isinstance(target_units, str) or target_units not in {"Pa", "hPa"}:
        raise ValueError("target_units must be 'Pa' or 'hPa'")
    if not isinstance(order, str) or order not in {"ascending", "descending"}:
        raise ValueError("order must be ascending or descending")
    name = _identify_pressure(dataset, pressure)
    dimension = dataset[name].dims[0]
    if ((name != "pressure" and "pressure" in dataset.variables)
            or (dimension != "pressure" and "pressure" in dataset.dims)):
        raise CoordinateError("Cannot normalize pressure: canonical target 'pressure' already exists")
    units = _pressure_units(dataset[name])
    values = _pressure_values(dataset, name)
    sorted_values, indexer = pressure_order(values, units, target_units, order)
    factor, _ = _pressure_factor(units, target_units)
    normalized = np.empty_like(sorted_values)
    normalized[indexer] = sorted_values
    bounds = _pressure_bounds(dataset, name, units, target_units, normalized)
    result = dataset.set_coords(name).copy(deep=False)
    # Old xarray versions can retain an auxiliary coordinate's stale PandasIndex
    # when its values change before rename/swap_dims. Rebuild that small index
    # from transformed pressure values rather than allowing it to restore Pa
    # values under an hPa label.
    if name in result.xindexes:
        result = result.drop_indexes(name)
    coordinate, removed = _converted_pressure_variable(result[name], normalized, target_units, factor)
    result = result.assign_coords({name: coordinate})
    invalidated = {"pressure": removed} if removed else {}
    bounds_units = None
    if bounds is not None:
        bounds_name, matrix, vertex_dimension, bounds_units, bounds_factor = bounds
        converted, removed = _converted_pressure_variable(
            result[bounds_name], matrix, target_units, bounds_factor, dims=(dimension, vertex_dimension),
        )
        result[bounds_name] = converted.transpose(*result[bounds_name].dims)
        if removed:
            invalidated[bounds_name] = removed
    mapping = {name: "pressure"}
    if name != "pressure":
        result = result.rename_vars(mapping) if dimension == "pressure" else result.rename(mapping)
    if result.pressure.dims[0] != "pressure":
        result = result.swap_dims({result.pressure.dims[0]: "pressure"})
    if "pressure" not in result.xindexes:
        result = result.set_xindex("pressure")
    # swap_dims changes the dimension but keeps any old dimension-coordinate
    # variable. CF coordinates/ancillary/formula references must still point to
    # that retained variable; only an actually renamed variable is rewritten.
    result = rename_cf_references(result, mapping)
    if dimension != "pressure":
        for variable in result.variables.values():
            for container in (variable.attrs, variable.encoding):
                methods = container.get("cell_methods")
                if isinstance(methods, str):
                    container["cell_methods"] = re.sub(
                        r"(?<!\S)([^\s:]+)(?=\s*:)",
                        lambda match: "pressure" if match[0] == dimension else match[0], methods,
                    )
    result.pressure.attrs.update(units=target_units, standard_name="air_pressure", axis="Z", positive="down")
    result = result.isel(pressure=indexer)
    # The shared kernel already checks these sorted physical values; assigning
    # them is unnecessary and would discard encoding/CF references restored above.
    assert np.array_equal(result.pressure.values, sorted_values)
    validate_metadata(result)
    result = ensure_lazy(result)
    result.set_close(dataset.close)
    return record_operation(result, "normalize_pressure", {
        "pressure": name, "source_units": units, "target_units": target_units, "order": order,
        "bounds_source_units": bounds_units, "removed_range_attributes": invalidated,
    })
