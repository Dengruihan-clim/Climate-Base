"""Strict rectilinear coordinate normalization using labelled lazy operations.

References
----------
CF 1.12 latitude, longitude and cell-boundary conventions:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
MetPy's axis identification is a design reference:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/xarray.py

Climate-Base improvements
-------------------------
Ambiguous axes and repeated cyclic endpoints fail explicitly. Coordinates and
bounds move together, and longitude bounds shift per cell instead of wrapping
their endpoints independently. Only small coordinate/bounds arrays are loaded;
meteorological fields remain Dask-backed. Independently implemented here.
"""

import numpy as np
import xarray as xr

from .exceptions import CoordinateError
from .metadata import _attribute, rename_cf_references, validate_metadata
from .provenance import record_operation
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
