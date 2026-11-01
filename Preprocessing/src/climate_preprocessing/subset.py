"""Explicit spatial selection on rectilinear grids, without interpolation.

Reference: xarray's public positional selection API:
https://docs.xarray.dev/en/stable/generated/xarray.Dataset.isel.html

Climate-Base improvement: longitude selection uses a directed eastward interval
and handles the seam in either longitude convention. No field values are loaded;
coordinate bounds and related variables are sliced along with the data.
"""

import numpy as np

from .coordinates import coordinate_values, identify_coordinate
from .exceptions import CoordinateError
from .provenance import record_operation
from .validation import ensure_lazy, require_dataset


def _pair(value, name):
    try:
        pair = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain two finite numbers") from error
    if len(pair) != 2 or not all(np.isfinite(pair)):
        raise ValueError(f"{name} must contain two finite numbers")
    return pair


def subset_region(
    dataset, *, latitude_bounds=None, longitude_bounds=None,
    latitude="latitude", longitude="longitude",
):
    """Select inclusive coordinate centers in a geographic region.

    Latitude bounds are (south, north). Longitude bounds are (west, east),
    interpreted eastward modulo 360; for example (170, -170) crosses the seam.
    A 360-degree span selects all longitudes; equal endpoints select one
    meridian. The input coordinate order is preserved and no interpolation or
    area clipping is performed. An empty selection raises ValueError.
    """
    require_dataset(dataset)
    lat_name = identify_coordinate(dataset, "latitude", latitude)
    lon_name = identify_coordinate(dataset, "longitude", longitude)
    lat = coordinate_values(dataset, lat_name)
    lon = coordinate_values(dataset, lon_name)
    if dataset[lat_name].dims[0] == dataset[lon_name].dims[0]:
        raise CoordinateError("Spatial selection requires rectilinear coordinates")
    if np.any(np.abs(lat) > 90) or np.any(np.diff(np.sort(lat)) <= 0):
        raise CoordinateError("Latitude coordinates must be distinct and within [-90, 90]")
    if np.any(np.diff(np.sort(lon % 360)) <= 1e-9):
        raise CoordinateError("Longitude coordinates contain duplicate cyclic points")
    indexers = {}
    parameters = {}
    if latitude_bounds is not None:
        south, north = _pair(latitude_bounds, "latitude_bounds")
        if not -90 <= south <= north <= 90:
            raise ValueError("latitude_bounds must satisfy -90 <= south <= north <= 90")
        indexers[dataset[lat_name].dims[0]] = np.flatnonzero((lat >= south) & (lat <= north))
        parameters["latitude_bounds"] = [south, north]
    if longitude_bounds is not None:
        west, east = _pair(longitude_bounds, "longitude_bounds")
        raw_span = east - west
        if abs(raw_span) > 360:
            raise ValueError("longitude_bounds cannot span more than 360 degrees")
        full_circle = abs(raw_span) == 360
        distance = (lon - west) % 360
        span = raw_span % 360
        selected = np.ones(lon.size, dtype=bool) if full_circle else distance <= span + 1e-9
        indexers[dataset[lon_name].dims[0]] = np.flatnonzero(selected)
        parameters["longitude_bounds"] = [west, east]
    if any(len(index) == 0 for index in indexers.values()):
        raise CoordinateError("Requested region contains no coordinate centers")
    result = ensure_lazy(dataset.isel(indexers))
    cropped_dimensions = {
        dimension for dimension, index in indexers.items()
        if len(index) != dataset.sizes[dimension]
    }
    removed_ranges = {}
    for name, variable in result.variables.items():
        if cropped_dimensions.intersection(variable.dims):
            removed = []
            for container_name, container in (("attrs", variable.attrs), ("encoding", variable.encoding)):
                if "actual_range" in container:
                    # Computing the new field range would defeat lazy selection.
                    # Physical valid_range limits remain valid and are retained.
                    del container["actual_range"]
                    removed.append(container_name)
            if removed:
                removed_ranges[name] = removed
    if removed_ranges:
        parameters["removed_actual_range"] = removed_ranges
    result.set_close(dataset.close)
    return record_operation(result, "subset_region", parameters)
