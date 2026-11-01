"""Compose the first strict xarray/Dask preprocessing workflow.

References
----------
xarray I/O and labeled transformations:
https://docs.xarray.dev/en/stable/user-guide/io.html

Climate-Base improvements
-------------------------
Independent orchestration with explicit coordinate/unit policies, no statistical
aggregation, strict option validation and a source lifetime tied to the result.
All field transformations stay Dask-backed; the writer is the compute boundary.
"""

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path

from .coordinates import normalize_coordinates, normalize_pressure
from .io import open_dataset, open_mfdataset
from .masking import mask_invalid
from .metadata import validate_metadata
from .subset import subset_region
from .time import subset_time, validate_time
from .units import convert_dataarray_units
from .provenance import record_operation


def prepare_dataset(
    inputs, *, chunks="auto", coordinates=None, time="time",
    longitude_convention="0_360", latitude_order="ascending", region=None, units=None,
    metadata_policy="strict", pressure=None, expected_time=None, time_range=None,
):
    """Return a lazy normalized Dataset for local rectilinear NetCDF files.

    Parameters
    ----------
    inputs : path-like or sequence of path-like
        Explicit files, not filename-based dates. Multiple files must satisfy
        the strict time/metadata/grid contract of ``open_mfdataset``.
    chunks : 'auto' or mapping
        Dask chunks in the input dimension names. ``None`` is rejected.
    coordinates : mapping, optional
        Explicit ``latitude`` / ``longitude`` source coordinate names. When
        absent, CF attributes identify them; ambiguity is an error.
    time : str
        Required decoded time coordinate name. Original calendar is retained.
    longitude_convention, latitude_order : str
        Explicit coordinate policies; no interpolation is performed.
    region : mapping, optional
        ``latitude_bounds`` and/or ``longitude_bounds`` in degrees. Longitude
        bounds may cross the dateline, using the subset function's convention.
    units : mapping, optional
        Field names to target units supported by ``convert_dataarray_units``.
    metadata_policy : {'strict', 'preserve_sources'}
        Multi-file attribute handling. The explicit ``preserve_sources`` policy
        records file-specific history/dates and descriptive ranges per source;
        scientific metadata, grids and calendars must still agree.
    pressure : mapping, optional
        Opt-in ``normalize_pressure`` options: ``pressure`` source name,
        ``target_units`` (Pa/hPa), and ``order``. An empty mapping requests CF
        identification. Surface data need no pressure coordinate by default.
    expected_time : sequence, optional
        Exact decoded dates or complete ISO dates in the source calendar. The
        entire input time index must match, before any time selection. No
        frequency, missing dates, or calendar conversion is inferred.
    time_range : mapping, optional
        Inclusive ``start`` / ``end`` instants for ``subset_time``. Complete ISO
        dates mean midnight; a bare year or month is not accepted.

    Returns
    -------
    xarray.Dataset
        Dask-backed fields. Use as a context manager or call ``close`` after
        computing/writing; do not close before executing the lazy graph.
    """
    if isinstance(inputs, (str, Path)):
        inputs = [inputs]
    inputs = list(inputs)
    if not inputs:
        raise ValueError("At least one input is required")
    if not isinstance(time, str) or not time:
        raise ValueError("time must be a coordinate name")
    if not isinstance(metadata_policy, str) or metadata_policy not in ("strict", "preserve_sources"):
        raise ValueError("metadata_policy must be 'strict' or 'preserve_sources'")
    for name, value, allowed in (
        ("coordinates", coordinates, {"latitude", "longitude"}),
        ("region", region, {"latitude_bounds", "longitude_bounds"}),
        ("pressure", pressure, {"pressure", "target_units", "order"}),
        ("time_range", time_range, {"start", "end"}),
    ):
        if value is not None and (not isinstance(value, Mapping) or set(value) - allowed):
            raise ValueError(f"Invalid {name}; supported keys are {sorted(allowed)}")
    coordinates = dict(coordinates or {})
    if any(not isinstance(value, str) or not value for value in coordinates.values()):
        raise ValueError("Coordinate mappings must contain nonempty names")
    if units is not None and (
        not isinstance(units, Mapping)
        or any(not isinstance(k, str) or not isinstance(v, str) or not v.strip() for k, v in units.items())
    ):
        raise ValueError("units must map variable names to nonempty target units")
    source = (
        open_dataset(inputs[0], chunks=chunks, time=time)
        if len(inputs) == 1 else open_mfdataset(
            inputs, time=time, chunks=chunks, metadata_policy=metadata_policy,
        )
    )
    try:
        validate_time(source, time=time, expected=expected_time)
        expected_summary = None
        if expected_time is not None:
            # Only the small, already validated time coordinate is inspected.
            # An ordered digest distinguishes internal gaps, not just endpoints.
            labels = [str(value) for value in source[time].to_numpy()]
            expected_summary = {
                "count": len(labels), "first": labels[0], "last": labels[-1],
                "calendar": (source[time].encoding.get("calendar") or source[time].attrs.get("calendar")
                             or source[time].dt.calendar),
                "sha256": hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest(),
                "digest_format": "UTF-8 JSON array of source time strings; compact separators",
            }
        validate_metadata(source)
        selected = subset_time(source, time=time, **time_range) if time_range is not None else source
        result = normalize_coordinates(
            selected, **coordinates, longitude_convention=longitude_convention,
            latitude_order=latitude_order,
        )
        if pressure is not None:
            result = normalize_pressure(result, **pressure)
        result = subset_region(result, **(region or {}))
        for name, target in (units or {}).items():
            if name not in result.data_vars:
                raise ValueError(f"Unit conversion requires a data variable: {name!r}")
            result[name] = convert_dataarray_units(result[name], target)
        result = mask_invalid(result)
        validate_metadata(result)
        result = record_operation(result, "prepare_dataset", {
            "coordinates": coordinates, "time": time, "chunks": chunks,
            "longitude_convention": longitude_convention, "latitude_order": latitude_order,
            "region": dict(region or {}), "units": dict(units or {}),
            "metadata_policy": metadata_policy,
            "pressure": dict(pressure) if pressure is not None else None,
            "expected_time": expected_summary,
            "time_range_selected": time_range is not None,
        })
        result.set_close(source.close)
        return result
    except BaseException:
        source.close()
        raise
