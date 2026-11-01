"""Strict, lazy concatenation of local NetCDF records along decoded time.

References
----------
Xarray combining and multi-file I/O APIs:
https://docs.xarray.dev/en/stable/user-guide/combining.html
https://docs.xarray.dev/en/stable/user-guide/io.html#reading-multi-file-datasets
CF 1.12 descriptive attributes and actual_range semantics:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html

Climate-Base improvements
-------------------------
No permissive outer alignment or compat='override': calendars, schemas,
non-concatenated coordinates and metadata must agree. Data fields are not read
to resolve conflicts; repeated static fields are rejected rather than guessed
equal. An explicit preserve_sources policy reconciles only an enumerated set of
file descriptions and descriptive ranges, keeping their original values in
per-source provenance. It never relaxes scientific metadata or scans a field
to recompute a range. Coordinates alone may be inspected eagerly. Semantic
variable attributes and decoded CF links survive concatenation in both policies.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import xarray as xr

from ..metadata import validate_metadata
from ..provenance import PROVENANCE_ATTRIBUTE, input_record, record_operation
from ..time import validate_time
from .netcdf import open_dataset

_CF_LINKS = ("bounds", "climatology", "coordinates", "grid_mapping", "cell_measures",
             "formula_terms", "ancillary_variables")
_SOURCE_GLOBAL_ATTRS = frozenset({"history", "CONVERSION_DATE", "date_created",
                                  "date_modified", "creation_date"})
# These describe the values in one source; valid_* defines admissible values
# and is deliberately excluded. No extrema are calculated during combination.
_SOURCE_RANGE_ATTRS = frozenset({"actual_range", "minimum_value", "maximum_value"})


def _equal(left, right):
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _different_attrs(left, right, excluded=()):
    names = (set(left) | set(right)) - set(excluded)
    return sorted(key for key in names
                  if key not in left or key not in right or not _equal(left[key], right[key]))


def _json_attribute(value):
    """Normalize small NetCDF attribute values without emitting invalid JSON.

    Nonfinite floating values and byte strings use explicit tagged records,
    avoiding ambiguous strings or the non-standard JSON NaN/Infinity literals.
    No data variables or coordinates are read by this helper.
    """
    if isinstance(value, np.ndarray):
        return _json_attribute(value.tolist())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if np.isfinite(number):
            return number
        label = "NaN" if np.isnan(number) else ("Infinity" if number > 0 else "-Infinity")
        return {"nonfinite": label}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.generic):
        return _json_attribute(value.item())
    if isinstance(value, (list, tuple)):
        return [_json_attribute(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise TypeError(f"Unsupported source metadata attribute type: {type(value).__name__}")


def _source_metadata(dataset):
    return {
        "global_attributes": {key: _json_attribute(value) for key, value in dataset.attrs.items()
                              if key in _SOURCE_GLOBAL_ATTRS},
        "variable_attributes": {
            name: {key: _json_attribute(value) for key, value in variable.attrs.items()
                   if key in _SOURCE_RANGE_ATTRS}
            for name, variable in dataset.variables.items()
            if _SOURCE_RANGE_ATTRS.intersection(variable.attrs)
        },
    }


def _calendar(coordinate):
    calendar = coordinate.encoding.get("calendar", coordinate.attrs.get("calendar", "standard"))
    return {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}.get(calendar, calendar)


def _metadata(variable):
    attributes = dict(variable.attrs)
    for key in _CF_LINKS:
        if key in variable.encoding:
            attributes[key] = variable.encoding[key]
    return attributes


def _check_compatible(reference, current, dimension, time, metadata_policy):
    if set(reference.variables) != set(current.variables):
        raise ValueError("Multi-file variable/coordinate names differ")
    if set(reference.coords) != set(current.coords):
        raise ValueError("Multi-file coordinate roles differ")
    if _calendar(reference[time]) != _calendar(current[time]):
        raise ValueError("Multi-file calendars differ; no implicit calendar conversion is allowed")
    left_attrs = {key: value for key, value in reference.attrs.items() if key != PROVENANCE_ATTRIBUTE}
    right_attrs = {key: value for key, value in current.attrs.items() if key != PROVENANCE_ATTRIBUTE}
    reconcile = metadata_policy == "preserve_sources"
    conflicts = _different_attrs(left_attrs, right_attrs, _SOURCE_GLOBAL_ATTRS if reconcile else ())
    if conflicts:
        raise ValueError("Multi-file global attributes differ: " + ", ".join(conflicts)
                         + "; resolve the conflict explicitly")
    for name in reference.variables:
        left, right = reference[name], current[name]
        if left.dims != right.dims or left.dtype != right.dtype:
            raise ValueError(f"Multi-file dimensions or decoded dtype differ for {name!r}")
        if any(left.sizes[dim] != right.sizes[dim] for dim in left.dims if dim != dimension):
            raise ValueError(f"Multi-file non-time sizes differ for {name!r}")
        conflicts = _different_attrs(_metadata(left), _metadata(right), _SOURCE_RANGE_ATTRS if reconcile else ())
        if conflicts:
            raise ValueError(f"Multi-file variable metadata differ for {name!r}: " + ", ".join(conflicts))
        if dimension not in left.dims:
            if name in reference.data_vars:
                raise ValueError(
                    f"Static data field {name!r} occurs in multiple files; read it separately. "
                    "Comparing its values would require eager field computation."
                )
            if left.size > 1_000_000:
                raise ValueError(f"Coordinate {name!r} exceeds the bounded comparison size")
            if not _equal(left.values, right.values):
                raise ValueError(f"Multi-file non-time coordinate {name!r} differs")


def open_mfdataset(paths, *, time="time", chunks="auto", engine="netcdf4", metadata_policy="strict") -> xr.Dataset:
    """Strictly concatenate local files and sort records by actual CF time.

    Each input has a nonempty, increasing decoded time coordinate. All other
    coordinates, field names, dtypes and scientific variable metadata agree
    exactly. A shared schema is required; no time reindexing or implicit
    member selection occurs. Missing-period checks need a declared expected
    index and are not inferred here.

    ``metadata_policy="strict"`` also requires all descriptive attributes to
    agree. ``"preserve_sources"`` permits differences only in global
    ``history``, ``CONVERSION_DATE``, ``date_created``, ``date_modified``,
    ``creation_date`` and variable ``actual_range``, ``minimum_value``,
    ``maximum_value``. Their original values are retained in each provenance
    input's ``source_metadata`` under ``global_attributes`` and
    ``variable_attributes``. For multiple sources these file-scoped attributes
    are removed from the combined view, even when equal; no range is recomputed.
    A single source retains its attributes. Units, CF links, calendars, grids,
    valid_* limits and every unlisted attribute remain strict in both modes.

    Static data fields repeated across files are currently rejected. This
    avoids scanning potentially large orography/mask fields to prove equality;
    static coordinates (including CF bounds/CRS) are compared, up to one
    million elements each. The result owns every source close callback.
    """
    if not isinstance(metadata_policy, str) or metadata_policy not in ("strict", "preserve_sources"):
        raise ValueError("metadata_policy must be 'strict' or 'preserve_sources'")
    if isinstance(paths, (str, Path)):
        paths = [paths]
    sources = [Path(path).expanduser().resolve(strict=True) for path in paths]
    if not sources:
        raise ValueError("At least one local input file is required")
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate input filenames are not allowed")
    opened = []
    try:
        for source in sources:
            dataset = open_dataset(source, chunks=chunks, engine=engine, time=time)
            opened.append(dataset)
            validate_time(dataset, time=time)
        dimension = opened[0][time].dims[0]
        if any(dataset[time].dims != (dimension,) for dataset in opened):
            raise ValueError("The time coordinate dimension differs across files")
        for dataset in opened[1:]:
            _check_compatible(opened[0], dataset, dimension, time, metadata_policy)
        # Only concatenate variables that actually carry the time dimension;
        # this also preserves static fields when given a single input file.
        # Equality of non-time coordinates was checked explicitly above.
        combined = xr.concat(
            opened, dim=dimension, data_vars="minimal", coords="minimal",
            compat="equals", join="exact", combine_attrs="drop",
        ).sortby(time)
        reconciled = metadata_policy == "preserve_sources" and len(opened) > 1
        combined.attrs = {key: value for key, value in opened[0].attrs.items()
                          if key != PROVENANCE_ATTRIBUTE and not (reconciled and key in _SOURCE_GLOBAL_ATTRS)}
        # combine_attrs="drop" also drops variable attrs in supported xarray
        # versions. Restore validated semantics explicitly, instead of relying
        # on an implicit first-source override in xarray's attribute merger.
        for name, source_variable in opened[0].variables.items():
            combined[name].attrs = {key: value for key, value in source_variable.attrs.items()
                                    if not (reconciled and key in _SOURCE_RANGE_ATTRS)}
            combined[name].encoding = dict(combined[name].encoding)
            for key in _CF_LINKS:
                if key in source_variable.encoding:
                    combined[name].encoding[key] = source_variable.encoding[key]
        validate_time(combined, time=time)
        validate_metadata(combined)

        def close_all():
            for source_dataset in opened:
                source_dataset.close()

        combined.set_close(close_all)
        source_records = []
        for source, source_dataset in zip(sources, opened):
            record = input_record(source)
            # Multiple sources have separate processing histories. Keep each
            # branch instead of falsely linearizing or discarding that history.
            if PROVENANCE_ATTRIBUTE in source_dataset.attrs:
                record["provenance"] = json.loads(source_dataset.attrs[PROVENANCE_ATTRIBUTE])
            if metadata_policy == "preserve_sources":
                record["source_metadata"] = _source_metadata(source_dataset)
            source_records.append(record)
        result = record_operation(
            combined, "open_mfdataset",
            {"engine": engine, "chunks": chunks, "time": time, "join": "exact",
             "duplicate_time": "error", "static_fields": "error", "metadata_policy": metadata_policy},
            inputs=source_records,
        )
    except BaseException:
        for dataset in opened:
            dataset.close()
        raise
    return result
