"""Explicit Dask execution and non-overwriting publication of a NetCDF file.

References
----------
Xarray serialization/encoding:
https://docs.xarray.dev/en/stable/user-guide/io.html#writing-encoded-data
Climate-Base's original atomic seasonal writer:
https://github.com/Dengruihan-clim/Climate-Base/blob/d0ad09e0ca1f9cc589469a0fe08a30f8e7be6601/Preprocessing/climate_preprocessing/io.py

Climate-Base improvements
-------------------------
This independent general-Dataset writer reconstructs disk encoding after
transformations, preserves CF relationship metadata, closes output stores on
failure, and publishes only a completed file without overwriting an existing
target. Hard-link publication applies to local filesystems, not Zarr/cloud.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import tempfile

import numpy as np
import xarray as xr

from ._time_roundtrip import validate_time_roundtrip
from ..metadata import validate_metadata
from ..provenance import record_operation
from ..validation import ensure_lazy

_CF_LINKS = ("bounds", "climatology", "coordinates", "grid_mapping", "cell_measures",
             "formula_terms", "ancillary_variables")
_TIME_UNITS = {"D": "days", "h": "hours", "m": "minutes", "s": "seconds",
               "ms": "milliseconds", "us": "microseconds", "ns": "nanoseconds"}


def _time_encodings(dataset, output, requested):
    """Use paired integer time encoding, shared by CF centers and bounds.

    Only bounded date labels and relationship metadata are inspected here.
    Date labels are checked after writing, before publication; meteorological
    fields stay lazy.
    """
    names = {name for name in dataset.variables if dataset[name].dtype.kind == "M"
             or (dataset[name].dtype.kind == "O"
                 and ("since" in str(output.get(name, {}).get("units", ""))
                      or isinstance(dataset.indexes.get(name), xr.CFTimeIndex)))}
    for name in names:
        index = dataset.indexes.get(name)
        if isinstance(index, xr.CFTimeIndex):
            output.setdefault(name, {}).setdefault("calendar", index.calendar)
    links = []
    for name in dataset.variables:
        for relationship in ("bounds", "climatology"):
            target = dataset[name].attrs.get(relationship)
            if target in dataset.variables and dataset[name].dtype.kind in "MO" and dataset[target].dtype.kind in "MO":
                links.append({name, target})
    # A CF linked object bound may have no independent time encoding.
    while any(link & names and not link <= names for link in links):
        for link in links:
            if link & names:
                names.update(link)
    groups = [{name} for name in sorted(names)]
    for link in links:
        connected = [group for group in groups if group & link]
        if connected:
            merged = set().union(*connected)
            groups = [group for group in groups if not group & link] + [merged]
    changes = {}
    aliases = {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}
    for group in groups:
        ordered = sorted(group)
        calendars = {output.get(name, {}).get("calendar") for name in ordered} - {None}
        if len({aliases.get(value, value) for value in calendars}) > 1:
            raise ValueError("Linked datetime coordinates and bounds require consistent calendars")
        explicit = [name for name in ordered if {"units", "dtype"} & set(requested.get(name, {}))]
        if explicit:
            pairs = []
            for name in explicit:
                settings = output.get(name, {})
                if settings.get("units") is None or settings.get("dtype") is None:
                    raise ValueError("Explicit datetime encoding requires both units and dtype")
                pairs.append((settings["units"], np.dtype(settings["dtype"])))
            if any(pair != pairs[0] for pair in pairs[1:]):
                raise ValueError("Explicit linked datetime encodings must use the same units and dtype")
            units, dtype = pairs[0]
        else:
            precisions = []
            for name in ordered:
                variable = dataset[name]
                unit = "us" if variable.dtype.kind == "O" else np.datetime_data(variable.dtype)[0]
                if unit == "ns":
                    if variable.size > 1_000_000:
                        raise ValueError(f"Output time verification for {name!r} exceeds the 1000000-entry limit")
                    dates = variable.to_numpy()
                    ticks = dates[~np.isnat(dates)].astype("int64")
                    # Microseconds remain readable by cftime/legacy NetCDF
                    # APIs. Use nanoseconds only when actual labels need them.
                    if not np.any(ticks % 1000):
                        unit = "us"
                precisions.append(unit)
            if any(unit not in _TIME_UNITS for unit in precisions):
                raise ValueError("Unsupported decoded datetime precision")
            unit = max(precisions, key=lambda value: list(_TIME_UNITS).index(value))
            units, dtype = f"{_TIME_UNITS[unit]} since 1970-01-01", "int64"
        for name in ordered:
            settings = output.setdefault(name, {})
            previous = settings.get("units")
            settings.update(units=units, dtype=dtype)
            if calendars:
                settings.setdefault("calendar", sorted(calendars)[0])
            changes[name] = {"source_units": previous, "output_units": units,
                             "output_dtype": str(np.dtype(dtype))}
    return changes


def _prepare(dataset, encoding):
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("write_dataset requires an xarray.Dataset")
    validate_metadata(dataset)
    if encoding is not None and not isinstance(encoding, Mapping):
        raise TypeError("encoding must map variable names to output encoding dictionaries")
    requested = {} if encoding is None else encoding
    if set(requested) - set(dataset.variables):
        raise ValueError("Output encoding names a variable that does not exist")
    result = dataset.copy(deep=False)
    result.attrs = dict(dataset.attrs)
    result.encoding = {}
    output_encoding = {}
    discarded_ranges = {}
    for name in result.variables:
        variable = result[name]
        source_encoding = dict(variable.encoding)
        variable.attrs = dict(variable.attrs)
        variable.encoding = {}
        for key in _CF_LINKS:
            if key in source_encoding:
                variable.attrs[key] = source_encoding[key]
        if "scale_factor" in variable.attrs or "add_offset" in variable.attrs:
            raise ValueError(
                f"Variable {name!r} has packing attributes: decode input first, "
                "then provide new packing only through output encoding"
            )
        if "scale_factor" in source_encoding or "add_offset" in source_encoding:
            # CF valid_range/min/max on packed data may describe encoded
            # integers, whereas this output contains decoded physical values.
            # A trustworthy range cannot be inferred from the attribute alone.
            removed = []
            for key in ("valid_min", "valid_max", "valid_range", "actual_range"):
                if key in variable.attrs:
                    variable.attrs.pop(key)
                    removed.append(key)
            if removed:
                discarded_ranges[name] = removed
        # Time metadata describe the decoded calendar, unlike stale field
        # dtype/packing/chunking settings, which must not be blindly copied.
        settings = {key: source_encoding[key] for key in ("units", "calendar")
                    if key in source_encoding and "since" in str(source_encoding.get("units", ""))}
        if name in requested:
            if not isinstance(requested[name], Mapping):
                raise TypeError(f"Output encoding for {name!r} must be a mapping")
            settings.update(dict(requested[name]))
        if settings:
            output_encoding[name] = settings
    reset_time_encoding = _time_encodings(result, output_encoding, requested)
    # Put only eager fields on the Dask path. Do not replace the user's
    # existing block boundaries with another automatic chunking decision.
    result = ensure_lazy(result)
    if discarded_ranges:
        result = record_operation(result, "reset_output_encoding", {
            "removed_range_attributes": discarded_ranges,
            "reason": "Source packed-value range is not valid for decoded output",
        })
    if reset_time_encoding:
        result = record_operation(result, "reset_time_encoding", {
            "variables": reset_time_encoding,
            "reason": "Pair time units/dtype and keep CF center/bounds encoding consistent",
        })
    result.set_close(dataset.close)
    return result, output_encoding


def write_dataset(dataset, path, *, encoding=None, engine="netcdf4") -> Path:
    """Compute and write a new local NetCDF file, publishing only on success.

    This is an explicit execution boundary: Dask schedules field computation
    through xarray's writer. Source files must remain available until it
    returns. Existing files, including dangling symlinks, are never replaced.
    Source encoding's packed dtype, scaling and chunk sizes are discarded;
    specify desired packing/compression explicitly in ``encoding``.

    The destination filesystem must support hard links. The caller owns input
    Dataset resources; writing does not close the input Dataset.
    """
    destination = Path(path).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Output exists; choose a new NetCDF filename")
    if engine not in {"netcdf4", "h5netcdf", "scipy"}:
        raise ValueError("Use an explicit NetCDF engine: netcdf4, h5netcdf, or scipy")
    prepared, output_encoding = _prepare(dataset, encoding)
    prepared = record_operation(
        prepared, "write_dataset", {"engine": engine, "target": destination.name},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(filename)
    try:
        # compute=True uses Dask for chunked fields and, unlike returning a
        # deferred writer, ensures xarray closes the output store on failure.
        prepared.to_netcdf(temporary, engine=engine, encoding=output_encoding, compute=True)
        with xr.open_dataset(temporary, engine=engine, decode_coords="all") as reopened:
            validate_metadata(reopened)
            if dict(reopened.sizes) != dict(prepared.sizes):
                raise ValueError("Written dimensions differ from the prepared Dataset")
            validate_time_roundtrip(dataset, reopened)
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
