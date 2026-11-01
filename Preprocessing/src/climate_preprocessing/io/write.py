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

import xarray as xr

from ..metadata import validate_metadata
from ..provenance import record_operation
from ..validation import ensure_lazy

_CF_LINKS = ("bounds", "climatology", "coordinates", "grid_mapping", "cell_measures",
             "formula_terms", "ancillary_variables")


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
    # Put only eager fields on the Dask path. Do not replace the user's
    # existing block boundaries with another automatic chunking decision.
    result = ensure_lazy(result)
    if discarded_ranges:
        result = record_operation(result, "reset_output_encoding", {
            "removed_range_attributes": discarded_ranges,
            "reason": "Source packed-value range is not valid for decoded output",
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
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
