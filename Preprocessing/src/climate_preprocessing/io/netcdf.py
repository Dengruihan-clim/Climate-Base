"""Local CF NetCDF reading with an explicit xarray/Dask execution contract.

References
----------
Xarray public I/O and backend API:
https://docs.xarray.dev/en/stable/user-guide/io.html

Climate-Base improvements
-------------------------
This independent wrapper requires chunked fields, validates CF references and
decoded time, preserves source closing, and records safe input identifiers.
It never selects an ensemble member or changes units implicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import xarray as xr

from ..metadata import validate_metadata
from ..provenance import input_record, record_operation
from ..time import validate_time


def _check_input_stamp(path, stamp):
    """Reject unavailable or changed input before a block enters the result."""
    try:
        current = Path(path).stat()
    except OSError as error:
        raise ValueError("Input became unavailable after lazy graph construction") from error
    if (current.st_size, current.st_mtime_ns) != stamp:
        raise ValueError("Input changed after lazy graph construction; reopen the Dataset")


def _guard_source_block(block, *, path, stamp):
    _check_input_stamp(path, stamp)
    return block


def _checked_chunks(chunks):
    if chunks is None or isinstance(chunks, bool):
        raise ValueError("chunks must enable Dask: use 'auto', a positive integer, or a mapping")
    if isinstance(chunks, str):
        if chunks != "auto":
            raise ValueError("The only supported chunk string is 'auto'")
    elif isinstance(chunks, int):
        if chunks <= 0:
            raise ValueError("The chunk size must be positive")
    elif not isinstance(chunks, Mapping):
        raise TypeError("chunks must be 'auto', a positive integer, or a mapping")
    else:
        for dimension, size in chunks.items():
            if not isinstance(dimension, str):
                raise TypeError("Chunk dimension names must be strings")
            if size == "auto":
                continue
            if isinstance(size, bool) or not isinstance(size, int) or (size <= 0 and size != -1):
                raise ValueError("Mapped chunk sizes must be positive integers, -1 (whole axis), or 'auto'")
    return dict(chunks) if isinstance(chunks, Mapping) else chunks


def open_dataset(path, *, chunks="auto", engine="netcdf4", time="time") -> xr.Dataset:
    """Open one local decoded CF Dataset with Dask-backed field arrays.

    Parameters
    ----------
    path : path-like
        Existing local NetCDF file. Remote resources are not fetched.
    chunks : 'auto', positive int, or mapping
        Dask chunk specification. ``None`` is rejected to prevent an accidental
        eager field path. Mapping values may also be -1 for the whole axis;
        unknown dimension names are errors. Coordinates may be loaded for
        indexing and checks.
    engine : str
        Explicit xarray NetCDF engine, normally ``'netcdf4'``.
    time : str
        Decoded time coordinate to validate when it is present.

    Returns
    -------
    xarray.Dataset
        Decoded fields and CF-related coordinates. Use as a context manager or
        call ``close()`` after computation/output. Lazy graphs need the source
        files to remain unchanged and available until execution completes.
        Every computed Dask field block verifies the original file size/mtime
        before contributing to the result. This is not a cryptographic hash.

    Notes
    -----
    The requested time coordinate is checked when present. Its absence is
    allowed for a static grid Dataset; ``open_mfdataset`` requires it. Field
    values are not scanned during opening.
    """
    chunks = _checked_chunks(chunks)
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Expected a local NetCDF file")
    if engine not in {"netcdf4", "h5netcdf", "scipy"}:
        raise ValueError("Use an explicit NetCDF engine: netcdf4, h5netcdf, or scipy")
    original = source.stat()
    stamp = original.st_size, original.st_mtime_ns
    raw = xr.open_dataset(source, engine=engine, chunks=chunks, decode_cf=False)
    try:
        if isinstance(chunks, Mapping):
            unknown = set(chunks) - set(raw.dims)
            if unknown:
                raise ValueError(f"Unknown chunk dimensions: {sorted(unknown)}")
        # Inspect the on-disk relationships before xarray's decoder can drop
        # a missing bounds/CRS reference with only a warning. Decoding these
        # Dask-backed variables builds graphs rather than computing fields.
        validate_metadata(raw)
        dataset = xr.decode_cf(raw, mask_and_scale=True, decode_times=True, decode_coords="all")
        dataset.set_close(raw.close)
        validate_metadata(dataset)
        if time in dataset.coords:
            validate_time(dataset, time=time)
        for name in dataset.data_vars:
            variable = dataset[name]
            if variable.chunks is not None:
                guarded = variable.data.map_blocks(
                    _guard_source_block, path=str(source), stamp=stamp,
                    dtype=variable.dtype,
                    meta=np.empty((0,) * variable.ndim, dtype=variable.dtype),
                )
                dataset[name] = variable.copy(data=guarded, deep=False)
        # Strip transport paths from returned encoding; backend handles retain
        # their own file references and do not depend on encoding['source'].
        dataset.encoding.pop("source", None)
        for variable in dataset.variables.values():
            variable.encoding.pop("source", None)
        result = record_operation(
            dataset, "open_dataset", {"engine": engine, "chunks": chunks, "time": time},
            inputs=[input_record(source)],
        )
        _check_input_stamp(source, stamp)
    except BaseException:
        raw.close()
        raise
    return result
