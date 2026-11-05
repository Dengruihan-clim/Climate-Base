"""Lazy local I/O; optional GRIB/Zarr backends load only when requested."""

from ._seasonal import write_seasonal, write_seasonal_stream
from .multifile import open_mfdataset
from .netcdf import open_dataset
from .write import write_dataset
from .grib import open_grib, inspect_grib_groups
from .zarr import open_zarr, write_zarr

__all__ = [
    "open_dataset", "open_mfdataset", "write_dataset",
    "write_seasonal", "write_seasonal_stream",
    "open_grib", "inspect_grib_groups", "open_zarr", "write_zarr",
]
