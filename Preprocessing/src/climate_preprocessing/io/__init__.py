"""Lazy NetCDF input and explicit output, with legacy seasonal exports."""

from ._seasonal import write_seasonal, write_seasonal_stream
from .multifile import open_mfdataset
from .netcdf import open_dataset
from .write import write_dataset

__all__ = [
    "open_dataset", "open_mfdataset", "write_dataset",
    "write_seasonal", "write_seasonal_stream",
]
