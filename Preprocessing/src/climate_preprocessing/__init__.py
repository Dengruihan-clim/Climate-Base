"""Strict xarray/Dask preprocessing with compatible monthly interfaces."""
from ._version import __version__
from .archive import Coordinates, FieldSpec, MonthlyArchive, MonthlyField
from .seasons import SeasonalField, iter_seasons, seasonal_mean, month_days, season_dates
from .units import convert_units, convert_dataarray_units
from .io import write_seasonal, write_seasonal_stream, open_dataset, open_mfdataset, write_dataset
from .io import open_grib, inspect_grib_groups, open_zarr, write_zarr
from .coordinates import normalize_coordinates, normalize_pressure
from .metadata import validate_metadata
from .time import validate_time, subset_time
from .subset import subset_region
from .masking import mask_invalid, apply_mask
from .pipeline import prepare_dataset
from .regrid import regrid_dataset, rectilinear_grid
from .vertical import hybrid_pressure, interpolate_pressure
from .adapters import adapt_product

__all__ = ["Coordinates", "FieldSpec", "MonthlyArchive", "MonthlyField", "SeasonalField",
           "iter_seasons", "seasonal_mean", "month_days", "season_dates", "convert_units", "write_seasonal", "write_seasonal_stream",
           "__version__", "convert_dataarray_units", "open_dataset", "open_mfdataset", "write_dataset",
           "normalize_coordinates", "normalize_pressure", "validate_metadata", "validate_time", "subset_time",
           "subset_region", "mask_invalid", "apply_mask", "prepare_dataset",
           "open_grib", "inspect_grib_groups", "open_zarr", "write_zarr",
           "regrid_dataset", "rectilinear_grid", "hybrid_pressure", "interpolate_pressure", "adapt_product"]
