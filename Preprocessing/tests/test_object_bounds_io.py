"""Object calendar bounds remain lazy and survive the existing NetCDF path."""

import cftime
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import open_dataset, write_dataset
from climate_preprocessing.validation import ensure_lazy


def test_cftime_bounds_lazy_chunk_and_netcdf_roundtrip(tmp_path):
    dates = [cftime.DatetimeNoLeap(2001, 1, 16), cftime.DatetimeNoLeap(2001, 2, 15)]
    bounds = [[cftime.DatetimeNoLeap(2001, 1, 1), cftime.DatetimeNoLeap(2001, 2, 1)],
              [cftime.DatetimeNoLeap(2001, 2, 1), cftime.DatetimeNoLeap(2001, 3, 1)]]
    dataset = xr.Dataset({"v": ("time", [1., 2.]), "time_bounds": (("time", "vertex"), bounds)},
                         coords={"time": ("time", dates, {"bounds": "time_bounds"})})
    dataset.time.encoding.update(units="days since 2001-01-01", calendar="noleap")
    dataset.time_bounds.encoding.update(units="days since 2001-01-01", calendar="noleap")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(True)):
        lazy = ensure_lazy(dataset)
    assert not tasks
    assert lazy.time_bounds.chunks is not None
    target = tmp_path / "bounds.nc"
    write_dataset(lazy, target)
    with open_dataset(target) as result:
        assert result.time.encoding["calendar"] == "noleap"
        assert result.time.encoding["units"] == result.time_bounds.encoding["units"]
        np.testing.assert_array_equal(result.time_bounds.compute(), bounds)
        np.testing.assert_array_equal(result.v.compute(), [1., 2.])


def test_eager_object_chunk_count_is_bounded_and_existing_blocks_retained():
    dates = np.full((300, 400), cftime.DatetimeNoLeap(2001, 1, 1), dtype=object)
    existing = xr.DataArray(np.ones((300, 400)), dims=("a", "b")).chunk({"a": 31, "b": 47})
    lazy = ensure_lazy(xr.Dataset({"date": (("a", "b"), dates), "v": existing}))
    assert np.prod([max(axis) for axis in lazy.date.chunks]) <= 65536
    assert lazy.v.chunks == existing.chunks


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
def test_chunked_datetime_preserves_nanosecond_labels_with_inherited_coarse_units(tmp_path, format):
    if format == "zarr":
        pytest.importorskip("zarr")
        from climate_preprocessing import open_zarr, write_zarr
        writer, reader, target = write_zarr, open_zarr, tmp_path / "nanoseconds.zarr"
    else:
        writer, reader, target = write_dataset, open_dataset, tmp_path / "nanoseconds.nc"
    dates = np.array([["2001-01-01T00:00:00.000000001", "2001-01-01T00:00:00.000000002"]], dtype="datetime64[ns]")
    dataset = xr.Dataset({"v": (("run", "step"), [[1., 2.]])},
                         coords={"valid_time": (("run", "step"), dates)}).chunk({"run": 1})
    dataset.valid_time.encoding.update(units="seconds since 1970-01-01", calendar="proleptic_gregorian")
    writer(dataset, target)
    with reader(target) as result:
        np.testing.assert_array_equal(result.valid_time.compute(), dates)


def test_chunked_explicit_time_precision_requires_pair_without_writing(tmp_path):
    dataset = xr.Dataset({"date": ("record", np.array(["2001-01-01"], dtype="datetime64[ns]"))}).chunk()
    dataset.date.encoding["units"] = "days since 2000-01-01"
    with pytest.raises(ValueError, match="both units and dtype"):
        write_dataset(dataset, tmp_path / "invalid.nc", encoding={"date": {"units": "hours since 2001-01-01"}})
    assert not (tmp_path / "invalid.nc").exists()


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
@pytest.mark.parametrize("relationship", ["bounds", "climatology"])
@pytest.mark.parametrize("inherited", [False, True])
def test_time_and_linked_bounds_share_precise_encoding(tmp_path, format, relationship, inherited):
    if format == "zarr":
        pytest.importorskip("zarr")
        from climate_preprocessing import open_zarr, write_zarr
        writer, reader = write_zarr, open_zarr
    else:
        writer, reader = write_dataset, open_dataset
    dates = [cftime.DatetimeNoLeap(2001, 1, 16, microsecond=1)]
    bounds = [[cftime.DatetimeNoLeap(2001, 1, 1, microsecond=1),
               cftime.DatetimeNoLeap(2001, 2, 1, microsecond=2)]]
    dataset = xr.Dataset({"v": ("time", [1.]), "edges": (("time", "vertex"), bounds)},
                         coords={"time": ("time", dates, {relationship: "edges"})})
    if inherited:
        dataset.time.encoding.update(units="days since 2001-01-01", calendar="noleap")
    target = tmp_path / ("dates.nc" if format == "netcdf" else "dates.zarr")
    writer(dataset, target)
    with reader(target) as result:
        assert result.time.encoding["units"] == result.edges.encoding["units"]
        np.testing.assert_array_equal(result.time, dates)
        np.testing.assert_array_equal(result.edges.compute(), bounds)


def test_explicit_incompatible_linked_time_encoding_rejected_before_write(tmp_path):
    dates = np.array(["2001-01-16"], dtype="datetime64[ns]")
    bounds = np.array([["2001-01-01", "2001-02-01"]], dtype="datetime64[ns]")
    dataset = xr.Dataset({"edges": (("time", "vertex"), bounds)},
                         coords={"time": ("time", dates, {"bounds": "edges"})})
    encoding = {"time": {"units": "days since 2000-01-01", "dtype": "int64"},
                "edges": {"units": "hours since 2000-01-01", "dtype": "int64"}}
    with pytest.raises(ValueError, match="same units and dtype"):
        write_dataset(dataset, tmp_path / "invalid.nc", encoding=encoding)
    assert not (tmp_path / "invalid.nc").exists()
