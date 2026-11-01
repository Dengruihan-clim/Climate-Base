"""Observable lazy/compatibility contracts; synthetic source data only."""

import json

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr
from netCDF4 import Dataset
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from climate_preprocessing import Coordinates, FieldSpec, MonthlyArchive, seasonal_mean
from climate_preprocessing.units import convert_dataarray_units
from test_archive import make_file


def monthly_archive(tmp_path, **kwargs):
    return MonthlyArchive(
        {"t": FieldSpec("monthly.nc", "temperature", Coordinates(pressure="lev"))},
        data_root=tmp_path, hash_inputs=False, **kwargs,
    )


def test_monthly_graph_does_not_read_field_and_preserves_old_result(tmp_path, monkeypatch):
    make_file(tmp_path / "monthly.nc", permute=True)
    field_reads, tasks = [], []
    original_getitem = NetCDF4ArrayWrapper.__getitem__

    def tracked_getitem(self, key):
        if self.variable_name == "temperature":
            field_reads.append(key)
        return original_getitem(self, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", tracked_getitem)
    source = monthly_archive(tmp_path)
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        field = source.read_lazy("t", 2000, 1, chunks={"latitude": 1, "longitude": 2})
    try:
        assert not tasks
        assert not field_reads
        assert isinstance(field.data, da.Array)
        assert field.dims == ("pressure", "latitude", "longitude")
        np.testing.assert_array_equal(field.latitude.data, [-90, 0, 90])
        np.testing.assert_array_equal(field.pressure.data, [100, 500, 1000])
        actual = field.compute()
        assert field_reads
        np.testing.assert_array_equal(actual.data, source.read("t", 2000, 1).values)
    finally:
        field.close()


def test_units_are_deferred_preserve_chunks_and_remove_packing():
    source = xr.DataArray(
        da.from_array([0., 20., np.inf, np.nan], chunks=2), dims="sample", name="temperature",
        coords={"sample": [1, 2, 3, 4]}, attrs={"units": "degree_Celsius", "long_name": "Test"},
    )
    source.encoding = {"scale_factor": 0.01, "add_offset": 250., "dtype": "int16", "_FillValue": -9999}
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        actual = convert_dataarray_units(source, "K")
    assert not tasks
    assert actual.chunks == source.chunks
    assert actual.name == source.name
    assert actual.attrs["long_name"] == "Test"
    assert actual.attrs["units"] == "K"
    assert source.attrs["units"] == "degree_Celsius"
    assert actual.encoding == {}
    assert "scale_factor" in source.encoding
    np.testing.assert_array_equal(actual.sample.data, source.sample.data)
    np.testing.assert_allclose(actual.compute().data, [273.15, 293.15, np.nan, np.nan], equal_nan=True)


def test_units_overflow_is_checked_when_block_executes():
    source = xr.DataArray(da.from_array([1e308], chunks=1), dims="x", attrs={"units": "hPa"})
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        actual = convert_dataarray_units(source, "Pa")
    assert not tasks
    with pytest.raises(ValueError, match="nonfinite values from finite"):
        actual.compute()


def test_eager_dataarray_enters_dask_and_override_is_audited():
    source = xr.DataArray([1., 2.], dims="x", attrs={"units": "incorrect"})
    actual = convert_dataarray_units(source, "K", source_units="degree_Celsius")
    assert isinstance(actual.data, da.Array)
    audit = json.loads(actual.attrs["climate_preprocessing_unit_transform"])
    assert audit["original_units"] == "incorrect"
    assert audit["source_units"] == "degree_Celsius"
    np.testing.assert_allclose(actual.compute().data, [274.15, 275.15])


def test_unit_conversion_removes_and_audits_stale_range_metadata_without_scan():
    source = xr.DataArray(
        da.from_array([20.], chunks=1), dims="x",
        attrs={"units": "degree_Celsius", "standard_name": "air_temperature",
               "valid_min": np.float32(-50), "valid_max": np.float32(50),
               "valid_range": np.array([-50., 50.]), "actual_range": np.array([20., 20.])},
    )
    source.encoding = {"climatology": "climatology_bounds", "ancillary_variables": "quality_flag"}
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        result = convert_dataarray_units(source, "K")
    assert not tasks
    removed = json.loads(result.attrs["climate_preprocessing_unit_transform"])["removed_attributes"]
    assert set(removed) == {"valid_min", "valid_max", "valid_range", "actual_range"}
    assert removed["valid_range"]["value"] == [-50., 50.]
    assert all(name not in result.attrs for name in removed)
    assert result.attrs["standard_name"] == "air_temperature"
    assert result.encoding["climatology"] == "climatology_bounds"
    assert result.encoding["ancillary_variables"] == "quality_flag"
    assert "valid_range" in source.attrs
    np.testing.assert_allclose(result.compute().data, [293.15])


def test_explicit_physical_transform_does_not_keep_old_standard_name():
    source = xr.DataArray([9.80665], dims="x", attrs={"units": "m2 s-2", "standard_name": "geopotential"})
    result = convert_dataarray_units(source, "m", scale=1 / 9.80665)
    assert "standard_name" not in result.attrs
    removed = json.loads(result.attrs["climate_preprocessing_unit_transform"])["removed_attributes"]
    assert removed["standard_name"]["value"] == "geopotential"
    assert source.attrs["standard_name"] == "geopotential"
    np.testing.assert_allclose(result.compute().data, [1.])


def test_temperature_difference_rejects_absolute_offset():
    source = xr.DataArray([2.], dims="x", attrs={"units": "K", "units_metadata": "temperature: difference"})
    with pytest.raises(ValueError, match="temperature difference"):
        convert_dataarray_units(source, "degree_Celsius")
    result = convert_dataarray_units(source, "degree_Celsius", scale=1., offset=0.)
    np.testing.assert_allclose(result.compute().data, [2.])


def test_changed_source_is_rejected_when_lazy_graph_computes(tmp_path):
    path = make_file(tmp_path / "monthly.nc")
    field = monthly_archive(tmp_path).read_lazy("t", 2000, 1)
    # Releasing the file handle allows this controlled mutation on Windows.
    field.close()
    with Dataset(path, "a") as ds:
        ds.changed_source = "changed after graph creation" * 100
    with pytest.raises(ValueError, match="Input changed after lazy graph construction"):
        field.compute()
    field.close()


@pytest.mark.parametrize("chunks", [None, {"unknown_axis": 1}])
def test_lazy_monthly_reader_rejects_invalid_chunk_contract(tmp_path, chunks):
    make_file(tmp_path / "monthly.nc")
    with pytest.raises(ValueError, match="chunks|chunk"):
        monthly_archive(tmp_path).read_lazy("t", 2000, 1, chunks=chunks)


def test_season_uses_lazy_monthly_interface_and_closes_sources(tmp_path, monkeypatch):
    make_file(tmp_path / "monthly.nc", dates=((1999, 12), (2000, 1), (2000, 2)),
              values=np.array([0., 10., 20.])[:, None, None, None])
    source = monthly_archive(tmp_path)
    original_read = source.read_lazy
    closed = []

    def tracked_read(name, year, month):
        field = original_read(name, year, month)
        close_source = field._close

        def close():
            closed.append((year, month))
            close_source()

        field.set_close(close)
        return field

    monkeypatch.setattr(source, "read_lazy", tracked_read)
    monkeypatch.setattr(source, "read", lambda *args: pytest.fail("season called eager monthly read"))
    actual = seasonal_mean(source, "t", 2000, 2000)
    np.testing.assert_allclose(actual.values, (10 * 31 + 20 * 29) / 91)
    assert closed == [(1999, 12), (2000, 1), (2000, 2)]


def test_incomplete_season_closes_already_opened_lazy_sources(tmp_path, monkeypatch):
    make_file(tmp_path / "monthly.nc", dates=((1999, 12), (2000, 1)))
    source = monthly_archive(tmp_path)
    original_read = source.read_lazy
    closed = []

    def tracked_read(name, year, month):
        field = original_read(name, year, month)
        close_source = field._close

        def close():
            closed.append((year, month))
            close_source()

        field.set_close(close)
        return field

    monkeypatch.setattr(source, "read_lazy", tracked_read)
    with pytest.raises(ValueError, match="found 0"):
        seasonal_mean(source, "t", 2000, 2000)
    assert closed == [(1999, 12), (2000, 1)]
