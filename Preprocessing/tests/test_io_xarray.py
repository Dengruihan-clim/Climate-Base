"""I/O contracts: deferred fields, strict merging, metadata and publication."""

from __future__ import annotations

import json
from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.io import open_dataset, open_mfdataset, write_dataset
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE, record_operation


def _dataset(times=("2000-01-01", "2000-02-01"), *, offset=0.0):
    values = np.arange(len(times) * 6, dtype=float).reshape(len(times), 2, 3) + offset
    result = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values, {"units": "K"})},
        coords={
            "time": np.array(times, dtype="datetime64[ns]"),
            "latitude": ("latitude", [-30.0, 30.0], {"units": "degrees_north", "standard_name": "latitude"}),
            "longitude": ("longitude", [0.0, 120.0, 240.0], {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={"title": "Synthetic CF example"},
    )
    return result


def _write_fixture(path, times=("2000-01-01", "2000-02-01"), *, offset=0.0, **options):
    dataset = _dataset(times, offset=offset)
    dataset.to_netcdf(path, engine="netcdf4", **options)
    return dataset


def test_single_read_is_lazy_and_does_not_expose_input_directory(tmp_path):
    source = tmp_path / "source.nc"
    expected = _write_fixture(source)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args[0])):
        dataset = open_dataset(source, chunks={"time": 1})
    try:
        assert tasks == []
        assert isinstance(dataset.temperature.data, da.Array)
        assert dataset.temperature.chunks[0] == (1, 1)
        assert str(tmp_path) not in dataset.attrs[PROVENANCE_ATTRIBUTE]
        assert "source" not in dataset.encoding
        assert all("source" not in variable.encoding for variable in dataset.variables.values())
        xr.testing.assert_allclose(dataset.temperature.compute(), expected.temperature)
    finally:
        dataset.close()


@pytest.mark.parametrize("chunks", [None, False, 0, "none"])
def test_read_requires_dask_chunks(tmp_path, chunks):
    path = tmp_path / "source.nc"
    _write_fixture(path)
    with pytest.raises((TypeError, ValueError), match="chunk"):
        open_dataset(path, chunks=chunks)


@pytest.mark.parametrize("chunks", [{"typo_dimension": 1}, {"time": 0}, {"time": -2}, {"time": True}])
def test_read_rejects_invalid_mapped_chunks(tmp_path, chunks):
    path = tmp_path / "source.nc"
    _write_fixture(path)
    with pytest.raises((TypeError, ValueError), match="[Cc]hunk"):
        open_dataset(path, chunks=chunks)


@pytest.mark.parametrize("attribute", ["bounds", "grid_mapping"])
def test_read_rejects_dangling_cf_reference_before_decoder_can_drop_it(tmp_path, attribute):
    path = tmp_path / "dangling.nc"
    dataset = _dataset()
    dataset.temperature.attrs[attribute] = "missing_related_variable"
    dataset.to_netcdf(path)
    with pytest.raises(ValueError, match="missing_related_variable"):
        open_dataset(path)


def test_nonstandard_calendar_is_preserved(tmp_path):
    path = tmp_path / "360day.nc"
    source = _dataset()
    source = source.assign_coords(time=xr.cftime_range("2000-01-30", periods=2, freq="30D", calendar="360_day"))
    source.to_netcdf(path, engine="netcdf4")
    with open_dataset(path) as result:
        assert result.time.dt.calendar == "360_day"
        assert isinstance(result.temperature.data, da.Array)


def test_changed_source_is_rejected_before_computed_values_are_returned(tmp_path):
    from netCDF4 import Dataset

    path = tmp_path / "source.nc"
    _write_fixture(path)
    source = open_dataset(path, chunks={"time": 1})
    # Close the original handle so Windows permits this controlled mutation.
    # The graph still refers to its backend, whose public contract requires
    # the source to remain unchanged until execution has completed.
    source.close()
    with Dataset(path, "a") as changed:
        changed.changed_source = "mutation after lazy graph construction" * 100
    with pytest.raises(ValueError, match="Input changed after lazy graph construction"):
        source.temperature.compute()


def test_changed_source_cannot_publish_output(tmp_path):
    from netCDF4 import Dataset

    path, destination = tmp_path / "source.nc", tmp_path / "output.nc"
    _write_fixture(path)
    source = open_dataset(path)
    source.close()
    with Dataset(path, "a") as changed:
        changed.changed_source = "changed before write" * 100
    with pytest.raises(ValueError, match="Input changed after lazy graph construction"):
        write_dataset(source, destination)
    assert not destination.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_multifile_sorts_actual_times_and_stays_lazy(tmp_path):
    late, early = tmp_path / "late.nc", tmp_path / "early.nc"
    _write_fixture(late, ("2000-03-01", "2000-04-01"), offset=20)
    _write_fixture(early)
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args[0])):
        dataset = open_mfdataset([late, early], chunks={"time": 1})
    try:
        assert tasks == []
        assert isinstance(dataset.temperature.data, da.Array)
        np.testing.assert_array_equal(dataset.time.dt.month, [1, 2, 3, 4])
        assert dataset.temperature.isel(time=0, latitude=0, longitude=0).compute().item() == 0
        assert dataset.temperature.isel(time=2, latitude=0, longitude=0).compute().item() == 20
        provenance = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])
        assert provenance["operations"][-1]["operation"] == "open_mfdataset"
        assert [item["name"] for item in provenance["operations"][-1]["inputs"]] == ["late.nc", "early.nc"]
    finally:
        dataset.close()


@pytest.mark.parametrize("conflict", ["time", "grid", "attributes", "static", "units"])
def test_multifile_rejects_ambiguous_merge(tmp_path, conflict):
    first, second = tmp_path / "a.nc", tmp_path / "b.nc"
    a = _dataset()
    b = _dataset(("2000-03-01", "2000-04-01"))
    if conflict == "time":
        b = _dataset(("2000-02-01", "2000-03-01"))
    elif conflict == "grid":
        b = b.assign_coords(longitude=b.longitude + 1)
        b.longitude.attrs = dict(a.longitude.attrs)
    elif conflict == "attributes":
        b.attrs["title"] = "Conflicting title"
    elif conflict == "static":
        a["terrain"] = (("latitude", "longitude"), np.zeros((2, 3)))
        b["terrain"] = (("latitude", "longitude"), np.zeros((2, 3)))
    else:
        b.temperature.attrs["units"] = "degC"
    a.to_netcdf(first)
    b.to_netcdf(second)
    with pytest.raises(ValueError):
        open_mfdataset([first, second])


def test_multifile_one_file_does_not_expand_static_field(tmp_path):
    source = _dataset()
    source["terrain"] = (("latitude", "longitude"), np.zeros((2, 3)))
    path = tmp_path / "one.nc"
    source.to_netcdf(path)
    with open_mfdataset([path]) as result:
        assert result.terrain.dims == ("latitude", "longitude")


def test_multifile_keeps_distinct_input_processing_histories(tmp_path):
    paths = [tmp_path / "a.nc", tmp_path / "b.nc"]
    for index, path in enumerate(paths):
        source = _dataset((f"2000-0{index + 1}-01",))
        record_operation(source, f"previous_operation_{index}").to_netcdf(path)
    with open_mfdataset(paths) as result:
        history = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])
        inputs = history["operations"][-1]["inputs"]
        for index, source in enumerate(inputs):
            operations = source["provenance"]["operations"]
            assert operations[0]["operation"] == f"previous_operation_{index}"
            assert operations[-1]["operation"] == "open_dataset"


def test_multifile_closes_every_source_on_success_and_failure(monkeypatch):
    from climate_preprocessing.io import multifile

    closed = []
    opened = []
    for number in (0, 1):
        ds = _dataset((f"2000-0{number + 1}-01",)).chunk("auto")
        ds.set_close(lambda index=number: closed.append(index))
        opened.append(ds)
    # Real paths are required but content is replaced by in-memory Datasets.
    paths = [Path(__file__), Path(multifile.__file__)]
    source_iter = iter(opened)
    monkeypatch.setattr(multifile, "open_dataset", lambda *args, **kwargs: next(source_iter))
    result = open_mfdataset(paths)
    assert closed == []
    result.close()
    assert closed == [0, 1]

    closed.clear()
    opened = [_dataset(("2000-01-01",)).chunk("auto") for _ in range(2)]
    for number, ds in enumerate(opened):
        ds.set_close(lambda index=number: closed.append(index))
    source_iter = iter(opened)
    with pytest.raises(ValueError):
        open_mfdataset(paths)
    assert closed == [0, 1]


def test_write_rebuilds_packing_and_preserves_missing_bounds(tmp_path):
    original_path, output_path = tmp_path / "packed.nc", tmp_path / "output.nc"
    source = _dataset()
    source.temperature.values[0, 0, 0] = np.nan
    source.temperature.attrs["valid_range"] = np.array([-3000, 3000], dtype=np.int16)
    source["latitude_bounds"] = (("latitude", "bnds"), [[-60.0, 0.0], [0.0, 60.0]])
    source.latitude.attrs["bounds"] = "latitude_bounds"
    source.to_netcdf(original_path, encoding={"temperature": {
        "dtype": "int16", "scale_factor": 0.1, "add_offset": 273.15, "_FillValue": -32768,
    }})
    with open_dataset(original_path) as dataset:
        assert dataset.temperature.encoding["scale_factor"] == 0.1
        original_encoding = dict(dataset.temperature.encoding)
        write_dataset(dataset, output_path)
        assert dataset.temperature.encoding == original_encoding
        with xr.open_dataset(output_path, decode_coords="all") as result:
            xr.testing.assert_allclose(result.temperature, dataset.temperature.compute())
            assert result.temperature.dtype == dataset.temperature.dtype
            assert np.issubdtype(result.temperature.dtype, np.floating)
            assert "scale_factor" not in result.temperature.encoding
            assert "valid_range" not in result.temperature.attrs
            assert result.latitude.encoding["bounds"] == "latitude_bounds"
            assert "latitude_bounds" in result.coords
            assert np.isnan(result.temperature.isel(time=0, latitude=0, longitude=0))


def test_write_is_execution_boundary_and_refuses_overwrite(tmp_path):
    calls = []

    @delayed
    def values():
        calls.append("computed")
        return np.ones((2, 2, 3))

    source = _dataset()
    source["temperature"] = xr.DataArray(
        da.from_delayed(values(), shape=(2, 2, 3), dtype=float),
        dims=("time", "latitude", "longitude"), attrs={"units": "K"},
    )
    path = tmp_path / "result.nc"
    assert calls == []
    write_dataset(source, path)
    assert calls == ["computed"]
    content = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_dataset(source, path)
    assert path.read_bytes() == content
    assert calls == ["computed"]


def test_write_preparation_keeps_existing_chunks_with_mixed_eager_fields():
    from climate_preprocessing.io.write import _prepare

    source = _dataset().chunk({"time": 1, "latitude": 1, "longitude": 2})
    source["eager_auxiliary"] = (("latitude", "longitude"), np.zeros((2, 3)))
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args[0])):
        prepared, _ = _prepare(source, None)
    assert tasks == []
    assert prepared.temperature.chunks == source.temperature.chunks
    assert isinstance(prepared.eager_auxiliary.data, da.Array)
    assert isinstance(source.eager_auxiliary.data, np.ndarray)


def test_failed_computation_leaves_no_output_or_partial(tmp_path):
    @delayed
    def fail():
        raise RuntimeError("synthetic task failure")

    source = _dataset()
    source["temperature"] = xr.DataArray(
        da.from_delayed(fail(), shape=(2, 2, 3), dtype=float),
        dims=("time", "latitude", "longitude"), attrs={"units": "K"},
    )
    path = tmp_path / "failed.nc"
    with pytest.raises(RuntimeError, match="synthetic task failure"):
        write_dataset(source, path)
    assert not path.exists()
    assert list(tmp_path.glob("*.partial")) == []


def test_operation_records_do_not_mutate_source_and_close_is_forwarded():
    source = _dataset().chunk("auto")
    closed = []
    source.set_close(lambda: closed.append(True))
    result = record_operation(source, "test", {"value": 1})
    assert PROVENANCE_ATTRIBUTE not in source.attrs
    assert isinstance(result.temperature.data, da.Array)
    assert result.temperature.data is source.temperature.data
    result.close()
    assert closed == [True]
