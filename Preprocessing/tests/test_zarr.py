"""Real local Zarr v2 roundtrips, lazy reads and failure-safe completion."""

from pathlib import Path
import importlib
import json
import os

import cftime
import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.io.zarr import open_zarr, write_zarr
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE

backend = importlib.import_module("climate_preprocessing.io.zarr")
pytestmark = pytest.mark.zarr


@pytest.fixture(autouse=True)
def real_backend():
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")


def source():
    values = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    values[1, 0, 1] = np.nan
    dataset = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values,
                         {"units": "K", "ancillary_variables": "qc", "coordinates": "latitude longitude"}),
         "qc": (("latitude", "longitude"), np.array([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int16),
                {"flag_values": [0, 1], "flag_meanings": "good bad"}),
         "keep": (("latitude", "longitude"), np.array([[True, False, True, False], [False, True, False, True]])),
         "station_id": ("latitude", np.array(["0001", "0010"])),
         "latitude_bounds": (("latitude", "vertex"), [[-15., -5.], [5., 15.]])},
        coords={"time": np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [-10., 10.], {"units": "degrees_north", "bounds": "latitude_bounds"}),
                "longitude": ("longitude", [0., 90., 180., 270.], {"units": "degrees_east"})},
    )
    return dataset.chunk({"time": 1, "latitude": 2, "longitude": 2})


def write_source(tmp_path):
    return write_zarr(source(), tmp_path / "weather.zarr")


def chunk_files(path, variable="temperature"):
    return sorted(item for item in (path / variable).rglob("*") if item.is_file() and not item.name.startswith("."))


def test_real_roundtrip_preserves_values_flags_strings_booleans_cf_and_sources(tmp_path):
    original = source()
    closed = []
    original.set_close(lambda: closed.append(True))
    path = write_zarr(original, tmp_path / "weather.zarr")
    assert closed == []
    assert (path / backend._COMPLETE).is_file()
    with open_zarr(path, chunks={"time": 1}) as result:
        assert all(isinstance(variable.data, da.Array) for variable in result.data_vars.values())
        actual = result.compute()
        for name in original.variables:
            assert actual[name].dims == original[name].dims
            np.testing.assert_equal(actual[name].data, original[name].compute().data)
        assert result.qc.dtype.kind in "iu"
        assert result.keep.dtype.kind == "b"
        assert result.station_id.isel(latitude=0).compute().item() == "0001"
        assert result.temperature.attrs["ancillary_variables"] == "qc"
        assert result.latitude.encoding["bounds"] == "latitude_bounds"
        events = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"]
        assert events[-1]["operation"] == "open_zarr"
        assert events[-1]["parameters"]["completion_verified"] is True
        assert events[-1]["inputs"][0]["name"] == "weather.zarr"
        assert str(tmp_path) not in result.attrs[PROVENANCE_ATTRIBUTE]
    assert not list(tmp_path.glob("*.partial"))
    original.close()
    assert closed == [True]


def test_open_builds_no_field_tasks_or_reads_and_preserves_native_chunks(tmp_path, monkeypatch):
    path = write_source(tmp_path)
    field_reads = []
    original = backend._ReadOnlyStore.__getitem__

    def read(self, key):
        if key.startswith("temperature/") and not Path(key).name.startswith("."):
            field_reads.append(key)
        return original(self, key)

    monkeypatch.setattr(backend._ReadOnlyStore, "__getitem__", read)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(True)):
        result = open_zarr(path, chunks={})
    try:
        assert not field_reads and not tasks
        assert result.temperature.chunks == ((1, 1, 1), (2,), (2, 2))
        result.temperature.isel(time=0, longitude=slice(0, 2)).compute()
        assert field_reads and len(set(field_reads)) == 1
    finally:
        result.close()


def test_source_integer_zero_and_floating_missingness_survive_packed_encoding(tmp_path):
    original = source()
    original.temperature.encoding.update(dtype="int16", scale_factor=0.25, _FillValue=-32768)
    original.temperature.attrs["actual_range"] = [0, 100]
    path = write_zarr(original, tmp_path / "packed.zarr", encoding={
        "temperature": {"dtype": "int16", "scale_factor": 0.25, "_FillValue": -32768, "chunks": (1, 2, 2)},
    })
    with open_zarr(path) as actual:
        np.testing.assert_allclose(actual.temperature.compute(), original.temperature.compute(), equal_nan=True)
        np.testing.assert_array_equal(actual.qc.compute(), original.qc.compute())
        assert "actual_range" not in actual.temperature.attrs
        assert actual.temperature.encoding["scale_factor"] == 0.25


def test_floating_integer_packing_requires_a_valid_explicit_fill_sentinel(tmp_path):
    for value in (None, np.nan, -32769, 1.25):
        with pytest.raises(ValueError, match="FillValue"):
            write_zarr(source(), tmp_path / "unsafe.zarr", encoding={
                "temperature": {"dtype": "int16", "_FillValue": value},
            })
    assert not (tmp_path / "unsafe.zarr").exists()


@pytest.mark.parametrize("calendar,dates", [
    ("noleap", [cftime.DatetimeNoLeap(2001, 2, 28), cftime.DatetimeNoLeap(2001, 3, 1)]),
    ("360_day", [cftime.Datetime360Day(2001, 2, 30), cftime.Datetime360Day(2001, 3, 1)]),
])
def test_nonstandard_calendar_and_time_bounds_roundtrip(tmp_path, calendar, dates):
    dataset = xr.Dataset({"v": ("time", [0., 1.]),
                          "time_bounds": (("time", "vertex"), [[dates[0], dates[0]], [dates[1], dates[1]]])},
                         coords={"time": dates})
    dataset.time.attrs["bounds"] = "time_bounds"
    dataset.time.encoding.update(units="days since 2000-01-01", calendar=calendar)
    dataset.time_bounds.encoding.update(units="days since 2000-01-01", calendar=calendar)
    path = write_zarr(dataset, tmp_path / "calendar.zarr")
    with open_zarr(path) as result:
        assert result.time.encoding["calendar"] == calendar
        assert result.time.encoding["bounds"] == "time_bounds"
        np.testing.assert_array_equal(result.time.data, dates)
        np.testing.assert_array_equal(result.time_bounds.compute(), dataset.time_bounds)


def test_external_store_requires_explicit_opt_in_and_stale_metadata_is_rejected(tmp_path):
    path = tmp_path / "external.zarr"
    source().to_zarr(path, consolidated=True, zarr_version=2)
    with pytest.raises(ValueError, match="completion marker"):
        open_zarr(path)
    with open_zarr(path, require_complete=False) as result:
        assert json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]["parameters"]["completion_verified"] is False
        np.testing.assert_array_equal(result.qc.compute(), source().qc.compute())
    attrs = path / "temperature/.zattrs"
    changed = json.loads(attrs.read_text())
    changed["units"] = "degree_Celsius"
    attrs.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="Consolidated"):
        open_zarr(path, require_complete=False)


def test_external_raw_cf_links_checked_before_decode_can_drop_them(tmp_path):
    dataset = source()
    dataset.latitude.attrs["bounds"] = "missing_bounds"
    path = tmp_path / "bad-cf.zarr"
    dataset.to_zarr(path, consolidated=True, zarr_version=2)
    with pytest.raises(ValueError, match="dangling"):
        open_zarr(path, require_complete=False)


@pytest.mark.parametrize("change", ["delete_chunk", "change_metadata", "change_size", "change_stamp"])
def test_marked_store_change_is_rejected_before_open(tmp_path, change):
    path = write_source(tmp_path)
    item = chunk_files(path)[0]
    if change == "delete_chunk":
        item.unlink()
    elif change == "change_metadata":
        item = path / "temperature/.zattrs"
        data = item.read_text().replace('"K"', '"C"')
        details = item.stat()
        item.write_text(data)
        os.utime(item, ns=(details.st_atime_ns, details.st_mtime_ns))
    elif change == "change_size":
        item.write_bytes(item.read_bytes() + b"x")
    else:
        details = item.stat()
        os.utime(item, ns=(details.st_atime_ns, details.st_mtime_ns + 10_000_000))
    with pytest.raises(ValueError, match="changed or are missing"):
        open_zarr(path)
    with pytest.raises(ValueError, match="changed or are missing"):
        open_zarr(path, require_complete=False)


@pytest.mark.parametrize("change", ["delete", "modify"])
def test_lazy_access_rejects_missing_or_changed_expected_chunk_instead_of_fill(tmp_path, change):
    path = write_source(tmp_path)
    with open_zarr(path) as dataset:
        item = chunk_files(path)[0]
        if change == "delete":
            item.unlink()
        else:
            item.write_bytes(item.read_bytes() + b"x")
        with pytest.raises(ValueError, match="unavailable|changed"):
            dataset.temperature.compute()


def test_all_missing_chunks_are_written_and_missing_one_is_detected(tmp_path):
    dataset = xr.Dataset({"v": ("x", np.full(4, np.nan))}, coords={"x": [0, 1, 2, 3]}).chunk({"x": 2})
    path = write_zarr(dataset, tmp_path / "missing.zarr")
    chunks = chunk_files(path, "v")
    assert len(chunks) == 2
    chunks[0].unlink()
    with pytest.raises(ValueError, match="missing"):
        open_zarr(path)


def test_completed_store_cannot_opt_out_of_invalid_manifest(tmp_path):
    path = write_source(tmp_path)
    marker = path / backend._COMPLETE
    data = json.loads(marker.read_text())
    data["schema_version"] = 99
    marker.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="completion manifest"):
        open_zarr(path, require_complete=False)


@pytest.mark.parametrize("chunks", [None, False, 0, {"time": 0}, {"unknown": 1}])
def test_invalid_chunks_are_rejected(tmp_path, chunks):
    path = write_source(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        open_zarr(path, chunks=chunks)


def test_v3_and_nested_groups_are_rejected(tmp_path):
    path = tmp_path / "v3.zarr"
    path.mkdir()
    (path / "zarr.json").write_text('{"zarr_format": 3, "node_type": "group"}')
    with pytest.raises(ValueError, match="v3"):
        open_zarr(path, require_complete=False)
    nested = tmp_path / "nested.zarr"
    source().to_zarr(nested, group="nested", consolidated=True, zarr_version=2)
    with pytest.raises(ValueError, match="Nested"):
        open_zarr(nested, require_complete=False)


@pytest.mark.parametrize("target", ["s3://bucket/store", "https://example.org/store", "zip::local.zip"])
def test_nonlocal_stores_rejected(target):
    with pytest.raises(ValueError, match="local"):
        open_zarr(target)
    with pytest.raises(ValueError, match="local"):
        write_zarr(source(), target)


def test_unsafe_chunk_overlap_and_fill_chunk_omission_rejected(tmp_path):
    with pytest.raises((ValueError, NotImplementedError), match="chunk"):
        write_zarr(source(), tmp_path / "overlap.zarr", encoding={"temperature": {"chunks": (2, 2, 4)}})
    with pytest.raises(ValueError, match="write_empty_chunks"):
        write_zarr(source(), tmp_path / "omit.zarr", encoding={"temperature": {"write_empty_chunks": False}})
    assert list(tmp_path.iterdir()) == []


def test_existing_output_never_evaluates_input_or_overwrites(tmp_path):
    path = write_source(tmp_path)
    marker = (path / backend._COMPLETE).read_bytes()

    @delayed
    def forbidden():
        raise AssertionError("field must not execute")

    dataset = xr.Dataset({"v": ("x", da.from_delayed(forbidden(), shape=(2,), dtype=float))})
    with pytest.raises(FileExistsError):
        write_zarr(dataset, path)
    assert (path / backend._COMPLETE).read_bytes() == marker


def test_failure_during_computation_removes_only_owned_stage(tmp_path):
    untouched = tmp_path / "unrelated.txt"
    untouched.write_text("retain")

    @delayed
    def failure():
        raise RuntimeError("deliberate field failure")

    dataset = xr.Dataset({"v": ("x", da.from_delayed(failure(), shape=(2,), dtype=float))})
    with pytest.raises(RuntimeError, match="field failure"):
        write_zarr(dataset, tmp_path / "bad.zarr")
    assert list(tmp_path.iterdir()) == [untouched]
    assert untouched.read_text() == "retain"


def test_failure_during_publication_cleans_owned_target(tmp_path, monkeypatch):
    link = backend.os.link
    calls = []

    def fail_after_one(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError("deliberate link failure")
        return link(src, dst)

    monkeypatch.setattr(backend.os, "link", fail_after_one)
    with pytest.raises(OSError, match="link failure"):
        write_zarr(source(), tmp_path / "bad.zarr")
    assert list(tmp_path.iterdir()) == []


def test_competing_writer_target_is_preserved(tmp_path, monkeypatch):
    publish = backend._publish

    def competitor(stage, destination, files):
        destination.mkdir()
        (destination / "other_writer.txt").write_text("keep")
        return publish(stage, destination, files)

    monkeypatch.setattr(backend, "_publish", competitor)
    with pytest.raises(FileExistsError):
        write_zarr(source(), tmp_path / "race.zarr")
    assert (tmp_path / "race.zarr/other_writer.txt").read_text() == "keep"
    assert list(tmp_path.iterdir()) == [tmp_path / "race.zarr"]


def test_foreign_file_during_failed_publication_prevents_destructive_cleanup(tmp_path, monkeypatch):
    link = backend.os.link
    calls = []

    def inject(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            (tmp_path / "race.zarr/other_writer.txt").write_text("keep")
            raise OSError("deliberate publication interruption")
        return link(src, dst)

    monkeypatch.setattr(backend.os, "link", inject)
    with pytest.raises(OSError, match="publication interruption"):
        write_zarr(source(), tmp_path / "race.zarr")
    assert (tmp_path / "race.zarr/other_writer.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="completion marker"):
        open_zarr(tmp_path / "race.zarr")


def test_completion_marker_is_last_and_partial_target_is_not_readable(tmp_path, monkeypatch):
    link = backend.os.link
    published = []

    def observe(src, dst):
        if Path(dst).name == backend._COMPLETE:
            with pytest.raises(ValueError, match="completion marker"):
                open_zarr(tmp_path / "completed.zarr")
        published.append(Path(dst).name)
        return link(src, dst)

    monkeypatch.setattr(backend.os, "link", observe)
    path = write_zarr(source(), tmp_path / "completed.zarr")
    assert published[-1] == backend._COMPLETE
    with open_zarr(path) as result:
        np.testing.assert_array_equal(result.qc.compute(), source().qc.compute())


def test_cached_cf_metadata_change_is_rejected_during_field_read(tmp_path):
    path = write_source(tmp_path)
    with open_zarr(path) as dataset:
        attrs = path / "temperature/.zattrs"
        details = attrs.stat()
        attrs.write_text(attrs.read_text().replace('"K"', '"C"'))
        os.utime(attrs, ns=(details.st_atime_ns, details.st_mtime_ns))
        with pytest.raises(ValueError, match="metadata changed"):
            dataset.temperature.compute()


def test_replaced_publication_target_never_receives_completion_marker(tmp_path, monkeypatch):
    link = backend.os.link
    changed = []
    destination = tmp_path / "race.zarr"

    def replace_once(src, dst):
        if not changed:
            changed.append(True)
            destination.rename(tmp_path / "moved-original-reservation")
            destination.mkdir()
            for name in source().variables:
                (destination / name).mkdir()
            (destination / "foreign.txt").write_text("retain")
        return link(src, dst)

    monkeypatch.setattr(backend.os, "link", replace_once)
    with pytest.raises(ValueError, match="target changed"):
        write_zarr(source(), destination)
    assert (destination / "foreign.txt").read_text() == "retain"
    assert not (destination / backend._COMPLETE).exists()


def test_foreign_empty_directory_is_preserved_on_publication_failure(tmp_path, monkeypatch):
    link = backend.os.link
    calls = []

    def inject(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            (tmp_path / "race.zarr/foreign-empty-directory").mkdir()
            raise OSError("deliberate publication interruption")
        return link(src, dst)

    monkeypatch.setattr(backend.os, "link", inject)
    with pytest.raises(OSError):
        write_zarr(source(), tmp_path / "race.zarr")
    assert (tmp_path / "race.zarr/foreign-empty-directory").is_dir()
    assert not (tmp_path / "race.zarr" / backend._COMPLETE).exists()


def test_missing_dependency_produces_actionable_error_without_creating_output(tmp_path, monkeypatch):
    original = backend.importlib.import_module

    def missing(name):
        if name == "zarr":
            raise ImportError("deliberately unavailable")
        return original(name)

    monkeypatch.setattr(backend.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"climate-preprocessing\[zarr\]"):
        write_zarr(source(), tmp_path / "new.zarr")
    assert not (tmp_path / "new.zarr").exists()


def test_linked_input_or_output_is_rejected_when_platform_allows_links(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("This Windows account cannot create symbolic links")
    with pytest.raises(ValueError, match="links"):
        open_zarr(linked)
    with pytest.raises(ValueError, match="links"):
        write_zarr(source(), linked / "new.zarr")
