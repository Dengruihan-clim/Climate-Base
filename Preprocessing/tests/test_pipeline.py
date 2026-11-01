"""User-visible lazy workflow, CLI and configuration integration contracts."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path

from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import prepare_dataset
from climate_preprocessing.cli import main
from climate_preprocessing.config import load_prepare_config
from test_archive import make_file


def test_prepare_graph_then_explicit_compute(tmp_path):
    path = make_file(tmp_path / "source.nc", units="degree_Celsius", values=1.)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        dataset = prepare_dataset(
            path, coordinates={"latitude": "lat", "longitude": "lon"},
            chunks={"time": 1}, region={"longitude_bounds": [270, 90]},
            units={"temperature": "K"},
        )
        try:
            assert dataset.temperature.chunks is not None
            assert not tasks
            np.testing.assert_array_equal(dataset.longitude, [0, 90, 270])
            np.testing.assert_allclose(dataset.temperature.compute(), 274.15)
            assert tasks
        finally:
            dataset.close()


def test_pipeline_closes_source_on_policy_failure(monkeypatch):
    source = xr.Dataset(
        {"v": (("time", "lat", "lon"), np.ones((1, 2, 2)))},
        coords={"time": np.array(["2000-01-01"], dtype="datetime64[ns]"),
                "lat": [0., 10.], "lon": [0., 90.]},
    ).chunk()
    closed = []
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr("climate_preprocessing.pipeline.open_dataset", lambda *a, **kw: source)
    with pytest.raises(ValueError, match="units"):
        prepare_dataset("unused.nc", coordinates={"latitude": "lat", "longitude": "lon"})
    assert closed == [True]


@pytest.mark.parametrize("changes", [
    {"schema_version": True}, {"schema_version": 2}, {"unknown": 1},
    {"input": []}, {"input": ["a.nc", "a.nc"]}, {"output": 123},
    {"metadata_policy": "override"}, {"metadata_policy": None}, {"metadata_policy": True},
])
def test_invalid_config_is_rejected_without_io(tmp_path, changes):
    value = {"schema_version": 1, "input": "missing.nc", **changes}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepare_config(path)


def test_example_cli_inspect_and_prepare_roundtrip(tmp_path):
    path = Path(__file__).resolve().parents[1] / "examples/netcdf_prepare.py"
    spec = importlib.util.spec_from_file_location("lazy_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = module.create_demo(tmp_path / "demo")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)), redirect_stdout(io.StringIO()) as stream:
        assert main(["inspect", str(config)]) == 0
    report = json.loads(stream.getvalue())
    assert not tasks
    assert report["field_values_read"] is False
    assert not (config.parent / "prepared.nc").exists()
    with redirect_stdout(io.StringIO()):
        assert main(["prepare", str(config)]) == 0
    with xr.open_dataset(config.parent / "prepared.nc") as result:
        assert result.temperature.attrs["units"] == "K"
        assert result.temperature.shape == (2, 3, 3)
        assert "prepare_dataset" in str(result.attrs)
        expected = (np.arange(24).reshape(2, 3, 4)[:, ::-1, :])[:, :, [2, 3, 1]] + 273.15
        np.testing.assert_allclose(result.temperature, expected)
    with pytest.raises(FileExistsError):
        main(["prepare", str(config)])


def test_cli_preserves_per_file_metadata_without_computing_inspect(tmp_path):
    from netCDF4 import Dataset
    from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE

    first = make_file(tmp_path / "early.nc", dates=[(2000, 1)], units="K", values=1.)
    second = make_file(tmp_path / "late.nc", dates=[(2001, 1)], units="K", values=2.)
    for path, history in ((first, "converted early"), (second, "converted late")):
        with Dataset(path, "a") as source:
            source.history = history
    config = tmp_path / "prepare.json"
    config.write_text(json.dumps({
        "schema_version": 1, "input": [second.name, first.name], "output": "prepared.nc",
        "coordinates": {"latitude": "lat", "longitude": "lon"},
        "metadata_policy": "preserve_sources", "chunks": {"time": 1},
    }), encoding="utf-8")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)), redirect_stdout(io.StringIO()):
        assert main(["inspect", str(config)]) == 0
    assert not tasks
    with redirect_stdout(io.StringIO()):
        assert main(["prepare", str(config)]) == 0
    with xr.open_dataset(tmp_path / "prepared.nc") as result:
        assert result.temperature.attrs["units"] == "K"
        np.testing.assert_allclose(result.temperature.isel(time=0), 1.)
        np.testing.assert_allclose(result.temperature.isel(time=1), 2.)
        assert "history" not in result.attrs
        events = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"]
        event = next(item for item in events if item["operation"] == "open_mfdataset")
        assert {item["source_metadata"]["global_attributes"]["history"] for item in event["inputs"]} == {
            "converted early", "converted late"}
        prepared = next(item for item in events if item["operation"] == "prepare_dataset")
        assert prepared["parameters"]["metadata_policy"] == "preserve_sources"
