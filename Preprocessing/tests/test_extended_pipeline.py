"""Cross-operation and configuration contracts for P2/P3 additions."""

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
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE


def example(directory):
    path = Path(__file__).resolve().parents[1] / "examples/pressure_time_mask.py"
    spec = importlib.util.spec_from_file_location("extended_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_example(directory)


def test_pressure_time_quality_mask_roundtrip(tmp_path):
    output = example(tmp_path / "example")
    with xr.open_dataset(output, engine="netcdf4") as result:
        np.testing.assert_array_equal(result.pressure, [500, 1000])
        assert result.pressure.attrs["units"] == "hPa"
        assert result.sizes["time"] == 2
        expected = np.arange(36, dtype=float).reshape(3, 2, 2, 3)[1:, ::-1, ::-1, :] + 250
        keep = np.array([[True, True, False], [True, False, True]])
        expected = np.where(keep, expected, np.nan)
        np.testing.assert_array_equal(result.temperature, expected)
        np.testing.assert_array_equal(result.quality, [[0, 0, 1], [0, 1, 0]])
        assert result.temperature.attrs["ancillary_variables"] == "quality"
        operations = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"]
        assert {item["operation"] for item in operations} >= {
            "normalize_pressure", "subset_time", "apply_mask", "prepare_dataset"}


def test_cli_new_options_inspect_without_field_tasks_and_complete_input_check(tmp_path):
    example(tmp_path / "example")
    config = tmp_path / "config.json"
    payload = {"schema_version": 1, "input": "example/synthetic.nc", "chunks": {"time": 1},
               "pressure": {"pressure": "level", "target_units": "Pa", "order": "descending"},
               "expected_time": ["2000-01-01", "2000-02-01", "2000-03-01"],
               "time_range": {"start": "2000-02-01", "end": "2000-03-01"}}
    config.write_text(json.dumps(payload), encoding="utf-8")
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(True)), redirect_stdout(io.StringIO()) as output:
        assert main(["inspect", str(config)]) == 0
    assert not tasks
    assert json.loads(output.getvalue())["dimensions"] == {
        "time": 2, "pressure": 2, "latitude": 2, "longitude": 3}
    payload["expected_time"] = payload["expected_time"][1:]
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        main(["inspect", str(config)])


def test_expected_time_failure_closes_source(monkeypatch):
    dataset = xr.Dataset(coords={"time": np.array(["2000-01-01"], dtype="datetime64[ns]")})
    closed = []
    dataset.set_close(lambda: closed.append(True))
    monkeypatch.setattr("climate_preprocessing.pipeline.open_dataset", lambda *a, **kw: dataset)
    with pytest.raises(ValueError):
        prepare_dataset("unused.nc", expected_time=["2000-02-01"])
    assert closed == [True]


@pytest.mark.parametrize("change", [
    {"pressure": []}, {"pressure": {"unknown": "level"}}, {"time_range": []},
    {"time_range": {"stop": "2000-01-01"}}, {"expected_time": []},
    {"expected_time": "2000-01-01"}, {"expected_time": [1]},
])
def test_new_config_shape_rejected_without_io(tmp_path, change):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "input": "missing.nc", **change}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepare_config(path)


def test_expected_summary_uses_decoded_cftime_labels_without_field_reads(monkeypatch):
    import hashlib

    import cftime
    import dask.array as da
    from dask import delayed

    dates = [cftime.DatetimeNoLeap(2001, month, 1) for month in (1, 2, 3)]
    field_reads, closed = [], []

    @delayed
    def read_field():
        field_reads.append(True)
        return np.arange(12, dtype=float).reshape(3, 2, 2)

    source = xr.Dataset(
        {"V": (("record", "lat", "lon"), da.from_delayed(read_field(), shape=(3, 2, 2), dtype=float),
               {"units": "m s-1"})},
        coords={"valid_time": ("record", da.from_array(np.array(dates, dtype=object), chunks=1)),
                "lat": ("lat", [-10., 10.], {"units": "degrees_north"}),
                "lon": ("lon", [0., 180.], {"units": "degrees_east"})},
    )
    assert "calendar" not in source.valid_time.attrs
    assert "calendar" not in source.valid_time.encoding
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr("climate_preprocessing.pipeline.open_dataset", lambda *args, **kwargs: source)
    with prepare_dataset(
        "unused.nc", time="valid_time", coordinates={"latitude": "lat", "longitude": "lon"},
        expected_time=["2001-01-01", "2001-02-01", "2001-03-01"],
        time_range={"start": "2001-02-01", "end": "2001-03-01"},
    ) as result:
        event = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]
        summary = event["parameters"]["expected_time"]
        labels = [str(value) for value in dates]
        assert summary["calendar"] == "noleap"
        assert summary["count"] == 3  # complete input is checked before the two-record selection
        assert summary["first"] == labels[0]
        assert summary["last"] == labels[-1]
        assert summary["sha256"] == hashlib.sha256(json.dumps(labels, separators=(",", ":")).encode()).hexdigest()
        assert result.sizes["record"] == 2
        assert isinstance(result.V.data, da.Array)
        assert field_reads == []
        assert closed == []
    assert closed == [True]
