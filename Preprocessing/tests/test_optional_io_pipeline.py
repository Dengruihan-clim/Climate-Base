"""Explicit optional-format orchestration with real backends, when installed."""

from contextlib import redirect_stdout
import io
import json
import os
import subprocess
import sys

from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import prepare_dataset
from climate_preprocessing.cli import main
from climate_preprocessing.config import load_prepare_config


def test_core_import_does_not_import_optional_backends():
    script = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'zarr', 'numcodecs', 'cfgrib', 'eccodes', 'xesmf', 'esmpy', 'ESMF', 'metpy', 'geocat'}:
            raise AssertionError('Core import attempted optional backend: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
import climate_preprocessing
from climate_preprocessing._version import __version__
assert climate_preprocessing.__version__ == __version__
"""
    subprocess.run([sys.executable, "-c", script], env=os.environ.copy(), check=True, timeout=60)


@pytest.mark.parametrize("changes", [
    {"input_format": "auto"}, {"input_format": True}, {"output_format": "grib"},
    {"output_format": None}, {"input_options": {"indexpath": "cache.idx"}},
    {"input_format": "grib", "input_options": {"errors": "ignore"}},
    {"input_format": "zarr", "input_options": {"consolidated": False}},
])
def test_config_rejects_backend_overrides_before_io(tmp_path, changes):
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps({"schema_version": 1, "input": "absent", **changes}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepare_config(path)


@pytest.mark.parametrize("options", [
    {"input_format": "auto"}, {"input_format": "grib", "input_options": {"indexpath": "x"}},
    {"input_format": "zarr", "metadata_policy": "preserve_sources"},
])
def test_prepare_rejects_invalid_backend_policy_before_io(options):
    with pytest.raises(ValueError):
        prepare_dataset("absent", **options)


@pytest.mark.parametrize("format", ["grib", "zarr"])
def test_optional_multiple_inputs_need_explicit_external_selection(format):
    with pytest.raises(ValueError, match="exactly one"):
        prepare_dataset(["absent1", "absent2"], input_format=format)


def test_netcdf_to_zarr_cli_then_lazy_zarr_inspect(tmp_path):
    pytest.importorskip("zarr")
    from climate_preprocessing.io.zarr import open_zarr

    source = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), np.arange(12.).reshape(2, 2, 3), {"units": "degree_Celsius"})},
        coords={"time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
                "lat": ("lat", [10., 0.], {"standard_name": "latitude", "units": "degrees_north"}),
                "lon": ("lon", [0., 90., 180.], {"standard_name": "longitude", "units": "degrees_east"})},
    )
    source.to_netcdf(tmp_path / "source.nc", engine="netcdf4")
    config = tmp_path / "prepare.json"
    config.write_text(json.dumps({"schema_version": 1, "input": "source.nc", "output": "result.zarr",
                                  "output_format": "zarr", "chunks": {"time": 1},
                                  "units": {"temperature": "K"}}), encoding="utf-8")
    with redirect_stdout(io.StringIO()) as output:
        assert main(["prepare", str(config)]) == 0
    assert json.loads(output.getvalue())["output_format"] == "zarr"
    with open_zarr(tmp_path / "result.zarr", chunks={}) as result:
        np.testing.assert_allclose(result.temperature.compute(), source.temperature.data[:, ::-1] + 273.15)
    config.write_text(json.dumps({"schema_version": 1, "input": "result.zarr", "input_format": "zarr",
                                  "expected_time": ["2000-01-01", "2000-02-01"],
                                  "time_range": {"start": "2000-02-01"}}), encoding="utf-8")
    tasks = []
    with Callback(pretask=lambda *a: tasks.append(True)), redirect_stdout(io.StringIO()) as output:
        assert main(["inspect", str(config)]) == 0
    assert not tasks
    assert json.loads(output.getvalue())["dimensions"]["time"] == 1


def test_grib_prepare_and_zarr_roundtrip_preserve_forecast_axes(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    pytest.importorskip("cfgrib")
    pytest.importorskip("zarr")
    from test_grib import write_grib
    from climate_preprocessing.io.zarr import open_zarr, write_zarr

    path = tmp_path / "forecast.grib2"
    truth = write_grib(path, eccodes, [
        {"step": 6, "dataDate": 20000101, "values": np.arange(6.) + 250},
        {"step": 6, "dataDate": 20000102, "values": np.arange(6.) + 260},
    ])
    tasks = []
    with Callback(pretask=lambda *a: tasks.append(True)):
        result = prepare_dataset(
            path, input_format="grib", input_options={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
            pressure={"pressure": "isobaricInhPa", "target_units": "Pa"},
            expected_time=["2000-01-01", "2000-01-02"],
        )
    try:
        assert not tasks
        assert result.sizes["step"] == 1 and result.sizes["time"] == 2
        assert result.valid_time.dims == ("time", "step")
        np.testing.assert_array_equal(result.pressure, [50000.])
        expected = np.stack(truth).reshape(2, 2, 3)[:, ::-1]
        np.testing.assert_allclose(result.t.squeeze(("pressure", "step")).compute(), expected)
        write_zarr(result, tmp_path / "forecast.zarr")
        with open_zarr(tmp_path / "forecast.zarr") as reopened:
            np.testing.assert_allclose(reopened.t.squeeze(("pressure", "step")).compute(), expected)
            np.testing.assert_array_equal(reopened.valid_time, result.valid_time)
            np.testing.assert_array_equal(reopened.step, result.step)
        assert not list(tmp_path.glob("*.idx"))
    finally:
        result.close()
