"""Real optional backend through preparation, JSON and output boundaries."""

from contextlib import redirect_stdout
import io
import json

from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import prepare_dataset, rectilinear_grid, regrid_dataset, write_dataset, open_dataset
from climate_preprocessing.cli import main
from climate_preprocessing.regrid import _xesmf

pytestmark = pytest.mark.regrid


@pytest.fixture(scope="module", autouse=True)
def real_backend():
    try:
        _xesmf.backend_versions()
    except ImportError as error:
        pytest.skip(f"Real native backend unavailable: {error}")


def input_file(tmp_path):
    source = rectilinear_grid([0, 1, 2], [0, 1, 2],
                             latitude_bounds=[-0.5, 0.5, 1.5, 2.5], longitude_bounds=[-0.5, 0.5, 1.5, 2.5])
    source = source.assign_coords(time=np.array(["2000-01-01"], dtype="datetime64[ns]"))
    source["t"] = (("time", "latitude", "longitude"), np.ones((1, 3, 3)) * 7, {"units": "K"})
    path = tmp_path / "input.nc"
    source.to_netcdf(path, engine="netcdf4")
    return path


def test_prepare_composes_conservative_lazy_bounds(tmp_path):
    path = input_file(tmp_path)
    with prepare_dataset(path, regrid={
        "method": "conservative_normed", "variables": ["t"],
        "target_grid": {"latitude": [0.5, 1.5], "longitude": [0.5, 1.5],
                        "latitude_bounds": [0, 1, 2], "longitude_bounds": [0, 1, 2]},
    }) as result:
        assert result.t.chunks is not None
        np.testing.assert_allclose(result.t.compute(), 7)
        assert result.latitude.attrs["bounds"] == "latitude_bounds"
        event = json.loads(result.attrs["climate_preprocessing_provenance"])["operations"][-1]
        assert event["parameters"]["regrid_requested"] is True


def test_cli_inspect_no_cache_write_prepare_roundtrips(tmp_path):
    input_file(tmp_path)
    config = {"schema_version": 1, "input": "input.nc", "output": "output.nc", "regrid": {
        "method": "bilinear", "variables": ["t"], "weights_dir": "weights",
        "target_grid": {"latitude": [0.5, 1.5], "longitude": [0.5, 1.5]},
    }}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with redirect_stdout(io.StringIO()) as stream:
        assert main(["inspect", str(path)]) == 0
    summary = json.loads(stream.getvalue())
    assert summary["field_values_read"] is False and summary["outputs_written"] is False
    assert summary["regrid_geometry_evaluated"] is True
    assert not (tmp_path / "weights").exists() and not (tmp_path / "output.nc").exists()
    with redirect_stdout(io.StringIO()):
        assert main(["prepare", str(path)]) == 0
    assert len(list((tmp_path / "weights").glob("*.nc"))) == 1
    with open_dataset(tmp_path / "output.nc") as result:
        np.testing.assert_allclose(result.t.compute(), 7)
        assert result.t.attrs["units"] == "K"


def test_decoded_cf_encoding_references_survive_actual_regrid(tmp_path):
    path = input_file(tmp_path)
    with xr.open_dataset(path, decode_coords="all", chunks={"time": 1}) as original:
        original["instrument"] = xr.DataArray(42.0, attrs={"long_name": "nonspatial calibration reference"})
        original.t.encoding["ancillary_variables"] = "instrument"
        original.t.encoding["coordinates"] = "time latitude longitude"
        target = rectilinear_grid([0.5, 1.5], [0.5, 1.5])
        result = regrid_dataset(original, target, variables="t", method="bilinear")
        assert result.t.attrs["ancillary_variables"] == "instrument"
        assert result.t.attrs["coordinates"] == "time latitude longitude"
        assert result.t.encoding == {}
        write_dataset(result, tmp_path / "decoded.nc")
    with open_dataset(tmp_path / "decoded.nc") as output:
        np.testing.assert_allclose(output.t.compute(), 7)


def test_nearest_same_shape_different_grids_do_not_reuse_cache(tmp_path):
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    source["t"] = (("latitude", "longitude"), np.arange(9.).reshape(3, 3), {"units": "K"})
    first = rectilinear_grid([0.1, 1.1], [0.1, 1.1])
    second = rectilinear_grid([0.9, 1.9], [0.9, 1.9])
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        one = regrid_dataset(source, first, variables="t", method="nearest_s2d", weights_dir=tmp_path / "weights")
        two = regrid_dataset(source, second, variables="t", method="nearest_s2d", weights_dir=tmp_path / "weights")
    assert tasks == []
    np.testing.assert_array_equal(one.t.compute(), [[0, 1], [3, 4]])
    np.testing.assert_array_equal(two.t.compute(), [[4, 5], [7, 8]])
    assert len(list((tmp_path / "weights").glob("*.nc"))) == 2
