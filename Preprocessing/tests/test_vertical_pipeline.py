"""Optional real vertical backend through preparation, files and CLI."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import importlib.util

from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import open_dataset, prepare_dataset, write_dataset
from climate_preprocessing.cli import main
from climate_preprocessing.vertical import _metpy
from test_vertical_contract import fixed_dataset

pytestmark = pytest.mark.vertical


def geometry_only(task_graphs):
    # xarray may execute small first/last time-bounds slices while determining
    # its decoded dtype. Those are permitted coordinate reads, not field scans.
    return all(all("time_bounds" in str(key) for key in graph
                   if str(key).startswith("original-open_dataset-")) for graph in task_graphs)


@pytest.fixture(scope="module", autouse=True)
def real_backend():
    try:
        _metpy.backend_versions()
    except ImportError as error:
        pytest.skip(str(error))


def test_prepare_fixed_input_retains_time_and_no_field_tasks(tmp_path):
    source = fixed_dataset()
    source.t.attrs.pop("cell_methods")
    write_dataset(source, tmp_path / "input.nc")
    tasks = []
    with Callback(start=lambda graph: tasks.append(tuple(graph))):
        result = prepare_dataset(tmp_path / "input.nc", vertical={
            "target_levels": [900, 700, 300], "variables": ["t"], "method": "linear", "surface_pressure": "ps",
        })
    try:
        assert geometry_only(tasks)
        actual = result.t.compute().to_numpy()
        assert np.isnan(actual[:, [0, 2]]).all()  # no support below valid bottom center or above top center
        np.testing.assert_array_equal(actual[:, 1], 1)
        np.testing.assert_array_equal(result.pressure, [900, 700, 300])
        # CF decode may attach the independent scalar CRS as a coordinate;
        # compare the time variable's actual labels/dimensions, not that view.
        xr.testing.assert_equal(result.time.variable, source.time.variable)
    finally:
        result.close()


def test_cli_inspect_then_prepare_and_output_no_overwrite(tmp_path):
    write_dataset(fixed_dataset(), tmp_path / "input.nc")
    config = {"schema_version": 1, "input": "input.nc", "output": "output.nc", "vertical": {
        "target_levels": [900, 700], "target_units": "hPa", "variables": ["t"], "method": "log",
    }}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    tasks = []
    with Callback(start=lambda graph: tasks.append(tuple(graph))), redirect_stdout(io.StringIO()) as output:
        assert main(["inspect", str(path)]) == 0
    summary = json.loads(output.getvalue())
    assert summary["vertical_field_graph_built"] and not summary["field_values_read"] and geometry_only(tasks)
    assert not (tmp_path / "output.nc").exists()
    with redirect_stdout(io.StringIO()):
        assert main(["prepare", str(path)]) == 0
    with open_dataset(tmp_path / "output.nc") as result:
        np.testing.assert_allclose(result.t.compute(), 1)
        assert result.pressure.attrs["units"] == "hPa"
        assert result.t.attrs["cell_methods"] == "time: mean area: mean"
    with pytest.raises(FileExistsError):
        main(["prepare", str(path)])


def test_real_hybrid_example_and_generated_configuration(tmp_path):
    path = Path(__file__).parents[1] / "examples/vertical_pressure.py"
    spec = importlib.util.spec_from_file_location("example_vertical_pressure", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = tmp_path / "demo"
    result_path = module.run_example(demo)
    with open_dataset(result_path) as result:
        expected = xr.DataArray([250., 270., 300.], dims="pressure")
        np.testing.assert_allclose(result.temperature.mean(("time", "latitude", "longitude")).compute(), expected)
        assert not {"lev", "a", "b", "p0"}.intersection(result.variables)
        assert "ps" in result
    with redirect_stdout(io.StringIO()):
        assert main(["inspect", str(demo / "prepare.json")]) == 0
        assert main(["prepare", str(demo / "prepare.json")]) == 0
    with open_dataset(demo / "cli_pressure.nc") as result:
        np.testing.assert_allclose(result.temperature.isel(time=0, latitude=0, longitude=0).compute(), expected)


def test_pipeline_closes_source_on_vertical_failure(monkeypatch):
    source = fixed_dataset()
    closed = []
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr("climate_preprocessing.pipeline.open_dataset", lambda *args, **kwargs: source)
    with pytest.raises(ValueError, match="vertical dimension"):
        prepare_dataset("unused", vertical={"target_levels": [700], "variables": ["ps"], "method": "linear"})
    assert closed == [True]


def test_pipeline_does_not_mask_invalid_surface_geometry_before_validation(tmp_path):
    source = fixed_dataset()
    source["ps"] = xr.full_like(source.ps, np.inf)
    source.ps.attrs = {"units": "hPa", "standard_name": "surface_air_pressure"}
    write_dataset(source, tmp_path / "invalid.nc")
    with prepare_dataset(tmp_path / "invalid.nc", vertical={
        "target_levels": [700], "variables": ["t"], "method": "linear", "surface_pressure": "ps",
    }) as result:
        with pytest.raises(ValueError, match="Surface pressure"):
            result.t.compute()


@pytest.mark.regrid
def test_hybrid_vertical_runs_before_horizontal_regridding(tmp_path):
    from climate_preprocessing.regrid import _xesmf

    try:
        _xesmf.backend_versions()
    except ImportError as error:
        pytest.skip(str(error))
    path = Path(__file__).parents[1] / "examples/vertical_pressure.py"
    spec = importlib.util.spec_from_file_location("example_vertical_combined", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    demo = tmp_path / "demo"
    module.run_example(demo)
    with prepare_dataset(demo / "hybrid.nc", vertical={
        "target_levels": [300, 500], "variables": ["temperature"], "method": "linear",
    }, regrid={"variables": ["temperature"], "method": "bilinear",
               "target_grid": {"latitude": [2, 8], "longitude": [102, 108]}}) as result:
        np.testing.assert_allclose(result.temperature.isel(time=0, latitude=0, longitude=0).compute(), [250, 270])
        operations = [item["operation"] for item in json.loads(result.attrs["climate_preprocessing_provenance"])["operations"]]
        assert operations.index("interpolate_pressure") < operations.index("regrid_dataset")
