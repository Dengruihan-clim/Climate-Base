"""Backend-independent P5 selection, metadata and configuration contracts."""

import json

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import rectilinear_grid, regrid_dataset
from climate_preprocessing.config import load_prepare_config
from climate_preprocessing.regrid.api import _assemble_result, _without_horizontal_methods, validate_regrid_options


def dataset():
    grid = rectilinear_grid([0, 10, 20], [0, 10, 20])
    grid["t"] = (("time", "latitude", "longitude"), da.ones((2, 3, 3), chunks=(1, 2, 2)),
                 {"units": "K", "cell_methods": "time: mean", "ancillary_variables": "qc",
                  "cell_measures": "area: cell_area", "grid_mapping": "crs"})
    grid["qc"] = (("latitude", "longitude"), np.ones((3, 3), dtype="i1"), {"flag_meanings": "good"})
    grid["cell_area"] = (("latitude", "longitude"), np.ones((3, 3)))
    grid["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    grid = grid.assign_coords(time=np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"))
    grid.time.attrs["bounds"] = "time_bounds"
    grid["time_bounds"] = (("time", "nv"), np.array(
        [["1999-12-15", "2000-01-15"], ["2000-01-15", "2000-02-15"]], dtype="datetime64[ns]"))
    return grid


@pytest.mark.parametrize("options,match", [
    ({"method": "magic"}, "method"),
    ({"method": []}, "method"),
    ({"periodic": 1}, "periodic"),
    ({"quantity": "extensive"}, "intensive"),
    ({"weights_dir": "s3://bucket/weights"}, "local directory"),
    ({"weights_dir": ""}, "local directory"),
    ({"method": "conservative", "periodic": True}, "periodic"),
    ({"variables": []}, "nonempty"),
    ({"variables": ["qc"]}, "QC"),
    ({"variables": ["cell_area"]}, "structural"),
    ({"variables": ["time_bounds"]}, "structural"),
    ({"variables": ["unknown"]}, "data variable"),
])
def test_invalid_request_fails_before_optional_backend(options, match):
    kwargs = {"method": "bilinear", "variables": ["t"], **options}
    with pytest.raises((ValueError, TypeError), match=match):
        regrid_dataset(dataset(), rectilinear_grid([2, 5], [2, 5]), **kwargs)


def test_integer_field_requires_explicit_physical_conversion():
    source = dataset()
    source["t"] = source.t.astype("int64")
    with pytest.raises(ValueError, match="floating-point"):
        regrid_dataset(source, rectilinear_grid([2, 5], [2, 5]), variables=["t"], method="bilinear")


def test_vector_rotation_not_implicitly_assumed():
    source = dataset()
    source.t.attrs["standard_name"] = "grid_eastward_wind"
    with pytest.raises(ValueError, match="rotation"):
        regrid_dataset(source, rectilinear_grid([2, 5], [2, 5]), variables=["t"], method="bilinear")


def test_metadata_cleanup_does_not_mutate_input_or_evaluate_fields():
    source = dataset()
    source.attrs["geospatial_lat_min"] = 0.0
    source.t.attrs.update(actual_range=[1, 1], valid_min=0, cell_methods="time: mean (interval: 1 month) area: mean")
    source.t.encoding.update(dtype="int16", scale_factor=0.1, coordinates="time latitude longitude")
    before = source.copy(deep=True)
    target = rectilinear_grid([2, 5], [2, 5], latitude_bounds=[0, 3, 7], longitude_bounds=[0, 3, 7])
    transformed = xr.DataArray(da.ones((2, 2, 2), chunks=(1, 2, 2)),
                               dims=("time", "latitude", "longitude"),
                               coords={"time": source.time, "latitude": target.latitude,
                                       "longitude": target.longitude}, attrs=dict(source.t.attrs), name="t")
    transformed.encoding = dict(source.t.encoding)
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        result, audit = _assemble_result(source, {"t": transformed}, target)
    assert calls == []
    xr.testing.assert_identical(source, before)
    assert "qc" not in result and "crs" not in result and "cell_area" not in result
    assert "cell_measures" not in result.t.attrs and "ancillary_variables" not in result.t.attrs
    assert "actual_range" not in result.t.attrs and "valid_min" not in result.t.attrs
    assert result.t.attrs["cell_methods"] == "time: mean (interval: 1 month)"
    assert result.t.attrs["units"] == "K" and result.t.encoding == {}
    assert result.t.attrs["coordinates"] == "time latitude longitude"
    xr.testing.assert_equal(result.time_bounds, source.time_bounds)
    assert "geospatial_lat_min" not in result.attrs
    assert set(audit["removed_variables"]) == {"qc", "crs", "cell_area"}


def test_result_retains_source_close_callback():
    source = dataset()
    closed = []
    source.set_close(lambda: closed.append(True))
    result, _ = _assemble_result(source, {"t": source.t}, source[["latitude", "longitude"]])
    assert closed == []
    result.close()
    assert closed == [True]


def test_decoded_scalar_crs_is_not_reintroduced_by_dataarray_coordinates():
    source = dataset().set_coords("crs")
    transformed = source.t.copy(deep=False)
    result, audit = _assemble_result(source, {"t": transformed}, source[["latitude", "longitude"]].drop_vars("crs"))
    assert "crs" not in result.variables
    assert "grid_mapping" not in result.t.attrs
    assert "crs" in audit["removed_variables"]


def test_hybrid_dependency_cannot_silently_disappear():
    source = dataset()
    source = source.assign_coords(lev=xr.DataArray(0.5, attrs={"formula_terms": "ps: ps a: a b: b"}))
    source["ps"] = (("latitude", "longitude"), np.ones((3, 3)))
    source["a"] = xr.DataArray(0.2)
    source["b"] = xr.DataArray(0.3)
    with pytest.raises(ValueError, match="formula_terms"):
        regrid_dataset(source, rectilinear_grid([2, 5], [2, 5]), variables=["t"], method="bilinear")


@pytest.mark.parametrize("value,expected", [
    ("time: mean area: mean", "time: mean"),
    ("latitude: longitude: mean time: sum", "time: sum"),
    ("time: mean (interval: 1 hour) area: mean", "time: mean (interval: 1 hour)"),
    ("area: mean", ""),
])
def test_horizontal_cell_method_removal_retains_time_semantics(value, expected):
    assert _without_horizontal_methods(value) == expected


@pytest.mark.parametrize("options", [{}, {"method": "bilinear", "target_grid": {}},
    {"method": "bilinear", "variables": [], "target_grid": {"latitude": [], "longitude": []}},
    {"method": "bilinear", "variables": ["t"], "target_grid": {"latitude": [], "longitude": [], "guess": True}},
    {"method": "bilinear", "variables": ["t"], "target_grid": [], "extra": 1}])
def test_config_regrid_schema_is_strict(options):
    with pytest.raises(ValueError):
        validate_regrid_options(options)


def test_cache_path_resolves_relative_to_configuration(tmp_path):
    config = {"schema_version": 1, "input": "input.nc", "regrid": {
        "variables": ["t"], "method": "bilinear", "weights_dir": "weights",
        "target_grid": {"latitude": [2, 5], "longitude": [2, 5]},
    }}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_prepare_config(path)
    assert loaded.options["regrid"]["weights_dir"] == tmp_path / "weights"
    assert not (tmp_path / "weights").exists()


def test_core_import_does_not_load_native_backend():
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", "import sys; import climate_preprocessing; "
                    "assert not {'xesmf', 'ESMF', 'esmpy'}.intersection(sys.modules)"], check=True)
