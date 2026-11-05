"""Backend-independent field selection, CF output and configuration contracts."""

import json

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import interpolate_pressure
from climate_preprocessing.config import load_prepare_config
from climate_preprocessing.vertical.interpolate import _assemble_result, validate_vertical_options
from climate_preprocessing.vertical.pressure import resolve_pressure


def fixed_dataset():
    source = xr.Dataset(coords={
        "lev": ("lev", [1000., 800., 600., 400.], {"standard_name": "air_pressure", "units": "hPa", "bounds": "lev_bounds"}),
        "time": np.array(["2001-01-01", "2001-02-01"], dtype="datetime64[ns]"),
        "latitude": ("latitude", [10., 20.], {"standard_name": "latitude", "units": "degrees_north"}),
        "longitude": ("longitude", [100., 110.], {"standard_name": "longitude", "units": "degrees_east"}),
    })
    source["t"] = (("time", "lev", "latitude", "longitude"), da.ones((2, 4, 2, 2), chunks=(1, 2, 1, 2)),
                   {"units": "K", "ancillary_variables": "qc calibration", "cell_methods": "time: mean lev: mean area: mean",
                    "cell_measures": "area: cell_area volume: cell_volume", "grid_mapping": "crs", "actual_range": [1, 1]})
    source["lev_bounds"] = (("lev", "nv"), [[1100., 900.], [900., 700.], [700., 500.], [500., 300.]])
    source["qc"] = xr.zeros_like(source.t, dtype="i1")
    source.qc.attrs["flag_meanings"] = "good"
    source["calibration"] = xr.DataArray(1.0, attrs={"long_name": "calibration factor"})
    source["cell_area"] = (("latitude", "longitude"), np.ones((2, 2)), {"units": "m2"})
    source["cell_volume"] = (("lev", "latitude", "longitude"), np.ones((4, 2, 2)), {"units": "m3"})
    source["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    source["ps"] = (("time", "latitude", "longitude"), da.ones((2, 2, 2), chunks=(1, 1, 2)) * 950,
                    {"standard_name": "surface_air_pressure", "units": "hPa"})
    source.time.attrs["bounds"] = "time_bounds"
    source["time_bounds"] = (("time", "nv"), np.array(
        [["2000-12-15", "2001-01-15"], ["2001-01-15", "2001-02-15"]], dtype="datetime64[ns]"))
    return source


@pytest.mark.parametrize("options,match", [
    ({"method": "nearest"}, "method"), ({"method": []}, "method"),
    ({"variables": []}, "nonempty"), ({"variables": ["qc"]}, "QC"),
    ({"variables": ["lev_bounds"]}, "structural"), ({"variables": ["ps"]}, "vertical dimension"),
    ({"surface_pressure": True}, "name"), ({"coordinate": ""}, "name"),
    ({"target_units": "m"}, "Pa"), ({"target_levels": [800, 600, 700]}, "monotonic"),
    ({"target_levels": [0, 100]}, "positive"), ({"target_levels": [500, 500]}, "distinct"),
])
def test_input_contract_failures_precede_numerical_backend(options, match):
    kwargs = {"target_levels": [900., 700.], "variables": ["t"], "method": "linear", **options}
    with pytest.raises((ValueError, TypeError), match=match):
        interpolate_pressure(fixed_dataset(), **kwargs)


def test_integer_target_fields_are_refused():
    source = fixed_dataset()
    source["t"] = source.t.astype("int64")
    with pytest.raises(ValueError, match="floating-point"):
        interpolate_pressure(source, [700], variables="t", method="log")


def test_foreign_surface_dimensions_cannot_broadcast_new_axes():
    source = fixed_dataset()
    source["foreign_ps"] = ("station", [100000., 101000.], {"units": "Pa"})
    source = source.assign_coords(station=["a", "b"])
    with pytest.raises(ValueError, match="subset"):
        interpolate_pressure(source, [700], variables="t", method="linear", surface_pressure="foreign_ps")


def test_vertical_output_rebuilds_cf_without_source_mutation_or_tasks():
    source = fixed_dataset()
    source.attrs.update(geospatial_vertical_min=400, geospatial_lat_min=10)
    source.t.encoding.update(scale_factor=0.1, dtype="int16", coordinates="time lev latitude longitude")
    before = source.copy(deep=True)
    spec = resolve_pressure(source, surface_pressure="ps")
    target = xr.DataArray([800., 600.], dims="pressure", name="pressure",
                          attrs={"standard_name": "air_pressure", "units": "hPa", "positive": "down"})
    field = source.t.isel(lev=[1, 2]).rename(lev="pressure").assign_coords(pressure=target)
    field.encoding = dict(source.t.encoding)
    calls = []
    with Callback(pretask=lambda *args: calls.append(args)):
        result, audit = _assemble_result(source, {"t": field}, target, spec)
    assert not calls
    xr.testing.assert_identical(source, before)
    assert not {"lev", "lev_bounds", "qc", "cell_volume"}.intersection(result.variables)
    assert result.t.attrs["ancillary_variables"] == "calibration"
    assert result.t.attrs["cell_measures"] == "area: cell_area"
    assert result.t.attrs["cell_methods"] == "time: mean area: mean"
    assert result.t.attrs["coordinates"] == "time pressure latitude longitude"
    assert result.t.attrs["grid_mapping"] == "crs" and result.t.attrs["units"] == "K"
    assert result.t.encoding == {} and "actual_range" not in result.t.attrs
    assert "geospatial_vertical_min" not in result.attrs and result.attrs["geospatial_lat_min"] == 10
    assert "ps" in result and "lev_bounds" in audit["removed_variables"]
    xr.testing.assert_equal(result.time_bounds, source.time_bounds)


def test_result_owns_source_close_callback():
    source = fixed_dataset()
    closed = []
    source.set_close(lambda: closed.append(True))
    spec = resolve_pressure(source)
    target = xr.DataArray([800., 600.], dims="pressure")
    field = source.t.isel(lev=[1, 2]).rename(lev="pressure").assign_coords(pressure=target)
    result, _ = _assemble_result(source, {"t": field}, target, spec)
    assert closed == []
    result.close()
    assert closed == [True]


def test_retained_surface_pressure_keeps_nonvertical_bounds():
    source = fixed_dataset()
    source["ps_bounds"] = xr.concat([source.ps - 1, source.ps + 1], dim="vertex")
    source.ps.attrs["bounds"] = "ps_bounds"
    spec = resolve_pressure(source, surface_pressure="ps")
    target = xr.DataArray([800., 600.], dims="pressure")
    field = source.t.isel(lev=[1, 2]).rename(lev="pressure").assign_coords(pressure=target)
    result, _ = _assemble_result(source, {"t": field}, target, spec)
    assert result.ps.attrs["bounds"] == "ps_bounds"
    xr.testing.assert_equal(result.ps_bounds, source.ps_bounds)


@pytest.mark.parametrize("change", [{"extrapolate": True}, {"method": "cubic"}, {"surface_pressure": [1]},
                                     {"target_levels": []}, {"target_units": "guess"}])
def test_configuration_rejects_unsupported_policies(change):
    with pytest.raises((ValueError, TypeError)):
        validate_vertical_options({"target_levels": [700], "method": "linear", "variables": ["t"], **change})


def test_json_configuration_reads_without_opening_data(tmp_path):
    config = {"schema_version": 1, "input": "absent.nc", "vertical": {
        "target_levels": [900, 700], "target_units": "hPa", "variables": ["t"], "method": "log", "surface_pressure": "ps",
    }}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    loaded = load_prepare_config(path)
    assert loaded.options["vertical"] == config["vertical"]


def test_retained_independent_formula_dependency_is_not_silently_deleted():
    source = fixed_dataset()
    source = source.assign_coords(extra=xr.DataArray(1., attrs={"formula_terms": "a: lev"}))
    with pytest.raises(ValueError, match="retained.*formula_terms"):
        interpolate_pressure(source, [700], variables="t", method="linear", coordinate="lev")
