"""Pressure coordinate physics, CF relations and lazy field reindexing contracts."""
import json

import cftime
import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.coordinates import normalize_pressure, pressure_order
from climate_preprocessing.io import open_dataset, write_dataset
from climate_preprocessing.metadata import validate_metadata
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE


def source_dataset(calls=None, units="Pa", levels=(100000., 50000., 10000.)):
    calls = [] if calls is None else calls
    values = np.arange(len(levels) * 4, dtype=np.float64).reshape(len(levels), 2, 2)

    @delayed
    def field():
        calls.append("field")
        return values

    field = da.from_delayed(field(), shape=values.shape, dtype=values.dtype)
    dataset = xr.Dataset({"temperature": (("lev", "y", "x"), field, {"units": "K"})}, coords={
        "lev": ("lev", np.asarray(levels), {"units": units, "standard_name": "air_pressure", "axis": "Z"}),
        "y": [0., 1.], "x": [0., 1.],
    })
    return dataset, values


def event(result):
    events = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"]
    return next(item for item in reversed(events) if item["operation"] == "normalize_pressure")


def test_pressure_conversion_and_synchronous_field_order_remain_lazy():
    calls = []
    source, values = source_dataset(calls)
    result = normalize_pressure(source)
    assert calls == []
    assert isinstance(result.temperature.data, da.Array)
    assert result.temperature.dims == ("pressure", "y", "x")
    np.testing.assert_array_equal(result.pressure, [100., 500., 1000.])
    assert result.pressure.attrs == {"units": "hPa", "standard_name": "air_pressure", "axis": "Z", "positive": "down"}
    np.testing.assert_array_equal(result.temperature.compute(), values[::-1])
    assert calls == ["field"]
    assert source.lev.attrs["units"] == "Pa"
    np.testing.assert_array_equal(source.lev, [100000., 50000., 10000.])
    assert event(result)["parameters"]["source_units"] == "Pa"


def test_millibar_alias_and_descending_pa_target_use_shared_kernel():
    source, values = source_dataset(units="millibars", levels=(100., 1000., 500.))
    result = normalize_pressure(source, target_units="Pa", order="descending")
    np.testing.assert_array_equal(result.pressure, [100000., 50000., 10000.])
    np.testing.assert_array_equal(result.temperature.compute(), values[[1, 2, 0]])
    normalized, indexer = pressure_order(np.array([100., 1000., 500.]), "millibars", "Pa", "descending")
    np.testing.assert_array_equal(normalized, result.pressure)
    np.testing.assert_array_equal(indexer, [1, 2, 0])


@pytest.mark.parametrize("relation_container", ["attrs", "encoding"])
def test_bounds_inherit_units_and_follow_level_reorder(relation_container):
    calls = []
    source, _ = source_dataset(calls)
    getattr(source.lev, relation_container)["bounds"] = "lev_bounds"
    original = np.array([[95000., 105000.], [45000., 55000.], [5000., 15000.]])
    source["lev_bounds"] = (("lev", "vertex"), original)
    result = normalize_pressure(source)
    assert calls == []
    np.testing.assert_array_equal(result.lev_bounds.compute(), original[::-1] / 100)
    assert result.lev_bounds.attrs["units"] == "hPa"
    assert getattr(result.pressure, relation_container)["bounds"] == "lev_bounds"
    validate_metadata(result)
    repeated = normalize_pressure(result)
    np.testing.assert_array_equal(repeated.pressure, result.pressure)
    np.testing.assert_array_equal(repeated.lev_bounds.compute(), result.lev_bounds.compute())
    assert calls == []


def test_bounds_can_declare_different_pressure_units_and_vertex_first_order():
    calls = []
    source, _ = source_dataset(calls, units="hPa", levels=(1000., 500., 100.))
    source.lev.attrs["bounds"] = "interfaces"
    original = np.array([[95000., 105000.], [45000., 55000.], [5000., 15000.]])
    source["interfaces"] = (("vertex", "lev"), original.T,
                             {"units": "Pa", "standard_name": "air_pressure"})
    result = normalize_pressure(source)
    assert calls == []
    assert result.interfaces.dims == ("vertex", "pressure")
    np.testing.assert_array_equal(result.interfaces.compute(), original[::-1].T / 100)
    assert event(result)["parameters"]["bounds_source_units"] == "Pa"
    assert source.interfaces.attrs["units"] == "Pa"
    normalize_pressure(result)


def test_auxiliary_pressure_becomes_dimension_and_cf_links_are_renamed():
    source, values = source_dataset()
    source = source.rename({"lev": "p"}).rename_dims({"p": "model_index"})
    source.temperature.encoding["coordinates"] = "p"
    source.temperature.attrs["cell_methods"] = "model_index: point"
    result = normalize_pressure(source, pressure="p")
    assert result.pressure.dims == ("pressure",)
    assert result.temperature.encoding["coordinates"] == "pressure"
    assert result.temperature.attrs["cell_methods"] == "pressure: point"
    np.testing.assert_array_equal(result.temperature.compute(), values[::-1])
    assert source.temperature.encoding["coordinates"] == "p"
    assert source.temperature.attrs["cell_methods"] == "model_index: point"


def test_retained_model_coordinate_references_do_not_turn_into_physical_pressure():
    source = xr.Dataset({
        "temperature": ("model_index", da.from_array([10., 20.], chunks=1), {
            "coordinates": "model_index p", "ancillary_variables": "model_index",
            "cell_methods": "model_index: point", "units": "K"}),
        "a": ("model_index", [0.2, 0.1]), "p0": 100000., "ps": 95000.,
    }, coords={
        "model_index": ("model_index", [7., 9.], {
            "axis": "Z", "units": "1", "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
            "formula_terms": "a: a b: model_index p0: p0 ps: ps"}),
        "p": ("model_index", [100000., 50000.], {"units": "Pa", "standard_name": "air_pressure"}),
    })
    result = normalize_pressure(source, pressure="p")
    np.testing.assert_array_equal(result.model_index, [9., 7.])
    assert result.temperature.attrs["coordinates"] == "model_index pressure"
    assert result.temperature.attrs["ancillary_variables"] == "model_index"
    assert result.temperature.attrs["cell_methods"] == "pressure: point"
    assert result.model_index.attrs["formula_terms"] == "a: a b: model_index p0: p0 ps: ps"
    assert source.temperature.attrs["coordinates"] == "model_index p"
    assert source.temperature.attrs["cell_methods"] == "model_index: point"
    validate_metadata(result)
    repeated = normalize_pressure(result)
    xr.testing.assert_equal(repeated.pressure, result.pressure)
    assert repeated.model_index.attrs["formula_terms"] == result.model_index.attrs["formula_terms"]


def test_auxiliary_coordinate_on_already_named_pressure_dimension_gets_an_index():
    source = xr.Dataset({"field": ("pressure", [10., 20.])}, coords={
        "p": ("pressure", [100000., 50000.], {"units": "Pa"}),
    })
    result = normalize_pressure(source, pressure="p")
    assert "pressure" in result.xindexes
    assert result.sel(pressure=500.).field.compute().item() == 20.


def test_pressure_unit_evidence_without_name_guessing_and_explicit_mapping():
    source, _ = source_dataset()
    source.lev.attrs = {"units": "Pa"}
    normalize_pressure(source)
    source.lev.attrs = {}
    with pytest.raises(ValueError, match="Expected one pressure"):
        normalize_pressure(source)
    with pytest.raises(ValueError, match="units"):
        normalize_pressure(source, pressure="lev")
    source.lev.attrs = {"units": "Pa"}
    source = source.reset_coords(drop=False)  # dimensional lev remains an index coordinate
    assert normalize_pressure(source, pressure="lev").sizes["pressure"] == 3


def test_unit_changes_and_packed_sources_invalidate_coordinate_ranges():
    source, _ = source_dataset()
    source.lev.attrs.update(valid_min=10000., actual_range=[10000., 100000.], minimum_value=10000.)
    source.lev.encoding.update(dtype="int16", scale_factor=.1, add_offset=100., _FillValue=-1,
                               valid_range=[0, 1000], units="Pa", preferred_chunks={"lev": 3})
    result = normalize_pressure(source)
    for key in ("valid_min", "actual_range", "minimum_value"):
        assert key not in result.pressure.attrs
    for key in ("dtype", "scale_factor", "add_offset", "_FillValue", "valid_range", "units", "preferred_chunks"):
        assert key not in result.pressure.encoding
    removed = event(result)["parameters"]["removed_range_attributes"]["pressure"]
    assert "attrs.actual_range" in removed and "encoding.valid_range" in removed
    assert source.lev.attrs["valid_min"] == 10000.
    assert source.lev.encoding["scale_factor"] == .1
    same_units = normalize_pressure(source, target_units="Pa")
    assert "actual_range" not in same_units.pressure.attrs  # packed-domain ambiguity remains


def test_bounds_packing_is_reset_without_losing_cf_relationship():
    source, _ = source_dataset()
    source.lev.encoding["bounds"] = "bounds"
    source["bounds"] = (("lev", "vertex"), [[95000., 105000.], [45000., 55000.], [5000., 15000.]])
    source.bounds.attrs["maximum_value"] = 105000.
    source.bounds.encoding.update(scale_factor=100., dtype="int16", actual_range=[50, 1050])
    result = normalize_pressure(source)
    assert result.pressure.encoding["bounds"] == "bounds"
    assert "maximum_value" not in result.bounds.attrs
    assert "scale_factor" not in result.bounds.encoding and "actual_range" not in result.bounds.encoding


@pytest.mark.parametrize("levels", [(0., 100., 500.), (-1., 100., 500.), (100., 100., 500.),
                                   (np.nan, 100., 500.), (np.inf, 100., 500.)])
def test_nonpositive_duplicate_or_nonfinite_levels_fail_without_field_tasks(levels):
    calls = []
    source, _ = source_dataset(calls, levels=levels)
    with pytest.raises(ValueError, match="positive|distinct|nonfinite"):
        normalize_pressure(source)
    assert calls == []


@pytest.mark.parametrize("attributes", [
    {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate"},
    {"standard_name": "model_level_number"}, {"standard_name": "height"},
    {"axis": "X"}, {"positive": "up"}, {"formula_terms": "ap: ap"},
])
def test_model_hybrid_height_or_contradictory_cf_evidence_is_rejected(attributes):
    calls = []
    source, _ = source_dataset(calls)
    source["ap"] = xr.DataArray(1.)
    source.lev.attrs.update(attributes)
    with pytest.raises(ValueError, match="pressure|Pressure|Hybrid|axis|positive"):
        normalize_pressure(source, pressure="lev")
    assert calls == []


@pytest.mark.parametrize("units", ["", "1", "m", "K", "hPa s-1", "psi"])
def test_unrecognized_or_nonpressure_units_are_rejected(units):
    source, _ = source_dataset(units=units)
    with pytest.raises(ValueError, match="units"):
        normalize_pressure(source)


def test_ambiguous_axes_and_canonical_name_collisions_are_not_overridden():
    source, _ = source_dataset()
    source = source.assign_coords(other=("lev", [100000., 50000., 10000.], {"units": "Pa"}))
    with pytest.raises(ValueError, match="Expected one pressure"):
        normalize_pressure(source)
    normalize_pressure(source, pressure="lev")
    source["pressure"] = xr.DataArray(0.)
    with pytest.raises(ValueError, match="already exists"):
        normalize_pressure(source, pressure="lev")


@pytest.mark.parametrize("arguments", [{"target_units": "mbar"}, {"target_units": None},
                                      {"order": "up"}, {"order": None}, {"pressure": True}])
def test_invalid_options_fail_before_computing_fields(arguments):
    calls = []
    source, _ = source_dataset(calls)
    with pytest.raises(ValueError):
        normalize_pressure(source, **arguments)
    assert calls == []


@pytest.mark.parametrize("matrix", [
    [[1., 2.], [45000., 55000.], [5000., 15000.]],
    [[0., 105000.], [45000., 55000.], [5000., 15000.]],
    [[100000., 100000.], [45000., 55000.], [5000., 15000.]],
    [[np.nan, 105000.], [45000., 55000.], [5000., 15000.]],
])
def test_invalid_pressure_bounds_are_rejected(matrix):
    source, _ = source_dataset()
    source.lev.attrs["bounds"] = "bounds"
    source["bounds"] = (("lev", "vertex"), matrix)
    with pytest.raises(ValueError, match="bounds|nonfinite"):
        normalize_pressure(source)


def test_bad_bounds_shape_and_nonpressure_units_are_rejected():
    source, _ = source_dataset()
    source.lev.attrs["bounds"] = "bounds"
    source["bounds"] = (("lev", "vertex"), np.ones((3, 3)))
    with pytest.raises(ValueError, match="size-2"):
        normalize_pressure(source)
    source = source.drop_vars("bounds")
    source["bounds"] = (("lev", "edge"), [[95000., 105000.], [45000., 55000.], [5000., 15000.]], {"units": "m"})
    with pytest.raises(ValueError, match="units"):
        normalize_pressure(source)


def test_raw_packing_declared_missing_and_non_numeric_coordinates_are_rejected():
    source, _ = source_dataset()
    source.lev.attrs["scale_factor"] = .1
    with pytest.raises(ValueError, match="CF-decode"):
        normalize_pressure(source)
    source.lev.attrs.pop("scale_factor")
    source.lev.attrs["missing_value"] = 50000.
    with pytest.raises(ValueError, match="declared missing"):
        normalize_pressure(source)
    source = source.assign_coords(lev=("lev", ["1000", "500", "100"], {"units": "hPa"}))
    with pytest.raises(ValueError, match="numeric"):
        normalize_pressure(source, pressure="lev")


def test_pressure_on_time_dimension_and_multidimensional_pressure_are_rejected():
    source = xr.Dataset(coords={"time": ("time", [0, 1], {"axis": "T"}),
                               "p": ("time", [1000., 900.], {"units": "hPa"})})
    with pytest.raises(ValueError, match="time or horizontal"):
        normalize_pressure(source, pressure="p")
    source = xr.Dataset(coords={"p": (("z", "x"), [[1000., 900.]], {"units": "hPa", "standard_name": "air_pressure"})})
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_pressure(source)


def test_pressure_roundtrip_retains_units_bounds_provenance_and_close(tmp_path):
    calls, closed = [], []
    source, values = source_dataset(calls)
    source.set_close(lambda: closed.append(True))
    source.lev.attrs["bounds"] = "bounds"
    source["bounds"] = (("lev", "vertex"), [[95000., 105000.], [45000., 55000.], [5000., 15000.]])
    source.temperature.encoding["coordinates"] = "lev"
    result = normalize_pressure(source)
    assert closed == [] and calls == []
    output = tmp_path / "normalized.nc"
    write_dataset(result, output)
    with open_dataset(output, chunks={}) as reopened:
        np.testing.assert_array_equal(reopened.pressure, [100., 500., 1000.])
        np.testing.assert_array_equal(reopened.temperature.compute(), values[::-1])
        np.testing.assert_array_equal(reopened.bounds.compute(), [[50., 150.], [450., 550.], [950., 1050.]])
        assert reopened.pressure.attrs["units"] == "hPa"
        # CF serialization may omit units on bounds when they equal the parent.
        assert reopened.bounds.attrs.get("units", reopened.pressure.attrs["units"]) == "hPa"
        assert reopened.pressure.encoding["bounds"] == "bounds"
        assert reopened.temperature.encoding["coordinates"] == "pressure"
        assert event(reopened)["parameters"]["target_units"] == "hPa"
    assert closed == []
    result.close()
    assert closed == [True]


@pytest.mark.parametrize("dimension_values,attributes", [
    (np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"), {}),
    ([cftime.DatetimeNoLeap(2000, 1, 1), cftime.DatetimeNoLeap(2000, 2, 1)], {}),
    ([0, 1], {"units": "hours since 2000-01-01"}),
    ([0., 1.], {"units": "degrees_north"}),
    ([0., 1.], {"units": "degrees_east"}),
])
def test_time_and_horizontal_evidence_cannot_be_replaced_by_pressure(dimension_values, attributes):
    source = xr.Dataset(coords={
        "dimension": ("dimension", dimension_values, attributes),
        "surface_pressure": ("dimension", [1000., 900.], {"units": "hPa", "standard_name": "air_pressure"}),
    })
    with pytest.raises(ValueError, match="time or horizontal"):
        normalize_pressure(source, pressure="surface_pressure")
