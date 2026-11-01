"""Behavior tests for lazy CF preprocessing, including scientific edge cases."""

import cftime
import json
import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.coordinates import normalize_coordinates
from climate_preprocessing.masking import mask_invalid
from climate_preprocessing.metadata import validate_metadata
from climate_preprocessing.subset import subset_region
from climate_preprocessing.time import validate_time
from climate_preprocessing.validation import ensure_lazy


def lazy_field(calls, *, lat=(30.0, -30.0), lon=(-90.0, 0.0, 90.0, 180.0)):
    @delayed
    def field():
        calls.append("field")
        return np.asarray(lat)[:, None] + np.cos(np.deg2rad(lon))[None, :]

    values = da.from_delayed(field(), shape=(len(lat), len(lon)), dtype=float)
    return xr.Dataset(
        {"temperature": (("y", "x"), values, {"units": "K", "cell_methods": "y: mean"})},
        coords={
            "phi": ("y", np.asarray(lat), {"standard_name": "latitude", "units": "degrees_north"}),
            "lambda": ("x", np.asarray(lon), {"standard_name": "longitude", "units": "degrees_east"}),
        },
        attrs={"title": "synthetic"},
    )


def test_normalization_and_masking_do_not_execute_field_tasks():
    calls = []
    source = lazy_field(calls)
    result = mask_invalid(normalize_coordinates(source))
    assert calls == []
    assert isinstance(result.temperature.data, da.Array)
    assert result.temperature.attrs["units"] == "K"
    assert result.temperature.attrs["cell_methods"] == "latitude: mean"
    assert source.temperature.attrs["cell_methods"] == "y: mean"
    np.testing.assert_array_equal(result.latitude, [-30, 30])
    np.testing.assert_array_equal(result.longitude, [0, 90, 180, 270])
    expected = np.array([-30, 30])[:, None] + np.array([1, 0, -1, 0])[None, :]
    np.testing.assert_allclose(result.temperature.compute(), expected, atol=1e-12)
    assert calls == ["field"]


def test_coordinate_bounds_shift_and_reorder_without_changing_width():
    calls = []
    source = lazy_field(calls)
    source["phi"].attrs["bounds"] = "phi_bounds"
    source["lambda"].encoding["bounds"] = "lambda_bounds"
    source["phi_bounds"] = (("y", "vertex"), [[25, 35], [-35, -25]])
    source["lambda_bounds"] = (("x", "vertex"), [[-100, -80], [-10, 10], [80, 100], [170, 190]])
    result = normalize_coordinates(source)
    assert calls == []
    np.testing.assert_array_equal(result.phi_bounds.compute(), [[-35, -25], [25, 35]])
    np.testing.assert_array_equal(result.lambda_bounds.compute(), [[-10, 10], [80, 100], [170, 190], [260, 280]])
    assert calls == []
    assert result.longitude.encoding["bounds"] == "lambda_bounds"
    validate_metadata(result)
    twice = normalize_coordinates(result)
    assert calls == []
    np.testing.assert_array_equal(twice.longitude, result.longitude)
    xr.testing.assert_equal(twice.lambda_bounds.compute(), result.lambda_bounds.compute())


def test_relationships_follow_explicit_renaming_and_source_is_unchanged():
    source = lazy_field([])
    source["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    source.temperature.encoding.update(coordinates="phi lambda", grid_mapping="crs: phi lambda")
    result = normalize_coordinates(source)
    assert result.temperature.encoding["coordinates"] == "latitude longitude"
    assert result.temperature.encoding["grid_mapping"] == "crs: latitude longitude"
    assert source.temperature.encoding["coordinates"] == "phi lambda"


@pytest.mark.parametrize("lon", [(0.0, 360.0), (-180.0, 180.0)])
def test_duplicate_cyclic_endpoints_are_rejected(lon):
    with pytest.raises(ValueError, match="cyclic"):
        normalize_coordinates(lazy_field([], lon=lon))


def test_ambiguous_and_projected_axes_are_not_guessed():
    source = lazy_field([])
    source["other_lat"] = ("y", [30.0, -30.0], {"standard_name": "latitude", "units": "degrees_north"})
    with pytest.raises(ValueError, match="Expected one latitude"):
        normalize_coordinates(source)
    normalize_coordinates(source, latitude="phi")
    source = lazy_field([])
    source["phi"].attrs = {"axis": "Y", "units": "m"}
    with pytest.raises(ValueError, match="degrees_north"):
        normalize_coordinates(source)


def test_explicit_names_work_without_standard_name_but_units_remain_required():
    source = lazy_field([])
    source.phi.attrs.pop("standard_name")
    source["lambda"].attrs.pop("standard_name")
    result = normalize_coordinates(source, latitude="phi", longitude="lambda")
    assert result.latitude.attrs["standard_name"] == "latitude"
    source.phi.attrs.pop("units")
    with pytest.raises(ValueError, match="units"):
        normalize_coordinates(source, latitude="phi", longitude="lambda")


def test_curvilinear_grid_and_unsafe_longitude_bounds_are_rejected():
    source = lazy_field([])
    source = source.drop_vars("phi").assign_coords(phi=(("y", "x"), np.zeros((2, 4))))
    source.phi.attrs.update(standard_name="latitude", units="degrees_north")
    with pytest.raises(ValueError, match="one-dimensional"):
        normalize_coordinates(source)
    source = lazy_field([])
    source["lambda"].attrs["bounds"] = "bounds"
    source["bounds"] = (("x", "vertex"), [[-100, -80], [350, 10], [80, 100], [170, 190]])
    with pytest.raises(ValueError, match="wrapped discontinuity"):
        normalize_coordinates(source)


def test_cross_seam_selection_is_lazy_and_uses_coordinate_centers():
    calls = []
    source = lazy_field(calls, lat=(-30.0, 30.0), lon=(-180.0, -170.0, 0.0, 170.0))
    canonical = normalize_coordinates(source, longitude_convention="-180_180")
    result = subset_region(canonical, latitude_bounds=(-40, 0), longitude_bounds=(170, -170))
    assert calls == []
    np.testing.assert_array_equal(result.latitude, [-30])
    np.testing.assert_array_equal(result.longitude, [-180, -170, 170])
    assert result.temperature.shape == (1, 3)
    assert isinstance(result.temperature.data, da.Array)
    globe = subset_region(canonical, longitude_bounds=(0, 360))
    assert globe.sizes["longitude"] == 4
    meridian = subset_region(canonical, longitude_bounds=(0, 0))
    np.testing.assert_array_equal(meridian.longitude, [0])
    with pytest.raises(ValueError, match="no coordinate centers"):
        subset_region(canonical, latitude_bounds=(80, 90))
    with pytest.raises(ValueError, match="360"):
        subset_region(canonical, longitude_bounds=(0, 721))


def test_masking_preserves_zero_negatives_and_integer_flags():
    ds = xr.Dataset({
        "field": ("point", da.from_array([0.0, -2.0, np.inf, np.nan], chunks=2), {"units": "K"}),
        "qc": ("point", np.array([0, 1, 2, 3], dtype=np.int16)),
    })
    result = mask_invalid(ds)
    assert isinstance(result.field.data, da.Array)
    np.testing.assert_allclose(result.field.compute(), [0, -2, np.nan, np.nan], equal_nan=True)
    assert result.qc.dtype == np.dtype("int16")
    np.testing.assert_array_equal(result.qc.compute(), ds.qc)
    assert np.isinf(ds.field.compute()[2])
    with pytest.raises(ValueError, match="Complex"):
        mask_invalid(xr.Dataset({"z": ("x", np.array([1 + 2j]))}))


def test_time_validation_preserves_noleap_and_360_day():
    for calendar in ("noleap", "360_day"):
        times = [cftime.datetime(2001, 2, 28, calendar=calendar), cftime.datetime(2001, 3, 1, calendar=calendar)]
        ds = xr.Dataset(coords={"time": times})
        validate_time(ds)
        assert ds.time.data[0].calendar == calendar
    ds = xr.Dataset(coords={"time": np.array(["2000-02-28", "2000-02-29", "2000-03-01"], dtype="datetime64[ns]")})
    validate_time(ds)


@pytest.mark.parametrize("times,match", [
    (np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]"), "NaT"),
    (np.array(["2000-01-01", "2000-01-01"], dtype="datetime64[ns]"), "duplicates"),
    (np.array(["2000-02-01", "2000-01-01"], dtype="datetime64[ns]"), "increasing"),
    ([0, 1], "CF-decoded"),
])
def test_invalid_time_indexes_fail(times, match):
    with pytest.raises(ValueError, match=match):
        validate_time(xr.Dataset(coords={"time": times}))


def test_mixed_calendars_fail_without_loading_fields():
    times = [cftime.DatetimeNoLeap(2000, 1, 1), cftime.Datetime360Day(2000, 2, 1)]
    with pytest.raises(ValueError, match="Mixed calendars"):
        validate_time(xr.Dataset(coords={"time": times}))


def test_metadata_references_are_checked_in_attrs_and_encoding():
    ds = xr.Dataset({"field": ("x", da.ones(2, chunks=1))}, coords={"x": [0, 1]})
    ds.field.encoding["ancillary_variables"] = "missing_qc"
    with pytest.raises(ValueError, match="missing_qc"):
        validate_metadata(ds)
    ds.field.encoding = {}
    ds.field.attrs["formula_terms"] = "a: x b: missing"
    with pytest.raises(ValueError, match="missing"):
        validate_metadata(ds)
    ds.field.attrs = {"coordinates": "x"}
    ds.field.encoding["coordinates"] = "other"
    with pytest.raises(ValueError, match="Conflicting"):
        validate_metadata(ds)


@pytest.mark.parametrize("reference", ["crs:", "crs:: x", "x crs: x"])
def test_invalid_expanded_grid_mapping_grammar_is_rejected(reference):
    ds = xr.Dataset({"field": ("x", [1.0, 2.0]), "crs": 0}, coords={"x": [0, 1]})
    ds.field.attrs["grid_mapping"] = reference
    with pytest.raises(ValueError, match="grid_mapping"):
        validate_metadata(ds)


def test_ensure_lazy_preserves_chunks_and_closes_the_original_resource():
    closed = []
    ds = xr.Dataset({"f": ("x", da.ones(6, chunks=2))})
    ds.set_close(lambda: closed.append(True))
    result = ensure_lazy(ds)
    assert result.f.chunks == ((2, 2, 2),)
    result.close()
    assert closed == [True]
    with pytest.raises(ValueError, match="chunks"):
        ensure_lazy(ds, chunks=None)


def test_ensure_lazy_keeps_existing_chunks_when_an_eager_field_is_added():
    ds = xr.Dataset({
        "lazy": ("x", da.ones(6, chunks=2)),
        "eager": ("x", np.arange(6)),
    })
    result = ensure_lazy(ds)
    assert result.lazy.chunks == ((2, 2, 2),)
    assert isinstance(result.eager.data, da.Array)
    explicit = ensure_lazy(ds, chunks={"x": 3})
    assert explicit.lazy.chunks == ((3, 3),)
    assert explicit.eager.chunks == ((3, 3),)


def test_wrapping_invalidates_and_audits_old_longitude_ranges():
    source = lazy_field([])
    source["lambda"].attrs.update(valid_min=-180.0, valid_max=180.0, actual_range=[-90, 180], bounds="lon_bounds")
    source["lambda"].encoding["valid_range"] = [-180.0, 180.0]
    source["lon_bounds"] = (("x", "vertex"), [[-100, -80], [-10, 10], [80, 100], [170, 190]])
    source.lon_bounds.attrs["actual_range"] = [-100.0, 190.0]
    result = normalize_coordinates(source)
    for key in ("valid_min", "valid_max", "valid_range", "actual_range"):
        assert key not in result.longitude.attrs
        assert key not in result.longitude.encoding
    assert "actual_range" not in result.lon_bounds.attrs
    assert source["lambda"].attrs["valid_min"] == -180.0
    audit = json.loads(result.attrs["climate_preprocessing_provenance"])
    removed = audit["operations"][-1]["parameters"]["removed_range_attributes"]
    assert "attrs.valid_min" in removed["longitude"]
    assert removed["lon_bounds"] == ["attrs.actual_range"]


def test_cropping_drops_obsolete_actual_ranges_without_reducing_fields():
    calls = []
    source = normalize_coordinates(lazy_field(calls))
    source.temperature.attrs.update(actual_range=[-31.0, 31.0], valid_range=[-100.0, 100.0])
    source.latitude.attrs["actual_range"] = [-30.0, 30.0]
    result = subset_region(source, latitude_bounds=(0, 90))
    assert calls == []
    assert "actual_range" not in result.temperature.attrs
    assert "actual_range" not in result.latitude.attrs
    assert result.temperature.attrs["valid_range"] == [-100.0, 100.0]
    assert source.temperature.attrs["actual_range"] == [-31.0, 31.0]
    audit = json.loads(result.attrs["climate_preprocessing_provenance"])
    assert audit["operations"][-1]["parameters"]["removed_actual_range"]["temperature"] == ["attrs"]
