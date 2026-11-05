"""CF hybrid-pressure physics and eager-geometry/lazy-field contracts."""

import json

from dask import delayed
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.vertical.pressure import hybrid_pressure, resolve_pressure, target_pressure


def hybrid_dataset(*, formula="a", ps_values=None, ps_units="Pa", calls=None):
    values = np.array([[100000., 80000.], [90000., np.nan]]) if ps_values is None else np.asarray(ps_values)
    if calls is None:
        surface = values
    else:
        @delayed
        def field():
            calls.append("surface")
            return values

        surface = da.from_delayed(field(), shape=values.shape, dtype=values.dtype).rechunk((1, 1))
    dataset = xr.Dataset({
        "a": ("lev", [.1, .2, .3], {"units": "1"}),
        "ap": ("lev", [10000., 20000., 30000.], {"units": "Pa"}),
        "b": ("lev", [.1, .3, .5], {"units": "1"}),
        "p0": ((), 100000., {"units": "Pa"}),
        "ps": (("time", "latitude"), surface, {"units": ps_units, "standard_name": "surface_air_pressure"}),
    }, coords={
        "lev": ("lev", [1., 2., 3.], {"units": "1", "axis": "Z",
                "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                "formula_terms": "a: a b: b p0: p0 ps: ps" if formula == "a" else "ap: ap b: b ps: ps"}),
        "time": ("time", np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]")),
        "latitude": ("latitude", [10., 20.], {"units": "degrees_north", "standard_name": "latitude"}),
    })
    return dataset


@pytest.mark.parametrize("formula", ["a", "ap"])
def test_both_cf_formulas_match_analytic_pressure_without_field_tasks(formula):
    calls = []
    dataset = hybrid_dataset(formula=formula, calls=calls)
    original = dataset.copy(deep=True)
    result = hybrid_pressure(dataset)
    assert calls == []
    assert isinstance(result.data, da.Array)
    assert result.dims == ("lev", "time", "latitude")
    assert result.chunks[1:] == dataset.ps.chunks
    assert result.attrs["units"] == "Pa"
    assert result.attrs["standard_name"] == "air_pressure"
    assert "formula_terms" not in result.lev.attrs
    assert "standard_name" not in result.lev.attrs
    source = json.loads(result.attrs["climate_preprocessing_pressure_source"])
    assert source["coordinate"] == "lev"
    assert source["terms"]["ps"] == "ps"
    expected = np.array([10000., 20000., 30000.])[:, None, None] + np.array([.1, .3, .5])[:, None, None] * np.array([
        [100000., 80000.], [90000., np.nan]])
    np.testing.assert_allclose(result.compute(), expected, rtol=0, atol=0, equal_nan=True)
    assert calls == ["surface"]
    assert dataset.lev.attrs == original.lev.attrs
    assert dataset.ps.attrs == original.ps.attrs


@pytest.mark.parametrize("formula", ["a", "ap"])
def test_pressure_units_convert_independently_for_ap_p0_and_ps(formula):
    dataset = hybrid_dataset(formula=formula, ps_values=[[1000., 800.], [900., np.nan]], ps_units="hPa")
    dataset.p0.data = np.array(1000.)
    dataset.p0.attrs["units"] = "hPa"
    dataset.ap.data = np.array([100., 200., 300.])
    dataset.ap.attrs["units"] = "millibars"
    expected = hybrid_pressure(hybrid_dataset(formula=formula)).compute()
    xr.testing.assert_allclose(hybrid_pressure(dataset).compute(), expected)


def test_formula_terms_in_encoding_and_pressure_surface_standard_name():
    dataset = hybrid_dataset(formula="ap")
    dataset.lev.encoding["formula_terms"] = dataset.lev.attrs.pop("formula_terms")
    dataset.ps.attrs["standard_name"] = "air_pressure"
    specification = resolve_pressure(dataset)
    assert specification.kind == "hybrid"
    assert specification.formula == {"kind": "ap_b_ps", "terms": {"ap": "ap", "b": "b", "ps": "ps"}}
    assert specification.structural_names == {"lev", "ap", "b", "ps"}


def test_distinct_formula_labels_may_share_a_dimensionless_coefficient():
    dataset = hybrid_dataset()
    dataset.lev.attrs["formula_terms"] = "a: a b: a p0: p0 ps: ps"
    pressure = hybrid_pressure(dataset).compute()
    np.testing.assert_array_equal(pressure[:, 0, 0], [20000., 40000., 60000.])


def test_declared_lazy_coefficients_are_geometry_but_surface_fields_stay_unread():
    calls = []
    dataset = hybrid_dataset(calls=calls)
    dataset["a"] = dataset.a.chunk()
    dataset["b"] = dataset.b.chunk()
    dataset["p0"] = xr.DataArray(da.from_array(np.array(100000.), chunks=()), attrs={"units": "Pa"})
    pressure = hybrid_pressure(dataset)
    assert calls == []
    assert isinstance(pressure.data, da.Array)


def test_auxiliary_hybrid_coordinate_preserves_its_vertical_index():
    dataset = hybrid_dataset().rename_vars({"lev": "hybrid"})
    dataset = dataset.assign_coords(lev=("lev", [11., 12., 13.]))
    pressure = hybrid_pressure(dataset, coordinate="hybrid")
    np.testing.assert_array_equal(pressure.lev, [11., 12., 13.])
    np.testing.assert_array_equal(pressure.hybrid, [1., 2., 3.])
    assert "formula_terms" not in pressure.hybrid.attrs
    assert "standard_name" not in pressure.hybrid.attrs
    assert pressure.dims == ("lev", "time", "latitude")


@pytest.mark.parametrize("metadata_container", ["attrs", "encoding"])
def test_standalone_labels_are_generic_and_source_hybrid_metadata_is_unchanged(metadata_container):
    dataset = hybrid_dataset()
    dataset.lev.attrs["long_name"] = "Original model level"
    if metadata_container == "encoding":
        for key in ("standard_name", "formula_terms"):
            dataset.lev.encoding[key] = dataset.lev.attrs.pop(key)
    original_attrs = dict(dataset.lev.attrs)
    original_encoding = dict(dataset.lev.encoding)
    pressure = hybrid_pressure(dataset)
    assert pressure.lev.attrs["units"] == "1"
    assert pressure.lev.attrs["axis"] == "Z"
    assert pressure.lev.attrs["long_name"] == "Original model level"
    for container in (pressure.lev.attrs, pressure.lev.encoding):
        assert "standard_name" not in container
        assert "formula_terms" not in container
    assert dataset.lev.attrs == original_attrs
    assert dataset.lev.encoding == original_encoding


def test_eager_scalar_surface_is_still_checked_lazily_and_close_is_shared():
    dataset = hybrid_dataset().drop_vars("ps")
    dataset["ps"] = xr.DataArray(100000., attrs={"units": "Pa"})
    closed = []
    dataset.set_close(lambda: closed.append("source"))
    pressure = hybrid_pressure(dataset)
    assert pressure.dims == ("lev",)
    assert isinstance(pressure.data, da.Array)
    np.testing.assert_array_equal(pressure.compute(), [20000., 50000., 80000.])
    assert closed == []
    pressure.close()
    assert closed == ["source"]


@pytest.mark.parametrize("invalid", [0., -1., np.inf, -np.inf])
def test_nonphysical_surface_fails_on_compute_only(invalid):
    calls = []
    dataset = hybrid_dataset(ps_values=[[invalid, 100000.], [90000., 80000.]], calls=calls)
    pressure = hybrid_pressure(dataset)
    assert calls == []
    with pytest.raises(ValueError, match="Surface pressure"):
        pressure.compute()


@pytest.mark.parametrize("intercept", [-100000., 1e308])
def test_nonphysical_derived_pressure_fails_on_compute(intercept):
    dataset = hybrid_dataset(formula="ap")
    dataset.ap.data = np.full(3, intercept)
    if intercept > 0:
        dataset.b.data = np.full(3, 1e308)
    pressure = hybrid_pressure(dataset)
    with pytest.raises(ValueError, match="Hybrid cell-center pressure"):
        pressure.compute()


def test_missing_surface_is_entire_missing_column_even_with_zero_b():
    dataset = hybrid_dataset(formula="ap")
    dataset.b.data = np.zeros(3)
    pressure = hybrid_pressure(dataset).compute()
    assert np.isnan(pressure[:, 1, 1]).all()
    np.testing.assert_array_equal(pressure[:, 0, 0], [10000., 20000., 30000.])


@pytest.mark.parametrize("terms,match", [
    ("a: a b: b p0: p0 ps: ps a: a", "duplicate"),
    ("a: a b: b ps: ps", "exactly"),
    ("ap: ap b: b ps: ps p0: p0", "exactly"),
    ("a: a ap: ap b: b p0: p0 ps: ps", "exactly"),
    ("ap: ap b: b ps: ps unknown: a", "exactly"),
    ("ap: missing b: b ps: ps", "dangling"),
])
def test_malformed_or_ambiguous_formulas_fail_eagerly(terms, match):
    dataset = hybrid_dataset()
    dataset.lev.attrs["formula_terms"] = terms
    with pytest.raises(ValueError, match=match):
        hybrid_pressure(dataset)


@pytest.mark.parametrize("variable,units", [("lev", "level"), ("a", None), ("b", "Pa"), ("p0", "K"), ("ps", "m")])
def test_explicit_dimensionless_and_pressure_units_are_required(variable, units):
    dataset = hybrid_dataset()
    if units is None:
        dataset[variable].attrs.pop("units")
    else:
        dataset[variable].attrs["units"] = units
    with pytest.raises(ValueError):
        hybrid_pressure(dataset)


@pytest.mark.parametrize("bad", ["interfaces", "bounds", "nonfinite", "scalar_reference", "vertical_ps", "integer_ps"])
def test_hybrid_geometry_cannot_be_reinterpreted(bad):
    dataset = hybrid_dataset()
    if bad == "interfaces":
        dataset["a"] = (("interface",), [0., .1, .2, .3], {"units": "1"})
    elif bad == "bounds":
        dataset["a"] = (("lev", "nv"), np.ones((3, 2)), {"units": "1"})
    elif bad == "nonfinite":
        dataset.a.data = np.array([.1, np.nan, .3])
    elif bad == "scalar_reference":
        dataset["p0"] = (("lev",), [100000., 100000., 100000.], {"units": "Pa"})
    elif bad == "vertical_ps":
        dataset["ps"] = (("lev",), [100000., 90000., 80000.], {"units": "Pa"})
    else:
        dataset["ps"] = dataset.ps.fillna(100000.).astype(int)
    with pytest.raises(ValueError):
        hybrid_pressure(dataset)


def test_coordinate_and_coefficient_bounds_are_never_selected_as_centers():
    dataset = hybrid_dataset()
    dataset["lev_bounds"] = (("lev", "nv"), [[.5, 1.5], [1.5, 2.5], [2.5, 3.5]])
    dataset.lev.attrs["bounds"] = "lev_bounds"
    with pytest.raises(ValueError, match="bounds/interface"):
        hybrid_pressure(dataset, coordinate="lev_bounds")
    specification = resolve_pressure(dataset)
    assert "lev_bounds" in specification.structural_names
    dataset.a.attrs["bounds"] = "a_bounds"
    dataset["a_bounds"] = (("lev",), [.1, .2, .3], {"units": "1"})
    dataset.lev.attrs["formula_terms"] = "a: a_bounds b: b p0: p0 ps: ps"
    with pytest.raises(ValueError, match="bounds/interfaces"):
        hybrid_pressure(dataset)


def test_hybrid_surface_override_must_be_its_cf_reference():
    dataset = hybrid_dataset()
    dataset["other_ps"] = dataset.ps
    resolve_pressure(dataset, surface_pressure="ps")
    with pytest.raises(ValueError, match="exact CF ps"):
        resolve_pressure(dataset, surface_pressure="other_ps")
    with pytest.raises(TypeError, match="variable-name"):
        resolve_pressure(dataset, surface_pressure=dataset.ps)


def fixed_dataset(levels=(1000., 500., 100.), units="hPa"):
    return xr.Dataset(coords={"level": ("level", np.asarray(levels), {"units": units, "standard_name": "air_pressure", "axis": "Z"})})


def test_fixed_pressure_retains_original_order_and_units_are_converted():
    dataset = fixed_dataset()
    specification = resolve_pressure(dataset)
    assert specification.kind == "fixed"
    assert specification.vertical_dim == "level"
    np.testing.assert_array_equal(specification.pressure, [100000., 50000., 10000.])
    np.testing.assert_array_equal(specification.pressure.level, [1000., 500., 100.])
    assert specification.pressure.attrs["units"] == "Pa"
    assert specification.surface_pressure is None
    assert dataset.level.attrs["units"] == "hPa"


def test_fixed_center_resolution_does_not_read_or_interpolate_bounds():
    dataset = fixed_dataset()

    @delayed
    def bounds():
        raise AssertionError("Interface geometry must not be read as cell centers")

    dataset["level_bounds"] = (("level", "nv"), da.from_delayed(bounds(), shape=(3, 2), dtype=float))
    dataset.level.attrs["bounds"] = "level_bounds"
    specification = resolve_pressure(dataset)
    assert "level_bounds" in specification.structural_names
    np.testing.assert_array_equal(specification.pressure, [100000., 50000., 10000.])


@pytest.mark.parametrize("levels", [(1000.,), (1000., 1000.), (1000., -100.), (1000., np.nan), (1000., 100., 500.)])
def test_bad_fixed_source_levels_are_rejected(levels):
    with pytest.raises(ValueError):
        resolve_pressure(fixed_dataset(levels))


def test_fixed_with_explicit_surface_pressure_and_ambiguous_combined_axes():
    dataset = fixed_dataset()
    dataset["ps"] = xr.DataArray(1000., attrs={"units": "hPa"})
    specification = resolve_pressure(dataset, surface_pressure="ps")
    assert isinstance(specification.surface_pressure.data, da.Array)
    assert float(specification.surface_pressure.compute()) == 100000.
    hybrid = hybrid_dataset().assign_coords(level=dataset.level)
    with pytest.raises(ValueError, match="Expected one fixed or hybrid"):
        resolve_pressure(hybrid)
    assert resolve_pressure(hybrid, coordinate="lev").kind == "hybrid"
    assert resolve_pressure(hybrid, coordinate="level").kind == "fixed"
    with pytest.raises(ValueError, match="hybrid_pressure requires"):
        hybrid_pressure(dataset)


def test_target_pressure_preserves_order_and_supports_one_level():
    result = target_pressure([850., 500., 200.], target_units="hPa")
    np.testing.assert_array_equal(result, [85000., 50000., 20000.])
    assert result.dims == ("pressure",)
    assert result.attrs["units"] == result.pressure.attrs["units"] == "Pa"
    np.testing.assert_array_equal(target_pressure([50000.], target_units="Pa"), [50000.])


@pytest.mark.parametrize("levels", [[], [0.], [np.inf], [500., 500.], [100., 900., 500.], [[500., 700.]]])
def test_invalid_target_geometry_is_rejected(levels):
    with pytest.raises(ValueError):
        target_pressure(levels)


def test_target_dataarray_declared_units_cannot_be_overridden_silently():
    levels = xr.DataArray([500., 700.], dims="lev", attrs={"units": "hPa"})
    with pytest.raises(ValueError, match="disagree"):
        target_pressure(levels, target_units="Pa")
    result = target_pressure(levels, target_units="hPa")
    np.testing.assert_array_equal(result, [50000., 70000.])
