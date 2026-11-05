"""Real MetPy pressure kernels plus optional GeoCAT public-API comparisons."""

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import interpolate_pressure, open_dataset, validate_metadata, write_dataset
from climate_preprocessing.vertical import _metpy

pytestmark = pytest.mark.vertical


@pytest.fixture(scope="module", autouse=True)
def real_metpy():
    try:
        return _metpy.backend_versions()
    except ImportError as error:
        pytest.skip(f"Real optional MetPy backend unavailable: {error}")


def analytic(pressure_pa, method):
    pressure_pa = np.asarray(pressure_pa, dtype=float)
    return 10 + 0.001 * pressure_pa if method == "linear" else 3 + 2 * np.log(pressure_pa)


def fixed_dataset(method="linear", levels=(100, 300, 600, 900)):
    dataset = xr.Dataset(coords={
        "level": ("level", np.array(levels, dtype=float),
                  {"standard_name": "air_pressure", "units": "hPa", "positive": "down", "axis": "Z"}),
    })
    dataset["temperature"] = ("level", analytic(np.array(levels) * 100, method), {"units": "K"})
    return dataset


def hybrid_dataset(formula="a", method="linear"):
    a = np.array([0.05, 0.10, 0.15, 0.20])
    b = np.array([0.10, 0.30, 0.50, 0.70])
    ps_hpa = np.array([[900, 1000], [950, 800]], dtype=float)
    levels = np.arange(4, dtype=float)
    terms = "a: a b: b ps: ps p0: p0" if formula == "a" else "ap: ap b: b ps: ps"
    dataset = xr.Dataset(coords={
        "level": ("level", levels, {"standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                                     "units": "1", "positive": "down", "axis": "Z", "formula_terms": terms}),
        "time": np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[ns]"),
        "column": [0, 1],
    })
    dataset["ps"] = (("time", "column"), ps_hpa, {"units": "hPa", "standard_name": "surface_air_pressure"})
    dataset["b"] = ("level", b, {"units": "1"})
    if formula == "a":
        dataset["a"] = ("level", a, {"units": "1"})
        dataset["p0"] = xr.DataArray(100000.0, attrs={"units": "Pa"})
    else:
        dataset["ap"] = ("level", a * 100000.0, {"units": "Pa"})
    truth_pa = a[None, :, None] * 100000.0 + b[None, :, None] * ps_hpa[:, None, :] * 100.0
    dataset["temperature"] = (("time", "level", "column"), analytic(truth_pa, method), {"units": "K"})
    return dataset


@pytest.mark.parametrize("method", ["linear", "log"])
@pytest.mark.parametrize("source_descending", [False, True])
@pytest.mark.parametrize("target_descending", [False, True])
def test_fixed_pressure_analytic_with_both_axis_orders(method, source_descending, target_descending):
    dataset = fixed_dataset(method)
    if source_descending:
        dataset = dataset.isel(level=slice(None, None, -1))
    targets = np.array([150, 450, 800], dtype=float)
    if target_descending:
        targets = targets[::-1]
    result = interpolate_pressure(dataset, targets, variables="temperature", method=method)
    assert isinstance(result.temperature.data, da.Array)
    np.testing.assert_array_equal(result.pressure, targets)
    np.testing.assert_allclose(result.temperature.compute(), analytic(targets * 100, method), rtol=2e-14)
    assert result.pressure.attrs["units"] == "hPa" and result.temperature.attrs["units"] == "K"


@pytest.mark.parametrize("method", ["linear", "log"])
def test_exact_endpoints_survive_without_extrapolation(method):
    dataset = fixed_dataset(method)
    targets = np.array([50, 100, 200, 900, 950], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method=method)
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[[0, 4]]).all()
    np.testing.assert_allclose(actual[1:4], analytic(targets[1:4] * 100, method), rtol=2e-14)


@pytest.mark.parametrize("method", ["linear", "log"])
@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_missing_original_level_blocks_only_adjacent_brackets_and_exact_hits_survive(method, missing):
    dataset = fixed_dataset(method)
    dataset.temperature.data[1] = missing
    targets = np.array([100, 200, 300, 450, 600, 750, 900], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method=method)
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[[1, 2, 3]]).all()
    valid = [0, 4, 5, 6]
    np.testing.assert_allclose(actual[valid], analytic(targets[valid] * 100, method), rtol=2e-14)


@pytest.mark.parametrize("surface_hpa", [700.0, 50.0, np.nan])
def test_fixed_surface_pressure_forbids_belowground_contributors_and_lower_extrapolation(surface_hpa):
    dataset = fixed_dataset()
    dataset["ps"] = xr.DataArray(surface_hpa, attrs={"units": "hPa"})
    targets = [100, 200, 600, 650, 700, 800, 900]
    result = interpolate_pressure(dataset, targets, variables="temperature", method="linear", surface_pressure="ps")
    actual = result.temperature.compute().to_numpy()
    if surface_hpa == 700:
        np.testing.assert_allclose(actual[:3], analytic(np.array(targets[:3]) * 100, "linear"))
        # 650/700 are physically above/on the ground but have no two valid
        # above-ground source centers. They must not use the 900-hPa level.
        assert np.isnan(actual[3:]).all()
    else:
        assert np.isnan(actual).all()


@pytest.mark.parametrize("surface_hpa", [0.0, -1.0, np.inf, -np.inf])
def test_invalid_surface_pressure_raises_only_at_explicit_compute(surface_hpa):
    dataset = fixed_dataset()
    dataset["ps"] = xr.DataArray(da.from_array(np.array(surface_hpa)), attrs={"units": "hPa"})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = interpolate_pressure(dataset, [200, 500], variables="temperature", method="linear",
                                      surface_pressure="ps")
    assert tasks == []
    with pytest.raises(ValueError, match="[Ss]urface pressure|surface_pressure"):
        result.temperature.compute()


@pytest.mark.parametrize("formula", ["a", "ap"])
@pytest.mark.parametrize("method", ["linear", "log"])
def test_both_cf_hybrid_formulas_against_pressure_analytic_truth(formula, method):
    dataset = hybrid_dataset(formula, method)
    targets = np.array([200, 500, 700], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method=method)
    expected = np.broadcast_to(analytic(targets * 100, method)[None, :, None], (2, 3, 2))
    np.testing.assert_allclose(result.temperature.compute(), expected, rtol=2e-14)
    assert result.temperature.dims == ("time", "pressure", "column")
    assert not {"a", "ap", "b", "p0", "level"}.intersection(result.variables)
    assert "ps" in result


def test_hybrid_descending_source_target_and_missing_surface_column():
    dataset = hybrid_dataset("ap", "log").isel(level=slice(None, None, -1))
    dataset.ps.data[0, 1] = np.nan
    targets = np.array([700, 500, 200], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method="log")
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[0, :, 1]).all()
    expected = analytic(targets * 100, "log")
    np.testing.assert_allclose(actual[0, :, 0], expected, rtol=2e-14)
    np.testing.assert_allclose(actual[1], np.broadcast_to(expected[:, None], (3, 2)), rtol=2e-14)


@pytest.mark.parametrize("a", [[0.1, 0.3, 0.2, 0.4], [0.1, 0.2, 0.2, 0.4]])
def test_invalid_hybrid_pressure_columns_are_not_silently_sorted(a):
    dataset = hybrid_dataset("a")
    dataset["a"] = ("level", a, {"units": "1"})
    dataset["b"] = ("level", np.zeros(4), {"units": "1"})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = interpolate_pressure(dataset, [150, 250], variables="temperature", method="linear")
    assert tasks == []
    with pytest.raises(ValueError, match="monotonic|duplicate"):
        result.temperature.compute()


def test_vertical_core_rechunk_preserves_other_axis_positions_chunks_and_lazy_tasks():
    dataset = fixed_dataset().drop_vars("temperature").assign_coords(column=[0, 1, 2], time=[0, 1])
    expected_source = np.broadcast_to(analytic(dataset.level.to_numpy() * 100, "linear")[None, :, None], (3, 4, 2))
    data = da.from_delayed(delayed(lambda: expected_source)(), shape=(3, 4, 2), dtype=float).rechunk((2, 2, 1))
    dataset["temperature"] = (("column", "level", "time"), data, {"units": "K"})
    dataset = dataset.assign_coords(location=("column", [10, 20, 30]), old_vertical_aux=("level", [1, 2, 3, 4]))
    before = dataset.copy(deep=False)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = interpolate_pressure(dataset, [200, 500, 700], variables="temperature", method="linear")
    assert tasks == []
    assert result.temperature.dims == ("column", "pressure", "time")
    assert result.temperature.chunks == ((2, 1), (3,), (1, 1))
    assert "location" in result.coords and "old_vertical_aux" not in result.variables
    assert dataset.identical(before)
    expected = np.broadcast_to(analytic(np.array([200, 500, 700]) * 100, "linear")[None, :, None], (3, 3, 2))
    np.testing.assert_allclose(result.temperature.compute(), expected)


def test_hybrid_geometry_chunks_follow_permuted_field_without_splitting_nonvertical_chunks():
    dataset = hybrid_dataset("ap", "log")
    dataset["temperature"] = dataset.temperature.transpose("column", "level", "time").chunk(
        {"column": 1, "level": 2, "time": 1})
    dataset["ps"] = dataset.ps.chunk({"time": 2, "column": 2})
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = interpolate_pressure(dataset, [200, 500, 700], variables="temperature", method="log")
    assert tasks == []
    assert result.temperature.dims == ("column", "pressure", "time")
    assert result.temperature.chunks == ((1, 1), (3,), (1, 1))
    expected = np.broadcast_to(analytic(np.array([200, 500, 700]) * 100, "log")[None, :, None], (2, 3, 2))
    np.testing.assert_allclose(result.temperature.compute(), expected, rtol=2e-14)


def test_canonical_pressure_source_axis_collision_preserves_requested_pa_levels():
    dataset = fixed_dataset().rename(level="pressure")
    targets = np.array([70000, 50000, 20000], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method="linear", target_units="Pa")
    np.testing.assert_array_equal(result.pressure, targets)
    np.testing.assert_allclose(result.temperature.compute(), analytic(targets, "linear"))
    assert result.temperature.dims == ("pressure",)


def test_cf_calendar_and_pressure_metadata_survive_real_netcdf_writeback(tmp_path):
    import cftime

    dataset = fixed_dataset().expand_dims(time=[cftime.Datetime360Day(2001, 2, 30)])
    dataset["level_bounds"] = (("level", "bounds"), [[50, 200], [200, 450], [450, 750], [750, 1000]])
    dataset.level.attrs["bounds"] = "level_bounds"
    dataset.temperature.attrs["cell_methods"] = "time: mean level: mean"
    dataset.temperature.encoding = {"scale_factor": 0.1, "add_offset": 0, "dtype": "int16",
                                     "coordinates": "time level"}
    result = interpolate_pressure(dataset, [200, 500], variables="temperature", method="linear")
    assert "level_bounds" not in result.variables
    assert result.temperature.attrs["cell_methods"] == "time: mean"
    assert result.temperature.encoding == {}
    validate_metadata(result)
    path = tmp_path / "pressure.nc"
    write_dataset(result, path)
    with open_dataset(path) as reopened:
        assert isinstance(reopened.time.to_numpy()[0], cftime.Datetime360Day)
        assert reopened.time.to_numpy()[0] == dataset.time.to_numpy()[0]
        assert reopened.pressure.attrs["units"] == "hPa"
        assert "formula_terms" not in reopened.pressure.attrs
        np.testing.assert_allclose(reopened.temperature.compute(), [[30, 60]])


@pytest.mark.geocat
@pytest.mark.parametrize("formula", ["a", "ap"])
@pytest.mark.parametrize("method", ["linear", "log"])
def test_optional_geocat_public_hybrid_reference(formula, method):
    geocat = pytest.importorskip("geocat.comp", reason="Optional GeoCAT reference is not a runtime dependency")
    dataset = hybrid_dataset(formula, method)
    targets = np.array([200, 500, 700], dtype=float)
    result = interpolate_pressure(dataset, targets, variables="temperature", method=method)
    reference = geocat.interp_hybrid_to_pressure(
        dataset.temperature, dataset.ps * 100.0,
        dataset.a if formula == "a" else dataset.ap, dataset.b,
        p0=100000.0 if formula == "a" else 1.0, new_levels=targets * 100,
        lev_dim="level", method=method, extrapolate=False,
    )
    reference = reference.transpose("time", "plev", "column")
    np.testing.assert_allclose(result.temperature.compute(), reference.to_numpy(), rtol=2e-14)
