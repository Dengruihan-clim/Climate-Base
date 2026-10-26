"""Analytic checks of the regional Hadley diagnostics, using synthetic data only."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scipy import stats

import hadley_ma2025 as hc


LAT = np.arange(-90.0, 91.0, 30.0)
LON = np.arange(0.0, 360.0, 60.0)
LEVEL = np.array([100.0, 500.0, 1000.0])


def wind_cube(years=range(1980, 1986), amplitudes=None):
    """A complete global, regular grid with a vertically constant divergent V."""
    years = np.asarray(list(years), dtype=int)
    if amplitudes is None:
        amplitudes = np.ones(years.size)
    values = np.broadcast_to(
        np.asarray(amplitudes, dtype=float)[:, None, None, None],
        (years.size, LEVEL.size, LAT.size, LON.size),
    ).copy()
    out = xr.DataArray(
        values,
        dims=("time", "level", "lat", "lon"),
        coords={
            "time": pd.to_datetime([f"{y}-10-15" for y in years]),
            "year": ("time", years),
            "level": LEVEL,
            "lat": LAT,
            "lon": LON,
        },
        name="vd",
        attrs={"units": "m s-1"},
    )
    out.level.attrs["units"] = "hPa"
    out.lat.attrs["units"] = "degrees_north"
    out.lon.attrs["units"] = "degrees_east"
    return out


def predictor(years, values):
    years = np.asarray(list(years), dtype=int)
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims="time",
        coords={
            # Deliberately use a different day: year coordinates control alignment.
            "time": pd.to_datetime([f"{y}-07-01" for y in years]),
            "year": ("time", years),
        },
        name="synthetic_index",
    )


def run_analysis(vd, index, **kwargs):
    settings = {
        "baseline_start": "1980",
        "baseline_end": "1985",
        "detrend_fields": False,
        "detrend_index": False,
        "main_lon_bounds": (0.0, 120.0),
        "sensitivity_lon_bounds": (300.0, 60.0),
    }
    settings.update(kwargs)
    return hc.analyze(vd, index, **settings)


def one_variable(dataset, suffix):
    names = [name for name in dataset.data_vars if name.endswith(suffix) and not name.startswith("index_")]
    assert len(names) == 1, (suffix, list(dataset.data_vars))
    return dataset[names[0]]


def test_constant_wind_integral_has_correct_pressure_and_longitude_factors():
    vd = wind_cube(amplitudes=np.arange(1.0, 7.0))
    result = hc.compute_local_mass_streamfunction(vd)
    radius = getattr(hc, "EARTH_RADIUS", 6.371e6)
    gravity = getattr(hc, "GRAVITY", 9.80665)
    coefficient = radius * np.cos(np.deg2rad(LAT)) * (2 * np.pi / LON.size) / gravity
    expected = (
        np.arange(1.0, 7.0)[:, None, None, None]
        * ((LEVEL - LEVEL[0]) * 100.0)[None, :, None, None]
        * coefficient[None, None, :, None]
    )
    expected = np.broadcast_to(expected, result.shape)
    np.testing.assert_allclose(result.transpose("time", "level", "lat", "lon"), expected, rtol=2e-6, atol=1e-6)
    assert result.attrs["units"] in ("kg s-1", "kg s^-1", "kg/s")
    assert np.all(result.sel(level=100.0) == 0)
    assert np.all(result.sel(level=1000.0, lat=0.0) > 0)


def test_pressure_order_changes_do_not_change_the_integral():
    vd = wind_cube()
    forward = hc.compute_local_mass_streamfunction(vd)
    backward = hc.compute_local_mass_streamfunction(vd.isel(level=slice(None, None, -1)))
    xr.testing.assert_allclose(forward.sortby("level"), backward.sortby("level"))


def test_explicit_pascal_pressure_matches_hectopascal_pressure():
    vd = wind_cube()
    pa = vd.assign_coords(level=LEVEL * 100)
    pa.level.attrs["units"] = "Pa"
    xr.testing.assert_allclose(hc.compute_local_mass_streamfunction(vd), hc.compute_local_mass_streamfunction(pa))


def test_analysis_pascal_pressure_does_not_create_a_mixed_unit_level_union():
    vd = wind_cube(amplitudes=[2, 3, 4, 2, 5, 1])
    pa = vd.assign_coords(level=LEVEL * 100)
    pa.level.attrs["units"] = "Pa"
    index = predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2])
    reference, converted = run_analysis(vd, index), run_analysis(pa, index)
    for key in ("psi", "intensity", "regional"):
        xr.testing.assert_allclose(reference[key], converted[key])
    assert converted["psi"].sizes["level"] == LEVEL.size
    np.testing.assert_array_equal(converted["psi"].level, LEVEL)
    assert np.isfinite(converted["psi"]["vd"]).all()
    assert np.isfinite(converted["psi"]["psi"]).all()


@pytest.mark.parametrize("bounds,count", [((0.0, 120.0), 3), ((300.0, 60.0), 3)])
def test_regional_sum_is_grid_count_times_mean_including_wrapped_sector(bounds, count):
    psi = hc.compute_local_mass_streamfunction(wind_cube())
    psi = psi * xr.DataArray(np.arange(1.0, LON.size + 1), dims="lon", coords={"lon": LON})
    mean = hc.regional_reduce_psi(psi, lon_bounds=bounds, reduction="mean")
    total = hc.regional_reduce_psi(psi, lon_bounds=bounds, reduction="sum")
    np.testing.assert_allclose(total, mean * count, rtol=2e-6, atol=1e-5)
    selected = [0, 1, 2] if bounds == (0.0, 120.0) else [0, 1, 5]
    np.testing.assert_allclose(mean, psi.isel(lon=selected).mean("lon"))
    assert "lon" not in mean.dims
    assert set(mean.dims) == {"time", "level", "lat"}


def test_horizontal_intensity_uses_absolute_value_before_vertical_maximum():
    psi = hc.compute_local_mass_streamfunction(wind_cube())
    psi.loc[{"level": 500.0}] = -2 * psi.sel(level=1000.0)
    intensity = hc.compute_hc_intensity(psi)
    expected = 2 * np.abs(psi.sel(level=1000.0))
    np.testing.assert_allclose(intensity, expected)


def test_horizontal_response_takes_maximum_before_regression():
    x = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    vd = wind_cube()
    # The integrated profiles are [0, 2+x, 2-x]. Their maximum absolute
    # value is 2+abs(x), which has zero regression on symmetric x.
    integral = np.column_stack([np.zeros(x.size), 2 + x, 2 - x])
    dp = np.diff(LEVEL) * 100.0
    nodes = np.zeros_like(integral)
    nodes[:, 1] = 2 * integral[:, 1] / dp[0]
    nodes[:, 2] = 2 * (integral[:, 2] - integral[:, 1]) / dp[1] - nodes[:, 1]
    vd.values[:] = nodes[:, :, None, None]
    output = run_analysis(vd, predictor(range(1980, 1986), x))
    regression = one_variable(output["intensity"], "regression")
    assert abs(float(regression.sel(lat=0.0).mean())) < 1e-6


@pytest.mark.parametrize("detrend", [False, True])
def test_predictor_is_aligned_then_detrended_and_standardized_on_common_years(detrend):
    years = np.arange(1980, 1986)
    common = np.array([1.0, 3.0, 4.0, -2.0, 2.0, 8.0])
    index = predictor(np.arange(1970, 1986), np.r_[np.arange(10) * 1000.0, common])
    output = run_analysis(wind_cube(amplitudes=20 + common), index, detrend_index=detrend)
    actual = output["predictor"]
    expected = common.copy()
    if detrend:
        fit = np.polyfit(np.arange(common.size), common, 1)
        expected -= np.polyval(fit, np.arange(common.size))
    expected = (expected - expected.mean()) / expected.std(ddof=1)
    assert actual.sizes["time"] == years.size
    np.testing.assert_array_equal(actual.year, years)
    np.testing.assert_allclose(actual, expected, atol=1e-8)
    assert abs(float(actual.mean())) < 1e-10
    assert float(actual.std(ddof=1)) == pytest.approx(1.0)


def test_regression_amplitude_is_per_common_period_sample_standard_deviation():
    x = np.array([1.0, 3.0, 4.0, -2.0, 2.0, 8.0])
    output = run_analysis(wind_cube(amplitudes=20 + x), predictor(range(1980, 1986), x))
    coefficient = getattr(hc, "EARTH_RADIUS", 6.371e6) * (2 * np.pi / LON.size) / getattr(hc, "GRAVITY", 9.80665)
    expected = coefficient * (LEVEL[-1] - LEVEL[0]) * 100.0 * x.std(ddof=1)
    regression = one_variable(output["intensity"], "regression")
    np.testing.assert_allclose(regression.sel(lat=0.0), expected, rtol=2e-6)


@pytest.mark.parametrize("values", [np.ones(6), np.full(6, np.nan)])
def test_invalid_predictor_variance_is_rejected(values):
    with pytest.raises(ValueError):
        run_analysis(wind_cube(), predictor(range(1980, 1986), values))


def test_less_than_three_common_years_is_rejected():
    with pytest.raises(ValueError):
        run_analysis(wind_cube(), predictor([1980, 1981], [0.0, 1.0]))


@pytest.mark.parametrize("defect", ["missing_longitude", "duplicate_longitude", "duplicate_pressure", "nan"])
def test_invalid_global_wind_inputs_are_rejected(defect):
    vd = wind_cube()
    if defect == "missing_longitude":
        vd = vd.isel(lon=slice(0, -1))
    elif defect == "duplicate_longitude":
        vd = vd.assign_coords(lon=[0, 60, 120, 180, 240, 240])
    elif defect == "duplicate_pressure":
        vd = vd.assign_coords(level=[100, 500, 500])
        vd.level.attrs["units"] = "hPa"
    else:
        vd.values[0, 1, 3, 0] = np.nan
    with pytest.raises(ValueError):
        hc.compute_local_mass_streamfunction(vd)


def test_latitude_outside_physical_range_is_rejected_by_the_core():
    vd = wind_cube().assign_coords(lat=np.arange(-120.0, 121.0, 40.0))
    with pytest.raises(ValueError, match="(?i)lat|90|range"):
        hc.compute_local_mass_streamfunction(vd)


@pytest.mark.parametrize("units", [None, "knots"])
@pytest.mark.parametrize("entrypoint", ["integral", "analysis"])
def test_unknown_or_missing_wind_speed_units_are_rejected(units, entrypoint):
    vd = wind_cube()
    vd.attrs = {} if units is None else {"units": units}
    with pytest.raises(ValueError, match="(?i)wind|unit|m/s|m s"):
        if entrypoint == "integral":
            hc.compute_local_mass_streamfunction(vd)
        else:
            run_analysis(vd, predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2]))


def test_common_dimension_names_and_pascal_pressure_are_normalized():
    ds = wind_cube().to_dataset(name="v").rename({"lat": "latitude", "lon": "longitude", "level": "plev"})
    ds = ds.assign_coords(plev=LEVEL * 100.0)
    ds.plev.attrs["units"] = "Pa"
    result = hc.normalize_wind_dataset(ds)
    assert {"lat", "lon", "level", "time"}.issubset(result.dims)
    np.testing.assert_allclose(np.sort(result.level), LEVEL)


def test_pressure_unit_is_required_when_normalizing_external_data():
    ds = wind_cube().to_dataset(name="v")
    ds.level.attrs = {}
    with pytest.raises(ValueError, match="(?i)pressure|unit"):
        hc.normalize_wind_dataset(ds)


def test_paired_regression_matches_scipy_and_marks_undefined_tests():
    index = hc.standardize_series(predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2]))
    field = xr.DataArray(
        np.column_stack([[1, 2, 4, 8, 16, 32], np.ones(6), [np.nan, 2, 3, 4, 5, 6]]),
        dims=("time", "lat"),
        coords={"time": index.time, "year": index.year, "lat": [0, 10, 20]},
        attrs={"units": "kg s-1"},
    )
    result = hc.regress_to_nino34(field, index, alpha=0.1)
    for lat in (0, 20):
        y = field.sel(lat=lat).values
        valid = np.isfinite(y)
        expected = stats.linregress(index.values[valid], y[valid])
        assert float(result["regression"].sel(lat=lat)) == pytest.approx(expected.slope)
        assert float(result["pvalue"].sel(lat=lat)) == pytest.approx(expected.pvalue)
        assert int(result["sample_count"].sel(lat=lat)) == int(valid.sum())
    assert np.isnan(result["pvalue"].sel(lat=10))
    assert not bool(result["significance"].sel(lat=10))


def monthly_dataset(times):
    base = wind_cube(years=[1980]).isel(time=0, drop=True)
    values = np.broadcast_to(np.arange(1.0, len(times) + 1)[:, None, None, None], (len(times), LEVEL.size, LAT.size, LON.size)).copy()
    vd = xr.DataArray(values, dims=("time", "level", "lat", "lon"), coords={"time": pd.to_datetime(times), "level": LEVEL, "lat": LAT, "lon": LON}, attrs=base.attrs)
    vd.level.attrs["units"] = "hPa"
    return vd.to_dataset(name="v")


def test_monthly_seasonal_means_use_actual_month_lengths():
    ds = monthly_dataset(["1980-09-01", "1980-10-01", "1980-11-01"])
    result = hc.seasonalize_winds(ds, season="SON", frequency="monthly")
    expected = (30 * 1 + 31 * 2 + 30 * 3) / 91
    np.testing.assert_allclose(result["v"], expected)
    assert result.sizes["time"] == 1


def test_djf_assigns_december_to_the_following_year_and_weights_leap_february():
    ds = monthly_dataset(["1979-12-01", "1980-01-01", "1980-02-01"])
    result = hc.seasonalize_winds(ds, season="DJF", frequency="monthly")
    np.testing.assert_allclose(result["v"], (31 * 1 + 31 * 2 + 29 * 3) / 91)
    np.testing.assert_array_equal(result.year, [1980])


def test_incomplete_season_is_not_used_as_a_complete_season():
    ds = monthly_dataset(["1980-09-01", "1980-10-01", "1980-11-01", "1981-09-01", "1981-10-01"])
    try:
        result = hc.seasonalize_winds(ds, season="SON", frequency="monthly")
    except ValueError as exc:
        assert any(word in str(exc).lower() for word in ("complete", "month", "season", "missing"))
    else:
        assert result.sizes["time"] == 1
        assert pd.to_datetime(result.time.values).year.tolist() == [1980]


def test_output_metadata_does_not_propagate_input_source_information():
    vd = wind_cube(amplitudes=[2, 3, 4, 2, 5, 1])
    vd.attrs["source"] = "PRIVATE_SOURCE_MARKER"
    vd.encoding["source"] = "PRIVATE_SOURCE_MARKER"
    output = run_analysis(vd, predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2]))
    for key in ("psi", "intensity", "regional"):
        ds = output[key]
        metadata = {"dataset": ds.attrs, "variables": {name: var.attrs for name, var in ds.variables.items()}}
        assert "PRIVATE_SOURCE_MARKER" not in json.dumps(metadata, default=str)


@pytest.mark.parametrize("coordinate_kind", ["time", "scalar"])
def test_auxiliary_source_coordinates_are_removed_from_every_output(coordinate_kind):
    vd = wind_cube(amplitudes=[2, 3, 4, 2, 5, 1])
    coordinate = ("time", ["PRIVATE_SOURCE_MARKER"] * vd.sizes["time"]) if coordinate_kind == "time" else "PRIVATE_SOURCE_MARKER"
    vd = vd.assign_coords(source_path=coordinate)
    allowed = {"time", "year", "level", "lat", "lon"}

    integrated = hc.compute_local_mass_streamfunction(vd)
    assert set(integrated.coords).issubset(allowed)
    results = run_analysis(vd, predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2]))
    for key in ("psi", "intensity", "regional"):
        assert set(results[key].coords).issubset(allowed)
    # Serialization must also sanitize a caller-created Dataset rather than
    # rely only on an earlier analysis having removed auxiliary coordinates.
    unsafe = integrated.to_dataset(name="psi").assign_coords(source_path=coordinate)
    serialized = hc._safe_netcdf(unsafe)
    assert set(serialized.coords).issubset(allowed)


@pytest.mark.parametrize("reverse", [False, True])
def test_edge_zero_crossing_is_interpolated_independently_of_latitude_order(reverse):
    lat = np.array([0, 10, 20, 30, 40], dtype=float)
    values = np.array([2, 5, 4, 1, -3], dtype=float)
    if reverse:
        lat, values = lat[::-1], values[::-1]
    core, edge = hc.find_hc_edge_latitude(lat, values)
    assert core == pytest.approx(10.0)
    assert edge == pytest.approx(32.5)


def test_edge_zero_crossing_does_not_bridge_missing_latitudes():
    core, edge = hc.find_hc_edge_latitude(
        np.array([0, 10, 20, 30, 40], dtype=float),
        np.array([2, 5, 4, np.nan, -3], dtype=float),
    )
    assert core == pytest.approx(10.0)
    assert np.isnan(edge)


def test_missing_edges_produce_zero_samples_and_no_claim_of_significance():
    edges = pd.DataFrame({"year": np.arange(1980, 1986), "hc_edge_latitude_deg": np.full(6, np.nan)})
    index = predictor(range(1980, 1986), [-2, 1, 0, -1, 3, 2])
    _, result = hc.edge_regression_stats(edges, index)
    assert int(result.loc[0, "n_samples"]) == 0
    assert np.isnan(result.loc[0, "correlation_pvalue"])
    assert np.isnan(result.loc[0, "regression_pvalue"])
    assert not result.loc[0, "significant"]
