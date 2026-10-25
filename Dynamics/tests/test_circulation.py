"""Analytic circulation, longitude geometry and fixed-support checks."""
import numpy as np
import pytest

from climate_dynamics.circulation import (mass_streamfunction, longitude_cell_weights,
                                          fixed_longitude_mean)


def test_linear_pressure_wind_integrates_exactly_from_supplied_reference():
    p = np.array([3000., 5000., 10000., 50000., 100000.])
    lat = np.array([-35., 0., 50.])
    a, b = np.array([1., -.5, 2.]), np.array([2e-5, -1e-5, 3e-5])
    v = a + p[:, None] * b
    radius, gravity = 3., 2.
    expected = (a * (p[:, None] - p[0]) + .5 * b * (p[:, None] ** 2 - p[0] ** 2)) * (2 * np.pi * radius / gravity * np.cos(np.deg2rad(lat)))
    psi, metadata = mass_streamfunction(v, p, lat, radius=radius, gravity=gravity)
    np.testing.assert_allclose(psi, expected, rtol=3e-15, atol=1e-8)
    np.testing.assert_array_equal(psi[0], 0.)
    assert metadata["reference_pressure_pa"] == 3000.
    assert "not integrated" in metadata["upper_boundary_assumption"]
    assert "2pi" in metadata["longitude_geometry"]
    assert metadata["formula_reference"]["doi"] == "10.5194/gmd-11-4339-2018"


def test_no_zero_pressure_assumption_or_extrapolation_is_silently_added():
    p = np.array([1000., 3000., 5000.])
    v = np.ones((3, 1))
    psi, _ = mass_streamfunction(v, p, [0.], radius=1., gravity=1.)
    np.testing.assert_allclose(psi[:, 0], 2 * np.pi * (p - 1000.))
    with pytest.raises(ValueError, match="first input pressure"):
        mass_streamfunction(v, p, [0.], reference_pressure_pa=0.)
    explicit, _ = mass_streamfunction(v, p, [0.], reference_pressure_pa=1000., radius=1., gravity=1.)
    np.testing.assert_array_equal(explicit, psi)


def test_missing_levels_never_restart_integration_and_real_zero_stays_zero():
    p = np.array([0., 1000., 3000., 7000.])
    v = np.ones((2, 4, 3))
    v[0, 2, 0] = np.nan
    v[0, 0, 1] = np.nan
    v[1, :, 2] = 0.
    v[1, 1, 0] = np.inf
    psi, metadata = mass_streamfunction(v, p, [-30., 0., 30.])
    assert np.isfinite(psi[0, :2, 0]).all()
    assert np.isnan(psi[0, 2:, 0]).all()
    assert np.isnan(psi[0, :, 1]).all()
    assert np.isnan(psi[1, 1:, 0]).all()
    np.testing.assert_array_equal(psi[1, :, 2], 0.)
    assert metadata["restart_below_missing"] is False


def test_arbitrary_leading_dimensions_and_explicit_axes():
    rng = np.random.default_rng(1711)
    p, lat = np.array([3000., 10000., 50000.]), np.array([-40., 0., 60., 80.])
    v = rng.normal(size=(2, 5, 3, 4))
    baseline, _ = mass_streamfunction(v, p, lat)
    transposed = v.transpose(2, 0, 1, 3)
    actual, metadata = mass_streamfunction(transposed, p, lat, pressure_axis=0, latitude_axis=3)
    np.testing.assert_array_equal(actual, baseline.transpose(2, 0, 1, 3))
    assert metadata["pressure_axis"] == 0
    assert metadata["latitude_axis"] == 3


def test_descending_latitude_and_exact_polar_zero_without_missing_fill():
    p, lat = [3000., 5000., 10000.], [90., 30., -90.]
    v = np.ones((3, 3))
    v[1, 2] = np.nan
    psi, _ = mass_streamfunction(v, p, lat)
    np.testing.assert_array_equal(psi[:, 0], 0.)
    assert np.isnan(psi[1:, 2]).all()


@pytest.mark.parametrize("change", ["pressure_order", "negative_pressure", "latitude", "axes", "complex", "radius", "gravity"])
def test_mass_input_validation(change):
    kwargs = {}
    v, p, lat = np.ones((3, 2)), np.array([1000., 3000., 5000.]), np.array([-20., 20.])
    if change == "pressure_order":
        p = p[::-1]
    elif change == "negative_pressure":
        p[0] = -1.
    elif change == "latitude":
        lat[0] = 91.
    elif change == "axes":
        kwargs["latitude_axis"] = -2
    elif change == "complex":
        v = v.astype(complex) + 1j
    elif change == "radius":
        kwargs["radius"] = True
    elif change == "gravity":
        kwargs["gravity"] = 0.
    with pytest.raises(ValueError):
        mass_streamfunction(v, p, lat, **kwargs)


def test_regional_endpoint_and_irregular_native_cell_intersections():
    expected = np.r_[.5, np.ones(9), .5]
    widths, meta = longitude_cell_weights(np.arange(70., 81.), (70., 80.))
    np.testing.assert_allclose(widths, expected, rtol=0, atol=1e-12)
    assert meta["requested_width_degrees"] == 10.
    irregular, _ = longitude_cell_weights([69., 71., 74., 78., 81.], (70., 80.))
    np.testing.assert_allclose(irregular, [0., 2.5, 3.5, 3.5, .5], atol=1e-12)
    descending, _ = longitude_cell_weights([81., 78., 74., 71., 69.], (70., 80.))
    np.testing.assert_allclose(descending, irregular[::-1], atol=1e-12)
    with pytest.raises(ValueError, match="cover"):
        longitude_cell_weights(np.arange(72., 79.), (70., 80.))


def test_date_line_crossing_and_global_cyclic_widths():
    widths, meta = longitude_cell_weights([165., 175., -175., -165.], (170., -170.))
    np.testing.assert_allclose(widths, [0., 10., 10., 0.], atol=1e-12)
    assert meta["requested_width_degrees"] == 20.
    cyclic, _ = longitude_cell_weights([0., 60., 180., 270.], cyclic=True)
    np.testing.assert_allclose(cyclic, [75., 90., 105., 90.], atol=1e-12)
    assert cyclic.sum() == 360.
    wrapped, _ = longitude_cell_weights([-90., 0., 90., 180.], (-180., 180.), cyclic=True)
    np.testing.assert_allclose(wrapped, 90.)


def test_explicit_native_bounds_and_no_double_counting():
    bounds = [[-30., 30.], [30., 150.], [150., 270.], [270., 330.]]
    weights, metadata = longitude_cell_weights([0., 90., 210., 300.], (350., 10.), cell_bounds=bounds)
    np.testing.assert_allclose(weights, [20., 0., 0., 0.])
    assert metadata["cell_edge_method"] == "explicit cell west/east bounds"
    with pytest.raises(ValueError, match="overlap"):
        longitude_cell_weights([0., 90.], cell_bounds=[[-90., 60.], [30., 270.]])
    with pytest.raises(ValueError, match="duplicate"):
        longitude_cell_weights([0., 90., 180., 360.], cyclic=True)
    with pytest.raises(ValueError, match="empty arc"):
        longitude_cell_weights([0., 90., 180., 270.], (0., 0.), cyclic=True)


def test_fixed_support_is_identical_across_all_fields_and_samples():
    lon = np.arange(70., 81.)
    weights = np.r_[.5, np.ones(9), .5]
    times = np.arange(5.)[:, None, None, None]
    a = np.broadcast_to(times + lon, (5, 1, 2, 11)).copy()
    b = 3 * a + 1
    a[0, 0, 0, 0] = np.nan
    b[-1, 0, 0, 1] = np.nan
    a[2, 0, 1, 2:5] = np.nan
    means, fraction, mask, meta = fixed_longitude_mean({"a": a, "b": b}, lon, longitude_bounds=(70., 80.))
    np.testing.assert_allclose(fraction, [[.85, .7]], atol=1e-14)
    np.testing.assert_array_equal(mask[0, 0], [False, False] + [True] * 9)
    expected_lon = (lon[2:] * weights[2:]).sum() / 8.5
    np.testing.assert_allclose(means["a"][:, 0, 0], np.arange(5.) + expected_lon)
    np.testing.assert_allclose(means["b"][:, 0, 0], 3 * (np.arange(5.) + expected_lon) + 1)
    assert np.isnan(means["a"][:, 0, 1]).all()
    assert np.isnan(means["b"][:, 0, 1]).all()
    assert "2pi" in meta["regional_streamfunction_interpretation"]


def test_weighted_threshold_is_inclusive_and_zero_is_observed():
    values = np.zeros((4, 1, 1, 11))
    values[0, 0, 0, 3:5] = np.nan
    weights = np.r_[.5, np.ones(9), .5]
    means, fraction, _, _ = fixed_longitude_mean({"v": values}, weights=weights, min_fraction=.8)
    assert fraction.item() == .8
    np.testing.assert_array_equal(means["v"], 0.)
    rejected, _, _, _ = fixed_longitude_mean({"v": values}, weights=weights, min_fraction=.81)
    assert np.isnan(rejected["v"]).all()
    missing = np.full((4, 1, 1, 11), np.nan)
    no_mean, no_fraction, no_mask, _ = fixed_longitude_mean({"v": missing}, weights=weights)
    assert np.isnan(no_mean["v"]).all()
    assert no_fraction.item() == 0.
    assert not no_mask.any()


def test_multiple_sample_axes_and_nonlast_longitude_preserve_output_axes():
    values = np.arange(2 * 3 * 2 * 1 * 4, dtype=float).reshape(2, 3, 2, 1, 4)
    values[0, 1, 0, 0, 0] = np.nan
    fields = {"a": values, "b": -2 * values}
    original, support, mask, _ = fixed_longitude_mean(fields, weights=np.ones(4), min_fraction=.5, sample_axes=(0, 1))
    changed_fields = {name: array.transpose(4, 2, 0, 1, 3) for name, array in fields.items()}
    changed, changed_support, changed_mask, metadata = fixed_longitude_mean(changed_fields, weights=np.ones(4),
                            min_fraction=.5, longitude_axis=0, sample_axes=(2, 3))
    np.testing.assert_array_equal(changed["a"], original["a"].transpose(2, 0, 1, 3))
    np.testing.assert_array_equal(changed_support, support)
    np.testing.assert_array_equal(changed_mask, mask)
    assert metadata["fixed_mask_axes"] == [1, 4, 0]


def test_static_fields_can_declare_no_sample_axes():
    a = np.array([[0., 1., 2., 3.], [4., 5., np.nan, 7.]])
    means, support, _, _ = fixed_longitude_mean({"a": a}, weights=np.ones(4), sample_axes=(), min_fraction=.5)
    np.testing.assert_allclose(means["a"], [1.5, 16 / 3])
    np.testing.assert_allclose(support, [1., .75])


@pytest.mark.parametrize("case", ["no_weights", "both", "negative", "min_fraction", "boolean_fraction", "sample_longitude", "wrong_shape", "wrong_arc"])
def test_fixed_mean_contract_rejects_ambiguous_or_invalid_inputs(case):
    field = np.ones((3, 2, 4))
    kwargs = {"weights": np.ones(4)}
    if case == "no_weights":
        kwargs = {}
    elif case == "both":
        kwargs["longitude"] = np.arange(4) * 90.
    elif case == "negative":
        kwargs["weights"] = -np.ones(4)
    elif case == "min_fraction":
        kwargs["min_fraction"] = 0.
    elif case == "boolean_fraction":
        kwargs["min_fraction"] = True
    elif case == "sample_longitude":
        kwargs["sample_axes"] = (-1,)
    elif case == "wrong_shape":
        kwargs["weights"] = np.ones(5)
    elif case == "wrong_arc":
        kwargs["longitude_bounds"] = (0., 90.)
    with pytest.raises(ValueError):
        fixed_longitude_mean({"a": field}, **kwargs)


@pytest.mark.parametrize("kind", ["boolean_wind", "string_wind", "string_pressure", "boolean_latitude", "string_radius", "string_gravity", "string_reference"])
def test_physical_inputs_never_coerce_strings_or_booleans(kind):
    v, p, lat = np.ones((3, 2)), np.array([1000., 3000., 5000.]), np.array([-20., 20.])
    kwargs = {}
    if kind == "boolean_wind":
        v = v.astype(bool)
    elif kind == "string_wind":
        v = v.astype(str)
    elif kind == "string_pressure":
        p = p.astype(str)
    elif kind == "boolean_latitude":
        lat = lat.astype(bool)
    elif kind == "string_radius":
        kwargs["radius"] = "6371000"
    elif kind == "string_gravity":
        kwargs["gravity"] = "9.80665"
    elif kind == "string_reference":
        kwargs["reference_pressure_pa"] = "1000"
    with pytest.raises(ValueError):
        mass_streamfunction(v, p, lat, **kwargs)


def test_finite_explicit_weights_cannot_overflow_their_total_or_exceed_globe():
    with pytest.raises(ValueError, match="finite positive width"):
        fixed_longitude_mean({"v": np.ones((2, 2))}, weights=[1e308, 1e308])
    with pytest.raises(ValueError, match="360 degrees"):
        fixed_longitude_mean({"v": np.ones((2, 2))}, weights=[200., 200.])
