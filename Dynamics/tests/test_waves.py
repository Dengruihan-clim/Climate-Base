"""Analytical, dimensional, finite-variation and missing-stencil wave checks."""
import numpy as np
import pytest

from climate_dynamics.waves import (
    EARTH_RADIUS_M, RWS_KEYS, _three_point_derivative,
    linear_rws, rossby_wave_source, tn_waf,
)


def _components(shape=(3, 4), seed=22):
    random = np.random.default_rng(seed)
    scales = dict(eta=1e-4, div=1e-6, udiv=10., vdiv=10., eta_x=1e-11, eta_y=1e-11)
    return {name: scale*random.normal(size=shape) for name, scale in scales.items()}


def _wave():
    lat = np.arange(-90., 90.1, 2.5)
    lon = np.arange(0., 360., 2.5)
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    psi = 2e7*np.cos(phi)*np.sin(lam)
    return lat, lon, psi, np.full(psi.shape, 20.), np.zeros(psi.shape)


def _literal_rws(components):
    return (-components['eta']*components['div']-components['udiv']*components['eta_x']
            -components['vdiv']*components['eta_y'])


def test_full_source_known_signs_and_term_closure():
    source = {name: np.full((2, 3), value) for name, value in
              dict(eta=2., div=7., udiv=23., vdiv=29., eta_x=17., eta_y=19.).items()}
    result = rossby_wave_source(source)
    assert set(result) == {'stretch', 'advection', 'rws'}
    np.testing.assert_array_equal(result['stretch'], -14.)
    np.testing.assert_array_equal(result['advection'], -942.)
    np.testing.assert_array_equal(result['rws'], -956.)
    np.testing.assert_array_equal(result['rws'], result['stretch']+result['advection'])


def test_linear_source_known_four_terms_and_exact_closure():
    mean = {name: np.full((2, 3), value) for name, value in
            dict(eta=2., div=7., udiv=23., vdiv=29., eta_x=17., eta_y=19.).items()}
    delta = {name: np.full((2, 3), value) for name, value in
             dict(eta=5., div=3., udiv=11., vdiv=13., eta_x=31., eta_y=37.).items()}
    result = linear_rws(mean, delta)
    expected = {'stretch_div': -6., 'stretch_vort': -35.,
                'adv_pert_wind': -434., 'adv_pert_vort': -1786., 'rws': -2261.}
    assert set(result) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(result[name], expected[name])
    np.testing.assert_array_equal(result['rws'], sum(result[name] for name in expected if name != 'rws'))


def test_linear_source_is_first_variation_and_has_correct_quadratic_remainder():
    mean, delta = _components(seed=13), _components(seed=51)
    epsilon = .01
    plus = {name: mean[name]+epsilon*delta[name] for name in RWS_KEYS}
    minus = {name: mean[name]-epsilon*delta[name] for name in RWS_KEYS}
    first_variation = (_literal_rws(plus)-_literal_rws(minus))/(2*epsilon)
    result = linear_rws(mean, delta)
    np.testing.assert_allclose(result['rws'], first_variation, rtol=2e-12, atol=1e-22)
    finite_change = _literal_rws({name: mean[name]+delta[name] for name in RWS_KEYS})-_literal_rws(mean)
    quadratic_remainder = _literal_rws(delta)
    np.testing.assert_allclose(finite_change-result['rws'], quadratic_remainder, rtol=3e-14, atol=1e-24)
    assert np.max(np.abs(quadratic_remainder)) > 1e-12
    doubled = linear_rws(mean, {name: 2*delta[name] for name in RWS_KEYS})
    for name in result:
        np.testing.assert_array_equal(doubled[name], 2*result[name])


@pytest.mark.parametrize('missing_key', RWS_KEYS)
def test_each_source_component_has_common_missing_mask(missing_key):
    fields = _components()
    fields[missing_key][0, 1] = np.nan
    result = rossby_wave_source(fields)
    for values in result.values():
        assert np.isnan(values[0, 1])
        assert np.isfinite(values[1:, :]).all()


@pytest.mark.parametrize('side', ['background', 'perturbation'])
def test_linear_source_masks_all_terms_and_does_not_modify_inputs(side):
    mean, delta = _components(seed=31), _components(seed=88)
    affected = mean if side == 'background' else delta
    affected['eta_x'] = np.ma.array(affected['eta_x'], mask=np.zeros((3, 4), dtype=bool))
    affected['eta_x'].mask[0, 1] = True
    affected['vdiv'][1, 2] = np.inf
    originals = [{name: np.ma.copy(value) for name, value in item.items()} for item in (mean, delta)]
    result = linear_rws(mean, delta)
    for values in result.values():
        assert np.isnan(values[0, 1]) and np.isnan(values[1, 2])
        assert np.isfinite(values[2, 0])
    for item, original in zip((mean, delta), originals):
        for name in RWS_KEYS:
            np.testing.assert_array_equal(item[name], original[name])
            np.testing.assert_array_equal(np.ma.getmaskarray(item[name]), np.ma.getmaskarray(original[name]))


def test_sources_broadcast_leading_dimensions_with_fixed_spatial_grid():
    mean, delta = _components(), _components(shape=(2, 1, 3, 4), seed=41)
    full = rossby_wave_source(delta)
    linear = linear_rws(mean, delta)
    for i in range(2):
        selected = {name: value[i, 0] for name, value in delta.items()}
        for name, values in rossby_wave_source(selected).items():
            np.testing.assert_array_equal(full[name][i, 0], values)
        for name, values in linear_rws(mean, selected).items():
            np.testing.assert_array_equal(linear[name][i, 0], values)


@pytest.mark.parametrize('invalid', [np.ones((2, 4)), np.ones(4), np.ones((0, 3, 4)),
                                   np.ones((3, 4), dtype=complex), np.ones((3, 4), dtype=bool)])
def test_source_rejects_invalid_spatial_shapes_and_nonreal_values(invalid):
    components = dict(_components(), eta=invalid)
    with pytest.raises(ValueError):
        rossby_wave_source(components)
    with pytest.raises(ValueError):
        linear_rws(_components(), components)


def test_source_rejects_missing_keys_and_incompatible_leading_shapes():
    with pytest.raises(ValueError, match='Missing'):
        rossby_wave_source({})
    with pytest.raises(ValueError, match='mapping'):
        linear_rws([], _components())
    with pytest.raises(ValueError, match='broadcast'):
        linear_rws(_components(shape=(2, 3, 4)), _components(shape=(5, 3, 4)))


def test_unrepresentable_source_terms_share_missing_output_mask():
    fields = _components()
    fields['eta'][0, 0], fields['div'][0, 0] = 1e308, 1e308
    for value in rossby_wave_source(fields).values():
        assert np.isnan(value[0, 0])
        assert not np.isinf(value).any()


def test_tn_degree_one_wave_analytic_zonal_flux_radians_and_periodic_seam():
    lat, lon, psi, u, v = _wave()
    result = tn_waf(psi, u, v, lat, lon)
    h = np.deg2rad(2.5)
    expected = (.2*(2e7)**2*np.cos(np.deg2rad(lat))[:, None]
                /(2*EARTH_RADIUS_M**2)*(np.sin(h)/h)**2)
    expected = np.broadcast_to(expected, psi.shape)
    rows = (np.abs(lat) > 10) & (np.abs(lat) < 90)
    assert set(result) == {'waf_x', 'waf_y'}
    np.testing.assert_allclose(result['waf_x'][rows], expected[rows], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(result['waf_y'][rows], 0., atol=2e-10)
    np.testing.assert_allclose(result['waf_x'][rows, 0], result['waf_x'][rows, -1], atol=2e-10)


def test_tn_both_components_match_phase_tilted_wave_formula():
    lat = np.arange(-80., 80.1, 2.)
    lon = np.arange(0., 360., 3.)
    phi, lam = np.deg2rad(lat)[:, None], np.deg2rad(lon)[None, :]
    amplitude, m, k = 7e6, 2, -3
    psi = amplitude*np.sin(m*lam+k*phi)
    u, v = np.full(psi.shape, 15.), np.full(psi.shape, 3.)
    result = tn_waf(psi, u, v, lat, lon, pressure_hpa=350.)
    dlambda, dphi = np.deg2rad(3.), np.deg2rad(2.)
    lambda_factor, phi_factor = np.sin(m*dlambda)/dlambda, np.sin(k*dphi)/dphi
    a = amplitude**2*lambda_factor**2
    b = amplitude**2*lambda_factor*phi_factor
    c = amplitude**2*phi_factor**2
    coefficient = .35*np.cos(phi)/(2*np.hypot(u, v)*EARTH_RADIUS_M**2)
    x = coefficient*(u*a/np.cos(phi)**2+v*b/np.cos(phi))
    y = coefficient*(u*b/np.cos(phi)+v*c)
    rows = (np.abs(lat) > 10) & (np.arange(lat.size) >= 2) & (np.arange(lat.size) < lat.size-2)
    np.testing.assert_allclose(result['waf_x'][rows], x[rows], rtol=5e-12, atol=2e-11)
    np.testing.assert_allclose(result['waf_y'][rows], y[rows], rtol=5e-12, atol=2e-11)


def test_tn_quadratic_amplitude_sign_pressure_and_radius_scaling():
    lat, lon, psi, u, v = _wave()
    base = tn_waf(psi, u, v, lat, lon)
    changes = [(tn_waf(2*psi, u, v, lat, lon), 4.),
               (tn_waf(-psi, u, v, lat, lon), 1.),
               (tn_waf(psi, u, v, lat, lon, pressure_hpa=100.), .5),
               (tn_waf(psi, u, v, lat, lon, radius=2*EARTH_RADIUS_M), .25),
               (tn_waf(psi, 2*u, 2*v, lat, lon), 1.)]
    for result, scale in changes:
        for name in base:
            np.testing.assert_allclose(result[name], scale*base[name], rtol=2e-12, atol=2e-12, equal_nan=True)


def test_tn_equator_poles_weak_wind_total_speed_and_westerly_masks():
    lat, lon, psi, u, v = _wave()
    row = np.flatnonzero(lat == 45.)[0]
    u[row, :5], v[row, :5] = [2., -20., 3., 0., 0.], [0., 0., 4., 10., 0.]
    result = tn_waf(psi, u, v, lat, lon)
    unrestricted = tn_waf(psi, u, v, lat, lon, require_westerly=False, min_speed=0.)
    masked_rows = (np.abs(lat) <= 10) | (np.abs(lat) == 90)
    for name in result:
        assert np.isnan(result[name][masked_rows]).all()
        assert np.isnan(result[name][row, [0, 1, 3, 4]]).all()
        assert np.isfinite(result[name][row, 2])  # speed is exactly 5 m/s
        assert np.isfinite(unrestricted[name][row, [0, 1, 2, 3]]).all()
        assert np.isnan(unrestricted[name][row, 4])  # never divide by zero wind
    np.testing.assert_array_equal(np.isfinite(result['waf_x']), np.isfinite(result['waf_y']))


def test_tn_masked_streamfunction_is_not_filled_and_inputs_are_unchanged():
    lat, lon, psi, u, v = _wave()
    row = np.flatnonzero(lat == 45.)[0]
    psi = np.ma.array(psi, mask=np.zeros(psi.shape, dtype=bool))
    psi.mask[row, 5] = True
    u[row, 10] = np.nan
    psi_original, u_original = psi.copy(), u.copy()
    result = tn_waf(psi, u, v, lat, lon)
    for value in result.values():
        assert np.isnan(value[row, [5, 10]]).all()
        assert np.isfinite(value[row, 25])
    np.testing.assert_array_equal(psi, psi_original)
    np.testing.assert_array_equal(psi.mask, psi_original.mask)
    np.testing.assert_array_equal(u, u_original)


def test_tn_leading_dimensions_match_independent_calls():
    lat, lon, psi, u, v = _wave()
    patterns = psi[None, None]*np.array([1., 2.])[:, None, None, None]
    v_background = np.stack((v, v+3., v+5.))[None]
    result = tn_waf(patterns, u, v_background, lat, lon)
    assert result['waf_x'].shape == (2, 3, lat.size, lon.size)
    for i in range(2):
        for j in range(3):
            expected = tn_waf(patterns[i, 0], u, v_background[0, j], lat, lon)
            for name in result:
                np.testing.assert_array_equal(result[name][i, j], expected[name])


def test_nonuniform_latitude_derivative_is_exact_for_quadratics_and_never_jumps_gaps():
    coordinates = np.array([-2., -1.7, -.8, 0., .2, .9, 2.2, 4.])
    values = coordinates**2+3*coordinates-8
    field = np.broadcast_to(values[None, :, None], (2, coordinates.size, 4)).copy()
    derivative = _three_point_derivative(field, coordinates, axis=-2)
    np.testing.assert_allclose(derivative, np.broadcast_to((2*coordinates+3)[None, :, None], field.shape), atol=2e-14)
    field[:, 3:5] = np.nan
    derivative = _three_point_derivative(field, coordinates, axis=-2)
    assert np.isnan(derivative[:, 3:5]).all()
    assert np.isfinite(derivative[:, [0, 1, 2, 5, 6, 7]]).all()
    # Isolated finite cells have no contiguous three-point stencil.
    field[:, 1:7] = np.nan
    assert np.isnan(_three_point_derivative(field, coordinates, axis=-2)).all()


@pytest.mark.parametrize('kwargs', [dict(pressure_hpa=0.), dict(pressure_hpa=np.nan),
                                   dict(pressure_hpa=[200.]), dict(min_speed=-1.),
                                   dict(min_abs_lat=90.), dict(radius=-1.),
                                   dict(radius=1e300), dict(radius=1e-300),
                                   dict(require_westerly='yes')])
def test_tn_rejects_invalid_scalar_parameters(kwargs):
    lat, lon, psi, u, v = _wave()
    with pytest.raises(ValueError):
        tn_waf(psi, u, v, lat, lon, **kwargs)


def test_tn_rejects_invalid_coordinates_and_incomplete_longitude():
    lat, lon, psi, u, v = _wave()
    invalid = [(psi[::-1], u, v, lat[::-1], lon),
               (psi[:, :-1], u[:, :-1], v[:, :-1], lat, lon[:-1]),
               (psi, u, v, lat, np.linspace(0., 360., lon.size)),
               (psi, u, v, lat, lon-180),
               (psi, u[:-1], v, lat, lon),
               (psi, u, v, lat[:-1], lon)]
    for args in invalid:
        with pytest.raises(ValueError):
            tn_waf(*args)


def test_tn_zero_pattern_is_zero_in_eligible_domain():
    lat, lon, psi, u, v = _wave()
    result = tn_waf(np.zeros(psi.shape), u, v+3., lat, lon)
    rows = (np.abs(lat) > 10) & (np.abs(lat) < 90)
    for value in result.values():
        np.testing.assert_array_equal(value[rows], 0.)
        assert np.isnan(value[~rows]).all()
