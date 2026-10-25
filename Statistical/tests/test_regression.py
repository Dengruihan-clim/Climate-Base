"""Independent SciPy and explicit joint-OLS checks; no external data."""
import numpy as np
import pytest
from scipy import stats

from climate_statistics.regression import linear_regression


def sample():
    rng = np.random.default_rng(821)
    years = np.arange(1980, 2030)
    controls = rng.normal(size=(len(years), 2))
    predictor = .6*controls[:, 0] + rng.normal(size=len(years))*2.5
    field = (3 + .03*(years-2000) + 1.7*controls[:, 0] - .6*controls[:, 1]
             + 2.2*predictor + rng.normal(size=len(years))*.3)
    return years, predictor, controls, field


def test_simple_ols_matches_scipy_and_scalar_output():
    year, x, _, y = sample()
    actual = linear_regression(x, y, year, include_trend=False)
    expected = stats.linregress(x, y)
    for key, value in (('beta', expected.slope), ('intercept', expected.intercept),
                       ('stderr', expected.stderr), ('r', expected.rvalue), ('p', expected.pvalue)):
        np.testing.assert_allclose(actual[key], value, atol=1e-12, rtol=1e-12)
    assert actual['beta'].shape == ()
    assert actual['n'] == len(year)
    assert actual['df'] == len(year)-2
    assert actual['control_beta'].shape == (0,)


def test_multiple_controls_joint_coefficients_partial_r_and_t_p():
    year, x, control, y = sample()
    result = linear_regression(x, y, year, controls=control)
    t = year-year.mean()
    design = np.column_stack((np.ones(len(year)), t, control, x))
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y-design@coefficients
    df = len(year)-design.shape[1]
    covariance = np.linalg.inv(design.T@design) * np.sum(residual**2)/df
    stderr = np.sqrt(covariance[-1, -1])
    nuisance = design[:, :-1]
    xr = x-nuisance@np.linalg.lstsq(nuisance, x, rcond=None)[0]
    yr = y-nuisance@np.linalg.lstsq(nuisance, y, rcond=None)[0]
    expected_r = stats.pearsonr(xr, yr).statistic
    expected_p = 2*stats.t.sf(abs(coefficients[-1]/stderr), df)
    np.testing.assert_allclose(result['beta'], coefficients[-1], rtol=1e-12)
    np.testing.assert_allclose(result['trend_beta'], coefficients[1], atol=1e-12)
    np.testing.assert_allclose(result['control_beta'], coefficients[2:4], atol=1e-12)
    np.testing.assert_allclose(result['intercept'], coefficients[0]-coefficients[1]*year.mean(), atol=1e-10)
    np.testing.assert_allclose(result['stderr_nominal'], stderr, rtol=1e-12)
    np.testing.assert_allclose(result['r'], expected_r, atol=1e-14)
    np.testing.assert_allclose(result['p_nominal'], expected_p, rtol=1e-10, atol=0)
    assert result['df'] == df
    assert result['rank'] == 5


def test_grid_specific_shared_pairing_includes_missing_predictor_and_controls():
    year, x, control, y = sample()
    x[2] = np.nan
    control[5, 1] = np.nan
    field = np.stack((y, y+.1), axis=1)
    field[[0, 4, 8, 13], 1] = np.nan
    result = linear_regression(x, field, year, control)
    for column in range(2):
        keep = np.isfinite(x) & np.isfinite(control).all(axis=1) & np.isfinite(field[:, column])
        design = np.column_stack((np.ones(keep.sum()), year[keep]-year[keep].mean(), control[keep], x[keep]))
        expected = np.linalg.lstsq(design, field[keep, column], rcond=None)[0]
        np.testing.assert_allclose(result['beta'][column], expected[-1], atol=1e-12)
        assert result['n'][column] == keep.sum()
        assert result['df'][column] == keep.sum()-5


def test_input_unit_scaling_and_arbitrarily_small_responses():
    year, x, control, y = sample()
    reference = linear_regression(x, y, year, control)
    scaled = linear_regression(x*1e80, y*1e-80, year, control*1e-80)
    np.testing.assert_allclose(scaled['beta']/1e-160, reference['beta'], rtol=1e-12, atol=0)
    np.testing.assert_allclose(scaled['control_beta'], reference['control_beta'], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaled['r'], reference['r'], atol=1e-14)
    tiny = linear_regression(x, x*1e-200, year)
    np.testing.assert_allclose(tiny['beta']/1e-200, 1., rtol=1e-12, atol=0)
    assert np.isfinite(tiny['r'])


def test_constant_response_and_rank_deficient_designs_are_explicit():
    year, x, control, _ = sample()
    constant = linear_regression(x, np.full(len(year), 7.), year)
    assert constant['beta'] == 0
    assert constant['intercept'] == 7
    assert constant['fit_valid']
    assert np.isnan(constant['r']) and np.isnan(constant['p'])
    for predictor, controls in ((np.ones(len(year)), None), (x, x),
                                (year.astype(float), None), (x, np.ones(len(year)))):
        result = linear_regression(predictor, x, year, controls)
        assert not result['fit_valid']
        assert result['rank'] < result['n_parameters']
        assert np.isnan(result['beta'])


def test_all_missing_and_short_local_samples_keep_actual_counts():
    year, x, _, y = sample()
    field = np.column_stack((y, np.full(len(year), np.nan), y))
    field[3:, 2] = np.nan
    result = linear_regression(x, field, year, min_samples=10)
    np.testing.assert_array_equal(result['n'], [50, 0, 3])
    assert result['fit_valid'][0]
    assert not result['fit_valid'][1:].any()
    assert np.isnan(result['beta'][1:]).all()


def test_lag1_partial_residuals_actual_gaps_and_adjusted_error():
    year, x, control, y = sample()
    keep = np.ones(len(year), bool)
    keep[[3, 9, 17, 23]] = False
    year, x, z, y = year[keep], x[keep], control[keep, 0], y[keep]
    result = linear_regression(x, y, year, z, inference='lag1_bartlett')
    nuisance = np.column_stack((np.ones(len(year)), year-year.mean(), z))
    xr = x-nuisance@np.linalg.lstsq(nuisance, x, rcond=None)[0]
    yr = y-nuisance@np.linalg.lstsq(nuisance, y, rcond=None)[0]
    adjacent = np.diff(year) == 1
    rx = stats.pearsonr(xr[:-1][adjacent], xr[1:][adjacent]).statistic
    ry = stats.pearsonr(yr[:-1][adjacent], yr[1:][adjacent]).statistic
    neff = min(len(year), len(year)*(1-rx*ry)/(1+rx*ry))
    df = neff-4
    r = stats.pearsonr(xr, yr).statistic
    expected_p = 2*stats.t.sf(abs(r)*np.sqrt(df/(1-r*r)), df)
    np.testing.assert_allclose(result['n_eff'], neff, rtol=1e-12)
    np.testing.assert_allclose(result['rho1_predictor'], rx, atol=1e-14)
    np.testing.assert_allclose(result['p'], expected_p, rtol=1e-10, atol=0)
    np.testing.assert_allclose(result['stderr'], result['stderr_nominal']*np.sqrt((len(year)-4)/df), rtol=1e-12)
    assert result['n_lag_pairs'] == adjacent.sum()


def test_unavailable_lag_diagnostics_do_not_replace_adjusted_p_with_nominal():
    year, x, _, y = sample()
    result = linear_regression(x, y, year*2, inference='lag1_bartlett')
    assert np.isfinite(result['beta']) and np.isfinite(result['p_nominal'])
    assert result['n_lag_pairs'] == 0
    assert np.isnan(result['p']) and np.isnan(result['stderr'])


def test_multidimensional_fields_masked_values_and_no_input_mutation():
    year, x, _, y = sample()
    field = np.broadcast_to(y[:, None, None], (len(year), 2, 3)).copy()
    masked = np.ma.array(field, mask=False)
    masked.mask[0, 1, 2] = True
    original = masked.copy()
    result = linear_regression(x, masked, year)
    assert result['beta'].shape == (2, 3)
    assert result['coefficients'].shape == (3, 2, 3)
    assert result['n'][1, 2] == len(year)-1
    np.testing.assert_array_equal(masked, original)
    np.testing.assert_array_equal(masked.mask, original.mask)


def test_unrepresentable_centering_is_an_explicit_value_error():
    years = np.arange(10)
    predictor = np.arange(10.) % 3
    with pytest.raises(ValueError, match='predictor centering'):
        linear_regression(np.full(10, 1e308), np.arange(10.), years)
    with pytest.raises(ValueError, match='response centering'):
        linear_regression(predictor, np.array([-1e308,1e308]*5), years)


@pytest.mark.parametrize('change', [
    {'years': [1, 2, 2, 4]}, {'years': [4, 3, 2, 1]}, {'years': [1., 2., 3., 4.5]},
    {'predictor': [1., 2., np.inf, 4.]}, {'field': np.ones((4, 0))},
    {'controls': np.ones((4, 2, 1))}, {'field': np.ones(4, complex)},
    {'include_trend': 'yes'}, {'min_samples': True}, {'min_samples': 3},
    {'inference': 'hac'}, {'min_lag_pairs': False}])
def test_invalid_arguments_are_rejected(change):
    options = {'predictor': np.array([1., 0., 2., 1.]), 'field': np.arange(4.), 'years': np.arange(4)}
    options.update(change)
    with pytest.raises(ValueError):
        linear_regression(**options)
