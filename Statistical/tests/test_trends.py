"""Analytic lines, SciPy trend references, and literal residual reconstruction."""
import numpy as np
import pytest
from scipy import stats

from climate_statistics.trends import circular_block_bootstrap, ols_trend, sen_trend


def test_ols_and_sen_use_actual_gapped_years_and_independent_scipy_results():
    years = np.array([1980,1981,1984,1985,1988,1990,1991,1993])
    values = np.array([1.,3.,2.,7.,5.,8.,11.,9.])
    actual = ols_trend(values, years)
    expected = stats.linregress(years, values)
    np.testing.assert_allclose(actual['slope'], expected.slope, rtol=1e-12)
    np.testing.assert_allclose(actual['stderr'], expected.stderr, rtol=1e-12)
    np.testing.assert_allclose(actual['p'], expected.pvalue, rtol=1e-12)
    np.testing.assert_allclose(actual['slope_per_decade'], 10*expected.slope)
    actual = sen_trend(values, years, confidence=.90)
    expected = stats.theilslopes(values, years, alpha=.90, method='separate')
    for key, reference in (('slope',expected.slope),('intercept',expected.intercept),
                           ('slope_ci_low',expected.low_slope),('slope_ci_high',expected.high_slope)):
        np.testing.assert_allclose(actual[key], reference, rtol=1e-12)
    np.testing.assert_allclose(actual['p_nominal'], stats.kendalltau(years, values).pvalue)


def test_per_cell_missing_constant_and_short_series():
    years = np.arange(12)+2000
    values = np.column_stack((years*.2, np.full(12,7.), years*.2))
    values[[1,4,9],0] = np.nan
    values[2:,2] = np.nan
    actual = sen_trend(values, years)
    np.testing.assert_array_equal(actual['n'], [9,12,2])
    np.testing.assert_allclose(actual['slope'][:2], [.2,0.], atol=1e-13)
    assert np.isnan(actual['slope'][2])
    assert np.isnan(actual['tau'][1]) and np.isnan(actual['p_nominal'][1])
    assert actual['slope_ci_low'][1] == actual['slope_ci_high'][1] == 0


@pytest.mark.parametrize('method', ['sen','ols'])
def test_exact_linear_bootstrap_constant_and_spatial_pairing(method):
    years = np.arange(20)+1980
    field = np.column_stack((2*(years-1980)+7., -3*(years-1980)+1., np.full(20,8.)))
    result = circular_block_bootstrap(field, years, n_resamples=41, block_length=3, seed=91, method=method)
    np.testing.assert_allclose(result['bootstrap_slopes'], np.broadcast_to([2.,-3.,0.], (41,3)), atol=1e-12)
    np.testing.assert_allclose(result['ci_low'], [2.,-3.,0.], atol=1e-12)
    rng = np.random.default_rng(821)
    y = .3*(years-years.mean())+rng.normal(size=20)
    paired = np.column_stack((y, 2*y+3))
    result = circular_block_bootstrap(paired, years, n_resamples=41, block_length=3, seed=91, method=method)
    np.testing.assert_allclose(result['bootstrap_slopes'][:,1], 2*result['bootstrap_slopes'][:,0], atol=1e-12)


def test_seed_batch_independence_and_literal_sen_residual_reconstruction():
    years = np.arange(1980, 1992)
    y = np.array([3.,2.,5.,4.,8.,7.,10.,11.,9.,15.,12.,16.])
    result = circular_block_bootstrap(y, years, n_resamples=23, block_length=3, seed=21, batch_size=4)
    again = circular_block_bootstrap(y, years, n_resamples=23, block_length=3, seed=21, batch_size=11)
    np.testing.assert_array_equal(result['resample_indices'], again['resample_indices'])
    np.testing.assert_array_equal(result['bootstrap_slopes'], again['bootstrap_slopes'])
    fit = stats.theilslopes(y, years, method='separate')
    line = np.median(y)+(years-np.median(years))*fit.slope
    residual = y-line
    residual -= residual.mean()
    expected = np.array([stats.theilslopes(line+residual[ix], years).slope for ix in result['resample_indices']])
    np.testing.assert_allclose(result['bootstrap_slopes'], expected, atol=1e-14)
    np.testing.assert_allclose(result['ci_low'], np.quantile(expected,.05))
    np.testing.assert_allclose(result['ci_high'], np.quantile(expected,.95))
    ix = result['resample_indices'].reshape(23,4,3)
    np.testing.assert_array_equal((np.diff(ix,axis=-1) % len(years)), 1)


def test_trend_serial_uncertainty_is_not_mislabeled_hac_or_bartlett():
    with pytest.raises(ValueError, match='nominal'):
        ols_trend(np.arange(10.), np.arange(10), inference='lag1_bartlett')


@pytest.mark.parametrize('change', [{'years':[1980,1981,1983,1984]}, {'field':[1.,2.,np.nan,4.]},
                                  {'block_length':0}, {'block_length':5}, {'block_length':True},
                                  {'n_resamples':1}, {'seed':-1}, {'method':'hac'},
                                  {'confidence':False}, {'batch_size':0}])
def test_bootstrap_rejects_gaps_missing_and_invalid_options(change):
    options = {'field':np.array([1.,3.,2.,5.]), 'years':np.arange(1980,1984), 'n_resamples':7}
    options.update(change)
    with pytest.raises(ValueError):
        circular_block_bootstrap(**options)


def test_trend_validation_and_masked_input():
    data = np.ma.array(np.arange(12.).reshape(6,2), mask=False)
    data.mask[0,0] = True
    actual = sen_trend(data, np.arange(6))
    np.testing.assert_array_equal(actual['n'], [5,6])
    for options in ({'confidence':1.}, {'min_samples':True}, {'min_samples':2}):
        with pytest.raises(ValueError):
            sen_trend(np.arange(6.), np.arange(6), **options)


def test_sen_confidence_does_not_silently_mean_the_complementary_level():
    years = np.arange(12)
    y = np.array([0.,1.,2.5,2.,5.,3.,7.,6.,9.,8.,9.,12.])
    for confidence in (.1, .49):
        with pytest.raises(ValueError, match='Sen confidence'):
            sen_trend(y, years, confidence=confidence)
    result = sen_trend(y, years, confidence=.5)
    reference = stats.theilslopes(y, years, alpha=.5)
    assert result['confidence'] == .5
    np.testing.assert_allclose(result['slope_ci_low'], reference.low_slope)
    np.testing.assert_allclose(result['slope_ci_high'], reference.high_slope)
    # Percentile and Student-t intervals have no complementary-alpha API.
    assert ols_trend(y, years, confidence=.1)['confidence'] == .1
    assert circular_block_bootstrap(y, years, n_resamples=7, confidence=.1)['confidence'] == .1
