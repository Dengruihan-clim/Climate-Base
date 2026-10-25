"""Independent lag-one formulas and explicit BH families."""
import numpy as np
import pytest
from scipy import stats

from climate_statistics.inference import benjamini_hochberg, lag1_bartlett


def test_lag_formula_calendar_gaps_and_missing_pairs():
    years = np.array([1980,1981,1982,1984,1985,1986,1987,1988,1990,1991,1992,1993])
    x = np.array([1.,2.,4.,8.,7.,3.,5.,10.,2.,6.,3.,8.])
    y = np.array([3.,6.,1.,5.,9.,np.nan,7.,4.,11.,10.,5.,9.])
    actual = lag1_bartlett(x, y, years, n_parameters=3)
    paired = np.isfinite(x) & np.isfinite(y)
    lag = paired[:-1] & paired[1:] & (np.diff(years) == 1)
    rx = stats.pearsonr(x[:-1][lag], x[1:][lag]).statistic
    ry = stats.pearsonr(y[:-1][lag], y[1:][lag]).statistic
    raw = paired.sum()*(1-rx*ry)/(1+rx*ry)
    effective = min(raw, paired.sum())
    r = stats.pearsonr(x[paired], y[paired]).statistic
    p = 2*stats.t.sf(abs(r)*np.sqrt((effective-3)/(1-r*r)), effective-3)
    for key, expected in (('rho1_x', rx), ('rho1_y', ry), ('n_eff_raw', raw),
                           ('n_eff', effective), ('p', p)):
        np.testing.assert_allclose(actual[key], expected, rtol=1e-12, atol=1e-14)
    assert actual['n_lag_pairs'] == 7 and actual['n'] == 11
    assert actual['df'] == actual['n_eff']-3


def test_raw_infinite_estimate_is_capped_and_missing_has_no_fallback():
    result = lag1_bartlett(np.arange(7.), [1.,-1.,1.,-1.,1.,-1.,1.], np.arange(7))
    assert np.isposinf(result['n_eff_raw'])
    assert result['n_eff'] == 7
    result = lag1_bartlett([1.,1.,1.,1.,2.], [0.,2.,1.,4.,3.], np.arange(5))
    assert np.isfinite(result['p_nominal'])
    assert np.isnan(result['rho1_x']) and np.isnan(result['p'])
    result = lag1_bartlett(np.arange(5.), [1.,3.,0.,2.,4.], np.arange(5)*2)
    assert result['n_lag_pairs'] == 0 and np.isnan(result['p'])


def test_scale_offset_masked_arrays_and_broadcast_predictor():
    rng = np.random.default_rng(321)
    x = rng.normal(size=30)
    y = rng.normal(size=(30, 2, 3))
    base = lag1_bartlett(x, y, np.arange(30))
    scaled = lag1_bartlett(x*1e-180, y*1e180, np.arange(30))
    np.testing.assert_allclose(scaled['r'], base['r'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(scaled['n_eff'], base['n_eff'], rtol=1e-12)
    masked = np.ma.array(y, mask=False)
    masked.mask[5, 0, 0] = True
    result = lag1_bartlett(x, masked, np.arange(30))
    assert result['n'][0, 0] == 29


def test_bh_known_values_ties_missing_and_shape():
    p = np.array([[.01, .04, .03], [.002, .5, np.nan]])
    actual = benjamini_hochberg(p)
    expected = np.array([[.025, .05, .05], [.01, .5, np.nan]])
    np.testing.assert_allclose(actual['q_values'], expected, equal_nan=True)
    np.testing.assert_array_equal(actual['reject'], [[True,True,True],[True,False,False]])
    assert actual['n_tests'] == 5
    ties = benjamini_hochberg([.02, .02, .8])
    np.testing.assert_allclose(ties['q_values'], [.03, .03, .8])


def test_bh_empty_and_all_missing_families_are_explicit():
    for p in (np.array([]), np.full((2, 3), np.nan)):
        result = benjamini_hochberg(p)
        assert result['n_tests'] == 0
        assert not result['reject'].any()
        assert np.isnan(result['q_values']).all()
    assert benjamini_hochberg(.01)['reject']


@pytest.mark.parametrize('p,alpha', [([-1.], .05), ([1.1], .05), ([np.inf], .05),
                                    ([.01], False), ([.01], 0), ([.01], np.nan),
                                    ([.01], 1.1), ([.01+0j], .05)])
def test_bh_invalid_inputs(p, alpha):
    with pytest.raises(ValueError):
        benjamini_hochberg(p, alpha)


@pytest.mark.parametrize('options', [{'years':[0,1,1,3]}, {'years':[0.,1.,2.,3.5]},
                                   {'n_parameters':True}, {'min_lag_pairs':2},
                                   {'y':[1.,2.,np.inf,4.]}, {'x':np.ones((4,2))}])
def test_lag_invalid_inputs(options):
    values = {'x':np.arange(4.),'y':np.array([1.,0.,3.,2.]),'years':np.arange(4)}
    values.update(options)
    with pytest.raises(ValueError):
        lag1_bartlett(**values)
