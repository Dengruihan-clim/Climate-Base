"""Independent joint-OLS and actual-year-lag inference checks."""
import unittest
import numpy as np
from scipy import stats
from lbm_forcing.statistics import partial_regression


class PartialRegressionTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(731)
        self.years = np.arange(1979, 2025)
        self.z = self.rng.normal(size=len(self.years))
        self.x = 0.8*self.z+self.rng.normal(size=len(self.years))
        self.x = (self.x-self.x.mean())/self.x.std(ddof=1)
        self.y = 1.3*self.x+2.1*self.z+0.03*(self.years-2000)+self.rng.normal(size=len(self.years))*.1

    def test_independent_joint_ols_and_original_predictor_sd(self):
        field = self.y[:, None, None]
        result = partial_regression(self.x, field, self.years, self.z)
        design = np.column_stack((np.ones(len(self.x)), self.years, self.z, self.x))
        expected = np.linalg.lstsq(design, self.y, rcond=None)[0][-1]
        self.assertAlmostEqual(result['beta'][0, 0], expected, places=12)
        scaled = partial_regression(self.x*2, field, self.years, self.z)
        self.assertAlmostEqual(scaled['beta'][0, 0], expected/2, places=12)
        n = len(self.x)
        t_nominal = abs(result['r'][0, 0])*np.sqrt((n-4)/(1-result['r'][0, 0]**2))
        self.assertAlmostEqual(result['p_nominal'][0, 0], 2*stats.t.sf(t_nominal, n-4), places=12)
        self.assertAlmostEqual(result['dof'][0, 0], result['n_eff'][0, 0]-4, places=12)

    def test_cell_specific_pairing_matches_ols_and_retains_below45_samples(self):
        field = np.column_stack((self.y, self.y+.2*self.rng.normal(size=len(self.x))))
        field[[0, 2, 8, 16, 22, 33], 1] = np.nan
        result = partial_regression(self.x, field, self.years, self.z)
        keep = np.isfinite(field[:, 1])
        design = np.column_stack((np.ones(keep.sum()), self.years[keep], self.z[keep], self.x[keep]))
        expected = np.linalg.lstsq(design, field[keep, 1], rcond=None)[0][-1]
        self.assertAlmostEqual(result['beta'][1], expected, places=12)
        self.assertEqual(result['n'][1], keep.sum())
        self.assertTrue(np.isfinite(result['p'][1]))

    def test_constant_and_tiny_responses_keep_valid_beta(self):
        field = np.column_stack((np.full(len(self.x), 7.), self.x*1e-18))
        result = partial_regression(self.x, field, self.years, self.z)
        self.assertEqual(result['beta'][0], 0.)
        self.assertAlmostEqual(result['beta'][1]/1e-18, 1., places=12)
        self.assertTrue(np.isnan(result['r'][0]))

    def test_shared_missing_patterns_preserve_component_closure(self):
        a = self.rng.normal(size=(len(self.x), 2, 3))
        b = self.rng.normal(size=a.shape)
        c = self.rng.normal(size=a.shape)
        a[[1, 3, 7], 0, 1] = np.nan
        b[np.isnan(a)] = np.nan
        c[np.isnan(a)] = np.nan
        betas = [partial_regression(self.x, f, self.years, self.z)['beta']
                 for f in (a+b+c, a, b, c)]
        np.testing.assert_allclose(betas[0], betas[1]+betas[2]+betas[3], atol=1e-15)

    def test_actual_year_lags_bartlett_independent_calculation(self):
        keep = np.ones(len(self.years), bool)
        keep[[3, 9, 11, 24, 34]] = False
        x, z, year, y = self.x[keep], self.z[keep], self.years[keep], self.y[keep]
        result = partial_regression(x, y[:, None], year, z)
        nuisance = np.column_stack((np.ones(len(year)), year-year.mean(), z))
        xr = x-nuisance@np.linalg.lstsq(nuisance, x, rcond=None)[0]
        yr = y-nuisance@np.linalg.lstsq(nuisance, y, rcond=None)[0]
        lookup = {value: i for i, value in enumerate(year)}
        inflation = 1.
        for lag in (1, 2, 3):
            pairs = [(i, lookup[value+lag]) for i, value in enumerate(year) if value+lag in lookup]
            left, right = np.array(pairs).T
            rx = np.corrcoef(xr[left], xr[right])[0, 1]
            ry = np.corrcoef(yr[left], yr[right])[0, 1]
            inflation += 2*len(left)/len(year)*rx*ry
        neff = min(len(year), len(year)/inflation)
        self.assertAlmostEqual(result['n_eff'][0], neff, places=11)
        self.assertAlmostEqual(result['dof'][0], neff-4, places=11)

    def test_rejects_nonfinite_and_collinear_global_controls(self):
        bad = [(self.x, np.full_like(self.z, 1.)), (self.x, self.x*2),
               (self.years.astype(float), self.z), (self.x, self.years.astype(float))]
        zbad = self.z.copy()
        zbad[3] = np.nan
        bad.append((self.x, zbad))
        xbad = self.x.copy()
        xbad[4] = np.inf
        bad.append((xbad, self.z))
        for x, z in bad:
            with self.subTest(x=x[:3], z=z[:3]), self.assertRaises(ValueError):
                partial_regression(x, self.y[:, None], self.years, z)

    def test_insufficient_and_locally_singular_samples_mask_beta(self):
        field = np.full((len(self.x), 2), np.nan)
        field[:9, 0] = self.y[:9]
        field[:10, 1] = self.y[:10]
        result = partial_regression(self.x, field, self.years, self.z)
        self.assertTrue(np.isnan(result['beta'][0]))
        self.assertTrue(np.isfinite(result['beta'][1]))
        # Predictor and control are independent over the full record, but
        # become identical over this particular cell's paired years.
        local_x = self.x.copy()
        local_x[:12] = self.z[:12]
        local_field = np.full((len(self.x), 1), np.nan)
        local_field[:12, 0] = self.y[:12]
        local = partial_regression(local_x, local_field, self.years, self.z)
        self.assertTrue(np.isnan(local['beta'][0]))

    def test_no_control_joint_ols_predictor_scale_and_degrees_of_freedom(self):
        field=self.y[:,None]
        result=partial_regression(self.x,field,self.years)
        design=np.column_stack((np.ones(len(self.x)),self.years,self.x))
        expected=np.linalg.lstsq(design,self.y,rcond=None)[0][-1]
        self.assertAlmostEqual(result['beta'][0],expected,places=12)
        scaled=partial_regression(self.x*2,field,self.years)
        self.assertAlmostEqual(scaled['beta'][0],expected/2,places=12)
        self.assertAlmostEqual(result['dof'][0],result['n_eff'][0]-3,places=12)
        r=result['r'][0]
        nominal_t=abs(r)*np.sqrt((len(self.x)-3)/(1-r*r))
        self.assertAlmostEqual(result['p_nominal'][0],
                               2*stats.t.sf(nominal_t,len(self.x)-3),places=12)
        self.assertIn('predictor_trend',result)
        self.assertIn('field_trend',result)
        self.assertNotIn('r_predictor_control',result)

    def test_no_control_paired_years_zero_tiny_beta_and_no_lag(self):
        field=np.column_stack((self.y,np.full(len(self.x),7.),self.x*1e-18))
        field[[0,2,8,16,22,33],0]=np.nan
        result=partial_regression(self.x,field,self.years,max_lag=0)
        keep=np.isfinite(field[:,0])
        design=np.column_stack((np.ones(keep.sum()),self.years[keep],self.x[keep]))
        expected=np.linalg.lstsq(design,field[keep,0],rcond=None)[0][-1]
        self.assertAlmostEqual(result['beta'][0],expected,places=12)
        self.assertEqual(result['n'][0],keep.sum())
        self.assertEqual(result['n_eff'][0],keep.sum())
        self.assertEqual(result['beta'][1],0.)
        self.assertAlmostEqual(result['beta'][2]/1e-18,1.,places=12)
        self.assertTrue(np.isnan(result['r'][1]))
        self.assertAlmostEqual(result['p'][0],result['p_nominal'][0],places=14)

    def test_no_control_actual_year_gaps_bartlett_independent_calculation(self):
        keep=np.ones(len(self.years),bool)
        keep[[3,9,11,24,34]]=False
        x,year,y=self.x[keep],self.years[keep],self.y[keep]
        result=partial_regression(x,y[:,None],year)
        nuisance=np.column_stack((np.ones(len(year)),year-year.mean()))
        xr=x-nuisance@np.linalg.lstsq(nuisance,x,rcond=None)[0]
        yr=y-nuisance@np.linalg.lstsq(nuisance,y,rcond=None)[0]
        lookup={value:i for i,value in enumerate(year)}
        inflation=1.
        for lag in (1,2,3):
            pairs=[(i,lookup[value+lag]) for i,value in enumerate(year)
                   if value+lag in lookup]
            left,right=np.array(pairs).T
            rx=np.corrcoef(xr[left],xr[right])[0,1]
            ry=np.corrcoef(yr[left],yr[right])[0,1]
            inflation+=2*len(left)/len(year)*rx*ry
        expected=min(len(year),len(year)/inflation)
        self.assertAlmostEqual(result['n_eff'][0],expected,places=11)
        self.assertAlmostEqual(result['dof'][0],expected-3,places=11)

    def test_generic_design_rejects_linear_predictor_and_bad_options(self):
        with self.assertRaisesRegex(ValueError,'detrending'):
            partial_regression(self.years.astype(float),self.y[:,None],self.years)
        for change in ({'min_years':True},{'max_lag':False},
                       {'control':self.z[:-1]},{'min_years':2}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                partial_regression(self.x,self.y[:,None],self.years,**change)


if __name__ == '__main__':
    unittest.main()
