import unittest
import numpy as np
from lbm_background.vertical import height_to_surface_pressure,pressure_to_sigma,sea_level_to_surface_pressure

class VerticalTests(unittest.TestCase):
    def test_height_log_pressure_and_extrapolation(self):
        p=np.array([1000.,850.,500.,100.,1.])
        h=(-8000*np.log(p/1000))[:,None,None]*np.ones((1,1,3))
        h[:2,0,1]=np.nan
        terrain=np.array([[0.,4000.,-100.]])
        result,qa=height_to_surface_pressure(h,p,terrain)
        np.testing.assert_allclose(result,1000*np.exp(-terrain/8000),rtol=1e-14)
        self.assertEqual(qa['below_lowest_valid_height_columns'],2)

    def test_missing_aware_spline_and_clamp(self):
        p=np.array([1000.,850.,500.,100.,1.])
        field=(10+2*np.log(p))[:,None,None]*np.ones((1,1,2))
        field[:2,0,1]=np.nan
        sigma=np.array([.999,.9,.3,.01])
        ps=np.array([[1020.,600.]])
        result,qa=pressure_to_sigma(field,p,ps,sigma)
        expected=10+2*np.log(np.minimum(sigma[:,None,None]*ps,np.array([[1000.,500.]])))
        np.testing.assert_allclose(result,expected,rtol=1e-14)
        self.assertEqual(qa['lower_endpoint_extension_by_sigma'],[2,1,0,0])

    def test_slp_reduction(self):
        result=sea_level_to_surface_pressure(np.array([1010.,1010.]),np.array([0.,2000.]),
                                             np.array([280.,280.]),np.array([.005,.005]),.995)
        self.assertEqual(result[0],1010.)
        self.assertTrue(750<result[1]<850)
        self.assertEqual(sea_level_to_surface_pressure(1010.,0.,280.,.005,.995),1010.)

    def test_invalid_pressure_or_surface_inputs_rejected(self):
        p=np.array([1000.,500.,100.])
        field=np.ones((3,1,2))
        ps=np.ones((1,2))*1000
        sigma=np.array([.995,.5,.01])
        for invalid in ([1000.,np.nan,100.],[1000.,0.,100.],
                        [100.,500.,1000.],[[1000.,500.,100.]]):
            with self.subTest(pressure=invalid), self.assertRaises(ValueError):
                pressure_to_sigma(field,invalid,ps,sigma)
        for invalid in (np.array([[np.nan,1000.]]),np.array([[0.,1000.]]),
                        np.array([[1000.]])):
            with self.subTest(surface=invalid.tolist()), self.assertRaises(ValueError):
                pressure_to_sigma(field,p,invalid,sigma)

    def test_invalid_sigma_inputs_rejected(self):
        for invalid in ([1.,.5],[.9,0.],[.5,.9],[.9,np.nan],.9,[],[.5,.5]):
            with self.subTest(sigma=invalid), self.assertRaises(ValueError):
                pressure_to_sigma(np.ones((3,1,1)),[1000.,500.,100.],
                                  np.array([[1000.]]),invalid)

    def test_height_shape_topography_and_missing_columns(self):
        p=np.array([1000.,500.,100.])
        height=np.array([0.,5000.,15000.])[:,None,None]
        with self.assertRaisesRegex(ValueError,'dimensions'):
            height_to_surface_pressure(height,p,np.ones((1,2)))
        with self.assertRaisesRegex(ValueError,'finite'):
            height_to_surface_pressure(height,p,np.array([[np.nan]]))
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            height_to_surface_pressure(np.array([0.,np.nan,np.nan])[:,None,None],
                                       p,np.array([[0.]]))

    def test_slp_invalid_scalar_sigma_physical_values_and_shapes(self):
        good=(np.array([1010.,1010.]),np.array([0.,2000.]),
              np.array([280.,280.]),np.array([.005,.005]))
        for invalid in (0.,1.,np.nan,np.array([.995])):
            with self.subTest(sigma=invalid), self.assertRaisesRegex(ValueError,'sigma'):
                sea_level_to_surface_pressure(*good,invalid)
        for field,value in ((0,[-1.,1010.]),(1,[0.,np.nan]),
                            (2,[0.,280.]),(3,[-.001,.005])):
            args=list(good); args[field]=np.array(value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                sea_level_to_surface_pressure(*args,.995)
        with self.assertRaisesRegex(ValueError,'shapes'):
            sea_level_to_surface_pressure(np.ones(2),np.ones(3),280.,.005,.995)

if __name__=='__main__':
    unittest.main()
