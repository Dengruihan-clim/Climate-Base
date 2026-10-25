"""Independent analytic checks for pressure-resolved apparent heating."""
import unittest
import numpy as np
from lbm_forcing.diagnostics import CP, RD, EARTH_RADIUS_M, q1_layers


class Q1LayerTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([10000., 17000., 34000., 59000., 90000.])
        self.lat = np.array([-60., -30., 0., 30., 60.])
        self.lon = np.arange(0., 360., 30.)
        self.shape = (len(self.p), len(self.lat), len(self.lon))
        self.t = np.full(self.shape, 260.)
        self.zero = np.zeros(self.shape)
        self.h0, self.h1 = 30.5*86400, 29.5*86400

    def calculate(self, **changes):
        args = dict(t_prev=self.t, t=self.t, t_next=self.t, u=self.zero,
                    v=self.zero, omega=self.zero, pressure_pa=self.p,
                    lat=self.lat, lon=self.lon, dt_prev_seconds=self.h0,
                    dt_next_seconds=self.h1)
        args.update(changes)
        return q1_layers(**args)

    def test_unequal_time_quadratic_and_native_levels_no_integral(self):
        rate = np.arange(1, 6)[:, None, None]/86400.
        acceleration = 1e-13
        prev = self.t-rate*self.h0+acceleration*self.h0**2
        nxt = self.t+rate*self.h1+acceleration*self.h1**2
        out = self.calculate(t_prev=prev, t_next=nxt)
        self.assertEqual(out['q1'].shape, self.shape)
        np.testing.assert_allclose(out['tendency'], np.broadcast_to(rate*86400, self.shape), atol=2e-14)
        np.testing.assert_allclose(out['q1'], out['tendency'], atol=1e-14)
        self.assertEqual(set(out), {'q1', 'tendency', 'horizontal', 'vertical'})

    def test_vertical_pressure_sign_and_units(self):
        slope = 3e-4
        temp = np.broadcast_to(240+slope*self.p[:, None, None], self.shape)
        omega = np.full(self.shape, -0.04)
        out = self.calculate(t_prev=temp, t=temp, t_next=temp, omega=omega)
        expected = omega*(slope-RD/CP*temp/self.p[:, None, None])*86400
        np.testing.assert_allclose(out['vertical'], expected, atol=5e-14)
        self.assertTrue(np.all(out['q1'] > 0))

    def test_dry_adiabat_has_negligible_vertical_heating(self):
        pressure = np.geomspace(10000., 100000., 121)
        temperature = np.broadcast_to((280*(pressure/100000)**(RD/CP))[:, None, None],
                                      (len(pressure), *self.shape[1:]))
        zeros = np.zeros_like(temperature)
        omega = np.full_like(temperature, -0.03)
        out = self.calculate(t_prev=temperature, t=temperature, t_next=temperature,
                             u=zeros, v=zeros, omega=omega, pressure_pa=pressure)
        reference = np.abs(omega*RD/CP*temperature/pressure[:, None, None])*86400
        self.assertLess(np.max(np.abs(out['vertical'])/reference), 2e-4)

    def test_cyclic_longitude_and_meridional_derivatives(self):
        phi, lam = np.deg2rad(self.lat), np.deg2rad(self.lon)
        temp = np.broadcast_to(260+2*phi[None, :, None]+3*np.sin(lam)[None, None, :], self.shape)
        u, v = np.full(self.shape, 8.), np.full(self.shape, -3.)
        out = self.calculate(t_prev=temp, t=temp, t_next=temp, u=u, v=v)
        step = np.deg2rad(30)
        dtdlam = 3*np.cos(lam)*np.sin(step)/step
        expected = (u*dtdlam[None, None, :]/(EARTH_RADIUS_M*np.cos(phi)[None, :, None])
                    +v*2/EARTH_RADIUS_M)*86400
        np.testing.assert_allclose(out['horizontal'], expected, atol=2e-14)
        self.assertGreater(out['horizontal'][0, 2, 0], 0)
        self.assertLess(out['horizontal'][0, 2, 6], 0)

    def test_surface_pressure_masks_each_month_independently(self):
        ps = np.full(self.shape[1:], 100000.)
        previous_ps, next_ps = ps.copy(), ps.copy()
        previous_ps[2, 2] = 80000.
        next_ps[2, 3] = 50000.
        out = self.calculate(surface_pressure=ps, surface_pressure_prev=previous_ps,
                             surface_pressure_next=next_ps)
        self.assertTrue(np.isfinite(out['q1'][-1, 2, 1]))
        self.assertTrue(np.isnan(out['q1'][-1, 2, 2]))
        self.assertTrue(np.isnan(out['q1'][-2:, 2, 3]).all())
        self.assertTrue(np.isfinite(out['q1'][2, 2, 3]))
        # A previous-month PS mask affects only the temporal stencil; it
        # must not contaminate a neighboring current-month spatial stencil.
        self.assertTrue(np.isfinite(out['q1'][-1, 2, 4]))

    def test_current_terrain_masks_before_contiguous_spatial_stencils(self):
        ps = np.full(self.shape[1:], 100000.)
        ps[2, :] = 50000.
        out = self.calculate(surface_pressure=ps)
        self.assertTrue(np.isnan(out['q1'][-2:, 2, :]).all())
        # Only two finite latitude points remain on either side of the
        # mountain at these levels: a three-point derivative cannot jump it.
        self.assertTrue(np.isnan(out['q1'][-2:]).all())
        self.assertTrue(np.isfinite(out['q1'][:3]).all())

    def test_common_mask_and_exact_term_closure(self):
        omega = self.zero.copy()
        omega[0, 1, 3] = np.nan
        out = self.calculate(omega=omega)
        common = np.isfinite(out['q1'])
        for values in out.values():
            np.testing.assert_array_equal(np.isfinite(values), common)
        np.testing.assert_array_equal(out['q1'], out['tendency']+out['horizontal']+out['vertical'])

    def test_poles_masked(self):
        out = self.calculate(lat=np.array([-90., -45., 0., 45., 90.]))
        for values in out.values():
            self.assertTrue(np.isnan(values[:, [0, -1], :]).all())
            self.assertTrue(np.isfinite(values[:, 1:-1]).all())

    def test_invalid_coordinates_times_shapes_and_ps_units(self):
        bad = [dict(pressure_pa=self.p[::-1]), dict(lat=self.lat[::-1]),
               dict(lon=np.linspace(0., 360., len(self.lon))),
               dict(lon=np.arange(12.)), dict(dt_prev_seconds=0),
               dict(dt_next_seconds=np.nan), dict(u=self.zero[:-1]),
               dict(surface_pressure=np.ones(self.shape[1:])*1000),
               dict(surface_pressure_prev=np.ones((3, 4))),
               dict(surface_pressure_next=np.ones(self.shape[1:])*-100)]
        for values in bad:
            with self.subTest(values=values.keys()), self.assertRaises(ValueError):
                self.calculate(**values)



if __name__ == '__main__':
    unittest.main()
