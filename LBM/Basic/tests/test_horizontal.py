"""Numerical checks for the LBM horizontal interpolation/filtering module."""

import unittest

import numpy as np
from scipy.special import gammaln, lpmv

from lbm_background.horizontal import gaussian_grid, periodic_bilinear, scalar_triangular_truncate


def real_harmonic(degree, order):
    lat, lon = gaussian_grid()
    scale = np.exp(0.5 * (np.log((2 * degree + 1) / 2)
                         + gammaln(degree - order + 1)
                         - gammaln(degree + order + 1)))
    meridional = lpmv(order, degree, np.sin(np.deg2rad(lat))) * scale
    return meridional[:, None] * np.cos(order * np.deg2rad(lon))[None, :]


class GaussianSpectralTests(unittest.TestCase):
    def test_grid(self):
        lat, lon = gaussian_grid()
        self.assertEqual((lat.size, lon.size), (64, 128))
        self.assertTrue(np.all(np.diff(lat) < 0))
        np.testing.assert_allclose(lat, -lat[::-1], atol=0, rtol=0)
        self.assertEqual(lon[0], 0)
        self.assertEqual(lon[-1], 357.1875)

    def test_constant(self):
        data = np.full((2, 64, 128), 287.125)
        np.testing.assert_allclose(scalar_triangular_truncate(data), data,
                                   atol=3e-9, rtol=0)

    def test_low_harmonics(self):
        data = np.stack([real_harmonic(2, 0), real_harmonic(12, 5),
                         real_harmonic(42, 42), real_harmonic(42, 17)])
        np.testing.assert_allclose(scalar_triangular_truncate(data), data,
                                   atol=2e-11, rtol=2e-11)

    def test_high_degree_removed(self):
        data = np.stack([real_harmonic(45, 0), real_harmonic(45, 11),
                         real_harmonic(45, 45)])
        np.testing.assert_allclose(scalar_triangular_truncate(data), 0,
                                   atol=2e-11, rtol=0)

    def test_idempotence(self):
        data = np.random.default_rng(20261019).normal(size=(2, 3, 64, 128))
        first = scalar_triangular_truncate(data)
        np.testing.assert_allclose(scalar_triangular_truncate(first), first,
                                   atol=2e-11, rtol=2e-11)

    def test_missing_rejected(self):
        data = np.ones((64, 128))
        data[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            scalar_triangular_truncate(data)


class BilinearTests(unittest.TestCase):
    def test_periodic_boundary(self):
        lat = [-90, 90]
        lon = [-180, -90, 0, 90]
        field = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=float)
        result = periodic_bilinear(field, lat, lon, [30, -30],
                                   [135, -225, -180, 180, 540])
        np.testing.assert_allclose(result, [[1.5, 1.5, 0, 0, 0]] * 2)

    def test_missing_corners_normalized(self):
        # At the midpoint each corner initially contributes one quarter.
        field = np.array([[2., 4.], [6., np.nan]])
        result = periodic_bilinear(field, [-1, 1], [-90, 90], [0], [0])
        np.testing.assert_allclose(result, [[4.]])
        masked = np.ma.array([[2., 4.], [6., 8.]],
                             mask=[[False, False], [False, True]])
        np.testing.assert_allclose(periodic_bilinear(
            masked, [-1, 1], [-90, 90], [0], [0]), [[4.]])

    def test_unequal_missing_weights_and_all_missing(self):
        field = np.array([[[2., 4.], [6., np.nan]],
                          [[np.nan, np.nan], [np.nan, np.nan]]])
        result = periodic_bilinear(field, [0, 1], [-90, 90], [.25], [-45])
        expected = (2 * .75 * .75 + 4 * .75 * .25 + 6 * .25 * .75) / .9375
        self.assertAlmostEqual(result[0, 0, 0], expected)
        self.assertTrue(np.isnan(result[1, 0, 0]))

    def test_zero_weight_valid_neighbor_does_not_fill_missing_node(self):
        field = np.array([[np.nan, 4.], [6., 8.]])
        result = periodic_bilinear(field, [0, 1], [-90, 90], [0], [-90])
        self.assertTrue(np.isnan(result[0, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
