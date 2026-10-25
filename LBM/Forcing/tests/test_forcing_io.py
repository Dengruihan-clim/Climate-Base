"""Analytic taper and independently decoded Fortran-layout checks."""
import struct
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from lbm_forcing.forcing_io import (FILE_BYTES, NLAT, NLEV, NLON, NRECORDS,
                            RECORD_BYTES, read_forcing_grd, region_taper,
                            write_forcing_ctl, write_forcing_grd)


class RegionTaperTests(unittest.TestCase):
    def test_periodic_seam_and_longitude_conventions(self):
        lon = [0, 27.5, 30, 32.5, 35, 60, 115, 117.5, 120, 122.5, 330]
        expected = np.array([[0, 0, 0, .5, 1, 1, 1, .5, 0, 0, 0]])
        np.testing.assert_allclose(region_taper([0], lon), expected, atol=2e-16)
        np.testing.assert_array_equal(region_taper([0], lon), region_taper([0], np.mod(lon, 360)))
        np.testing.assert_array_equal(region_taper([0], lon), region_taper([0], np.asarray(lon) + 720))

    def test_four_edges_corners_and_half_ramps(self):
        lat = [-47.5, -45, -42.5, -40, 0, 10, 12.5, 15, 17.5]
        lon = [27.5, 30, 32.5, 35, 60, 115, 117.5, 120, 122.5]
        edge = np.array([0, 0, .5, 1, 1, 1, .5, 0, 0])
        expected = edge[:, None] * edge[None, :]
        actual = region_taper(lat, lon)
        np.testing.assert_allclose(actual, expected, atol=3e-16)
        np.testing.assert_array_equal(actual[[0, 1, -2, -1]], 0)
        np.testing.assert_array_equal(actual[:, [0, 1, -2, -1]], 0)
        self.assertAlmostEqual(actual[2, 2], .25)

    def test_gaussian_grid_exact_exterior_zero_and_finite(self):
        lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(64)[0]))[::-1]
        lon = np.arange(128) * 2.8125
        window = region_taper(lat, lon)
        outside = ((lat[:, None] <= -45) | (lat[:, None] >= 15)
                   | (lon[None, :] <= 30) | (lon[None, :] >= 120))
        np.testing.assert_array_equal(window[outside], 0)
        self.assertTrue(np.all(np.isfinite(window)))
        self.assertGreaterEqual(window.min(), 0)
        self.assertEqual(window.max(), 1)

    def test_region_across_360_and_arbitrary_coordinate_order(self):
        lat = [5, -5, 0]
        lon = [355, 360, 365, 350, 20, 15, 17.5, 347.5]
        actual = region_taper(lat, lon, south=-10, north=10, west=350, east=20)
        expected = np.tile([1, 1, 1, 0, 0, 1, .5, 0], (3, 1))
        np.testing.assert_allclose(actual, expected, atol=2e-16)

    def test_reject_invalid_coordinates_and_bounds(self):
        for kwargs in ({"width": 0}, {"width": 40}, {"south": 15},
                       {"west": 0, "east": 360}, {"north": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                region_taper([0], [0], **kwargs)
        for lat, lon in (([[0]], [0]), ([91], [0]), ([0], [float("inf")]),
                         (np.ma.array([0], mask=[True]), [0])):
            with self.subTest(lat=lat, lon=lon), self.assertRaises(ValueError):
                region_taper(lat, lon)


class ForcingFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "forcing.grd"
        self.heating = np.zeros((NLEV, NLAT, NLON), dtype=np.float64)
        self.heating[0, 0, 0] = 1.25e-5
        self.heating[0, -1, 0] = -2.5e-5
        self.heating[19, 17, 43] = -8.125e-6

    def test_hand_decoded_binary_layout_and_north_to_south(self):
        path = write_forcing_grd(self.binary, self.heating)
        self.assertEqual(path, self.binary.resolve())
        contents = path.read_bytes()
        self.assertEqual(len(contents), 61 * (128 * 64 * 4 + 8))
        record_length = 128 * 64 * 4 + 8
        for index in range(61):
            start = index * record_length
            self.assertEqual(struct.unpack(">i", contents[start:start + 4]), (32768,))
            self.assertEqual(struct.unpack(">i", contents[start + record_length - 4:start + record_length]), (32768,))
            payload = contents[start + 4:start + record_length - 4]
            if index < 40 or index == 60:
                self.assertEqual(payload, bytes(32768))
        for level, j, i in ((0, 0, 0), (0, 63, 0), (19, 17, 43)):
            offset = (40 + level) * record_length + 4 + (j * 128 + i) * 4
            value = struct.unpack(">f", contents[offset:offset + 4])[0]
            self.assertEqual(value, float(np.float32(self.heating[level, j, i])))
        self.assertNotEqual(contents[40 * record_length + 4:40 * record_length + 8],
                            struct.pack("<f", self.heating[0, 0, 0]))

    def test_strict_reader_with_independent_manual_fixture(self):
        marker = struct.pack(">i", 32768)
        manual = bytearray()
        for index in range(61):
            plane = np.zeros((64, 128), dtype=">f4")
            if 40 <= index < 60:
                plane[:] = index - 45.25
                plane[1, 2] = index * .125
            manual.extend(marker + plane.tobytes(order="C") + marker)
        self.binary.write_bytes(manual)
        actual = read_forcing_grd(self.binary)
        self.assertEqual(set(actual), {"vor", "div", "tmp", "ps"})
        np.testing.assert_array_equal(actual["vor"], 0)
        np.testing.assert_array_equal(actual["div"], 0)
        np.testing.assert_array_equal(actual["ps"], 0)
        self.assertEqual(actual["tmp"].shape, (20, 64, 128))
        self.assertEqual(actual["tmp"][0, 0, 0], -5.25)
        self.assertEqual(actual["tmp"][19, 1, 2], 59 * .125)

    def test_roundtrip_and_refuse_overwrite_without_temporary_leaks(self):
        write_forcing_grd(self.binary, self.heating)
        original = self.binary.read_bytes()
        np.testing.assert_array_equal(read_forcing_grd(self.binary)["tmp"], self.heating.astype(np.float32))
        with self.assertRaises(FileExistsError):
            write_forcing_grd(self.binary, self.heating * 2)
        self.assertEqual(self.binary.read_bytes(), original)
        self.assertEqual(list(self.root.iterdir()), [self.binary])

    def test_binary_rejects_missing_complex_and_float32_overflow(self):
        for value in (float("nan"), float("inf"), 1e100):
            bad = self.heating.copy()
            bad[3, 4, 5] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                write_forcing_grd(self.binary, bad)
        masked = np.ma.array(self.heating, mask=False)
        masked.mask[3, 4, 5] = True
        for bad in (masked, self.heating.astype(complex), self.heating[:-1]):
            with self.subTest(dtype=bad.dtype), self.assertRaises(ValueError):
                write_forcing_grd(self.binary, bad)
        self.assertFalse(self.binary.exists())

    def test_concurrent_target_is_preserved_and_temporary_cleaned(self):
        def concurrent_writer(_source, target):
            Path(target).write_bytes(b"concurrent result")
            raise FileExistsError("destination was concurrently created")

        with mock.patch("lbm_forcing.forcing_io.os.link", side_effect=concurrent_writer):
            with self.assertRaises(FileExistsError):
                write_forcing_grd(self.binary, self.heating)
        self.assertEqual(self.binary.read_bytes(), b"concurrent result")
        self.assertEqual(list(self.root.iterdir()), [self.binary])

    def test_reader_rejects_corruption_nonthermal_and_nonfinite(self):
        write_forcing_grd(self.binary, self.heating)
        original = self.binary.read_bytes()
        for offset, replacement in ((0, struct.pack("<i", 32768)),
                                    (32772, struct.pack(">i", 123)),
                                    (4, struct.pack(">f", 1)),
                                    (40 * 32776 + 4, struct.pack(">f", float("nan")))):
            bad = bytearray(original)
            bad[offset:offset + 4] = replacement
            self.binary.write_bytes(bad)
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                read_forcing_grd(self.binary)
        for bad in (original[:-1], original + b"x", original[:32776]):
            self.binary.write_bytes(bad)
            with self.subTest(size=len(bad)), self.assertRaises(ValueError):
                read_forcing_grd(self.binary)


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "forcing.ctl"
        self.sigma = np.geomspace(.995, .0083, 20)
        self.lat = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(64)[0]))[::-1]
        self.lon = np.arange(128) * 2.8125

    def test_descriptor_coordinates_units_order_and_relative_dataset(self):
        write_forcing_ctl(self.path, self.root / "data" / "forcing.grd",
                          self.sigma, self.lat, self.lon)
        descriptor = self.path.read_text("ascii")
        self.assertEqual(descriptor.splitlines()[0], "DSET ^data/forcing.grd")
        self.assertIn("OPTIONS SEQUENTIAL YREV BIG_ENDIAN", descriptor)
        self.assertIn("TITLE Custom LBM thermal forcing",descriptor)
        self.assertIn("XDEF 128 LINEAR 0. 2.8125", descriptor)
        ytext = descriptor.split("YDEF 64 LEVELS\n")[1].split("ZDEF")[0]
        ztext = descriptor.split("ZDEF 20 LEVELS\n")[1].split("TDEF")[0]
        np.testing.assert_allclose(np.fromstring(ytext, sep=" "), self.lat[::-1], atol=1e-10)
        np.testing.assert_allclose(np.fromstring(ztext, sep=" "), self.sigma, atol=1e-12)
        self.assertTrue(np.all(np.diff(np.fromstring(ytext, sep=" ")) > 0))
        self.assertIn("vor 20 99", descriptor)
        self.assertIn("div 20 99", descriptor)
        self.assertIn("tmp 20 99 heating [K/s]", descriptor)
        self.assertIn("ps 0 99", descriptor)
        self.assertLess(descriptor.index("vor 20"), descriptor.index("div 20"))
        self.assertLess(descriptor.index("div 20"), descriptor.index("tmp 20"))
        self.assertLess(descriptor.index("tmp 20"), descriptor.index("ps 0"))
        with self.assertRaises(FileExistsError):
            write_forcing_ctl(self.path, "forcing.grd", self.sigma, self.lat, self.lon)

    def test_rounded_reference_gaussian_latitudes_accepted(self):
        write_forcing_ctl(self.path, "forcing.grd", self.sigma, np.round(self.lat, 4), self.lon)
        self.assertEqual(self.path.read_text("ascii").splitlines()[0], "DSET ^forcing.grd")

    def test_reject_wrong_direction_grid_or_level_count(self):
        for sigma, lat, lon in ((self.sigma[::-1], self.lat, self.lon),
                                (self.sigma[:-1], self.lat, self.lon),
                                (self.sigma, self.lat[::-1], self.lon),
                                (self.sigma, np.linspace(90, -90, 64), self.lon),
                                (self.sigma, self.lat, self.lon + 1)):
            with self.subTest(), self.assertRaises(ValueError):
                write_forcing_ctl(self.path, "forcing.grd", sigma, lat, lon)
        self.assertFalse(self.path.exists())

    def test_title_is_configurable_and_single_line_ascii(self):
        for title in ('', 'two\nlines', 'two\rlines', 'x'*129, '\u70ed\u6e90'):
            with self.subTest(title=title),self.assertRaises(ValueError):
                write_forcing_ctl(self.path,'forcing.grd',self.sigma,self.lat,self.lon,
                                  title=title)
        write_forcing_ctl(self.path,'forcing.grd',self.sigma,self.lat,self.lon,
                          title='Signed seasonal heating')
        self.assertIn('TITLE Signed seasonal heating',self.path.read_text('ascii'))


if __name__ == "__main__":
    unittest.main()
