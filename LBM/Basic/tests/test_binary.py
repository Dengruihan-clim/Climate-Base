"""Binary format checks using synthetic metadata and fields only.

The fixtures reproduce the documented record layout. They are not model
reference data and are not suitable for a real LBM integration.
"""
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from lbm_background.binary import (
    FIELD_ORDER, FIELD_UNITS, GTOOL_ITEMS, NX, NY, NZ, read_gtool,
    read_sigma, t42_coordinates, write_background, write_grads_ctl,
    write_grads_sequential)


def _record(handle, payload, endian='>'):
    marker = struct.pack(endian + 'i', len(payload))
    handle.write(marker + payload + marker)


def _header(item, units, shape, axes):
    nz, ny, nx = shape
    result = [''] * 64
    for index, value in {
            0: '9010', 1: 'SYNTHETIC TEST', 2: item, 15: units,
            28: axes[0], 29: '1', 30: str(nx),
            31: axes[1], 32: '1', 33: str(ny),
            34: axes[2], 35: '1', 36: str(nz),
            37: 'UR4', 63: str(nx * ny * nz)}.items():
        result[index] = value
    return ''.join(value.ljust(16) for value in result).encode('ascii')


def synthetic_fields():
    k, y, x = np.indices((NZ, NY, NX))
    return {
        'U': (k + y / 100 + x / 1000).astype(np.float32),
        'V': (-k / 10 + y / 200 + x / 2000).astype(np.float32),
        'T': (290 - 3 * k + y / 1000 + x / 10000).astype(np.float32),
        'PS': (980 + y[:1] / 100 + x[:1] / 1000).astype(np.float32),
        'Q': (.01 * np.exp(-k / 3) + y * 0 + x * 0).astype(np.float32)}


def write_synthetic_template(path, endian='>'):
    """Create a structurally valid fixture, never a production basic state."""
    fields = synthetic_fields()
    with Path(path).open('wb') as handle:
        for name, item, units in zip(FIELD_ORDER, GTOOL_ITEMS, FIELD_UNITS):
            values = fields[name]
            axes = ('GLON128', 'GGLA64', 'SFC1' if name == 'PS' else 'CSIG20')
            _record(handle, _header(item, units, values.shape, axes), endian)
            _record(handle, values.astype(endian + 'f4').tobytes(), endian)
    return fields


def write_synthetic_sigma(path, sigma):
    values = np.asarray(sigma, dtype='>f4').reshape(1, 1, -1)
    with Path(path).open('wb') as handle:
        _record(handle, _header('CSIG20', 'ND', values.shape,
                               ('SIGTEST', 'Y1', 'Z1')))
        _record(handle, values.tobytes())


class LbmBinaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.template = self.root / 'synthetic.t42l20'
        self.fields = write_synthetic_template(self.template)
        self.sigma = np.linspace(.995, .01, NZ)
        self.sigma_file = self.root / 'synthetic_sigma'
        write_synthetic_sigma(self.sigma_file, self.sigma)

    def test_template_layout_and_sigma(self):
        records = read_gtool(self.template)
        self.assertEqual(tuple(record.item for record in records), GTOOL_ITEMS)
        self.assertEqual([record.data.shape for record in records],
                         [(NZ, NY, NX)] * 3 + [(1, NY, NX), (NZ, NY, NX)])
        self.assertEqual(self.template.stat().st_size, 2659408)
        self.assertTrue(all(record.endian == '>' for record in records))
        np.testing.assert_array_equal(read_sigma(self.sigma_file),
                                      self.sigma.astype(np.float32))
        lon, lat = t42_coordinates()
        self.assertEqual(lon[0], 0)
        self.assertEqual(lon[-1], 357.1875)
        self.assertAlmostEqual(lat[0], 87.8637988, places=6)
        self.assertAlmostEqual(lat[-1], -87.8637988, places=6)

    def test_native_roundtrip_and_axis_order(self):
        fields = {key: value.copy() for key, value in self.fields.items()}
        fields['U'] = (np.arange(NZ)[:, None, None] * 10000
                       + np.arange(NY)[None, :, None] * NX
                       + np.arange(NX)[None, None, :])
        fields['PS'] = fields['PS'][0]
        destination = self.root / 'custom.t42l20'
        write_background(self.template, destination, fields)
        for name, record in zip(FIELD_ORDER, read_gtool(destination)):
            expected = np.asarray(fields[name], np.float32).reshape(record.data.shape)
            np.testing.assert_array_equal(record.data, expected)
            self.assertEqual(record.header[26].strip(), '00010101 000000')
            self.assertEqual(int(record.header[24]), 0)
            self.assertEqual(int(record.header[63]), record.data.size)
            self.assertEqual(record.header[62].strip(), 'CUSTOM')
            self.assertEqual(''.join(record.header[16:24]).strip(),
                             'Custom LBM basic state')
        raw = destination.read_bytes()
        self.assertEqual(raw[:4], b'\x00\x00\x04\x00')
        self.assertEqual(struct.unpack('>f', raw[1036 + 4 * NX:1040 + 4 * NX])[0], NX)
        self.assertEqual(destination.stat().st_size, 2659408)

    def test_grads_matches_native_and_has_surface_field(self):
        grd, ctl = self.root / 'background.grd', self.root / 'background.ctl'
        write_grads_sequential(grd, self.fields)
        self.assertEqual(grd.stat().st_size, 81 * (NX * NY * 4 + 8))
        with grd.open('rb') as handle:
            for name in FIELD_ORDER:
                for plane in self.fields[name]:
                    marker = handle.read(4)
                    self.assertEqual(struct.unpack('>i', marker)[0], NX * NY * 4)
                    data = np.frombuffer(handle.read(NX * NY * 4), '>f4').reshape(NY, NX)
                    self.assertEqual(handle.read(4), marker)
                    np.testing.assert_array_equal(data, plane)
            self.assertEqual(handle.read(), b'')
        write_grads_ctl(ctl, grd, read_sigma(self.sigma_file))
        text = ctl.read_text()
        self.assertIn('OPTIONS sequential big_endian yrev', text)
        self.assertIn('p 0 99', text)
        self.assertIn('00Z01JAN0001', text)
        self.assertIn('DSET ^background.grd', text)

    def test_corruption_and_missing_values_are_rejected(self):
        dest = self.root / 'bad'
        data = bytearray(self.template.read_bytes())
        data[1031] ^= 1
        dest.write_bytes(data)
        with self.assertRaises(ValueError):
            read_gtool(dest)
        dest.write_bytes(b'original output remains untouched')
        for name, value in (('PS', self.fields['PS'] * 100),
                            ('T', np.full(self.fields['T'].shape, np.nan))):
            fields = dict(self.fields, **{name: value})
            with self.subTest(field=name), self.assertRaises(ValueError):
                write_background(self.template, dest, fields)
            self.assertEqual(dest.read_bytes(), b'original output remains untouched')

    def test_little_endian_readable_but_not_valid_template(self):
        little = self.root / 'little_endian'
        write_synthetic_template(little, '<')
        records = read_gtool(little)
        self.assertTrue(all(record.endian == '<' for record in records))
        with self.assertRaisesRegex(ValueError, 'big-endian'):
            write_background(little, self.root / 'output', self.fields)

    def test_template_units_and_field_shape_are_checked(self):
        data = bytearray(self.template.read_bytes())
        data[4 + 15 * 16:4 + 16 * 16] = b'Pa'.ljust(16)
        bad = self.root / 'wrong_units'
        bad.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'units'):
            write_background(bad, self.root / 'output', self.fields)
        fields = dict(self.fields, Q=np.ones((NZ - 1, NY, NX)))
        with self.assertRaisesRegex(ValueError, 'shape'):
            write_background(self.template, self.root / 'output', fields)

    def test_sigma_invalid_bounds_order_and_length(self):
        for sigma in (np.linspace(.995, .01, NZ - 1),
                      np.linspace(.01, .995, NZ),
                      np.linspace(1, .01, NZ),
                      np.linspace(.995, 0, NZ),
                      np.full(NZ, np.nan)):
            with self.subTest(sigma=sigma.tolist()):
                write_synthetic_sigma(self.sigma_file, sigma)
                with self.assertRaises(ValueError):
                    read_sigma(self.sigma_file)

    def test_truncated_and_unsupported_records(self):
        bad = self.root / 'bad'
        data = self.template.read_bytes()
        bad.write_bytes(data[:-1])
        with self.assertRaisesRegex(ValueError, 'Truncated'):
            read_gtool(bad)
        wrong = bytearray(data)
        wrong[4 + 37 * 16:4 + 38 * 16] = b'UR8'.ljust(16)
        bad.write_bytes(wrong)
        with self.assertRaisesRegex(ValueError, 'UR4'):
            read_gtool(bad)


if __name__ == '__main__':
    unittest.main(verbosity=2)
