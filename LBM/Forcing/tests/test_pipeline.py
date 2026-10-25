"""Analytic end-to-end tests using invented CF inputs and model metadata."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset
import numpy as np

from lbm_forcing.config import load_config
from lbm_forcing.converter import convert, check_conversion
from lbm_forcing.demo import create_demo
from lbm_forcing.forcing_io import read_forcing_grd
from lbm_forcing.pipeline import diagnose, check_diagnosis, midpoint_seconds


def _float(variable):
    return np.ma.asarray(variable[:], dtype=np.float64).filled(np.nan)


def _analytic_pattern(lat, lon):
    return np.cos(np.deg2rad(lat))[:, None] * np.cos(np.deg2rad(lon))[None, :]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def _configs(self, **kwargs):
        paths = create_demo(self.root / 'case', **kwargs)
        return (load_config(paths['diagnose'], 'diagnose'),
                load_config(paths['convert'], 'convert'), paths)

    def test_complete_climatology_conversion_readback_and_checks(self):
        diagnosis, conversion, paths = self._configs()
        self.assertEqual(paths['diagnose'].name, 'diagnose.json')
        self.assertEqual(paths['convert'].name, 'convert.json')
        dout, fout = Path(diagnosis['output']), Path(conversion['output'])
        checked = check_diagnosis(diagnosis)
        self.assertFalse(checked['raw_data_checked'])
        self.assertFalse(checked['outputs_written'])
        self.assertEqual(checked['target_shape'], [19, 36])
        self.assertFalse(dout.exists())
        self.assertFalse(fout.exists())
        result = diagnose(diagnosis)
        self.assertEqual(result['state'], 'complete')
        self.assertEqual(result['seasonal_files'], 2)
        self.assertEqual(result['analysis'], 'climatology')
        with Dataset(dout / 'diagnosed_heating.nc') as ds:
            lat, lon = ds['lat'][:], ds['lon'][:]
            expected = np.broadcast_to(.3 * _analytic_pattern(lat, lon),
                                       ds['heating_k_day'].shape).copy()
            expected[:, [0, -1], :] = np.nan
            np.testing.assert_allclose(_float(ds['heating_k_day']), expected,
                                       atol=5e-11, rtol=1e-10, equal_nan=True)
            total = _float(ds['q1_climatology'])
            terms = [_float(ds[name + '_climatology'])
                     for name in ('tendency', 'horizontal', 'vertical')]
            np.testing.assert_allclose(total, sum(terms), atol=1e-14, equal_nan=True)
            self.assertEqual(float(np.nanmax(abs(terms[1]))), 0.)
            self.assertEqual(float(np.nanmax(abs(terms[2]))), 0.)
            self.assertTrue(np.isnan(total[:, [0, -1], :]).all())
            self.assertTrue(np.isfinite(total[:, 1:-1, :]).all())
            np.testing.assert_allclose(_float(ds['heating_w_kg']), expected * (1004. / 86400.),
                                       atol=1e-12, rtol=1e-10, equal_nan=True)
        before = {path.relative_to(self.root).as_posix() for path in self.root.rglob('*')}
        planned = check_conversion(conversion)
        self.assertFalse(planned['files_written'])
        self.assertFalse(planned['output_exists'])
        self.assertEqual(before, {path.relative_to(self.root).as_posix() for path in self.root.rglob('*')})
        result = convert(conversion)
        self.assertEqual(result['state'], 'complete')
        actual = read_forcing_grd(fout / result['forcing_grd'])
        for key in ('vor', 'div', 'ps'):
            self.assertTrue(np.all(actual[key] == 0))
        with Dataset(fout / 'forcing.nc') as ds:
            np.testing.assert_array_equal(ds['temperature_forcing_k_s'][:], actual['tmp'])
            np.testing.assert_array_equal(ds['heating_k_day'][:], actual['tmp'].astype(float) * 86400.)
            lat, lon = ds['lat'][:], ds['lon'][:]
            outside = ((lat[:, None] <= -45.) | (lat[:, None] >= 15.)
                       | (lon[None, :] <= 30.) | (lon[None, :] >= 120.))
            self.assertTrue(np.all(actual['tmp'][:, outside] == 0))
            # In the flat full-weight core the pressure-independent analytic
            # heating remains unchanged by vertical interpolation. Compute
            # the source-grid bilinear interpolation independently here.
            y = (lat + 90.) / 10.
            i = np.clip(np.floor(y).astype(int), 0, 17)
            fy = y - i
            x = lon / 10.
            j = np.floor(x).astype(int) % 36
            fx = x - np.floor(x)
            source_lat = np.linspace(-90., 90., 19)
            source_lon = np.arange(36) * 10.
            source = .3 * _analytic_pattern(source_lat, source_lon)
            interpolated = ((1-fy[:, None]) * ((1-fx[None]) * source[i[:, None], j[None]]
                            + fx[None] * source[i[:, None], (j+1)[None] % 36])
                            + fy[:, None] * ((1-fx[None]) * source[(i+1)[:, None], j[None]]
                            + fx[None] * source[(i+1)[:, None], (j+1)[None] % 36]))
            core = ((lat[:, None] >= -40.) & (lat[:, None] <= 10.)
                    & (lon[None, :] >= 35.) & (lon[None, :] <= 115.))
            self.assertTrue(np.any(actual['tmp'] > 0))
            self.assertTrue(np.any(actual['tmp'] < 0))
            np.testing.assert_allclose(actual['tmp'][:, core].astype(float) * 86400.,
                                       np.broadcast_to(interpolated[core], (20, core.sum())),
                                       rtol=3e-7, atol=1e-11)
        for folder in (dout, fout):
            manifest = json.loads((folder / 'output_manifest.json').read_text())
            for name, item in manifest.items():
                self.assertEqual(hashlib.sha256((folder / name).read_bytes()).hexdigest(), item['sha256'])
            self.assertNotIn(str(self.root), (folder / 'input_manifest.json').read_text())
        before = hashlib.sha256((dout / 'diagnosed_heating.nc').read_bytes()).hexdigest()
        with self.assertRaises(FileExistsError):
            diagnose(diagnosis)
        with self.assertRaises(FileExistsError):
            convert(conversion)
        with self.assertRaises(FileExistsError):
            create_demo(self.root / 'case')
        self.assertEqual(hashlib.sha256((dout / 'diagnosed_heating.nc').read_bytes()).hexdigest(), before)

    def test_regression_known_coefficient_and_original_predictor_scale(self):
        diagnosis, _, _ = self._configs(regression=True)
        result = diagnose(diagnosis)
        self.assertEqual(result['analysis'], 'regression')
        path = Path(diagnosis['output']) / 'diagnosed_heating.nc'
        with Dataset(path) as ds:
            expected = np.broadcast_to(.075 * _analytic_pattern(ds['lat'][:], ds['lon'][:]),
                                       ds['heating_k_day'].shape).copy()
            expected[:, [0, -1], :] = np.nan
            original = _float(ds['heating_k_day'])
            np.testing.assert_allclose(original, expected, rtol=1e-9, atol=5e-11, equal_nan=True)
            self.assertFalse(np.isclose(np.std(ds['predictor'][:], ddof=1), 1.))
            self.assertTrue(np.all(_float(ds['q1_n'])[:, 1:-1] == 12))
            self.assertEqual(ds['q1_predictor_trend'].units,
                             diagnosis['regression']['predictor_units'] + ' year-1')
            self.assertEqual(ds['q1_beta'].per_predictor_unit,
                             diagnosis['regression']['predictor_units'])
        csv_path = Path(diagnosis['regression']['file'])
        with csv_path.open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        with csv_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=('year', 'predictor', 'control'))
            writer.writeheader()
            for row in rows:
                row['predictor'] = str(float(row['predictor']) * 2.)
                writer.writerow(row)
        diagnosis['output'] = str(self.root / 'scaled_diagnosis_output')
        diagnose(diagnosis)
        with Dataset(Path(diagnosis['output']) / 'diagnosed_heating.nc') as ds:
            np.testing.assert_allclose(_float(ds['heating_k_day']), original / 2.,
                                       atol=5e-11, rtol=1e-9, equal_nan=True)

    def test_360_day_calendar_midpoints_and_analytic_heating(self):
        diagnosis, _, _ = self._configs(calendar='360_day')
        seconds = 86400.
        self.assertEqual(midpoint_seconds(2001, 1, '360_day')
                         - midpoint_seconds(2000, 12, '360_day'), 30. * seconds)
        self.assertEqual(midpoint_seconds(2001, 2, '360_day')
                         - midpoint_seconds(2001, 1, '360_day'), 30. * seconds)
        self.assertEqual(midpoint_seconds(2001, 1, 'standard')
                         - midpoint_seconds(2000, 12, 'standard'), 31. * seconds)
        self.assertEqual(midpoint_seconds(2001, 2, 'standard')
                         - midpoint_seconds(2001, 1, 'standard'), 29.5 * seconds)
        diagnose(diagnosis)
        with Dataset(Path(diagnosis['output']) / 'diagnosed_heating.nc') as ds:
            self.assertEqual(ds.calendar, '360_day')
            expected = np.broadcast_to(.3 * _analytic_pattern(ds['lat'][:], ds['lon'][:]),
                                       ds['heating_k_day'].shape).copy()
            expected[:, [0, -1], :] = np.nan
            np.testing.assert_allclose(_float(ds['heating_k_day']), expected,
                                       atol=5e-11, rtol=1e-10, equal_nan=True)

    def test_missing_halo_records_failed_state(self):
        diagnosis, _, _ = self._configs()
        with Dataset(Path(diagnosis['data_root']) / 'monthly.nc', 'a') as ds:
            ds['time'][4] = ds['time'][4] + 45.  # March moved into an unused April.
        with self.assertRaisesRegex(ValueError, '200103; found 0'):
            diagnose(diagnosis)
        output = Path(diagnosis['output'])
        state = json.loads((output / 'build_status.json').read_text())
        self.assertEqual(state['state'], 'failed')
        self.assertFalse((output / 'diagnosed_heating.nc').exists())
        self.assertNotIn(str(self.root), (output / 'build_status.json').read_text())

    def test_month_specific_terrain_mask_survives_seasonal_cache(self):
        diagnosis, _, _ = self._configs()
        with Dataset(Path(diagnosis['data_root']) / 'monthly.nc', 'a') as ds:
            ds['PS'][2, 9, 6] = 60000.  # January 2001, latitude 0, longitude 60.
        diagnose(diagnosis)
        output = Path(diagnosis['output'])
        with Dataset(output / 'seasonal' / '2001.nc') as ds:
            common = np.isfinite(_float(ds['q1']))
            self.assertFalse(common[-1, 9, 6])
            self.assertTrue(common[:-1, 9, 6].all())
            for name in ('tendency', 'horizontal', 'vertical'):
                np.testing.assert_array_equal(np.isfinite(_float(ds[name])), common)
            self.assertEqual(ds['ps_min'][9, 6], 60000.)
            self.assertAlmostEqual(ds['ps_mean'][9, 6], 260000. / 3., places=9)
        with Dataset(output / 'diagnosed_heating.nc') as ds:
            self.assertEqual(ds['q1_climatology_n'][-1, 9, 6], 1)
            self.assertTrue(np.all(ds['q1_climatology_n'][:-1, 9, 6] == 2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
