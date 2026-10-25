"""Synthetic CF monthly input, pressure masks and calendar regression tests."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import cftime
from netCDF4 import Dataset, date2num

from lbm_forcing.monthly_inputs import MonthlyReader, neighbor_month, season_dates, strict_bilinear


class MonthlyReaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        self.lat = np.array([-90.0, 90.0])
        self.lon = np.array([0.0, 90.0, 180.0, 270.0])

    def tearDown(self):
        self.tmp.cleanup()

    def config(self):
        return {"_config_dir": self.base, "data_root": "data",
                "fields": {key: {"files": "{data_root}/source.nc", "variable": key}
                           for key in ("T", "U", "V", "OMEGA", "PS")},
                "integrity": {"hash_inputs": True}}

    def fixture(self, filename="source.nc", dates=None, lat=None, lon=None,
                pressure=None, pressure_units="hPa", units=None, values=None,
                ps_lat=None, ps_lon=None, calendar="standard", permute=False,
                singleton=False):
        path = self.data / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        dates = dates or [(2001, 1), (2001, 2), (2001, 3)]
        lat = self.lat if lat is None else np.asarray(lat)
        lon = self.lon if lon is None else np.asarray(lon)
        pressure = [1000.0, 500.0, 100.0] if pressure is None else pressure
        ps_lat = lat if ps_lat is None else np.asarray(ps_lat)
        ps_lon = lon if ps_lon is None else np.asarray(ps_lon)
        units = {"T": "K", "U": "m s-1", "V": "m/s", "OMEGA": "Pa s-1", "PS": "Pa", **(units or {})}
        defaults = {"T": 280.0, "U": 10.0, "V": 2.0, "OMEGA": 0.1, "PS": 101000.0}
        values = values or {}
        with Dataset(path, "w") as ds:
            dimensions = {"time": len(dates), "lev": len(pressure), "lat": len(lat), "lon": len(lon),
                          "pslat": len(ps_lat), "pslon": len(ps_lon)}
            if singleton:
                dimensions["member"] = 1
            for name, count in dimensions.items():
                ds.createDimension(name, count)
            time = ds.createVariable("time", "f8", ("time",))
            time.units = "days since 1900-01-01 00:00:00"
            time.calendar = calendar
            time[:] = date2num([datetime(y, m, 15) for y, m in dates], time.units, calendar=calendar)
            for name, axis, unit in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east"),
                                     ("pslat", ps_lat, "degrees_north"), ("pslon", ps_lon, "degrees_east"),
                                     ("lev", pressure, pressure_units)):
                variable = ds.createVariable(name, "f8", (name,))
                variable.units = unit
                variable[:] = axis
            for key, default in defaults.items():
                dims = ("time", "pslat", "pslon") if key == "PS" else ("time", "lev", "lat", "lon")
                shape = tuple(len(ds.dimensions[d]) for d in dims)
                data = np.empty(shape, dtype=np.float64)
                for index, date in enumerate(dates):
                    setting = values.get(key, {})
                    data[index] = setting.get(date, default) if isinstance(setting, dict) else setting
                if singleton:
                    dims = ("member",) + dims
                    data = data[None]
                if permute:
                    order = tuple(reversed(range(len(dims))))
                    dims = tuple(dims[i] for i in order)
                    data = np.transpose(data, order)
                variable = ds.createVariable(key, "f8", dims, fill_value=-999.0)
                variable.units = units[key]
                variable[:] = np.ma.masked_invalid(data)
        return path

    def reader(self, config=None, target_lat=None, target_lon=None):
        config = self.config() if config is None else config
        config["fields"]["PS"]["coordinates"] = {"latitude": "pslat", "longitude": "pslon"}
        return MonthlyReader(config, np.array([0.0]) if target_lat is None else target_lat,
                             np.array([45.0]) if target_lon is None else target_lon)

    def test_native_pressure_mask_precedes_strict_interpolation(self):
        ps = np.full((2, 4), 120000.0)
        ps[0, 0] = 80000.0
        temperature = np.full((3, 2, 4), 280.0)
        temperature[0, 0, 0] = 1000.0  # Underground value must never enter a stencil.
        self.fixture(values={"PS": ps, "T": temperature})
        reader = self.reader()
        ps_target, _ = reader.read("PS", 2001, 1)
        self.assertEqual(ps_target[0, 0], 110000.0)
        temperature, pressure = reader.read("T", 2001, 1)
        np.testing.assert_array_equal(pressure, [10000.0, 50000.0, 100000.0])
        self.assertTrue(np.isnan(temperature[2, 0, 0]))
        np.testing.assert_allclose(temperature[:2], 280.0)

    def test_neighbouring_temperature_uses_its_own_monthly_ps(self):
        self.fixture(values={"PS": {(2001, 1): 101000.0, (2001, 2): 45000.0, (2001, 3): 101000.0}})
        reader = self.reader()
        january, _ = reader.read("T", 2001, 1)
        february, _ = reader.read("T", 2001, 2)
        self.assertTrue(np.isfinite(january).all())
        self.assertTrue(np.isnan(february[1:]).all())
        self.assertTrue(np.isfinite(february[0]).all())
        january_again, _ = reader.read("T", 2001, 1)
        np.testing.assert_array_equal(january, january_again)
        reader.read("T", 2001, 3)
        self.assertLessEqual(len(reader._ps_cache), 2)

    def test_strict_corners_and_zero_weight_missing_corner(self):
        values = np.ones((2, 4))
        values[0, 0] = np.nan
        result = strict_bilinear(values, self.lat, self.lon, [0.0], [45.0])
        self.assertTrue(np.isnan(result).all())
        exact = strict_bilinear(values, self.lat, self.lon, [90.0], [0.0])
        np.testing.assert_array_equal(exact, [[1.0]])
        with self.assertRaisesRegex(ValueError, "extrapolation"):
            strict_bilinear(values, [-80.0, 80.0], self.lon, [89.0], [0.0])

    def test_missing_surface_pressure_propagates_to_every_level(self):
        ps = np.full((2, 4), 101000.0)
        ps[0, 0] = np.nan
        self.fixture(values={"PS": ps})
        reader = self.reader()
        field, _ = reader.read("U", 2001, 1)
        self.assertTrue(np.isnan(field).all())
        self.assertGreater(reader.qa["records"][-1]["native_underground_or_missing_ps_values"], 0)

    def test_units_coordinates_dimension_order_and_singletons(self):
        raw_lat = np.array([90.0, -90.0])
        raw_lon = np.array([-180.0, -90.0, 0.0, 90.0])
        wind = (np.arange(3)[:, None, None] * 10.0 + raw_lat[None, :, None] / 90.0
                + 3 * np.cos(np.deg2rad(raw_lon))[None, None, :])
        self.fixture(lat=raw_lat, lon=raw_lon, pressure=[100000.0, 50000.0, 10000.0], pressure_units="Pa",
                     units={"T": "degree_Celsius", "OMEGA": "hPa/s", "PS": "hPa"},
                     values={"U": wind, "T": 6.85, "OMEGA": 0.02, "PS": 1010.0}, permute=True, singleton=True)
        reader = self.reader(target_lat=[30.0, -30.0], target_lon=[0.0, 180.0])
        field, pressure = reader.read("U", 2001, 1)
        expected = np.array([30.0, -30.0])[:, None] / 90.0 + np.array([3.0, -3.0])[None, :]
        np.testing.assert_allclose(field[0], expected + 20.0)
        np.testing.assert_allclose(field[2], expected)
        np.testing.assert_array_equal(pressure, [10000.0, 50000.0, 100000.0])
        np.testing.assert_allclose(reader.read("T", 2001, 1)[0], 280.0)
        np.testing.assert_allclose(reader.read("OMEGA", 2001, 1)[0], 2.0)
        np.testing.assert_allclose(reader.read("PS", 2001, 1)[0], 101000.0)

    def test_geometric_vertical_velocity_rejected(self):
        self.fixture(units={"OMEGA": "m/s"})
        with self.assertRaisesRegex(ValueError, "geometric m/s"):
            self.reader().read("OMEGA", 2001, 1)

    def test_different_surface_pressure_grid_is_mapped_to_native_grid(self):
        self.fixture(lat=[-90.0, 0.0, 90.0], lon=np.arange(8) * 45.0,
                     ps_lat=[-90.0, 90.0], ps_lon=[0.0, 90.0, 180.0, 270.0])
        reader = self.reader()
        field, _ = reader.read("V", 2001, 1)
        np.testing.assert_allclose(field, 2.0)

    def test_actual_cf_month_missing_and_duplicate_records(self):
        path = self.fixture(dates=[(2002, 1)])
        with self.assertRaisesRegex(ValueError, "200101; found 0"):
            self.reader().read("T", 2001, 1)
        path.unlink()
        self.fixture()
        shutil.copyfile(self.data / "source.nc", self.data / "duplicate.nc")
        config = self.config()
        config["fields"]["T"]["files"] = "{data_root}/*.nc"
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.reader(config).read("T", 2001, 1)

    def test_pressure_change_rejected_without_intersection(self):
        self.fixture()
        self.fixture(filename="wind.nc", pressure=[1000.0, 400.0, 100.0])
        config = self.config()
        config["fields"]["U"]["files"] = "{data_root}/wind.nc"
        reader = self.reader(config)
        reader.read("T", 2001, 1)
        with self.assertRaisesRegex(ValueError, "Pressure levels differ"):
            reader.read("U", 2001, 1)

    def test_calendar_identity_preserved_and_mixed_calendars_rejected(self):
        self.fixture(calendar="360_day")
        reader = self.reader()
        self.assertIsNone(reader.calendar)
        reader.read("T", 2001, 1)
        self.assertEqual(reader.calendar, "360_day")
        self.fixture(filename="surface.nc", calendar="noleap")
        config = self.config()
        config["fields"]["PS"]["files"] = "{data_root}/surface.nc"
        with self.assertRaisesRegex(ValueError, "Mixed CF calendars"):
            self.reader(config).read("T", 2001, 1)

    def test_standard_and_gregorian_are_equivalent_but_noleap_is_not(self):
        self.fixture(calendar="standard")
        self.fixture(filename="surface.nc", calendar="gregorian")
        config = self.config()
        config["fields"]["PS"]["files"] = "{data_root}/surface.nc"
        reader = self.reader(config)
        reader.read("T", 2001, 1)
        self.assertEqual(reader.calendar, "proleptic_gregorian")
        self.fixture(filename="surface.nc", calendar="noleap")
        with self.assertRaisesRegex(ValueError, "Mixed CF calendars"):
            self.reader(config).read("T", 2001, 1)

    def test_modern_standard_and_proleptic_sources_share_real_month_midpoints(self):
        dates = [(2000, 1), (2000, 2), (2000, 3)]
        self.fixture(dates=dates, calendar="standard")
        self.fixture(filename="surface.nc", dates=dates, calendar="proleptic_gregorian")
        config = self.config()
        config["fields"]["PS"]["files"] = "{data_root}/surface.nc"
        reader = self.reader(config)
        temperature, _ = reader.read("T", 2000, 2)
        self.assertTrue(np.isfinite(temperature).all())
        self.assertEqual(reader.calendar, "proleptic_gregorian")
        self.assertEqual(set(reader.qa["source_calendars"]), {"standard", "proleptic_gregorian"})

        def midpoint(year, month, calendar):
            next_year, next_month = neighbor_month(year, month, 1)
            endpoints = [cftime.datetime(year, month, 1, calendar=calendar),
                         cftime.datetime(next_year, next_month, 1, calendar=calendar)]
            return float(np.mean(cftime.date2num(endpoints, "seconds since 1970-01-01", calendar=calendar)))

        for year in (2000, 2001):
            for month in (1, 2, 3):
                self.assertEqual(midpoint(year, month, "standard"), midpoint(year, month, reader.calendar))
        self.assertEqual(midpoint(2000, 2, reader.calendar) - midpoint(2000, 1, reader.calendar), 30 * 86400)
        self.assertEqual(midpoint(2001, 2, reader.calendar) - midpoint(2001, 1, reader.calendar), 29.5 * 86400)

    def test_pre_1583_standard_and_proleptic_sources_are_not_merged(self):
        dates = [(1500, 1), (1500, 2)]
        self.fixture(dates=dates, calendar="standard")
        self.fixture(filename="surface.nc", dates=dates, calendar="proleptic_gregorian")
        config = self.config()
        config["fields"]["PS"]["files"] = "{data_root}/surface.nc"
        with self.assertRaisesRegex(ValueError, "Mixed CF calendars"):
            self.reader(config).read("T", 1500, 1)

    def test_calendar_normalization_happens_after_original_time_decoding(self):
        source = self.fixture(dates=[(2001, 1)], calendar="standard")
        surface = self.fixture(filename="surface.nc", dates=[(2001, 1)], calendar="proleptic_gregorian")
        for path, calendar in ((source, "standard"), (surface, "proleptic_gregorian")):
            with Dataset(path, "a") as ds:
                ds["time"].units = "days since 1500-01-01 00:00:00"
                ds["time"][:] = date2num([datetime(2001, 1, 1)], ds["time"].units, calendar=calendar)
        config = self.config()
        config["fields"]["PS"]["files"] = "{data_root}/surface.nc"
        reader = self.reader(config)
        temperature, _ = reader.read("T", 2001, 1)
        self.assertTrue(np.isfinite(temperature).all())
        self.assertEqual(reader.calendar, "proleptic_gregorian")

    def test_provenance_hash_relative_paths_and_receipts(self):
        path = self.fixture()
        reader = self.reader()
        reader.read("T", 2001, 1)
        self.assertEqual(len(reader.inventory), 1)
        record = reader.inventory[0]
        self.assertEqual(record["path"], "source.nc")
        self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertNotIn(str(self.base), json.dumps(reader.inventory))
        receipt = Path(str(path) + ".verified.json")
        receipt.write_text(json.dumps({"bytes": path.stat().st_size, "sha256": "invalid"}))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.reader().read("T", 2001, 1)

    def test_templates_and_season_neighbours(self):
        for year, month in ((2000, 12), (2001, 1), (2001, 2)):
            self.fixture(filename=f"{year}/month_{year}{month:02d}.nc", dates=[(year, month)])
        config = self.config()
        for spec in config["fields"].values():
            spec["files"] = "{data_root}/{year}/month_{yyyymm}.nc"
        reader = self.reader(config)
        reader.read("T", 2000, 12)
        reader.read("T", 2001, 1)
        self.assertEqual(neighbor_month(2001, 1, -1), (2000, 12))
        self.assertEqual(neighbor_month(2000, 12, 1), (2001, 1))
        self.assertEqual(season_dates({"start_year": 2001, "end_year": 2001, "months": [12, 1, 2]}),
                         [(2001, 2000, 12), (2001, 2001, 1), (2001, 2001, 2)])

    def test_global_grid_and_pressure_level_contract(self):
        path = self.fixture(lon=[0.0, 10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, "global grid"):
            self.reader().read("T", 2001, 1)
        path.unlink()
        self.fixture(pressure=[1000.0, 500.0])
        with self.assertRaisesRegex(ValueError, "At least three"):
            self.reader().read("T", 2001, 1)

    def test_gaussian_source_returns_missing_polar_rows_without_extrapolating(self):
        gaussian = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(4)[0]))
        field = np.ones((3, 4, 4))
        result = strict_bilinear(field, gaussian, self.lon, [90.0, 0.0, -90.0], [45.0])
        self.assertTrue(np.isnan(result[:, (0, 2), :]).all())
        np.testing.assert_array_equal(result[:, 1, :], np.ones((3, 1)))
        with self.assertRaisesRegex(ValueError, "Nonpolar target"):
            strict_bilinear(field, gaussian, self.lon, [80.0], [45.0])
        self.fixture(lat=gaussian)
        reader = self.reader(target_lat=[90.0, 0.0, -90.0])
        temperature, _ = reader.read("T", 2001, 1)
        self.assertTrue(np.isnan(temperature[:, (0, 2), :]).all())
        self.assertTrue(np.isfinite(temperature[:, 1, :]).all())

    def test_gaussian_surface_pressure_masks_native_poles_instead_of_filling(self):
        gaussian = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(4)[0]))
        self.fixture(lat=[-90.0, -30.0, 30.0, 90.0], ps_lat=gaussian)
        reader = self.reader()
        temperature, _ = reader.read("T", 2001, 1)
        self.assertTrue(np.isfinite(temperature).all())
        self.assertGreater(reader.qa["records"][-1]["native_underground_or_missing_ps_values"], 0)

    def test_negative_direction_coordinate_units_are_rejected(self):
        path = self.fixture()
        with Dataset(path, "a") as ds:
            ds["lon"].units = "degrees_west"
        with self.assertRaisesRegex(ValueError, "degrees_east"):
            self.reader().read("T", 2001, 1)
        with Dataset(path, "a") as ds:
            ds["lon"].units = "degrees_east"
            ds["pslat"].units = "degrees_south"
        with self.assertRaisesRegex(ValueError, "degrees_north"):
            self.reader().read("T", 2001, 1)


if __name__ == "__main__":
    unittest.main()
