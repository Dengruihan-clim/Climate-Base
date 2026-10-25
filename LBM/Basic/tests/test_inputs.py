"""Synthetic monthly CF fixtures: no network, external data or credentials."""
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
from netCDF4 import Dataset, date2num

from lbm_background.inputs import load_climatology, season_dates, STANDARD_GRAVITY


class MonthlyInputsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = Path(self.tmp.name)
        self.data = self.base / "data"
        self.data.mkdir()
        self.lat = np.array([-90.0, 90.0])
        self.lon = np.array([0.0, 90.0, 180.0, 270.0])
        self.target_lat = np.array([30.0, -30.0])
        self.target_lon = np.array([0.0, 180.0])

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, **changes):
        result = {
            "_config_dir": self.base,
            "data_root": "data",
            "period": {"start_year": 1991, "end_year": 1991,
                       "months": [12, 1, 2], "minimum_valid_years": 1},
            "coordinates": {"time": "time", "latitude": "lat", "longitude": "lon", "pressure": "lev"},
            "fields": {key: {"files": "{data_root}/*.nc", "variable": key}
                       for key in ("U", "V", "T", "Q", "H", "SLP")},
            "integrity": {"require_receipts": False, "hash_inputs": True},
        }
        result.update(changes)
        return result

    def fixture(self, filename="monthly.nc", dates=None, units=None, values=None,
                lat=None, lon=None, pressure=None, pressure_units="hPa", permute=False):
        path = self.data / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        dates = dates or [(1990, 12), (1991, 1), (1991, 2)]
        lat = self.lat if lat is None else np.asarray(lat)
        lon = self.lon if lon is None else np.asarray(lon)
        pressure = np.array([1000.0, 500.0]) if pressure is None else np.asarray(pressure)
        units = {"U": "m s-1", "V": "m/s", "T": "K", "Q": "kg kg-1", "H": "m", "SLP": "Pa", **(units or {})}
        values = values or {}
        defaults = {"U": 10.0, "V": 1.0, "T": 280.0, "Q": 0.005, "H": 100.0, "SLP": 100000.0}
        with Dataset(path, "w") as ds:
            for name, count in (("time", len(dates)), ("lev", len(pressure)), ("lat", len(lat)), ("lon", len(lon))):
                ds.createDimension(name, count)
            time = ds.createVariable("time", "f8", ("time",))
            time.units = "days since 1980-01-01 00:00:00"
            time.calendar = "noleap"
            time[:] = date2num([datetime(y, m, 15) for y, m in dates], time.units, calendar=time.calendar)
            for name, axis, unit in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east"), ("lev", pressure, pressure_units)):
                var = ds.createVariable(name, "f8", (name,))
                var[:] = axis
                var.units = unit
            for key, default in defaults.items():
                dims = ("time", "lev", "lat", "lon") if key != "SLP" else ("time", "lat", "lon")
                shape = tuple(len(ds.dimensions[d]) for d in dims)
                data = np.empty(shape, dtype=np.float64)
                for i, date in enumerate(dates):
                    value = values.get(key, {}).get(date, default) if isinstance(values.get(key, {}), dict) else values[key]
                    data[i] = value
                if permute:
                    order = tuple(reversed(range(len(dims))))
                    dims = tuple(dims[i] for i in order)
                    data = np.transpose(data, order)
                var = ds.createVariable(key, "f8", dims, fill_value=-999.0)
                var.units = units[key]
                var[:] = np.ma.masked_invalid(data)
        return path

    def load(self, config=None):
        return load_climatology(config or self.config(), self.target_lat, self.target_lon)

    def test_winter_years_and_contiguous_seasons(self):
        self.assertEqual(season_dates({"start_year": 1991, "end_year": 1991, "months": [12, 1, 2]}),
                         [(1991, 1990, 12), (1991, 1991, 1), (1991, 1991, 2)])
        self.assertEqual(season_dates({"start_year": 2001, "end_year": 2001, "months": [3, 4, 5]}),
                         [(2001, 2001, 3), (2001, 2001, 4), (2001, 2001, 5)])
        for months in ([12, 2], [12, 1, 12], [], [0, 1]):
            with self.assertRaises(ValueError):
                season_dates({"start_year": 1991, "end_year": 1991, "months": months})

    def test_annual_multirecord_selection_equal_month_and_year_weights(self):
        dates = [(1990, 12), (1991, 1), (1991, 2), (1991, 12), (1992, 1), (1992, 2), (1991, 7)]
        vals = dict(zip(dates, [100.0, 10.0, 2.0, 300.0, 30.0, 6.0, 999.0]))
        self.fixture(dates=dates, values={"U": vals})
        config = self.config()
        config["period"]["end_year"] = 1992
        fields, pressure, inventory, qa = self.load(config)
        np.testing.assert_allclose(fields["U"], (200.0 + 20.0 + 4.0) / 3.0)
        np.testing.assert_array_equal(pressure, [1000.0, 500.0])
        self.assertEqual(len(inventory), 1)
        self.assertEqual(inventory[0]["path"], "monthly.nc")
        self.assertEqual(len(inventory[0]["records"]), 36)
        self.assertIsNotNone(inventory[0]["sha256"])
        self.assertEqual([r["years"] for r in qa["U"]["monthly_climatologies"]], [2, 2, 2])

    def test_normalizes_coordinates_dimension_order_and_units(self):
        raw_lat = np.array([90.0, -90.0])
        raw_lon = np.array([-180.0, -90.0, 0.0, 90.0])
        wind = (np.arange(2)[:, None, None] * 10.0
                + raw_lat[None, :, None] / 90.0
                + 3.0 * np.cos(np.deg2rad(raw_lon))[None, None, :])
        self.fixture(lat=[90.0, -90.0], lon=[-180.0, -90.0, 0.0, 90.0],
                     pressure=[50000.0, 100000.0], pressure_units="Pa", permute=True,
                     units={"H": "m**2 s**-2", "Q": "g kg-1", "T": "degree_Celsius"},
                     values={"H": 100.0 * STANDARD_GRAVITY, "Q": 5.0, "T": 6.85, "U": wind})
        fields, pressure, _, _ = self.load()
        np.testing.assert_array_equal(pressure, [1000.0, 500.0])
        np.testing.assert_allclose(fields["H"], 100.0)
        np.testing.assert_allclose(fields["Q"], 0.005)
        np.testing.assert_allclose(fields["T"], 280.0)
        np.testing.assert_allclose(fields["SLP"], 1000.0)
        expected = self.target_lat[:, None] / 90.0 + np.array([3.0, -3.0])[None, :]
        np.testing.assert_allclose(fields["U"][0], expected + 10.0)
        np.testing.assert_allclose(fields["U"][1], expected)

    def test_units_override_is_explicit_and_recorded(self):
        self.fixture(units={"Q": "broken_metadata"})
        with self.assertRaisesRegex(ValueError, "Unsupported Q units"):
            self.load()
        config = self.config()
        config["fields"]["Q"]["units_override"] = "kg kg-1"
        _, _, _, qa = self.load(config)
        self.assertEqual(qa["Q"]["source_units"], ["broken_metadata"])
        self.assertEqual(qa["Q"]["units_override"], "kg kg-1")

    def test_missing_or_duplicate_CF_month_rejected(self):
        path = self.fixture(dates=[(1990, 12), (1991, 1)])
        with self.assertRaisesRegex(ValueError, "199102; found 0"):
            self.load()
        path.unlink()
        self.fixture()
        shutil.copyfile(self.data / "monthly.nc", self.data / "duplicate.nc")
        with self.assertRaisesRegex(ValueError, "found 2"):
            self.load()

    def test_filename_does_not_override_actual_CF_month(self):
        self.fixture(filename="199012.nc", dates=[(2000, 1)])
        with self.assertRaisesRegex(ValueError, "199012; found 0"):
            self.load()

    def test_receipt_optional_but_verified_when_present(self):
        path = self.fixture()
        config = self.config()
        config["integrity"]["require_receipts"] = True
        with self.assertRaisesRegex(ValueError, "Missing required"):
            self.load(config)
        receipt = Path(str(path) + ".verified.json")
        receipt.write_text(json.dumps({"bytes": path.stat().st_size,
                                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))
        _, _, inventory, _ = self.load(config)
        self.assertTrue(inventory[0]["receipt_verified"])
        receipt.write_text(json.dumps({"bytes": path.stat().st_size, "sha256": "bad"}))
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.load(config)

    def test_missing_monthly_cells_remain_missing_and_minimum_years_applies(self):
        dates = [(1990, 12), (1991, 1), (1991, 2), (1991, 12), (1992, 1), (1992, 2)]
        self.fixture(dates=dates, values={"U": {(1990, 12): np.nan}})
        config = self.config()
        config["period"]["end_year"] = 1992
        fields, _, _, qa = self.load(config)
        self.assertTrue(np.isfinite(fields["U"]).all())
        self.assertGreater(qa["U"]["monthly_climatologies"][0]["partially_sampled_cells"], 0)
        config["period"]["minimum_valid_years"] = 2
        fields, _, _, _ = self.load(config)
        self.assertTrue(np.isnan(fields["U"]).all())

    def test_pressure_change_between_records_is_rejected(self):
        self.fixture(filename="a.nc", dates=[(1990, 12)], pressure=[1000.0, 500.0])
        self.fixture(filename="b.nc", dates=[(1991, 1), (1991, 2)], pressure=[1000.0, 400.0])
        with self.assertRaisesRegex(ValueError, "Pressure levels differ"):
            self.load()

    def test_regional_grid_and_duplicate_longitude_endpoint_rejected(self):
        path = self.fixture(lon=[0.0, 10.0, 20.0, 30.0])
        with self.assertRaisesRegex(ValueError, "global grid is required"):
            self.load()
        path.unlink()
        self.fixture(lon=[0.0, 90.0, 180.0, 360.0])
        with self.assertRaisesRegex(ValueError, "duplicate cyclic endpoints"):
            self.load()

    def test_field_glob_template_and_optional_surface_pressure(self):
        for year, month in ((1990, 12), (1991, 1), (1991, 2)):
            self.fixture(filename=f"{year}/sample_{year}{month:02d}.nc", dates=[(year, month)])
        config = self.config()
        for spec in config["fields"].values():
            spec["files"] = "{data_root}/{year}/sample_{yyyymm}.nc"
        config["fields"]["PS_SOURCE"] = deepcopy(config["fields"]["SLP"])
        fields, _, inventory, _ = self.load(config)
        np.testing.assert_allclose(fields["PS_SOURCE"], 1000.0)
        self.assertEqual(len(inventory), 3)
        self.assertTrue(all(not Path(record["path"]).is_absolute() for record in inventory))

    def test_coordinate_names_can_be_overridden_per_field(self):
        path = self.fixture()
        with Dataset(path, "a") as ds:
            ds.renameVariable("lat", "latitude")
            ds.renameVariable("lon", "longitude")
            ds.renameVariable("lev", "level")
        config = self.config()
        for spec in config["fields"].values():
            spec["coordinates"] = {"latitude": "latitude", "longitude": "longitude", "pressure": "level"}
        fields, _, _, _ = self.load(config)
        np.testing.assert_allclose(fields["U"], 10.0)

    def test_wrong_coordinate_direction_units_are_rejected(self):
        path = self.fixture()
        with Dataset(path, "a") as ds:
            ds["lon"].units = "degrees_west"
        with self.assertRaisesRegex(ValueError, "degrees_east"):
            self.load()


if __name__ == "__main__":
    unittest.main()
