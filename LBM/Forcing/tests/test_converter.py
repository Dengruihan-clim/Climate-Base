"""Synthetic Gtool/CF converter tests; no real background or reanalysis data."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

from lbm_forcing.converter import CP, SECONDS_PER_DAY, check_conversion, convert
from lbm_forcing.forcing_io import FILE_BYTES, read_forcing_grd, region_taper
from lbm_forcing.reference.horizontal import gaussian_grid


def _gtool_record(stream, item, values, units, axes):
    values = np.asarray(values, dtype=np.float32)
    nz, ny, nx = values.shape
    header = ["" for _ in range(64)]
    header[0], header[2], header[15], header[37] = "9010", item, units, "UR4"
    for start, end, count in ((29, 30, nx), (32, 33, ny), (35, 36, nz)):
        header[start], header[end] = "1", str(count)
    for index, axis in zip((28, 31, 34), axes):
        header[index] = axis
    header[63] = str(values.size)
    for payload in (("".join(value.ljust(16) for value in header)).encode("ascii"), values.astype(">f4").tobytes()):
        marker = struct.pack(">i", len(payload))
        stream.write(marker + payload + marker)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sigma = np.linspace(0.95, 0.05, 20).astype(np.float32)
        self.model()
        self.heating()

    def tearDown(self):
        self.tmp.cleanup()

    def model(self, ps=1000.0, filename="background.t42l20"):
        path = self.root / filename
        with path.open("wb") as stream:
            for name, item, units, value in (("U", "GAU", "m/s", 1.0), ("V", "GAV", "m/s", 0.0),
                                             ("T", "GAT", "K", 280.0), ("PS", "GAPS", "hPa", ps),
                                             ("Q", "GAQ", "kg/kg", 0.001)):
                count = 1 if name == "PS" else 20
                values = np.broadcast_to(np.asarray(value), (count, 64, 128))
                _gtool_record(stream, item, values, units, ("GLON128", "GGLA64", "SFC1" if name == "PS" else "CSIG20"))
        sigma_path = self.root / "GTAXLOC.CSIG20"
        with sigma_path.open("wb") as stream:
            _gtool_record(stream, "CSIG20", self.sigma.reshape(1, 1, 20), "1", ("CSIG20", "SFC1", "SFC1"))
        return path

    def heating(self, units="K day-1", values=None, pressure=None, pressure_units="hPa",
                descending_lat=False, negative_lon=False, permute=False, time_size=None,
                integrated=None, units_lat="degrees_north", units_lon="degrees_east"):
        path = self.root / "heating.nc"
        p = np.array([100.0, 300.0, 700.0, 1000.0]) if pressure is None else np.asarray(pressure)
        lat = np.array([-90.0, -30.0, 30.0, 90.0])
        if descending_lat:
            lat = lat[::-1]
        lon = np.arange(8) * 45.0
        if negative_lon:
            lon = np.arange(-4, 4) * 45.0
        if values is None:
            data = np.broadcast_to(np.cos(np.deg2rad(lon))[None, None, :], (len(p), len(lat), len(lon))).copy()
        elif callable(values):
            data = np.broadcast_to(values(p, lat, lon), (len(p), len(lat), len(lon))).copy()
        else:
            data = np.broadcast_to(values, (len(p), len(lat), len(lon))).copy()
        with Dataset(path, "w") as ds:
            for name, count in (("plev", len(p)), ("lat", len(lat)), ("lon", len(lon))):
                ds.createDimension(name, count)
            for name, values_axis, unit in (("plev", p, pressure_units), ("lat", lat, units_lat), ("lon", lon, units_lon)):
                var = ds.createVariable(name, "f8", (name,))
                var.units = unit
                var[:] = values_axis
            dims = ("plev", "lat", "lon")
            if time_size is not None:
                ds.createDimension("time", time_size)
                dims = ("time",) + dims
                data = np.broadcast_to(data[None], (time_size,) + data.shape)
            if permute:
                dims = dims[::-1]
                data = np.transpose(data, tuple(reversed(range(data.ndim))))
            var = ds.createVariable("heating", "f8", dims, fill_value=-999.0)
            var[:] = np.ma.masked_invalid(data)
            var.units = units
            if integrated is not None:
                ds.pressure_integrated = integrated
        return path

    def config(self, output="output"):
        return {"_config_dir": self.root, "name": "synthetic_forcing", "output": str(self.root / output),
                "input": {"file": str(self.root / "heating.nc"), "variable": "heating",
                          "coordinates": {"pressure": "plev", "latitude": "lat", "longitude": "lon"}},
                "model": {"background": str(self.root / "background.t42l20"), "sigma": str(self.root / "GTAXLOC.CSIG20")},
                "region": {"south": -45.0, "north": 15.0, "west": 45.0, "east": 90.0, "width": 5.0},
                "amplitude_multiplier": 1.0}

    def test_precheck_writes_nothing_and_provenance_is_private(self):
        report = check_conversion(self.config())
        self.assertEqual(report["state"], "checked")
        self.assertFalse(report["files_written"])
        self.assertFalse((self.root / "output").exists())
        self.assertEqual(report["target_shape"], [20, 64, 128])
        self.assertNotIn(str(self.root), json.dumps(report))
        self.assertEqual([r["role"] for r in report["input_manifest"]], ["heating", "background", "sigma"])

    def test_full_roundtrip_61_records_exact_zeros_and_signed_amplitude(self):
        config = self.config()
        config["region"]["east"] = 135.0
        config["amplitude_multiplier"] = 2.0
        status = convert(config)
        self.assertEqual(status["state"], "complete")
        output = Path(config["output"])
        grd = output / "synthetic_forcing.grd"
        self.assertEqual(grd.stat().st_size, FILE_BYTES)
        forcing = read_forcing_grd(grd)
        lat, lon = gaussian_grid()
        taper = region_taper(lat, lon, **config["region"])
        self.assertTrue(np.all(forcing["tmp"][:, taper == 0] == 0))
        self.assertTrue(np.any(forcing["tmp"] > 0))
        self.assertTrue(np.any(forcing["tmp"] < 0))
        for key in ("vor", "div", "ps"):
            self.assertTrue(np.all(forcing[key] == 0))
        with Dataset(output / "forcing.nc") as ds:
            np.testing.assert_array_equal(ds["temperature_forcing_k_s"][:], forcing["tmp"])
            np.testing.assert_array_equal(ds["heating_k_day"][:], forcing["tmp"].astype(np.float64) * SECONDS_PER_DAY)
            np.testing.assert_allclose(ds["target_pressure_hpa"][:], self.sigma[:, None, None] * 1000.0 + np.zeros((20, 64, 128)), rtol=1e-7)
        quality = json.loads((output / "quality_report.json").read_text())
        self.assertEqual(quality["binary_records"], 61)
        self.assertTrue(quality["netcdf_equals_binary_exactly"])
        self.assertFalse(quality["spectral_filter_applied"])

    def test_energy_and_per_second_units_match_kelvin_per_day(self):
        for index, (units, value) in enumerate((("K day-1", 1.0), ("K s-1", 1.0 / SECONDS_PER_DAY),
                                               ("W kg-1", CP / SECONDS_PER_DAY))):
            self.heating(units=units, values=value)
            config = self.config(output=f"units_{index}")
            convert(config)
            field = read_forcing_grd(Path(config["output"]) / "synthetic_forcing.grd")["tmp"]
            lat, lon = gaussian_grid()
            taper = region_taper(lat, lon, **config["region"])
            expected = np.broadcast_to((taper / SECONDS_PER_DAY)[None], field.shape).astype(np.float32)
            np.testing.assert_array_equal(field, expected)

    def test_coordinate_transpose_pressure_units_and_singleton(self):
        self.heating(values=1.0, pressure=[100000.0, 70000.0, 30000.0, 10000.0], pressure_units="Pa",
                     descending_lat=True, negative_lon=True, permute=True, time_size=1)
        report = check_conversion(self.config())
        self.assertEqual(report["input_pressure_hpa_bounds"], [100.0, 1000.0])
        convert(self.config())
        field = read_forcing_grd(self.root / "output/synthetic_forcing.grd")["tmp"]
        self.assertGreater(float(field.max()), 0.0)

    def test_model_surface_pressure_changes_log_pressure_interpolation(self):
        self.heating(values=lambda p, lat, lon: np.log(p)[:, None, None])
        first = self.config(output="ps1000")
        convert(first)
        second = self.config(output="ps800")
        second["model"]["background"] = str(self.model(ps=800.0, filename="lower_ps.t42l20"))
        convert(second)
        a = read_forcing_grd(self.root / "ps1000/synthetic_forcing.grd")["tmp"]
        b = read_forcing_grd(self.root / "ps800/synthetic_forcing.grd")["tmp"]
        lat, lon = gaussian_grid()
        iy, ix = np.argmin(np.abs(lat)), np.argmin(np.abs(lon - 67.5))
        self.assertAlmostEqual(float((b[0, iy, ix] - a[0, iy, ix]) * SECONDS_PER_DAY), float(np.log(0.8)), places=5)

    def test_missing_vertical_levels_are_discarded_but_not_zero_filled(self):
        values = np.array([2.0, np.nan, np.nan, 2.0])[:, None, None]
        self.heating(values=values)
        config = self.config()
        convert(config)
        field = read_forcing_grd(self.root / "output/synthetic_forcing.grd")["tmp"]
        lat, lon = gaussian_grid()
        taper = region_taper(lat, lon, **config["region"])
        expected = np.broadcast_to((2.0 * taper / SECONDS_PER_DAY)[None], field.shape).astype(np.float32)
        np.testing.assert_array_equal(field, expected)
        quality = json.loads((self.root / "output/quality_report.json").read_text())
        self.assertGreater(quality["endpoint_diagnostics"]["missing_pressure_values_discarded"], 0)

    def test_insufficient_active_levels_fail_before_output_creation(self):
        self.heating(values=np.array([np.nan, np.nan, 1.0, np.nan])[:, None, None])
        with self.assertRaisesRegex(ValueError, "at least two finite"):
            convert(self.config())
        self.assertFalse((self.root / "output").exists())

    def test_integrated_energy_and_compound_predictor_units_are_rejected(self):
        for units in ("W m-2", "K day-1 std-1"):
            self.heating(units=units)
            with self.assertRaisesRegex(ValueError, "unintegrated"):
                check_conversion(self.config())
        for flag in ("true", 1):
            self.heating(integrated=flag)
            with self.assertRaisesRegex(ValueError, "pressure-integrated"):
                check_conversion(self.config())

    def test_extra_non_singleton_and_wrong_coordinate_direction_rejected(self):
        self.heating(time_size=2)
        with self.assertRaisesRegex(ValueError, "non-singleton"):
            check_conversion(self.config())
        self.heating(units_lon="degrees_west")
        with self.assertRaisesRegex(ValueError, "degrees_east"):
            check_conversion(self.config())

    def test_existing_output_is_never_changed(self):
        output = self.root / "output"
        output.mkdir()
        sentinel = output / "keep.txt"
        sentinel.write_text("original")
        with self.assertRaises(FileExistsError):
            convert(self.config())
        self.assertEqual(sentinel.read_text(), "original")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_failed_status_omits_private_paths(self):
        with patch("lbm_forcing.converter.write_forcing_grd", side_effect=OSError(str(self.root / "private/filename"))):
            with self.assertRaises(OSError):
                convert(self.config())
        state_text = (self.root / "output/build_status.json").read_text()
        self.assertNotIn(str(self.root), state_text)
        state = json.loads(state_text)
        self.assertEqual(state["state"], "failed")
        self.assertEqual(state["phase"], "writing_binary")

    def test_units_override_and_invalid_background_ps(self):
        self.heating(units="legacy_label", values=1.0)
        config = self.config()
        config["input"]["units_override"] = "K/day"
        report = check_conversion(config)
        self.assertEqual(report["unit_transform"]["source_units"], "legacy_label")
        self.assertEqual(report["unit_transform"]["units_override"], "K/day")
        config["model"]["background"] = str(self.model(ps=100000.0, filename="bad_ps.t42l20"))
        with self.assertRaisesRegex(ValueError, "100..1200 hPa"):
            check_conversion(config)

    def test_predictor_scale_metadata_survives_independently_of_multiplier(self):
        self.heating(values=1.0)
        with Dataset(self.root / "heating.nc", "a") as ds:
            ds.analysis_kind = "regression"
            ds.predictor_units = "K of original supplied index"
            ds.predictor_scaling = "original supplied values; never restandardized"
            ds["heating"].amplitude_interpretation = "per one supplied predictor unit"
            ds.createDimension("year", 3)
            predictor = ds.createVariable("predictor", "f8", ("year",))
            predictor.units = "original predictor metadata"
            predictor[:] = [1.0, 2.0, 3.0]
        config = self.config()
        config["amplitude_multiplier"] = 3.0
        report = check_conversion(config)
        self.assertEqual(report["input_scale"]["predictor_units"], "K of original supplied index")
        convert(config)
        output = Path(config["output"])
        with Dataset(output / "forcing.nc") as ds:
            self.assertEqual(ds.amplitude_interpretation, "per one supplied predictor unit")
            self.assertEqual(ds.analysis_kind, "regression")
            self.assertEqual(ds.predictor_units, "K of original supplied index")
            self.assertEqual(ds.source_predictor_variable_units, "original predictor metadata")
            self.assertEqual(ds.amplitude_multiplier, 3.0)
            self.assertEqual(ds.forcing_predictor_increment, 3.0)
            self.assertIn("+3", ds.forcing_amplitude_interpretation)
        quality = json.loads((output / "quality_report.json").read_text())
        self.assertEqual(quality["input_scale"]["amplitude_interpretation"], "per one supplied predictor unit")
        self.assertEqual(quality["forcing_predictor_increment"], 3.0)
        field = read_forcing_grd(output / "synthetic_forcing.grd")["tmp"]
        lat, lon = gaussian_grid()
        taper = region_taper(lat, lon, **config["region"])
        expected = np.broadcast_to((3.0 * taper / SECONDS_PER_DAY)[None], field.shape).astype(np.float32)
        np.testing.assert_array_equal(field, expected)


if __name__ == "__main__":
    unittest.main()
