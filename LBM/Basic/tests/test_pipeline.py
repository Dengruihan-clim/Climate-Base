"""Full synthetic construction checks; no remote data or model binary needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset
import numpy as np

from lbm_background.binary import FIELD_ORDER, read_gtool
from lbm_background.config import load_config
from lbm_background.demo import create_demo
from lbm_background.pipeline import build, preflight


class PipelineTests(unittest.TestCase):
    def test_complete_build_with_cf_input_and_binary_readback(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(create_demo(Path(temporary) / "case"))
            output = Path(config["output"])
            report = preflight(config)
            self.assertFalse(output.exists())
            self.assertFalse(report["reanalysis_data_checked"])
            self.assertEqual(report["required_calendar_months"], ["2000-12", "2001-01", "2001-02"])
            result = build(config)
            self.assertEqual(result["state"], "complete")
            self.assertEqual(json.loads((output / "build_status.json").read_text())["state"], "complete")
            records = read_gtool(output / result["background"])
            with Dataset(output / "background_sigma.nc") as ds:
                self.assertEqual(ds["U"].shape, (20, 64, 128))
                self.assertEqual(ds["PS"].units, "hPa")
                self.assertGreater(np.max(abs(ds["V"][:])), 1.)
                self.assertGreaterEqual(ds["Q"][:].min(), 0.)
                self.assertGreater(np.diff(ds["potential_temperature"][:], axis=0).min(), 0.)
                for key, record in zip(FIELD_ORDER, records):
                    expected = ds[key][:].reshape(record.data.shape)
                    np.testing.assert_array_equal(record.data, expected)
            quality = json.loads((output / "quality_report.json").read_text())
            self.assertTrue(quality["binary_roundtrip_exact_float32"])
            self.assertEqual(quality["climatology"]["months"], [12, 1, 2])
            self.assertEqual(quality["static_stability"]["negative_layer_pair_fraction"], 0.)
            manifest = json.loads((output / "output_manifest.json").read_text())
            for name, details in manifest.items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), details["sha256"])
            self.assertNotIn(str(Path(temporary)), (output / "input_manifest.json").read_text())
            with self.assertRaises(FileExistsError):
                build(config)

    def test_missing_required_field_fails_with_reviewable_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(create_demo(Path(temporary) / "case"))
            config["fields"]["H"]["variable"] = "not_present"
            with self.assertRaises(ValueError):
                build(config)
            output = Path(config["output"])
            state = json.loads((output / "build_status.json").read_text())
            self.assertEqual(state["state"], "failed")
            self.assertFalse((output / (config["name"] + ".t42l20")).exists())

    def test_missing_optional_ps_is_reported_without_blocking_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config(create_demo(Path(temporary) / "case"))
            with Dataset(Path(config["data_root"]) / "monthly.nc", "a") as ds:
                ds["PS_SOURCE"][:] = np.nan
            result = build(config)
            self.assertEqual(result["state"], "complete")
            quality = json.loads((Path(config["output"]) / "quality_report.json").read_text())
            comparison = quality["surface_pressure_difference_vs_source_hPa"]
            self.assertEqual(comparison["finite_comparison_cells"], 0)
            self.assertEqual(comparison["status"], "no_valid_comparison")
            self.assertIsNone(comparison["rms"])
            self.assertTrue(quality["netcdf_roundtrip_exact_float32"])

    def test_config_rejects_path_name_and_receipt_boolean_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = create_demo(Path(temporary) / "case")
            config = json.loads(path.read_text())
            config["name"] = "../unsafe"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)
            config["name"] = "valid"
            config["integrity"]["require_receipts"] = "false"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
