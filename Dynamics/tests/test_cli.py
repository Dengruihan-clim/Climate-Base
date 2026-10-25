"""Configuration, numerical smoke checks, references and output protection."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from climate_dynamics.__main__ import run_config


def _demo(path):
    script = Path(__file__).resolve().parents[1] / "examples/generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("dynamics_demo", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.create_demo(path)


def test_four_numpy_diagnostics_and_method_references(tmp_path):
    destination = tmp_path / "demo"
    _demo(destination)
    expected_references = {"tn_waf": "takaya_nakamura_2001", "linear_rws": "sardeshmukh_hoskins_1988",
                           "rossby_wave_source": "sardeshmukh_hoskins_1988", "mass_streamfunction": "adam_et_al_2018"}
    for method, reference in expected_references.items():
        result = run_config(destination / (method + ".json"))
        assert result["status"] == "written"
        with np.load(destination / (method + "_result.npz"), allow_pickle=False) as data:
            metadata = json.loads(str(data["_metadata_json"]))
            assert metadata["references"][reference]["doi"]
            assert str(tmp_path) not in json.dumps(metadata)
            if method == "tn_waf":
                assert np.isfinite(data["waf_x"]).any()
                assert np.isnan(data["waf_x"][[0, -1]]).all()
                assert "quadratic" in metadata["amplitude_convention"]
            elif method == "linear_rws":
                np.testing.assert_allclose(data["rws"], sum(data[key] for key in
                    ("stretch_div", "stretch_vort", "adv_pert_wind", "adv_pert_vort")), rtol=0, atol=0)
            elif method == "rossby_wave_source":
                np.testing.assert_array_equal(data["rws"], data["stretch"] + data["advection"])
            else:
                assert data["mass_streamfunction"].shape == (3, 6, 5)
                np.testing.assert_array_equal(data["mass_streamfunction"][:, 0], 0.)
                np.testing.assert_array_equal(data["longitude_support_fraction"], 1.)
                assert metadata["mass_integration"]["reference_pressure_pa"] == 3000.


def test_existing_output_is_never_overwritten(tmp_path):
    destination = tmp_path / "demo"
    _demo(destination)
    config = destination / "tn_waf.json"
    run_config(config)
    output = destination / "tn_waf_result.npz"
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        run_config(config)
    assert output.read_bytes() == before


def test_wrong_physical_units_fail_before_publication(tmp_path):
    destination = tmp_path / "demo"
    _demo(destination)
    config_path = destination / "mass_streamfunction.json"
    config = json.loads(config_path.read_text())
    config["units"]["pressure"] = "hPa"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pressure"):
        run_config(config_path)
    assert not (destination / "mass_streamfunction_result.npz").exists()


def test_npz_object_arrays_are_never_unpickled(tmp_path):
    np.savez(tmp_path / "data.npz", lat=np.array([object()], dtype=object), lon=[0.])
    config = {"method": "rossby_wave_source", "input": "data.npz", "output": "result.npz", "units": {}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Object arrays"):
        run_config(tmp_path / "config.json")
    assert not (tmp_path / "result.npz").exists()


def test_dimensional_predictor_scale_is_recorded_without_rescaling(tmp_path):
    destination = tmp_path / "demo"
    _demo(destination)
    config_path = destination / "linear_rws.json"
    run_config(config_path)
    with np.load(destination / "linear_rws_result.npz") as source:
        reference = source["rws"].copy()
    config = json.loads(config_path.read_text())
    config.update(predictor_unit="K", output="linear_per_K.npz")
    config_path.write_text(json.dumps(config))
    run_config(config_path)
    with np.load(destination / "linear_per_K.npz") as source:
        np.testing.assert_array_equal(source["rws"], reference)
        metadata = json.loads(str(source["_metadata_json"]))
        assert metadata["output_units"]["rws"] == "s-2 per (K)"
