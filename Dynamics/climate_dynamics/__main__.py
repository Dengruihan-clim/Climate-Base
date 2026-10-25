"""Run a configured dynamical diagnostic on explicitly named numeric NPZ arrays.

Methods and original references: Takaya & Nakamura (2001), Sardeshmukh &
Hoskins (1988), Dawson (2016), Adams & Swarztrauber (1999), Oort & Yienger
(1996), Adam et al. (2018).
DOIs and implementation boundaries are recorded in citations.py/REFERENCES.md.
Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import __version__, diagnose_wind, fixed_longitude_mean, mass_streamfunction
from .citations import CITATIONS, METHOD_REFERENCES
from .waves import RWS_KEYS, linear_rws, rossby_wave_source, tn_waf


def _path(value, parent):
    if not isinstance(value, str) or not value or "<" in value or ">" in value:
        raise ValueError("Replace input/output path placeholders")
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path


def _array(archive, name):
    if name not in archive.files:
        raise ValueError("Missing input array: " + name)
    value = archive[name]
    if value.dtype.kind not in "fiu":
        raise ValueError("Input arrays must be real numeric arrays: " + name)
    return np.asarray(value, dtype=float)


def _units(declared, expected):
    if not isinstance(declared, dict):
        raise ValueError("units must explicitly declare the numeric archive's physical units")
    for name, unit in expected.items():
        if declared.get(name) != unit:
            raise ValueError(f"Declare units.{name}={unit!r}; this runner performs no implicit unit conversion")


def _coordinates(data):
    lat, lon = _array(data, "lat"), _array(data, "lon")
    if (lat.ndim != 1 or not lat.size or not np.isfinite(lat).all() or np.any(np.abs(lat) > 90)
            or (lat.size > 1 and not (np.all(np.diff(lat) > 0) or np.all(np.diff(lat) < 0)))):
        raise ValueError("lat must be finite monotonic latitude degrees within [-90,90]")
    if (lon.ndim != 1 or not lon.size or not np.isfinite(lon).all()
            or (lon.size > 1 and not (np.all(np.diff(lon) > 0) or np.all(np.diff(lon) < 0)))
            or len(np.unique(lon % 360)) != lon.size):
        raise ValueError("lon must be finite monotonic longitude degrees without cyclic duplicates")
    return lat, lon


def _field(data, name, lat, lon):
    value = _array(data, name)
    if value.ndim < 2 or value.shape[-2:] != (lat.size, lon.size):
        raise ValueError("Final field axes must match lat/lon: " + name)
    return value


def _save_new(path, arrays, metadata):
    if path.exists() or path.is_symlink():
        raise FileExistsError("Output exists; choose a new filename")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez(stream, **arrays, _metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False)))
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_config(config_path):
    """Write one new NPZ result with method-specific literature metadata."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"method", "input", "output", "units"}
    if (not isinstance(config, dict) or required - config.keys()
            or config.keys() - (required | {"options", "amplitude_description", "predictor_unit"})):
        raise ValueError("Configuration has missing or unknown keys")
    method = config["method"]
    if method not in METHOD_REFERENCES:
        raise ValueError("Unknown diagnostic method")
    predictor_unit = config.get("predictor_unit", "1")
    if (not isinstance(predictor_unit, str) or not predictor_unit.strip()
            or ("predictor_unit" in config and method != "linear_rws")):
        raise ValueError("predictor_unit must be nonempty text and is supported only by linear_rws")
    options = config.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")
    output = _path(config["output"], config_path.parent)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output exists; choose a new filename")
    input_path = _path(config["input"], config_path.parent)
    initial_stat = input_path.stat()
    initial_stamp = initial_stat.st_size, initial_stat.st_mtime_ns
    with input_path.open("rb") as stream:
        input_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    metadata = {"package": "climate-base-dynamics", "version": __version__, "method": method,
                "authors": "RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring)",
                "numpy_version": np.__version__,
                "input_sha256": input_hash, "declared_input_units": config["units"],
                "options": options,
                "references": {key: CITATIONS[key] for key in METHOD_REFERENCES[method]}}
    if "amplitude_description" in config:
        if not isinstance(config["amplitude_description"], str):
            raise ValueError("amplitude_description must be text")
        metadata["amplitude_description"] = config["amplitude_description"]
    loaded = np.load(input_path, allow_pickle=False)
    if not isinstance(loaded, np.lib.npyio.NpzFile):
        raise ValueError("Input must be a numeric NPZ archive")
    with loaded as data:
        lat, lon = _coordinates(data)
        arrays = {"lat": lat, "lon": lon}
        if method == "tn_waf":
            _units(config["units"], {"psi": "m2 s-1", "u_background": "m s-1", "v_background": "m s-1"})
            result = tn_waf(*[_field(data, key, lat, lon) for key in ("psi", "u_background", "v_background")],
                            lat, lon, **options)
            arrays.update(result)
            metadata.update(output_units={key: "m2 s-2" for key in result},
                            amplitude_convention="quadratic in supplied streamfunction amplitude; not a linear regression coefficient")
        elif method in {"rossby_wave_source", "linear_rws"}:
            _units(config["units"], {"eta": "s-1", "div": "s-1", "udiv": "m s-1", "vdiv": "m s-1",
                                      "eta_x": "s-1 m-1", "eta_y": "s-1 m-1"})
            if options:
                raise ValueError("RWS assembly has no configurable numerical options")
            if method == "rossby_wave_source":
                result = rossby_wave_source({key: _field(data, key, lat, lon) for key in RWS_KEYS})
            else:
                background = {key: _field(data, "background_" + key, lat, lon) for key in RWS_KEYS}
                perturbation = {key: _field(data, "perturbation_" + key, lat, lon) for key in RWS_KEYS}
                result = linear_rws(background, perturbation)
                metadata["linearization"] = "first variation only; perturbation eta and gradients exclude the fixed planetary contribution"
            arrays.update(result)
            metadata["output_units"] = {key: "s-2" for key in result}
            if method == "linear_rws":
                metadata["predictor_unit"] = predictor_unit
                metadata["perturbation_unit_convention"] = "Each perturbation component is expressed in its declared physical base unit per predictor_unit"
                metadata["coefficient_scale"] = "Preserves the supplied perturbation/component coefficient scale"
                if predictor_unit != "1":
                    metadata["output_units"] = {key: "s-2 per (" + predictor_unit + ")" for key in result}
        elif method == "wind":
            _units(config["units"], {"u": "m s-1", "v": "m s-1"})
            result = diagnose_wind(_field(data, "u", lat, lon), _field(data, "v", lat, lon), lat, lon, **options)
            arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
            metadata["wind_diagnostic"] = {key: value for key, value in result.items() if key not in arrays}
            metadata["output_units"] = result["units"]
            metadata["backend_versions"] = {}
            for distribution in ("windspharm", "pyspharm"):
                try:
                    metadata["backend_versions"][distribution] = importlib.metadata.version(distribution)
                except importlib.metadata.PackageNotFoundError:
                    metadata["backend_versions"][distribution] = "metadata unavailable"
        else:
            _units(config["units"], {"v": "m s-1", "pressure": "Pa"})
            allowed = {"longitude_bounds", "cyclic", "min_fraction", "reference_pressure_pa", "radius", "gravity"}
            if options.keys() - allowed:
                raise ValueError("Unknown mass_streamfunction options")
            v = _field(data, "v", lat, lon)
            pressure = _array(data, "pressure")
            if v.ndim != 4 or pressure.ndim != 1 or v.shape[1] != pressure.size:
                raise ValueError("Mass CLI requires v=(sample,pressure,lat,lon) and pressure=(pressure,)")
            averaging = {key: options[key] for key in ("longitude_bounds", "cyclic", "min_fraction") if key in options}
            integrating = {key: options[key] for key in ("reference_pressure_pa", "radius", "gravity") if key in options}
            means, support, mask, average_metadata = fixed_longitude_mean({"v": v}, longitude=lon, **averaging)
            psi, integration_metadata = mass_streamfunction(means["v"], pressure, lat, **integrating)
            arrays = {"lat": lat, "pressure": pressure, "v_mean": means["v"], "mass_streamfunction": psi,
                      "longitude_support_fraction": support, "fixed_longitude_mask": mask, "lon": lon}
            metadata.update(longitude_average=average_metadata, mass_integration=integration_metadata,
                            output_units={"v_mean": "m s-1", "mass_streamfunction": "kg s-1",
                                          "longitude_support_fraction": "1", "pressure": "Pa"})
    final_stat = input_path.stat()
    if (final_stat.st_size, final_stat.st_mtime_ns) != initial_stamp:
        raise ValueError("Input changed during hashing or calculation; no result was published")
    _save_new(output, arrays, metadata)
    return {"status": "written", "method": method, "output_arrays": sorted(arrays),
            "reference_keys": list(metadata["references"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_config(arguments.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
