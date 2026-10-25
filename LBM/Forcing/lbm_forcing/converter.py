"""Convert a static pressure-level heating field to T42L20 LBM forcing.

Stage two intentionally differs from the diagnostic input reader: missing
horizontal corners are renormalized, missing vertical levels are discarded,
and nearest valid pressure endpoints are extended.  Only columns inside the
configured taper are interpolated.  No spectral filtering, normalization,
vertical integration or additional pressure masking is performed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

from .forcing_io import (NRECORDS, read_forcing_grd, region_taper,
                         write_forcing_ctl, write_forcing_grd)
from .reference.binary import (FIELD_ORDER, FIELD_UNITS, GTOOL_ITEMS,
                               read_gtool, read_sigma)
from .reference.horizontal import gaussian_grid, periodic_bilinear
from .reference.vertical import pressure_to_sigma


CP = 1004.0
SECONDS_PER_DAY = 86400.0


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path, value):
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _unit_key(value):
    value = str(value).strip().lower()
    for before, after in (("−", "-"), ("⁻", "-"), ("¹", "1"), ("²", "2"),
                          ("**", ""), ("^", ""), ("·", ""), ("_", ""), (".", "")):
        value = value.replace(before, after)
    return re.sub(r"\s+", "", value)


def _to_k_day(field, units):
    unit = _unit_key(units)
    if unit in {"k/day", "k/d", "kday-1", "kd-1", "kelvin/day", "kelvinday-1"}:
        factor = 1.0
    elif unit in {"k/s", "ks-1", "ksec-1", "kelvin/s", "kelvinsecond-1"}:
        factor = SECONDS_PER_DAY
    elif unit in {"w/kg", "wkg-1", "watt/kg", "watts/kg", "jkg-1s-1", "j/kg/s"}:
        factor = SECONDS_PER_DAY / CP
    else:
        raise ValueError("Heating must have unintegrated K/day, K/s or W/kg units; per-predictor and W/m2 units must be converted explicitly")
    return field * factor, {"input_units": str(units), "output_units": "K day-1",
                            "multiplicative_factor": factor, "cp_J_kg_K": CP}


def _is_true(value):
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="replace")
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    if isinstance(value, (list, tuple, np.ndarray)):
        return any(_is_true(item) for item in np.asarray(value).ravel())
    return value is not None and bool(value == 1)


def _plain_attribute(value):
    """Keep a NetCDF metadata value without copying unrelated path attributes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return _plain_attribute(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain_attribute(v) for v in value]
    return value


def _scale_metadata(ds, variable):
    scale = {}
    if "amplitude_interpretation" in variable.ncattrs():
        scale["amplitude_interpretation"] = _plain_attribute(variable.getncattr("amplitude_interpretation"))
    for name in ("analysis_kind", "analysis_type", "analysis", "predictor_scaling"):
        if name in ds.ncattrs():
            scale[name] = _plain_attribute(ds.getncattr(name))
    if "predictor_units" in ds.ncattrs():
        scale["dataset_predictor_units"] = _plain_attribute(ds.getncattr("predictor_units"))
        scale["predictor_units"] = scale["dataset_predictor_units"]
    if "predictor" in ds.variables and "units" in ds.variables["predictor"].ncattrs():
        scale["predictor_variable_units"] = _plain_attribute(ds.variables["predictor"].getncattr("units"))
        scale.setdefault("predictor_units", scale["predictor_variable_units"])
    return scale


def _axis(ds, name, description):
    if name not in ds.variables or ds.variables[name].ndim != 1:
        raise ValueError(f"Missing one-dimensional {description} coordinate {name!r}")
    variable = ds.variables[name]
    values = np.ma.asarray(variable[:], dtype=np.float64).filled(np.nan)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{description} coordinate is empty or contains missing values")
    return values, variable.dimensions[0], str(getattr(variable, "units", ""))


def _read_heating(config):
    try:
        from netCDF4 import Dataset
    except ImportError as error:
        raise ImportError("Heating conversion requires netCDF4; install this package's dependencies") from error
    spec = config["input"]
    path = Path(spec["file"])
    names = {"pressure": "plev", "latitude": "lat", "longitude": "lon",
             **spec.get("coordinates", {})}
    with Dataset(path) as ds:
        name = spec["variable"]
        if name not in ds.variables:
            raise ValueError(f"Missing configured heating variable {name!r}")
        variable = ds.variables[name]
        if _is_true(getattr(ds, "pressure_integrated", False)) or _is_true(getattr(variable, "pressure_integrated", False)):
            raise ValueError("A pressure-integrated field cannot be used as layer heating")
        pressure, p_dim, p_units = _axis(ds, names["pressure"], "pressure")
        lat, y_dim, y_units = _axis(ds, names["latitude"], "latitude")
        lon, x_dim, x_units = _axis(ds, names["longitude"], "longitude")
        if _unit_key(y_units) not in {"degreesnorth", "degreenorth", "degreesn", "degreen"}:
            raise ValueError("Latitude coordinate units must explicitly indicate degrees_north")
        if _unit_key(x_units) not in {"degreeseast", "degreeeast", "degreese", "degreee"}:
            raise ValueError("Longitude coordinate units must explicitly indicate degrees_east")
        p_unit = _unit_key(p_units)
        if p_unit in {"pa", "pascal", "pascals"}:
            pressure *= 0.01
            p_factor = 0.01
        elif p_unit in {"hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars"}:
            p_factor = 1.0
        else:
            raise ValueError("Pressure coordinate units must be Pa or hPa")
        positive = str(getattr(ds.variables[names["pressure"]], "positive", "down")).lower()
        if positive != "down":
            raise ValueError("Pressure coordinate positive direction must be down")
        p_order, y_order = np.argsort(pressure), np.argsort(lat)
        pressure, lat = pressure[p_order], lat[y_order]
        lon = lon % 360.0
        x_order = np.argsort(lon)
        lon = lon[x_order]
        if pressure.size < 2 or np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0):
            raise ValueError("At least two distinct positive pressure coordinates are required")
        if lat.size < 2 or np.any(np.abs(lat) > 90) or np.any(np.diff(lat) <= 0):
            raise ValueError("Latitude coordinate must have distinct values in [-90,90]")
        if lon.size < 4 or np.any(np.diff(lon) <= 1e-9):
            raise ValueError("Global longitude must contain distinct samples and no repeated cyclic endpoint")
        gaps = np.diff(np.r_[lon, lon[0] + 360.0])
        if gaps.max() > 1.5 * np.median(gaps) + 1e-8:
            raise ValueError("A global longitude grid is required; regional cyclic gaps are unsupported")
        dimensions = [p_dim, y_dim, x_dim]
        if len(set(dimensions)) != 3 or any(dim not in variable.dimensions for dim in dimensions):
            raise ValueError("Heating variable dimensions disagree with its configured coordinate dimensions")
        selectors, retained = [], []
        for dim in variable.dimensions:
            if dim in dimensions:
                selectors.append(slice(None))
                retained.append(dim)
            elif len(ds.dimensions[dim]) == 1:
                selectors.append(0)
            else:
                raise ValueError(f"Static heating has an unselected non-singleton dimension {dim!r}")
        values = np.ma.asarray(variable[tuple(selectors)], dtype=np.float64).filled(np.nan)
        values = np.transpose(values, [retained.index(dim) for dim in dimensions])
        for axis, order in enumerate((p_order, y_order, x_order)):
            values = np.take(values, order, axis=axis)
        values[~np.isfinite(values)] = np.nan
        source_units = str(getattr(variable, "units", ""))
        applied_units = spec.get("units_override", source_units)
        values, transform = _to_k_day(values, applied_units)
        scale = _scale_metadata(ds, variable)
    transform.update({"source_units": source_units, "units_override": spec.get("units_override"),
                      "pressure_input_units": p_units, "pressure_to_hpa_factor": p_factor,
                      "source_missing_values": int(np.sum(~np.isfinite(values))), "input_scale": scale})
    return values, pressure, lat, lon, transform


def _read_model(config):
    records = read_gtool(config["model"]["background"])
    if len(records) != 5 or tuple(record.item for record in records) != GTOOL_ITEMS:
        raise ValueError("Background must contain the ordered Gtool U,V,T,PS,Q fields")
    for kind, units, record in zip(FIELD_ORDER, FIELD_UNITS, records):
        expected_shape = (1 if kind == "PS" else 20, 64, 128)
        expected_axes = ("GLON128", "GGLA64", "SFC1" if kind == "PS" else "CSIG20")
        axes = tuple(record.header[index].strip() for index in (28, 31, 34))
        if record.endian != ">" or record.data.shape != expected_shape or axes != expected_axes or record.units != units:
            raise ValueError("Background must use verified big-endian T42L20 axes, dimensions and field units")
        if not np.isfinite(record.data).all() or np.any(record.data == -999.0):
            raise ValueError("Background contains missing or nonfinite values")
    ps = records[3].data[0].astype(np.float64)
    if np.any((ps < 100.0) | (ps > 1200.0)):
        raise ValueError("Background PS must be finite surface pressure in 100..1200 hPa")
    sigma = read_sigma(config["model"]["sigma"])
    lat, lon = gaussian_grid()
    return ps, sigma, lat, lon


def _prepare(config):
    name = config.get("name", "forcing")
    if (not isinstance(name, str) or not name or len(name) > 80
            or name in {".", ".."} or any(c in name for c in "/\\\n\r")):
        raise ValueError("name must be a safe single ASCII filename component")
    try:
        name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("name must use ASCII characters") from error
    amplitude = float(config.get("amplitude_multiplier", 1.0))
    if not np.isfinite(amplitude):
        raise ValueError("amplitude_multiplier must be finite")
    values, pressure, lat, lon, units = _read_heating(config)
    model_ps, sigma, target_lat, target_lon = _read_model(config)
    taper = region_taper(target_lat, target_lon, **config.get("region", {}))
    active = taper > 0.0
    if not np.any(active):
        raise ValueError("Configured region contains no active T42 Gaussian-grid points")
    regridded = periodic_bilinear(values, lat, lon, target_lat, target_lon)
    counts = np.sum(np.isfinite(regridded), axis=0)
    insufficient = active & (counts < 2)
    if np.any(insufficient):
        raise ValueError(f"Active columns require at least two finite heating levels; {int(insufficient.sum())} columns are insufficient")
    manifest = []
    for role, path in (("heating", config["input"]["file"]), ("background", config["model"]["background"]),
                       ("sigma", config["model"]["sigma"])):
        path = Path(path)
        manifest.append({"role": role, "file": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)})
    return {"name": name, "amplitude": amplitude, "heating": regridded, "pressure": pressure,
            "ps": model_ps, "sigma": sigma, "lat": target_lat, "lon": target_lon,
            "taper": taper, "active": active, "counts": counts, "units": units,
            "scale": units["input_scale"], "manifest": manifest}


def _forcing_scale(data):
    original = data["scale"].get("amplitude_interpretation")
    if original == "per one supplied predictor unit":
        return {"forcing_amplitude_interpretation": f"Response to a {data['amplitude']:+g} change in the original supplied predictor units",
                "forcing_predictor_increment": data["amplitude"]}
    return {"forcing_amplitude_interpretation": f"Supplied static heating multiplied by {data['amplitude']:g}; source scale is retained separately"}


def check_conversion(config):
    """Read/validate heating and model references; never create output files."""
    data = _prepare(config)
    return {"state": "checked", "name": data["name"], "target_shape": [20, 64, 128],
            "input_pressure_levels": int(len(data["pressure"])),
            "input_pressure_hpa_bounds": [float(data["pressure"].min()), float(data["pressure"].max())],
            "active_columns": int(data["active"].sum()),
            "minimum_active_finite_levels": int(data["counts"][data["active"]].min()),
            "background_ps_hpa_bounds": [float(data["ps"].min()), float(data["ps"].max())],
            "unit_transform": data["units"], "amplitude_multiplier": data["amplitude"],
            "input_scale": data["scale"], **_forcing_scale(data),
            "input_manifest": data["manifest"], "output_exists": Path(config["output"]).exists(),
            "files_written": False}


def _write_netcdf(path, data, thermal):
    from netCDF4 import Dataset
    path = Path(path)
    temporary = path.with_name("." + path.name + ".partial")
    try:
        with Dataset(temporary, "w", format="NETCDF4") as ds:
            for name, size in (("sigma", 20), ("lat", 64), ("lon", 128)):
                ds.createDimension(name, size)
            for name, values, units in (("sigma", data["sigma"], "1"), ("lat", data["lat"], "degrees_north"),
                                         ("lon", data["lon"], "degrees_east")):
                var = ds.createVariable(name, "f8", (name,))
                var[:] = values
                var.units = units
            ds["sigma"].standard_name = "atmosphere_sigma_coordinate"
            ds["sigma"].positive = "down"
            ds["sigma"].formula_terms = "sigma: sigma ps: model_ps ptop: ptop"
            ds.createVariable("ptop", "f8").assignValue(0.0)
            ds["ptop"].units = "hPa"
            three = ("sigma", "lat", "lon")
            two = ("lat", "lon")
            fields = (("temperature_forcing_k_s", thermal, "K s-1", "f4", three),
                      ("heating_k_day", thermal.astype(np.float64) * SECONDS_PER_DAY, "K day-1", "f8", three),
                      ("target_pressure_hpa", data["sigma"][:, None, None] * data["ps"][None], "hPa", "f8", three),
                      ("model_ps", data["ps"], "hPa", "f8", two),
                      ("taper", data["taper"], "1", "f8", two),
                      ("valid_source_level_count", np.where(data["active"], data["counts"], 0), "1", "i4", two))
            for name, values, units, dtype, dims in fields:
                var = ds.createVariable(name, dtype, dims, zlib=True, complevel=4)
                var[:] = values
                var.units = units
            ds["heating_k_day"].comment = "Computed from the exact float32 binary temperature forcing, multiplied by 86400"
            ds["valid_source_level_count"].comment = "Finite source levels after horizontal interpolation; zero outside the active taper"
            ds.setncatts({"title": data["name"] + " T42L20 thermal forcing", "authors": "RH_Deng; Codex (OpenAI)",
                          "created_utc": _now(), "pressure_integrated": "false", "cp_J_kg_K": CP,
                          "amplitude_multiplier": data["amplitude"],
                          "horizontal_method": "periodic bilinear with finite-corner weight renormalization",
                          "vertical_method": "natural cubic spline in log pressure; missing levels discarded and nearest valid endpoints extended",
                          "spatial_support": "raised-cosine taper with exact exterior/boundary zeros; no spectral filtering or normalization"})
            ds.input_scale_metadata = json.dumps(data["scale"], ensure_ascii=False)
            for name in ("amplitude_interpretation", "analysis_kind", "analysis_type", "analysis", "predictor_scaling", "predictor_units"):
                if name in data["scale"]:
                    ds.setncattr(name, data["scale"][name])
            if "predictor_variable_units" in data["scale"]:
                ds.source_predictor_variable_units = data["scale"]["predictor_variable_units"]
            ds.setncatts(_forcing_scale(data))
        if path.exists() or path.is_symlink():
            raise FileExistsError("Refusing to overwrite a NetCDF output")
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def convert(config):
    """Create a new forcing directory and verify all binary/NetCDF payloads."""
    output = Path(config["output"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output directory already exists; choose a new directory")
    # Validate data and interpolation before creating any output directory.
    data = _prepare(config)
    active = data["active"]
    sigma_heating, endpoints = pressure_to_sigma(data["heating"][::-1, active], data["pressure"][::-1],
                                                data["ps"][active], data["sigma"])
    thermal = np.zeros((20, 64, 128), dtype=np.float64)
    thermal[:, active] = sigma_heating * data["taper"][active] * data["amplitude"] / SECONDS_PER_DAY
    with np.errstate(over="ignore", invalid="ignore"):
        thermal = thermal.astype(np.float32)
    if not np.isfinite(thermal).all():
        raise ValueError("Converted forcing cannot be represented as finite float32 values")
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "build_status.json"
    phase = "starting"
    try:
        _json(state_path, {"state": "running", "started_utc": _now(), "name": data["name"]})
        _json(output / "input_manifest.json", data["manifest"])
        phase = "writing_binary"
        grd = output / (data["name"] + ".grd")
        ctl = output / (data["name"] + ".ctl")
        write_forcing_grd(grd, thermal)
        write_forcing_ctl(ctl, grd, data["sigma"], data["lat"], data["lon"], title=data["name"])
        phase = "checking_binary"
        binary = read_forcing_grd(grd)
        np.testing.assert_array_equal(binary["tmp"], thermal)
        for name in ("vor", "div", "ps"):
            if np.any(binary[name] != 0):
                raise ValueError("Unexpected nonthermal binary forcing")
        if np.any(binary["tmp"][:, ~active] != 0):
            raise ValueError("Exterior or boundary forcing is not exactly zero")
        phase = "writing_netcdf"
        nc_path = output / "forcing.nc"
        _write_netcdf(nc_path, data, binary["tmp"])
        phase = "checking_netcdf"
        from netCDF4 import Dataset
        with Dataset(nc_path) as ds:
            np.testing.assert_array_equal(ds["temperature_forcing_k_s"][:], binary["tmp"])
            np.testing.assert_array_equal(ds["heating_k_day"][:], binary["tmp"].astype(np.float64) * SECONDS_PER_DAY)
        actual_day = binary["tmp"].astype(np.float64) * SECONDS_PER_DAY
        quality = {"name": data["name"], "active_columns": int(active.sum()),
                   "minimum_active_finite_levels": int(data["counts"][active].min()),
                   "unit_transform": data["units"], "cp_J_kg_K": CP,
                   "input_scale": data["scale"], **_forcing_scale(data),
                   "amplitude_multiplier": data["amplitude"], "region": config.get("region", {}),
                   "bounds": {"temperature_forcing_k_s": [float(thermal.min()), float(thermal.max())],
                              "heating_k_day_from_float32": [float(actual_day.min()), float(actual_day.max())],
                              "model_ps_hpa": [float(data["ps"].min()), float(data["ps"].max())]},
                   "sign_counts": {"positive": int(np.sum(thermal > 0)), "negative": int(np.sum(thermal < 0)),
                                   "zero": int(np.sum(thermal == 0))},
                   "endpoint_diagnostics": endpoints,
                   "binary_records": NRECORDS, "binary_temperature_roundtrip_exact_float32": True,
                   "binary_nonthermal_fields_exact_zero": True,
                   "exterior_and_boundary_exact_zero": True, "netcdf_equals_binary_exactly": True,
                   "spectral_filter_applied": False, "pressure_mask_or_integration_applied": False,
                   "normalization_applied": False}
        _json(output / "quality_report.json", quality)
        files = {path.name: {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                 for path in sorted(output.iterdir()) if path.is_file()
                 and path.name not in {"build_status.json", "output_manifest.json"}}
        _json(output / "output_manifest.json", files)
        status = {"state": "complete", "completed_utc": _now(), "name": data["name"],
                  "forcing_grd": grd.name, "forcing_ctl": ctl.name, "forcing_nc": nc_path.name,
                  "binary_records": NRECORDS, "binary_and_netcdf_roundtrip": True}
        _json(state_path, status)
        return {**status, "output": str(output)}
    except Exception as error:
        # Exceptions can contain private absolute filenames.  Keep only the
        # safe stage/type in the shareable status; let the caller see details.
        _json(state_path, {"state": "failed", "updated_utc": _now(), "phase": phase,
                           "error_type": type(error).__name__,
                           "message": "Conversion failed; inspect the exception in the calling session"})
        raise
