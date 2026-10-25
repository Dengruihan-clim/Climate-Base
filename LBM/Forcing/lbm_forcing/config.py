"""Portable configuration paths and explicit scientific choices."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Paths must be nonempty strings")
    path = Path(os.path.expandvars(value)).expanduser()
    return str((path if path.is_absolute() else base / path).resolve())


def load_config(path, command):
    path = Path(path).expanduser().resolve()
    return normalize_config(json.loads(path.read_text(encoding="utf-8-sig")), path.parent, command)


def normalize_config(value, directory, command):
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a JSON object")
    config = deepcopy(value)
    base = Path(directory).resolve()
    config["_config_dir"] = base
    if not isinstance(config.get("name"), str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", config["name"]):
        raise ValueError("name requires 1..80 safe ASCII filename characters")
    config["output"] = _path(config.get("output"), base)
    config.setdefault("source_product", "User-supplied reanalysis")
    if command == "diagnose":
        from .monthly_inputs import season_dates, neighbor_month
        dates = season_dates(config.get("period", {}))
        for _, year, month in dates:
            neighbor_month(year, month, -1)
            neighbor_month(year, month, 1)
        config["data_root"] = _path(config.get("data_root"), base)
        fields = config.get("fields", {})
        if set(fields) != {"T", "U", "V", "OMEGA", "PS"}:
            raise ValueError("fields must define exactly T, U, V, OMEGA and PS")
        for key, spec in fields.items():
            if not isinstance(spec, dict) or not all(isinstance(spec.get(k), str) and spec[k] for k in ("files", "variable")):
                raise ValueError(f"fields.{key} needs files and variable")
        step = config.setdefault("grid", {}).setdefault("step_degrees", 2.5)
        if isinstance(step, bool) or not isinstance(step, (int, float)) or not math.isfinite(step) or not 0 < step <= 30:
            raise ValueError("grid.step_degrees must be finite and between zero and 30")
        if any(not math.isclose(span / step, round(span / step), abs_tol=1e-8) for span in (180., 360.)):
            raise ValueError("grid.step_degrees must divide both 180 and 360 exactly")
        integrity = config.setdefault("integrity", {})
        for key, default in (("hash_inputs", True), ("require_receipts", False)):
            integrity.setdefault(key, default)
            if not isinstance(integrity[key], bool):
                raise ValueError(f"integrity.{key} must be boolean")
        regression = config.setdefault("regression", None)
        if regression is not None:
            if not isinstance(regression, dict):
                raise ValueError("regression must be null or an object")
            regression["file"] = _path(regression.get("file"), base)
            for key, default in (("year_column", "year"), ("predictor_column", "predictor"), ("predictor_units", "original supplied unit")):
                regression.setdefault(key, default)
                if not isinstance(regression[key], str) or not regression[key]:
                    raise ValueError(f"regression.{key} must be a nonempty string")
            regression.setdefault("control_column", None)
            parameters = 4 if regression["control_column"] else 3
            count = config["period"]["end_year"] - config["period"]["start_year"] + 1
            regression.setdefault("min_years", 10)
            regression.setdefault("max_lag", 3)
            for key in ("min_years", "max_lag"):
                if isinstance(regression[key], bool) or not isinstance(regression[key], int):
                    raise ValueError(f"regression.{key} must be integer")
            if not parameters <= regression["min_years"] <= count or regression["max_lag"] < 0:
                raise ValueError("Regression requires enough configured years and a nonnegative maximum lag")
    elif command == "convert":
        inputs = config.get("input", {})
        inputs["file"] = _path(inputs.get("file"), base)
        if not isinstance(inputs.get("variable"), str) or not inputs["variable"]:
            raise ValueError("input.variable must explicitly name the static pressure-level heating")
        config["input"] = inputs
        model = config.get("model", {})
        for key in ("background", "sigma"):
            model[key] = _path(model.get(key), base)
        config["model"] = model
        region = config.setdefault("region", {"south": -45., "north": 15., "west": 30., "east": 120., "width": 5.})
        from .forcing_io import region_taper
        region_taper([0.], [60.], **region)
        amplitude = config.setdefault("amplitude_multiplier", 1.)
        if isinstance(amplitude, bool) or not isinstance(amplitude, (int, float)) or not math.isfinite(amplitude):
            raise ValueError("amplitude_multiplier must be a finite real number")
    else:
        raise ValueError("Unknown command")
    return config
