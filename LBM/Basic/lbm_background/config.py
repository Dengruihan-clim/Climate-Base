"""Load a portable JSON configuration; paths are relative to its directory."""
from copy import deepcopy
import json
import os
from pathlib import Path
import re


def _path(value, base):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Configured paths must be nonempty strings")
    candidate = Path(os.path.expandvars(value)).expanduser()
    return (candidate if candidate.is_absolute() else base / candidate).resolve()


def load_config(path):
    path = Path(path).expanduser().resolve()
    with path.open(encoding="utf-8-sig") as stream:
        config = json.load(stream)
    return normalize_config(config, path.parent)


def normalize_config(config, directory):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a JSON object")
    config = deepcopy(config)
    base = Path(directory).resolve()
    config["_config_dir"] = base
    name = config.get("name", "")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", name):
        raise ValueError("name must be 1..80 ASCII filename characters: letters, digits, dot, underscore, hyphen")
    for key in ("data_root", "output"):
        if key not in config:
            raise ValueError(f"Missing configuration key: {key}")
        config[key] = str(_path(config[key], base))
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must specify topography, sigma and template reference files")
    for key in ("topography", "sigma", "template"):
        if key not in model:
            raise ValueError(f"Missing model.{key}")
        model[key] = str(_path(model[key], base))
    if len(set(model[key] for key in ("topography", "sigma", "template"))) != 3:
        raise ValueError("The three model reference files must be distinct")
    if not isinstance(config.get("period"), dict):
        raise ValueError("period must specify start_year, end_year and months")
    from .inputs import REQUIRED_FIELDS, season_dates
    season_dates(config["period"])
    period = config["period"]
    minimum = period.setdefault("minimum_valid_years", 1)
    number = period["end_year"] - period["start_year"] + 1
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= number:
        raise ValueError("period.minimum_valid_years must be between one and the climatology year count")
    fields = config.get("fields")
    if not isinstance(fields, dict) or not set(REQUIRED_FIELDS).issubset(fields):
        raise ValueError(f"fields must define {', '.join(REQUIRED_FIELDS)}")
    extra = set(fields) - set(REQUIRED_FIELDS) - {"PS_SOURCE"}
    if extra:
        raise ValueError(f"Unknown input field keys: {sorted(extra)}")
    for key, field in fields.items():
        if not isinstance(field, dict) or not all(isinstance(field.get(k), str) and field[k] for k in ("files", "variable")):
            raise ValueError(f"fields.{key} requires nonempty files and variable strings")
    for section, defaults in (("integrity", {"hash_inputs": True, "require_receipts": False}),
                              ("diagnostics", {"plot": False})):
        settings = config.setdefault(section, {})
        if not isinstance(settings, dict):
            raise ValueError(f"{section} must be an object")
        for key, value in defaults.items():
            settings.setdefault(key, value)
            if not isinstance(settings[key], bool):
                raise ValueError(f"{section}.{key} must be true or false")
    product = config.setdefault("source_product", "User-supplied monthly reanalysis")
    if not isinstance(product, str) or not product.strip():
        raise ValueError("source_product must be a nonempty string")
    return config
