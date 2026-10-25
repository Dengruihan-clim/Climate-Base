"""Explicit affine unit transformations; no guessing from variable names."""
from __future__ import annotations
import re
import numpy as np


def unit_key(value):
    value = str(value).strip().lower()
    for old, new in (("−", "-"), ("⁻", "-"), ("¹", "1"), ("²", "2"),
                     ("**", ""), ("^", ""), ("·", ""), ("_", ""), (".", "")):
        value = value.replace(old, new)
    return re.sub(r"\s+", "", value)


# Each entry converts to its group's base unit by value*scale+offset.
_GROUPS = (
    ("temperature", 1., 0., ("k", "kelvin", "kelvins")),
    ("temperature", 1., 273.15, ("degc", "degreecelsius", "degreescelsius", "celsius")),
    ("pressure", 1., 0., ("pa", "pascal", "pascals")),
    ("pressure", 100., 0., ("hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars")),
    ("mass_fraction", 1., 0., ("kgkg-1", "kg/kg", "1", "dimensionless")),
    ("mass_fraction", .001, 0., ("gkg-1", "g/kg")),
    ("velocity", 1., 0., ("ms-1", "m/s", "msec-1")),
    ("pressure_velocity", 1., 0., ("pas-1", "pa/s", "pasec-1")),
    ("pressure_velocity", 100., 0., ("hpas-1", "hpa/s", "hpasec-1")),
    ("length", 1., 0., ("m", "meter", "meters", "metre", "metres")),
    ("length", .001, 0., ("mm", "millimeter", "millimeters", "millimetre", "millimetres")),
    ("heating", 1., 0., ("ks-1", "k/s", "ksec-1")),
    ("heating", 1. / 86400., 0., ("kday-1", "k/day", "kd-1", "k/d")),
)
_UNITS = {alias: (group, scale, offset) for group, scale, offset, aliases in _GROUPS for alias in aliases}


def same_units(a, b):
    """True only for equivalent representations, not merely same dimension."""
    aa, bb = unit_key(a), unit_key(b)
    return aa == bb or (aa in _UNITS and bb in _UNITS and _UNITS[aa] == _UNITS[bb])


def conversion_plan(source_units, target_units=None, *, scale=None, offset=0.):
    if not str(source_units).strip():
        raise ValueError("Source units are missing; correct metadata or declare units_override")
    if scale is not None:
        if not target_units:
            raise ValueError("An explicit scale requires explicit target_units")
        scale, offset = float(scale), float(offset)
        if not np.isfinite(scale) or not np.isfinite(offset):
            raise ValueError("Explicit scale and offset must be finite")
        return scale, offset, str(target_units)
    if offset != 0:
        raise ValueError("Explicit offset requires explicit scale, including scale=1")
    target_units = str(source_units) if target_units is None else str(target_units)
    if same_units(source_units, target_units):
        return 1., 0., target_units
    source, target = _UNITS.get(unit_key(source_units)), _UNITS.get(unit_key(target_units))
    if source is None or target is None or source[0] != target[0]:
        raise ValueError("Unsupported or physically incompatible unit conversion; provide a documented explicit scale/offset")
    return source[1] / target[1], (source[2] - target[2]) / target[1], target_units


def convert_units(values, source_units, target_units, *, scale=None, offset=0.):
    """Return a float64 copy using only a known or explicitly declared transform.

    Geopotential to height, accumulated precipitation to rates, and energy to
    temperature tendencies require explicit physical assumptions and are not
    inferred by this function. Masked values and NaN/Inf become NaN.
    """
    factor, shift, _ = conversion_plan(source_units, target_units, scale=scale, offset=offset)
    return _apply_affine(values, factor, shift)


def _apply_affine(values, factor, shift):
    if np.iscomplexobj(values):
        raise ValueError("Complex values are unsupported; no imaginary component is discarded")
    source = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    finite = np.isfinite(source)
    with np.errstate(over="ignore", invalid="ignore"):
        result = source * factor + shift
    if np.any(finite & ~np.isfinite(result)):
        raise ValueError("Unit conversion produced nonfinite values from finite input")
    return np.where(finite, result, np.nan)
