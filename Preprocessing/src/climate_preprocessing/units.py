"""Explicit affine unit transforms with xarray/Dask execution.

References
----------
The conversion table and strict errors originate in Climate-Base 0.1.0.
MetPy's unit-aware adapter design was consulted, not copied:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/xarray.py
The block execution follows xarray's public ``apply_ufunc`` interface:
https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html

Climate-Base improvements
-------------------------
DataArray transforms preserve coordinates and attributes, retain Dask chunks,
reject finite-input overflow when a block executes, and discard obsolete
packing encodings and range metadata. Removed metadata are recorded without
scanning the field. Custom affine conversions also clear an unverified CF
standard_name. The legacy ndarray API computes only at its return boundary.
This module intentionally keeps the existing small, explicit conversion table;
it does not claim the general unit vocabulary of Pint or MetPy.
"""
from __future__ import annotations
import json
import re
import numpy as np
import xarray as xr


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
    if np.iscomplexobj(values):
        raise ValueError("Complex values are unsupported; no imaginary component is discarded")
    # This compatibility entry accepts already-materialized values. New large
    # workflows should call convert_dataarray_units without crossing this eager
    # return boundary.
    source = np.ma.asarray(values, dtype=np.float64).filled(np.nan)
    field = xr.DataArray(source).chunk("auto")
    converted = convert_dataarray_units(
        field, target_units, source_units=source_units, scale=scale, offset=offset
    )
    return converted.compute().data


def convert_dataarray_units(data, target_units, *, source_units=None, scale=None, offset=0.):
    """Convert a real DataArray using a deferred, chunk-preserving transform.

    Parameters
    ----------
    data : xarray.DataArray
        Input values. Eager arrays are converted to Dask chunks; existing Dask
        chunks are preserved. Coordinates are not converted by this operation.
    target_units : str
        A known compatible unit or a target paired with an explicit scale.
    source_units : str, optional
        Explicit source metadata override. Otherwise ``data.attrs['units']``
        is required. Both original and applied source labels enter the audit.
    scale, offset : float, optional
        Explicit affine transform under the existing conversion-plan contract.

    Returns
    -------
    xarray.DataArray
        Dask-backed float64 field. Coordinates and descriptive/CF relationship
        metadata survive; the unit label and conversion audit change. Range
        metadata and disk packing are cleared because their original encoded
        representation cannot be inferred safely. Custom affine conversion
        also clears standard_name for the caller to reassess. Removed metadata
        are retained in the audit. No field values are computed here.

    Notes
    -----
    Missing/Inf source values become NaN. Overflow from finite input raises
    ValueError during compute/write, not during construction of the graph.
    Scalar metadata errors and complex dtypes raise immediately.
    """
    if not isinstance(data, xr.DataArray):
        raise TypeError("Unit conversion requires an xarray.DataArray")
    if np.issubdtype(data.dtype, np.complexfloating):
        raise ValueError("Complex values are unsupported; no imaginary component is discarded")
    original_units = str(data.attrs.get("units", ""))
    applied_units = original_units if source_units is None else str(source_units)
    factor, shift, output_units = conversion_plan(
        applied_units, target_units, scale=scale, offset=offset
    )
    if data.attrs.get("units_metadata") == "temperature: difference" and shift != 0:
        raise ValueError(
            "A temperature difference cannot use an absolute-temperature offset; "
            "supply a documented explicit scale/offset for the difference"
        )
    lazy = data if data.chunks is not None else data.chunk("auto")
    result = xr.apply_ufunc(
        _apply_affine,
        lazy,
        kwargs={"factor": factor, "shift": shift},
        dask="parallelized",
        output_dtypes=[np.float64],
        keep_attrs=True,
    )
    result.attrs = dict(data.attrs)
    result.attrs["units"] = output_units
    removed = {}
    for key in ("valid_min", "valid_max", "valid_range", "actual_range"):
        if key in result.attrs:
            removed[key] = {
                "value": _audit_value(result.attrs.pop(key)),
                "reason": "Source range may refer to packed values or previous units; not rescanned",
            }
    if scale is not None and "standard_name" in result.attrs:
        removed["standard_name"] = {
            "value": _audit_value(result.attrs.pop("standard_name")),
            "reason": "Explicit affine conversion may change the physical quantity; caller must reassess",
        }
    result.attrs["climate_preprocessing_unit_transform"] = json.dumps(
        {"original_units": original_units, "source_units": applied_units,
         "target_units": output_units, "scale": factor, "offset": shift,
         "removed_attributes": removed},
        sort_keys=True, allow_nan=False,
    )
    # Never apply stale scale_factor/add_offset or a source integer dtype again
    # when this transformed floating field is serialized.
    result.encoding = {
        key: value for key, value in data.encoding.items()
        if key in {"coordinates", "grid_mapping", "bounds", "climatology", "cell_measures",
                   "formula_terms", "ancillary_variables"}
    }
    result.set_close(data.close)
    return result


def _audit_value(value):
    """Serialize small attribute values only, without consulting field data."""
    if isinstance(value, np.ndarray):
        return _audit_value(value.tolist())
    if isinstance(value, np.generic):
        return _audit_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_audit_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


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
