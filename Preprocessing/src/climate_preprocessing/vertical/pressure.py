"""Explicit fixed and CF hybrid pressure geometry, independent of interpolation.

References
----------
CF 1.12 Appendix D, atmosphere hybrid sigma pressure coordinate:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
GeoCAT hybrid-pressure/interpolation API, a design and comparison reference:
https://geocat-comp.readthedocs.io/en/latest/user_api/generated/geocat.comp.interpolation.interp_hybrid_to_pressure.html

Climate-Base improvements
-------------------------
Independently implemented CF algebra recognizes exactly a*p0+b*ps or ap+b*ps,
without guessing from variable names. Existing pressure-unit/coordinate helpers
are reused. Coefficients are bounded, eagerly inspected cell-center geometry;
surface pressure and derived fields remain Dask-backed. Labels must agree exactly,
missing surface pressure produces a missing column, and nonphysical pressure
raises only when its scheduled block executes. No interface-to-center conversion,
vertical interpolation, extrapolation, or field computation occurs at construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import json

import dask.array as da
import numpy as np
import xarray as xr

from ..coordinates import (
    _attribute, _check_pressure_semantics, _identify_pressure, _nonvertical_dimension,
    _pressure_factor, _pressure_units, _pressure_values, pressure_order,
)
from ..exceptions import CoordinateError
from ..metadata import _pairs, validate_metadata
from ..validation import require_dataset

_HYBRID = "atmosphere_hybrid_sigma_pressure_coordinate"
_LIMIT = 1_000_000
_PACKING = {"scale_factor", "add_offset", "_FillValue", "missing_value"}
_PRESSURE_ATTRS = {"standard_name": "air_pressure", "units": "Pa", "positive": "down"}


@dataclass(frozen=True)
class PressureSpec:
    """Pressure in Pa and the original vertical dimension, without reordering."""

    pressure: xr.DataArray
    vertical_dim: str
    coordinate_name: str
    surface_pressure: xr.DataArray | None
    kind: str
    formula: dict
    structural_names: set[str]


def _bounds_names(dataset):
    return {
        value for var in dataset.variables.values() for attr in ("bounds", "climatology")
        if (value := _attribute(var, attr)) is not None
    }


def _choose_coordinate(dataset, coordinate):
    bounded = _bounds_names(dataset)
    if coordinate is not None:
        if not isinstance(coordinate, str) or not coordinate or coordinate not in dataset.variables:
            raise CoordinateError("coordinate must name an existing pressure or hybrid coordinate")
        if coordinate in bounded:
            raise CoordinateError("A bounds/interface variable cannot be selected as a cell-center pressure coordinate")
        return coordinate
    formula_variables = {
        reference for var in dataset.variables.values()
        if (formula := _attribute(var, "formula_terms")) is not None
        for _, reference in _pairs(formula, "formula_terms")
    }
    candidates = []
    for name, var in dataset.variables.items():
        if name in bounded:
            continue
        standard = _attribute(var, "standard_name")
        if name in formula_variables and standard != _HYBRID:
            continue
        if standard in {"air_pressure", _HYBRID}:
            candidates.append(name)
        elif ((name in dataset.coords and var.ndim == 1)
              or str(_attribute(var, "axis") or "").upper() == "Z"):
            try:
                _pressure_units(var)
            except CoordinateError:
                continue
            candidates.append(name)
    if len(candidates) != 1:
        raise CoordinateError(f"Expected one fixed or hybrid pressure coordinate; found {candidates}; select coordinate explicitly")
    return candidates[0]


def _geometry(variable, label, *, minimum=2):
    if variable.ndim != 1 or variable.size < minimum or variable.size > _LIMIT:
        raise CoordinateError(f"{label} requires {minimum}..{_LIMIT} one-dimensional cell-center values")
    if variable.dtype.kind not in "iuf":
        raise CoordinateError(f"{label} must be real numeric geometry")
    if _PACKING.intersection(variable.attrs):
        raise CoordinateError(f"CF-decode {label} packing and missing values before pressure construction")
    values = np.asarray(variable.to_numpy(), dtype=np.float64)
    if not np.isfinite(values).all():
        raise CoordinateError(f"{label} must contain finite values")
    return values


def _monotonic(values, label):
    delta = np.diff(values)
    if not (np.all(delta > 0) or np.all(delta < 0)):
        raise CoordinateError(f"{label} must be unique and strictly monotonic; no implicit sorting is performed")


def _dimension_labels(variable, *, label):
    coords = {}
    for dimension in variable.dims:
        if dimension not in variable.coords or variable[dimension].dims != (dimension,):
            raise CoordinateError(f"{label} requires explicit one-dimensional labels for {dimension!r}")
        if dimension not in variable.indexes or not variable.get_index(dimension).is_unique:
            raise CoordinateError(f"{label} dimension {dimension!r} must have unique coordinate labels")
        coordinate = variable[dimension]
        if coordinate.dtype.kind in "iuf" and not np.isfinite(coordinate.to_numpy()).all():
            raise CoordinateError(f"{label} dimension {dimension!r} has nonfinite labels")
        if coordinate.dtype.kind == "M" and np.isnat(coordinate.to_numpy()).any():
            raise CoordinateError(f"{label} dimension {dimension!r} has missing date labels")
        # A standalone Variable prevents auxiliary indexes attached to a
        # DataArray label from leaking into later exact-alignment operations.
        copied = coordinate.variable.copy(deep=False)
        copied.attrs = {key: value for key, value in coordinate.attrs.items()
                        if key not in {"bounds", "climatology", "formula_terms"}}
        copied.encoding = {key: value for key, value in coordinate.encoding.items()
                           if key not in {"bounds", "climatology", "formula_terms"}}
        if _attribute(coordinate, "standard_name") == _HYBRID:
            # Without formula dependencies this is a generic source level
            # label, not a self-contained CF hybrid-coordinate definition.
            copied.attrs.pop("standard_name", None)
            copied.encoding.pop("standard_name", None)
            copied.attrs.setdefault("long_name", "Source hybrid coordinate label")
        coords[dimension] = copied
    return coords


def _coefficient(dataset, name, coordinate, *, dimensionless):
    var = dataset[name]
    vertical = coordinate.dims[0]
    if var.dims != (vertical,) or var.size != coordinate.size:
        raise CoordinateError(f"Hybrid coefficient {name!r} must use the same cell-center dimension; interfaces/bounds are unsupported")
    labels = _dimension_labels(var, label=f"Hybrid coefficient {name!r}")
    if not var.get_index(vertical).equals(coordinate.get_index(vertical)):
        raise CoordinateError(f"Hybrid coefficient {name!r} labels must exactly match the selected coordinate")
    values = _geometry(var, f"Hybrid coefficient {name!r}")
    if dimensionless:
        if _attribute(var, "units") != "1":
            raise CoordinateError(f"Hybrid coefficient {name!r} requires explicit dimensionless units '1'")
    else:
        factor, shift = _pressure_factor(_pressure_units(var), "Pa")
        with np.errstate(over="ignore", invalid="ignore"):
            values = values * factor + shift
        if not np.isfinite(values).all():
            raise CoordinateError(f"Hybrid coefficient {name!r} overflows pressure conversion")
    return xr.DataArray(values, dims=(vertical,), coords=labels)


def _checked_surface_block(values):
    values = np.asarray(values, dtype=np.float64)
    if np.any(np.isinf(values)) or np.any(np.isfinite(values) & (values <= 0)):
        raise CoordinateError("Surface pressure must be positive and finite, or NaN for a missing column")
    return values


def _surface_pressure(dataset, name, vertical):
    if not isinstance(name, str) or not name or name not in dataset.variables:
        raise CoordinateError("surface_pressure must name an existing pressure field")
    var = dataset[name]
    if vertical in var.dims:
        raise CoordinateError("Surface pressure must not contain the vertical dimension")
    if var.dtype.kind != "f":
        raise CoordinateError("Surface pressure must be a floating-point scalar or labelled field")
    if _PACKING.intersection(var.attrs):
        raise CoordinateError("CF-decode surface-pressure packing and missing values before interpolation")
    standard = _attribute(var, "standard_name")
    if standard not in {None, "air_pressure", "surface_air_pressure"}:
        raise CoordinateError("Surface pressure has incompatible standard_name metadata")
    if _attribute(var, "formula_terms") is not None:
        raise CoordinateError("Surface pressure must already be physical pressure, not a vertical formula")
    factor, shift = _pressure_factor(_pressure_units(var), "Pa")
    labels = _dimension_labels(var, label="Surface pressure")
    source = xr.DataArray(var.data, dims=var.dims, coords=labels, name=name)
    if source.chunks is None:
        # xarray's scalar .chunk() path may remain eager in older versions.
        source = xr.DataArray(da.from_array(source.data, chunks="auto"), dims=source.dims,
                              coords=labels, name=name)
    with xr.set_options(arithmetic_join="exact"):
        pressure = source.astype(np.float64) * factor + shift
    checked = xr.apply_ufunc(_checked_surface_block, pressure, dask="parallelized", output_dtypes=[np.float64],
                             keep_attrs=False, join="exact")
    checked.name = name
    checked.attrs = {**_PRESSURE_ATTRS, "standard_name": "surface_air_pressure"}
    checked.set_close(dataset.close)
    return checked


def _hybrid_block(intercept, coefficient, surface):
    with np.errstate(over="ignore", invalid="ignore"):
        pressure = intercept + coefficient * surface
    valid_surface = ~np.isnan(surface)
    if np.any(valid_surface & (~np.isfinite(pressure) | (pressure <= 0))):
        raise CoordinateError("Hybrid cell-center pressure must be positive and finite wherever surface pressure is valid")
    return np.where(valid_surface, pressure, np.nan).astype(np.float64)


def _structural_names(dataset, names):
    result = set(names)
    for name in tuple(result):
        for attribute in ("bounds", "climatology"):
            value = _attribute(dataset[name], attribute)
            if value is not None:
                result.add(value)
    return result


def _resolve_hybrid(dataset, name, surface_pressure):
    coordinate = dataset[name]
    values = _geometry(coordinate, "Hybrid coordinate")
    _monotonic(values, "Hybrid coordinate")
    vertical = coordinate.dims[0]
    labels = _dimension_labels(coordinate, label="Hybrid coordinate")
    if _attribute(coordinate, "units") != "1":
        raise CoordinateError("Hybrid coordinate requires explicit dimensionless units '1'")
    axis = _attribute(coordinate, "axis")
    if axis is not None and axis.upper() != "Z":
        raise CoordinateError("Hybrid coordinate axis must be Z")
    if vertical in dataset.coords and _nonvertical_dimension(dataset[vertical]):
        raise CoordinateError("Hybrid coordinate cannot use a horizontal or time dimension")
    formula = _attribute(coordinate, "formula_terms")
    if formula is None:
        raise CoordinateError("Hybrid coordinate requires explicit CF formula_terms")
    pairs = _pairs(formula, "formula_terms")
    if len({term for term, _ in pairs}) != len(pairs):
        raise CoordinateError("Hybrid formula_terms contains duplicate term labels")
    terms = dict(pairs)
    if set(terms) == {"a", "b", "p0", "ps"}:
        kind = "a_p0_b_ps"
    elif set(terms) == {"ap", "b", "ps"}:
        kind = "ap_b_ps"
    else:
        raise CoordinateError("Hybrid formula_terms must be exactly a,b,p0,ps or ap,b,ps")
    if terms.get("p0") == terms["ps"]:
        raise CoordinateError("Hybrid scalar reference p0 must be distinct from the lazily evaluated ps field")
    if set(terms.values()) & _bounds_names(dataset):
        raise CoordinateError("Hybrid formulas require cell-center coefficients, not referenced bounds/interfaces")
    if surface_pressure is not None and surface_pressure != terms["ps"]:
        raise CoordinateError("A hybrid surface_pressure override must name its exact CF ps variable")
    surface = _surface_pressure(dataset, terms["ps"], vertical)
    b = _coefficient(dataset, terms["b"], coordinate, dimensionless=True)
    if kind == "a_p0_b_ps":
        a = _coefficient(dataset, terms["a"], coordinate, dimensionless=True)
        p0 = dataset[terms["p0"]]
        if p0.ndim != 0 or p0.dtype.kind not in "iuf" or _PACKING.intersection(p0.attrs):
            raise CoordinateError("Hybrid p0 must be a decoded numeric scalar reference pressure")
        factor, shift = _pressure_factor(_pressure_units(p0), "Pa")
        reference = float(p0.to_numpy()) * factor + shift
        if not np.isfinite(reference) or reference <= 0:
            raise CoordinateError("Hybrid p0 must be positive and finite")
        with np.errstate(over="ignore", invalid="ignore"):
            intercept = a * reference
        if not np.isfinite(intercept.to_numpy()).all():
            raise CoordinateError("Hybrid a*p0 coefficient overflows physical pressure")
    else:
        intercept = _coefficient(dataset, terms["ap"], coordinate, dimensionless=False)
    # Coefficients carry only their common dimension labels; the exact join
    # precludes silent union/intersection when broadcasting with surface fields.
    pressure = xr.apply_ufunc(_hybrid_block, intercept, b, surface, dask="parallelized",
                             output_dtypes=[np.float64], keep_attrs=False, join="exact")
    pressure = pressure.transpose(vertical, *surface.dims)
    if name != vertical:
        independent = coordinate.variable.copy(deep=False)
        independent.attrs = {key: value for key, value in coordinate.attrs.items()
                             if key not in {"bounds", "climatology", "formula_terms"}}
        independent.attrs.pop("standard_name", None)
        independent.attrs.setdefault("long_name", "Source hybrid coordinate label")
        independent.encoding = {}
        labels[name] = independent
    pressure = pressure.assign_coords(labels)
    pressure.name = "air_pressure"
    pressure.attrs = dict(_PRESSURE_ATTRS)
    pressure.attrs["climate_preprocessing_pressure_source"] = json.dumps({
        "kind": kind, "coordinate": name, "terms": terms,
        "formula": "p = a*p0 + b*ps" if kind == "a_p0_b_ps" else "p = ap + b*ps",
        "output_units": "Pa",
    }, sort_keys=True)
    pressure.encoding = {}
    pressure.set_close(dataset.close)
    return PressureSpec(pressure, vertical, name, surface, "hybrid",
                        {"kind": kind, "terms": terms}, _structural_names(dataset, {name, *terms.values()}))


def resolve_pressure(dataset, *, coordinate=None, surface_pressure=None):
    """Resolve fixed or hybrid cell-center pressure without executing fields.

    Automatic recognition requires one unambiguous CF hybrid or physical-pressure
    axis; a fixed and hybrid axis together require explicit selection. Fixed levels
    retain their original increasing/decreasing order. Hybrid coefficients must
    lie on the same labelled one-dimensional cell-center axis. Referenced bounds,
    n+1 interfaces and two-endpoint arrays are never reinterpreted as centers.

    ``surface_pressure`` is an optional variable-name string. For hybrid formulas
    it may only repeat the CF ``ps`` name; fixed pressure may use an explicit
    scalar or nonvertical field. The interpolation caller additionally checks that
    its dimensions/labels match every selected field. NaN surface pressure means
    a missing column; all other invalid pressures fail on explicit computation.
    Only bounded coordinate/coefficient/reference arrays are read eagerly.
    """
    require_dataset(dataset)
    if surface_pressure is not None and not isinstance(surface_pressure, str):
        raise TypeError("surface_pressure must be a variable-name string or None")
    validate_metadata(dataset)
    name = _choose_coordinate(dataset, coordinate)
    if _attribute(dataset[name], "standard_name") == _HYBRID:
        return _resolve_hybrid(dataset, name, surface_pressure)
    name = _identify_pressure(dataset, name)
    original = dataset[name]
    values = _pressure_values(dataset, name)
    if values.size < 2:
        raise CoordinateError("Source pressure requires at least two cell-center levels")
    normalized, order = pressure_order(values, _pressure_units(original), "Pa")
    pressures = np.empty_like(normalized)
    pressures[order] = normalized
    _monotonic(pressures, "Fixed pressure coordinate")
    vertical = original.dims[0]
    labels = _dimension_labels(original, label="Fixed pressure coordinate")
    pressure = xr.DataArray(pressures, dims=(vertical,), coords=labels, name="air_pressure", attrs=_PRESSURE_ATTRS)
    pressure.set_close(dataset.close)
    surface = _surface_pressure(dataset, surface_pressure, vertical) if surface_pressure is not None else None
    structural = _structural_names(dataset, {name} | ({surface_pressure} if surface_pressure else set()))
    return PressureSpec(pressure, vertical, name, surface, "fixed", {"kind": "fixed", "terms": {}}, structural)


def hybrid_pressure(dataset, *, coordinate=None):
    """Build lazy Pa pressure from exactly one explicitly defined CF hybrid axis.

    The formulas are ``p = a*p0 + b*ps`` (a/b in units ``1``) and
    ``p = ap + b*ps`` (ap in pressure units). p0/ap/ps require explicit physical
    units accepted by the existing Pa/hPa conversion table. Coefficients are
    bounded same-dimension cell-center arrays, not interfaces or bounds. Labels
    and field chunks remain intact; a missing ps creates an all-NaN column.

    The returned DataArray is named ``air_pressure`` with units Pa, contains the
    original vertical dimension and ps dimensions, and shares the source close
    callback. Standalone labels omit Dataset-only formula/bounds links and the
    hybrid standard_name, preserving generic level labels and their units;
    source metadata is unchanged. Physical pressure checks execute lazily.
    """
    specification = resolve_pressure(dataset, coordinate=coordinate)
    if specification.kind != "hybrid":
        raise CoordinateError("hybrid_pressure requires an atmosphere_hybrid_sigma_pressure_coordinate")
    return specification.pressure


def target_pressure(target_levels, *, target_units="hPa"):
    """Return an eager canonical Pa target coordinate, preserving requested order.

    One or more positive, finite, distinct pressure levels are required. Multiple
    targets must be strictly monotonic. No sorting, inferred range, or frequency
    generation occurs. If a DataArray declares units, they must agree physically
    with the explicit Pa/hPa ``target_units`` argument. Bounds are not inferred.
    """
    if not isinstance(target_units, str) or target_units not in {"Pa", "hPa"}:
        raise ValueError("target_units must be Pa or hPa")
    if isinstance(target_levels, xr.DataArray):
        original = target_levels
        _check_pressure_semantics(original)
        declared = _attribute(original, "units")
        if declared is not None and _pressure_factor(declared, "Pa") != _pressure_factor(target_units, "Pa"):
            raise CoordinateError("Target DataArray units disagree with target_units")
    else:
        raw = np.asanyarray(target_levels)
        if raw.ndim != 1:
            raise CoordinateError("target_levels must be a one-dimensional pressure vector")
        original = xr.DataArray(raw, dims=("target",))
    values = _geometry(original, "Target pressure", minimum=1)
    normalized, order = pressure_order(values, target_units, "Pa")
    pressures = np.empty_like(normalized)
    pressures[order] = normalized
    _monotonic(pressures, "Target pressure")
    attrs = {**_PRESSURE_ATTRS, "axis": "Z"}
    return xr.DataArray(pressures, dims=("pressure",), coords={"pressure": ("pressure", pressures, attrs)},
                        name="pressure", attrs=attrs)
