"""Strict labelled pressure interpolation, without evaluating field graphs.

References
----------
CF 1.12 Appendix D, atmosphere hybrid sigma pressure coordinates:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
GeoCAT-comp v2026.04.0 delegates linear/log interpolation to MetPy:
https://github.com/NCAR/geocat-comp/blob/v2026.04.0/geocat/comp/interpolation.py
MetPy v1.7.1 public one-dimensional interpolation APIs:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/interpolate/one_dimension.py

Climate-Base improvements
-------------------------
Independent CF validation and output adaptation around the upstream numerical
kernel. Explicit fields/method/units; no extrapolation, layer-gap bridging or
implicit label alignment. Only vertical geometry is inspected eagerly. Column
checks and physical interpolation execute in Dask tasks, not during assembly.
"""

from collections.abc import Mapping
import re

import numpy as np
import xarray as xr

from ..coordinates import _pressure_factor
from ..masking import _mask_targets
from ..metadata import _attribute, _pairs, _references, _without_cell_methods, validate_metadata
from ..provenance import record_operation
from ..validation import ensure_lazy, require_dataset
from .pressure import resolve_pressure, target_pressure

_RELATIONS = {"coordinates", "ancillary_variables", "grid_mapping", "cell_measures",
              "formula_terms", "bounds", "climatology"}
_RANGES = {"actual_range", "valid_range", "valid_min", "valid_max", "minimum_value", "maximum_value"}


def validate_vertical_options(options):
    """Check the explicit preparation/JSON contract before opening inputs."""
    allowed = {"target_levels", "variables", "method", "coordinate", "target_units", "surface_pressure"}
    if not isinstance(options, Mapping) or set(options) - allowed:
        raise ValueError(f"Invalid vertical options; supported keys are {sorted(allowed)}")
    if not {"target_levels", "variables", "method"}.issubset(options):
        raise ValueError("vertical requires target_levels, variables and method")
    if not isinstance(options["method"], str) or options["method"] not in {"linear", "log"}:
        raise ValueError("Vertical method must be explicitly 'linear' or 'log'")
    for key in ("coordinate", "surface_pressure"):
        if options.get(key) is not None and (not isinstance(options[key], str) or not options[key]):
            raise ValueError(f"{key} must name an input variable")
    names = options["variables"]
    names = [names] if isinstance(names, str) else names
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names)):
        raise ValueError("vertical variables must explicitly select unique nonempty field names")
    target_pressure(options["target_levels"], target_units=options.get("target_units", "hPa"))


def _check_field_geometry(field, specification):
    vertical = specification.vertical_dim
    if vertical not in field.dims:
        raise ValueError(f"Field {field.name!r} lacks source vertical dimension {vertical!r}")
    if field.dtype.kind != "f":
        raise ValueError("Vertical interpolation requires floating-point physical fields, not integer/category data")
    for geometry in (specification.pressure, specification.surface_pressure):
        if geometry is None:
            continue
        if set(geometry.dims) - set(field.dims):
            raise ValueError("Pressure geometry dimensions must be a subset of each selected field's dimensions")
        for dimension in geometry.dims:
            if geometry.sizes[dimension] != field.sizes[dimension]:
                raise ValueError(f"Pressure/field dimension sizes differ: {dimension}")
            if dimension in geometry.indexes or dimension in field.indexes:
                if dimension not in geometry.indexes or dimension not in field.indexes:
                    raise ValueError(f"Pressure/field coordinate labels must both be explicit: {dimension}")
                if not geometry.get_index(dimension).equals(field.get_index(dimension)):
                    raise ValueError(f"Pressure/field coordinate labels do not exactly match: {dimension}")
        for name, coordinate in geometry.coords.items():
            if name in geometry.dims or vertical in coordinate.dims:
                continue
            if name not in field.coords or coordinate.dims != field[name].dims:
                raise ValueError(f"Pressure auxiliary coordinate {name!r} is absent from the field")
            if coordinate.size > 1_000_000 or not coordinate.equals(field[name]):
                raise ValueError(f"Pressure auxiliary coordinate {name!r} must exactly match within bounded size")


def _assemble_result(dataset, transformed, target, specification):
    """Keep unaffected CF meaning and discard invalid source vertical metadata."""
    vertical = specification.vertical_dim
    names = set(transformed)
    ps_name = specification.surface_pressure.name if specification.surface_pressure is not None else None
    keep_surface = {ps_name} if ps_name is not None else set()
    pending = list(keep_surface)
    while pending:
        current = pending.pop()
        for attribute in _RELATIONS:
            value = _attribute(dataset[current], attribute)
            if value is None:
                continue
            for reference in _references(attribute, value):
                if reference not in keep_surface and vertical not in dataset[reference].dims:
                    keep_surface.add(reference)
                    pending.append(reference)
    obsolete = set(specification.structural_names) - keep_surface
    retained = [name for name, var in dataset.variables.items()
                if vertical not in var.dims and name not in obsolete and name not in names]
    if "pressure" in retained:
        raise ValueError("Output pressure coordinate conflicts with an independent source variable")
    transformed = {
        name: field.drop_vars([coord for coord in field.coords
                               if coord in obsolete or vertical in field[coord].dims])
        for name, field in transformed.items()
    }
    independent = xr.Dataset(
        data_vars={name: dataset[name].variable.copy(deep=False) for name in retained if name not in dataset.coords},
        coords={name: dataset[name].variable.copy(deep=False) for name in retained if name in dataset.coords},
    )
    result = xr.merge([independent, xr.Dataset(coords={"pressure": target}), xr.Dataset(transformed)],
                      compat="equals", join="exact", combine_attrs="override")
    result.attrs = dict(dataset.attrs)
    removed = {}
    for key in list(result.attrs):
        if key.startswith("geospatial_vertical_"):
            removed.setdefault("dataset", []).append(key)
            del result.attrs[key]
    source_axes = {specification.coordinate_name, vertical, "air_pressure"}
    for name, variable in result.variables.items():
        for label, container in (("attrs", variable.attrs), ("encoding", variable.encoding)):
            if name in names:
                for key in _RANGES:
                    if key in container:
                        removed.setdefault(name, []).append(f"{label}.{key}")
                        del container[key]
                methods = container.get("cell_methods")
                if isinstance(methods, str) and any(re.search(rf"(?<!\S){re.escape(axis)}\s*:", methods) for axis in source_axes):
                    kept = _without_cell_methods(methods, source_axes)
                    removed.setdefault(name, []).append(f"{label}.cell_methods")
                    if kept:
                        container["cell_methods"] = kept
                    else:
                        del container["cell_methods"]
            for attribute in _RELATIONS:
                value = container.get(attribute)
                if value is None:
                    continue
                references = _references(attribute, value)
                if attribute == "coordinates" and name in names:
                    references = ["pressure" if ref in {specification.coordinate_name, vertical} else ref
                                  for ref in references]
                    value = " ".join(dict.fromkeys(references))
                    container[attribute] = value
                missing = [ref for ref in references if ref not in result.variables]
                if not missing:
                    continue
                if name not in names and attribute in {"formula_terms", "bounds", "climatology"}:
                    raise ValueError(f"Vertical interpolation would invalidate retained {name!r} {attribute}")
                removed.setdefault(name, []).append(f"{label}.{attribute}")
                if attribute in {"coordinates", "ancillary_variables"}:
                    kept = [ref for ref in references if ref in result.variables]
                    if kept:
                        container[attribute] = " ".join(dict.fromkeys(kept))
                    else:
                        del container[attribute]
                elif attribute == "cell_measures":
                    kept = [f"{kind}: {ref}" for kind, ref in _pairs(value, attribute) if ref in result.variables]
                    if kept:
                        container[attribute] = " ".join(kept)
                    else:
                        del container[attribute]
                else:
                    del container[attribute]
        if name in names:
            for key in _RELATIONS | {"cell_methods"}:
                if key in variable.encoding:
                    variable.attrs[key] = variable.encoding.pop(key)
                    removed.setdefault(name, []).append(f"encoding.{key} -> attrs.{key}")
            if variable.encoding:
                removed.setdefault(name, []).extend(f"encoding.{key}" for key in variable.encoding)
            variable.encoding = {}
    validate_metadata(result)
    result = ensure_lazy(result)
    result.set_close(dataset.close)
    return result, {"removed_variables": sorted(set(dataset.variables) - set(result.variables)),
                    "removed_attributes": removed}


def interpolate_pressure(
    dataset, target_levels, *, variables, method, coordinate=None,
    target_units="hPa", surface_pressure=None,
):
    """Interpolate explicit fields to positive pressure centers, without extrapolation.

    ``method`` is explicitly ``linear`` in pressure or ``log`` in log-pressure.
    Target centers use ``target_units`` (Pa/hPa), must be finite, positive,
    distinct and strictly monotonic in either order, and retain requested order.
    The output vertical dimension/coordinate is ``pressure`` in those units.
    No target bounds are guessed; interpolated point values are not layer means.

    Source is a fixed physical pressure coordinate or a CF hybrid sigma-pressure
    coordinate using exactly ``p=a*p0+b*ps`` or ``p=ap+b*ps``. Coefficients must
    describe the same centers/dimension as each selected floating-point field;
    interface/bounds coefficients are not reinterpreted as centers. Small axes
    and coefficients are inspected eagerly. Model names never replace CF evidence.

    No extrapolation is supported. Outside the source-center span returns NaN.
    Source NaN/inf values invalidate their adjacent intervals; they are never
    removed to bridge a gap. An exact target hit retains the finite original
    source value even beside a missing level. Source pressures must be strictly
    monotonic per column; sorting a nonmonotonic hybrid column is not allowed.
    Invalid lazy geometry raises only when the affected task executes.

    For fixed levels, optionally name a ``surface_pressure`` variable to exclude
    source/target pressures greater than the local surface pressure. Without it,
    terrain is unknown and only source-center support limits are applied. Hybrid
    formulas always use their CF ps term; another surface variable is rejected.
    Missing ps makes the whole column missing; nonpositive or infinite ps fails.
    A target below the lowest valid above-ground center stays missing even if it
    is physically above ground: no lower extrapolation is inferred.

    Field and ps computations remain Dask-backed. The complete source vertical
    column is rechunked into one core chunk; nonvertical field chunks are retained.
    Choose those chunks to fit memory. Optional MetPy supplies the actual kernel,
    as in GeoCAT; no MetPy import is needed for other preprocessing features.
    Output retains selected fields and independent source variables, including
    surface pressure. Source-level QC, coefficients, bounds and obsolete CF links
    are removed/audited; units and time/calendar/horizontal metadata remain.
    The result shares the source close callback. Keep input resources open until
    explicit compute/write; the caller owns cleanup after an API error.
    """
    require_dataset(dataset)
    options = dict(target_levels=target_levels, variables=variables, method=method, coordinate=coordinate,
                   target_units=target_units, surface_pressure=surface_pressure)
    validate_vertical_options(options)
    validate_metadata(dataset)
    names = _mask_targets(dataset, variables)
    specification = resolve_pressure(dataset, coordinate=coordinate, surface_pressure=surface_pressure)
    target_pa = target_pressure(target_levels, target_units=target_units)
    factor, shift = _pressure_factor("Pa", target_units)
    target = xr.DataArray(np.asarray(target_pa.data) * factor + shift, dims="pressure", name="pressure",
                          attrs={"standard_name": "air_pressure", "units": target_units, "positive": "down", "axis": "Z"})
    if "pressure" in names:
        raise ValueError("Selected field name conflicts with output pressure coordinate")
    if specification.vertical_dim != "pressure" and "pressure" in dataset.dims:
        raise ValueError("Output pressure dimension conflicts with an independent source dimension")
    for name in names:
        _check_field_geometry(dataset[name], specification)
    placeholders = {name: xr.DataArray([], dims="_vertical_metadata_probe", attrs=dict(dataset[name].attrs))
                    for name in names}
    for name in names:
        placeholders[name].encoding = dict(dataset[name].encoding)
    _assemble_result(dataset, placeholders, target, specification)
    from . import _metpy

    versions = _metpy.backend_versions()
    transformed = {}
    for name in names:
        field = _metpy.interpolate_field(
            dataset[name], specification.pressure, target_pa,
            vertical_dim=specification.vertical_dim, method=method,
            surface_pressure=specification.surface_pressure,
        )
        transformed[name] = field.assign_coords(pressure=target)
        transformed[name].encoding = dict(dataset[name].encoding)
    result, audit = _assemble_result(dataset, transformed, target, specification)
    return record_operation(result, "interpolate_pressure", {
        "variables": names, "method": method, "coordinate": specification.coordinate_name,
        "source_kind": specification.kind, "formula": specification.formula,
        "target_units": target_units, "target_levels": target.to_numpy().tolist(),
        "surface_pressure": specification.surface_pressure.name if specification.surface_pressure is not None else None,
        "extrapolate": False, "missing_value_policy": "original_adjacent_intervals_and_exact_hits",
        "vertical_rechunk": "complete source column", "backend_versions": versions, **audit,
    })
