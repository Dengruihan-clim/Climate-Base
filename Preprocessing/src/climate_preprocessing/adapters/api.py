"""Explicit lazy adapters for three monthly pressure-level representations.

References
----------
Product scope and official metadata references are in ``profiles.py``.
CF 1.12 metadata links and coordinates:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
Xarray exact labelled selection and rename:
https://docs.xarray.dev/en/stable/user-guide/indexing.html

Climate-Base improvements
-------------------------
Independently implemented adapters require field-role declarations, physical
unit evidence, exact explicit version/member selectors, and conflict-free CF
renaming. Units, values, axis ordering, forecasts, and averaging windows are
never guessed or numerically transformed. Unknown provenance remains unverified.
Only bounded coordinate/metadata arrays are inspected; field graphs stay lazy.
"""

from collections.abc import Mapping
import re

import numpy as np

from ..coordinates import (
    _UNITS, _unit_key, _checked_bounds, _check_pressure_semantics,
    _pressure_bounds, _pressure_factor, _pressure_values, coordinate_values, pressure_order,
)
from ..masking import _mask_targets
from ..metadata import _attribute, rename_cf_references, validate_metadata
from ..provenance import record_operation
from ..time import validate_time
from ..units import conversion_plan, unit_key
from ..validation import ensure_lazy, require_dataset
from .profiles import ERA5_PARAMETERS, KNOWN_SELECTION_AXES, PROFILES, ROLE_UNITS

_AXES = {"time", "pressure", "latitude", "longitude"}
_STANDARD = {"time": "time", "pressure": "air_pressure", "latitude": "latitude", "longitude": "longitude"}
_AXIS_TAG = {"time": "T", "pressure": "Z", "latitude": "Y", "longitude": "X"}
_DESCRIPTIVE = {"actual_range", "minimum_value", "maximum_value"}
_SHAPE_ENCODING = {"chunksizes", "chunks", "preferred_chunks", "original_shape", "contiguous"}
_LIMIT = 1_000_000


def validate_adapter_options(options):
    """Validate explicit options without reading a dataset or opening a file."""
    allowed = {"product", "variables", "coordinates", "selectors"}
    if not isinstance(options, Mapping) or set(options) - allowed:
        raise ValueError(f"Invalid adapter options; supported keys are {sorted(allowed)}")
    if not isinstance(options.get("product"), str) or options["product"] not in PROFILES:
        raise ValueError(f"product must explicitly select one of {sorted(PROFILES)}")
    variables = options.get("variables")
    if (not isinstance(variables, Mapping) or not variables or set(variables) - set(ROLE_UNITS)
            or any(not isinstance(name, str) or not name for name in variables.values())):
        raise ValueError(f"variables must map supported canonical roles to explicit source names: {sorted(ROLE_UNITS)}")
    if len(set(variables.values())) != len(variables):
        raise ValueError("A source variable cannot be assigned to multiple physical roles")
    coordinates = options.get("coordinates")
    if coordinates is not None and (
        not isinstance(coordinates, Mapping) or set(coordinates) - _AXES
        or any(not isinstance(name, str) or not name for name in coordinates.values())
    ):
        raise ValueError("coordinates may map time, pressure, latitude and longitude to source names")
    names = {**PROFILES[options["product"]]["coordinates"], **dict(coordinates or {})}
    if len(set(names.values())) != 4:
        raise ValueError("The four source axes must have distinct names")
    selectors = options.get("selectors")
    if selectors is not None and (not isinstance(selectors, Mapping) or any(
        not isinstance(name, str) or not name or type(value) not in (str, int, float)
        or isinstance(value, float) and not np.isfinite(value)
        for name, value in selectors.items()
    )):
        raise ValueError("selectors require names and exact JSON string/integer/finite-float scalar labels")
    if set(selectors or {}) & set(names.values()):
        raise ValueError("Product selectors may not collapse the four physical axes")


def adapter_source_time(options):
    """Return the raw time name needed before per-file adaptation."""
    validate_adapter_options(options)
    return (options.get("coordinates") or {}).get("time", PROFILES[options["product"]]["coordinates"]["time"])


def _label_equal(actual, requested):
    if isinstance(actual, np.generic):
        actual = actual.item()
    if isinstance(actual, (bool, np.bool_)):
        return False
    # Avoid equating a string ID such as '0001' with an integer member label.
    if isinstance(requested, str):
        return isinstance(actual, str) and actual == requested
    return isinstance(actual, (int, float)) and np.isfinite(actual) and actual == requested


def _select(dataset, selectors, axis_names, source_fields):
    required = set(KNOWN_SELECTION_AXES) & (set(dataset.variables) | set(dataset.dims))
    for name in source_fields:
        required.update(set(dataset[name].dims) - set(axis_names))
    if required - set(selectors):
        raise ValueError(f"Explicit selectors are required for version/member/extra axes: {sorted(required - set(selectors))}")
    result = dataset.copy(deep=False)
    collapsed = set()
    observed = {}
    for name, requested in selectors.items():
        if name not in result.coords:
            raise ValueError(f"Selector {name!r} requires an explicit label coordinate")
        coordinate = result[name]
        if coordinate.ndim == 0:
            actual = np.asarray(coordinate.data).item()
            if not _label_equal(actual, requested):
                raise ValueError(f"Scalar selector {name!r} does not exactly match its label")
            observed[name] = {"label": requested, "source_size": 1, "source_scalar": True}
            continue
        if coordinate.dims != (name,) or coordinate.size > _LIMIT or name not in result.indexes:
            raise ValueError(f"Selector {name!r} requires a bounded one-dimensional self-labelled index")
        index = result.get_index(name)
        if not index.is_unique:
            raise ValueError(f"Selector {name!r} has duplicate labels")
        matches = [index for index, value in enumerate(np.asarray(coordinate.data)) if _label_equal(value, requested)]
        if len(matches) != 1:
            raise ValueError(f"Selector {name!r} requires exactly one exact matching label")
        observed[name] = {"label": requested, "source_size": coordinate.size, "source_scalar": False}
        # Locate the exact label first. isel then uses that verified position;
        # requested integers are never interpreted as positional indexes.
        result = result.isel({name: matches[0]}, drop=False)
        collapsed.add(name)
    removed = {}
    for name, original in dataset.variables.items():
        if not collapsed.intersection(original.dims):
            continue
        variable = result[name]
        for label, container in (("attrs", variable.attrs), ("encoding", variable.encoding)):
            for key in _DESCRIPTIVE:
                if key in container:
                    container.pop(key)
                    removed.setdefault(name, []).append(f"{label}.{key}")
        for key in _SHAPE_ENCODING:
            if key in variable.encoding:
                variable.encoding.pop(key)
                removed.setdefault(name, []).append(f"encoding.{key}")
    return result, observed, removed


def validate_product_time(dataset, *, time="time"):
    """Validate decoded monotonic time and at most one record per year/month.

    Called both per source and after multi-file combination. Labels are not
    inferred averaging windows, and intervening missing months remain allowed.
    """
    validate_time(dataset, time=time)
    dates = dataset[time]
    if dates.size > _LIMIT:
        raise ValueError("Monthly time coordinate exceeds bounded validation size")
    months = list(zip(dates.dt.year.to_numpy().tolist(), dates.dt.month.to_numpy().tolist()))
    if len(months) != len(set(months)):
        raise ValueError("Monthly product profiles require at most one record per calendar year/month")


def _axes(dataset, names):
    for role, name in names.items():
        if name not in dataset.coords or dataset[name].dims != (name,) or dataset[name].size == 0:
            raise ValueError(f"{role} requires a nonempty one-dimensional self-dimension coordinate {name!r}")
        if dataset[name].size > _LIMIT:
            raise ValueError(f"{role} coordinate exceeds bounded validation size")
        standard = _attribute(dataset[name], "standard_name")
        if standard is not None and standard != _STANDARD[role]:
            raise ValueError(f"Conflicting {role} standard_name {standard!r}")
        axis = _attribute(dataset[name], "axis")
        if axis is not None and axis.upper() != _AXIS_TAG[role]:
            raise ValueError(f"Conflicting {role} axis metadata")
    validate_product_time(dataset, time=names["time"])
    for role in ("latitude", "longitude"):
        name = names[role]
        if _unit_key(_attribute(dataset[name], "units")) not in _UNITS[role]:
            raise ValueError(f"{role} needs explicit physical angular units; product names are not unit evidence")
        values = coordinate_values(dataset, name)
        if role == "latitude":
            if np.any(np.abs(values) > 90) or np.unique(values).size != values.size:
                raise ValueError("Latitude centers must be distinct and within [-90, 90]")
        elif np.any(np.diff(np.sort(values % 360)) <= 1e-9):
            raise ValueError("Longitude contains duplicate cyclic centers")
        bounds = _checked_bounds(dataset, name, values, role)
        if bounds is not None:
            bound_units = _attribute(dataset[bounds[0]], "units")
            if bound_units is not None and _unit_key(bound_units) not in _UNITS[role]:
                raise ValueError(f"{role} bounds have incompatible units")
    p = names["pressure"]
    _check_pressure_semantics(dataset[p])
    units = _attribute(dataset[p], "units")
    values = _pressure_values(dataset, p)
    pressure_order(values, units, "Pa")
    if values.size > 1 and not (np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)):
        raise ValueError("Fixed pressure levels must be strictly monotonic in their original order")
    factor, shift = _pressure_factor(units, "Pa")
    _pressure_bounds(dataset, p, units, "Pa", values * factor + shift)


def _units_and_semantics(variable, role, time_name):
    units = _attribute(variable, "units")
    if units is None or not units.strip():
        raise ValueError(f"Mapped {role} requires explicit physical units")
    key = unit_key(units)
    if role == "geopotential":
        if key not in {"m2s-2", "m2/s2", "m2sec-2", "m2/sec2"}:
            raise ValueError("geopotential requires explicit squared-length/squared-time units, not height")
    elif role == "geopotential_height" and key in {"gpm", "geopotentialmeter", "geopotentialmeters"}:
        pass
    else:
        try:
            conversion_plan(units, ROLE_UNITS[role])
        except ValueError as error:
            raise ValueError(f"Units {units!r} are not a supported physical dimension for {role}") from error
    standard = _attribute(variable, "standard_name")
    if standard is not None and standard != role:
        raise ValueError(f"Existing standard_name {standard!r} conflicts with declared role {role!r}")
    if role == "air_temperature" and _attribute(variable, "units_metadata") == "temperature: difference":
        raise ValueError("A temperature difference is not an absolute air_temperature field")
    methods = _attribute(variable, "cell_methods")
    if methods:
        searchable = re.sub(r"\([^)]*\)", "", methods)
        for axes, method in re.findall(r"(?<!\S)((?:[^\s:()]+\s*:\s*)+)([^\s:()]+)", searchable):
            if {axis.strip() for axis in axes.split(":")[:-1]} & {"time", time_name} and method != "mean":
                raise ValueError("A monthly mean profile conflicts with existing non-mean time cell_methods")


def _metadata_text(value, label):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return value.strip()
    if type(value) is int or type(value) is float and np.isfinite(value) and value.is_integer():
        return str(int(value))
    raise ValueError(f"Product metadata {label!r} must have a scalar documented representation")


def _evidence(dataset, product, variables):
    evidence, unverified = [], []
    scopes = [("dataset", dataset.attrs)] + [(name, dataset[name].attrs) for name in variables.values()]
    for attribute, allowed in PROFILES[product]["identity"].items():
        found = False
        for scope, attrs in scopes:
            if attribute in attrs:
                found = True
                value = _metadata_text(attrs[attribute], attribute)
                checked = value
                if attribute == "identifier_product_doi":
                    checked = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I).casefold()
                    allowed = {item.casefold() for item in allowed}
                if checked not in allowed:
                    raise ValueError(f"Conflicting {product} identity/statistic metadata {scope}.{attribute}={value!r}")
                evidence.append({"scope": scope, "attribute": attribute, "value": value})
        if not found:
            unverified.append(attribute)
    for role, name in variables.items():
        attrs = dataset[name].attrs
        if product == "era5-monthly-pressure-levels":
            parameter, short = ERA5_PARAMETERS[role]
            expected = {"ecmwf_parameter": str(parameter), "GRIB_paramId": str(parameter),
                        "ecmwf_local_table": "128", "GRIB_table2Version": "128",
                        "short_name": short, "GRIB_shortName": short}
            for attribute, required in expected.items():
                if attribute in attrs:
                    value = _metadata_text(attrs[attribute], attribute)
                    if value != required:
                        raise ValueError(f"ERA5 parameter metadata {name}.{attribute} conflicts with declared {role}")
                    evidence.append({"scope": name, "attribute": attribute, "value": value})
                else:
                    unverified.append(f"{name}.{attribute}")
        if role in {"eastward_wind", "northward_wind"}:
            for attribute in ("resolution_and_component_flags", "GRIB_uvRelativeToGrid"):
                if attribute not in attrs:
                    unverified.append(f"{name}.{attribute}")
                    continue
                value = _metadata_text(attrs[attribute], attribute)
                if attribute == "GRIB_uvRelativeToGrid":
                    valid = value == "0"
                elif value.isdigit():
                    integer = int(value)
                    valid = 0 <= integer <= 255 and not (integer & 8)
                else:
                    valid = " ".join(value.split()).lower() == (
                        "resolved u- and v-components of vector quantities relative to easterly and northerly directions"
                    )
                if not valid:
                    raise ValueError("Grid-relative or unknown wind-component flags cannot be declared eastward/northward")
                evidence.append({"scope": name, "attribute": attribute, "value": value})
    return {"status": "caller_declared_profile_with_partial_metadata_checks",
            "verified_metadata": evidence, "unverified_metadata": sorted(unverified),
            "provider_authenticated": False, "averaging_window_inferred": False}


def adapt_product(dataset, *, product, variables, coordinates=None, selectors=None):
    """Select explicit labels and rename a declared monthly pressure product.

    ``variables`` maps canonical physical roles to source field names. Four axes
    use profile defaults or explicit ``coordinates`` mappings. Their units, CF
    meaning, distinct geometry, fixed-pressure monotonicity and decoded monthly
    record uniqueness are checked. No coordinate sorting or unit conversion is
    performed. Known version/member axes and all other extra dimensions of the
    mapped fields require exact scalar ``selectors``, including singleton axes
    and known scalar coordinates. Selected coordinates remain as scalars.

    Only mapped fields are claimed as adapted. Other fields remain unchanged
    except for shared explicit selection and CF renaming. Missing identity or
    averaging evidence is recorded as unverified; names are never certification.
    Conflicting available evidence fails. Existing non-CF/ad-hoc standard_names
    require explicit scientific resolution; no undocumented aliases are guessed.
    Result resources share the caller's source lifetime; errors do not close it.
    """
    require_dataset(dataset)
    options = dict(product=product, variables=variables, coordinates=coordinates, selectors=selectors)
    validate_adapter_options(options)
    validate_metadata(dataset)
    variables = dict(variables)
    names = {**PROFILES[product]["coordinates"], **dict(coordinates or {})}
    selectors = dict(selectors or {})
    _mask_targets(dataset, list(variables.values()))
    selected, selections, removed = _select(dataset, selectors, names.values(), variables.values())
    _axes(selected, names)
    for role, name in variables.items():
        if set(selected[name].dims) != set(names.values()):
            raise ValueError(f"Mapped field {name!r} must have exactly the four physical axes after explicit selection")
        if (product == "merra2-m2imnpasm-5.12.4" and role == "geopotential_height"
                and _attribute(selected[name], "standard_name") != "geopotential_height"):
            raise ValueError(
                "MERRA-2 ASM layer heights are not automatically geopotential height; "
                "this role requires an existing explicit standard_name='geopotential_height'"
            )
        _units_and_semantics(selected[name], role, names["time"])
    evidence = _evidence(selected, product, variables)
    mapping = {source: role for role, source in {**names, **variables}.items()}
    if len(mapping) != len(names) + len(variables):
        raise ValueError("Coordinate and field mappings must use distinct source variables")
    mapping = {source: target for source, target in mapping.items() if source != target}
    for source, target in mapping.items():
        if target in selected.variables or target in selected.dims:
            raise ValueError(f"Cannot rename {source!r}: target {target!r} already exists")
    result = selected.rename(mapping)
    result = rename_cf_references(result, mapping)
    added, migrated = {}, {}
    for role, source in {**names, **variables}.items():
        target = mapping.get(source, source)
        variable = result[target]
        standard = _STANDARD.get(role, role)
        if _attribute(variable, "standard_name") is None:
            variable.attrs["standard_name"] = standard
            added[target] = standard
        units = _attribute(variable, "units")
        if role != "time" and units is not None and "units" not in variable.attrs:
            variable.attrs["units"] = units
            variable.encoding.pop("units", None)
            migrated[target] = "encoding.units -> attrs.units (unchanged unit label)"
    result = ensure_lazy(result)
    result.set_close(dataset.close)
    validate_metadata(result)
    return record_operation(result, "adapt_product", {
        "product": product, "variables": variables, "coordinates": names,
        "selectors": selections, "rename": mapping, "identity": evidence,
        "added_standard_names": added, "units_metadata_migrations": migrated,
        "removed_selection_metadata": removed,
        "values_transformed": False, "units_converted": False,
        "scope": "explicitly mapped fields; no provider authentication or statistical conversion",
    })
