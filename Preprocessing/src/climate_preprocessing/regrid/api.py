"""Explicit scalar horizontal remapping with lazy xarray fields.

References
----------
xESMF 0.9.2 public Regridder API and ESMF regridding methods:
https://xesmf.readthedocs.io/en/stable/user_api.html
https://earthsystemmodeling.org/regrid/
CF 1.12 variable relationships and cell measures:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html

Climate-Base improvements
-------------------------
Independent adapter, not a new interpolation kernel: explicit field selection,
validated geometry/masks, immutable content-addressed weights, and audited CF
metadata cleanup. Geometry and weights are eager; meteorological fields stay
Dask-backed. No default extrapolation, dynamic missing-value renormalization,
vector rotation, unit conversion, or statistical aggregation is performed.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import xarray as xr

from ..masking import _mask_targets
from ..metadata import _references, _without_cell_methods, validate_metadata
from ..provenance import record_operation
from ..validation import ensure_lazy, require_dataset
from .grids import validate_grid
from .weights import build_or_load_weights, validate_cached_weights, weight_key, weight_metadata

METHODS = frozenset({"bilinear", "nearest_s2d", "nearest_d2s", "conservative", "conservative_normed"})
_HORIZONTAL = {"latitude", "longitude"}
_RELATIONS = {"coordinates", "ancillary_variables", "grid_mapping", "cell_measures",
              "formula_terms", "bounds", "climatology"}
_RANGES = {"actual_range", "valid_range", "valid_min", "valid_max", "minimum_value", "maximum_value"}


def _validate_cache_path(value):
    if value is not None and (
        not isinstance(value, (str, Path)) or not str(value).strip() or "://" in str(value)
    ):
        raise ValueError("weights_dir must be a nonempty local directory path, not an object-store URL")


def _without_horizontal_methods(value):
    return _without_cell_methods(value, {"latitude", "longitude", "area"})


def validate_regrid_options(options):
    """Validate the non-executable preparation/configuration remapping schema."""
    allowed = {"target_grid", "variables", "method", "periodic", "weights_dir", "quantity"}
    if not isinstance(options, Mapping) or set(options) - allowed:
        raise ValueError(f"Invalid regrid options; supported keys are {sorted(allowed)}")
    if not {"target_grid", "variables", "method"}.issubset(options):
        raise ValueError("regrid requires target_grid, variables and method")
    target = options["target_grid"]
    if isinstance(target, Mapping):
        grid_keys = {"latitude", "longitude", "latitude_bounds", "longitude_bounds"}
        if set(target) - grid_keys or not {"latitude", "longitude"}.issubset(target):
            raise ValueError("target_grid requires latitude/longitude and optional explicit axis bounds")
    elif not isinstance(target, xr.Dataset):
        raise ValueError("target_grid must be a Dataset or coordinate mapping")
    if not isinstance(options["method"], str) or options["method"] not in METHODS:
        raise ValueError(f"Unknown regrid method; choose from {sorted(METHODS)}")
    if type(options.get("periodic", False)) is not bool:
        raise ValueError("periodic must be boolean")
    if options.get("quantity", "intensive") != "intensive":
        raise ValueError("Only intensive scalar regridding is supported")
    _validate_cache_path(options.get("weights_dir"))
    names = options["variables"]
    if isinstance(names, str):
        names = [names]
    if (not isinstance(names, (list, tuple)) or not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names)):
        raise ValueError("regrid variables must explicitly select unique field names")


def _assemble_result(dataset, transformed, target):
    """Rebuild CF links on the new grid without ever reading field values."""
    # decode_coords='all' can attach an obsolete scalar CRS to each DataArray;
    # excluding it from the independent Dataset alone would reintroduce it here.
    transformed = {
        name: field.drop_vars([coord for coord, value in field.coords.items()
                               if "grid_mapping_name" in value.attrs])
        for name, field in transformed.items()
    }
    names = set(transformed)
    # Retain only grid-independent source variables and coordinates. All source
    # spatial auxiliaries, QC flags, cell areas and unselected fields are dropped.
    retained = [name for name, var in dataset.variables.items()
                if not _HORIZONTAL.intersection(var.dims)
                and "grid_mapping_name" not in var.attrs and name not in names]
    collision = set(retained) & set(target.variables)
    # A vertex dimension is often shared with time/pressure bounds. It is safe
    # only when its labels are identical; merge below uses join='exact'.
    for name in collision:
        if not dataset[name].identical(target[name]):
            raise ValueError(f"Target grid conflicts with retained source variable {name!r}")
    independent = xr.Dataset(
        data_vars={name: dataset[name].variable.copy(deep=False) for name in retained if name not in dataset.coords},
        coords={name: dataset[name].variable.copy(deep=False) for name in retained if name in dataset.coords},
    )
    result = xr.merge([independent, target, xr.Dataset(transformed)],
                      compat="equals", join="exact", combine_attrs="override")
    result.attrs = dict(dataset.attrs)
    removed = {}
    # Remove geographic coverage claims of the input domain, without inventing
    # a new ACDD range or a new area/CRS from center coordinates.
    for key in list(result.attrs):
        if key.startswith("geospatial_"):
            removed.setdefault("dataset", []).append(key)
            del result.attrs[key]
    for name, var in result.variables.items():
        for container_name, container in (("attrs", var.attrs), ("encoding", var.encoding)):
            if name in names:
                for key in list(container):
                    if key in _RANGES or key in {"grid_mapping", "cell_measures"}:
                        removed.setdefault(name, []).append(f"{container_name}.{key}")
                        del container[key]
                # Keep time/vertical cell-method semantics. A source horizontal
                # aggregation cannot describe the new interpolation footprint.
                cell_methods = container.get("cell_methods", "")
                if isinstance(cell_methods, str) and re.search(
                    r"(?:^|\s)(?:latitude|longitude|area)\s*:", cell_methods
                ):
                    removed.setdefault(name, []).append(f"{container_name}.cell_methods")
                    retained_methods = _without_horizontal_methods(cell_methods)
                    if retained_methods:
                        container["cell_methods"] = retained_methods
                    else:
                        del container["cell_methods"]
            for attribute in _RELATIONS:
                value = container.get(attribute)
                if value is None:
                    continue
                references = _references(attribute, value)
                missing = [ref for ref in references if ref not in result.variables]
                if not missing:
                    continue
                if attribute in {"formula_terms", "bounds", "climatology"}:
                    raise ValueError(f"Regridding would invalidate {name!r} {attribute}: {missing}")
                removed.setdefault(name, []).append(f"{container_name}.{attribute}")
                if attribute in {"coordinates", "ancillary_variables"}:
                    kept = [ref for ref in references if ref in result.variables]
                    if kept:
                        container[attribute] = " ".join(kept)
                    else:
                        del container[attribute]
                else:
                    del container[attribute]
        if name in names:
            # CF decoding may move scientific references into encoding. Keep
            # their validated, cleaned meaning in attrs before dropping storage
            # dtype/chunks/compression/packing, and record that migration.
            for key in _RELATIONS | {"cell_methods"}:
                if key in var.encoding:
                    var.attrs[key] = var.encoding[key]
                    removed.setdefault(name, []).append(f"encoding.{key} -> attrs.{key}")
                    del var.encoding[key]
            # All old disk encodings are invalid after numerical resampling;
            # date/vertical auxiliary encodings remain on independent variables.
            if var.encoding:
                removed.setdefault(name, []).extend(f"encoding.{key}" for key in var.encoding)
            var.encoding = {}
    validate_metadata(result)
    result = ensure_lazy(result)
    result.set_close(dataset.close)
    return result, {"removed_variables": sorted(set(dataset.variables) - set(result.variables)),
                    "removed_attributes": removed}


def regrid_dataset(
    dataset, target_grid, *, variables, method, periodic=False,
    source_mask=None, target_mask=None, weights_dir=None, quantity="intensive",
):
    """Regrid explicit scalar fields on ascending rectilinear geographic grids.

    Source and target must have canonical ``latitude`` / ``longitude`` dimension
    coordinates with degree units. Use ``normalize_coordinates`` first where
    necessary and ``rectilinear_grid`` to construct an explicit target. Descending,
    duplicate, curvilinear and projected grids are rejected. Conservative methods
    require contiguous CF bounds on both axes; bounds are never guessed.

    Methods are bilinear, nearest_s2d (one source value per destination),
    nearest_d2s (sums sources choosing each destination), conservative (normalizes
    overlap by the full destination area), and conservative_normed (normalizes by
    covered destination area). Only the full-area conservative method preserves
    the area integral on matching coverage/geometry. Inputs are scalar intensive
    floating-point values; integer/category fields are refused (convert physical
    values explicitly after checking precision). Cell totals/extensive quantities require a separately validated area
    conversion. nearest_d2s is an explicitly requested scatter-sum, so constants
    and intensive magnitudes are not generally preserved by that method.

    ``periodic=True`` is only for nonconservative global source grids with explicit
    longitude bounds proving a 360-degree span; conservative geometry handles
    closure itself and rejects this flag. There is no extrapolation option.
    Unmapped or masked target cells are NaN (nearest methods have their native
    global nearest-point domain). Source NaNs propagate through nonzero weights;
    ``skipna`` renormalization is deliberately unavailable. Static masks must be
    eager labelled boolean DataArrays with exact geometry labels: True is valid.
    Weight geometry/mask inspection and ESMF weight generation are eager, never
    the source field graph. Nonhorizontal chunks are retained; horizontal output
    chunks are bounded by the source chunk and target axis sizes. This is not a
    whole-graph memory bound; choose time/level chunks for the available memory.

    ``weights_dir`` enables a local content-addressed cache. Coordinates, bounds,
    masks, method, periodicity and backend versions all contribute to its key.
    Corrupt/incomplete caches fail instead of being silently reused or replaced.
    Without it no weights file is written. Dataset provenance stores the digest,
    not private paths. Outputs contain selected fields and independent source
    variables plus target grid/bounds. Obsolete spatial fields, QC/area/CRS links
    and ranges are removed and audited, never interpolated implicitly. Unsupported
    hybrid formula dependencies fail. Source units and time/calendar labels remain.
    The returned Dataset shares the caller's source close callback; keep the
    source open until explicit compute/write, and close it yourself on errors.
    """
    require_dataset(dataset)
    require_dataset(target_grid)
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(f"method must explicitly select one of {sorted(METHODS)}")
    if type(periodic) is not bool:
        raise ValueError("periodic must be boolean")
    if quantity != "intensive":
        raise ValueError("Only scalar intensive fields are supported; extensive cell totals require area conversion")
    _validate_cache_path(weights_dir)
    conservative = method in {"conservative", "conservative_normed"}
    if conservative and periodic:
        raise ValueError("Conservative geometry handles closure; periodic=True is not used by ESMF")
    validate_metadata(dataset)
    validate_metadata(target_grid)
    names = _mask_targets(dataset, variables)
    for name in names:
        var = dataset[name]
        if var.dtype.kind != "f":
            raise ValueError("Regridding requires floating-point scalar physical fields; integer/category fields are unsupported")
        if not _HORIZONTAL.issubset(var.dims):
            raise ValueError(f"Regrid field {name!r} must have both latitude and longitude dimensions")
        if var.attrs.get("standard_name", "").startswith(("grid_eastward_", "grid_northward_")):
            raise ValueError("Grid-relative vector components need a separately validated rotation")
    source = validate_grid(dataset, mask=source_mask, require_bounds=conservative, periodic=periodic)
    target = validate_grid(target_grid, mask=target_mask, require_bounds=conservative)
    # Reconstruct only validated geometry, rather than admitting arbitrary target
    # data variables into the transformed output.
    from .grids import rectilinear_grid
    output_grid = rectilinear_grid(target.latitude, target.longitude,
                                  latitude_bounds=target.latitude_bounds,
                                  longitude_bounds=target.longitude_bounds)
    if set(names) & set(output_grid.variables):
        raise ValueError("Selected field name conflicts with target geometry")
    # Fail CF closure/collision checks before loading a heavy optional backend.
    placeholders = {name: xr.DataArray(
        [], dims=("_regrid_metadata_probe",), attrs=dict(dataset[name].attrs), name=name
    ) for name in names}
    for name in names:
        placeholders[name].encoding = dict(dataset[name].encoding)
    _assemble_result(dataset, placeholders, output_grid)

    from . import _xesmf
    versions = _xesmf.backend_versions()
    parameters = dict(method=method, periodic=periodic, backend_versions=versions)
    metadata = weight_metadata(source, target, **parameters)
    key = weight_key(source, target, **parameters)
    reused = False
    if weights_dir is None:
        regridder = _xesmf.build_regridder(source.grid, target.grid, method=method, periodic=periodic)
    else:
        created = []

        def builder(path):
            operator = _xesmf.build_regridder(source.grid, target.grid, method=method, periodic=periodic)
            _xesmf.save_weights(operator, path)
            created.append(operator)

        path, reused = build_or_load_weights(weights_dir, key, metadata, builder)
        regridder = created[0] if created else _xesmf.build_regridder(
            source.grid, target.grid, method=method, periodic=periodic, filename=path, reuse_weights=True,
        )
        validate_cached_weights(weights_dir, key, metadata)
    transformed = {}
    for name in names:
        transformed[name] = _xesmf.apply_regridder(
            regridder, dataset[name], output_latitude=target.latitude, output_longitude=target.longitude,
        )
        transformed[name].encoding = dict(dataset[name].encoding)
    result, cleanup = _assemble_result(dataset, transformed, output_grid)
    return record_operation(result, "regrid_dataset", {
        "variables": names, "method": method, "periodic": periodic, "quantity": quantity,
        "skipna": False, "unmapped_to_nan": True, "weight_key": key,
        "weights_cached": weights_dir is not None, "weights_reused": reused,
        "backend_versions": versions, "source_mask": source_mask is not None,
        "target_mask": target_mask is not None, **cleanup,
    })
