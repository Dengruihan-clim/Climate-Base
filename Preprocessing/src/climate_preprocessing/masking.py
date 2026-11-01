"""Explicit lazy masking; no physical-range guesses or missing-value filling.

References: xarray's labelled conditional operation and exact alignment:
https://docs.xarray.dev/en/stable/generated/xarray.DataArray.where.html
https://docs.xarray.dev/en/stable/generated/xarray.align.html
https://docs.xarray.dev/en/stable/generated/xarray.apply_ufunc.html
CF 1.12 related variables and flags:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html

Climate-Base improvements: preserve valid zero and negative values, reject
complex fields, retain QC/source variables, and leave field evaluation to Dask.
Explicit user masks require exact coordinate labels, selected target variables
and a documented reason. QC/structural variables stay intact; obsolete ranges
and unsafe disk encodings are removed without reading mask or field values.
This is independently implemented using public xarray/Dask operations.
"""

from collections.abc import Sequence

import numpy as np
import xarray as xr

from .metadata import _attribute, _references, validate_metadata
from .provenance import record_operation
from .validation import ensure_lazy, require_dataset

_CF_LINKS = frozenset({"bounds", "climatology", "coordinates", "grid_mapping", "cell_measures",
                       "formula_terms", "ancillary_variables"})
_STRUCTURAL_LINKS = _CF_LINKS - {"ancillary_variables"}
_FLAG_ATTRIBUTES = frozenset({"flag_values", "flag_masks", "flag_meanings"})
_DESCRIPTIVE_RANGES = ("actual_range", "minimum_value", "maximum_value")
_VALID_RANGES = ("valid_min", "valid_max", "valid_range")


def _mask_targets(dataset, variables):
    if isinstance(variables, str):
        names = [variables]
    elif isinstance(variables, Sequence):
        names = list(variables)
    else:
        raise TypeError("variables must explicitly name a nonempty sequence of data variables")
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("variables must explicitly name a nonempty sequence of data variables")
    if len(names) != len(set(names)):
        raise ValueError("variables must not contain duplicate names")
    protected = set(dataset.coords)
    for variable in dataset.variables.values():
        for attribute in _STRUCTURAL_LINKS:
            value = _attribute(variable, attribute)
            if value is not None:
                protected.update(_references(attribute, value))
    for name in names:
        if name not in dataset.data_vars:
            raise ValueError(f"Expected a data variable target, got {name!r}")
        variable = dataset[name]
        if name in protected or "grid_mapping_name" in variable.attrs:
            raise ValueError(f"CF structural variable {name!r} cannot be masked")
        if _FLAG_ATTRIBUTES.intersection(variable.attrs) or _FLAG_ATTRIBUTES.intersection(variable.encoding):
            raise ValueError(f"QC flag variable {name!r} cannot be masked; retain flags and select physical fields")
        if variable.dtype.kind not in "iuf":
            raise ValueError(f"Mask target {name!r} must be a real numeric field, not boolean, complex or nonnumeric")
        if {"scale_factor", "add_offset", "_FillValue", "missing_value"}.intersection(variable.attrs):
            raise ValueError(f"Decode packing and fill metadata for {name!r} before applying a user mask")
    return names


def _check_mask_coordinates(variable, mask):
    extra = set(mask.dims) - set(variable.dims)
    if extra:
        raise ValueError(f"Mask has dimensions absent from target {variable.name!r}: {sorted(extra)}")
    for dimension in mask.dims:
        for label, array in (("mask", mask), ("target", variable)):
            if dimension not in array.coords or array[dimension].dims != (dimension,):
                raise ValueError(f"The {label} requires a one-dimensional label coordinate for {dimension!r}")
            if dimension not in array.indexes:
                raise ValueError(f"The {label} requires a coordinate index for {dimension!r}")
            if not array.get_index(dimension).is_unique:
                raise ValueError(f"The {label} coordinate {dimension!r} contains duplicate labels")
        if not variable.get_index(dimension).equals(mask.get_index(dimension)):
            raise ValueError(f"Mask coordinate labels for {dimension!r} do not exactly match the target")
    # Auxiliary/scalar coordinates can identify another time, level or grid
    # despite matching dimension labels. Check bounded coordinates only; the
    # boolean condition and physical field arrays are never inspected here.
    for name, coordinate in mask.coords.items():
        if name in mask.dims:
            continue
        if name not in variable.coords or coordinate.dims != variable[name].dims:
            raise ValueError(f"Mask auxiliary coordinate {name!r} is absent or incompatible with the target")
        if max(coordinate.size, variable[name].size) > 1_000_000:
            raise ValueError(f"Mask auxiliary coordinate {name!r} exceeds the bounded comparison size")
        if not coordinate.equals(variable[name]):
            raise ValueError(f"Mask auxiliary coordinate {name!r} does not exactly match the target")


def _masked_integer_block(values, condition):
    """Promote integers for NaN without silently rounding retained values."""
    values = np.asanyarray(values)
    floating = values.astype(np.float64)
    # Comparing float64 directly to int64/uint64 would coerce both sides and
    # miss losses above 2**53. Roundtrip to the original integer dtype instead.
    # Overflow on masked-out extrema is irrelevant, and must not reject them.
    with np.errstate(invalid="ignore", over="ignore"):
        restored = floating.astype(values.dtype)
    if np.any(condition & (restored != values)):
        raise ValueError("Retained integer mask values cannot be represented exactly with floating NaN")
    return np.where(condition, floating, np.nan)


def apply_mask(dataset, mask, *, variables, reason):
    """Apply an explicit labelled boolean condition without computing fields.

    ``True`` retains the original value; ``False`` produces missing (NaN).
    ``mask`` is a boolean DataArray or a data-variable name within ``dataset``.
    Its dimensions must be a subset of every explicitly selected target's
    dimensions. Shared dimension indexes must be present, unique and exactly
    equal in value and order; no reordering, interpolation, inner join or
    positional alignment is performed. Scalar boolean DataArrays broadcast
    to all selected values. Small auxiliary coordinates, when attached, must
    also match and are not introduced into the output.

    ``variables`` is a nonempty name or sequence of names, never an implicit
    selection of the whole Dataset. Coordinates, referenced CF structural
    variables, boolean conditions and flag-labelled QC variables cannot be
    targets. Numeric ancillary quantities without flag metadata can be selected
    explicitly. Integer physical fields promote to float64 to represent NaN;
    retained integers that cannot be represented exactly fail when their Dask
    block executes. Masked-out extrema do not trigger that check. Input
    packing/fill attributes must already be CF-decoded.

    A nonempty ``reason`` is required for provenance, including the origin and
    meaning of an external condition. Records identify the condition's name,
    dimensions and shape; its values are neither scanned nor hashed. Source
    units and CF links survive. Descriptive ranges and old disk encodings are
    removed and recorded; valid_* limits are retained unless source packing
    makes their physical interpretation unsafe. No range is recomputed.

    The returned Dataset retains the input Dataset close callback. An external
    mask may own an independent backend: the caller must keep it open until
    computation finishes and close it separately. This function does not take
    ownership of that backend and does not close caller resources on errors.
    """
    require_dataset(dataset)
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must document the origin and meaning of the mask")
    validate_metadata(dataset)
    names = _mask_targets(dataset, variables)
    if isinstance(mask, str):
        if mask not in dataset.data_vars:
            raise ValueError(f"Expected a mask data variable, got {mask!r}")
        if mask in names:
            raise ValueError("The mask condition cannot also be a target variable")
        condition = dataset[mask]
        identifier = {"kind": "dataset_variable", "name": mask}
    elif isinstance(mask, xr.DataArray):
        condition = mask
        if mask.name is not None and not isinstance(mask.name, str):
            raise TypeError("An external mask name must be a string or None")
        identifier = {"kind": "external_dataarray", "name": mask.name}
    else:
        raise TypeError("mask must be a boolean xarray.DataArray or a Dataset data-variable name")
    if condition.dtype.kind != "b":
        raise TypeError("mask must have boolean dtype; explicitly derive a boolean condition from QC flags")
    for name in names:
        _check_mask_coordinates(dataset[name], condition)
    identifier.update(dimensions=list(condition.dims), shape=list(condition.shape))
    # Validate auxiliary coordinates above, then keep only dimension labels on
    # the condition so xr.where cannot introduce foreign scalar/grid metadata.
    condition = condition.reset_coords(drop=True)
    if condition.chunks is None:
        condition = condition.chunk("auto")
    result = ensure_lazy(dataset)
    removed_ranges = {}
    removed_encodings = {}
    for name in names:
        variable, aligned = xr.align(result[name], condition, join="exact", copy=False)
        if variable.dtype.kind in "iu":
            masked = xr.apply_ufunc(
                _masked_integer_block, variable, aligned,
                dask="parallelized", output_dtypes=[np.float64], keep_attrs=True,
            )
        else:
            masked = variable.where(aligned)
        masked = masked.transpose(*variable.dims)
        masked.attrs = dict(variable.attrs)
        packed = "scale_factor" in variable.encoding or "add_offset" in variable.encoding
        ranges = _DESCRIPTIVE_RANGES + (_VALID_RANGES if packed else ())
        removed = []
        for key in ranges:
            if key in masked.attrs:
                masked.attrs.pop(key)
                removed.append(f"attrs.{key}")
            if key in variable.encoding:
                removed.append(f"encoding.{key}")
        if removed:
            removed_ranges[name] = removed
        removed_encodings[name] = sorted(key for key in variable.encoding if key not in _CF_LINKS)
        masked.encoding = {key: value for key, value in variable.encoding.items() if key in _CF_LINKS}
        result[name] = masked
    result.set_close(dataset.close)
    validate_metadata(result)
    return record_operation(result, "apply_mask", {
        "variables": names, "reason": reason.strip(), "mask": identifier,
        "true_means": "retain", "false_means": "missing", "alignment": "exact",
        "removed_range_attributes": removed_ranges, "removed_encoding_keys": removed_encodings,
    })


def mask_invalid(dataset, *, variables=None):
    """Replace floating NaN/Inf by NaN lazily for explicit data variables.

    By default floating data variables are selected; integer flags and string
    identifiers are preserved. Explicit integer variables remain unchanged.
    Coordinates are never masked. Encoded fill values should first be decoded
    by the reader; this function does not guess undocumented sentinel values.
    """
    require_dataset(dataset)
    if variables is None:
        names = [name for name, value in dataset.data_vars.items() if value.dtype.kind in "fc"]
    else:
        names = [variables] if isinstance(variables, str) else list(variables)
    if len(names) != len(set(names)):
        raise ValueError("variables must not contain duplicate names")
    result = ensure_lazy(dataset)
    for name in names:
        if name not in result.data_vars:
            raise ValueError(f"Expected a data variable, got {name!r}")
        variable = result[name]
        if variable.dtype.kind == "c":
            raise ValueError(f"Complex field {name!r} is unsupported")
        if variable.dtype.kind not in "iufb":
            raise ValueError(f"Variable {name!r} is not numeric")
        if variable.dtype.kind == "f":
            masked = variable.where(np.isfinite(variable))
            masked.attrs = dict(variable.attrs)
            masked.encoding = dict(variable.encoding)
            result[name] = masked
    result.set_close(dataset.close)
    return record_operation(result, "mask_invalid", {"variables": names})
