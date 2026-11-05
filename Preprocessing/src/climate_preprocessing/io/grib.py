"""Strict local GRIB reads through optional cfgrib/ecCodes, with lazy fields.

References
----------
ECMWF cfgrib public xarray backend and hypercube/filter contract (0.9.15.1):
https://github.com/ecmwf/cfgrib/tree/0.9.15.1
https://github.com/ecmwf/cfgrib/blob/0.9.15.1/cfgrib/xarray_plugin.py
ecCodes Python message/header APIs:
https://confluence.ecmwf.int/display/ECC/GRIB+examples
GRIB2 section framing (single-field check only, no value decoder):
https://codes.ecmwf.int/grib/format/grib2/

Climate-Base improvements
-------------------------
Independent adapter, not a copied decoder. Header-only preflight rejects mixed
representations, duplicate forecast slots and incomplete Cartesian layouts
instead of silently dropping messages or inventing missing records. Persistent
indexes and geographic caches are disabled; inputs remain read-only. Forecast
reference time, step and valid_time remain distinct; statistical step windows
are retained in source metadata and never converted to instantaneous values.
Only headers/geographic coordinates are inspected during opening. Meteorological
arrays are Dask-backed; their original source size/mtime is checked on execution.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
from importlib import metadata
import math
from pathlib import Path

import numpy as np
import xarray as xr

from ..metadata import validate_metadata
from ..provenance import input_record, record_operation
from ..time import validate_time
from .netcdf import _checked_chunks, _check_input_stamp, _guard_source_block

_GROUP_KEYS = ("edition", "centre", "subCentre", "dataType", "typeOfLevel", "stepType", "gridType",
               "md5GridSection", "productDefinitionTemplateNumber", "stepUnits")
_FIELD_KEYS = ("paramId", "shortName", "level", "dataDate", "dataTime", "startStep", "endStep", "number")
_FILTER_KEYS = frozenset((*_GROUP_KEYS, *_FIELD_KEYS, "stepRange"))
_REQUIRED_KEYS = {"edition", "typeOfLevel", "stepType", "gridType", "md5GridSection", "paramId",
                  "level", "dataDate", "dataTime", "startStep", "endStep", "stepUnits"}


def _backends():
    try:
        eccodes = importlib.import_module("eccodes")
        cfgrib = importlib.import_module("cfgrib")
        native = str(eccodes.codes_get_api_version())
    except (ImportError, RuntimeError, OSError) as error:
        raise ImportError(
            "GRIB support requires optional cfgrib and a working ecCodes library. "
            "Install climate-preprocessing[grib] in the selected environment; "
            "if ecCodes cannot load, install its native library for your platform."
        ) from error
    return eccodes, cfgrib, native


def _filters(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("filter_by_keys must be a mapping of explicit GRIB header keys to scalar values")
    if set(value) - _FILTER_KEYS:
        raise ValueError(f"Unsupported GRIB filter keys: {sorted(set(value) - _FILTER_KEYS)}")
    result = {}
    for key, item in value.items():
        if isinstance(item, bool) or not isinstance(item, (str, int, float)):
            raise TypeError("GRIB filters require string, integer or finite float scalar values")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("GRIB filter values must be finite")
        result[key] = item
    return result


def _source(path):
    if isinstance(path, str) and "://" in path:
        raise ValueError("GRIB input must be a local file, not a URL")
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Expected one local GRIB file")
    info = source.stat()
    return source, (info.st_size, info.st_mtime_ns)


def _backend_filters(filters):
    # CfMessage exposes computed endStep/stepUnits in HOURS. Public filters and
    # inspection use original ecCodes headers, so typed keys deliberately bypass
    # those computed aliases. Fractional vertical levels likewise need float.
    aliases = {"stepUnits": "stepUnits:int", "startStep": "startStep:int", "endStep": "endStep:int", "level": "level:float"}
    return {aliases.get(key, key): value for key, value in filters.items()}


def _header(eccodes, message, key):
    if not eccodes.codes_is_defined(message, key) or eccodes.codes_is_missing(message, key):
        return None
    if key == "level":
        value = eccodes.codes_get(message, key, float)
    elif key in {"startStep", "endStep", "stepUnits"}:
        # Recent ecCodes may return '30m' for the untyped start/end aliases.
        # Integer values plus the explicitly retained units avoid ambiguity.
        value = eccodes.codes_get(message, key, int)
    else:
        value = eccodes.codes_get(message, key)
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        raise ValueError(f"GRIB header {key!r} must be a scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite GRIB header {key!r}")
    return value


def _single_field_grib2(stream, offset, length):
    """Inspect GRIB2 section framing, without reading the packed data payload.

    ecCodes' global multi-field setting differs from cfgrib's iteration mode.
    Count data sections directly so a hidden second field cannot evade header
    preflight. This validates framing only; ecCodes remains the field decoder.
    """
    stream.seek(offset)
    indicator = stream.read(16)
    if len(indicator) != 16 or indicator[:4] != b"GRIB" or indicator[7] != 2:
        raise ValueError("Invalid GRIB2 indicator section")
    physical_length = int.from_bytes(indicator[8:16], "big")
    position, end = offset + 16, offset + physical_length - 4
    fields = 0
    while position < end:
        stream.seek(position)
        header = stream.read(5)
        if len(header) != 5:
            raise ValueError("Truncated GRIB2 section header")
        size, number = int.from_bytes(header[:4], "big"), header[4]
        if size < 5 or number not in range(1, 8) or position + size > end:
            raise ValueError("Invalid GRIB2 section framing")
        fields += number == 7
        position += size
    stream.seek(end)
    if position != end or stream.read(4) != b"7777":
        raise ValueError("GRIB2 message lacks a complete end section")
    if fields != 1:
        raise ValueError(
            "Multi-field GRIB2 messages are not supported; split them into single-field "
            "messages explicitly before opening, so no field can be silently discarded"
        )
    if physical_length != length:
        raise ValueError("GRIB2 physical message length differs from the decoded header")


def _scan(source, eccodes, filters):
    """Request header keys, never values/data/bitmap arrays.

    ecCodes may read complete compressed messages even with headers_only=True;
    this is a no-field-decode contract, not a header-byte-only I/O guarantee.
    """
    records = []
    end = 0
    keys = tuple(dict.fromkeys((*_GROUP_KEYS, *_FIELD_KEYS, *filters)))
    try:
        with source.open("rb") as stream, source.open("rb", buffering=0) as framing:
            while True:
                message = eccodes.codes_grib_new_from_file(stream, headers_only=True)
                if message is None:
                    break
                try:
                    offset = int(eccodes.codes_get(message, "offset"))
                    length = int(eccodes.codes_get(message, "totalLength"))
                    if offset != end or length <= 0:
                        raise ValueError("GRIB input must contain contiguous complete messages without envelopes or stray bytes")
                    end = offset + length
                    if _header(eccodes, message, "second") not in (None, 0):
                        raise ValueError("GRIB reference times with nonzero seconds are unsupported; cfgrib would truncate them")
                    record = {key: _header(eccodes, message, key) for key in keys}
                    if record["edition"] == 2:
                        _single_field_grib2(framing, offset, length)
                    if any(record.get(key) is None for key in _REQUIRED_KEYS):
                        raise ValueError("GRIB message lacks required grid, pressure/level or forecast-time header evidence")
                    if all(record[key] == value for key, value in filters.items()):
                        records.append(record)
                finally:
                    eccodes.codes_release(message)
    except (OSError, ValueError):
        raise
    except Exception as error:
        raise ValueError("Cannot read complete GRIB headers; check the input and ecCodes definitions") from error
    if end != source.stat().st_size:
        raise ValueError("GRIB input has trailing, truncated or non-message bytes")
    if not records:
        raise ValueError("No GRIB messages match the requested filters")
    return records


def _group(record):
    return tuple(record[key] for key in _GROUP_KEYS)


def _describe_groups(records):
    groups = {}
    for record in records:
        groups.setdefault(_group(record), []).append(record)
    result = []
    for key, group in groups.items():
        result.append({
            "filter_by_keys": {name: value for name, value in zip(_GROUP_KEYS, key) if value is not None},
            "messages": len(group),
            "parameters": sorted({record["paramId"] for record in group}),
            "short_names": sorted({record["shortName"] for record in group if record["shortName"] is not None}),
            "levels": sorted({record["level"] for record in group}),
            "forecast_reference_times": sorted({(record["dataDate"], record["dataTime"]) for record in group}),
            "step_windows": sorted({(record["startStep"], record["endStep"]) for record in group}),
            "forecast_windows": sorted({(record["dataDate"], record["dataTime"], record["startStep"], record["endStep"])
                                        for record in group}),
        })
    return result


def inspect_grib_groups(path):
    """Return header-derived groups for explicit filtering; do not open fields.

    The list contains group filters, parameter IDs, levels, forecast reference
    dates/times and original start/end step windows. Dates/times are GRIB header
    integers, not filenames. No group is chosen or merged automatically. Header
    scanning reads encoded messages but never decodes their field values, and
    writes no .idx or cache files. Only pure concatenated local, single-field
    GRIB messages are accepted. Multi-field GRIB2 is rejected explicitly.
    """
    source, stamp = _source(path)
    eccodes, _, _ = _backends()
    records = _scan(source, eccodes, {})
    _check_input_stamp(source, stamp)
    return _describe_groups(records)


def _check_layout(records):
    if len({_group(record) for record in records}) != 1:
        raise ValueError(
            "Selected GRIB messages have heterogeneous level types, step types, grids or product definitions. "
            "Use inspect_grib_groups and explicit filter_by_keys; groups are never merged automatically."
        )
    windows = {}
    parameters = {}
    for record in records:
        parameters.setdefault(record["paramId"], []).append(record)
        key = (record["dataDate"], record["dataTime"], record["endStep"])
        windows.setdefault(key, set()).add(record["startStep"])
    if any(len(starts) != 1 for starts in windows.values()):
        raise ValueError("GRIB fields have conflicting statistical windows at the same reference/end time; filter explicitly")
    for parameter, rows in parameters.items():
        slots = set()
        for record in rows:
            # cfgrib's step coordinate denotes the forecast end. Distinct time
            # windows with the same end must not overwrite one another.
            slot = (record["dataDate"], record["dataTime"], record["endStep"], record["level"], record["number"])
            if slot in slots:
                raise ValueError(f"Duplicate or conflicting GRIB forecast slot for paramId={parameter}; select a unique product")
            slots.add(slot)
        sizes = [len({(row["dataDate"], row["dataTime"]) for row in rows})]
        sizes.extend(len({row[key] for row in rows}) for key in ("endStep", "level", "number"))
        if len(slots) != math.prod(sizes):
            raise ValueError(
                f"Incomplete GRIB message hypercube for paramId={parameter}; "
                "select a complete run/step/level/member group explicitly instead of filling absent messages"
            )


def open_grib(path, *, chunks="auto", filter_by_keys=None):
    """Open one explicit local GRIB group as a Dask-backed Dataset.

    ``filter_by_keys`` accepts only documented scalar header keys (see
    ``inspect_grib_groups``). Mixed grids/level types/step types/product templates,
    duplicate message slots and sparse forecast hypercubes are rejected. Use
    explicit filters to select a compatible group; no values are aggregated,
    no members are selected implicitly and absent messages are never filled.
    Each physical GRIB message must contain one field; split multi-field GRIB2
    explicitly first. Envelopes and stray/truncated bytes are unsupported.
    Nonzero seconds in the reference timestamp are rejected because cfgrib's
    reference-time conversion currently has minute precision.

    Singleton time/step/level/member axes remain dimensions. ``time`` is the
    forecast reference time; ``step`` is the lead time; ``valid_time`` may be
    two-dimensional and is never flattened or deduplicated. GRIB statistical
    step types remain on variables; distinct source start/end windows and their
    units are retained in the provenance group metadata. This is not a forecast
    accumulation conversion or a calendar/coordinate normalization operation.

    Only header keys and geographic coordinates are inspected eagerly; ecCodes
    may still read complete compressed message bytes. Keep the
    source unchanged until computing/writing, then close the returned Dataset.
    Executed field blocks check source size/mtime; this is not a content hash.
    Persistent .idx files and cfgrib's geographic cache are disabled.
    """
    chunks = _checked_chunks(chunks)
    filters = _filters(filter_by_keys)
    source, stamp = _source(path)
    eccodes, cfgrib, native = _backends()
    records = _scan(source, eccodes, filters)
    _check_layout(records)
    _check_input_stamp(source, stamp)
    dataset = None
    try:
        dataset = xr.open_dataset(
            source, engine="cfgrib", chunks=chunks, cache=False,
            backend_kwargs={"indexpath": "", "filter_by_keys": _backend_filters(filters), "errors": "raise",
                            "cache_geo_coords": False, "squeeze": False,
                            "values_dtype": np.dtype("float64")},
        )
        if isinstance(chunks, Mapping) and set(chunks) - set(dataset.dims):
            raise ValueError(f"Unknown chunk dimensions: {sorted(set(chunks) - set(dataset.dims))}")
        if not dataset.data_vars or any(value.chunks is None for value in dataset.data_vars.values()):
            raise ValueError("cfgrib did not provide Dask-backed meteorological fields")
        if {value.attrs.get("GRIB_paramId") for value in dataset.data_vars.values()} != {
            record["paramId"] for record in records
        }:
            raise ValueError("cfgrib parameter selection differs from the inspected GRIB headers")
        validate_metadata(dataset)
        if "time" in dataset.coords:
            validate_time(dataset)
        for name in dataset.data_vars:
            variable = dataset[name]
            guarded = variable.data.map_blocks(
                _guard_source_block, path=str(source), stamp=stamp, dtype=variable.dtype,
                meta=np.empty((0,) * variable.ndim, dtype=variable.dtype),
            )
            dataset[name] = variable.copy(data=guarded, deep=False)
        dataset.encoding.pop("source", None)
        for variable in dataset.variables.values():
            variable.encoding.pop("source", None)
        # cfgrib constructs this history itself from backend options including
        # the absolute transport path; it is not an original GRIB history field.
        dataset.attrs.pop("history", None)
        result = record_operation(dataset, "open_grib", {
            "chunks": chunks, "filter_by_keys": filters, "indexpath": "", "cache_geo_coords": False,
            "selected_messages": len(records), "source_groups": _describe_groups(records),
            "cfgrib_version": getattr(cfgrib, "__version__", metadata.version("cfgrib")),
            "eccodes_version": native, "forecast_time_policy": "retain_reference_time_step_and_valid_time",
            "backend_history_policy": "generated_path_history_replaced_by_structured_provenance",
        }, inputs=[input_record(source)])
        _check_input_stamp(source, stamp)
        return result
    except BaseException:
        if dataset is not None:
            dataset.close()
        raise
