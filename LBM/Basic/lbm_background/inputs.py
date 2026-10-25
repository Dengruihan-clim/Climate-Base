"""Read monthly CF NetCDF reanalysis and make equal-month climatologies.

This module never infers dates from filenames.  Input file patterns select
candidates, and decoded CF time values select the one record for each month.
Fields may use different global rectilinear grids, but a field's grid and all
three-dimensional fields' pressure levels must remain consistent over time.
"""
from __future__ import annotations

import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .horizontal import periodic_bilinear


REQUIRED_FIELDS = ("U", "V", "T", "Q", "H", "SLP")
PRESSURE_FIELDS = frozenset(("U", "V", "T", "Q", "H"))
STANDARD_GRAVITY = 9.80665
DEFAULT_COORDINATES = {
    "time": "time", "latitude": "lat", "longitude": "lon", "pressure": "lev"
}


def season_dates(period):
    """Return ``(season_year, calendar_year, month)`` in season order.

    The year label is the year of the last configured month.  Thus DJF 1991
    contains 1990-12, 1991-01 and 1991-02.  Only contiguous, nonrepeated
    month sequences of one to twelve months are accepted.
    """
    start, end = period.get("start_year"), period.get("end_year")
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (start, end)):
        raise ValueError("period.start_year and end_year must be integers")
    if not 1 <= start <= end <= 9999:
        raise ValueError("period years must satisfy 1 <= start_year <= end_year <= 9999")
    months = period.get("months", [12, 1, 2])
    if not isinstance(months, (list, tuple)) or not 1 <= len(months) <= 12:
        raise ValueError("period.months must contain one to twelve contiguous months")
    if any(isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 12 for m in months):
        raise ValueError("period.months entries must be integers in 1..12")
    if len(set(months)) != len(months) or any(b != a % 12 + 1 for a, b in zip(months, months[1:])):
        raise ValueError("period.months must be contiguous and contain no repeated month")
    if start == 1 and any(m > months[-1] for m in months):
        raise ValueError("This season would require calendar year zero")
    return [(year, year - int(month > months[-1]), month)
            for year in range(start, end + 1) for month in months]


def _unit_key(value):
    value = str(value).strip().lower()
    for before, after in (("−", "-"), ("⁻", "-"), ("¹", "1"), ("²", "2"),
                          ("**", ""), ("^", ""), ("·", ""), ("_", ""), (".", "")):
        value = value.replace(before, after)
    return re.sub(r"\s+", "", value)


def _pressure_factor(units):
    key = _unit_key(units)
    if key in {"pa", "pascal", "pascals"}:
        return 0.01
    if key in {"hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars"}:
        return 1.0
    raise ValueError(f"Unsupported pressure units {units!r}; use Pa or hPa")


def _convert_units(values, key, units):
    unit = _unit_key(units)
    if key in ("U", "V"):
        if unit in {"ms-1", "m/s", "msec-1", "meterspersecond", "metrespersecond"}:
            return values
    elif key == "T":
        if unit in {"k", "kelvin", "kelvins"}:
            return values
        if unit in {"degc", "degreecelsius", "degreescelsius", "celsius"}:
            return values + 273.15
    elif key == "Q":
        if unit in {"kgkg-1", "kg/kg", "1", "dimensionless"}:
            return values
        if unit in {"gkg-1", "g/kg"}:
            return values * 0.001
    elif key == "H":
        if unit in {"m", "meter", "meters", "metre", "metres", "gpm", "geopotentialmeter", "geopotentialmetre"}:
            return values
        if unit in {"m2s-2", "m2/s2", "m2sec-2"}:
            return values / STANDARD_GRAVITY
    elif key in ("SLP", "PS_SOURCE"):
        return values * _pressure_factor(units)
    raise ValueError(f"Unsupported {key} units {units!r}; correct metadata or set an explicit units_override")


def _coordinate(ds, name, label):
    if name not in ds.variables:
        raise ValueError(f"Missing {label} coordinate {name!r}")
    variable = ds.variables[name]
    if label in ("latitude", "longitude"):
        allowed = ({"degreesnorth", "degreenorth", "degreesn", "degreen"} if label == "latitude"
                   else {"degreeseast", "degreeeast", "degreese", "degreee"})
        if _unit_key(getattr(variable, "units", "")) not in allowed:
            expected = "degrees_north" if label == "latitude" else "degrees_east"
            raise ValueError(f"{label} coordinate units must describe {expected}; radians or west/south-positive axes must be converted first")
    if variable.ndim != 1 or len(variable.dimensions) != 1:
        raise ValueError(f"{label} coordinate must be one-dimensional")
    values = np.ma.asarray(variable[:], dtype=np.float64).filled(np.nan)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{label} coordinate contains missing/non-finite values")
    return values, variable.dimensions[0]


def _normalized_grid(ds, names, is_pressure):
    lat, lat_dim = _coordinate(ds, names["latitude"], "latitude")
    lon, lon_dim = _coordinate(ds, names["longitude"], "longitude")
    if lat.size < 2 or lon.size < 4 or np.any(np.abs(lat) > 90):
        raise ValueError("A global rectilinear grid requires valid latitude and at least four longitude samples")
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    if np.any(np.diff(lat) <= 0):
        raise ValueError("Latitude coordinate contains duplicates")
    lon = lon % 360.0
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    if np.any(np.diff(lon) <= 1e-9):
        raise ValueError("Longitude contains duplicate cyclic endpoints; remove the repeated 0/360 column")
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    # A large closing gap indicates a regional field that must not be wrapped
    # around the globe.  Modest irregularities are allowed on global grids.
    if gaps.max() > 1.5 * np.median(gaps) + 1e-8:
        raise ValueError("Longitude coverage is regional or has a large cyclic gap; a global grid is required")
    dimensions = [lat_dim, lon_dim]
    orders = [lat_order, lon_order]
    pressure = None
    if is_pressure:
        pressure, pressure_dim = _coordinate(ds, names["pressure"], "pressure")
        pressure *= _pressure_factor(getattr(ds.variables[names["pressure"]], "units", ""))
        pressure_order = np.argsort(pressure)[::-1]
        pressure = pressure[pressure_order]
        if pressure.size < 2 or np.any(pressure <= 0) or np.any(np.diff(pressure) >= 0):
            raise ValueError("Pressure levels must be distinct, positive and have at least two values")
        dimensions.insert(0, pressure_dim)
        orders.insert(0, pressure_order)
    if len(set(dimensions)) != len(dimensions):
        raise ValueError("Horizontal and pressure coordinates must use distinct dimensions")
    return lat, lon, pressure, dimensions, orders


def _read_record(ds, key, spec, names, record_index):
    varname = spec.get("variable")
    if not varname or varname not in ds.variables:
        raise ValueError(f"Missing configured {key} variable {varname!r}")
    var = ds.variables[varname]
    lat, lon, pressure, dimensions, orders = _normalized_grid(ds, names, key in PRESSURE_FIELDS)
    time_variable = ds.variables[names["time"]]
    time_dim = time_variable.dimensions[0]
    if time_dim not in var.dimensions or time_dim in dimensions:
        raise ValueError(f"{key} must have a distinct time dimension")
    if any(dim not in var.dimensions for dim in dimensions):
        raise ValueError(f"{key} dimensions do not contain its configured coordinate dimensions")
    selection, retained = [], []
    for dim in var.dimensions:
        if dim == time_dim:
            selection.append(record_index)
        elif dim in dimensions:
            selection.append(slice(None))
            retained.append(dim)
        elif len(ds.dimensions[dim]) == 1:
            selection.append(0)
        else:
            raise ValueError(f"{key} has unconfigured non-singleton dimension {dim!r}; select it explicitly in the source data")
    values = np.ma.asarray(var[tuple(selection)], dtype=np.float64).filled(np.nan)
    values = np.transpose(values, [retained.index(dim) for dim in dimensions])
    for axis, order in enumerate(orders):
        values = np.take(values, order, axis=axis)
    source_units = str(getattr(var, "units", ""))
    units = spec.get("units_override", source_units)
    values = _convert_units(values, key, units)
    values[~np.isfinite(values)] = np.nan
    return values, lat, lon, pressure, source_units


def _fingerprint(path, display, integrity):
    receipt_path = Path(str(path) + ".verified.json")
    receipt = None
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
    elif integrity.get("require_receipts", False):
        raise ValueError(f"Missing required .verified.json receipt: {display}")
    size = path.stat().st_size
    digest = None
    if receipt is not None or integrity.get("hash_inputs", True):
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if receipt is not None:
        if receipt.get("bytes") != size or receipt.get("sha256") != digest:
            raise ValueError(f"Input receipt size/SHA256 mismatch: {display}")
    return {"path": display, "bytes": size, "sha256": digest,
            "receipt_verified": receipt is not None, "records": []}


def load_climatology(config, target_lat, target_lon):
    """Load configured monthly data and return fields, levels, provenance, QA.

    Returned units are U/V m s-1, T K, Q kg kg-1, H m and SLP/PS_SOURCE hPa.
    The pressure axis is hPa from surface toward top.  Three-dimensional
    field arrays use (pressure, target latitude, target longitude).
    """
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as exc:
        raise ImportError("Reading monthly input requires netCDF4; install the package dependencies") from exc
    specs = config.get("fields", {})
    missing = set(REQUIRED_FIELDS) - set(specs)
    unknown = set(specs) - set(REQUIRED_FIELDS) - {"PS_SOURCE"}
    if missing or unknown:
        raise ValueError(f"Configured fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    period = config.get("period", {})
    dates = season_dates(period)
    years = period["end_year"] - period["start_year"] + 1
    minimum = period.get("minimum_valid_years", 1)
    if isinstance(minimum, bool) or not isinstance(minimum, int) or not 1 <= minimum <= years:
        raise ValueError("period.minimum_valid_years must be an integer from one to the number of season years")
    base = Path(config.get("_config_dir", ".")).resolve()
    data_root = Path(config.get("data_root", ".")).expanduser()
    if not data_root.is_absolute():
        data_root = base / data_root
    data_root = data_root.resolve()
    integrity = config.get("integrity", {})
    coord_defaults = {**DEFAULT_COORDINATES, **config.get("coordinates", {})}
    manifests, date_cache, result, qa = {}, {}, {}, {}
    pressure_reference = None

    def label(path):
        try:
            return path.relative_to(data_root).as_posix()
        except ValueError:
            # Do not leak absolute home/project directory names into output.
            return f"external_input_{len(manifests) + 1:04d}/{path.name}"

    for key in (*REQUIRED_FIELDS, *(("PS_SOURCE",) if "PS_SOURCE" in specs else ())):
        spec = specs[key]
        if not isinstance(spec, dict) or not isinstance(spec.get("files"), str):
            raise ValueError(f"fields.{key}.files must be a filename/glob template string")
        names = {**coord_defaults, **spec.get("coordinates", {})}
        reference = None
        month_means, month_qa, source_units = [], [], set()
        for month in period.get("months", [12, 1, 2]):
            total = count = None
            selected_dates = [(cy, m) for _, cy, m in dates if m == month]
            for year, date_month in selected_dates:
                yyyymm = f"{year:04d}{date_month:02d}"
                try:
                    pattern = spec["files"].format(data_root=data_root.as_posix(), year=year,
                                                   month=date_month, yyyymm=yyyymm)
                except (KeyError, ValueError) as exc:
                    raise ValueError(f"Invalid filename template for {key}; supported placeholders are data_root/year/month/yyyymm") from exc
                pattern_path = Path(pattern).expanduser()
                if not pattern_path.is_absolute():
                    pattern_path = base / pattern_path
                candidates = sorted({Path(p).resolve() for p in glob.glob(str(pattern_path)) if Path(p).is_file()})
                if not candidates:
                    raise ValueError(f"No input files match {key} for {yyyymm}")
                matches = []
                for path in candidates:
                    cache_key = (path, names["time"])
                    if cache_key not in date_cache:
                        with Dataset(path) as ds:
                            name = names["time"]
                            if name not in ds.variables or ds.variables[name].ndim != 1:
                                raise ValueError(f"{key}: missing one-dimensional CF time coordinate")
                            time_var = ds.variables[name]
                            if not getattr(time_var, "units", ""):
                                raise ValueError(f"{key}: CF time coordinate lacks units")
                            raw_times = np.ma.asarray(time_var[:])
                            if np.ma.getmaskarray(raw_times).any() or not np.isfinite(raw_times).all():
                                raise ValueError(f"{key}: CF time coordinate contains missing values")
                            decoded = num2date(raw_times, time_var.units,
                                               calendar=getattr(time_var, "calendar", "standard"))
                            date_cache[cache_key] = [(int(value.year), int(value.month)) for value in np.atleast_1d(decoded)]
                    matches.extend((path, index) for index, date in enumerate(date_cache[cache_key])
                                   if date == (year, date_month))
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one CF monthly record for {key} {yyyymm}; found {len(matches)}")
                path, index = matches[0]
                if path not in manifests:
                    manifests[path] = _fingerprint(path, label(path), integrity)
                with Dataset(path) as ds:
                    values, lat, lon, pressure, units = _read_record(ds, key, spec, names, index)
                source_units.add(units)
                if reference is None:
                    reference = (lat, lon, pressure)
                if not np.array_equal(lat, reference[0]) or not np.array_equal(lon, reference[1]):
                    raise ValueError(f"Source horizontal grid changes over time for {key}")
                if pressure is not None:
                    if pressure_reference is None:
                        pressure_reference = pressure.copy()
                    if not np.array_equal(pressure, pressure_reference):
                        raise ValueError(f"Pressure levels differ between records or fields: {key}")
                if total is None:
                    total = np.zeros(values.shape, dtype=np.float64)
                    count = np.zeros(values.shape, dtype=np.int32)
                valid = np.isfinite(values)
                total += np.where(valid, values, 0.0)
                count += valid
                manifests[path]["records"].append({"field": key, "month": yyyymm, "time_index": index})
            average = np.full(total.shape, np.nan)
            np.divide(total, count, out=average, where=count >= minimum)
            interpolated = periodic_bilinear(average, reference[0], reference[1], target_lat, target_lon)
            month_means.append(interpolated)
            nonzero = count[count > 0]
            month_qa.append({"month": month, "years": years, "minimum_valid_years_required": minimum,
                             "missing_all_years_cells": int(np.sum(count == 0)),
                             "partially_sampled_cells": int(np.sum((count > 0) & (count < years))),
                             "cells_below_minimum_valid_years": int(np.sum(count < minimum)),
                             "minimum_nonzero_year_count": int(nonzero.min()) if nonzero.size else 0,
                             "target_missing_cells": int(np.sum(~np.isfinite(interpolated)))})
        # np.mean deliberately preserves missing months rather than weighting
        # whichever months happen to survive at an underground grid cell.
        result[key] = np.mean(np.stack(month_means), axis=0)
        qa[key] = {"source_units": sorted(source_units), "units_override": spec.get("units_override"),
                   "monthly_climatologies": month_qa,
                   "seasonal_missing_cells": int(np.sum(~np.isfinite(result[key])))}
    inventory = list(manifests.values())
    return result, pressure_reference, inventory, qa
