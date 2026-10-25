"""Monthly CF inputs for pressure-coordinate heating diagnostics.

No field is averaged here.  In particular, every neighbouring temperature
record is masked with its own month's surface pressure before regridding.
Missing or underground interpolation corners are never renormalized away.
"""
from __future__ import annotations

from collections import OrderedDict
import glob
import hashlib
import json
from pathlib import Path
import re

import numpy as np


DEFAULT_COORDINATES = {
    "time": "time", "latitude": "lat", "longitude": "lon", "pressure": "lev"
}
KINDS = ("T", "U", "V", "OMEGA", "PS")


def season_dates(period):
    """Return (season year, calendar year, month), labelled by final month."""
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


def neighbor_month(year, month, offset):
    """Return an adjacent calendar month without assuming its day count."""
    if not 1 <= int(month) <= 12 or not 1 <= int(year) <= 9999 or int(offset) != offset:
        raise ValueError("Invalid year, month or integer month offset")
    index = int(year) * 12 + int(month) - 1 + int(offset)
    result_year, result_month = divmod(index, 12)
    if not 1 <= result_year <= 9999:
        raise ValueError("Neighbour month lies outside calendar years 1..9999")
    return result_year, result_month + 1


def _unit_key(value):
    value = str(value).strip().lower()
    for before, after in (("−", "-"), ("⁻", "-"), ("¹", "1"), ("²", "2"),
                          ("**", ""), ("^", ""), ("·", ""), ("_", ""), (".", "")):
        value = value.replace(before, after)
    return re.sub(r"\s+", "", value)


def _pressure_factor(units):
    unit = _unit_key(units)
    if unit in {"pa", "pascal", "pascals"}:
        return 1.0
    if unit in {"hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars"}:
        return 100.0
    raise ValueError(f"Unsupported pressure units {units!r}; use Pa or hPa")


def _convert(values, kind, units):
    unit = _unit_key(units)
    if kind == "T":
        if unit in {"k", "kelvin", "kelvins"}:
            return values, "K", "identity"
        if unit in {"degc", "degreecelsius", "degreescelsius", "celsius"}:
            return values + 273.15, "K", "add 273.15"
    elif kind in {"U", "V"}:
        if unit in {"ms-1", "m/s", "msec-1", "meterspersecond", "metrespersecond"}:
            return values, "m s-1", "identity"
    elif kind == "PS":
        factor = _pressure_factor(units)
        return values * factor, "Pa", f"multiply by {factor:g}"
    elif kind == "OMEGA":
        if unit in {"pas-1", "pa/s", "pasec-1"}:
            return values, "Pa s-1", "identity"
        if unit in {"hpas-1", "hpa/s", "hpasec-1", "mbs-1", "mb/s", "mbars-1", "mbar/s"}:
            return values * 100.0, "Pa s-1", "multiply by 100"
        if unit in {"ms-1", "m/s", "msec-1"}:
            raise ValueError("OMEGA requires pressure vertical velocity in Pa/s or hPa/s; geometric m/s velocity cannot be substituted")
    raise ValueError(f"Unsupported {kind} units {units!r}; correct metadata or set an explicit units_override")


def _axis(ds, name, description):
    if name not in ds.variables or ds.variables[name].ndim != 1:
        raise ValueError(f"Missing one-dimensional {description} coordinate {name!r}")
    var = ds.variables[name]
    values = np.ma.asarray(var[:], dtype=np.float64).filled(np.nan)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{description} coordinate is empty or contains missing values")
    units = _unit_key(getattr(var, "units", ""))
    if description == "latitude" and units not in {"degreesnorth", "degreenorth", "degreesn", "degreen"}:
        raise ValueError("Latitude coordinate units must explicitly indicate degrees_north")
    if description == "longitude" and units not in {"degreeseast", "degreeeast", "degreese", "degreee"}:
        raise ValueError("Longitude coordinate units must explicitly indicate degrees_east")
    return values, var.dimensions[0]


def _global_grid(lat, lon):
    if lat.ndim != 1 or lon.ndim != 1 or lat.size < 2 or lon.size < 4:
        raise ValueError("A global rectilinear grid needs at least two latitudes and four longitudes")
    if not np.isfinite(lat).all() or not np.isfinite(lon).all() or np.any(np.abs(lat) > 90):
        raise ValueError("Horizontal coordinates contain invalid values")
    lat_order = np.argsort(lat)
    lat = lat[lat_order]
    if np.any(np.diff(lat) <= 0):
        raise ValueError("Latitude contains duplicate coordinates")
    lon = lon % 360.0
    lon_order = np.argsort(lon)
    lon = lon[lon_order]
    if np.any(np.diff(lon) <= 1e-9):
        raise ValueError("Longitude contains duplicate cyclic endpoints")
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    if gaps.max() > 1.5 * np.median(gaps) + 1e-8:
        raise ValueError("A global grid is required; longitude has a regional cyclic gap")
    return lat, lon, lat_order, lon_order


def strict_bilinear(field, lat, lon, target_lat, target_lon):
    """Cyclic bilinear interpolation requiring every positive-weight corner.

    Source axes must be increasing, latitude within [-90, 90], longitude
    without a repeated cyclic endpoint, and globally covering longitude.
    A NaN at a corner of exactly zero weight has no effect.  No latitude
    extrapolation, nearest-neighbour filling or missing-corner weight
    renormalization is performed.  Exact polar targets outside a Gaussian
    source's latitude range are returned as NaN rows, without extrapolation;
    other out-of-range target latitudes are rejected.
    """
    data = np.ma.asarray(field, dtype=np.float64).filled(np.nan)
    lat, lon, target_lat, target_lon = [np.asarray(a, dtype=np.float64)
                                        for a in (lat, lon, target_lat, target_lon)]
    if data.ndim < 2 or data.shape[-2:] != (lat.size, lon.size):
        raise ValueError("Input field dimensions do not match its horizontal coordinates")
    if any(a.ndim != 1 or not np.isfinite(a).all() for a in (lat, lon, target_lat, target_lon)):
        raise ValueError("Bilinear coordinate axes must be finite and one-dimensional")
    if lat.size < 2 or lon.size < 4 or np.any(np.diff(lat) <= 0) or np.any(np.diff(lon) <= 0):
        raise ValueError("Bilinear source axes must increase and represent a global grid")
    gaps = np.diff(np.r_[lon, lon[0] + 360.0])
    if lon[-1] - lon[0] >= 360 or gaps.max() > 1.5 * np.median(gaps) + 1e-8:
        raise ValueError("Bilinear source longitude must cover the globe without a cyclic duplicate")
    outside = (target_lat < lat[0]) | (target_lat > lat[-1])
    missing_poles = outside & np.isclose(np.abs(target_lat), 90.0, rtol=0, atol=1e-10)
    if np.any(outside & ~missing_poles):
        raise ValueError("Nonpolar target latitude lies outside the source grid; extrapolation is forbidden; source coverage must include all nonpolar targets")
    # A clipped working coordinate keeps bilinear weights valid.  The rows
    # are explicitly discarded below, so no endpoint extrapolation is used.
    working_lat = np.clip(target_lat, lat[0], lat[-1])
    wrapped = (target_lon - lon[0]) % 360.0 + lon[0]
    ix = np.searchsorted(lon, wrapped, side="right") - 1
    jx = (ix + 1) % lon.size
    right = np.where(jx == 0, lon[0] + 360, lon[jx])
    wx = (wrapped - lon[ix]) / (right - lon[ix])
    iy = np.clip(np.searchsorted(lat, working_lat, side="right") - 1, 0, lat.size - 2)
    jy = iy + 1
    wy = (working_lat - lat[iy]) / (lat[jy] - lat[iy])
    output = np.zeros(data.shape[:-2] + (target_lat.size, target_lon.size), dtype=np.float64)
    valid = np.ones(output.shape, dtype=bool)
    for y, yw in ((iy, 1.0 - wy), (jy, wy)):
        for x, xw in ((ix, 1.0 - wx), (jx, wx)):
            corner = data[..., y[:, None], x[None, :]]
            weight = yw[:, None] * xw[None, :]
            present = np.isfinite(corner)
            valid &= present | (weight == 0)
            output += np.where(present, corner, 0.0) * weight
    output[~valid] = np.nan
    output[..., missing_poles, :] = np.nan
    return output


def _calendar_name(value):
    value = str(value).strip().lower()
    return {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}.get(value, value)


def _comparison_calendar(source_calendar, selected_year):
    """Use Gregorian equivalence only after the historical reform period.

    This helper is used after decoding with the original file calendar.
    Modern ERA5 sources often label identical Gregorian months differently.
    Ancient mixed standard/proleptic dates retain their distinct semantics.
    """
    calendar = _calendar_name(source_calendar)
    if selected_year >= 1583 and calendar in {"standard", "proleptic_gregorian"}:
        return "proleptic_gregorian"
    return calendar


class MonthlyReader:
    """Read individual masked monthly fields and retain bounded provenance.

    All three-dimensional arrays are (pressure, latitude, longitude), with
    pressure strictly increasing in Pa and common across all records and
    variables.  ``calendar`` is None until the first selected record.
    Only two months of two-dimensional native/target surface pressure are
    cached; no high-resolution three-dimensional field arrays are retained.
    """

    def __init__(self, config, target_lat, target_lon):
        self.config = config
        self.target_lat = np.asarray(target_lat, dtype=np.float64)
        self.target_lon = np.asarray(target_lon, dtype=np.float64)
        if any(a.ndim != 1 or not np.isfinite(a).all() or not a.size for a in (self.target_lat, self.target_lon)):
            raise ValueError("Target latitude/longitude must be finite nonempty one-dimensional arrays")
        self.specs = config.get("fields", {})
        missing, unknown = set(KINDS) - set(self.specs), set(self.specs) - set(KINDS)
        if missing or unknown:
            raise ValueError(f"Configured fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
        for kind, spec in self.specs.items():
            if not isinstance(spec, dict) or not isinstance(spec.get("files"), str) or not spec.get("variable"):
                raise ValueError(f"fields.{kind} requires files and variable")
        self.base = Path(config.get("_config_dir", ".")).resolve()
        self.data_root = Path(config.get("data_root", ".")).expanduser()
        if not self.data_root.is_absolute():
            self.data_root = self.base / self.data_root
        self.data_root = self.data_root.resolve()
        self.coordinates = {**DEFAULT_COORDINATES, **config.get("coordinates", {})}
        self.integrity = config.get("integrity", {})
        self._calendar = None
        self._pressure = None
        self._dates = {}
        self._manifest = OrderedDict()
        self._grid_reference = {}
        self._ps_cache = OrderedDict()
        self.qa = {"records": [], "calendar": None, "source_calendars": [], "pressure_pa": None,
                   "mask_rule": "native monthly PS before strict regridding, target monthly PS afterward",
                   "surface_pressure_cache_months": 2}

    @property
    def calendar(self):
        return self._calendar

    @property
    def inventory(self):
        return list(self._manifest.values())

    def _label(self, path):
        try:
            return path.relative_to(self.data_root).as_posix()
        except ValueError:
            return f"external_input_{len(self._manifest) + 1:04d}/{path.name}"

    def _register(self, path, kind, year, month, index):
        if path not in self._manifest:
            label = self._label(path)
            size = path.stat().st_size
            receipt_path = Path(str(path) + ".verified.json")
            receipt = None
            if receipt_path.exists():
                receipt = json.loads(receipt_path.read_text(encoding="utf-8-sig"))
            elif self.integrity.get("require_receipts", False):
                raise ValueError(f"Missing required verified receipt: {label}")
            digest = None
            if receipt is not None or self.integrity.get("hash_inputs", True):
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if receipt is not None and (receipt.get("bytes") != size or receipt.get("sha256") != digest):
                raise ValueError(f"Receipt size/SHA256 mismatch: {label}")
            self._manifest[path] = {"path": label, "bytes": size, "sha256": digest,
                                    "receipt_verified": receipt is not None, "records": []}
        record = {"field": kind, "month": f"{year:04d}{month:02d}", "time_index": index}
        if record not in self._manifest[path]["records"]:
            self._manifest[path]["records"].append(record)

    def _selection(self, kind, year, month, Dataset, num2date):
        spec = self.specs[kind]
        names = {**self.coordinates, **spec.get("coordinates", {})}
        try:
            pattern = spec["files"].format(data_root=self.data_root.as_posix(), year=year, month=month,
                                           yyyymm=f"{year:04d}{month:02d}")
        except (KeyError, ValueError) as exc:
            raise ValueError(f"Invalid {kind} file template; supported placeholders: data_root/year/month/yyyymm") from exc
        path_pattern = Path(pattern).expanduser()
        if not path_pattern.is_absolute():
            path_pattern = self.base / path_pattern
        files = sorted({Path(p).resolve() for p in glob.glob(str(path_pattern)) if Path(p).is_file()})
        matches = []
        for path in files:
            cache_key = (path, names["time"])
            if cache_key not in self._dates:
                with Dataset(path) as ds:
                    name = names["time"]
                    if name not in ds.variables or ds.variables[name].ndim != 1:
                        raise ValueError(f"{kind}: missing one-dimensional CF time coordinate")
                    var = ds.variables[name]
                    units = getattr(var, "units", "")
                    if not units:
                        raise ValueError(f"{kind}: CF time coordinate lacks units")
                    raw = np.ma.asarray(var[:])
                    if np.ma.getmaskarray(raw).any() or not np.isfinite(raw).all():
                        raise ValueError(f"{kind}: CF time contains missing values")
                    source_calendar = str(getattr(var, "calendar", "standard")).strip().lower()
                    decoded = num2date(raw, units, calendar=source_calendar)
                    self._dates[cache_key] = ([(int(d.year), int(d.month)) for d in np.atleast_1d(decoded)], source_calendar)
            dates, source_calendar = self._dates[cache_key]
            calendar = _comparison_calendar(source_calendar, year)
            matches.extend((path, index, calendar, source_calendar) for index, date in enumerate(dates) if date == (year, month))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one CF monthly record for {kind} {year:04d}{month:02d}; found {len(matches)}")
        path, index, calendar, source_calendar = matches[0]
        if self._calendar is None:
            self._calendar = calendar
            self.qa["calendar"] = calendar
        elif self._calendar != calendar:
            raise ValueError(f"Mixed CF calendars are unsupported: {self._calendar} versus {calendar}")
        if source_calendar not in self.qa["source_calendars"]:
            self.qa["source_calendars"].append(source_calendar)
        self._register(path, kind, year, month, index)
        return path, index, names

    def _native(self, kind, year, month):
        try:
            from netCDF4 import Dataset, num2date
        except ImportError as exc:
            raise ImportError("Monthly CF inputs require netCDF4; install this package's dependencies") from exc
        path, index, names = self._selection(kind, year, month, Dataset, num2date)
        spec = self.specs[kind]
        with Dataset(path) as ds:
            varname = spec["variable"]
            if varname not in ds.variables:
                raise ValueError(f"Missing configured {kind} variable {varname!r}")
            var = ds.variables[varname]
            lat, lat_dim = _axis(ds, names["latitude"], "latitude")
            lon, lon_dim = _axis(ds, names["longitude"], "longitude")
            lat, lon, lat_order, lon_order = _global_grid(lat, lon)
            dimensions, orders = [lat_dim, lon_dim], [lat_order, lon_order]
            pressure = None
            if kind != "PS":
                pressure, pressure_dim = _axis(ds, names["pressure"], "pressure")
                pressure *= _pressure_factor(getattr(ds.variables[names["pressure"]], "units", ""))
                p_order = np.argsort(pressure)
                pressure = pressure[p_order]
                if pressure.size < 3 or np.any(pressure <= 0) or np.any(np.diff(pressure) <= 0):
                    raise ValueError("At least three distinct positive pressure levels are required")
                dimensions.insert(0, pressure_dim)
                orders.insert(0, p_order)
                if self._pressure is None:
                    self._pressure = pressure.copy()
                    self.qa["pressure_pa"] = pressure.tolist()
                elif not np.array_equal(self._pressure, pressure):
                    raise ValueError("Pressure levels differ between variables or records; no silent intersection is allowed")
            if len(set(dimensions)) != len(dimensions):
                raise ValueError("Configured spatial coordinates must use distinct dimensions")
            time_dim = ds.variables[names["time"]].dimensions[0]
            if time_dim not in var.dimensions or time_dim in dimensions or any(d not in var.dimensions for d in dimensions):
                raise ValueError(f"{kind} variable dimensions disagree with configured coordinates")
            selectors, retained = [], []
            for dim in var.dimensions:
                if dim == time_dim:
                    selectors.append(index)
                elif dim in dimensions:
                    selectors.append(slice(None))
                    retained.append(dim)
                elif len(ds.dimensions[dim]) == 1:
                    selectors.append(0)
                else:
                    raise ValueError(f"{kind} has an unconfigured non-singleton dimension {dim!r}")
            values = np.ma.asarray(var[tuple(selectors)], dtype=np.float64).filled(np.nan)
            values = np.transpose(values, [retained.index(d) for d in dimensions])
            for axis, order in enumerate(orders):
                values = np.take(values, order, axis=axis)
            source_units = str(getattr(var, "units", ""))
            applied_units = spec.get("units_override", source_units)
            values, output_units, transform = _convert(values, kind, applied_units)
            values[~np.isfinite(values)] = np.nan
        reference = self._grid_reference.setdefault(kind, (lat, lon))
        if not np.array_equal(reference[0], lat) or not np.array_equal(reference[1], lon):
            raise ValueError(f"Source horizontal grid changes between records for {kind}")
        metadata = {"field": kind, "month": f"{year:04d}{month:02d}",
                    "source_units": source_units, "units_override": spec.get("units_override"),
                    "output_units": output_units, "unit_transform": transform,
                    "source_missing_values": int(np.sum(~np.isfinite(values)))}
        return values, lat, lon, pressure, metadata

    def _surface(self, year, month):
        date = (year, month)
        if date in self._ps_cache:
            self._ps_cache.move_to_end(date)
            return self._ps_cache[date]
        values, lat, lon, _, metadata = self._native("PS", year, month)
        nonpositive = int(np.sum(values <= 0))
        values[values <= 0] = np.nan
        target = strict_bilinear(values, lat, lon, self.target_lat, self.target_lon)
        metadata.update({"native_nonpositive_pressure_values": nonpositive,
                         "target_missing_values": int(np.sum(~np.isfinite(target)))})
        self.qa["records"].append(metadata)
        record = (values, lat, lon, target)
        self._ps_cache[date] = record
        while len(self._ps_cache) > 2:
            self._ps_cache.popitem(last=False)
        return record

    def read(self, kind, year, month):
        """Read one month; T/U/V/OMEGA use that same month's native PS mask."""
        if kind not in KINDS:
            raise ValueError(f"Unsupported field {kind!r}; expected one of {KINDS}")
        if isinstance(year, bool) or isinstance(month, bool) or int(year) != year or int(month) != month or not 1 <= year <= 9999 or not 1 <= month <= 12:
            raise ValueError("year/month must be integers in the supported calendar range")
        year, month = int(year), int(month)
        if kind == "PS":
            return self._surface(year, month)[3].copy(), None
        ps_native, ps_lat, ps_lon, ps_target = self._surface(year, month)
        values, lat, lon, pressure, metadata = self._native(kind, year, month)
        local_ps = strict_bilinear(ps_native, ps_lat, ps_lon, lat, lon)
        underground = (~np.isfinite(local_ps)[None, :, :]) | (pressure[:, None, None] > local_ps[None, :, :])
        values[underground] = np.nan
        output = strict_bilinear(values, lat, lon, self.target_lat, self.target_lon)
        target_underground = (~np.isfinite(ps_target)[None, :, :]) | (pressure[:, None, None] > ps_target[None, :, :])
        output[target_underground] = np.nan
        metadata.update({"native_underground_or_missing_ps_values": int(np.sum(underground)),
                         "target_underground_or_missing_ps_values": int(np.sum(target_underground)),
                         "target_missing_values": int(np.sum(~np.isfinite(output))),
                         "finite_target_values": int(np.sum(np.isfinite(output)))})
        self.qa["records"].append(metadata)
        return output, pressure.copy()
