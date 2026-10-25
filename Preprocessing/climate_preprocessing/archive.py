"""Strict, streaming monthly CF NetCDF input on native rectilinear grids."""
from __future__ import annotations
from dataclasses import dataclass, field
import glob
import hashlib
from pathlib import Path
from collections.abc import Mapping, Sequence
import numpy as np
from .units import conversion_plan, same_units, unit_key, _apply_affine
from ._validation import checked_integer, checked_bool


@dataclass(frozen=True)
class Coordinates:
    time: str = "time"
    latitude: str = "lat"
    longitude: str = "lon"
    pressure: str | None = None


@dataclass(frozen=True)
class FieldSpec:
    files: str | Path | Sequence[str | Path]
    variable: str
    coordinates: Coordinates = field(default_factory=Coordinates)
    expected_units: str | Sequence[str] | None = None
    target_units: str | None = None
    units_override: str | None = None
    scale: float | None = None
    offset: float = 0.


@dataclass
class MonthlyField:
    values: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    pressure: np.ndarray | None
    units: str
    calendar: str
    source_calendar: str
    year: int
    month: int
    metadata: dict


def _backend():
    try:
        from netCDF4 import Dataset, num2date
    except ImportError as error:
        raise ImportError("Monthly CF reading requires netCDF4; install the declared dependencies") from error
    return Dataset, num2date


def _calendar(source, year, policy):
    source = str(source).strip().lower()
    canonical = {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}.get(source, source)
    if policy == "modern_gregorian" and year >= 1583 and canonical in {"standard", "proleptic_gregorian"}:
        return "proleptic_gregorian"
    return canonical


def _axis(ds, name, meaning):
    if name not in ds.variables or ds.variables[name].ndim != 1:
        raise ValueError(f"Missing one-dimensional {meaning} coordinate {name!r}")
    var = ds.variables[name]
    values = np.ma.asarray(var[:], dtype=np.float64).filled(np.nan)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"{meaning} coordinate is empty or nonfinite")
    units = str(getattr(var, "units", ""))
    if meaning == "latitude" and unit_key(units) not in {"degreesnorth", "degreenorth", "degreesn", "degreen"}:
        raise ValueError("Latitude units must explicitly indicate degrees_north")
    if meaning == "longitude" and unit_key(units) not in {"degreeseast", "degreeeast", "degreese", "degreee"}:
        raise ValueError("Longitude units must explicitly indicate degrees_east")
    return values, var.dimensions[0], units


class MonthlyArchive:
    """Index configured local monthly files and read one field record at a time.

    Each file must contain strictly increasing CF times, at most one record
    per calendar month, rectilinear 1-D coordinates, and explicit field units.
    Coordinates and output units are fixed within a field over all reads.
    ``shared_grid=True`` also requires all fields to share horizontal and
    pressure coordinates.  No arrays of field data are cached.
    """

    def __init__(self, fields: Mapping[str, FieldSpec | dict], *, data_root=".",
                 longitude="0_360", latitude_order="ascending", pressure_units="hPa",
                 pressure_order="ascending", require_global_longitude=False,
                 shared_grid=False, calendar_policy="strict", hash_inputs=True):
        if not fields:
            raise ValueError("At least one configured field is required")
        if longitude not in {"0_360", "-180_180"} or latitude_order not in {"ascending", "descending"}:
            raise ValueError("Unsupported longitude convention or latitude_order")
        if pressure_units not in {"Pa", "hPa"} or pressure_order not in {"ascending", "descending"}:
            raise ValueError("pressure_units must be Pa/hPa and pressure_order ascending/descending")
        if calendar_policy not in {"strict", "modern_gregorian"}:
            raise ValueError("calendar_policy must be strict or modern_gregorian")
        self.data_root = Path(data_root).expanduser().resolve()
        self.fields = {}
        for name, spec in fields.items():
            if isinstance(spec, dict):
                spec = dict(spec)
                spec["coordinates"] = Coordinates(**spec.get("coordinates", {}))
                spec = FieldSpec(**spec)
            if not isinstance(spec, FieldSpec) or not spec.variable or not isinstance(spec.coordinates, Coordinates):
                raise ValueError(f"Invalid field specification {name!r}")
            self.fields[name] = spec
        self.longitude_convention = longitude
        self.latitude_order = latitude_order
        self.pressure_units = pressure_units
        self.pressure_order = pressure_order
        self.require_global_longitude = checked_bool(require_global_longitude, "require_global_longitude")
        self.shared_grid = checked_bool(shared_grid, "shared_grid")
        self.calendar_policy = calendar_policy
        self.hash_inputs = checked_bool(hash_inputs, "hash_inputs")
        self.calendar = None
        self.audit = {"calendar_policy": calendar_policy, "source_calendars": [], "unit_transforms": {}}
        self._time_cache = {}
        self._manifest = {}
        self._reference = {}
        self._shared_reference = None
        self._shared_pressure = None

    @property
    def inventory(self):
        return [dict(value) for value in self._manifest.values()]

    def _stamp(self, path):
        info = path.stat()
        return info.st_size, info.st_mtime_ns

    def _times(self, path, name):
        key, stamp = (path, name), self._stamp(path)
        if key in self._time_cache:
            cached = self._time_cache[key]
            if cached[0] != stamp:
                raise ValueError("An input file changed after indexing; construct a new archive")
            return cached[1], cached[2]
        Dataset, num2date = _backend()
        with Dataset(path) as ds:
            if name not in ds.variables or ds.variables[name].ndim != 1:
                raise ValueError("A nonempty one-dimensional CF time coordinate is required")
            var = ds.variables[name]
            raw = np.ma.asarray(var[:], dtype=np.float64).filled(np.nan)
            if not raw.size or not np.isfinite(raw).all() or np.any(np.diff(raw) <= 0):
                raise ValueError("CF time must be finite and strictly increasing within each file")
            if not getattr(var, "units", ""):
                raise ValueError("CF time coordinate is missing units")
            source_calendar = str(getattr(var, "calendar", "standard")).strip().lower()
            decoded = num2date(raw, var.units, calendar=source_calendar)
            dates = tuple((int(d.year), int(d.month)) for d in np.atleast_1d(decoded))
            if len(set(dates)) != len(dates):
                raise ValueError("Input is not monthly: repeated CF calendar months within one file")
        self._time_cache[key] = stamp, dates, source_calendar
        return dates, source_calendar

    def _select(self, field_name, year, month):
        if field_name not in self.fields:
            raise KeyError(f"Unknown configured field {field_name!r}")
        year = checked_integer(year, "year", 1, 9999)
        month = checked_integer(month, "month", 1, 12)
        spec = self.fields[field_name]
        patterns = (spec.files,) if isinstance(spec.files, (str, Path)) else tuple(spec.files)
        candidates = set()
        for pattern in patterns:
            try:
                rendered = str(pattern).format(data_root=self.data_root.as_posix(), year=year,
                                               month=month, yyyymm=f"{year:04d}{month:02d}")
            except (KeyError, ValueError) as error:
                raise ValueError("Invalid files template; supported keys: data_root/year/month/yyyymm") from error
            candidate = Path(rendered).expanduser()
            if not candidate.is_absolute():
                candidate = self.data_root / candidate
            candidates.update(Path(value).resolve() for value in glob.glob(str(candidate)) if Path(value).is_file())
        matches = []
        for path in sorted(candidates):
            dates, source_calendar = self._times(path, spec.coordinates.time)
            matches.extend((path, index, source_calendar) for index, date in enumerate(dates) if date == (year, month))
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one CF monthly record for {field_name} {year:04d}{month:02d}; found {len(matches)}")
        path, index, source_calendar = matches[0]
        calendar = _calendar(source_calendar, year, self.calendar_policy)
        if self.calendar is None:
            self.calendar = calendar
        elif self.calendar != calendar:
            raise ValueError("Mixed incompatible calendars; use separate archives or explicit modern_gregorian policy for modern Gregorian aliases")
        if source_calendar not in self.audit["source_calendars"]:
            self.audit["source_calendars"].append(source_calendar)
        return spec, path, index, source_calendar, calendar

    def _register(self, path, field_name, year, month, index):
        stamp = self._stamp(path)
        if path not in self._manifest:
            try:
                label = path.relative_to(self.data_root).as_posix()
            except ValueError:
                label = f"external_input_{len(self._manifest) + 1:04d}/{path.name}"
            digest = None
            if self.hash_inputs:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                if self._stamp(path) != stamp:
                    raise ValueError("Input changed during provenance hashing")
            self._manifest[path] = {"path": label, "bytes": stamp[0], "sha256": digest, "records": []}
        record = {"field": field_name, "year": int(year), "month": int(month), "time_index": index}
        if record not in self._manifest[path]["records"]:
            self._manifest[path]["records"].append(record)

    def _layout(self, ds, spec, field_name, index):
        coords = spec.coordinates
        if spec.variable not in ds.variables:
            raise ValueError(f"Missing configured variable {spec.variable!r}")
        variable = ds.variables[spec.variable]
        lat, lat_dim, _ = _axis(ds, coords.latitude, "latitude")
        lon, lon_dim, _ = _axis(ds, coords.longitude, "longitude")
        if lat.size < 2 or np.any(np.abs(lat) > 90):
            raise ValueError("At least two valid latitude coordinates are required")
        y_order = np.argsort(lat)
        if self.latitude_order == "descending":
            y_order = y_order[::-1]
        lat = lat[y_order]
        if np.any(np.diff(np.sort(lat)) <= 0):
            raise ValueError("Latitude coordinates must be distinct")
        lon = lon % 360.0 if self.longitude_convention == "0_360" else (lon + 180.0) % 360.0 - 180.0
        x_order = np.argsort(lon)
        lon = lon[x_order]
        if lon.size < 2 or np.any(np.diff(lon) <= 1e-9):
            raise ValueError("Longitude coordinates must be distinct without a repeated cyclic endpoint")
        if self.require_global_longitude:
            gaps = np.diff(np.r_[lon, lon[0] + 360.0])
            if lon.size < 4 or gaps.max() > 1.5 * np.median(gaps) + 1e-8:
                raise ValueError("Global longitude coverage is required; a large cyclic gap was found")
        dimensions, orders = [lat_dim, lon_dim], [y_order, x_order]
        pressure = None
        if coords.pressure is not None:
            pressure, p_dim, p_units = _axis(ds, coords.pressure, "pressure")
            factor, shift, _ = conversion_plan(p_units, self.pressure_units)
            # Only true pressure dimensions, not arbitrary affine conversions.
            if unit_key(p_units) not in {"pa", "pascal", "pascals", "hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars"}:
                raise ValueError("Pressure coordinate units must be Pa or hPa")
            if str(getattr(ds.variables[coords.pressure], "positive", "down")).lower() != "down":
                raise ValueError("Pressure positive direction must be down")
            pressure = _apply_affine(pressure, factor, shift)
            p_order = np.argsort(pressure)
            if self.pressure_order == "descending":
                p_order = p_order[::-1]
            pressure = pressure[p_order]
            if np.any(pressure <= 0) or np.any(np.diff(np.sort(pressure)) <= 0):
                raise ValueError("Pressure coordinates must be distinct and positive")
            dimensions.insert(0, p_dim)
            orders.insert(0, p_order)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("Configured spatial coordinates must use distinct dimensions")
        time_dim = ds.variables[coords.time].dimensions[0]
        if time_dim not in variable.dimensions or time_dim in dimensions or any(d not in variable.dimensions for d in dimensions):
            raise ValueError("Variable dimensions do not match the configured time/spatial coordinates")
        selectors, retained = [], []
        for dim in variable.dimensions:
            if dim == time_dim:
                selectors.append(index)
            elif dim in dimensions:
                selectors.append(slice(None))
                retained.append(dim)
            elif len(ds.dimensions[dim]) == 1:
                selectors.append(0)
            else:
                raise ValueError(f"Unselected non-singleton dimension {dim!r}; preprocess/select it explicitly")
        source_units = str(getattr(variable, "units", "")).strip()
        applied_units = spec.units_override if spec.units_override is not None else source_units
        if spec.expected_units is not None:
            allowed = (spec.expected_units,) if isinstance(spec.expected_units, str) else spec.expected_units
            if not any(same_units(applied_units, value) for value in allowed):
                raise ValueError(f"Source units for {field_name} do not match the declared expected_units")
        factor, shift, output_units = conversion_plan(applied_units, spec.target_units, scale=spec.scale, offset=spec.offset)
        reference = (lat.copy(), lon.copy(), None if pressure is None else pressure.copy(), output_units)
        if field_name in self._reference:
            self._same_grid(self._reference[field_name], reference, check_units=True)
        else:
            self._reference[field_name] = reference
        if self.shared_grid:
            if self._shared_reference is not None:
                if not np.array_equal(self._shared_reference[0], lat) or not np.array_equal(self._shared_reference[1], lon):
                    raise ValueError("Fields do not share the same normalized horizontal grid")
            else:
                self._shared_reference = reference
            if pressure is not None:
                if self._shared_pressure is None:
                    self._shared_pressure = pressure.copy()
                elif not np.array_equal(self._shared_pressure, pressure):
                    raise ValueError("Three-dimensional fields do not share the same pressure levels")
        transform = {"source_units": source_units, "units_override": spec.units_override,
                     "target_units": output_units, "scale": factor, "offset": shift}
        transforms = self.audit["unit_transforms"].setdefault(field_name, [])
        if transform not in transforms:
            transforms.append(transform)
        return variable, tuple(selectors), retained, dimensions, orders, reference, transform

    @staticmethod
    def _same_grid(a, b, *, check_units):
        for left, right in zip(a[:3], b[:3]):
            if (left is None) != (right is None) or (left is not None and not np.array_equal(left, right)):
                raise ValueError("Normalized horizontal grid or pressure levels changed; no automatic regridding/intersection is allowed")
        if check_units and not same_units(a[3], b[3]):
            raise ValueError("Output units changed between records; specify a consistent target_units")

    def _record(self, field_name, year, month, *, data):
        spec, path, index, source_calendar, calendar = self._select(field_name, year, month)
        Dataset, _ = _backend()
        stamp = self._stamp(path)
        if stamp != self._time_cache[(path, spec.coordinates.time)][0]:
            raise ValueError("Input changed after time selection")
        with Dataset(path) as ds:
            variable, selectors, retained, dimensions, orders, reference, transform = self._layout(ds, spec, field_name, index)
            values = None
            if data:
                raw = variable[selectors]
                if np.iscomplexobj(raw):
                    raise ValueError("Monthly field values must be real")
                values = np.ma.asarray(raw, dtype=np.float64).filled(np.nan)
                values = np.transpose(values, [retained.index(d) for d in dimensions])
                for axis, order in enumerate(orders):
                    values = np.take(values, order, axis=axis)
                values = _apply_affine(values, transform["scale"], transform["offset"])
        if self._stamp(path) != stamp:
            raise ValueError("Input changed during reading")
        self._register(path, field_name, year, month, index)
        lat, lon, pressure, units = reference
        metadata = {"field": field_name, "year": int(year), "month": int(month),
                    "calendar": calendar, "source_calendar": source_calendar, "unit_transform": transform,
                    "pressure_units": self.pressure_units if pressure is not None else None}
        if not data:
            return {**metadata, "shape": ([len(pressure)] if pressure is not None else []) + [len(lat), len(lon)],
                    "latitude_bounds": [float(lat.min()), float(lat.max())],
                    "longitude_bounds": [float(lon.min()), float(lon.max())],
                    "pressure": None if pressure is None else pressure.tolist(), "field_values_read": False}
        return MonthlyField(values, lat.copy(), lon.copy(), None if pressure is None else pressure.copy(),
                            units, calendar, source_calendar, int(year), int(month), metadata)

    def inspect(self, name, year, month):
        """Validate selected metadata/time/grid/units without reading field data."""
        return self._record(name, year, month, data=False)

    def read(self, name, year, month):
        """Return one selected monthly field; dates always come from CF time."""
        return self._record(name, year, month, data=True)
