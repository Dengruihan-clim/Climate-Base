"""Explicit decoded time validation and selection, without aggregation.

References
----------
CF 1.12 time/calendar conventions and xarray's public indexing/cftime model:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
https://docs.xarray.dev/en/stable/user-guide/weather-climate.html
https://docs.xarray.dev/en/stable/user-guide/indexing.html

Climate-Base improvements
-------------------------
This independent implementation rejects undecoded, missing, mixed-calendar or
unordered indexes. An explicit expected index detects missing endpoints as well
as interior records without inferring a frequency. Inclusive temporal selection
uses exact coordinate centers and preserves CF-related variables and source
ownership. Complete ISO strings mean precise instants, never partial-date
expansion. Fields remain Dask-backed; no calendar conversion or reduction occurs.
"""

from datetime import date, datetime
import re

import cftime
import numpy as np

from .exceptions import TimeCoordinateError
from .provenance import record_operation
from .validation import ensure_lazy, require_dataset

_ALIASES = {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}
_ISO = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?)?$")
_NUMPY_DATE = re.compile(r"^([+-]?\d{4,})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\.(\d{9})$")
_DESCRIPTIVE_RANGES = ("actual_range", "minimum_value", "maximum_value")


def _calendar_name(value):
    name = str(value).strip().lower()
    return _ALIASES.get(name, name)


def _declared_calendar(index):
    attrs = getattr(index, "attrs", {})
    encoding = getattr(index, "encoding", {})
    declared = [container["calendar"] for container in (attrs, encoding) if "calendar" in container]
    if len({_calendar_name(value) for value in declared}) > 1:
        raise TimeCoordinateError("Conflicting time calendar metadata")
    return _calendar_name(declared[0]) if declared else None


def _date_key(value):
    """Compare decoded labels without overflow from datetime64 unit promotion."""
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise TimeCoordinateError("Time coordinate contains NaT")
        if np.datetime_data(value.dtype)[0] in {"ps", "fs", "as"}:
            raise TimeCoordinateError("Time precision finer than nanoseconds is unsupported")
        match = _NUMPY_DATE.fullmatch(np.datetime_as_string(value, unit="ns"))
        if match is None:
            raise TimeCoordinateError("Unsupported decoded datetime64 label")
        return tuple(int(part) for part in match.groups())
    if not isinstance(value, (date, datetime, cftime.datetime)):
        raise TimeCoordinateError("Time coordinate must contain decoded datetime/cftime values")
    if not isinstance(value.year, (int, np.integer)):
        raise TimeCoordinateError("Time coordinate contains a missing date")
    if isinstance(value, datetime) and value.tzinfo is not None and value.utcoffset() is not None:
        raise TimeCoordinateError("Timezone-aware dates require an explicit timezone policy")
    return (int(value.year), int(value.month), int(value.day), int(getattr(value, "hour", 0)),
            int(getattr(value, "minute", 0)), int(getattr(value, "second", 0)),
            int(getattr(value, "microsecond", 0)) * 1000 + int(getattr(value, "nanosecond", 0)))


def _describe(values, *, declared=None, label="Time coordinate"):
    if np.ndim(values) != 1 or len(values) == 0:
        raise TimeCoordinateError(f"{label} must be a nonempty one-dimensional index")
    keys, kinds, calendars, year_zero = [], set(), set(), set()
    for value in values:
        if isinstance(value, cftime.datetime):
            kinds.add("cftime")
            calendars.add(_calendar_name(value.calendar))
            year_zero.add(value.has_year_zero)
        elif isinstance(value, (np.datetime64, date, datetime)):
            kinds.add("gregorian")
        else:
            raise TimeCoordinateError(f"{label} must be CF-decoded before validation")
        keys.append(_date_key(value))
    if len(kinds) != 1 or len(calendars) > 1 or len(year_zero) > 1:
        raise TimeCoordinateError("Mixed calendars or date representations are not accepted")
    kind = next(iter(kinds))
    if kind == "cftime":
        calendar = next(iter(calendars))
        if declared is not None and declared != calendar:
            raise TimeCoordinateError("Decoded dates and declared calendar differ")
    else:
        calendar = declared or "proleptic_gregorian"
        if calendar not in {"standard", "proleptic_gregorian"}:
            raise TimeCoordinateError("datetime64/Python dates cannot represent the declared calendar")
    if any(left >= right for left, right in zip(keys[:-1], keys[1:])):
        raise TimeCoordinateError(f"{label} must be strictly increasing without duplicates")
    return {"keys": keys, "kind": kind, "calendar": calendar,
            "has_year_zero": next(iter(year_zero)) if year_zero else None}


def _source_index(dataset, time):
    require_dataset(dataset)
    if not isinstance(time, str) or time not in dataset.coords or dataset[time].ndim != 1 or dataset[time].size == 0:
        raise TimeCoordinateError(f"A nonempty one-dimensional decoded {time!r} coordinate is required")
    coordinate = dataset[time]
    return _describe(np.asarray(coordinate.data), declared=_declared_calendar(coordinate))


def _precise_date(value, source, label):
    if isinstance(value, str):
        match = _ISO.fullmatch(value)
        if match is None:
            raise TimeCoordinateError(
                f"{label} requires YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS[.fraction]; "
                "partial dates and timezones are not accepted"
            )
        year, month, day, hour, minute, second, fraction = match.groups()
        parts = tuple(int(part or 0) for part in (year, month, day, hour, minute, second))
        nanosecond = int((fraction or "").ljust(9, "0"))
        try:
            if source["kind"] == "cftime":
                if nanosecond % 1000:
                    raise TimeCoordinateError("cftime endpoints cannot represent sub-microsecond precision")
                value = cftime.datetime(*parts, microsecond=nanosecond // 1000,
                                        calendar=source["calendar"], has_year_zero=source["has_year_zero"])
            else:
                value = np.datetime64(value.replace(" ", "T"))
                if _date_key(value) != (*parts, nanosecond):
                    raise TimeCoordinateError(f"{label} cannot be represented exactly as datetime64")
        except (TypeError, ValueError, OverflowError) as error:
            raise TimeCoordinateError(f"Invalid {label} for calendar {source['calendar']!r}: {error}") from error
    info = _describe([value], label=label)
    if info["kind"] != source["kind"]:
        raise TimeCoordinateError(f"{label} uses a different decoded calendar representation")
    if info["kind"] == "cftime" and (
        info["calendar"] != source["calendar"] or info["has_year_zero"] != source["has_year_zero"]
    ):
        raise TimeCoordinateError(f"{label} calendar differs from the source calendar")
    return value


def validate_time(dataset, *, time="time", expected=None):
    """Validate decoded ordering and, optionally, an exact expected time index.

    ``expected`` is a nonempty 1-D sequence/index/DataArray of decoded dates or
    complete ISO strings. Strings are parsed as exact instants in the source
    calendar; date-only strings mean midnight. Explicit calendar metadata and
    cftime calendars must agree. Unlabelled datetime64/Python expectations use
    the source's Gregorian representation; cftime and Gregorian types are not
    silently converted. Matching checks every label and both endpoints.

    Only time labels are read. Without ``expected``, an irregular increasing
    index remains valid: sampling frequency and missing periods are not inferred.
    """
    source = _source_index(dataset, time)
    if expected is None:
        return
    raw = np.asarray(expected.data if hasattr(expected, "dims") else expected)
    if raw.ndim != 1 or raw.size == 0:
        raise TimeCoordinateError("Expected time index must be a nonempty one-dimensional index")
    declared = _declared_calendar(expected)
    if declared is not None and declared != source["calendar"]:
        raise TimeCoordinateError("Expected time calendar differs from the source calendar")
    dates = [_precise_date(value, source, "Expected time") for value in raw]
    expectation = _describe(dates, declared=declared, label="Expected time index")
    if source["keys"] != expectation["keys"]:
        raise TimeCoordinateError(
            "Time coordinate does not match the explicit expected index "
            f"({len(source['keys'])} actual records, {len(expectation['keys'])} expected); "
            "missing, extra or shifted records, including endpoints, are not accepted"
        )


def _format_key(key):
    year, month, day, hour, minute, second, nanosecond = key
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}.{nanosecond:09d}"


def subset_time(dataset, *, start=None, end=None, time="time"):
    """Select inclusive decoded coordinate centers in an exact time interval.

    Bounds accept decoded scalar dates or complete ISO strings (YYYY-MM-DD or
    YYYY-MM-DDTHH:MM:SS[.fraction]). A date means midnight, including for ``end``;
    it does not expand to the whole day/month. ``None`` leaves that side open.
    Invalid/reversed ranges and empty selections raise ``TimeCoordinateError``.

    All variables sharing the time dimension, including bounds and auxiliary
    coordinates, are sliced together. Source attributes and CF links remain;
    stale descriptive ranges on cropped variables are removed and audited,
    without recalculating extrema. Returned fields are lazy, and closing the
    result closes the source. No calendar conversion or aggregation occurs.
    """
    source = _source_index(dataset, time)
    lower = _date_key(_precise_date(start, source, "start")) if start is not None else None
    upper = _date_key(_precise_date(end, source, "end")) if end is not None else None
    if lower is not None and upper is not None and lower > upper:
        raise TimeCoordinateError("start must not be later than end")
    selected = [index for index, key in enumerate(source["keys"])
                if (lower is None or key >= lower) and (upper is None or key <= upper)]
    if not selected:
        raise TimeCoordinateError("Requested time range contains no coordinate centers")
    dimension = dataset[time].dims[0]
    result = ensure_lazy(dataset.isel({dimension: slice(selected[0], selected[-1] + 1)}))
    removed = {}
    if len(selected) != len(source["keys"]):
        for name, variable in result.variables.items():
            if dimension not in variable.dims:
                continue
            variable.attrs = dict(variable.attrs)
            variable.encoding = dict(variable.encoding)
            locations = {}
            for location, container in (("attrs", variable.attrs), ("encoding", variable.encoding)):
                keys = [key for key in _DESCRIPTIVE_RANGES if key in container]
                for key in keys:
                    del container[key]
                if keys:
                    locations[location] = keys
            if locations:
                removed[name] = locations
    result.set_close(dataset.close)
    return record_operation(result, "subset_time", {
        "time": time, "start": _format_key(lower) if lower is not None else None,
        "end": _format_key(upper) if upper is not None else None,
        "closed": "both", "calendar": source["calendar"], "selected_records": len(selected),
        "removed_descriptive_ranges": removed,
    })
