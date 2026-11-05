"""Check decoded date precision at an explicit output execution boundary.

References
----------
CF time/calendar conventions and xarray's public CF encoding/decoding APIs:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
https://docs.xarray.dev/en/stable/user-guide/time-series.html

Climate-Base improvements
-------------------------
An integer storage dtype alone does not prove a date roundtrip: supported
xarray versions can take floating intermediates during encoding or decoding.
This independent check compares decoded date labels before publishing output.
Only explicitly identified date/time arrays are read, with a fixed entry limit;
meteorological numeric fields and arbitrary object-valued fields are untouched.
"""

from datetime import date, datetime

import cftime
import numpy as np

from ..time import _date_key, _declared_calendar
from ..validation import require_dataset

_MAX_ENTRIES = 1_000_000
_CALENDARS = frozenset({"standard", "proleptic_gregorian", "noleap", "all_leap", "360_day", "julian"})
_ALIASES = {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}


def _decoded_time(variable):
    return variable.dtype.kind in "MO" and any(
        "since" in str(container.get("units", "")) for container in (variable.attrs, variable.encoding)
    )


def _missing(value):
    if isinstance(value, np.datetime64):
        return bool(np.isnat(value))
    if value is None:
        return True
    # cftime arrays may represent missing timestamps with an object NaN.
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    # pandas.NaT is a datetime subclass with missing year metadata, whereas
    # a valid Python/cftime date has an integer year.
    return isinstance(value, (date, datetime)) and not isinstance(value.year, (int, np.integer))


def _labels_and_calendar(variable):
    values = variable.to_numpy().ravel()
    keys, calendars, year_zero = [], set(), set()
    declared = _declared_calendar(variable)
    for value in values:
        if _missing(value):
            keys.append(None)
            continue
        keys.append(_date_key(value))
        if isinstance(value, cftime.datetime):
            name = str(value.calendar).lower()
            calendars.add(_ALIASES.get(name, name))
            year_zero.add(bool(value.has_year_zero))
        elif isinstance(value, (np.datetime64, date, datetime)):
            calendars.add(declared or "proleptic_gregorian")
        else:
            raise ValueError("Output date check encountered a value that is not a decoded datetime")
    if len(calendars) > 1 or len(year_zero) > 1:
        raise ValueError("Output date check does not accept mixed calendars or year-zero conventions")
    inferred = next(iter(calendars)) if calendars else None
    if declared is not None and inferred is not None and declared != inferred:
        raise ValueError("Declared output calendar differs from decoded date values")
    calendar = declared or inferred or ("proleptic_gregorian" if variable.dtype.kind == "M" else None)
    if calendar not in _CALENDARS:
        raise ValueError("Output date check requires a known, unambiguous calendar")
    return keys, calendar, next(iter(year_zero)) if year_zero else None


def _compatible_calendars(left, right, labels):
    if left == right:
        return True
    if {left, right} == {"standard", "proleptic_gregorian"}:
        # Gregorian dates agree after the calendar reform. Earlier dates have
        # distinct physical meanings even when their year/month/day labels match.
        return all(key is None or key[:3] >= (1582, 10, 15) for key in labels)
    return False


def validate_time_roundtrip(source, reopened):
    """Reject altered decoded time labels/calendars before publishing a file.

    Compare source/reopened datetime variables identified by reopened datetime
    dtype and CF ``units`` containing ``since``. Source variables must themselves
    have datetime64/object dtype; numeric fields are never read or converted.
    Datetime64 source variables must retain their CF date representation.

    This function is only for an explicit write/verification boundary. It loads
    at most 1,000,000 entries per date/time/bounds/valid-time variable, rejects
    larger date arrays, and compares exact nanosecond/microsecond date tuples.
    Matching missing dates (including NaT) are allowed. Calendar aliases are
    respected; standard/proleptic Gregorian are compatible only for dates on or
    after 1582-10-15. Other calendar changes are rejected. Inputs remain open and
    unchanged. Return None on success; raise ValueError before output publication
    if a backend lost precision or changed time semantics.
    """
    require_dataset(source)
    require_dataset(reopened)
    for name, original in source.variables.items():
        if original.dtype.kind == "M" and (name not in reopened.variables or not _decoded_time(reopened[name])):
            raise ValueError(f"Output lost the decoded CF datetime representation of {name!r}")
    for name, output in reopened.variables.items():
        if not _decoded_time(output) or name not in source.variables:
            continue
        original = source[name]
        if original.dtype.kind not in "MO":
            continue
        if original.dims != output.dims or original.shape != output.shape:
            raise ValueError(f"Output time dimensions changed for {name!r}")
        if original.size > _MAX_ENTRIES:
            raise ValueError(f"Output time verification for {name!r} exceeds the {_MAX_ENTRIES}-entry limit")
        expected, source_calendar, source_year_zero = _labels_and_calendar(original)
        actual, output_calendar, output_year_zero = _labels_and_calendar(reopened[name])
        if not _compatible_calendars(source_calendar, output_calendar, expected):
            raise ValueError(f"Output calendar changed for {name!r}: {source_calendar} -> {output_calendar}")
        if (source_year_zero is not None and output_year_zero is not None
                and source_year_zero != output_year_zero
                and any(key is not None and key[0] <= 0 for key in expected)):
            raise ValueError(f"Output changed the year-zero convention for {name!r}")
        if expected != actual:
            raise ValueError(f"Output time precision or missing-date pattern changed for {name!r}")
