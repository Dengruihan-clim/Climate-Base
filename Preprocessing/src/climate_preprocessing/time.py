"""Decoded time-coordinate validation without temporal aggregation.

References: CF 1.12 time and calendar conventions, and xarray's cftime model:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
https://docs.xarray.dev/en/stable/user-guide/weather-climate.html

Climate-Base improvements: numeric undecoded times, NaT, mixed calendars,
duplicates and non-increasing indexes fail explicitly. Field graphs are untouched.
This independent implementation deliberately does not infer sampling frequency.
"""

from datetime import date, datetime

import numpy as np

from .exceptions import TimeCoordinateError
from .validation import require_dataset


def validate_time(dataset, *, time="time"):
    """Validate a nonempty, one-dimensional, strictly increasing decoded time.

    Supports numpy datetime64 and cftime-like calendar dates. Only the time
    coordinate is loaded. Missing sampling periods are not inferred: an irregular
    but increasing series is valid. Calendar conversion is never automatic.
    """
    require_dataset(dataset)
    if time not in dataset.coords or dataset[time].ndim != 1 or dataset[time].size == 0:
        raise TimeCoordinateError(f"A nonempty one-dimensional decoded {time!r} coordinate is required")
    values = np.asarray(dataset[time].data)
    if np.issubdtype(values.dtype, np.datetime64):
        if np.isnat(values).any():
            raise TimeCoordinateError("Time coordinate contains NaT")
    elif values.dtype.kind == "O":
        calendars = set()
        aliases = {"gregorian": "standard", "365_day": "noleap", "366_day": "all_leap"}
        for value in values:
            if not isinstance(value, (date, datetime)) and not (
                hasattr(value, "calendar") and hasattr(value, "year") and hasattr(value, "month")
            ):
                raise TimeCoordinateError("Time coordinate must contain decoded datetime/cftime values")
            if not isinstance(value.year, (int, np.integer)):
                raise TimeCoordinateError("Time coordinate contains a missing date")
            calendar = str(getattr(value, "calendar", "proleptic_gregorian")).lower()
            calendars.add(aliases.get(calendar, calendar))
        if len(calendars) != 1:
            raise TimeCoordinateError("Mixed calendars are not accepted")
    else:
        raise TimeCoordinateError("Time coordinate must be CF-decoded before validation")
    try:
        increasing = all(left < right for left, right in zip(values[:-1], values[1:]))
    except (TypeError, ValueError) as error:
        raise TimeCoordinateError("Time values are not mutually comparable") from error
    if not increasing:
        raise TimeCoordinateError("Time coordinate must be strictly increasing without duplicates")
