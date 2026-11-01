"""Compatibility seasonal reductions using xarray/Dask and the CF calendar.

References
----------
Season definitions and complete-cell rules are the author's Climate-Base 0.1.0
methods. Array alignment and lazy reduction use xarray's public API:
https://docs.xarray.dev/en/stable/user-guide/dask.html
https://docs.xarray.dev/en/stable/generated/xarray.concat.html

Climate-Base improvements
-------------------------
Monthly fields remain deferred through weighting and completeness checks, then
one season is computed at the existing ndarray return boundary. All input
resources close even if selection or computation fails. This is an internal
execution refactor of the existing compatibility API, not a new statistical
feature, and no upstream algorithm was copied.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import xarray as xr
from ._validation import checked_integer


def _months(months):
    months = tuple(months)
    if not 1 <= len(months) <= 12:
        raise ValueError("Season months must contain one to twelve integer months in 1..12")
    months = tuple(checked_integer(month, "season month", 1, 12) for month in months)
    if len(set(months)) != len(months) or any(b != a % 12 + 1 for a, b in zip(months, months[1:])):
        raise ValueError("Season months must be contiguous and nonrepeated")
    return months


def season_dates(start_year, end_year, months=(12, 1, 2)):
    """Return (season_year, calendar_year, month), labelled by the final month."""
    months = _months(months)
    start_year = checked_integer(start_year, "start_year", 1, 9999)
    end_year = checked_integer(end_year, "end_year", 1, 9999)
    if start_year > end_year:
        raise ValueError("Season years must satisfy 1 <= start_year <= end_year <= 9999")
    dates = [(int(year), int(year) - int(month > months[-1]), int(month))
             for year in range(int(start_year), int(end_year) + 1) for month in months]
    if any(y < 1 for _, y, _ in dates):
        raise ValueError("Requested season requires calendar year zero")
    return dates


def month_days(year, month, calendar="standard"):
    """Actual calendar-month duration, including leap/nonstandard calendars."""
    import cftime
    year = checked_integer(year, "year", 1, 9999)
    month = checked_integer(month, "month", 1, 12)
    next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
    endpoints = [cftime.datetime(int(year), int(month), 1, calendar=calendar),
                 cftime.datetime(int(next_year), int(next_month), 1, calendar=calendar)]
    values = cftime.date2num(endpoints, "days since 0001-01-01", calendar=calendar)
    return float(values[1] - values[0])


@dataclass
class SeasonalField:
    values: np.ndarray
    years: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray
    pressure: np.ndarray | None
    units: str
    calendar: str
    months: tuple[int, ...]
    weighting: str


def iter_seasons(archive, name, start_year, end_year, *, months=(12, 1, 2), weighting="days"):
    """Yield one complete season at a time, never filling partial cells.

    Missing input months raise.  A cell is finite only when every season
    month is finite.  ``days`` is appropriate for monthly mean rates/states;
    ``equal`` must be chosen explicitly for equal-month or month-end states.
    Monthly accumulated quantities need a separate, explicitly chosen sum;
    this API intentionally does not infer or perform it.
    """
    if weighting not in {"days", "equal"}:
        raise ValueError("weighting must be days or equal")
    months = _months(months)
    dates = season_dates(start_year, end_year, months)
    for position in range(0, len(dates), len(months)):
        season = dates[position:position + len(months)]
        monthly, weights = [], []
        try:
            for _, year, month in season:
                current = archive.read_lazy(name, year, month)
                monthly.append(current)
                weights.append(month_days(year, month, current.attrs["calendar"])
                               if weighting == "days" else 1.)
            first = monthly[0]
            series = xr.concat(
                monthly, dim=xr.IndexVariable("season_month", np.arange(len(months))),
                coords="minimal", compat="equals", join="exact",
            )
            weight_array = xr.DataArray(weights, dims="season_month").chunk({"season_month": 1})
            finite = xr.apply_ufunc(np.isfinite, series, dask="allowed")
            total = (series.where(finite, 0.) * weight_array).sum("season_month", skipna=False)
            reduced = (total / sum(weights)).where(finite.all("season_month"))
            # This compute is intentional: SeasonalField retains its historical
            # ndarray contract while all field arithmetic above stays deferred.
            values = reduced.compute().data
            result = SeasonalField(
                values[None], np.array([season[0][0]], dtype=np.int32),
                first.coords["latitude"].data.copy(), first.coords["longitude"].data.copy(),
                first.coords["pressure"].data.copy() if "pressure" in first.dims else None,
                first.attrs["units"], first.attrs["calendar"], months, weighting,
            )
        finally:
            for current in monthly:
                current.close()
        yield result


def seasonal_mean(archive, name, start_year, end_year, *, months=(12, 1, 2), weighting="days"):
    """Return a seasonal series; use iter_seasons for bounded output memory."""
    records = list(iter_seasons(archive, name, start_year, end_year, months=months, weighting=weighting))
    first = records[0]
    return SeasonalField(np.concatenate([record.values for record in records], axis=0),
                         np.concatenate([record.years for record in records]),
                         first.latitude, first.longitude, first.pressure, first.units,
                         first.calendar, first.months, first.weighting)
