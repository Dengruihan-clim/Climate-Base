"""Explicit sampling contracts and precise, lazy time-coordinate selection."""

from datetime import datetime, timezone
import json

import cftime
import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.exceptions import TimeCoordinateError
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE
from climate_preprocessing.time import subset_time, validate_time


def _dataset(times=None):
    if times is None:
        times = np.array(["2001-01-01T00:00:00", "2001-01-01T12:00:00",
                          "2001-01-02T00:00:00", "2001-01-02T12:00:00"], dtype="datetime64[ns]")
    count = len(times)
    source = xr.Dataset(
        {"V": (("time", "x"), da.arange(count * 2, chunks=2).reshape((count, 2)),
               {"units": "m s-1", "cell_methods": "time: point", "actual_range": [0, count * 2 - 1],
                "minimum_value": 0, "maximum_value": count * 2 - 1, "valid_range": [-100, 100]}),
         "quality": ("time", da.arange(count, chunks=2))},
        coords={"time": times, "x": [0, 1]},
        attrs={"history": "original observations", "source": "test fixture"},
    )
    return source


def test_expected_index_is_optional_and_never_infers_frequency():
    dates = np.array(["2000-01-01", "2000-01-07", "2000-01-10"], dtype="datetime64[ns]")
    source = _dataset(dates)
    validate_time(source)
    validate_time(source, expected=dates)
    validate_time(source, expected=["2000-01-01", "2000-01-07", "2000-01-10"])
    validate_time(source, expected=source.get_index("time"))


@pytest.mark.parametrize("indices", [[1, 2], [0, 1], [0, 2]])
def test_explicit_index_rejects_missing_first_last_or_interior_record(indices):
    dates = np.array(["2000-01-01", "2000-02-01", "2000-03-01"], dtype="datetime64[ns]")
    source = _dataset(dates[indices])
    validate_time(source)
    with pytest.raises(TimeCoordinateError, match="expected index"):
        validate_time(source, expected=dates)


@pytest.mark.parametrize("expected", [
    ["2001-01-01", "2001-01-02"],
    ["2001-01-01", "2001-01-01T12:00:00", "2001-01-02", "2001-01-02T13:00:00"],
])
def test_expected_index_rejects_extra_or_shifted_records(expected):
    with pytest.raises(TimeCoordinateError, match="expected index"):
        validate_time(_dataset(), expected=expected)


@pytest.mark.parametrize("expected", [[], "2001-01-01", [["2001-01-01"]],
                                      ["2001-01-01", "2001-01-01"],
                                      ["2001-01-02", "2001-01-01"], [0, 1]])
def test_invalid_expected_index_is_rejected(expected):
    with pytest.raises(TimeCoordinateError):
        validate_time(_dataset(), expected=expected)


def test_exact_midnight_end_preserves_related_variables_source_and_laziness():
    source = _dataset()
    dates = source.time.values
    bounds = np.column_stack([dates, dates + np.timedelta64(12, "h")])
    source = source.assign_coords(time_bounds=(("time", "bnds"), bounds))
    source.time.attrs["bounds"] = "time_bounds"
    source.time.encoding["calendar"] = "gregorian"
    source.V.encoding["actual_range"] = np.array([0, 7])
    closed, tasks = [], []
    source.set_close(lambda: closed.append(True))
    with Callback(pretask=lambda *args: tasks.append(args)):
        validate_time(source, expected=["2001-01-01", "2001-01-01T12:00:00",
                                        "2001-01-02", "2001-01-02T12:00:00"])
        result = subset_time(source, start="2001-01-01", end="2001-01-02")
    assert not tasks
    assert result.sizes["time"] == 3  # midnight end does not include the day's later record
    assert isinstance(result.V.data, da.Array)
    assert isinstance(result.quality.data, da.Array)
    assert result.attrs["history"] == source.attrs["history"]
    assert result.attrs["source"] == source.attrs["source"]
    assert result.time.attrs["bounds"] == "time_bounds"
    assert result.time.encoding["calendar"] == "gregorian"
    np.testing.assert_array_equal(result.time_bounds, bounds[:3])
    assert result.V.attrs["units"] == "m s-1"
    assert result.V.attrs["cell_methods"] == "time: point"
    assert result.V.attrs["valid_range"] == [-100, 100]
    assert not {"actual_range", "minimum_value", "maximum_value"} & result.V.attrs.keys()
    assert "actual_range" not in result.V.encoding
    assert source.V.attrs["actual_range"] == [0, 7]
    np.testing.assert_array_equal(source.V.encoding["actual_range"], [0, 7])
    np.testing.assert_array_equal(result.V.compute(), np.arange(6).reshape(3, 2))
    np.testing.assert_array_equal(result.quality.compute(), [0, 1, 2])
    event = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]
    assert event["operation"] == "subset_time"
    assert event["parameters"]["closed"] == "both"
    assert event["parameters"]["end"] == "2001-01-02T00:00:00.000000000"
    assert event["parameters"]["removed_descriptive_ranges"]["V"]["encoding"] == ["actual_range"]
    assert closed == []
    result.close()
    assert closed == [True]


@pytest.mark.parametrize("start,end,indices", [
    (None, None, [0, 1, 2, 3]),
    ("2001-01-02", None, [2, 3]),
    (None, "2001-01-01", [0]),
    ("2001-01-01T12:00:00", "2001-01-01T12:00:00", [1]),
])
def test_closed_and_one_sided_time_intervals(start, end, indices):
    source = _dataset()
    with subset_time(source, start=start, end=end) as result:
        np.testing.assert_array_equal(result.time, source.time.values[indices])
        if len(indices) == source.sizes["time"]:
            assert result.V.attrs["actual_range"] == source.V.attrs["actual_range"]


@pytest.mark.parametrize("bound", ["2001", "2001-01", "2001-01-01T12:00", "2001-01-01Z",
                                    "2001-01-01T00:00:00+00:00", "2001-02-30"])
def test_partial_timezone_and_invalid_iso_bounds_never_expand_selection(bound):
    with pytest.raises(TimeCoordinateError):
        subset_time(_dataset(), start=bound)
    with pytest.raises(TimeCoordinateError):
        validate_time(_dataset(), expected=[bound])


def test_reversed_or_empty_intervals_are_errors():
    with pytest.raises(TimeCoordinateError, match="later than"):
        subset_time(_dataset(), start="2001-01-02", end="2001-01-01")
    with pytest.raises(TimeCoordinateError, match="no coordinate centers"):
        subset_time(_dataset(), start="2002-01-01")
    with pytest.raises(TimeCoordinateError, match="no coordinate centers"):
        subset_time(_dataset(), start="2001-01-01T01:00:00", end="2001-01-01T02:00:00")


@pytest.mark.parametrize("calendar,labels", [
    ("360_day", ["2001-02-29", "2001-02-30", "2001-03-01"]),
    ("noleap", ["2001-02-28", "2001-03-01", "2001-03-02"]),
    ("all_leap", ["2001-02-28", "2001-02-29", "2001-03-01"]),
    ("julian", ["1900-02-28", "1900-02-29", "1900-03-01"]),
    ("proleptic_gregorian", ["2000-02-28", "2000-02-29", "2000-03-01"]),
    ("standard", ["1582-10-04", "1582-10-15", "1582-10-16"]),
])
def test_calendar_specific_expected_strings_and_closed_intervals(calendar, labels):
    dates = [cftime.datetime(*map(int, label.split("-")), calendar=calendar) for label in labels]
    source = _dataset(dates)
    source.time.encoding["calendar"] = calendar
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        validate_time(source, expected=labels)
        validate_time(source, expected=source.get_index("time"))
        result = subset_time(source, start=labels[1], end=dates[2])
    assert not tasks
    np.testing.assert_array_equal(result.time, dates[1:])
    assert result.time.data[0].calendar == dates[0].calendar
    assert result.time.encoding["calendar"] == calendar


def test_cftime_invalid_dates_and_submicrosecond_bounds_are_rejected():
    source = _dataset([cftime.DatetimeNoLeap(2001, 2, 28), cftime.DatetimeNoLeap(2001, 3, 1)])
    with pytest.raises(TimeCoordinateError, match="Invalid start"):
        subset_time(source, start="2001-02-29")
    with pytest.raises(TimeCoordinateError, match="sub-microsecond"):
        subset_time(source, start="2001-02-28T00:00:00.000000001")


def test_cftime_microsecond_precision_is_exact():
    dates = [cftime.Datetime360Day(2001, 2, 30, microsecond=microsecond) for microsecond in (0, 1, 2)]
    source = _dataset(dates)
    result = subset_time(source, start="2001-02-30T00:00:00.000001", end="2001-02-30T00:00:00.000002")
    np.testing.assert_array_equal(result.time, dates[1:])
    validate_time(result, expected=["2001-02-30T00:00:00.000001", "2001-02-30T00:00:00.000002"])


def test_nanosecond_interval_and_expected_index_are_exact():
    dates = np.datetime64("2001-01-01T00:00:00", "ns") + np.arange(4).astype("timedelta64[ns]")
    source = _dataset(dates)
    result = subset_time(source, start="2001-01-01T00:00:00.000000001", end="2001-01-01T00:00:00.000000002")
    np.testing.assert_array_equal(result.time, dates[1:3])
    validate_time(result, expected=["2001-01-01T00:00:00.000000001", "2001-01-01T00:00:00.000000002"])
    with pytest.raises(TimeCoordinateError, match="expected index"):
        validate_time(result, expected=["2001-01-01T00:00:00", "2001-01-01T00:00:00.000000002"])


def test_explicit_expected_calendar_metadata_must_agree():
    source = _dataset()
    source.time.encoding["calendar"] = "gregorian"
    expected = source.time.copy(deep=True)
    expected.encoding["calendar"] = "standard"
    validate_time(source, expected=expected)
    expected.encoding["calendar"] = "proleptic_gregorian"
    with pytest.raises(TimeCoordinateError, match="calendar differs"):
        validate_time(source, expected=expected)


@pytest.mark.parametrize("declared", ["unknown_calendar", "360_day", "noleap"])
def test_datetime64_with_incompatible_declared_calendar_is_rejected(declared):
    source = _dataset()
    source.time.encoding["calendar"] = declared
    with pytest.raises(TimeCoordinateError, match="declared calendar"):
        validate_time(source)


def test_cftime_calendar_mismatch_is_not_silently_converted():
    source = _dataset([cftime.DatetimeNoLeap(2001, 1, 1), cftime.DatetimeNoLeap(2001, 2, 1)])
    for value in (cftime.Datetime360Day(2001, 1, 1), np.datetime64("2001-01-01")):
        with pytest.raises(TimeCoordinateError, match="calendar"):
            subset_time(source, start=value)
        with pytest.raises(TimeCoordinateError, match="calendar"):
            validate_time(source, expected=[value])
    source.time.attrs["calendar"] = "360_day"
    with pytest.raises(TimeCoordinateError, match="declared calendar differ"):
        validate_time(source)


def test_named_auxiliary_time_coordinate_selects_its_underlying_dimension():
    source = _dataset().rename_dims({"time": "record"}).rename_vars({"time": "valid_time"})
    result = subset_time(source, time="valid_time", start="2001-01-02")
    assert result.sizes["record"] == 2
    assert result.V.dims == ("record", "x")
    np.testing.assert_array_equal(result.valid_time, source.valid_time.values[2:])
    validate_time(result, time="valid_time", expected=source.valid_time.values[2:])


def test_python_datetime_bounds_are_precise_and_timezone_aware_bounds_rejected():
    source = _dataset()
    result = subset_time(source, start=datetime(2001, 1, 1, 12), end=datetime(2001, 1, 2))
    assert result.sizes["time"] == 2
    with pytest.raises(TimeCoordinateError, match="Timezone-aware"):
        subset_time(source, start=datetime(2001, 1, 1, tzinfo=timezone.utc))
