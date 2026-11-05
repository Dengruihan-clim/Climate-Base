"""Output date checks never scan meteorological fields or publish lossy time."""

from pathlib import Path

import cftime
import dask.array as da
from dask import delayed
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.io import open_dataset, write_dataset
from climate_preprocessing.io._time_roundtrip import validate_time_roundtrip
from climate_preprocessing.time import _date_key


def decoded(dates, *, calendar="proleptic_gregorian"):
    dataset = xr.Dataset({"valid_time": ("record", dates)})
    dataset.valid_time.encoding.update(units="seconds since 1970-01-01", calendar=calendar)
    return dataset


def test_time_check_compares_ns_and_nat_without_reading_numeric_fields():
    @delayed
    def forbidden():
        raise AssertionError("Meteorological field must not be read by date verification")

    values = np.array(["2000-01-01T00:00:00.000000001", "NaT"], dtype="datetime64[ns]")
    original = decoded(values)
    original["temperature"] = ("record", da.from_delayed(forbidden(), shape=(2,), dtype=float))
    output = decoded(values.copy())
    output["temperature"] = original.temperature
    validate_time_roundtrip(original, output)
    output.valid_time.data[0] = np.datetime64("2000-01-01T00:00:00.000000000")
    with pytest.raises(ValueError, match="precision"):
        validate_time_roundtrip(original, output)


def test_time_check_keeps_arbitrary_object_fields_unread():
    @delayed
    def forbidden():
        raise AssertionError("Object station metadata must not be read as time")

    original = xr.Dataset({"station_id": ("station", da.from_delayed(forbidden(), shape=(2,), dtype=object))})
    validate_time_roundtrip(original, original.copy(deep=False))


def test_time_check_cftime_microseconds_and_calendar_aliases():
    source = decoded([cftime.DatetimeNoLeap(5000, 1, 1, microsecond=1)], calendar="365_day")
    exact = decoded([cftime.DatetimeNoLeap(5000, 1, 1, microsecond=1)], calendar="noleap")
    validate_time_roundtrip(source, exact)
    rounded = decoded([cftime.DatetimeNoLeap(5000, 1, 1)], calendar="noleap")
    with pytest.raises(ValueError, match="precision"):
        validate_time_roundtrip(source, rounded)
    changed = decoded([cftime.Datetime360Day(5000, 1, 1, microsecond=1)], calendar="360_day")
    with pytest.raises(ValueError, match="calendar changed"):
        validate_time_roundtrip(source, changed)


def test_time_check_modern_gregorian_can_change_decoded_representation():
    source = decoded([cftime.DatetimeGregorian(2000, 1, 1, microsecond=1)], calendar="gregorian")
    output = decoded(np.array(["2000-01-01T00:00:00.000001"], dtype="datetime64[ns]"))
    validate_time_roundtrip(source, output)


def test_time_check_pre_reform_gregorian_semantics_cannot_change():
    source = decoded([cftime.DatetimeGregorian(1500, 1, 1)], calendar="standard")
    output = decoded([cftime.DatetimeProlepticGregorian(1500, 1, 1)], calendar="proleptic_gregorian")
    with pytest.raises(ValueError, match="calendar changed"):
        validate_time_roundtrip(source, output)


def test_time_check_missing_pattern_and_date_representation_cannot_disappear():
    source = decoded(np.array(["2000-01-01", "NaT"], dtype="datetime64[ns]"))
    output = decoded(np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="precision|missing"):
        validate_time_roundtrip(source, output)
    output = xr.Dataset({"valid_time": ("record", [0, 1])})
    with pytest.raises(ValueError, match="representation"):
        validate_time_roundtrip(source, output)


def test_time_check_entry_limit_is_checked_before_reading_time_arrays():
    @delayed
    def forbidden():
        raise AssertionError("Oversized time field must be rejected without reading")

    data = da.from_delayed(forbidden(), shape=(1_000_001,), dtype="datetime64[ns]")
    source = decoded(data)
    with pytest.raises(ValueError, match="entry limit"):
        validate_time_roundtrip(source, source.copy(deep=False))


@pytest.mark.parametrize("format", ["netcdf", "zarr"])
@pytest.mark.parametrize("case", ["nanoseconds_and_nat", "far_cftime_microseconds"])
def test_writer_either_preserves_exact_time_or_rejects_before_publication(tmp_path, format, case):
    if format == "zarr":
        pytest.importorskip("zarr")
        from climate_preprocessing.io.zarr import open_zarr, write_zarr

        writer, reader = write_zarr, open_zarr
    else:
        writer, reader = write_dataset, open_dataset
    if case == "nanoseconds_and_nat":
        source = decoded(np.array(["2000-01-01T00:00:00.000000001", "NaT"], dtype="datetime64[ns]"))
    else:
        source = decoded([cftime.DatetimeNoLeap(5000, 1, 1, microsecond=1),
                          cftime.DatetimeNoLeap(5000, 1, 1, microsecond=2)], calendar="noleap")
    source = source.chunk({"record": 1})
    destination = tmp_path / ("output.nc" if format == "netcdf" else "output.zarr")
    try:
        writer(source, destination)
    except ValueError as error:
        assert "time precision" in str(error) or "missing-date" in str(error)
        assert not destination.exists()
    else:
        with reader(destination) as result:
            validate_time_roundtrip(source, result)
            original = source.valid_time.compute().data
            actual = result.valid_time.compute().data
            for left, right in zip(original, actual):
                if isinstance(left, np.datetime64) and np.isnat(left):
                    assert isinstance(right, np.datetime64) and np.isnat(right)
                else:
                    assert _date_key(left) == _date_key(right)
    assert not list(Path(tmp_path).glob("*.partial"))
