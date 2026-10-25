from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from climate_preprocessing import Coordinates, FieldSpec, MonthlyArchive, season_dates, seasonal_mean, iter_seasons, month_days
from climate_preprocessing.io import write_seasonal, write_seasonal_stream
from climate_preprocessing.__main__ import main
from test_archive import make_file


def source(tmp_path, dates, *, calendar="standard", values=None):
    make_file(tmp_path / "monthly.nc", dates=dates, calendar=calendar, values=values)
    return MonthlyArchive({"t": FieldSpec("*.nc", "temperature", Coordinates(pressure="lev"))}, data_root=tmp_path)


def test_winter_years_and_invalid_month_sequences():
    assert season_dates(2000, 2000) == [(2000, 1999, 12), (2000, 2000, 1), (2000, 2000, 2)]
    assert season_dates(2001, 2001, (3, 4, 5)) == [(2001, 2001, 3), (2001, 2001, 4), (2001, 2001, 5)]
    for months in ((), (12, 2), (12, 1, 12), (0, 1)):
        with pytest.raises(ValueError):
            season_dates(2000, 2000, months)


@pytest.mark.parametrize("calendar,year,february", [("standard", 2000, 29), ("standard", 2001, 28),
                                                  ("noleap", 2000, 28), ("all_leap", 2001, 29), ("360_day", 2001, 30)])
def test_actual_calendar_month_weights(tmp_path, calendar, year, february):
    values = np.array([0., 10., 20.])[:, None, None, None]
    dates = ((year - 1, 12), (year, 1), (year, 2))
    archive = source(tmp_path, dates, calendar=calendar, values=values)
    actual = seasonal_mean(archive, "t", year, year)
    december = 30 if calendar == "360_day" else 31
    january = december
    expected = (10 * january + 20 * february) / (december + january + february)
    np.testing.assert_allclose(actual.values, expected)
    assert month_days(year, 2, calendar) == february
    equal = seasonal_mean(archive, "t", year, year, weighting="equal")
    np.testing.assert_allclose(equal.values, 10.)


def test_missing_month_fails_and_partial_cells_remain_missing(tmp_path):
    dates = ((1999, 12), (2000, 1))
    archive = source(tmp_path, dates)
    with pytest.raises(ValueError, match="found 0"):
        seasonal_mean(archive, "t", 2000, 2000)
    values = np.array([1., np.nan, 3.])[:, None, None, None]
    archive = source(tmp_path, dates + ((2000, 2),), values=values)
    assert np.isnan(seasonal_mean(archive, "t", 2000, 2000).values).all()


def test_iterator_and_atomic_non_overwrite_export(tmp_path):
    dates = ((1999, 12), (2000, 1), (2000, 2), (2000, 12), (2001, 1), (2001, 2))
    archive = source(tmp_path, dates)
    records = list(iter_seasons(archive, "t", 2000, 2001))
    assert [record.years.tolist() for record in records] == [[2000], [2001]]
    result = seasonal_mean(archive, "t", 2000, 2001)
    output = tmp_path / "seasonal.nc"
    write_seasonal(output, result, pressure_units="hPa", inventory=archive.inventory, audit=archive.audit)
    with Dataset(output) as ds:
        np.testing.assert_array_equal(ds["season_year"][:], [2000, 2001])
        np.testing.assert_array_equal(ds["seasonal_value"][:], result.values)
        assert ds["pressure"].units == "hPa"
        assert str(tmp_path) not in ds.input_inventory
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_seasonal(output, result, pressure_units="hPa")
    assert output.read_bytes() == original


def test_synthetic_generator_and_cli(tmp_path):
    script = Path(__file__).resolve().parents[1] / "examples/generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("synthetic_example", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config_path = module.create_synthetic(tmp_path / "demo")
    with redirect_stdout(io.StringIO()) as stream:
        assert main(["check", str(config_path)]) == 0
    report = json.loads(stream.getvalue())
    assert report["outputs_written"] is False
    with redirect_stdout(io.StringIO()):
        assert main(["seasonal", str(config_path)]) == 0
    with Dataset(config_path.parent / "seasonal.nc") as ds:
        assert ds["seasonal_value"].units == "K"
        assert ds["seasonal_value"].shape == (2, 3, 3, 4)
    with pytest.raises(FileExistsError):
        main(["seasonal", str(config_path)])


@pytest.mark.parametrize("year,month", [(2000.5, 1), (2000, 1.0), (True, 1), (2000, np.bool_(True))])
def test_month_days_does_not_truncate_or_accept_boolean_dates(year, month):
    with pytest.raises(ValueError, match="integer"):
        month_days(year, month)


def test_streaming_export_does_not_publish_partial_result(tmp_path):
    archive = source(tmp_path, ((1999, 12), (2000, 1), (2000, 2)))
    records = iter_seasons(archive, "t", 2000, 2001)
    path = tmp_path / "incomplete.nc"
    with pytest.raises(ValueError, match="found 0"):
        write_seasonal_stream(path, records, pressure_units="hPa")
    assert not path.exists()
    assert not list(tmp_path.glob("*.partial"))
