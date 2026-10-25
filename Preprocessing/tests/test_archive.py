from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import pytest
from netCDF4 import Dataset, date2num
from climate_preprocessing import Coordinates, FieldSpec, MonthlyArchive, convert_units


def make_file(path, dates=((2000, 1), (2000, 2)), *, units="K", values=None,
              lat=(90., 0., -90.), lon=(-180., -90., 0., 90.),
              pressure=(100000., 50000., 10000.), p_units="Pa", calendar="standard",
              time_origin="days since 1900-01-01", permute=False, extra=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Dataset(path, "w") as ds:
        for name, count in (("time", len(dates)), ("lat", len(lat)), ("lon", len(lon)), ("member", extra)):
            ds.createDimension(name, count)
        if pressure is not None:
            ds.createDimension("lev", len(pressure))
        t = ds.createVariable("time", "f8", ("time",))
        t.units, t.calendar = time_origin, calendar
        t[:] = date2num([datetime(y, m, 1) for y, m in dates], t.units, calendar=calendar)
        for name, axis, unit in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
            var = ds.createVariable(name, "f8", (name,))
            var.units, var[:] = unit, axis
        if pressure is not None:
            var = ds.createVariable("lev", "f8", ("lev",))
            var.units, var[:] = p_units, pressure
        dims = ("member", "time") + (("lev",) if pressure is not None else ()) + ("lat", "lon")
        shape = tuple(len(ds.dimensions[d]) for d in dims)
        data = np.full(shape, 280.) if values is None else np.broadcast_to(values, shape)
        if permute:
            dims = dims[::-1]
            data = np.transpose(data, tuple(reversed(range(data.ndim))))
        var = ds.createVariable("temperature", "f8", dims, fill_value=-9999.)
        var.units = units
        var[:] = np.ma.masked_invalid(data)
    return path


def archive(tmp_path, files="*.nc", **kwargs):
    spec = FieldSpec(files, "temperature", coordinates=Coordinates(pressure="lev"))
    return MonthlyArchive({"t": spec}, data_root=tmp_path, **kwargs)


def test_normalized_pressure_axes_dimensions_and_values(tmp_path):
    p, lat, lon = np.array([100000., 50000., 10000.]), np.array([90., 0., -90.]), np.array([-180., -90., 0., 90.])
    values = np.arange(3)[:, None, None] * 10 + lat[None, :, None] / 90 + 3 * np.cos(np.deg2rad(lon))[None, None, :]
    make_file(tmp_path / "a.nc", values=values, permute=True)
    field = archive(tmp_path).read("t", 2000, 1)
    np.testing.assert_array_equal(field.latitude, [-90, 0, 90])
    np.testing.assert_array_equal(field.longitude, [0, 90, 180, 270])
    np.testing.assert_array_equal(field.pressure, [100, 500, 1000])
    np.testing.assert_allclose(field.values[0], 20 + np.array([-1, 0, 1])[:, None] + [3, 0, -3, 0], atol=1e-14)


def test_configurable_coordinate_orders(tmp_path):
    make_file(tmp_path / "a.nc")
    field = archive(tmp_path, longitude="-180_180", latitude_order="descending", pressure_units="Pa", pressure_order="descending").read("t", 2000, 1)
    np.testing.assert_array_equal(field.latitude, [90, 0, -90])
    np.testing.assert_array_equal(field.longitude, [-180, -90, 0, 90])
    np.testing.assert_array_equal(field.pressure, [100000, 50000, 10000])


def test_actual_cf_time_selected_not_filename(tmp_path):
    make_file(tmp_path / "year_1990.nc", dates=((2000, 1), (2000, 2)))
    assert archive(tmp_path).read("t", 2000, 2).month == 2
    with pytest.raises(ValueError, match="found 0"):
        archive(tmp_path).read("t", 1990, 1)


def test_missing_and_duplicate_months_fail(tmp_path):
    path = make_file(tmp_path / "a.nc")
    with pytest.raises(ValueError, match="found 0"):
        archive(tmp_path).read("t", 2000, 3)
    shutil.copyfile(path, tmp_path / "duplicate.nc")
    with pytest.raises(ValueError, match="found 2"):
        archive(tmp_path).read("t", 2000, 1)


def test_daily_or_unsorted_time_is_not_a_monthly_source(tmp_path):
    make_file(tmp_path / "a.nc", dates=((2000, 1), (2000, 1)))
    with pytest.raises(ValueError, match="strictly increasing"):
        archive(tmp_path).read("t", 2000, 1)
    make_file(tmp_path / "a.nc", dates=((2000, 2), (2000, 1)))
    with pytest.raises(ValueError, match="strictly increasing"):
        archive(tmp_path).read("t", 2000, 1)


def test_same_month_two_distinct_time_labels_fail(tmp_path):
    path = make_file(tmp_path / "a.nc")
    with Dataset(path, "a") as ds:
        ds["time"][:] = date2num([datetime(2000, 1, 1), datetime(2000, 1, 15)], ds["time"].units)
    with pytest.raises(ValueError, match="repeated CF calendar months"):
        archive(tmp_path).read("t", 2000, 1)


def test_changed_grid_or_pressure_fails_without_silent_intersection(tmp_path):
    make_file(tmp_path / "jan.nc", dates=((2000, 1),))
    make_file(tmp_path / "feb.nc", dates=((2000, 2),), pressure=(100000., 40000., 10000.))
    source = archive(tmp_path)
    source.read("t", 2000, 1)
    with pytest.raises(ValueError, match="pressure levels changed"):
        source.read("t", 2000, 2)


def test_explicit_unit_conversion_and_wrong_unit_rejected(tmp_path):
    make_file(tmp_path / "a.nc", units="degree_Celsius", values=6.85)
    spec = FieldSpec("*.nc", "temperature", Coordinates(pressure="lev"), expected_units="degree_Celsius", target_units="K")
    source = MonthlyArchive({"t": spec}, data_root=tmp_path)
    np.testing.assert_allclose(source.read("t", 2000, 1).values, 280.)
    assert source.audit["unit_transforms"]["t"][0]["offset"] == 273.15
    with pytest.raises(ValueError, match="incompatible"):
        convert_units([1.], "m/s", "Pa/s")
    with pytest.raises(ValueError, match="incompatible"):
        convert_units([1.], "m2 s-2", "m")
    np.testing.assert_allclose(convert_units([9.80665], "m2 s-2", "m", scale=1 / 9.80665), [1.])


def test_units_override_is_recorded_and_not_silently_assumed(tmp_path):
    make_file(tmp_path / "a.nc", units="legacy", values=1.)
    spec = FieldSpec("*.nc", "temperature", Coordinates(pressure="lev"), expected_units="K", units_override="K")
    source = MonthlyArchive({"t": spec}, data_root=tmp_path)
    assert source.read("t", 2000, 1).metadata["unit_transform"]["source_units"] == "legacy"
    assert source.audit["unit_transforms"]["t"][0]["units_override"] == "K"
    with pytest.raises(ValueError, match="expected_units"):
        MonthlyArchive({"t": FieldSpec("*.nc", "temperature", Coordinates(pressure="lev"), expected_units="K")}, data_root=tmp_path).read("t", 2000, 1)


def test_units_may_not_change_implicitly_between_files(tmp_path):
    make_file(tmp_path / "jan.nc", dates=((2000, 1),), units="K")
    make_file(tmp_path / "feb.nc", dates=((2000, 2),), units="degree_Celsius")
    source = archive(tmp_path)
    source.read("t", 2000, 1)
    with pytest.raises(ValueError, match="Output units changed"):
        source.read("t", 2000, 2)
    spec = FieldSpec("*.nc", "temperature", Coordinates(pressure="lev"), target_units="K")
    source = MonthlyArchive({"t": spec}, data_root=tmp_path)
    assert source.read("t", 2000, 1).units == source.read("t", 2000, 2).units == "K"


def test_strict_and_explicit_modern_calendar_policy(tmp_path):
    make_file(tmp_path / "jan.nc", dates=((2000, 1),), calendar="standard", time_origin="days since 1500-01-01")
    make_file(tmp_path / "feb.nc", dates=((2000, 2),), calendar="proleptic_gregorian", time_origin="days since 1500-01-01")
    source = archive(tmp_path)
    source.read("t", 2000, 1)
    with pytest.raises(ValueError, match="Mixed incompatible calendars"):
        source.read("t", 2000, 2)
    source = archive(tmp_path, calendar_policy="modern_gregorian")
    assert source.read("t", 2000, 1).calendar == source.read("t", 2000, 2).calendar == "proleptic_gregorian"


def test_modern_calendar_policy_does_not_merge_ancient_or_noleap(tmp_path):
    make_file(tmp_path / "jan.nc", dates=((1500, 1),), calendar="standard")
    make_file(tmp_path / "feb.nc", dates=((1500, 2),), calendar="proleptic_gregorian")
    source = archive(tmp_path, calendar_policy="modern_gregorian")
    source.read("t", 1500, 1)
    with pytest.raises(ValueError, match="Mixed incompatible calendars"):
        source.read("t", 1500, 2)


def test_hashes_relative_provenance_and_metadata_inspection(tmp_path):
    path = make_file(tmp_path / "a.nc")
    source = archive(tmp_path)
    metadata = source.inspect("t", 2000, 1)
    assert metadata["field_values_read"] is False
    assert source.inventory[0]["path"] == "a.nc"
    assert source.inventory[0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert str(tmp_path) not in json.dumps(source.inventory)


def test_changed_file_after_indexing_is_rejected(tmp_path):
    path = make_file(tmp_path / "a.nc")
    source = archive(tmp_path)
    source.inspect("t", 2000, 1)
    with Dataset(path, "a") as ds:
        ds.modified = "different source"
    with pytest.raises(ValueError, match="changed after indexing"):
        source.read("t", 2000, 2)


@pytest.mark.parametrize("coordinate,units,error", [("lat", "degrees_south", "degrees_north"), ("lon", "degrees_west", "degrees_east")])
def test_coordinate_units_are_not_guessed(tmp_path, coordinate, units, error):
    path = make_file(tmp_path / "a.nc")
    with Dataset(path, "a") as ds:
        ds[coordinate].units = units
    with pytest.raises(ValueError, match=error):
        archive(tmp_path).read("t", 2000, 1)


def test_duplicate_cyclic_endpoint_and_optional_global_check(tmp_path):
    make_file(tmp_path / "a.nc", lon=(0., 90., 180., 360.))
    with pytest.raises(ValueError, match="cyclic endpoint"):
        archive(tmp_path).read("t", 2000, 1)
    make_file(tmp_path / "a.nc", lon=(0., 10., 20., 30.))
    assert archive(tmp_path).read("t", 2000, 1).longitude.size == 4
    with pytest.raises(ValueError, match="Global longitude"):
        archive(tmp_path, require_global_longitude=True).read("t", 2000, 1)


def test_extra_non_singleton_dimension_requires_explicit_selection(tmp_path):
    make_file(tmp_path / "a.nc", extra=2)
    with pytest.raises(ValueError, match="non-singleton"):
        archive(tmp_path).read("t", 2000, 1)


def test_monthly_template_and_shared_surface_grid(tmp_path):
    make_file(tmp_path / "2000/field_200001.nc", dates=((2000, 1),))
    make_file(tmp_path / "surface.nc", pressure=None)
    fields = {"t": FieldSpec("{year}/field_{yyyymm}.nc", "temperature", Coordinates(pressure="lev")),
              "surface": FieldSpec("surface.nc", "temperature")}
    source = MonthlyArchive(fields, data_root=tmp_path, shared_grid=True)
    assert source.read("t", 2000, 1).pressure is not None
    assert source.read("surface", 2000, 1).pressure is None


@pytest.mark.parametrize("value", [2000.5, 2000.0, True, np.bool_(True), "2000"])
def test_year_values_are_explicit_integers(tmp_path, value):
    with pytest.raises(ValueError, match="integer"):
        archive(tmp_path).read("t", value, 1)


@pytest.mark.parametrize("option", ["shared_grid", "hash_inputs", "require_global_longitude"])
def test_boolean_options_reject_strings(tmp_path, option):
    with pytest.raises(ValueError, match="boolean"):
        archive(tmp_path, **{option: "false"})


def test_unit_conversion_rejects_complex_and_finite_overflow():
    with pytest.raises(ValueError, match="Complex"):
        convert_units([1 + 2j], "K", "K")
    with pytest.raises(ValueError, match="nonfinite values from finite"):
        convert_units([1e308], "hPa", "Pa")
    assert np.isnan(convert_units([np.inf], "K", "K")).all()
