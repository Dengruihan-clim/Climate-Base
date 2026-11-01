"""Real-file-shaped contracts for explicitly preserving per-source metadata.

These small synthetic fixtures model annual ERA5 V files without requiring HPC
data. They distinguish descriptive history/ranges from scientific metadata and
verify that reconciliation remains lazy and survives a NetCDF roundtrip.
"""

import json
from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.io import open_dataset, open_mfdataset, write_dataset
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE


def _annual_source(year, offset=0):
    values = np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) - 12 + offset
    source = xr.Dataset(
        {
            "V": (("time", "level", "latitude", "longitude"), values, {
                "units": "m s**-1", "standard_name": "northward_wind",
                "long_name": "V component of wind", "cell_methods": "time: mean",
                "ancillary_variables": "utc_date",
                "minimum_value": np.float32(values.min()),
                "maximum_value": np.float32(values.max()),
                "actual_range": np.array([values.min(), values.max()], dtype=np.float32),
                "valid_min": np.float32(-100), "valid_max": np.float32(100),
            }),
            "utc_date": ("time", np.array([year * 1000000 + 10100, year * 1000000 + 20100], dtype=np.int32),
                         {"long_name": "UTC date yyyymmddhh"}),
        },
        coords={
            "time": np.array([f"{year}-01-01", f"{year}-02-01"], dtype="datetime64[ns]"),
            "level": ("level", [1., 2.], {"units": "hPa", "long_name": "pressure level"}),
            "latitude": ("latitude", [30., -30.], {"units": "degrees_north"}),
            "longitude": ("longitude", [0., 120., 240.], {"units": "degrees_east"}),
        },
        attrs={"Conventions": "CF-1.6", "title": "Synthetic ERA5-like annual wind",
               "history": f"Converted source year {year}", "CONVERSION_DATE": f"{year + 1}-01-01"},
    )
    source.time.encoding["calendar"] = "gregorian"
    return source


def _write_pair(tmp_path, sources=None):
    sources = sources or [_annual_source(2003), _annual_source(2004, offset=50)]
    paths = [tmp_path / "2003.nc", tmp_path / "2004.nc"]
    for path, source in zip(paths, sources):
        source.to_netcdf(path, engine="netcdf4")
    return paths, sources


def _merge_event(dataset):
    operations = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])["operations"]
    return next(event for event in reversed(operations) if event["operation"] == "open_mfdataset")


def test_strict_default_still_rejects_file_scoped_history(tmp_path):
    paths, _ = _write_pair(tmp_path)
    with pytest.raises(ValueError, match="Multi-file global attributes differ") as failure:
        open_mfdataset(paths)
    assert "history" in str(failure.value)
    assert "CONVERSION_DATE" in str(failure.value)


def test_preserve_sources_is_lazy_sorted_and_retains_scientific_metadata(tmp_path):
    paths, sources = _write_pair(tmp_path)
    original_bytes = [path.read_bytes() for path in paths]
    tasks = []
    with Callback(posttask=lambda *args: tasks.append(args[0])):
        result = open_mfdataset(paths[::-1], chunks={"time": 1}, metadata_policy="preserve_sources")
    try:
        assert tasks == []
        assert isinstance(result.V.data, da.Array)
        assert result.V.dims == ("time", "level", "latitude", "longitude")
        np.testing.assert_array_equal(result.time.dt.year, [2003, 2003, 2004, 2004])
        assert result.V.attrs["units"] == "m s**-1"
        assert result.V.attrs["standard_name"] == "northward_wind"
        assert result.V.attrs["cell_methods"] == "time: mean"
        assert result.V.attrs["valid_min"] == -100
        assert result.V.attrs["valid_max"] == 100
        assert result.V.attrs.get("ancillary_variables", result.V.encoding.get("ancillary_variables")) == "utc_date"
        assert result.level.attrs["units"] == "hPa"
        assert result.attrs["Conventions"] == "CF-1.6"
        assert "history" not in result.attrs and "CONVERSION_DATE" not in result.attrs
        assert not {"actual_range", "minimum_value", "maximum_value"} & result.V.attrs.keys()
        expected = np.concatenate([source.V.values for source in sources], axis=0)
        np.testing.assert_allclose(result.V.compute().values, expected)
        np.testing.assert_array_equal(result.utc_date.compute(), np.concatenate([s.utc_date.values for s in sources]))

        event = _merge_event(result)
        assert event["parameters"]["metadata_policy"] == "preserve_sources"
        assert [record["name"] for record in event["inputs"]] == ["2004.nc", "2003.nc"]
        for record, source in zip(event["inputs"], sources[::-1]):
            metadata = record["source_metadata"]
            assert metadata["global_attributes"] == {
                "history": source.attrs["history"], "CONVERSION_DATE": source.attrs["CONVERSION_DATE"]}
            ranges = metadata["variable_attributes"]["V"]
            assert ranges["actual_range"] == source.V.attrs["actual_range"].tolist()
            assert ranges["minimum_value"] == float(source.V.attrs["minimum_value"])
            assert ranges["maximum_value"] == float(source.V.attrs["maximum_value"])
            assert "units" not in ranges
        assert str(tmp_path) not in result.attrs[PROVENANCE_ATTRIBUTE]
    finally:
        result.close()
    assert [path.read_bytes() for path in paths] == original_bytes


def test_missing_allowed_metadata_is_preserved_as_absence_not_filled(tmp_path):
    first, second = _annual_source(2003), _annual_source(2004)
    del second.attrs["history"]
    second.attrs["date_created"] = "2005-02-03"
    del second.V.attrs["actual_range"]
    paths, _ = _write_pair(tmp_path, [first, second])
    with open_mfdataset(paths, metadata_policy="preserve_sources") as result:
        records = _merge_event(result)["inputs"]
        assert "history" in records[0]["source_metadata"]["global_attributes"]
        assert "history" not in records[1]["source_metadata"]["global_attributes"]
        assert records[1]["source_metadata"]["global_attributes"]["date_created"] == "2005-02-03"
        assert "actual_range" not in records[1]["source_metadata"]["variable_attributes"]["V"]
        assert "date_created" not in result.attrs


@pytest.mark.parametrize("attribute,value", [
    ("units", "km h-1"),
    ("standard_name", "eastward_wind"),
    ("cell_methods", "time: point"),
    ("valid_min", np.float32(-90)),
    ("valid_max", np.float32(90)),
    ("unrecognized_attribute", "needs review"),
])
def test_preserve_sources_never_overrides_scientific_or_unknown_variable_attrs(tmp_path, attribute, value):
    first, second = _annual_source(2003), _annual_source(2004)
    second.V.attrs[attribute] = value
    paths, _ = _write_pair(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Multi-file variable metadata differ") as failure:
        open_mfdataset(paths, metadata_policy="preserve_sources")
    assert attribute in str(failure.value)


def test_unknown_global_metadata_is_not_ignored(tmp_path):
    first, second = _annual_source(2003), _annual_source(2004)
    second.attrs["title"] = "Another product"
    paths, _ = _write_pair(tmp_path, [first, second])
    with pytest.raises(ValueError, match="Multi-file global attributes differ") as failure:
        open_mfdataset(paths, metadata_policy="preserve_sources")
    assert "title" in str(failure.value)


@pytest.mark.parametrize("conflict", ["calendar", "grid", "cf_reference"])
def test_preserve_sources_retains_calendar_grid_and_cf_link_checks(tmp_path, conflict):
    first, second = _annual_source(2003), _annual_source(2004)
    if conflict == "calendar":
        second.time.encoding["calendar"] = "proleptic_gregorian"
        expected = "calendars differ"
    elif conflict == "grid":
        second = second.assign_coords(longitude=second.longitude + 0.5)
        second.longitude.attrs = dict(first.longitude.attrs)
        expected = "non-time coordinate"
    else:
        first["other_date"] = first.utc_date.copy()
        second["other_date"] = second.utc_date.copy()
        second.V.attrs["ancillary_variables"] = "other_date"
        expected = "variable metadata differ"
    paths, _ = _write_pair(tmp_path, [first, second])
    with pytest.raises(ValueError, match=expected):
        open_mfdataset(paths, metadata_policy="preserve_sources")


def test_combined_output_roundtrips_units_links_and_source_histories(tmp_path):
    paths, _ = _write_pair(tmp_path)
    output = tmp_path / "combined.nc"
    with open_mfdataset(paths, metadata_policy="preserve_sources") as combined:
        event = _merge_event(combined)
        write_dataset(combined, output)
        with open_dataset(output) as reopened:
            assert reopened.V.attrs["units"] == "m s**-1"
            assert reopened.V.attrs["standard_name"] == "northward_wind"
            assert reopened.V.attrs.get("ancillary_variables", reopened.V.encoding.get("ancillary_variables")) == "utc_date"
            assert _merge_event(reopened)["inputs"] == event["inputs"]
            assert not {"actual_range", "minimum_value", "maximum_value"} & reopened.V.attrs.keys()
            xr.testing.assert_allclose(reopened.V.compute(), combined.V.compute())


def test_strict_concat_and_write_preserve_variable_semantics(tmp_path):
    first, second = _annual_source(2003), _annual_source(2004)
    second.attrs = dict(first.attrs)
    paths, sources = _write_pair(tmp_path, [first, second])
    output = tmp_path / "strict_combined.nc"
    with open_mfdataset(paths) as combined:
        assert _merge_event(combined)["parameters"]["metadata_policy"] == "strict"
        assert combined.V.attrs["units"] == "m s**-1"
        assert combined.V.attrs["standard_name"] == "northward_wind"
        assert combined.V.attrs["cell_methods"] == "time: mean"
        assert combined.V.attrs.get("ancillary_variables", combined.V.encoding.get("ancillary_variables")) == "utc_date"
        assert combined.level.attrs["units"] == "hPa"
        write_dataset(combined, output)
        with open_dataset(output) as reopened:
            assert reopened.V.attrs["units"] == "m s**-1"
            assert reopened.V.attrs["standard_name"] == "northward_wind"
            assert reopened.V.attrs["cell_methods"] == "time: mean"
            assert reopened.V.attrs.get("ancillary_variables", reopened.V.encoding.get("ancillary_variables")) == "utc_date"
            assert reopened.level.attrs["units"] == "hPa"
            np.testing.assert_allclose(reopened.V.compute(), np.concatenate([source.V.values for source in sources]))


def test_nonfinite_numpy_range_attributes_use_valid_json_source_records(tmp_path):
    first, second = _annual_source(2003), _annual_source(2004)
    first.V.attrs["actual_range"] = np.array([np.nan, np.inf], dtype=np.float32)
    first.V.attrs["minimum_value"] = np.float32(np.nan)
    first.V.attrs["maximum_value"] = np.float32(np.inf)
    second.V.attrs["minimum_value"] = np.float32(-np.inf)
    paths, sources = _write_pair(tmp_path, [first, second])

    def reject_nonstandard_json_constant(value):
        raise AssertionError(f"Non-standard JSON constant in provenance: {value}")

    with open_mfdataset(paths, metadata_policy="preserve_sources") as combined:
        parsed = json.loads(combined.attrs[PROVENANCE_ATTRIBUTE], parse_constant=reject_nonstandard_json_constant)
        records = parsed["operations"][-1]["inputs"]
        first_ranges = records[0]["source_metadata"]["variable_attributes"]["V"]
        second_ranges = records[1]["source_metadata"]["variable_attributes"]["V"]
        assert first_ranges["actual_range"] == [{"nonfinite": "NaN"}, {"nonfinite": "Infinity"}]
        assert first_ranges["minimum_value"] == {"nonfinite": "NaN"}
        assert first_ranges["maximum_value"] == {"nonfinite": "Infinity"}
        assert second_ranges["minimum_value"] == {"nonfinite": "-Infinity"}
        assert not {"actual_range", "minimum_value", "maximum_value"} & combined.V.attrs.keys()
        np.testing.assert_allclose(combined.V.compute(), np.concatenate([source.V.values for source in sources]))


def test_single_source_policy_preserves_original_ranges_and_history(tmp_path):
    source = _annual_source(2003)
    path = tmp_path / "one.nc"
    source.to_netcdf(path, engine="netcdf4")
    with open_mfdataset([path], metadata_policy="preserve_sources") as result:
        assert result.attrs["history"] == source.attrs["history"]
        np.testing.assert_array_equal(result.V.attrs["actual_range"], source.V.attrs["actual_range"])
        assert result.V.attrs["minimum_value"] == source.V.attrs["minimum_value"]


@pytest.mark.parametrize("policy", [None, False, "", "ignore", {}, []])
def test_invalid_policy_has_an_explicit_error(tmp_path, policy):
    source = tmp_path / "one.nc"
    _annual_source(2003).to_netcdf(source, engine="netcdf4")
    with pytest.raises((TypeError, ValueError), match="metadata_policy"):
        open_mfdataset([source], metadata_policy=policy)


def test_preserve_sources_closes_inputs_on_success_and_metadata_failure(monkeypatch):
    from climate_preprocessing.io import multifile

    paths = [Path(__file__), Path(multifile.__file__)]
    for conflict in (False, True):
        closed = []
        sources = [_annual_source(2003).chunk(), _annual_source(2004).chunk()]
        if conflict:
            sources[1].V.attrs["units"] = "km h-1"
        for index, source in enumerate(sources):
            source.set_close(lambda number=index: closed.append(number))
        original_attrs = [dict(source.attrs) for source in sources]
        original_ranges = [source.V.attrs["actual_range"].copy() for source in sources]
        iterator = iter(sources)
        monkeypatch.setattr(multifile, "open_dataset", lambda *args, **kwargs: next(iterator))
        if conflict:
            with pytest.raises(ValueError, match="units"):
                open_mfdataset(paths, metadata_policy="preserve_sources")
        else:
            result = open_mfdataset(paths, metadata_policy="preserve_sources")
            assert closed == []
            result.close()
        assert closed == [0, 1]
        for source, attributes, ranges in zip(sources, original_attrs, original_ranges):
            assert source.attrs == attributes
            np.testing.assert_array_equal(source.V.attrs["actual_range"], ranges)
