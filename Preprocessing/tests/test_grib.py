"""Real ecCodes-generated GRIB messages, forecast semantics and lazy contracts.

Fixtures use ecCodes' built-in regular_ll_sfc_grib2 template with entirely
synthetic values authored here. No real data or external binary is bundled.
"""

import json
from pathlib import Path

import dask.array as da
from dask.callbacks import Callback
import numpy as np
import pytest

from climate_preprocessing.io.grib import open_grib, inspect_grib_groups
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE


@pytest.fixture
def eccodes():
    pytest.importorskip("cfgrib")
    module = pytest.importorskip("eccodes")
    try:
        message = module.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    except (RuntimeError, OSError) as error:
        pytest.skip(f"Native ecCodes samples unavailable: {error}")
    module.codes_release(message)
    return module


def write_grib(path, eccodes, records):
    """Generate pure concatenated regular-grid messages; return packed truth."""
    truth = []
    with Path(path).open("wb") as stream:
        for options in records:
            options = dict(options)
            values = np.asarray(options.pop("values", np.arange(6, dtype=float) + 250), dtype=float)
            message = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
            try:
                settings = {
                    "Ni": 3, "Nj": 2, "latitudeOfFirstGridPointInDegrees": 10.,
                    "latitudeOfLastGridPointInDegrees": 0., "longitudeOfFirstGridPointInDegrees": 0.,
                    "longitudeOfLastGridPointInDegrees": 20., "iDirectionIncrementInDegrees": 10.,
                    "jDirectionIncrementInDegrees": 10., "dataDate": 20000101, "dataTime": 0,
                    "typeOfLevel": "isobaricInhPa", "level": 500, "shortName": "t", "step": 0,
                    "bitsPerValue": 24,
                }
                settings.update(options)
                for key, value in settings.items():
                    eccodes.codes_set(message, key, value)
                if np.isnan(values).any():
                    eccodes.codes_set(message, "bitmapPresent", 1)
                    eccodes.codes_set(message, "missingValue", 9999.)
                    values = np.where(np.isnan(values), 9999., values)
                eccodes.codes_set_values(message, values)
                packed = eccodes.codes_get_values(message)
                truth.append(np.where(packed == 9999., np.nan, packed))
                eccodes.codes_write(message, stream)
            finally:
                eccodes.codes_release(message)
    return truth


def test_real_grib_open_builds_no_field_tasks_or_decodes_and_preserves_units(tmp_path, eccodes, monkeypatch):
    path = tmp_path / "field.grib2"
    truth = write_grib(path, eccodes, [{}])[0]
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    fields, tasks = [], []
    original = eccodes.codes_get_array

    def read_array(message, key, *args, **kwargs):
        if key in {"values", "codedValues"}:
            fields.append(key)
        return original(message, key, *args, **kwargs)

    monkeypatch.setattr(eccodes, "codes_get_array", read_array)
    with Callback(pretask=lambda *args: tasks.append(True)):
        dataset = open_grib(path, chunks={"time": 1})
    try:
        assert not tasks and not fields
        assert isinstance(dataset.t.data, da.Array)
        assert dataset.t.dtype == np.float64
        assert dataset.sizes["time"] == dataset.sizes["step"] == dataset.sizes["isobaricInhPa"] == 1
        assert dataset.t.attrs["units"] == "K"
        assert dataset.time.attrs["standard_name"] == "forecast_reference_time"
        assert dataset.valid_time.dims == ("time", "step")
        assert str(tmp_path) not in repr(dataset.attrs)
        assert all("source" not in variable.encoding for variable in dataset.variables.values())
        np.testing.assert_array_equal(dataset.t.compute().data.reshape(-1), truth)
        assert fields
        event = json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]
        assert event["parameters"]["selected_messages"] == 1
        assert event["inputs"][0]["name"] == path.name
    finally:
        dataset.close()
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]
    assert before == (path.stat().st_size, path.stat().st_mtime_ns)


def test_header_group_inspection_and_explicit_heterogeneous_level_selection(tmp_path, eccodes):
    path = tmp_path / "mixed.grib2"
    write_grib(path, eccodes, [{}, {"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"}])
    groups = inspect_grib_groups(path)
    assert len(groups) == 2
    assert {group["filter_by_keys"]["typeOfLevel"] for group in groups} == {"isobaricInhPa", "heightAboveGround"}
    with pytest.raises(ValueError, match="heterogeneous"):
        open_grib(path)
    for group in groups:
        with open_grib(path, filter_by_keys=group["filter_by_keys"]) as selected:
            assert len(selected.data_vars) == 1
            assert all(isinstance(variable.data, da.Array) for variable in selected.data_vars.values())
    assert not list(tmp_path.glob("*.idx"))


def test_forecast_reference_step_and_repeated_valid_times_are_not_flattened(tmp_path, eccodes):
    path = tmp_path / "forecast.grib2"
    records = [{"dataTime": hour * 100, "step": step, "values": np.full(6, hour + step)}
               for hour in (0, 6) for step in (0, 6)]
    write_grib(path, eccodes, records)
    with open_grib(path) as result:
        assert result.sizes["time"] == result.sizes["step"] == 2
        assert result.valid_time.dims == ("time", "step")
        assert result.valid_time.data[0, 1] == result.valid_time.data[1, 0]
        np.testing.assert_array_equal(result.t.compute().data[:, :, 0, 0, 0], [[0, 6], [6, 12]])


def test_explicit_filter_rejects_no_matching_messages(tmp_path, eccodes):
    path = tmp_path / "field.grib2"
    write_grib(path, eccodes, [{}])
    with pytest.raises(ValueError, match="No GRIB messages"):
        open_grib(path, filter_by_keys={"level": 700})


def test_duplicate_messages_are_rejected_instead_of_last_message_winning(tmp_path, eccodes):
    path = tmp_path / "duplicates.grib2"
    write_grib(path, eccodes, [{}, {"values": np.full(6, 999.)}])
    with pytest.raises(ValueError, match="Duplicate"):
        open_grib(path)


def test_sparse_message_cube_requires_explicit_selection(tmp_path, eccodes):
    path = tmp_path / "sparse.grib2"
    write_grib(path, eccodes, [{"dataTime": 0, "step": 0}, {"dataTime": 0, "step": 6}, {"dataTime": 600, "step": 0}])
    with pytest.raises(ValueError, match="Incomplete"):
        open_grib(path)
    with open_grib(path, filter_by_keys={"dataTime": 0}) as result:
        assert result.sizes["time"] == 1
        assert result.sizes["step"] == 2


def test_grid_change_requires_explicit_group(tmp_path, eccodes):
    path = tmp_path / "grids.grib2"
    write_grib(path, eccodes, [{}, {"dataTime": 600, "latitudeOfFirstGridPointInDegrees": 20.,
                                  "latitudeOfLastGridPointInDegrees": 10.}])
    groups = inspect_grib_groups(path)
    assert len(groups) == 2
    with pytest.raises(ValueError, match="heterogeneous"):
        open_grib(path)
    with open_grib(path, filter_by_keys=groups[1]["filter_by_keys"]) as result:
        np.testing.assert_array_equal(result.latitude, [20., 10.])


def test_bitmap_missing_values_and_precise_packed_values_are_preserved(tmp_path, eccodes):
    path = tmp_path / "missing.grib2"
    values = np.array([250., np.nan, 250.1, 250.2, 250.3, 250.4])
    truth = write_grib(path, eccodes, [{"values": values}])[0]
    assert np.any(truth[np.isfinite(truth)] != truth[np.isfinite(truth)].astype("float32").astype("float64"))
    with open_grib(path) as result:
        np.testing.assert_allclose(result.t.compute().data.reshape(-1), truth, rtol=0, atol=0, equal_nan=True)


def test_grib_valid_zero_and_negative_values_survive_bitmap(tmp_path, eccodes):
    path = tmp_path / "zero.grib2"
    truth = write_grib(path, eccodes, [{"values": [-1., 0., 1., np.nan, 2., 3.]}])[0]
    with open_grib(path) as result:
        np.testing.assert_allclose(result.t.compute().data.reshape(-1), truth, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("chunks", [None, False, 0, "none", {"time": 0}, {"time": True}])
def test_invalid_chunks_rejected_without_backend_or_input(tmp_path, chunks):
    with pytest.raises((ValueError, TypeError), match="chunk"):
        open_grib(tmp_path / "missing.grib2", chunks=chunks)


@pytest.mark.parametrize("filters", [{"values": 0}, {"level": [500]}, {"level": True}, {"level": float("nan")}, []])
def test_unsafe_or_nonscalar_filters_rejected_without_reading_input(tmp_path, filters):
    with pytest.raises((ValueError, TypeError)):
        open_grib(tmp_path / "missing.grib2", filter_by_keys=filters)


def test_optional_backend_failure_is_actionable(tmp_path, monkeypatch):
    import climate_preprocessing.io.grib as module

    path = tmp_path / "input.grib2"
    path.touch()
    original = module.importlib.import_module

    def missing(name):
        if name == "eccodes":
            raise ImportError("native backend unavailable")
        return original(name)

    monkeypatch.setattr(module.importlib, "import_module", missing)
    with pytest.raises(ImportError, match=r"climate-preprocessing\[grib\]"):
        open_grib(path)


def test_changed_source_is_rejected_on_deferred_execution(tmp_path, eccodes):
    path = tmp_path / "source.grib2"
    write_grib(path, eccodes, [{}])
    with open_grib(path) as result:
        with path.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(ValueError, match="Input changed"):
            result.t.compute()


@pytest.mark.parametrize("extra", [b"stray", b"\x00"])
def test_non_message_tail_is_rejected(tmp_path, eccodes, extra):
    path = tmp_path / "tail.grib2"
    write_grib(path, eccodes, [{}])
    with path.open("ab") as stream:
        stream.write(extra)
    with pytest.raises(ValueError, match="trailing|non-message"):
        open_grib(path)


def test_unknown_chunk_dimension_rejected_and_no_index_created(tmp_path, eccodes):
    path = tmp_path / "unknown.grib2"
    write_grib(path, eccodes, [{}])
    with pytest.raises(ValueError, match="Unknown chunk"):
        open_grib(path, chunks={"unknown": 1})
    assert sorted(p.name for p in tmp_path.iterdir()) == [path.name]


def test_accumulation_windows_and_step_type_remain_explicit(tmp_path, eccodes):
    path = tmp_path / "accumulation.grib2"
    records = [{"stepType": "accum", "startStep": 0, "endStep": end} for end in (6, 12)]
    write_grib(path, eccodes, records)
    with open_grib(path) as result:
        variable = next(iter(result.data_vars.values()))
        assert variable.attrs["GRIB_stepType"] == "accum"
        np.testing.assert_array_equal(result.step.data, np.array([6, 12], dtype="timedelta64[h]"))
        event = json.loads(result.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]
        group = event["parameters"]["source_groups"][0]
        assert group["step_windows"] == [[0, 6], [0, 12]]
        assert group["forecast_windows"] == [[20000101, 0, 0, 6], [20000101, 0, 0, 12]]
        assert group["filter_by_keys"]["stepUnits"] == 1  # GRIB hours, not calendar months


def test_mixed_step_types_require_filtering(tmp_path, eccodes):
    path = tmp_path / "step_types.grib2"
    write_grib(path, eccodes, [{"step": 6}, {"stepType": "accum", "startStep": 0, "endStep": 6}])
    with pytest.raises(ValueError, match="heterogeneous"):
        open_grib(path)
    with open_grib(path, filter_by_keys={"stepType": "accum"}) as result:
        assert all(variable.attrs["GRIB_stepType"] == "accum" for variable in result.data_vars.values())


def test_conflicting_statistical_windows_cannot_share_a_forecast_slot(tmp_path, eccodes):
    path = tmp_path / "windows.grib2"
    write_grib(path, eccodes, [
        {"stepType": "accum", "startStep": 0, "endStep": 6},
        {"stepType": "accum", "startStep": 3, "endStep": 6},
    ])
    with pytest.raises(ValueError, match="conflicting statistical windows"):
        open_grib(path)


def test_inspection_never_requests_field_arrays(tmp_path, eccodes, monkeypatch):
    path = tmp_path / "inspect.grib2"
    write_grib(path, eccodes, [{}, {"level": 1000}])
    original = eccodes.codes_get_array

    def read_array(message, key, *args, **kwargs):
        assert key not in {"values", "codedValues", "bitmap"}
        return original(message, key, *args, **kwargs)

    monkeypatch.setattr(eccodes, "codes_get_array", read_array)
    groups = inspect_grib_groups(path)
    assert groups[0]["messages"] == 2
    assert groups[0]["levels"] == [500., 1000.]


def test_backend_dataset_closed_when_post_open_validation_fails(tmp_path, eccodes, monkeypatch):
    import climate_preprocessing.io.grib as module

    path = tmp_path / "close.grib2"
    write_grib(path, eccodes, [{}])
    original = module.xr.open_dataset
    closed = []

    def opened(*args, **kwargs):
        result = original(*args, **kwargs)
        original_close = result._close

        def close():
            closed.append(True)
            if original_close is not None:
                original_close()

        result.set_close(close)
        return result

    monkeypatch.setattr(module.xr, "open_dataset", opened)
    with pytest.raises(ValueError, match="Unknown chunk"):
        open_grib(path, chunks={"unknown": 1})
    assert closed == [True]


def test_native_minute_step_filters_do_not_use_cfgrib_hour_aliases(tmp_path, eccodes):
    path = tmp_path / "minutes.grib2"
    write_grib(path, eccodes, [{"stepUnits": 0, "endStep": 30}, {"stepUnits": 0, "endStep": 90}])
    group = inspect_grib_groups(path)[0]
    assert group["filter_by_keys"]["stepUnits"] == 0
    with open_grib(path, filter_by_keys={**group["filter_by_keys"], "endStep": 30}) as result:
        np.testing.assert_array_equal(result.step.data, np.array([30], dtype="timedelta64[m]"))
        assert result.sizes["step"] == 1


@pytest.mark.parametrize("second_parameter", [130, 131])
def test_multifield_physical_message_is_rejected_before_any_field_can_be_lost(tmp_path, eccodes, second_parameter):
    path = tmp_path / "multifield.grib2"
    separate = tmp_path / "separate.grib2"
    write_grib(separate, eccodes, [{}, {"paramId": second_parameter, "values": np.full(6, 350.)}])
    multi = eccodes.codes_grib_multi_new()
    try:
        with separate.open("rb") as stream:
            for index in range(2):
                handle = eccodes.codes_grib_new_from_file(stream)
                try:
                    eccodes.codes_grib_multi_append(handle, 0 if index == 0 else 4, multi)
                finally:
                    eccodes.codes_release(handle)
        with path.open("wb") as stream:
            eccodes.codes_grib_multi_write(multi, stream)
    finally:
        eccodes.codes_grib_multi_release(multi)
    with pytest.raises(ValueError, match="Multi-field GRIB2"):
        inspect_grib_groups(path)
    with pytest.raises(ValueError, match="Multi-field GRIB2"):
        open_grib(path)


def test_nonzero_reference_seconds_are_rejected_instead_of_truncated(tmp_path, eccodes):
    path = tmp_path / "seconds.grib2"
    write_grib(path, eccodes, [{"second": 30}])
    with pytest.raises(ValueError, match="nonzero seconds"):
        open_grib(path)
