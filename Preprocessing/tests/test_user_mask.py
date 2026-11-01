"""Explicit user-mask contracts: labelled alignment, provenance and lazy I/O."""

import json

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.io import open_dataset, write_dataset
from climate_preprocessing.masking import apply_mask
from climate_preprocessing.provenance import PROVENANCE_ATTRIBUTE


def source():
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3) - 5
    return xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"), values,
                         {"units": "K", "standard_name": "air_temperature", "actual_range": [-5., 6.],
                          "minimum_value": -5., "maximum_value": 6., "valid_range": [-100., 100.],
                          "ancillary_variables": "qc uncertainty", "grid_mapping": "crs"}),
         "uncertainty": (("latitude", "longitude"), np.ones((2, 3)), {"units": "K"}),
         "qc": (("latitude", "longitude"), np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int16),
                {"flag_values": [0, 1], "flag_meanings": "good bad"}),
         "keep": (("latitude", "longitude"), [[True, False, True], [False, True, True]]),
         "latitude_bounds": (("latitude", "vertex"), [[-35., -25.], [25., 35.]]),
         "crs": ((), 0, {"grid_mapping_name": "latitude_longitude"})},
        coords={"time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [-30., 30.], {"units": "degrees_north", "bounds": "latitude_bounds"}),
                "longitude": ("longitude", [0., 120., 240.], {"units": "degrees_east"})},
    )


def event(dataset):
    return json.loads(dataset.attrs[PROVENANCE_ATTRIBUTE])["operations"][-1]


def test_user_mask_broadcasts_without_changing_qc_mask_or_source_metadata():
    dataset = source().chunk({"time": 1, "latitude": 1})
    dataset.temperature.encoding = {"dtype": "int16", "_FillValue": -32768,
                                    "coordinates": "latitude longitude", "ancillary_variables": "qc uncertainty"}
    original_attrs = dict(dataset.temperature.attrs)
    original_encoding = dict(dataset.temperature.encoding)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = apply_mask(dataset, "keep", variables=["temperature"], reason="Keep cells selected by supplied mask v1")
    assert not tasks
    assert isinstance(result.temperature.data, da.Array)
    assert result.temperature.dims == dataset.temperature.dims
    expected = np.where(dataset.keep.compute().values[None, :, :], dataset.temperature.compute().values, np.nan)
    np.testing.assert_allclose(result.temperature.compute(), expected, equal_nan=True)
    assert result.temperature.attrs["units"] == "K"
    assert result.temperature.attrs["standard_name"] == "air_temperature"
    assert result.temperature.attrs["valid_range"] == [-100., 100.]
    assert not {"actual_range", "minimum_value", "maximum_value"}.intersection(result.temperature.attrs)
    assert result.temperature.encoding == {"coordinates": "latitude longitude", "ancillary_variables": "qc uncertainty"}
    for name in ("qc", "keep", "uncertainty", "latitude_bounds", "crs"):
        xr.testing.assert_identical(result[name].compute(), dataset[name].compute())
    assert dataset.temperature.attrs == original_attrs
    assert dataset.temperature.encoding == original_encoding
    operation = event(result)
    assert operation["operation"] == "apply_mask"
    assert operation["parameters"]["alignment"] == "exact"
    assert operation["parameters"]["mask"] == {
        "kind": "dataset_variable", "name": "keep", "dimensions": ["latitude", "longitude"], "shape": [2, 3]}
    assert "attrs.actual_range" in operation["parameters"]["removed_range_attributes"]["temperature"]
    assert "dtype" in operation["parameters"]["removed_encoding_keys"]["temperature"]


def test_external_mask_and_field_are_not_evaluated_until_compute():
    calls = []

    @delayed
    def values():
        calls.append("field")
        return np.array([0., -2., 4.])

    @delayed
    def selected():
        calls.append("mask")
        return np.array([True, True, False])

    dataset = xr.Dataset({"field": ("x", da.from_delayed(values(), (3,), float), {"units": "K"})},
                         coords={"x": [10, 20, 30]})
    mask = xr.DataArray(da.from_delayed(selected(), (3,), bool), dims="x", coords={"x": [10, 20, 30]},
                        name="land_selection_v2")
    result = apply_mask(dataset, mask, variables="field", reason="External land-selection product v2; land=True")
    assert calls == []
    np.testing.assert_allclose(result.field.compute(), [0., -2., np.nan], equal_nan=True)
    assert sorted(calls) == ["field", "mask"]
    assert event(result)["parameters"]["mask"]["kind"] == "external_dataarray"
    assert event(result)["parameters"]["mask"]["name"] == "land_selection_v2"


@pytest.mark.parametrize("coordinates", [[240., 120., 0.], [0., 120., 241.], [0., 120.], [0., 120., 120.]])
def test_coordinate_misalignment_or_duplicates_are_rejected(coordinates):
    mask = xr.DataArray(np.ones(len(coordinates), dtype=bool), dims="longitude", coords={"longitude": coordinates})
    with pytest.raises(ValueError, match="coordinate|labels"):
        apply_mask(source(), mask, variables="temperature", reason="Explicit external mask")


@pytest.mark.parametrize("missing", ["mask", "target"])
def test_unlabelled_positional_alignment_is_not_allowed(missing):
    dataset = xr.Dataset({"field": ("x", [1., 2.])}, coords={"x": [0, 1]})
    mask = xr.DataArray([True, False], dims="x", coords={"x": [0, 1]})
    if missing == "mask":
        mask = mask.drop_vars("x")
    else:
        dataset = dataset.drop_vars("x")
    with pytest.raises(ValueError, match="coordinate"):
        apply_mask(dataset, mask, variables="field", reason="Coordinate omission test")


def test_mask_dimensions_must_be_subset_of_every_selected_target():
    dataset = source()
    time_mask = xr.DataArray([True, False], dims="time", coords={"time": dataset.time})
    with pytest.raises(ValueError, match="dimensions absent"):
        apply_mask(dataset, time_mask, variables=["temperature", "uncertainty"], reason="Time QC rule")


@pytest.mark.parametrize("mask", [np.array([True, False]), [True, False], xr.DataArray([0, 1], dims="x"),
                                 xr.DataArray([1., np.nan], dims="x")])
def test_mask_requires_explicit_boolean_dataarray(mask):
    with pytest.raises(TypeError, match="boolean|DataArray"):
        apply_mask(source(), mask, variables="temperature", reason="Invalid mask type")


@pytest.mark.parametrize("variables", [None, [], (), "", ["temperature", "temperature"], [123]])
def test_targets_must_be_explicit_nonempty_names(variables):
    with pytest.raises((TypeError, ValueError), match="variables"):
        apply_mask(source(), "keep", variables=variables, reason="Explicit selection required")


@pytest.mark.parametrize("reason", [None, "", " ", 123])
def test_reason_is_required_for_traceability(reason):
    with pytest.raises(ValueError, match="reason"):
        apply_mask(source(), "keep", variables="temperature", reason=reason)


@pytest.mark.parametrize("name,match", [("latitude", "data variable"), ("latitude_bounds", "structural"),
                                      ("crs", "structural"), ("qc", "QC flag"), ("keep", "numeric|condition")])
def test_coordinate_structure_flags_and_condition_cannot_be_targets(name, match):
    with pytest.raises(ValueError, match=match):
        apply_mask(source(), "keep", variables=name, reason="Reject protected variable")


def test_numeric_ancillary_can_be_explicitly_masked_and_integer_field_promotes():
    dataset = source()
    dataset["count"] = (("latitude", "longitude"), np.arange(6, dtype=np.int16).reshape(2, 3))
    result = apply_mask(dataset, "keep", variables=["uncertainty", "count"], reason="Explicitly selected quantities")
    assert result["count"].dtype.kind == "f"
    np.testing.assert_allclose(result["count"].compute(), [[0, np.nan, 2], [np.nan, 4, 5]], equal_nan=True)
    np.testing.assert_allclose(result.uncertainty.compute(), [[1, np.nan, 1], [np.nan, 1, 1]], equal_nan=True)
    xr.testing.assert_identical(result.qc.compute(), dataset.qc)


def test_scalar_mask_and_matching_auxiliary_coordinates_are_explicit():
    dataset = source().assign_coords(height=2.)
    matching = xr.DataArray(False, coords={"height": 2.}, name="whole_field_rejection")
    result = apply_mask(dataset, matching, variables="temperature", reason="Declared rejection of whole field")
    assert np.isnan(result.temperature.compute()).all()
    assert result.height == 2.
    with pytest.raises(ValueError, match="auxiliary coordinate"):
        apply_mask(dataset, matching.assign_coords(height=10.), variables="temperature", reason="Wrong measurement height")
    with pytest.raises(ValueError, match="auxiliary coordinate"):
        apply_mask(dataset, xr.DataArray(True, coords={"new_coordinate": 1}), variables="temperature",
                   reason="No new coordinate may be introduced")


def test_unsafe_packing_is_removed_and_masked_fields_roundtrip(tmp_path):
    dataset = source()
    dataset.temperature.encoding.update(dtype="int16", scale_factor=0.1, add_offset=273., _FillValue=-32768)
    output = tmp_path / "masked.nc"
    result = apply_mask(dataset, "keep", variables="temperature", reason="Supplied quality selection v3")
    assert "valid_range" not in result.temperature.attrs
    assert not {"dtype", "scale_factor", "add_offset", "_FillValue"}.intersection(result.temperature.encoding)
    write_dataset(result, output)
    with open_dataset(output) as reopened:
        np.testing.assert_allclose(reopened.temperature.compute(), result.temperature.compute(), equal_nan=True)
        assert reopened.temperature.attrs["units"] == "K"
        links = {**reopened.temperature.attrs, **reopened.temperature.encoding}
        assert links["grid_mapping"] == "crs"
        assert links["ancillary_variables"] == "qc uncertainty"
        operations = json.loads(reopened.attrs[PROVENANCE_ATTRIBUTE])["operations"]
        assert next(item for item in operations if item["operation"] == "apply_mask")["parameters"]["reason"] == (
            "Supplied quality selection v3")
        np.testing.assert_array_equal(reopened.qc.compute(), dataset.qc)
        assert reopened.qc.dtype == dataset.qc.dtype
        np.testing.assert_array_equal(reopened.qc.attrs["flag_values"], dataset.qc.attrs["flag_values"])
        assert reopened.qc.attrs["flag_meanings"] == dataset.qc.attrs["flag_meanings"]


def test_raw_packing_attributes_require_decoding_before_masking():
    dataset = source()
    dataset.temperature.attrs["scale_factor"] = .1
    with pytest.raises(ValueError, match="Decode"):
        apply_mask(dataset, "keep", variables="temperature", reason="Input must be decoded first")


def test_result_closes_dataset_but_external_mask_is_owned_by_caller():
    dataset = source()
    closed = []
    dataset.set_close(lambda: closed.append("dataset"))
    mask = dataset.keep.copy()
    mask.set_close(lambda: closed.append("mask"))
    result = apply_mask(dataset, mask, variables="temperature", reason="External mask with independent resource")
    assert closed == []
    result.close()
    assert closed == ["dataset"]
    mask.close()
    assert closed == ["dataset", "mask"]


def test_rejected_alignment_does_not_close_caller_owned_resources():
    dataset = source()
    closed = []
    dataset.set_close(lambda: closed.append("dataset"))
    mask = dataset.keep.isel(longitude=slice(0, 2))
    mask.set_close(lambda: closed.append("mask"))
    with pytest.raises(ValueError, match="exactly match"):
        apply_mask(dataset, mask, variables="temperature", reason="Reject partial labels")
    assert closed == []


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_precision_check_only_applies_to_retained_values(dtype):
    dataset = xr.Dataset({"count": ("x", np.array([2**53 + 1, np.iinfo(dtype).max, 3], dtype=dtype))},
                         coords={"x": [0, 1, 2]}).chunk({"x": 1})
    mask = xr.DataArray([True, False, True], dims="x", coords={"x": dataset.x})
    calls = []
    with Callback(pretask=lambda *args: calls.append(True)):
        result = apply_mask(dataset, mask, variables="count", reason="Counts must retain exact integer values")
    assert not calls
    with pytest.raises(ValueError, match="represented exactly"):
        result["count"].compute()
    mask = mask.copy(data=np.array([False, False, True]))
    result = apply_mask(dataset, mask, variables="count", reason="Only a representable count is retained")
    np.testing.assert_allclose(result["count"].compute(), [np.nan, np.nan, 3.], equal_nan=True)
    all_missing = apply_mask(dataset, xr.DataArray(False), variables="count", reason="All counts are excluded")
    assert np.isnan(all_missing["count"].compute()).all()


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_exactly_representable_large_integers_survive_masking(dtype):
    dataset = xr.Dataset({"count": ("x", np.array([2**53 + 2], dtype=dtype))}, coords={"x": [0]})
    result = apply_mask(dataset, xr.DataArray(True), variables="count", reason="Representable large count")
    assert int(result["count"].compute().item()) == 2**53 + 2
