"""Product selection precedes strict combination; IO and CLI preserve semantics."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path

from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import open_dataset, prepare_dataset
from climate_preprocessing.cli import main
from climate_preprocessing.config import load_prepare_config


PRODUCT = "era5-monthly-pressure-levels"


def options(**kwargs):
    return {"product": PRODUCT, "variables": {"northward_wind": "V"}, **kwargs}


def fixture(month="2001-01-01", versions=(1, 5), *, time="time"):
    values = np.array(versions, dtype=float)[None, :, None, None, None] * 100
    values = values + np.arange(8).reshape(1, 1, 2, 2, 2)
    return xr.Dataset({
        "V": ((time, "expver", "level", "latitude", "longitude"), values,
              {"units": "m s**-1", "cell_methods": f"{time}: mean (comment: {time}: original)",
               "rda_dataset": "ds633.1", "ecmwf_parameter": 132}),
        "utc_date": (time, [2001010100], {"long_name": "Auxiliary source date code"}),
    }, coords={
        time: np.array([month], dtype="datetime64[ns]"),
        "expver": list(versions),
        "level": ("level", [500., 850.], {"units": "hPa"}),
        "latitude": ("latitude", [20., 10.], {"units": "degrees_north"}),
        "longitude": ("longitude", [-10., 10.], {"units": "degrees_east"}),
    })


def test_per_file_selection_precedes_strict_combination(tmp_path):
    late = fixture("2001-02-01", (5, 1), time="valid_time")
    early = fixture("2001-01-01", (1,), time="valid_time")
    late.to_netcdf(tmp_path / "late.nc")
    early.to_netcdf(tmp_path / "early.nc")
    graphs = []
    with Callback(start=lambda graph: graphs.append(tuple(graph))):
        result = prepare_dataset(
            [tmp_path / "late.nc", tmp_path / "early.nc"],
            adapter=options(coordinates={"time": "valid_time"}, selectors={"expver": 1}),
            expected_time=["2001-01-01", "2001-02-01"], chunks={"valid_time": 1},
        )
    try:
        assert not graphs
        assert result.expver.ndim == 0 and result.expver.item() == 1
        assert result.northward_wind.attrs["cell_methods"] == "time: mean (comment: valid_time: original)"
        assert "utc_date" in result
        expected = np.broadcast_to(np.arange(8).reshape(1, 2, 2, 2) + 100, (2, 2, 2, 2))[:, :, ::-1, ::-1]
        np.testing.assert_array_equal(result.northward_wind.compute(), expected)
        np.testing.assert_array_equal(result.longitude, [10, 350])
        provenance = json.loads(result.attrs["climate_preprocessing_provenance"])
        combined = next(item for item in provenance["operations"] if item["operation"] == "open_mfdataset")
        assert len(combined["inputs"]) == 2
        for source in combined["inputs"]:
            assert any(item["operation"] == "adapt_product" for item in source["provenance"]["operations"])
    finally:
        result.close()


@pytest.mark.parametrize("change", ["grid", "unit", "calendar", "duplicate_time"])
def test_adapter_does_not_relax_multifile_conflicts(tmp_path, change):
    first, second = fixture(), fixture("2001-02-01")
    if change == "grid":
        second = second.assign_coords(latitude=("latitude", [21., 10.], {"units": "degrees_north"}))
    elif change == "unit":
        second.V.attrs["units"] = "K"
    elif change == "calendar":
        second.time.encoding["calendar"] = "noleap"
    else:
        second = fixture()
    first.to_netcdf(tmp_path / "first.nc")
    second.to_netcdf(tmp_path / "second.nc")
    with pytest.raises(ValueError):
        prepare_dataset([tmp_path / "first.nc", tmp_path / "second.nc"], adapter=options(selectors={"expver": 1}))


def test_adapter_expected_time_checks_entire_series_before_selection(tmp_path):
    fixture().to_netcdf(tmp_path / "input.nc")
    with pytest.raises(ValueError, match="expected index"):
        prepare_dataset(tmp_path / "input.nc", adapter=options(selectors={"expver": 1}),
                        expected_time=["2000-12-01", "2001-01-01"], time_range={"start": "2001-01-01"})


def test_monthly_record_uniqueness_is_checked_across_files(tmp_path):
    fixture("2001-01-01").to_netcdf(tmp_path / "first.nc")
    fixture("2001-01-15").to_netcdf(tmp_path / "second.nc")
    with pytest.raises(ValueError, match="month"):
        prepare_dataset([tmp_path / "first.nc", tmp_path / "second.nc"],
                        adapter=options(selectors={"expver": 1}))


def test_preserve_sources_keeps_raw_ranges_before_stream_selection(tmp_path):
    for index, month in enumerate(("2001-01-01", "2001-02-01")):
        source = fixture(month)
        source.attrs["history"] = f"synthetic processing branch {index}"
        source.V.attrs["actual_range"] = [index, 1000 + index]
        source.to_netcdf(tmp_path / f"source{index}.nc")
    with prepare_dataset([tmp_path / "source0.nc", tmp_path / "source1.nc"],
                         adapter=options(selectors={"expver": 1}), metadata_policy="preserve_sources") as result:
        assert "actual_range" not in result.northward_wind.attrs
        operations = json.loads(result.attrs["climate_preprocessing_provenance"])["operations"]
        records = next(item for item in operations if item["operation"] == "open_mfdataset")["inputs"]
        for index, record in enumerate(records):
            original = record["source_metadata"]
            assert original["variable_attributes"]["V"]["actual_range"] == [index, 1000 + index]
            assert original["global_attributes"]["history"] == f"synthetic processing branch {index}"


def test_config_and_cli_inspect_then_prepare(tmp_path):
    fixture(time="valid_time").to_netcdf(tmp_path / "input.nc")
    config = {"schema_version": 1, "input": "input.nc", "output": "output.nc",
              "adapter": options(coordinates={"time": "valid_time"}, selectors={"expver": 5})}
    path = tmp_path / "prepare.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    graphs = []
    with Callback(start=lambda graph: graphs.append(tuple(graph))), redirect_stdout(io.StringIO()) as output:
        assert main(["inspect", str(path)]) == 0
    summary = json.loads(output.getvalue())
    assert summary["product_adapter"] == PRODUCT
    assert not summary["field_values_read"] and not summary["outputs_written"] and not graphs
    assert not (tmp_path / "output.nc").exists()
    with redirect_stdout(io.StringIO()):
        assert main(["prepare", str(path)]) == 0
    with open_dataset(tmp_path / "output.nc") as result:
        np.testing.assert_array_equal(result.northward_wind.compute(),
                                      (np.arange(8).reshape(1, 2, 2, 2) + 500)[:, :, ::-1, ::-1])
        assert result.northward_wind.attrs["units"] == "m s**-1"
        assert result.expver.item() == 5
    with pytest.raises(FileExistsError):
        main(["prepare", str(path)])


@pytest.mark.parametrize("adapter", [
    {"product": "wrf", "variables": {"northward_wind": "V"}},
    {"product": PRODUCT, "variables": {"northward_wind": "V"}, "merge_expver": True},
    {"product": PRODUCT, "variables": {"northward_wind": "V"}, "selectors": {"expver": [1, 5]}},
])
def test_config_rejects_unsupported_adapter_before_opening(tmp_path, adapter):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema_version": 1, "input": "missing.nc", "adapter": adapter}), encoding="utf-8")
    with pytest.raises(ValueError):
        load_prepare_config(path)


@pytest.mark.parametrize("kwargs", [{"time": "valid_time"}, {"coordinates": {"latitude": "lat"}}])
def test_top_level_options_use_post_adapter_names(kwargs):
    with pytest.raises(ValueError, match="canonical"):
        prepare_dataset("missing.nc", adapter=options(), **kwargs)


def test_raw_grib_is_not_claimed_as_monthly_product_adaptation():
    with pytest.raises(ValueError, match="not raw GRIB"):
        prepare_dataset("missing.grib", input_format="grib", adapter=options())


def test_single_failure_closes_the_opened_source(monkeypatch):
    source = fixture()
    closed = []
    source.set_close(lambda: closed.append(True))
    monkeypatch.setattr("climate_preprocessing.pipeline.open_dataset", lambda *a, **k: source)
    with pytest.raises(ValueError):
        prepare_dataset("unused", adapter=options())  # expver declaration required
    assert closed == [True]


def test_multifile_failure_closes_successful_and_failed_branches(tmp_path, monkeypatch):
    from climate_preprocessing.io import multifile

    for name in ("first", "second"):
        (tmp_path / f"{name}.nc").touch()
    sources = [fixture(), fixture("2001-02-01", (5,))]
    closed = []
    for index, source in enumerate(sources):
        source.set_close(lambda i=index: closed.append(i))
    monkeypatch.setattr(multifile, "open_dataset", lambda *a, **k: sources.pop(0))
    with pytest.raises(ValueError):
        prepare_dataset([tmp_path / "first.nc", tmp_path / "second.nc"], adapter=options(selectors={"expver": 1}))
    assert sorted(closed) == [0, 1]


def test_product_example_and_generated_cli(tmp_path):
    path = Path(__file__).parents[1] / "examples/product_adapters.py"
    spec = importlib.util.spec_from_file_location("example_product_adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / "example"
    module.run_example(directory)
    with redirect_stdout(io.StringIO()):
        assert main(["inspect", str(directory / "prepare.json")]) == 0
        assert main(["prepare", str(directory / "prepare.json")]) == 0
    with open_dataset(directory / "prepared.nc") as result:
        np.testing.assert_array_equal(result.northward_wind.compute(), 7.)
        assert result.expver.item() == 1


@pytest.mark.zarr
def test_derived_local_zarr_preserves_explicit_product_selection(tmp_path):
    backend = pytest.importorskip("zarr")
    if not backend.__version__.startswith("2."):
        pytest.skip("The local writer contract requires Zarr v2")
    from climate_preprocessing import write_zarr

    write_zarr(fixture(), tmp_path / "input.zarr")
    with prepare_dataset(tmp_path / "input.zarr", input_format="zarr",
                         adapter=options(selectors={"expver": 1})) as result:
        np.testing.assert_array_equal(result.northward_wind.compute(),
                                      (np.arange(8).reshape(1, 2, 2, 2) + 100)[:, :, ::-1, ::-1])
        assert result.expver.item() == 1


@pytest.mark.vertical
def test_adapter_then_optional_vertical_interpolation(tmp_path):
    from climate_preprocessing.vertical import _metpy

    try:
        _metpy.backend_versions()
    except ImportError as error:
        pytest.skip(str(error))
    fixture().to_netcdf(tmp_path / "input.nc")
    with prepare_dataset(tmp_path / "input.nc", adapter=options(selectors={"expver": 1}),
                         vertical={"target_levels": [675], "variables": ["northward_wind"], "method": "linear"}) as result:
        np.testing.assert_array_equal(result.northward_wind.compute(), np.array([[[[105, 104], [103, 102]]]]))
