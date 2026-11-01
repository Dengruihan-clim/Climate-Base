"""Small local fixtures test the HPC runner, not claims about real-data coverage."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def validator():
    path = Path(__file__).resolve().parents[1] / "tools" / "validate_real_data.py"
    spec = importlib.util.spec_from_file_location("validate_real_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_input(path, year):
    values = np.arange(2 * 3 * 20 * 40, dtype=np.float32).reshape(2, 3, 20, 40) - 100
    values[0, 0, 5, 1] = np.nan
    dataset = xr.Dataset(
        {"V": (("time", "level", "latitude", "longitude"), values,
               {"units": "m s**-1", "minimum_value": -year}),
         "utc_date": ("time", [year * 1000000 + 10100, year * 1000000 + 20100])},
        coords={"time": np.array([f"{year}-01-01", f"{year}-02-01"], dtype="datetime64[ns]"),
                "level": ("level", [1., 2., 3.], {"units": "hPa", "positive": "down"}),
                "latitude": ("latitude", np.linspace(10, -10, 20), {"units": "degrees_north"}),
                "longitude": ("longitude", np.arange(40, dtype=float), {"units": "degrees_east"})},
        attrs={"history": f"converted {year}", "rda_dataset_group": "ERA5 monthly mean test fixture"},
    )
    dataset.to_netcdf(path, engine="netcdf4", encoding={"V": {"zlib": True, "chunksizes": (1, 3, 20, 40)}})
    return path


def test_runner_reports_each_real_data_stage(tmp_path, validator):
    files = [make_input(tmp_path / f"{year}.nc", year) for year in (2003, 2004)]
    before = [validator.stamp(path) for path in files]
    output = tmp_path / "report"
    code = validator.main(["--input", str(files[0]), "--input", str(files[1]),
                           "--output-dir", str(output), "--pressure", "level", "--monthly-product"])
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert code == 0, [case for case in report["cases"] if case["status"] == "FAIL"]
    cases = {case["case"]: case for case in report["cases"]}
    assert cases["original_multifile_contract"]["status"] == "EXPECTED_REJECTION"
    assert cases["derived_multifile_time_sort"]["status"] == "PASS"
    assert cases["monthly_archive_derived_roundtrip"]["status"] == "PASS"
    assert cases["field_laziness"]["details"]["field_blocks_during_construction"] == 0
    assert cases["write_existing_output_rejected"]["status"] == "EXPECTED_REJECTION"
    assert [validator.stamp(path) for path in files] == before
    assert len((output / "results.jsonl").read_text(encoding="utf-8").splitlines()) == len(report["cases"])
    assert report["metadata"]["sources"][0]["selected_dimensions"] == {
        "time": 1, "level": 2, "latitude": 16, "longitude": 32}


def test_native_chunk_budget_skips_computation_and_keeps_report(tmp_path, validator):
    path = make_input(tmp_path / "input.nc", 2003)
    output = tmp_path / "report"
    assert validator.main(["--input", str(path), "--output-dir", str(output), "--pressure", "level",
                           "--allow-native-chunk-mib", "0.001"]) == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    cases = {case["case"]: case for case in report["cases"]}
    assert cases["source_1_bounded_read"]["status"] == "NOT_APPLICABLE"
    assert cases["original_sources_unchanged"]["status"] == "PASS"
    assert cases["monthly_archive"]["status"] == "NOT_APPLICABLE"
    assert not (output / "prepared_sample.nc").exists()


def test_report_continues_after_case_failure(tmp_path, validator):
    report = validator.Report(tmp_path / "report")
    report.case("failure", "test", lambda: 1 / 0)
    report.case("continued", "test", lambda: {"continued": True})
    assert report.finish() == 1
    assert [case["status"] for case in report.cases] == ["FAIL", "PASS"]


def test_unknown_member_dimension_needs_explicit_selection(validator):
    dataset = xr.Dataset(
        {"V": (("time", "member", "latitude", "longitude"), np.ones((1, 2, 2, 2)))},
        coords={"time": [0], "latitude": [1., 0.], "longitude": [0., 1.]},
    )
    args = validator.parser().parse_args(["--input", "unused", "--output-dir", "unused"])
    with pytest.raises(validator.NotApplicable, match="--select"):
        validator.selection(dataset, args)
    args.select = ["member=1"]
    _, sample, _ = validator.selection(dataset, args)
    assert sample.sizes["member"] == 1
