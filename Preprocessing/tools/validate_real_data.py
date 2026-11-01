"""Bounded, offline real-NetCDF validation for a shared HPC login node.

References: xarray backend indexing and Dask scheduling documentation:
https://docs.xarray.dev/en/stable/user-guide/io.html
https://docs.dask.org/en/stable/scheduling.html

Climate-Base improvements: explicit tiny source hyperslabs, storage-chunk budget,
per-case durable reports, unchanged-source checks, and a clear distinction between
original-input contracts and tests on derived real-data subsets. No statistical
aggregation, full-input hashing, source rewriting, or implicit member selection.
Original lazy graphs are inspected without computing their large storage chunks.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import faulthandler
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# Set before importing numerical libraries; the script owns this process only.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import dask  # noqa: E402 -- numerical imports must follow thread limits
import dask.array as da  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

import climate_preprocessing as cp  # noqa: E402


class NotApplicable(Exception):
    """A declared prerequisite or resource budget prevents this case."""


class ExpectedRejection(Exception):
    """The API rejected an independently verified input-contract violation."""


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class Report:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.cases = []
        self.metadata = {}

    def case(self, name, basis, operation):
        started = time.monotonic()
        result = {"case": name, "data_basis": basis, "status": "PASS"}
        value = None
        try:
            value = operation()
            if isinstance(value, dict):
                result["details"] = value
        except (NotApplicable, ExpectedRejection) as error:
            result.update(status=("NOT_APPLICABLE" if isinstance(error, NotApplicable)
                                  else "EXPECTED_REJECTION"), reason=str(error))
        except Exception as error:
            result.update(status="FAIL", error=type(error).__name__, reason=str(error),
                          traceback=traceback.format_exc())
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        self.cases.append(result)
        with (self.directory / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(result, default=json_default, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(result, default=json_default, ensure_ascii=False), flush=True)
        return value

    def finish(self):
        counts = {status: sum(case["status"] == status for case in self.cases)
                  for status in ("PASS", "FAIL", "NOT_APPLICABLE", "EXPECTED_REJECTION")}
        result = {"metadata": self.metadata, "counts": counts, "cases": self.cases}
        (self.directory / "report.json").write_text(
            json.dumps(result, default=json_default, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return 1 if counts["FAIL"] else 0


def stamp(path):
    info = Path(path).stat()
    return {"basename": Path(path).name, "size_bytes": info.st_size, "mtime_ns": info.st_mtime_ns}


def parse_select(items):
    result = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or name in result:
            raise ValueError("--select requires unique DIM=NONNEGATIVE_INDEX entries")
        result[name] = int(value)
        if result[name] < 0:
            raise ValueError("--select indices must be nonnegative")
    return result


def selection(dataset, args):
    """Use one time, two levels and a contiguous 16 x 32 native-grid hyperslab."""
    if args.variable not in dataset.data_vars:
        raise ValueError(f"Missing requested field {args.variable!r}")
    indexers = {}
    roles = ((args.time, 1), (args.latitude, 16), (args.longitude, 32))
    if args.pressure:
        roles += ((args.pressure, 2),)
    for name, length in roles:
        if name not in dataset.coords or dataset[name].ndim != 1:
            raise ValueError(f"Expected one-dimensional coordinate {name!r}")
        dimension = dataset[name].dims[0]
        size = dataset.sizes[dimension]
        if size == 0:
            raise ValueError(f"Empty dimension {dimension!r}")
        start = max(0, (size - length) // 2) if name == args.latitude else 0
        indexers[dimension] = slice(start, min(size, start + length))
    for dimension, index in parse_select(args.select).items():
        if dimension in indexers or dimension not in dataset.sizes:
            raise ValueError(f"Invalid or already bounded selection dimension {dimension!r}")
        if index >= dataset.sizes[dimension]:
            raise ValueError(f"Selection index exceeds dimension {dimension!r}")
        indexers[dimension] = slice(index, index + 1)
    for dimension in dataset[args.variable].dims:
        if dimension not in indexers:
            if dataset.sizes[dimension] != 1:
                raise NotApplicable(f"Unselected dimension {dimension!r}; provide --select DIM=INDEX")
            indexers[dimension] = slice(0, 1)
    selected = dataset.isel(indexers)
    # Bound all retained fields and CF-related auxiliary variables, not just V.
    estimated = sum(variable.size * max(variable.dtype.itemsize, 8)
                    for variable in selected.variables.values())
    if estimated > 16 * 1024 ** 2:
        raise NotApplicable(f"Selected dataset exceeds 16 MiB payload budget: {estimated} bytes")
    return indexers, selected, estimated


def inspect_source(path, args):
    with xr.open_dataset(path, engine="netcdf4", chunks=None, cache=False,
                         decode_coords="all") as dataset:
        indexers, selected, estimated = selection(dataset, args)
        chunks = {}
        native_peak = 0
        variables = {}
        for name, variable in dataset.variables.items():
            native = variable.encoding.get("chunksizes")
            native_bytes = (math.prod(native) * variable.dtype.itemsize
                            if native else 0)
            if name in dataset.data_vars:
                native_peak = max(native_peak, native_bytes)
            variables[name] = {"dimensions": variable.dims, "dtype": str(variable.dtype),
                               "attrs": dict(variable.attrs), "native_chunks": native,
                               "native_decoded_chunk_bytes_conservative": native_bytes}
        # Match storage chunks for metadata-only graph construction. Small field
        # sampling below uses backend indexing BEFORE Dask graph construction.
        variable = dataset[args.variable]
        native = variable.encoding.get("chunksizes")
        for dimension, length in zip(variable.dims, native or variable.shape):
            chunks[dimension] = int(length)
        partitions = sum(math.prod(math.ceil(dataset.sizes[d] / chunks.get(d, dataset.sizes[d]))
                                   for d in variable.dims) for variable in dataset.data_vars.values())
        times = dataset[args.time]
        manifest = {**stamp(path), "dimensions": dict(dataset.sizes), "global_attrs": dict(dataset.attrs),
                    "variables": variables, "data_variables": list(dataset.data_vars),
                    "time_dimension": times.dims[0], "time_first": str(times.values[0]),
                    "time_last": str(times.values[-1]),
                    "calendar": times.encoding.get("calendar", "standard"),
                    "selection": {key: [value.start, value.stop] for key, value in indexers.items()},
                    "selection_semantics": "native positional slices [start, stop), no aggregation",
                    "selected_dimensions": dict(selected.sizes), "selected_payload_bytes_upper_bound": estimated,
                    "native_chunk_bytes_upper_bound": native_peak, "graph_chunks": chunks,
                    "estimated_dask_partitions": partitions}
        return manifest


def graph_case(path, manifest, args):
    if manifest["estimated_dask_partitions"] > 20000:
        raise NotApplicable("Original graph exceeds 20,000 partition budget")
    with cp.open_dataset(path, chunks=manifest["graph_chunks"], time=args.time) as dataset:
        cp.validate_metadata(dataset)
        cp.validate_time(dataset, time=args.time)
        if not isinstance(dataset[args.variable].data, da.Array):
            raise AssertionError("Original field is not Dask-backed")
        return {"field_values_read": False, "field_chunks": dataset[args.variable].chunks,
                "scope": "original source metadata, CF decoding and lazy graph only"}


def sample_source(path, manifest, args):
    limit = args.allow_native_chunk_mib * 1024 ** 2
    if manifest["native_chunk_bytes_upper_bound"] > limit:
        raise NotApplicable("Native decoded storage chunk exceeds explicit budget: "
                            f"{manifest['native_chunk_bytes_upper_bound']} > {int(limit)} bytes")
    with xr.open_dataset(path, engine="netcdf4", chunks=None, cache=False,
                         decode_coords="all") as dataset:
        _, selected, _ = selection(dataset, args)
        # Only this boundary reads original fields. Contiguous hyperslabs avoid
        # repeated native compressed-chunk reads from many small Dask tasks.
        sample = selected.load()
        for variable in sample.variables.values():
            variable.encoding.pop("source", None)
            variable.encoding.pop("original_shape", None)
            variable.encoding.pop("chunksizes", None)
            variable.encoding.pop("preferred_chunks", None)
        return sample


def equal_attrs(left, right):
    if set(left) != set(right):
        return False
    return all(np.array_equal(left[key], right[key]) for key in left)


def original_multifile(paths, manifests, args):
    if len(paths) != 2:
        raise NotApplicable("Two explicit source files are required")
    if any(item["estimated_dask_partitions"] > 20000 for item in manifests):
        raise NotApplicable("Original graph exceeds partition budget")
    left, right = manifests
    verified = []
    if not equal_attrs(left["global_attrs"], right["global_attrs"]):
        verified.append("Multi-file global attributes differ")
    for name in left["variables"].keys() & right["variables"].keys():
        if not equal_attrs(left["variables"][name]["attrs"], right["variables"][name]["attrs"]):
            verified.append("Multi-file variable metadata differ")
        if (name in left["data_variables"] and name in right["data_variables"]
                and left["time_dimension"] not in left["variables"][name]["dimensions"]):
            verified.append(f"Static data field {name!r} occurs in multiple files")
    try:
        with cp.open_mfdataset(list(reversed(paths)), chunks=left["graph_chunks"], time=args.time) as dataset:
            cp.validate_time(dataset, time=args.time)
            assert isinstance(dataset[args.variable].data, da.Array)
            return {"field_values_read": False, "time_count": dataset[args.time].size}
    except ValueError as error:
        if any(message in str(error) for message in verified):
            raise ExpectedRejection(f"Independently verified metadata conflict: {error}") from error
        raise


def assert_field(actual, expected):
    assert actual.dims == expected.dims
    for dimension in actual.dims:
        np.testing.assert_array_equal(actual[dimension].values, expected[dimension].values)
    np.testing.assert_allclose(actual.compute().values, expected.values, rtol=1e-6, atol=0, equal_nan=True)


def independent_normalized(sample, args):
    field = sample[args.variable]
    renames = {args.latitude: "latitude", args.longitude: "longitude"}
    renames = {old: new for old, new in renames.items() if old != new}
    field = field.rename(renames)
    field = field.assign_coords(longitude=np.mod(field.longitude.values, 360.0))
    return field.sortby("latitude").sortby("longitude")


def transform_cases(report, sample, args):
    if sample is None:
        report.case("derived_core_tests", "derived_real_subset", lambda: (_ for _ in ()).throw(
            NotApplicable("Bounded original-source sample was unavailable")))
        return None
    lazy = sample.chunk({dimension: min(size, 16) for dimension, size in sample.sizes.items()})
    expected = independent_normalized(sample, args)
    state = {}

    def normalization():
        result = cp.normalize_coordinates(lazy, latitude=args.latitude, longitude=args.longitude)
        assert isinstance(result[args.variable].data, da.Array)
        assert_field(result[args.variable], expected)
        state["normalized"] = result
        return {"comparison": "independent modulo-360 and latitude/index sorting", "dask_backed": True}
    report.case("normalize_coordinates", "derived_real_subset", normalization)

    def crop():
        if "normalized" not in state:
            raise NotApplicable("Coordinate normalization prerequisite failed")
        base = state["normalized"]
        longitude, latitude = base.longitude.values, base.latitude.values
        edge = min(4, max(1, len(longitude) // 2))
        lo, hi = float(longitude[-edge]), float(longitude[edge - 1])
        lat_lo, lat_hi = float(latitude[len(latitude) // 4]), float(latitude[3 * len(latitude) // 4])
        result = cp.subset_region(base, latitude_bounds=(lat_lo, lat_hi), longitude_bounds=(lo, hi))
        mask_lon = np.mod(longitude - lo, 360) <= np.mod(hi - lo, 360)
        mask_lat = (latitude >= lat_lo) & (latitude <= lat_hi)
        direct = expected.isel(latitude=np.flatnonzero(mask_lat), longitude=np.flatnonzero(mask_lon))
        assert_field(result[args.variable], direct)
        state["cropped"] = result
        return {"longitude_bounds": [lo, hi], "latitude_bounds": [lat_lo, lat_hi],
                "comparison": "independent directed eastward modulo-360 center selection"}
    report.case("subset_region_wrapped_longitude", "derived_real_subset", crop)

    def masking():
        result = cp.mask_invalid(lazy, variables=[args.variable])
        direct = sample[args.variable].where(np.isfinite(sample[args.variable]))
        assert_field(result[args.variable], direct)
        return {"negative_and_zero_values": "retained", "fill_or_interpolation": False}
    report.case("mask_invalid", "derived_real_subset", masking)

    def units():
        unit = sample[args.variable].attrs.get("units")
        if not unit:
            raise NotApplicable("Source field has no declared physical units")
        converted = cp.convert_dataarray_units(lazy[args.variable], unit)
        assert_field(converted, sample[args.variable])
        assert converted.attrs["units"] == unit
        return {"source_units": unit, "target_units": unit,
                "scope": "identity conversion for actual field units; no invented wind conversion"}
    report.case("convert_dataarray_units", "derived_real_subset", units)

    def laziness():
        calls = []

        def observed(block):
            calls.append(True)
            return block
        observed_dataset = lazy.copy()
        source = lazy[args.variable]
        observed_dataset[args.variable] = source.copy(data=source.data.map_blocks(
            observed, dtype=source.dtype, meta=np.empty((0,) * source.ndim, dtype=source.dtype)))
        result = cp.normalize_coordinates(observed_dataset, latitude=args.latitude, longitude=args.longitude)
        result = cp.subset_region(result)
        result = cp.mask_invalid(result, variables=[args.variable])
        unit = source.attrs.get("units")
        if unit:
            result[args.variable] = cp.convert_dataarray_units(result[args.variable], unit)
        assert not calls, "Field blocks ran during transform construction"
        result[args.variable].compute()
        assert calls, "Explicit compute did not execute observed field blocks"
        return {"field_blocks_during_construction": 0, "field_blocks_at_explicit_compute": len(calls)}
    report.case("field_laziness", "derived_real_subset", laziness)

    def writing():
        if "cropped" not in state:
            raise NotApplicable("Regional subset prerequisite failed")
        output = report.directory / "prepared_sample.nc"
        cp.write_dataset(state["cropped"], output)
        with cp.open_dataset(output, chunks={}, time=args.time) as reopened:
            cp.validate_metadata(reopened)
            cp.validate_time(reopened, time=args.time)
            assert_field(reopened[args.variable], state["cropped"][args.variable].compute())
        state["output"] = output
        return {"output": output.name, "size_bytes": output.stat().st_size}
    report.case("write_and_reopen", "derived_real_subset", writing)

    def pipeline():
        if "output" not in state:
            raise NotApplicable("Successful write prerequisite failed")
        unit = sample[args.variable].attrs.get("units")
        with cp.prepare_dataset(
            state["output"], chunks={}, time=args.time,
            coordinates={"latitude": "latitude", "longitude": "longitude"},
            units={args.variable: unit} if unit else None,
        ) as prepared:
            assert isinstance(prepared[args.variable].data, da.Array)
            direct = state["cropped"][args.variable].compute()
            direct = direct.where(np.isfinite(direct))
            assert_field(prepared[args.variable], direct)
        return {"dask_backed": True, "comparison": "composed pipeline against individual transformations"}
    report.case("prepare_dataset_pipeline", "derived_real_subset", pipeline)

    def overwrite():
        if "output" not in state:
            raise NotApplicable("Successful write prerequisite failed")
        before = state["output"].read_bytes()
        try:
            cp.write_dataset(state["cropped"], state["output"])
        except FileExistsError as error:
            assert state["output"].read_bytes() == before
            raise ExpectedRejection("Existing output rejected and small derived bytes unchanged") from error
        raise AssertionError("Writer unexpectedly overwrote an existing output")
    report.case("write_existing_output_rejected", "derived_real_subset", overwrite)
    return lazy


def derived_multifile(report, samples, args):
    if len(samples) != 2 or any(sample is None for sample in samples):
        raise NotApplicable("Two bounded original-source samples are required")
    # Explicit harmonization is confined to generated fixtures, with actual
    # source values/time retained. This does not claim original files combine.
    reference_attrs = dict(samples[0].attrs)
    reference_vars = {name: dict(value.attrs) for name, value in samples[0].variables.items()}
    paths = []
    for index, sample in enumerate(samples):
        derived = sample.copy(deep=True)
        derived.attrs = reference_attrs.copy()
        for name in derived.variables:
            if name in reference_vars:
                derived[name].attrs = reference_vars[name].copy()
        path = report.directory / f"derived_source_{index + 1}.nc"
        cp.write_dataset(derived.chunk(), path)
        paths.append(path)
    with cp.open_mfdataset(list(reversed(paths)), chunks={}, time=args.time) as combined:
        expected = xr.concat([sample[args.variable] for sample in samples], dim=args.time).sortby(args.time)
        assert_field(combined[args.variable], expected)
    return {"files": [path.name for path in paths], "original_attributes_harmonized": True,
            "values_and_times": "unaltered real-source subsets", "input_order": "reversed",
            "scope": "derived compatible-shard concatenation, separate from original-file contract"}


def monthly_cases(report, paths, manifests, samples, args):
    if not args.monthly_product:
        report.case("monthly_archive", "original_source", lambda: (_ for _ in ()).throw(
            NotApplicable("Monthly product not explicitly confirmed; hourly data are never reinterpreted")))
        return
    product = ""
    location = "user assertion only"
    if manifests:
        product = manifests[0]["variables"][args.variable]["attrs"].get("rda_dataset_group", "")
        if product:
            location = f"{args.variable}.attrs.rda_dataset_group"
        else:
            product = manifests[0]["global_attrs"].get("rda_dataset_group", "")
            if product:
                location = "global attrs.rda_dataset_group"
    report.metadata["monthly_product_confirmation"] = {
        "explicit_flag": True, "source_product_attribute": product, "attribute_location": location}
    coordinates = cp.Coordinates(time=args.time, latitude=args.latitude,
                                 longitude=args.longitude, pressure=args.pressure)
    spec = cp.FieldSpec(files=[str(paths[0])], variable=args.variable, coordinates=coordinates)
    first = samples[0] if samples else None

    def date():
        if first is None:
            raise NotApplicable("Bounded sample unavailable")
        times = first[args.time]
        return int(times.dt.year.values[0]), int(times.dt.month.values[0])

    def original():
        year, month = date()
        archive = cp.MonthlyArchive({"field": spec}, hash_inputs=False)
        metadata = archive.inspect("field", year, month)
        chunks = {"latitude": 64, "longitude": 128}
        if args.pressure:
            chunks["pressure"] = 2
        with archive.read_lazy("field", year, month, chunks=chunks) as field:
            assert isinstance(field.data, da.Array)
        return {"inspection": metadata, "field_values_read": False,
                "scope": "original monthly inspection and lazy construction; no second large-chunk read"}
    report.case("monthly_archive_original_lazy", "original_source", original)

    def derived():
        year, month = date()
        path = report.directory / "derived_monthly.nc"
        cp.write_dataset(first.chunk(), path)
        archive = cp.MonthlyArchive({"field": cp.FieldSpec(
            files=[str(path)], variable=args.variable, coordinates=coordinates)}, hash_inputs=False)
        with archive.read_lazy("field", year, month, chunks="auto") as field:
            actual = field.compute()
        compatible = archive.read("field", year, month)
        np.testing.assert_allclose(compatible.values, actual.values, equal_nan=True)
        direct = independent_normalized(first, args).isel({args.time: 0}, drop=True)
        if args.pressure:
            direct = direct.sortby(args.pressure)
        dimensions = ([args.pressure] if args.pressure else []) + ["latitude", "longitude"]
        direct = direct.squeeze(drop=True).transpose(*dimensions)
        np.testing.assert_allclose(actual.values, direct.values, rtol=1e-6, equal_nan=True)
        return {"lazy_and_legacy_eager_equal": True, "legacy_eager_scope": "tiny derived monthly subset only",
                "year": year, "month": month, "hash_inputs": False}
    report.case("monthly_archive_derived_roundtrip", "derived_real_subset", derived)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", action="append", required=True, help="Explicit original NetCDF; at most twice")
    result.add_argument("--output-dir", required=True, help="New report directory; must not exist")
    result.add_argument("--variable", default="V")
    result.add_argument("--latitude", default="latitude")
    result.add_argument("--longitude", default="longitude")
    result.add_argument("--time", default="time")
    result.add_argument("--pressure", default=None)
    result.add_argument("--select", action="append", default=[], metavar="DIM=INDEX")
    result.add_argument("--monthly-product", action="store_true",
                        help="Explicit assertion from product metadata, never inferred from sample spacing")
    result.add_argument("--allow-native-chunk-mib", type=float, default=192,
                        help="Conservative decoded bytes per original compressed chunk; default 192 MiB")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    paths = [Path(path).expanduser().resolve(strict=True) for path in args.input]
    if not 1 <= len(paths) <= 2 or len(set(paths)) != len(paths):
        raise ValueError("Provide one or two distinct explicit input files")
    if not 0 < args.allow_native_chunk_mib <= 512:
        raise ValueError("Native chunk budget must be in (0, 512] MiB")
    parse_select(args.select)
    report = Report(args.output_dir)
    before = [stamp(path) for path in paths]
    versions = {}
    for package in ("climate-preprocessing", "xarray", "dask", "numpy", "netCDF4", "cftime"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "source checkout" if package == "climate-preprocessing" else "unavailable"
    report.metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "versions": versions,
                       "python": sys.version, "executable": sys.executable,
                       "resource_limits": {"dask_scheduler": "threads", "dask_workers": 2, "blas_threads": 1,
                                           "maximum_source_files": 2, "times_per_source": 1,
                                           "pressure_levels": 2, "latitude_points": 16, "longitude_points": 32,
                                           "selected_payload_limit_bytes": 16 * 1024 ** 2,
                                           "native_chunk_limit_mib": args.allow_native_chunk_mib},
                       "source_identity_policy": "basename, size and mtime; no full-file hash",
                       "original_field_read_policy": "one contiguous backend hyperslab per source, then tiny Dask fixtures",
                       "scope": "basic preprocessing only; no statistical calculations", "sources": []}
    manifests, samples = [], []
    faulthandler.enable()
    faulthandler.dump_traceback_later(120, repeat=True)
    try:
        with dask.config.set(scheduler="threads", num_workers=2):
            for index, path in enumerate(paths, 1):
                manifest = report.case(f"source_{index}_metadata", "original_source",
                                       lambda path=path: inspect_source(path, args))
                if manifest is None:
                    samples.append(None)
                    continue
                manifests.append(manifest)
                report.metadata["sources"].append(manifest)
                report.case(f"source_{index}_open_dataset", "original_source",
                            lambda path=path, manifest=manifest: graph_case(path, manifest, args))
                sample = report.case(f"source_{index}_bounded_read", "original_source",
                                     lambda path=path, manifest=manifest: sample_source(path, manifest, args))
                samples.append(sample)
            if len(manifests) == len(paths):
                report.case("original_multifile_contract", "original_source",
                            lambda: original_multifile(paths, manifests, args))
            transform_cases(report, samples[0] if samples else None, args)
            report.case("derived_multifile_time_sort", "derived_real_subset",
                        lambda: derived_multifile(report, samples, args))
            monthly_cases(report, paths, manifests, samples, args)

            def unchanged():
                after = [stamp(path) for path in paths]
                assert before == after, "Original input size or mtime changed during validation"
                return {"original_files_unchanged": True, "identity": after}
            report.case("original_sources_unchanged", "original_source", unchanged)
    except Exception as error:
        def unexpected(error=error):
            raise error
        report.case("unexpected_runner_failure", "runner", unexpected)
    finally:
        faulthandler.cancel_dump_traceback_later()
    return report.finish()


if __name__ == "__main__":
    raise SystemExit(main())
