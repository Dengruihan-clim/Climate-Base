"""Synthetic NetCDF -> completed local Zarr v2 -> lazy read.

References: public xarray/Zarr v2 I/O, independently composed by Climate-Base.
Install climate-preprocessing[zarr] first. No network request or real data.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from climate_preprocessing import open_zarr, prepare_dataset, write_zarr


def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = xr.Dataset(
        {"temperature": (("time", "latitude", "longitude"),
                         np.arange(12, dtype="float32").reshape(2, 2, 3), {"units": "degree_Celsius"})},
        coords={"time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
                "latitude": ("latitude", [10., 0.], {"standard_name": "latitude", "units": "degrees_north"}),
                "longitude": ("longitude", [0., 10., 20.], {"standard_name": "longitude", "units": "degrees_east"})},
    )
    input_path = directory / "synthetic.nc"
    source.to_netcdf(input_path, engine="netcdf4")
    with prepare_dataset(input_path, chunks={"time": 1}, units={"temperature": "K"}) as prepared:
        output = write_zarr(prepared, directory / "prepared.zarr")
    configuration = {
        "schema_version": 1, "input": "prepared.zarr", "input_format": "zarr",
        "expected_time": ["2000-01-01", "2000-02-01"], "time_range": {"start": "2000-02-01"},
    }
    (directory / "inspect.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    with open_zarr(output) as reopened:
        return {"store": str(output), "dimensions": dict(reopened.sizes),
                "field_is_lazy": reopened.temperature.chunks is not None,
                "inspect_config": str(directory / "inspect.json")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(run_example(parser.parse_args().directory), indent=2))
