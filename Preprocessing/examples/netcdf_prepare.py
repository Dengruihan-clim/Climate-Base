"""Run a wholly synthetic xarray/Dask preprocessing example, without network.

Reference: xarray's public Dataset/Dask APIs. Climate-Base adds a strict,
auditable read-normalize-subset-convert-write workflow, without aggregation.
"""

import argparse
import json
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

from climate_preprocessing.cli import main


def create_demo(directory):
    """Create synthetic input and a schema-v1 config in a new directory."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    field = da.arange(24, chunks=12, dtype=float).reshape((2, 3, 4))
    dataset = xr.Dataset(
        {"temperature": (("time", "lat", "lon"), field, {"units": "degree_Celsius"})},
        coords={
            "time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
            "lat": ("lat", [45., 0., -45.], {"units": "degrees_north", "standard_name": "latitude"}),
            "lon": ("lon", [-180., -90., 0., 90.], {"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={"description": "Synthetic example, not observations or reanalysis"},
    )
    dataset.to_netcdf(directory / "source.nc", engine="netcdf4")
    config = {
        "schema_version": 1,
        "input": "source.nc",
        "output": "prepared.nc",
        "coordinates": {"latitude": "lat", "longitude": "lon"},
        "chunks": {"time": 1, "lat": 2, "lon": 2},
        "longitude_convention": "0_360",
        "latitude_order": "ascending",
        "region": {"latitude_bounds": [-45, 45], "longitude_bounds": [270, 90]},
        "units": {"temperature": "K"},
    }
    path = directory / "prepare.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="New directory, never overwritten")
    config_path = create_demo(parser.parse_args().directory)
    main(["inspect", str(config_path)])
    main(["prepare", str(config_path)])
