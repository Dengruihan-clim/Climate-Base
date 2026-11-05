"""Synthetic fixed/hybrid pressure interpolation and explicit CLI configuration.

References: CF1.12 hybrid formulas; GeoCAT/MetPy public interpolation APIs.
Climate-Base improvements: CF evidence, lazy application and explicit surface
masking. This creates analytical fixtures only, not observations or a forecast.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from climate_preprocessing import hybrid_pressure, interpolate_pressure, write_dataset


def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = xr.Dataset(coords={
        "lev": ("lev", [0., 1., 2., 3.], {
            "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate", "units": "1", "axis": "Z",
            "formula_terms": "a: a b: b p0: p0 ps: ps",
        }),
        "latitude": ("latitude", [0., 10.], {"units": "degrees_north", "standard_name": "latitude"}),
        "longitude": ("longitude", [100., 110.], {"units": "degrees_east", "standard_name": "longitude"}),
        "time": np.array(["2000-01-01", "2000-02-01"], dtype="datetime64[ns]"),
    })
    source["a"] = ("lev", [0.1, 0.15, 0.2, 0.1], {"units": "1"})
    source["b"] = ("lev", [0.1, 0.3, 0.5, 0.8], {"units": "1"})
    source["p0"] = xr.DataArray(100000., attrs={"units": "Pa"})
    source["ps"] = (("time", "latitude", "longitude"), np.full((2, 2, 2), 95000.),
                    {"units": "Pa", "standard_name": "surface_air_pressure"})
    source = source.chunk({"time": 1, "lev": 2, "latitude": 1, "longitude": 2})
    pressure = hybrid_pressure(source)
    dimensions = ("time", "lev", "latitude", "longitude")
    # Keep the source Dataset's original CF hybrid coordinate metadata. The
    # standalone pressure DataArray has self-contained, generic level labels.
    values = (220 + pressure * 0.001).transpose(*dimensions).data
    source["temperature"] = (dimensions, values, {"units": "K", "long_name": "Synthetic linear-pressure temperature"})
    write_dataset(source, directory / "hybrid.nc")
    result = interpolate_pressure(source, [300., 500., 800.], variables="temperature", method="linear")
    path = write_dataset(result, directory / "pressure.nc")
    config = {"schema_version": 1, "input": "hybrid.nc", "output": "cli_pressure.nc", "chunks": {"time": 1, "lev": 2},
              "vertical": {"target_levels": [300, 500, 800], "target_units": "hPa", "method": "linear",
                           "variables": ["temperature"], "coordinate": "lev"}}
    (directory / "prepare.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(run_example(parser.parse_args().directory))
