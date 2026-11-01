"""Synthetic pressure/time/mask workflow; no downloads or statistical reduction.

References: public xarray/Dask APIs and the CF pressure/time conventions.
Climate-Base improvements: compose explicit, audited policies and write only a
new output. This generated field is an analytical example, not observations.
"""

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from climate_preprocessing import apply_mask, prepare_dataset, write_dataset


def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    times = ["2000-01-01", "2000-02-01", "2000-03-01"]
    source = xr.Dataset(
        {"temperature": (("time", "level", "lat", "lon"),
                         np.arange(36, dtype=float).reshape(3, 2, 2, 3) + 250,
                         {"units": "K", "ancillary_variables": "quality"}),
         "quality": (("lat", "lon"), np.array([[0, 1, 0], [0, 0, 1]], dtype="int8"),
                     {"flag_values": np.array([0, 1], dtype="int8"), "flag_meanings": "good excluded"})},
        coords={"time": np.array(times, dtype="datetime64[ns]"),
                "level": ("level", [100000., 50000.], {"units": "Pa", "standard_name": "air_pressure"}),
                "lat": ("lat", [10., 0.], {"units": "degrees_north", "standard_name": "latitude"}),
                "lon": ("lon", [0., 90., 180.], {"units": "degrees_east", "standard_name": "longitude"})},
    )
    source_path = directory / "synthetic.nc"
    source.to_netcdf(source_path, engine="netcdf4")
    with prepare_dataset(
        source_path, chunks={"time": 1}, expected_time=times,
        pressure={"pressure": "level", "target_units": "hPa", "order": "ascending"},
        time_range={"start": "2000-02-01", "end": "2000-03-01"},
    ) as prepared:
        masked = apply_mask(prepared, prepared.quality == 0, variables=["temperature"],
                            reason="Synthetic quality flag: retain quality == 0")
        return write_dataset(masked, directory / "prepared.nc")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(run_example(parser.parse_args().directory))
