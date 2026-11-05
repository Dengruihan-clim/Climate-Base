"""Analytical scalar remapping with explicit geometry and reusable weights.

References: xESMF public API. Climate-Base improvements: named scalar selection,
CF metadata cleanup, lazy fields and validated cache identity. No data download.
Run in an isolated environment with xESMF AND working ESMPy/ESMF installed.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from climate_preprocessing import rectilinear_grid, regrid_dataset, write_dataset


def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    source = rectilinear_grid(np.arange(-20, 21, 10), np.arange(100, 141, 10))
    source["temperature"] = (280 + 0.1 * source.latitude + 0.2 * source.longitude).transpose("latitude", "longitude")
    source.temperature.attrs.update(units="K", long_name="Synthetic affine temperature")
    source = source.chunk({"latitude": 3, "longitude": 3})
    target = rectilinear_grid([-15, -5, 5, 15], [105, 115, 125, 135])
    result = regrid_dataset(source, target, variables=["temperature"], method="bilinear",
                            weights_dir=directory / "weights")
    output = write_dataset(result, directory / "regridded.nc")
    # Source file and config make the same operation reviewable through CLI.
    source = source.expand_dims(time=np.array(["2000-01-01"], dtype="datetime64[ns]"))
    write_dataset(source, directory / "input.nc")
    config = {"schema_version": 1, "input": "input.nc", "output": "cli_regridded.nc", "regrid": {
        "variables": ["temperature"], "method": "bilinear", "weights_dir": "weights",
        "target_grid": {"latitude": [-15, -5, 5, 15], "longitude": [105, 115, 125, 135]},
    }}
    (directory / "prepare.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(run_example(parser.parse_args().directory))
