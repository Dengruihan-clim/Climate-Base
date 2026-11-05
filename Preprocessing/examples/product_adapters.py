"""Declared monthly pressure products with explicit, lazy stream selection.

These tiny fixtures illustrate naming contracts only. They are not original
ERA5, MERRA-2 or JRA-3Q observations or product authenticity certificates.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from climate_preprocessing import adapt_product, write_dataset


def run_example(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    products = (
        ("era5-monthly-pressure-levels", "level", "latitude", "longitude", "V"),
        ("merra2-m2imnpasm-5.12.4", "lev", "lat", "lon", "V"),
        ("jra3q-monthly-isobaric-analysis", "pressure_level", "lat", "lon", "vgrd-pres-an-ll125-mn"),
    )
    for index, (product, pressure, latitude, longitude, field) in enumerate(products):
        source = xr.Dataset({
            field: (("time", pressure, latitude, longitude), np.full((1, 2, 2, 2), 7.),
                    {"units": "m s-1", "standard_name": "northward_wind", "cell_methods": "time: mean"}),
        }, coords={
            "time": np.array(["2000-01-01"], dtype="datetime64[ns]"),
            pressure: (pressure, [500., 850.], {"units": "hPa"}),
            latitude: (latitude, [20., 10.], {"units": "degrees_north"}),
            longitude: (longitude, [100., 110.], {"units": "degrees_east"}),
        }, attrs={"title": "Synthetic product-schema illustration; not original provider data"}).chunk({"time": 1})
        selectors = None
        if index == 0:
            source = source.expand_dims(expver=[1])
            selectors = {"expver": 1}
            write_dataset(source, directory / "era5_like.nc")
        adapted = adapt_product(source, product=product, variables={"northward_wind": field}, selectors=selectors)
        write_dataset(adapted, directory / f"adapted_{index}.nc")
    config = {
        "schema_version": 1, "input": "era5_like.nc", "output": "prepared.nc", "chunks": {"time": 1},
        "adapter": {"product": products[0][0], "variables": {"northward_wind": "V"}, "selectors": {"expver": 1}},
    }
    (directory / "prepare.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8", newline="\n")
    return directory / "prepare.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(run_example(parser.parse_args().directory))
