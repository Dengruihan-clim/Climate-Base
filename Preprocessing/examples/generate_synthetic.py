"""Create a small, wholly synthetic monthly dataset and runnable config."""
from datetime import datetime
import argparse
import json
from pathlib import Path
import numpy as np
from netCDF4 import Dataset, date2num


def create_synthetic(directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    dates = [(1999, 12), (2000, 1), (2000, 2), (2000, 12), (2001, 1), (2001, 2)]
    with Dataset(directory / "synthetic_months.nc", "w") as ds:
        for name, count in (("time", 6), ("level", 3), ("latitude", 3), ("longitude", 4)):
            ds.createDimension(name, count)
        time = ds.createVariable("time", "f8", ("time",))
        time.units, time.calendar = "days since 1900-01-01", "standard"
        time[:] = date2num([datetime(y, m, 15) for y, m in dates], time.units, calendar=time.calendar)
        for name, values, units in (("level", [100000., 50000., 10000.], "Pa"),
                                    ("latitude", [90., 0., -90.], "degrees_north"),
                                    ("longitude", [-180., -90., 0., 90.], "degrees_east")):
            var = ds.createVariable(name, "f8", (name,))
            var.units = units
            var[:] = values
        var = ds.createVariable("temperature", "f8", ("time", "level", "latitude", "longitude"))
        var.units = "degree_Celsius"
        var[:] = np.array([0., 10., 20., 1., 11., 21.])[:, None, None, None] + np.zeros((6, 3, 3, 4))
        ds.description = "Wholly synthetic example; not observations or reanalysis"
    config = {"data_root": ".", "fields": {"temperature": {"files": "synthetic_months.nc", "variable": "temperature",
              "coordinates": {"time": "time", "latitude": "latitude", "longitude": "longitude", "pressure": "level"},
              "expected_units": "degree_Celsius", "target_units": "K"}},
              "archive": {"pressure_units": "hPa", "hash_inputs": True},
              "analysis": {"field": "temperature", "start_year": 2000, "end_year": 2001, "months": [12, 1, 2], "weighting": "days"},
              "output": "seasonal.nc"}
    path = directory / "config.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="A new directory; existing directories are never overwritten")
    print(create_synthetic(parser.parse_args().directory))
