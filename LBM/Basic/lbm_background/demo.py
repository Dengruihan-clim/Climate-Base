"""Generate synthetic CF input and synthetic Gtool references for software QA.

These invented fields and headers must not be used as a scientific model state.
No real model assets or reanalysis observations are distributed here.
"""
from datetime import datetime
import json
from pathlib import Path

from netCDF4 import Dataset, date2num
import numpy as np

from . import binary


def _record(item, units, data, axes):
    data = np.asarray(data, dtype=np.float32)
    nz, ny, nx = data.shape
    header = [" " * 16 for _ in range(64)]
    for index, value in {0: "9010", 1: "SYNTHETIC", 2: item, 13: "SYNTHETIC QA", 15: units,
                         24: "0", 26: "00010101 000000", 28: axes[0], 29: "1", 30: str(nx),
                         31: axes[1], 32: "1", 33: str(ny), 34: axes[2], 35: "1", 36: str(nz),
                         37: "UR4", 38: "-999", 63: str(data.size)}.items():
        if len(value) > 16:
            raise ValueError("Synthetic header field is too long")
        header[index] = value.ljust(16)
    return "".join(header).encode("ascii"), np.asarray(data, dtype=">f4").tobytes()


def write_synthetic_references(directory):
    """Write invented T42 headers, flat model terrain and a 20-level sigma axis."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    sigma = np.geomspace(.995, .0083, 20).astype(np.float32)
    topography = np.zeros((1, 64, 128), np.float32)
    with (directory / "grz.t42").open("wb") as stream:
        for payload in _record("GRZ", "m", topography, ("GLON128", "GGLA64", "SFC1")):
            binary._write_record(stream, payload, ">")
    with (directory / "GTAXLOC.CSIG20").open("wb") as stream:
        for payload in _record("CSIG20", "ND", sigma.reshape(1, 1, 20), ("CSIG20", "NUMBER", "SFC1")):
            binary._write_record(stream, payload, ">")
    with (directory / "synthetic_template.t42l20").open("wb") as stream:
        for key, item, units in zip(binary.FIELD_ORDER, binary.GTOOL_ITEMS, binary.FIELD_UNITS):
            size = 1 if key == "PS" else 20
            value = {"PS": 1000., "T": 250., "Q": .001}.get(key, 1.)
            data = np.full((size, 64, 128), value, np.float32)
            for payload in _record(item, units, data, ("GLON128", "GGLA64", "SFC1" if key == "PS" else "CSIG20")):
                binary._write_record(stream, payload, ">")
    return {"topography": str(directory / "grz.t42"), "sigma": str(directory / "GTAXLOC.CSIG20"),
            "template": str(directory / "synthetic_template.t42l20")}


def create_demo(directory):
    """Create a new directory with a complete, portable synthetic DJF case."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    references = write_synthetic_references(directory / "model_reference")
    data_dir = directory / "data"
    data_dir.mkdir()
    lat = np.linspace(-90., 90., 19)
    lon = np.arange(0., 360., 10.)
    pressure = np.array([1000., 850., 700., 500., 300., 200., 100., 10.])
    dates = [datetime(2000, 12, 15), datetime(2001, 1, 15), datetime(2001, 2, 15)]
    p = pressure[:, None, None]
    phi = np.deg2rad(lat)[None, :, None]
    lam = np.deg2rad(lon)[None, None, :]
    base = np.zeros((len(pressure), len(lat), len(lon)))
    units = {"U": "m s-1", "V": "m s-1", "T": "K", "Q": "kg kg-1", "H": "m",
             "SLP": "Pa", "PS_SOURCE": "Pa"}
    with Dataset(data_dir / "monthly.nc", "w", format="NETCDF4") as ds:
        for name, size in (("time", 3), ("lev", len(pressure)), ("lat", len(lat)), ("lon", len(lon))):
            ds.createDimension(name, size)
        for name, values, unit in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east"),
                                    ("lev", pressure, "hPa")):
            variable = ds.createVariable(name, "f8", (name,))
            variable[:] = values
            variable.units = unit
        time = ds.createVariable("time", "f8", ("time",))
        time.units = "days since 2000-01-01 00:00:00"
        time.calendar = "standard"
        time[:] = date2num(dates, time.units, time.calendar)
        for key, unit in units.items():
            dims = ("time", "lat", "lon") if key in ("SLP", "PS_SOURCE") else ("time", "lev", "lat", "lon")
            variable = ds.createVariable(key, "f4", dims)
            variable.units = unit
            for index in range(3):
                values = {"U": base + 20. * np.cos(phi) + 2. * np.sin(lam) + index,
                          "V": base + 3. * np.sin(phi) * np.cos(lam),
                          "T": base + 220. + 65. * np.sqrt(p / 1000.),
                          "Q": base + .01 * (p / 1000.) ** 2,
                          "H": base - 7000. * np.log(p / 1000.) + 30.,
                          "SLP": np.full((len(lat), len(lon)), 101000.),
                          "PS_SOURCE": np.full((len(lat), len(lon)), 100800.)}[key]
                variable[index] = values
        ds.title = "SYNTHETIC SOFTWARE TEST; NOT REAL REANALYSIS"
    config = {"name": "synthetic_djf_2001", "source_product": "SYNTHETIC SOFTWARE TEST ONLY",
              "data_root": "data", "output": "output",
              "period": {"start_year": 2001, "end_year": 2001, "months": [12, 1, 2], "minimum_valid_years": 1},
              "coordinates": {"time": "time", "latitude": "lat", "longitude": "lon", "pressure": "lev"},
              "fields": {key: {"files": "{data_root}/monthly.nc", "variable": key} for key in units},
              "model": {key: Path(value).relative_to(directory).as_posix() for key, value in references.items()},
              "integrity": {"hash_inputs": True, "require_receipts": False}, "diagnostics": {"plot": False}}
    config_path = directory / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "SYNTHETIC_ONLY.txt").write_text("Invented software-test inputs. Not an atmospheric analysis or a scientific LBM background.\n", encoding="utf-8")
    return config_path
