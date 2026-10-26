"""Generate public, analytic example winds and an index; no observations are used."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def generate(output_dir: Path, *, first_year: int = 1980, years: int = 12) -> None:
    if years < 4:
        raise ValueError("Use at least four years for the regression example.")
    output_dir.mkdir(parents=True, exist_ok=False)
    time = pd.date_range(f"{first_year}-01-01", periods=12 * years, freq="MS")
    level = np.array([100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 1000.0])
    lat = np.arange(90.0, -90.1, -10.0)
    lon = np.arange(0.0, 360.0, 10.0)
    year_index = np.arange(years, dtype=float)
    index = np.sin(1.7 * year_index) + 0.35 * np.cos(0.8 * year_index)
    monthly_index = np.repeat(index, 12)
    phi = np.deg2rad(lat)[None, None, :, None]
    pressure_profile = 4.0 * np.cos(np.pi * (level - level[0]) / (level[-1] - level[0]))
    shape = (time.size, level.size, lat.size, lon.size)
    # The axisymmetric u field is nondivergent. The v field is purely divergent.
    # Both are smooth at the poles and have only low spherical-harmonic degrees.
    u = np.broadcast_to(20.0 * np.cos(phi), shape).astype(np.float32).copy()
    v = np.broadcast_to(
        pressure_profile[None, :, None, None]
        * (np.sin(phi) - 0.5 + 0.05 * monthly_index[:, None, None, None])
        * np.cos(phi),
        shape,
    ).astype(np.float32).copy()
    coords = {"time": time, "level": level, "lat": lat, "lon": lon}
    ds = xr.Dataset(
        {"u": (tuple(coords), u), "v": (tuple(coords), v), "vd": (tuple(coords), v)},
        coords=coords,
        attrs={"description": "Analytic synthetic fields; not observational results."},
    )
    ds["level"].attrs["units"] = "hPa"
    ds["lat"].attrs["units"] = "degrees_north"
    ds["lon"].attrs["units"] = "degrees_east"
    for name in ("u", "v", "vd"):
        ds[name].attrs["units"] = "m s-1"
    ds.to_netcdf(output_dir / "winds.nc", engine="scipy")
    pd.DataFrame({"year": first_year + year_index.astype(int), "nino34": index}).to_csv(
        output_dir / "index.csv", index=False
    )
    print("Created synthetic winds.nc and index.csv.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--first-year", type=int, default=1980)
    parser.add_argument("--years", type=int, default=12)
    args = parser.parse_args()
    generate(args.output_dir, first_year=args.first_year, years=args.years)
