"""Check the real Helmholtz backend against analytic global winds (no mocks)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hadley_ma2025 import compute_seasonal_divergent_wind  # noqa: E402


def validate() -> dict[str, float | str]:
    lat = np.arange(-90.0, 90.1, 5.0)
    lon = np.arange(0.0, 360.0, 5.0)
    coords = {
        "time": pd.to_datetime(["1981-10-01", "1982-10-01"]),
        "level": [200.0, 500.0],
        "lat": lat,
        "lon": lon,
    }
    phi = np.deg2rad(lat)[None, None, :, None]
    shape = (2, 2, lat.size, lon.size)
    amplitudes = np.array([[2.0, 3.0], [4.0, 1.5]])[:, :, None, None]
    u = xr.DataArray(np.broadcast_to(20.0 * np.cos(phi), shape).copy(), dims=tuple(coords), coords=coords)
    v = xr.DataArray(
        np.broadcast_to(amplitudes * (np.sin(phi) - 0.5) * np.cos(phi), shape).copy(),
        dims=tuple(coords), coords=coords,
    )
    for field in (u, v):
        field.attrs["units"] = "m s-1"
        field["level"].attrs["units"] = "hPa"
    actual = compute_seasonal_divergent_wind(u, v, truncation=6)
    vd_error = actual["vd"] - v
    relative_rms = float(np.sqrt((vd_error ** 2).mean()) / np.sqrt((v ** 2).mean()))
    ud_maximum = float(np.abs(actual["ud"]).max())
    tolerance = 2.0e-5
    if relative_rms > tolerance or ud_maximum > tolerance:
        raise AssertionError("The real spectral backend failed the analytic irrotational-wind check.")
    return {
        "status": "passed",
        "validation": "analytic_global_rotational_u_and_divergent_v",
        "vd_relative_rms_error": relative_rms,
        "ud_max_absolute_error_m_s": ud_maximum,
        "tolerance": tolerance,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
