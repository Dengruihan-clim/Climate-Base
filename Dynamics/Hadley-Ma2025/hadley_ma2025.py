"""Regional Hadley circulation diagnosed from divergent meridional wind.

Method reference: Ma, W., Wang, Y., Miao, Y., and Li, X. (2025),
Geophysical Research Letters, 52, e2025GL115344.
https://doi.org/10.1029/2025GL115344 (Section 2.2, Equation 3).

Scientific dependencies: NumPy, pandas, xarray, SciPy. Total-wind input also
requires windspharm or skyborn; plots require matplotlib. Optional GRIB input
requires cfgrib. No spectral or plotting package is imported by the core API.

The pressure integral starts at the uppermost supplied level, where psi is
defined as zero. No lower-boundary residual correction is applied. Horizontal
intensity is max_level(abs(psi)) BEFORE anomaly calculation and regression.
Regional sections retain signed psi and reduce longitude BEFORE regression.
These choices describe this implementation, rather than certifying agreement
with every detail of a particular publication.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr

EARTH_RADIUS = 6.371e6
GRAVITY = 9.80665
DEFAULT_PRESSURE_LEVELS = (100., 150., 200., 250., 300., 400., 500., 600., 700., 850., 1000.)
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


def _validate_wind_units(field: xr.DataArray) -> None:
    unit = str(field.attrs.get("units", "")).strip().lower().replace(" ", "")
    unit = unit.replace("−", "-").replace("⁻¹", "-1").replace("{", "").replace("}", "")
    if unit not in {"m/s", "ms-1", "ms^-1", "ms**-1", "meter/second", "metre/second",
                    "meters/second", "metres/second", "metersecond-1", "metresecond-1"}:
        raise ValueError("Wind units must explicitly be metres per second (for example 'm s-1' or 'm/s').")


def _pressure_factor(units: str) -> float:
    unit = str(units).strip().lower().replace(" ", "")
    if unit in {"hpa", "hectopascal", "hectopascals", "mb", "mbar", "millibar", "millibars"}:
        return 1.
    if unit in {"pa", "pascal", "pascals"}:
        return .01
    if unit in {"kpa", "kilopascal", "kilopascals"}:
        return 10.
    raise ValueError("Pressure units must explicitly be Pa, hPa, mbar, mb, or kPa.")


def normalize_wind_dataset(ds: xr.Dataset, *, pressure_units: str | None = None) -> xr.Dataset:
    """Normalize coordinate names and pressure units; require explicit units."""
    aliases = {"latitude": "lat", "longitude": "lon", "plev": "level", "pressure": "level",
               "lev": "level", "isobaricInhPa": "level", "isobaricInPa": "level",
               "valid_time": "time"}
    rename = {old: new for old, new in aliases.items() if old in ds.dims and new not in ds.dims}
    out = ds.rename(rename)
    for name in ("level", "lat", "lon"):
        if name not in out.coords or out[name].ndim != 1:
            raise ValueError(f"A one-dimensional {name} coordinate is required.")
    units = pressure_units if pressure_units is not None else out["level"].attrs.get("units", "")
    levels = np.asarray(out["level"].values, dtype=float) * _pressure_factor(units)
    if levels.size < 2 or not np.isfinite(levels).all() or np.any(levels <= 0):
        raise ValueError("At least two finite, positive pressure levels are required.")
    if np.unique(levels).size != levels.size:
        raise ValueError("Duplicate pressure levels are not allowed.")
    out = out.assign_coords(level=levels)
    out["level"].attrs = {"units": "hPa", "positive": "down"}
    for name in ("lat", "lon"):
        vals = np.asarray(out[name].values, dtype=float)
        if not np.isfinite(vals).all() or np.unique(vals).size != vals.size:
            raise ValueError(f"{name} must contain finite, unique coordinates.")
    out = out.assign_coords(lon=np.mod(np.asarray(out["lon"].values, dtype=float), 360.))
    if np.unique(out["lon"].values).size != out.sizes["lon"]:
        raise ValueError("Longitude contains a duplicate cyclic endpoint.")
    out = out.sortby("level").sortby("lat").sortby("lon")
    out["lat"].attrs = {"units": "degrees_north"}
    out["lon"].attrs = {"units": "degrees_east"}
    if "time" not in out.dims:
        if "year" not in out.dims:
            raise ValueError("A time dimension, or a year dimension for seasonal input, is required.")
        years = np.asarray(out["year"].values, dtype=int)
        out = out.rename({"year": "time"}).assign_coords(time=pd.to_datetime([f"{year:04d}-01-01" for year in years]))
        out = out.assign_coords(year=("time", years))
    return out


def validate_global_grid(field: xr.DataArray, *, require_poles: bool = True) -> None:
    """Check the full regular global grid required for the spectral workflow."""
    for name in ("lat", "lon"):
        if name not in field.coords or field[name].ndim != 1:
            raise ValueError(f"A one-dimensional {name} coordinate is required.")
    lon = np.sort(np.mod(np.asarray(field["lon"].values, dtype=float), 360.))
    lat = np.sort(np.asarray(field["lat"].values, dtype=float))
    if lon.size < 4 or not np.isfinite(lon).all() or np.unique(lon).size != lon.size:
        raise ValueError("Global longitude must have at least four finite, unique cells.")
    expected = 360. / lon.size
    gaps = np.diff(np.r_[lon, lon[0] + 360.])
    if not np.allclose(gaps, expected, rtol=1.e-5, atol=1.e-5):
        raise ValueError("Longitude must cover the complete globe with a regular cyclic spacing.")
    if lat.size < 3 or not np.isfinite(lat).all() or np.any(np.diff(lat) <= 0):
        raise ValueError("Latitude must contain at least three finite, unique points.")
    if np.any(lat < -90.) or np.any(lat > 90.):
        raise ValueError("Latitude values must lie within -90 to 90 degrees.")
    if not np.allclose(np.diff(lat), np.diff(lat)[0], rtol=1.e-5, atol=1.e-5):
        raise ValueError("Latitude must be regularly spaced.")
    if require_poles and not np.allclose([lat[0], lat[-1]], [-90., 90.], atol=1.e-5):
        raise ValueError("The global regular latitude grid must include both poles.")
    if not np.isfinite(np.asarray(field.values)).all():
        raise ValueError("Global wind fields must be finite and contain no missing values.")


def _seasonal_time(year: int, season: str) -> pd.Timestamp:
    return pd.Timestamp(year=year, month={"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}[season], day=1)


def seasonalize_winds(ds: xr.Dataset, *, season: str = "SON", frequency: str = "monthly",
                      start_year: int | None = None, end_year: int | None = None) -> xr.Dataset:
    """Days-weighted complete seasons; December belongs to the next DJF year."""
    season = season.upper()
    if season not in SEASON_MONTHS:
        raise ValueError("Season must be DJF, MAM, JJA, or SON.")
    if "time" not in ds.dims:
        raise ValueError("A time dimension is required.")
    if frequency == "seasonal":
        years = _years(ds)
        if np.unique(years).size != years.size:
            raise ValueError("Seasonal input must contain exactly one sample per year.")
        out = ds.assign_coords(time=[_seasonal_time(int(year), season) for year in years], year=("time", years))
    elif frequency == "monthly":
        try:
            months = np.asarray(ds["time"].dt.month.values, dtype=int)
            years = np.asarray(ds["time"].dt.year.values, dtype=int)
            days = np.asarray(ds["time"].dt.days_in_month.values, dtype=float)
        except (AttributeError, TypeError) as exc:
            raise ValueError("Monthly input requires decoded calendar time coordinates.") from exc
        period_keys = years * 100 + months
        if np.unique(period_keys).size != period_keys.size:
            raise ValueError("Monthly input contains duplicate calendar months.")
        season_year = years + ((months == 12) & (season == "DJF")).astype(int)
        selected = np.isin(months, SEASON_MONTHS[season])
        pieces = []
        for year in np.unique(season_year[selected]):
            indices = np.where(selected & (season_year == year))[0]
            if indices.size != 3 or set(months[indices]) != set(SEASON_MONTHS[season]):
                continue
            subset = ds.isel(time=indices)
            weight = xr.DataArray(days[indices], dims="time", coords={"time": subset["time"]})
            # skipna=False prevents a partly missing season from looking complete.
            weighted = (subset * weight).sum("time", skipna=False) / float(weight.sum())
            for name in weighted.data_vars:
                weighted[name].attrs = dict(ds[name].attrs)
            weighted = weighted.expand_dims(time=[_seasonal_time(int(year), season)])
            weighted = weighted.assign_coords(year=("time", [int(year)]))
            pieces.append(weighted)
        if not pieces:
            raise ValueError("No complete three-month target seasons were found.")
        out = xr.concat(pieces, dim="time")
    else:
        raise ValueError("Frequency must be monthly or seasonal.")
    years = _years(out)
    mask = np.ones(years.shape, dtype=bool)
    if start_year is not None:
        mask &= years >= start_year
    if end_year is not None:
        mask &= years <= end_year
    out = out.isel(time=np.where(mask)[0]).sortby("time")
    if not out.sizes.get("time", 0):
        raise ValueError("No seasonal wind samples remain in the requested years.")
    return out


def _years(field: xr.DataArray | xr.Dataset) -> np.ndarray:
    if "year" in field.coords and field["year"].dims == ("time",):
        years = np.asarray(field["year"].values, dtype=int)
    else:
        try:
            years = np.asarray(field["time"].dt.year.values, dtype=int)
        except (AttributeError, TypeError) as exc:
            raise ValueError("Seasonal data require a year(time) coordinate or datetime time.") from exc
    return years


def _align_common_years(field: xr.DataArray, predictor: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    if predictor.dims != ("time",):
        raise ValueError("Predictor must be one-dimensional with dimension time.")
    fy, py = _years(field), _years(predictor)
    if np.unique(fy).size != fy.size or np.unique(py).size != py.size:
        raise ValueError("Wind and index must each have one seasonal sample per year.")
    finite_predictor = np.isfinite(np.asarray(predictor.values, dtype=float))
    common = np.intersect1d(fy, py[finite_predictor])
    if common.size < 3:
        raise ValueError("At least three common years with a finite index are required.")
    fi = np.array([np.where(fy == year)[0][0] for year in common])
    pi = np.array([np.where(py == year)[0][0] for year in common])
    aligned_field = field.isel(time=fi).assign_coords(year=("time", common))
    aligned_predictor = predictor.isel(time=pi).assign_coords(time=aligned_field["time"], year=("time", common))
    if not np.isfinite(np.asarray(aligned_predictor.values)).all() or float(aligned_predictor.std()) <= 0:
        raise ValueError("The index must have nonzero finite variance over the common years.")
    return aligned_field, aligned_predictor


def detrend_along_time(field: xr.DataArray) -> xr.DataArray:
    """Remove an intercept and linear slope using actual seasonal years."""
    years = _years(field).astype(float)
    x = xr.DataArray(years - years.mean(), dims="time", coords={"time": field["time"]})
    valid = np.isfinite(field)
    n = valid.sum("time")
    xm = x.where(valid).sum("time", skipna=True) / n.where(n > 0)
    ym = field.where(valid).sum("time", skipna=True) / n.where(n > 0)
    dx, dy = x - xm, field - ym
    denominator = (dx ** 2).where(valid).sum("time", skipna=True)
    slope = (dx * dy).where(valid).sum("time", skipna=True) / denominator.where(denominator > 0)
    out = (field - (ym + slope * dx)).where(valid & (n >= 2))
    out.attrs = {"units": field.attrs.get("units", ""), "detrend_method": "linear_common_period"}
    return out.transpose(*field.dims)


def standardize_series(series: xr.DataArray) -> xr.DataArray:
    values = np.asarray(series.values, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("The index requires at least three finite common seasonal samples.")
    std = float(values.std(ddof=1))
    tolerance = np.finfo(float).eps * max(1., float(np.max(np.abs(values)))) * 100.
    if not np.isfinite(std) or std <= tolerance:
        raise ValueError("The index has no usable variance after processing the common period.")
    out = (series - float(values.mean())) / std
    out.attrs = {"units": "1", "standardization": "common_period_sample_standard_deviation_ddof_1",
                 "standardized_mean_removed": float(values.mean()), "standardized_std_removed": std}
    return out


def compute_seasonal_divergent_wind(u_seasonal: xr.DataArray, v_seasonal: xr.DataArray, *,
                                   truncation: int | None = None) -> xr.Dataset:
    """Globally decompose total wind; import the optional spectral backend here."""
    u, v = xr.align(u_seasonal, v_seasonal, join="exact")
    _validate_wind_units(u)
    _validate_wind_units(v)
    validate_global_grid(u)
    validate_global_grid(v)
    u = _clean_core_array(u, units="m s-1")
    v = _clean_core_array(v, units="m s-1")
    try:
        from windspharm.xarray import VectorWind
    except ImportError:
        try:
            from skyborn.windspharm.xarray import VectorWind
        except ImportError as exc:
            raise ImportError("Total-wind input requires windspharm or skyborn with its spectral backend. "
                              "Prepared divergent input can use --input-kind divergent --v-var vd.") from exc
    ud_list, vd_list = [], []
    for level in np.asarray(u["level"].values, dtype=float):
        ul = u.sel(level=level).sortby("lat", ascending=False).sortby("lon").transpose("lat", "lon", "time")
        vl = v.sel(level=level).sortby("lat", ascending=False).sortby("lon").transpose("lat", "lon", "time")
        ud, vd = VectorWind(ul, vl).irrotationalcomponent(truncation=truncation)
        ud_list.append(ud.expand_dims(level=[level]))
        vd_list.append(vd.expand_dims(level=[level]))
    out = xr.Dataset({"ud": xr.concat(ud_list, "level"), "vd": xr.concat(vd_list, "level")})
    out = out.sortby("level").sortby("lat").sortby("lon").transpose("time", "level", "lat", "lon")
    out = out.assign_coords(year=("time", _years(u)))
    out["level"].attrs = {"units": u["level"].attrs.get("units", "hPa")}
    for name in ("ud", "vd"):
        out[name].attrs = {"units": "m s-1", "method": "global_Helmholtz_irrotational_component"}
    return out


def compute_local_mass_streamfunction(vd: xr.DataArray) -> xr.DataArray:
    """Integrate vd from the supplied top pressure; level defaults to hPa.

    The prepared core API follows the hPa coordinate convention. If level has
    explicit Pa/kPa units it is converted; CLI input always requires units.
    """
    required = {"time", "level", "lat", "lon"}
    if set(vd.dims) != required:
        raise ValueError("vd must have precisely time, level, lat, and lon dimensions.")
    _validate_wind_units(vd)
    validate_global_grid(vd, require_poles=False)
    vd = _clean_core_array(vd, units="m s-1")
    factor = _pressure_factor(vd["level"].attrs.get("units", "hPa"))
    field = vd.assign_coords(level=np.asarray(vd["level"].values, dtype=float) * factor).sortby("level")
    field = field.transpose("time", "level", "lat", "lon")
    levels = np.asarray(field["level"].values, dtype=float)
    if levels.size < 2 or not np.isfinite(levels).all() or np.any(levels <= 0) or np.any(np.diff(levels) <= 0):
        raise ValueError("At least two distinct, finite, positive pressure levels are required.")
    values = np.asarray(field.values, dtype=np.float64)
    intervals = .5 * (values[:, 1:] + values[:, :-1]) * (np.diff(levels) * 100.)[None, :, None, None]
    top = np.where(np.isfinite(values[:, :1]), 0., np.nan)
    cumulative = np.concatenate([top, np.cumsum(intervals, axis=1)], axis=1)
    delta_lon = 2. * np.pi / field.sizes["lon"]
    coefficient = EARTH_RADIUS * np.cos(np.deg2rad(field["lat"].values)) * delta_lon / GRAVITY
    psi = xr.DataArray(cumulative * coefficient[None, None, :, None], dims=field.dims,
                       coords=field.coords, name="psi")
    psi.attrs = {"units": "kg s-1", "method": "local_mass_streamfunction_from_divergent_meridional_wind",
                 "top_boundary_hpa": float(levels.min()), "bottom_level_hpa": float(levels.max()),
                 "top_boundary_value": 0., "residual_correction": "none", "delta_lon_rad": delta_lon}
    psi["level"].attrs = {"units": "hPa", "positive": "down"}
    return psi


def compute_hc_intensity(psi: xr.DataArray) -> xr.DataArray:
    out = np.abs(psi).max("level", skipna=True).rename("hc_intensity")
    out.attrs = {"units": "kg s-1", "definition": "max_level(abs(original_psi))"}
    return out


def regional_reduce_psi(psi: xr.DataArray, *, lon_bounds: tuple[float, float], reduction: str) -> xr.DataArray:
    lon = np.mod(np.asarray(psi["lon"].values, dtype=float), 360.)
    west, east = np.mod(lon_bounds, 360.)
    if np.isclose(west, east):
        mask = np.ones(lon.shape, dtype=bool)
    elif west <= east:
        mask = (lon >= west) & (lon <= east)
    else:
        mask = (lon >= west) | (lon <= east)
    if not mask.any():
        raise ValueError("The longitude sector contains no grid cells.")
    subset = psi.isel(lon=np.where(mask)[0])
    if reduction == "sum":
        out = subset.sum("lon", skipna=True, min_count=1)
    elif reduction == "mean":
        out = subset.mean("lon", skipna=True)
    else:
        raise ValueError("Regional reduction must be mean or sum.")
    out = out.transpose("time", "level", "lat").rename("psi_regional")
    out.attrs = {"units": "kg s-1", "regional_reduction": reduction,
                 "lon_min": float(lon_bounds[0]), "lon_max": float(lon_bounds[1]),
                 "lon_count": int(mask.sum()), "longitude_endpoints": "inclusive"}
    return out


def compute_climatology_anomaly(field: xr.DataArray, *, baseline_start: str = "1981",
                                baseline_end: str = "2010", detrend: bool = True,
                                anomaly_name: str = "anomaly") -> tuple[xr.DataArray, xr.DataArray]:
    # Year-only boundaries include their complete seasons, including DJF labels.
    def boundary_year(value: str) -> int:
        return int(str(value)[:4])
    years = _years(field)
    selected = (years >= boundary_year(baseline_start)) & (years <= boundary_year(baseline_end))
    if not selected.any():
        raise ValueError("The baseline contains no samples within the common analysis years.")
    clim = field.isel(time=np.where(selected)[0]).mean("time", skipna=True).rename(f"{field.name or 'field'}_climatology")
    anomaly = (field - clim).rename(anomaly_name)
    units = str(field.attrs.get("units", ""))
    clim.attrs = {"units": units, "baseline_start_year": boundary_year(baseline_start),
                  "baseline_end_year": boundary_year(baseline_end), "baseline_sample_count": int(selected.sum())}
    anomaly.attrs = dict(clim.attrs)
    if detrend:
        anomaly = detrend_along_time(anomaly).rename(anomaly_name)
        anomaly.attrs.update(clim.attrs)
    anomaly.attrs["detrend_method"] = "linear_common_period" if detrend else "none"
    return clim, anomaly


def regress_to_nino34(field_anomaly: xr.DataArray, predictor: xr.DataArray, *, alpha: float = .1) -> dict[str, Any]:
    """OLS slope and two-sided Pearson t-test, with explicit finite-pair counts."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between zero and one.")
    field, index = _align_common_years(field_anomaly, predictor)
    valid = np.isfinite(field) & np.isfinite(index)
    n = valid.sum("time").astype(np.int32)
    xm = index.where(valid).sum("time", skipna=True) / n.where(n > 0)
    ym = field.where(valid).sum("time", skipna=True) / n.where(n > 0)
    dx, dy = index - xm, field - ym
    sxx = (dx ** 2).where(valid).sum("time", skipna=True)
    syy = (dy ** 2).where(valid).sum("time", skipna=True)
    sxy = (dx * dy).where(valid).sum("time", skipna=True)
    slope = (sxy / sxx.where(sxx > 0)).where(n >= 3).rename("regression")
    r = (sxy / np.sqrt(sxx * syy).where((sxx > 0) & (syy > 0))).clip(-1., 1.)
    r = r.where(n >= 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = np.abs(r) * np.sqrt((n - 2) / (1. - r ** 2))
        p_values = 2. * stats.t.sf(np.asarray(tval.values), np.asarray(n.values) - 2)
    pvalue = xr.DataArray(p_values, dims=r.dims, coords=r.coords, name="pvalue").where(np.isfinite(r))
    significant = ((pvalue < alpha) & np.isfinite(pvalue)).rename("significance")
    slope.attrs = {"units": f"{field.attrs.get('units', 'kg s-1')} per index standard deviation",
                   "test": "ordinary_least_squares", "predictor": "standardized_seasonal_index"}
    pvalue.attrs = {"units": "1", "test": "two_sided_Pearson_t", "effective_sample_size_adjustment": "none"}
    significant.attrs = {"alpha": float(alpha), "missing_or_undefined_test": "false"}
    n = n.rename("sample_count")
    n.attrs = {"units": "1", "description": "number of finite paired annual samples"}
    years = _years(field)
    return {"regression": slope, "pvalue": pvalue, "significance": significant, "sample_count": n,
            "predictor_common": index, "field_common": field, "actual_years": (int(years.min()), int(years.max()))}


def _clean_core_array(field: xr.DataArray, *, units: str) -> xr.DataArray:
    coords = {}
    for name, coord in field.coords.items():
        if name not in {"time", "year", "level", "lat", "lon"}:
            continue
        attrs = {}
        if name == "level":
            factor = _pressure_factor(coord.attrs.get("units", "hPa"))
            attrs = {"units": {.01: "Pa", 1.: "hPa", 10.: "kPa"}[factor], "positive": "down"}
        elif name == "lat":
            attrs = {"units": "degrees_north"}
        elif name == "lon":
            attrs = {"units": "degrees_east"}
        coords[name] = (coord.dims, coord.data, attrs)
    return xr.DataArray(field.data, dims=field.dims, coords=coords, name=field.name, attrs={"units": units})


def find_hc_edge_latitude(lat: np.ndarray, psi_500: np.ndarray, *,
                          core_bounds: tuple[float, float] = (0., 25.),
                          search_bounds: tuple[float, float] = (20., 40.)) -> tuple[float, float]:
    """First poleward zero crossing after the strongest core in the core band.

    Undefined cores/edges return NaN. Missing grid points are not bridged, and
    an identically zero core has no identifiable cell boundary.
    """
    latitude, values = np.asarray(lat, dtype=float), np.asarray(psi_500, dtype=float)
    if latitude.ndim != 1 or latitude.shape != values.shape:
        raise ValueError("Edge diagnosis requires matching one-dimensional latitude and psi arrays.")
    if core_bounds[0] >= core_bounds[1] or search_bounds[0] >= search_bounds[1]:
        raise ValueError("Core and search latitude bounds must increase.")
    order = np.argsort(latitude)
    latitude, values = latitude[order], values[order]
    valid = np.isfinite(latitude) & np.isfinite(values)
    core = valid & (latitude >= core_bounds[0]) & (latitude <= core_bounds[1])
    if valid.sum() < 3 or not core.any():
        return np.nan, np.nan
    core_indices = np.where(core)[0]
    core_index = int(core_indices[np.argmax(np.abs(values[core]))])
    core_lat = float(latitude[core_index])
    if values[core_index] == 0.:
        return np.nan, np.nan
    lower, upper = max(float(search_bounds[0]), core_lat), float(search_bounds[1])
    for i in range(latitude.size - 1):
        lat0, lat1 = latitude[i:i + 2]
        if lat0 < lower or lat1 > upper or not valid[i:i + 2].all():
            continue
        value0, value1 = values[i:i + 2]
        if value0 == 0.:
            return core_lat, float(lat0)
        if value1 == 0.:
            return core_lat, float(lat1)
        if np.signbit(value0) != np.signbit(value1):
            return core_lat, float(lat0 - value0 * (lat1 - lat0) / (value1 - value0))
    return core_lat, np.nan


def compute_hc_edge_timeseries(psi_reg: xr.DataArray, *, target_level: float = 500., smooth: bool = True,
                               core_bounds: tuple[float, float] = (0., 25.),
                               search_bounds: tuple[float, float] = (20., 40.)) -> pd.DataFrame:
    """Interpolate a pressure slice and diagnose each year's northern edge."""
    section = psi_reg.sortby("level").interp(level=float(target_level)).transpose("time", "lat")
    if smooth:
        # Keep missing central grid points missing rather than fill data gaps.
        section = section.rolling(lat=3, center=True, min_periods=1).mean().where(np.isfinite(section))
    years = _years(psi_reg)
    rows = []
    for i, year in enumerate(years):
        core, edge = find_hc_edge_latitude(section["lat"].values, section.isel(time=i).values,
                                          core_bounds=core_bounds, search_bounds=search_bounds)
        rows.append({"time": str(psi_reg["time"].values[i])[:10], "year": int(year),
                     "hc_core_latitude_deg": core, "hc_edge_latitude_deg": edge})
    return pd.DataFrame(rows)


def edge_regression_stats(edge_df: pd.DataFrame, predictor: xr.DataArray, *, detrend: bool = True,
                           alpha: float = .1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit finite edge/index pairs only; absent boundaries retain n_samples=0."""
    pred = pd.DataFrame({"year": _years(predictor), "index_for_regression": np.asarray(predictor.values, dtype=float)})
    merged = edge_df.merge(pred, on="year", how="inner", validate="one_to_one").sort_values("year").reset_index(drop=True)
    finite = np.isfinite(merged["hc_edge_latitude_deg"]) & np.isfinite(merged["index_for_regression"])
    count = int(finite.sum())
    merged["hc_edge_processed_deg"] = np.nan
    merged["index_standardized_for_edge"] = np.nan
    row = {"n_samples": count, "start_year": np.nan, "end_year": np.nan,
           "correlation": np.nan, "correlation_pvalue": np.nan,
           "regression_slope_deg_per_std_index": np.nan, "regression_intercept_deg": np.nan,
           "regression_pvalue": np.nan, "regression_stderr": np.nan, "significant": False,
           "alpha": float(alpha), "detrend_edge": bool(detrend)}
    if count:
        years = merged.loc[finite, "year"].to_numpy(dtype=int)
        row.update(start_year=int(years.min()), end_year=int(years.max()))
    if count >= 3:
        coords = {"time": pd.to_datetime([f"{year:04d}-01-01" for year in years]), "year": ("time", years)}
        y = xr.DataArray(merged.loc[finite, "hc_edge_latitude_deg"].to_numpy(dtype=float), dims="time", coords=coords)
        x = xr.DataArray(merged.loc[finite, "index_for_regression"].to_numpy(dtype=float), dims="time", coords=coords)
        if detrend:
            x, y = detrend_along_time(x), detrend_along_time(y)
        try:
            x = standardize_series(x)
        except ValueError:
            return merged, pd.DataFrame([row])
        merged.loc[finite, "hc_edge_processed_deg"] = y.values
        merged.loc[finite, "index_standardized_for_edge"] = x.values
        yvalues = np.asarray(y.values, dtype=float)
        tolerance = np.finfo(float).eps * max(1., float(np.max(np.abs(yvalues)))) * 100.
        if np.std(yvalues, ddof=1) > tolerance:
            fit = stats.linregress(x.values, yvalues)
            row.update(correlation=float(fit.rvalue), correlation_pvalue=float(fit.pvalue),
                regression_slope_deg_per_std_index=float(fit.slope), regression_intercept_deg=float(fit.intercept),
                regression_pvalue=float(fit.pvalue), regression_stderr=float(fit.stderr), significant=bool(fit.pvalue < alpha))
    return merged, pd.DataFrame([row])


def analyze(vd: xr.DataArray, predictor: xr.DataArray, *, baseline_start: str = "1981", baseline_end: str = "2010",
            detrend_fields: bool = True, detrend_index: bool = True, regional_reduction: str = "mean",
            main_lon_bounds: tuple[float, float] = (60., 120.),
            sensitivity_lon_bounds: tuple[float, float] = (70., 110.), alpha: float = .1,
            edge_target_level: float = 500., smooth_edge: bool = True,
            edge_core_bounds: tuple[float, float] = (0., 25.),
            edge_search_bounds: tuple[float, float] = (20., 40.)) -> dict[str, Any]:
    """Analyze prepared seasonal vd and index after aligning their common years.

    Returns psi, intensity and regional xarray Datasets, the processed predictor
    DataArray, and a pandas summary DataFrame. Core pressure coordinates are hPa
    unless explicit recognized pressure units are present.
    """
    _validate_wind_units(vd)
    aligned, raw_index = _align_common_years(vd, predictor)
    pressure_factor = _pressure_factor(aligned["level"].attrs.get("units", "hPa"))
    aligned = aligned.assign_coords(level=np.asarray(aligned["level"].values, dtype=float) * pressure_factor)
    aligned["level"].attrs = {"units": "hPa", "positive": "down"}
    aligned = _clean_core_array(aligned, units="m s-1")
    raw_index = _clean_core_array(raw_index, units="1")
    processed = detrend_along_time(raw_index) if detrend_index else raw_index.copy()
    index = standardize_series(processed).rename("index_for_regression")
    index.attrs["detrend_method"] = "linear_common_period" if detrend_index else "none"
    psi = compute_local_mass_streamfunction(aligned)
    psi_clim, psi_anom = compute_climatology_anomaly(psi, baseline_start=baseline_start, baseline_end=baseline_end,
                                                   detrend=detrend_fields, anomaly_name="psi_anomaly")
    intensity = compute_hc_intensity(psi)
    intensity_clim, intensity_anom = compute_climatology_anomaly(intensity, baseline_start=baseline_start,
            baseline_end=baseline_end, detrend=detrend_fields, anomaly_name="hc_intensity_anomaly")
    regression = regress_to_nino34(intensity_anom, index, alpha=alpha)
    intensity_ds = xr.Dataset({"hc_intensity": intensity, "hc_intensity_climatology": intensity_clim,
            "hc_intensity_anomaly": intensity_anom, "index_for_regression": index,
            **{f"hc_intensity_{name}": regression[name] for name in ("regression", "pvalue", "significance", "sample_count")}})
    regional_vars = {}
    summary = []
    edge_tables, edge_stats_tables = [], []
    for label, bounds in (("main", main_lon_bounds), ("sensitivity", sensitivity_lon_bounds)):
        reduced = regional_reduce_psi(psi, lon_bounds=bounds, reduction=regional_reduction)
        clim, anom = compute_climatology_anomaly(reduced, baseline_start=baseline_start, baseline_end=baseline_end,
                                                 detrend=detrend_fields, anomaly_name=f"{label}_psi_anomaly")
        case = regress_to_nino34(anom, index, alpha=alpha)
        regional_vars.update({f"{label}_psi": reduced, f"{label}_psi_climatology": clim, f"{label}_psi_anomaly": anom})
        regional_vars.update({f"{label}_psi_{name}": case[name] for name in ("regression", "pvalue", "significance", "sample_count")})
        edge = compute_hc_edge_timeseries(reduced, target_level=edge_target_level, smooth=smooth_edge,
                                         core_bounds=edge_core_bounds, search_bounds=edge_search_bounds)
        edge_merged, edge_stats = edge_regression_stats(edge, index, detrend=detrend_fields, alpha=alpha)
        for table in (edge_merged, edge_stats):
            table["region"] = label
            table["lon_min"], table["lon_max"] = bounds
        edge_tables.append(edge_merged)
        edge_stats_tables.append(edge_stats)
        summary.append({"region": label, "lon_min": bounds[0], "lon_max": bounds[1], "lon_count": reduced.attrs["lon_count"],
                        "regional_reduction": regional_reduction, "common_start_year": int(_years(aligned).min()),
                        "common_end_year": int(_years(aligned).max()), "common_sample_count": aligned.sizes["time"],
                        "pressure_top_hpa": float(psi["level"].min()), "pressure_bottom_hpa": float(psi["level"].max()),
                        "alpha": alpha, "detrend_fields": detrend_fields, "detrend_index": detrend_index})
    psi_ds = xr.Dataset({"psi": psi, "psi_climatology": psi_clim, "psi_anomaly": psi_anom,
                        "vd": aligned, "index_raw_common": raw_index, "index_for_regression": index})
    regional_ds = xr.Dataset(regional_vars)
    common_settings = {"method": "regional_Hadley_local_mass_streamfunction", "baseline_start": str(baseline_start),
            "baseline_end": str(baseline_end), "common_start_year": int(_years(aligned).min()),
            "common_end_year": int(_years(aligned).max()), "common_sample_count": aligned.sizes["time"],
            "detrend_fields": int(detrend_fields), "detrend_index": int(detrend_index), "alpha": float(alpha),
            "regional_reduction": regional_reduction, "top_boundary": "zero_at_uppermost_selected_level",
            "bottom_residual_correction": "none", "significance_test": "two_sided_Pearson_t_no_autocorrelation_adjustment",
            "edge_target_level_hpa": float(edge_target_level), "edge_smoothing": "3_point_latitude" if smooth_edge else "none",
            "edge_core_lat_min": float(edge_core_bounds[0]), "edge_core_lat_max": float(edge_core_bounds[1]),
            "edge_search_lat_min": float(edge_search_bounds[0]), "edge_search_lat_max": float(edge_search_bounds[1])}
    for ds in (psi_ds, intensity_ds, regional_ds):
        ds.attrs = dict(common_settings)
    intensity_ds.attrs["intensity_definition"] = "max_level(abs(original_psi))_before_anomaly_and_regression"
    return {"psi": psi_ds, "intensity": intensity_ds, "regional": regional_ds,
            "predictor": index, "summary": pd.DataFrame(summary),
            "edge_timeseries": pd.concat(edge_tables, ignore_index=True),
            "edge_statistics": pd.concat(edge_stats_tables, ignore_index=True)}


def read_index_csv(path: str | Path, *, index_column: str = "nino34", season: str = "SON") -> xr.DataArray:
    table = pd.read_csv(path)
    if index_column not in table:
        raise ValueError("The requested index column is absent from the CSV.")
    if "season" in table:
        table = table.loc[table["season"].astype(str).str.upper() == season.upper()].copy()
    if table.empty:
        raise ValueError("No rows match the requested index season.")
    values = pd.to_numeric(table[index_column], errors="coerce").to_numpy(dtype=float)
    if "year" in table:
        years = pd.to_numeric(table["year"], errors="raise").to_numpy(dtype=int)
        if np.unique(years).size != years.size:
            raise ValueError("Year-format index CSV must contain one target-season row per year.")
        times = [_seasonal_time(int(year), season) for year in years]
        out = xr.DataArray(values, dims="time", coords={"time": times, "year": ("time", years)}, name="index")
    elif "time" in table:
        times = pd.DatetimeIndex(pd.to_datetime(table["time"], errors="raise"))
        out = xr.DataArray(values, dims="time", coords={"time": times}, name="index")
        if pd.Index(times.year).has_duplicates:
            out = seasonalize_winds(out.to_dataset(), season=season, frequency="monthly")["index"]
        else:
            out = out.assign_coords(year=("time", times.year))
    else:
        raise ValueError("Index CSV requires either year or time, plus the requested index column.")
    return out.sortby("time")


def _read_winds(paths: list[str], *, variables: list[str] | None = None,
                pressure_levels: list[float] | None = None, pressure_units: str | None = None) -> xr.Dataset:
    """Select requested variables and levels before loading arrays into memory."""
    opened = []
    for item in paths:
        suffix = Path(item).suffix.lower()
        kwargs = {"engine": "cfgrib", "backend_kwargs": {"indexpath": ""}} if suffix in {".grib", ".grb", ".grib2", ".grb2"} else {}
        opened.append(xr.open_dataset(item, **kwargs))
    try:
        combined = opened[0] if len(opened) == 1 else xr.combine_by_coords(opened, combine_attrs="drop")
        combined = normalize_wind_dataset(combined, pressure_units=pressure_units)
        if variables is not None:
            if any(name not in combined for name in variables):
                raise ValueError("One or more requested wind variables are absent from the input dataset.")
            combined = combined[variables]
        if pressure_levels is not None:
            available = np.asarray(combined["level"].values, dtype=float)
            requested = np.asarray(pressure_levels, dtype=float)
            keep = np.isclose(available[:, None], requested[None, :], rtol=0, atol=1.e-4).any(axis=1)
            if int(keep.sum()) < 2:
                raise ValueError("At least two requested pressure levels must be present in the input.")
            combined = combined.isel(level=np.where(keep)[0])
        return combined.load()
    finally:
        for dataset in opened:
            dataset.close()


def _safe_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """Discard inherited source metadata and encodings before serialization."""
    out = ds.drop_vars([name for name in ds.coords if name not in {"time", "year", "level", "lat", "lon"}]).copy(deep=False)
    safe_keys = {"units", "method", "definition", "top_boundary_hpa", "bottom_level_hpa", "top_boundary_value",
            "residual_correction", "delta_lon_rad", "regional_reduction", "lon_min", "lon_max", "lon_count",
            "longitude_endpoints", "baseline_start_year", "baseline_end_year", "baseline_sample_count",
            "detrend_method", "test", "predictor", "effective_sample_size_adjustment", "alpha",
            "missing_or_undefined_test", "description", "standardization", "standardized_mean_removed",
            "standardized_std_removed", "positive"}
    for name in out.variables:
        out[name].encoding = {}
        if name in ds.coords:
            if name == "level":
                out[name].attrs = {"units": "hPa", "positive": "down"}
            elif name == "lat":
                out[name].attrs = {"units": "degrees_north"}
            elif name == "lon":
                out[name].attrs = {"units": "degrees_east"}
            else:
                out[name].attrs = {}
        elif name in {"vd", "index_raw_common"}:
            out[name].attrs = {"units": "m s-1" if name == "vd" else "1"}
        else:
            out[name].attrs = {key: value for key, value in ds[name].attrs.items() if key in safe_keys}
        if out[name].dtype == bool:
            out[name] = out[name].astype(np.int8)
    out.encoding = {}
    return out


def save_plots(results: dict[str, Any], output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    intensity = results["intensity"]
    response = intensity["hc_intensity_regression"] / 1.e9
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    limit = float(np.nanmax(np.abs(response.values))) if np.isfinite(response.values).any() else 1.
    limit = max(limit, np.finfo(float).eps)
    image = ax.pcolormesh(response["lon"], response["lat"], response, shading="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    sig = intensity["hc_intensity_significance"].values
    yy, xx = np.meshgrid(response["lat"].values, response["lon"].values, indexing="ij")
    ax.scatter(xx[sig], yy[sig], s=1, color="black", alpha=.45)
    ax.set(xlabel="Longitude (degrees east)", ylabel="Latitude (degrees north)", title="Hadley intensity response to seasonal index")
    fig.colorbar(image, ax=ax, label="10⁹ kg s⁻¹ per index standard deviation")
    fig.savefig(output_dir / "horizontal_response.png", dpi=180)
    plt.close(fig)
    regional = results["regional"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    maximum = max(float(np.nanmax(np.abs(regional[f"{label}_psi_regression"].values)))
                  if np.isfinite(regional[f"{label}_psi_regression"].values).any() else 0. for label in ("main", "sensitivity")) / 1.e9
    maximum = max(maximum, np.finfo(float).eps)
    for ax, label in zip(axes, ("main", "sensitivity")):
        response = regional[f"{label}_psi_regression"] / 1.e9
        image = ax.pcolormesh(response["lat"], response["level"], response, shading="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
        climatology = regional[f"{label}_psi_climatology"].transpose("level", "lat") / 1.e9
        values = np.asarray(climatology.values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size and np.ptp(finite) > 0:
            # Select contour intervals from the actual scale, including small
            # longitude-mean transports, instead of fixed 2 or 5 unit steps.
            maximum_clim = float(np.max(np.abs(finite)))
            raw_step = maximum_clim / 5.
            exponent = 10. ** np.floor(np.log10(raw_step))
            mantissa = raw_step / exponent
            step = next(value for value in (1., 2., 2.5, 5., 10.) if value >= mantissa) * exponent
            levels = np.arange(-np.ceil(maximum_clim / step), np.ceil(maximum_clim / step) + 1.) * step
            for sign in (-1, 1):
                selected = levels[(levels * sign > 0) & (levels > finite.min()) & (levels < finite.max())]
                if selected.size:
                    contours = ax.contour(climatology["lat"], climatology["level"], values, levels=selected,
                                          colors="black", linewidths=.65, linestyles="--" if sign < 0 else "-")
                    ax.clabel(contours, inline=True, fmt="%.2g", fontsize=6)
            if finite.min() < 0 < finite.max():
                ax.contour(climatology["lat"], climatology["level"], values, levels=[0.], colors="black", linewidths=1.6)
        shifted = climatology + response
        shifted_values = np.asarray(shifted.values, dtype=float)
        shifted_finite = shifted_values[np.isfinite(shifted_values)]
        if shifted_finite.size and shifted_finite.min() < 0 < shifted_finite.max():
            ax.contour(shifted["lat"], shifted["level"], shifted_values, levels=[0.],
                       colors=".3", linestyles="--", linewidths=1.5)
        sig = regional[f"{label}_psi_significance"].values
        yy, xx = np.meshgrid(response["level"].values, response["lat"].values, indexing="ij")
        ax.scatter(xx[sig], yy[sig], s=2, color="black", alpha=.45)
        attrs = regional[f"{label}_psi"].attrs
        ax.set(xlabel="Latitude (degrees north)", ylabel="Pressure (hPa)",
               title=f"{label.capitalize()}: {attrs['lon_min']:g}–{attrs['lon_max']:g}°E ({attrs['regional_reduction']})")
        ax.set_yscale("log")
        ax.set_ylim(float(response["level"].max()), float(response["level"].min()))
        ticks = np.asarray(response["level"].values, dtype=float)
        ax.set_yticks(ticks, labels=[f"{value:g}" for value in ticks])
    fig.suptitle("Shading: index response; contours: climatology; gray dashed: climatology + 1 SD zero line", fontsize=10)
    fig.colorbar(image, ax=list(axes), label="10⁹ kg s⁻¹ per index standard deviation")
    fig.savefig(output_dir / "vertical_response.png", dpi=180)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--winds", nargs="+", required=True, help="NetCDF input files; optional GRIB requires cfgrib.")
    parser.add_argument("--index", required=True, help="CSV with year/time and the requested index column.")
    parser.add_argument("--index-column", default="nino34")
    parser.add_argument("--season", choices=tuple(SEASON_MONTHS), default="SON")
    parser.add_argument("--frequency", choices=("monthly", "seasonal"), default="monthly")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--baseline-start", default="1981")
    parser.add_argument("--baseline-end", default="2010")
    parser.add_argument("--input-kind", choices=("total", "divergent"), default="total")
    parser.add_argument("--u-var", default="u")
    parser.add_argument("--v-var", default="v")
    parser.add_argument("--wind-units", help="Explicit override for wind variable units, e.g. 'm s-1'.")
    parser.add_argument("--pressure-units", help="Explicit override for pressure coordinate units.")
    parser.add_argument("--pressure-levels", type=float, nargs="+", default=list(DEFAULT_PRESSURE_LEVELS), help="Selected levels in hPa.")
    parser.add_argument("--regional-reduction", choices=("mean", "sum"), default="mean")
    parser.add_argument("--main-lon-min", type=float, default=60.)
    parser.add_argument("--main-lon-max", type=float, default=120.)
    parser.add_argument("--sensitivity-lon-min", type=float, default=70.)
    parser.add_argument("--sensitivity-lon-max", type=float, default=110.)
    parser.add_argument("--detrend-fields", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detrend-index", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--edge-target-level", type=float, default=500.)
    parser.add_argument("--edge-core-lat-min", type=float, default=0.)
    parser.add_argument("--edge-core-lat-max", type=float, default=25.)
    parser.add_argument("--edge-search-lat-min", type=float, default=20.)
    parser.add_argument("--edge-search-lat-max", type=float, default=40.)
    parser.add_argument("--smooth-edge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--truncation", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing this script's named outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.start_year is not None and args.end_year is not None and args.start_year > args.end_year:
        raise ValueError("start-year must not exceed end-year.")
    output_names = ["mass_streamfunction.nc", "horizontal_intensity.nc", "regional_sections.nc", "predictor.csv", "summary.csv",
                    "edge_timeseries.csv", "edge_statistics.csv"]
    if not args.no_plots:
        output_names += ["horizontal_response.png", "vertical_response.png"]
    if not args.overwrite and any((args.output_dir / name).exists() for name in output_names):
        raise FileExistsError("Named outputs already exist; use --overwrite to replace them.")
    variables = [args.v_var] if args.input_kind == "divergent" else [args.u_var, args.v_var]
    ds = _read_winds(args.winds, variables=variables, pressure_levels=args.pressure_levels, pressure_units=args.pressure_units)
    requested = np.asarray(args.pressure_levels, dtype=float)
    if requested.size < 2 or not np.isfinite(requested).all() or np.unique(requested).size != requested.size or np.any(requested <= 0):
        raise ValueError("Requested pressure levels must contain at least two distinct positive values.")
    available = np.asarray(ds["level"].values, dtype=float)
    indices = []
    for level in sorted(requested):
        matches = np.where(np.isclose(available, level, rtol=0, atol=1.e-4))[0]
        if matches.size != 1:
            raise ValueError(f"Requested pressure level {level:g} hPa is missing or ambiguous.")
        indices.append(int(matches[0]))
    if any(name not in ds for name in variables):
        raise ValueError("One or more requested wind variables are absent from the input dataset.")
    ds = ds[variables].isel(level=indices)
    for name in variables:
        if set(ds[name].dims) != {"time", "level", "lat", "lon"}:
            raise ValueError("Wind variables must have only time, level, lat, and lon dimensions.")
        if args.wind_units is not None:
            ds[name].attrs["units"] = args.wind_units
        _validate_wind_units(ds[name])
    seasonal = seasonalize_winds(ds, season=args.season, frequency=args.frequency, start_year=args.start_year, end_year=args.end_year)
    index = read_index_csv(args.index, index_column=args.index_column, season=args.season)
    # Restrict wind samples before decomposition as well as before detrending.
    aligned_v, aligned_index = _align_common_years(seasonal[args.v_var], index)
    seasonal = seasonal.sel(time=aligned_v["time"])
    for name in variables:
        validate_global_grid(seasonal[name], require_poles=True)
    if args.input_kind == "total":
        divwind = compute_seasonal_divergent_wind(seasonal[args.u_var], seasonal[args.v_var], truncation=args.truncation)
        vd = divwind["vd"]
    else:
        vd = seasonal[args.v_var].rename("vd")
    results = analyze(vd, aligned_index, baseline_start=args.baseline_start, baseline_end=args.baseline_end,
            detrend_fields=args.detrend_fields, detrend_index=args.detrend_index, regional_reduction=args.regional_reduction,
            main_lon_bounds=(args.main_lon_min, args.main_lon_max),
            sensitivity_lon_bounds=(args.sensitivity_lon_min, args.sensitivity_lon_max), alpha=args.alpha,
            edge_target_level=args.edge_target_level, smooth_edge=args.smooth_edge,
            edge_core_bounds=(args.edge_core_lat_min, args.edge_core_lat_max),
            edge_search_bounds=(args.edge_search_lat_min, args.edge_search_lat_max))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key, filename in (("psi", "mass_streamfunction.nc"), ("intensity", "horizontal_intensity.nc"), ("regional", "regional_sections.nc")):
        results[key].attrs.update({"season": args.season, "input_kind": args.input_kind, "input_frequency": args.frequency,
                                  "spectral_truncation": "none" if args.truncation is None else int(args.truncation)})
        _safe_netcdf(results[key]).to_netcdf(args.output_dir / filename)
    predictor = results["predictor"]
    pd.DataFrame({"year": _years(predictor), "index_standardized": predictor.values}).to_csv(args.output_dir / "predictor.csv", index=False)
    results["summary"].to_csv(args.output_dir / "summary.csv", index=False)
    results["edge_timeseries"].to_csv(args.output_dir / "edge_timeseries.csv", index=False)
    results["edge_statistics"].to_csv(args.output_dir / "edge_statistics.csv", index=False)
    if not args.no_plots:
        save_plots(results, args.output_dir)
    print(f"Completed {args.season}: {predictor.sizes['time']} common years; regional reduction={args.regional_reduction}.")


if __name__ == "__main__":
    main()
