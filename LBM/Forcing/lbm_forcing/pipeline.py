"""Monthly apparent-heating diagnosis, seasonal means and optional regression."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import cftime
from netCDF4 import Dataset
import numpy as np

from .diagnostics import CP, RD, EARTH_RADIUS_M, q1_layers
from .monthly_inputs import MonthlyReader, season_dates, neighbor_month
from .statistics import partial_regression

TERMS = ("q1", "tendency", "horizontal", "vertical")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def midpoint_seconds(year, month, calendar):
    """True calendar-month midpoint, independent of the product's time label."""
    next_year, next_month = neighbor_month(year, month, 1)
    start = cftime.datetime(year, month, 1, calendar=calendar)
    end = cftime.datetime(next_year, next_month, 1, calendar=calendar)
    return float(np.mean(cftime.date2num([start, end], "seconds since 1970-01-01", calendar=calendar)))


def calculation_grid(config):
    step = config["grid"]["step_degrees"]
    return np.linspace(-90., 90., round(180. / step) + 1), np.arange(round(360. / step)) * step


def check_diagnosis(config):
    dates = season_dates(config["period"])
    selected = sorted({(year, month) for _, year, month in dates})
    halo = sorted(set(selected) | {neighbor_month(year, month, offset) for year, month in selected for offset in (-1, 1)})
    lat, lon = calculation_grid(config)
    return {"state": "configuration_valid", "mode": "regression" if config["regression"] else "climatology",
            "target_shape": [len(lat), len(lon)], "season_years": [config["period"]["start_year"], config["period"]["end_year"]],
            "selected_months": [f"{y:04d}-{m:02d}" for y, m in selected],
            "temperature_and_ps_months_including_halo": [f"{y:04d}-{m:02d}" for y, m in halo],
            "raw_data_checked": False, "predictor_csv_checked": False, "outputs_written": False}


def load_predictors(config, years):
    settings = config["regression"]
    if settings is None:
        return None, None
    with Path(settings["file"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = [settings["year_column"], settings["predictor_column"]]
    if settings["control_column"]:
        required.append(settings["control_column"])
    if not rows or any(key not in rows[0] for key in required):
        raise ValueError("Predictor CSV lacks the configured column names")
    records = {}
    for row in rows:
        year = float(row[settings["year_column"]])
        if not np.isfinite(year) or year != round(year) or int(year) in records:
            raise ValueError("CSV season years must be finite, unique integers")
        records[int(year)] = row
    if any(int(year) not in records for year in years):
        raise ValueError("Predictor CSV is missing configured season years")
    x = np.array([float(records[int(y)][settings["predictor_column"]]) for y in years])
    z = None if not settings["control_column"] else np.array([float(records[int(y)][settings["control_column"]]) for y in years])
    if not np.isfinite(x).all() or (z is not None and not np.isfinite(z).all()):
        raise ValueError("Predictor and optional control must be finite")
    # Reject globally singular input designs before reading atmospheric data.
    partial_regression(x, np.zeros((len(years), 1)), years, z,
                       min_years=settings["min_years"], max_lag=settings["max_lag"])
    return x, z


def _coordinates(ds, years, pressure, lat, lon):
    for name, values, units in (("year", years, "season ending year"), ("pressure_hpa", pressure / 100., "hPa"),
                                 ("lat", lat, "degrees_north"), ("lon", lon, "degrees_east")):
        ds.createDimension(name, len(values))
        variable = ds.createVariable(name, "i4" if name == "year" else "f8", (name,))
        variable[:] = values
        variable.units = units
    ds["pressure_hpa"].positive = "down"
    ds["pressure_hpa"].standard_name = "air_pressure"


def _variable(ds, name, dims, units):
    variable = ds.createVariable(name, "f8", dims, zlib=True, complevel=3, fill_value=np.nan)
    variable.units = units
    return variable


def _season(reader, dates, lat, lon):
    temperatures, surfaces, pressure = {}, {}, None

    def get_t(year, month):
        nonlocal pressure
        key = (year, month)
        if key not in temperatures:
            temperatures[key], p = reader.read("T", year, month)
            if pressure is not None and not np.array_equal(pressure, p):
                raise ValueError("Pressure grid changed")
            pressure = p
        return temperatures[key]

    def get_ps(year, month):
        key = (year, month)
        if key not in surfaces:
            surfaces[key], _ = reader.read("PS", year, month)
        return surfaces[key]

    values = {name: [] for name in TERMS}
    for _, year, month in dates:
        previous = neighbor_month(year, month, -1)
        following = neighbor_month(year, month, 1)
        tprev, t, tnext = get_t(*previous), get_t(year, month), get_t(*following)
        winds = [reader.read(kind, year, month)[0] for kind in ("U", "V", "OMEGA")]
        calendar = reader.calendar
        middle = midpoint_seconds(year, month, calendar)
        result = q1_layers(tprev, t, tnext, *winds, pressure, lat, lon,
                           middle - midpoint_seconds(*previous, calendar),
                           midpoint_seconds(*following, calendar) - middle,
                           surface_pressure=get_ps(year, month), surface_pressure_prev=get_ps(*previous),
                           surface_pressure_next=get_ps(*following))
        for name in TERMS:
            values[name].append(result[name] * (CP / 86400.))
    ps_min = np.min(np.stack(list(surfaces.values())), axis=0)
    ps_mean = np.mean(np.stack([get_ps(y, m) for _, y, m in dates]), axis=0)
    seasonal = {name: np.mean(np.stack(fields), axis=0) for name, fields in values.items()}
    common = np.isfinite(ps_min)[None] & (pressure[:, None, None] <= ps_min[None])
    common &= np.logical_and.reduce([np.isfinite(value) for value in seasonal.values()])
    seasonal = {name: np.where(common, value, np.nan) for name, value in seasonal.items()}
    if not np.isfinite(seasonal["q1"]).any():
        raise ValueError("No valid seasonal heating cells; inspect input coverage and terrain masks")
    closure = seasonal["q1"] - sum(seasonal[name] for name in TERMS[1:])
    error = float(np.nanmax(abs(closure)))
    if error > 1e-10 * max(1., float(np.nanmax(abs(seasonal["q1"])))):
        raise ValueError("Seasonal Q1 terms do not close")
    return seasonal, pressure, ps_min, ps_mean, error


def diagnose(config):
    years = np.arange(config["period"]["start_year"], config["period"]["end_year"] + 1)
    x, control = load_predictors(config, years)
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    state = output / "build_status.json"
    write_json(state, {"state": "running", "started_utc": now()})
    try:
        result = _diagnose(config, years, x, control, output)
    except BaseException as error:
        write_json(state, {"state": "failed", "error_type": type(error).__name__,
                           "message": "Inspect the terminal error; existing outputs were preserved."})
        raise
    write_json(state, result)
    return result


def _diagnose(config, years, x, control, output):
    lat, lon = calculation_grid(config)
    reader = MonthlyReader(config, lat, lon)
    (output / "seasonal").mkdir()
    all_dates = season_dates(config["period"])
    caches, closure = [], []
    for year in years:
        dates = [row for row in all_dates if row[0] == year]
        fields, pressure, ps_min, ps_mean, error = _season(reader, dates, lat, lon)
        closure.append(error)
        path = output / "seasonal" / f"{year}.nc"
        temporary = Path(str(path) + ".partial")
        with Dataset(temporary, "w") as ds:
            _coordinates(ds, [year], pressure, lat, lon)
            for name, values in fields.items():
                _variable(ds, name, ("pressure_hpa", "lat", "lon"), "W kg-1")[:] = np.ma.masked_invalid(values)
            for name, values in (("ps_min", ps_min), ("ps_mean", ps_mean)):
                _variable(ds, name, ("lat", "lon"), "Pa")[:] = np.ma.masked_invalid(values)
            ds.calendar = reader.calendar
            ds.pressure_integrated = 0
            ds.complete = 1
        temporary.replace(path)
        caches.append(path)
    target = output / "diagnosed_heating.nc"
    temporary = Path(str(target) + ".partial")
    dims = ("pressure_hpa", "lat", "lon")
    finite_selected = 0
    with Dataset(temporary, "w") as ds:
        _coordinates(ds, years, pressure, lat, lon)
        ds.source_product = config["source_product"]
        ds.authors = "RH_Deng; Codex (OpenAI, tool organization and refactoring)"
        ds.analysis_kind = "regression" if x is not None else "climatology"
        ds.pressure_integrated = 0
        ds.calendar = reader.calendar
        ds.CP_J_kg_K, ds.RD_J_kg_K, ds.earth_radius_m = CP, RD, EARTH_RADIUS_M
        ds.method = "Full monthly PS-masked fields; calendar-midpoint unequal time derivative; equal-month complete seasonal means; no column integration"
        ds.predictor_scaling = "Original supplied values; no residual restandardization"
        if x is not None:
            ds.predictor_units = config["regression"]["predictor_units"]
        ds.created_utc = now()
        if x is not None:
            _variable(ds, "predictor", ("year",), config["regression"]["predictor_units"])[:] = x
            if control is not None:
                _variable(ds, "control", ("year",), "original supplied control unit")[:] = control
        selected = _variable(ds, "heating_k_day", dims, "K day-1")
        selected.amplitude_interpretation = "per one supplied predictor unit" if x is not None else "seasonal climatology"
        _variable(ds, "heating_w_kg", dims, "W kg-1")
        for name in ("ps_min", "ps_mean"):
            variable = _variable(ds, name, ("year", "lat", "lon"), "Pa")
            for index, path in enumerate(caches):
                with Dataset(path) as source:
                    variable[index] = source[name][:]
        for level in range(len(pressure)):
            fields = {name: [] for name in TERMS}
            for path in caches:
                with Dataset(path) as source:
                    for name in TERMS:
                        fields[name].append(np.ma.asarray(source[name][level], dtype=float).filled(np.nan))
            for name, values in fields.items():
                values = np.asarray(values)
                valid = np.isfinite(values)
                counts = valid.sum(axis=0)
                mean = np.divide(np.where(valid, values, 0.).sum(axis=0), counts,
                                 out=np.full(counts.shape, np.nan), where=counts > 0)
                stats = {"climatology": mean, "climatology_n": counts}
                if x is not None:
                    stats.update(partial_regression(x, values, years, control,
                                                    min_years=config["regression"]["min_years"], max_lag=config["regression"]["max_lag"]))
                for key, data in stats.items():
                    field_name = name + "_" + key
                    if field_name not in ds.variables:
                        units = ("W kg-1" if key in ("climatology", "beta") else
                                 "W kg-1 year-1" if key == "field_trend" else
                                 config["regression"]["predictor_units"] + " year-1" if key == "predictor_trend" else "1")
                        created = _variable(ds, field_name, dims, units)
                        if key == "beta":
                            created.per_predictor_unit = config["regression"]["predictor_units"]
                    ds[field_name][level] = np.ma.masked_invalid(data)
                if name == "q1":
                    energy = stats["beta"] if x is not None else mean
                    forcing = energy * (86400. / CP)
                    selected[level] = np.ma.masked_invalid(forcing)
                    ds["heating_w_kg"][level] = np.ma.masked_invalid(energy)
                    finite_selected += int(np.isfinite(forcing).sum())
                    if x is not None:
                        if "q1_beta_k_day" not in ds.variables:
                            _variable(ds, "q1_beta_k_day", dims, "K day-1")
                        ds["q1_beta_k_day"][level] = np.ma.masked_invalid(forcing)
        if finite_selected == 0:
            raise ValueError("No finite selected heating; inspect regression sampling and source masks")
        ds.complete = 1
    temporary.replace(target)
    with Dataset(target) as ds:
        kday = np.ma.asarray(ds["heating_k_day"][:]).filled(np.nan)
        energy = np.ma.asarray(ds["heating_w_kg"][:]).filled(np.nan)
        np.testing.assert_allclose(kday, energy * (86400. / CP), rtol=1e-14, atol=1e-14, equal_nan=True)
    inventory = reader.inventory
    if config["regression"]:
        path = Path(config["regression"]["file"])
        inventory.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path), "role": "predictor CSV"})
    write_json(output / "input_manifest.json", inventory)
    quality = {"analysis": "regression" if x is not None else "climatology", "CP": CP,
               "finite_selected_cells": finite_selected, "maximum_seasonal_term_closure_error_W_kg": max(closure),
               "unit_conversion_readback": True, "calendar": reader.calendar, "season_years": years.tolist(),
               "months": config["period"].get("months", [12, 1, 2]), "monthly_inputs": reader.qa,
               "submonthly_transient_heat_transport_resolved": False, "pressure_integrated": False,
               "python": platform.python_version()}
    write_json(output / "quality_report.json", quality)
    manifest = {p.relative_to(output).as_posix(): {"bytes": p.stat().st_size, "sha256": sha256(p)}
                for p in sorted(output.rglob("*")) if p.is_file() and p.name not in ("output_manifest.json", "build_status.json")}
    write_json(output / "output_manifest.json", manifest)
    return {"state": "complete", "completed_utc": now(), "output": "diagnosed_heating.nc", "seasonal_files": len(caches),
            "analysis": quality["analysis"], "finite_selected_cells": finite_selected}
