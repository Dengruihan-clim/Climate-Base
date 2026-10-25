"""Scientific pipeline adapted from the verified Python LBM builders."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import netCDF4
from netCDF4 import Dataset
import numpy as np
import scipy

from . import __version__
from . import binary
from .horizontal import gaussian_grid, scalar_triangular_truncate
from .inputs import load_climatology, season_dates
from .vertical import (RAIR, height_to_surface_pressure, pressure_to_sigma,
                       sea_level_to_surface_pressure)


GUIDE = "https://github.com/HanXiang0721/Note-Numerical-Methods/blob/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md"


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_references(config):
    model = config["model"]
    for key in ("topography", "sigma", "template"):
        if not Path(model[key]).is_file():
            raise FileNotFoundError(f"Provide model.{key}: {model[key]}")
    sigma = binary.read_sigma(model["sigma"])
    records = binary.read_gtool(model["topography"])
    if len(records) != 1:
        raise ValueError("The topography reference must contain exactly one GRZ field")
    record = records[0]
    axes = tuple(record.header[i].strip() for i in (28, 31))
    if (record.item != "GRZ" or record.units not in ("m", "gpm")
            or record.data.shape != (1, 64, 128) or axes != ("GLON128", "GGLA64")):
        raise ValueError("Topography must be GRZ in metres on the north-to-south 128x64 T42 Gaussian grid")
    topography = record.data[0].astype(np.float64)
    if not np.isfinite(topography).all() or np.any(topography == binary.MISSING):
        raise ValueError("Model topography must contain finite valid heights")
    binary._validate_template(binary.read_gtool(model["template"]))
    return sigma, topography


def preflight(config):
    sigma, topography = read_references(config)
    dates = season_dates(config["period"])
    return {"state": "configuration_and_model_references_valid",
            "name": config["name"], "target": "T42L20, 128x64 Gaussian, north to south",
            "sigma_levels": len(sigma), "model_reference_files": 3,
            "topography_range_m": [float(topography.min()), float(topography.max())],
            "season_years": [config["period"]["start_year"], config["period"]["end_year"]],
            "months": config["period"].get("months", [12, 1, 2]),
            "required_calendar_months": [f"{year:04d}-{month:02d}" for _, year, month in dates],
            "reanalysis_data_checked": False, "outputs_written": False}


def save_netcdf(path, fields, lat, lon, vertical, vertical_name, attrs, extras=None):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    with Dataset(temporary, "w", format="NETCDF4") as ds:
        for dimension, length in (("lat", len(lat)), ("lon", len(lon)), (vertical_name, len(vertical))):
            ds.createDimension(dimension, length)
        for name, values, units in (("lat", lat, "degrees_north"), ("lon", lon, "degrees_east"),
                                     (vertical_name, vertical, "1" if vertical_name == "sigma" else "hPa")):
            variable = ds.createVariable(name, "f8", (name,))
            variable[:] = values
            variable.units = units
        ds["lat"].standard_name = "latitude"
        ds["lon"].standard_name = "longitude"
        ds[vertical_name].positive = "down"
        if vertical_name == "sigma":
            ds["sigma"].standard_name = "atmosphere_sigma_coordinate"
            ds["sigma"].formula_terms = "sigma: sigma ps: PS ptop: ptop"
            ds.createVariable("ptop", "f8").assignValue(0.)
            ds["ptop"].units = "hPa"
        else:
            ds[vertical_name].standard_name = "air_pressure"
        units = {"U": "m s-1", "V": "m s-1", "T": "K", "Q": "kg kg-1", "H": "m",
                 "PS": "hPa", "PS_SOURCE": "hPa", "PS_HEIGHT": "hPa", "SLP": "hPa",
                 "topography": "m", "pressure": "hPa", "potential_temperature": "K"}
        for key, values in {**fields, **(extras or {})}.items():
            dims = (vertical_name, "lat", "lon") if np.ndim(values) == 3 else ("lat", "lon")
            variable = ds.createVariable(key, "f4", dims, zlib=True, complevel=4, fill_value=-999.)
            variable[:] = np.ma.masked_invalid(values)
            variable.units = units[key]
            variable.long_name = key
        ds.setncatts(attrs)
    temporary.replace(path)


def quality_checks(fields, sigma, qa):
    bounds = {"U": (-200, 200), "V": (-150, 150), "T": (150, 340), "PS": (350, 1100), "Q": (0, .05)}
    summary = {}
    for key, (lower, upper) in bounds.items():
        values = fields[key]
        if not np.isfinite(values).all() or values.min() < lower or values.max() > upper:
            raise ValueError(f"Invalid final {key}; expected finite values between {lower} and {upper}")
        summary[key] = {"min": float(values.min()), "max": float(values.max()),
                        "mean": float(values.mean()), "finite": True}
        if key != "Q":
            projected = scalar_triangular_truncate(values)
            relative = float(np.max(np.abs(projected - values)) / max(1., np.max(np.abs(values))))
            if relative > 1e-11:
                raise ValueError(f"Unexpected spectral power above T42 in {key}")
            summary[key]["relative_reprojection_error_before_float32"] = relative
    pressure = sigma[:, None, None] * fields["PS"][None]
    theta = fields["T"] * (1000. / pressure) ** (RAIR / 1004.6)
    delta = np.diff(theta, axis=0)
    qa["static_stability"] = {"minimum_adjacent_theta_difference_K": float(delta.min()),
                              "negative_layer_pair_fraction": float(np.mean(delta < 0)),
                              "columns_with_any_negative_difference": int(np.any(delta < 0, axis=0).sum()),
                              "pairs_below_minus_one_K": int((delta < -1).sum()),
                              "temperature_was_not_stability_adjusted": True}
    if delta.min() < -10 or np.mean(delta < 0) > .1:
        raise ValueError("Large static-stability problem; inspect temperature interpolation")
    qa["final_field_statistics"] = summary
    return pressure, theta


def plot_checks(path, fields, sigma, lat, lon, pressure_fields, plev, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, key, label, cmap in ((axes[0, 0], "U", "Zonal-mean U (m/s)", "RdBu_r"),
                                  (axes[0, 1], "T", "Zonal-mean T (K)", "viridis")):
        image = ax.contourf(lat, sigma, fields[key].mean(axis=-1), levels=21, cmap=cmap)
        ax.set_yscale("log")
        ax.set_ylim(1., sigma[-1])
        ax.set(title=label, xlabel="Latitude", ylabel="Sigma")
        figure.colorbar(image, ax=ax)
    image = axes[1, 0].pcolormesh(lon, lat, fields["PS"], shading="auto")
    axes[1, 0].set_title("Surface pressure on LBM topography (hPa)")
    figure.colorbar(image, ax=axes[1, 0])
    index = int(np.argmin(abs(plev - 200)))
    image = axes[1, 1].pcolormesh(lon, lat, pressure_fields["U"][index], shading="auto", cmap="RdBu_r")
    axes[1, 1].set_title(f"Source climatology U at {plev[index]:g} hPa (m/s)")
    figure.colorbar(image, ax=axes[1, 1])
    for ax in axes[1]:
        ax.set(xlabel="Longitude E", ylabel="Latitude")
    figure.suptitle(title)
    figure.savefig(path, dpi=170)
    figure.savefig(Path(path).with_suffix(".pdf"))
    plt.close(figure)


def build(config):
    """Build one configured seasonal climatology into a new output directory."""
    sigma, topography = read_references(config)
    if config["diagnostics"]["plot"]:
        try:
            import matplotlib  # noqa: F401 -- fail before creating output if optional dependency is absent
        except ImportError as error:
            raise RuntimeError("diagnostics.plot requires the optional matplotlib dependency") from error
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    state_path = output / "build_status.json"
    write_json(state_path, {"state": "running", "started_utc": now()})
    try:
        result = _build(config, output, sigma, topography)
    except BaseException as error:
        write_json(state_path, {"state": "failed", "updated_utc": now(),
                                "error_type": type(error).__name__,
                                "message": "Build failed. Inspect the terminal error; existing outputs were preserved."})
        raise
    write_json(state_path, result)
    return result


def _build(config, output, sigma, topography):
    lat, lon = gaussian_grid()
    period = config["period"]
    months = period.get("months", [12, 1, 2])
    title = f"{config['name']} {period['start_year']}-{period['end_year']}"
    attrs = {"Conventions": "CF-1.10", "title": title + " stationary dry LBM T42L20 basic state",
             "authors": "RH_Deng; Codex (OpenAI, AI-assisted refactoring)",
             "source_product": config["source_product"], "reference_guide": GUIDE,
             "climatology_start_year": period["start_year"], "climatology_end_year": period["end_year"],
             "climatology_months": ",".join(map(str, months)),
             "year_definition": "Year of the last configured season month; preceding December belongs to prior calendar year",
             "averaging": "Valid-year mean of each calendar month, then equal weights across configured months",
             "missing_values": "Monthly means require configured valid-year count; all monthly climatologies required; missing bilinear corners renormalized; invalid pressure levels discarded",
             "horizontal_method": "Periodic bilinear to exact 128x64 Gaussian grid; all sigma levels scalar triangular spectral projection at total degree 42",
             "vertical_method": "Natural cubic spline in ln(p); nearest valid endpoint extended outside profile",
             "surface_pressure_method": "ecmsbs SLP/virtual-temperature lapse correction to supplied LBM topography",
             "dry_mode": "Complete U V T PS Q; Q retained for virtual temperature and file compatibility",
             "created_utc": now(), "model_time_label": "0001-01-01 00:00:00; stationary climatology"}
    pressure_fields, plev, inventory, qa = load_climatology(config, lat, lon)
    write_json(output / "input_manifest.json", inventory)
    save_netcdf(output / "pressure_climatology_t42.nc", pressure_fields, lat, lon, plev, "plev", attrs)
    ps_height, qa["height_to_surface_pressure"] = height_to_surface_pressure(pressure_fields["H"], plev, topography)
    ps_height = scalar_triangular_truncate(ps_height)
    t_low, qa["temperature_for_surface_pressure"] = pressure_to_sigma(pressure_fields["T"], plev, ps_height, sigma[:1])
    q_low, qa["humidity_for_surface_pressure"] = pressure_to_sigma(pressure_fields["Q"], plev, ps_height, sigma[:1])
    t_low = scalar_triangular_truncate(t_low[0])
    q_low = scalar_triangular_truncate(q_low[0])
    qa["surface_humidity_negative_values_clipped"] = int((q_low < 0).sum())
    ps = sea_level_to_surface_pressure(pressure_fields["SLP"], topography, t_low, np.maximum(q_low, 0.), sigma[0])
    ps = scalar_triangular_truncate(ps)
    if "PS_SOURCE" in pressure_fields:
        difference = ps - pressure_fields["PS_SOURCE"]
        valid = difference[np.isfinite(difference)]
        qa["surface_pressure_difference_vs_source_hPa"] = {
            "min": float(valid.min()) if valid.size else None,
            "max": float(valid.max()) if valid.size else None,
            "rms": float(np.sqrt(np.mean(valid ** 2))) if valid.size else None,
            "finite_comparison_cells": int(valid.size),
            "status": "compared" if valid.size else "no_valid_comparison"}
    final = {"PS": ps}
    for key in ("U", "V", "T", "Q"):
        values, qa[key + "_pressure_to_sigma"] = pressure_to_sigma(pressure_fields[key], plev, ps, sigma)
        values = scalar_triangular_truncate(values)
        if key == "Q":
            qa["humidity_floor"] = {"negative_values": int((values < 0).sum()),
                                     "minimum_before_floor": float(values.min()),
                                     "rule": "Q>=0 after scalar T42 projection; passive dry-mode humidity only"}
            values = np.maximum(values, 0.)
        final[key] = values
    pressure, theta = quality_checks(final, sigma, qa)
    attrs["humidity_floor"] = json.dumps(qa["humidity_floor"])
    extras = {"topography": topography, "pressure": pressure, "potential_temperature": theta, "PS_HEIGHT": ps_height}
    if "PS_SOURCE" in pressure_fields:
        extras["PS_SOURCE"] = pressure_fields["PS_SOURCE"]
    save_netcdf(output / "background_sigma.nc", final, lat, lon, sigma, "sigma", attrs, extras)
    with Dataset(output / "background_sigma.nc") as ds:
        for key in binary.FIELD_ORDER:
            np.testing.assert_array_equal(ds[key][:], np.asarray(final[key], np.float32))
    qa["netcdf_roundtrip_exact_float32"] = True
    stem = config["name"] + ".t42l20"
    binary.write_background(config["model"]["template"], output / stem, final, title=title, year=1)
    binary.write_grads_sequential(output / (stem + ".grd"), final)
    binary.write_grads_ctl(output / (stem + ".ctl"), output / (stem + ".grd"), sigma, title=title, year=1)
    for record, key in zip(binary.read_gtool(output / stem), binary.FIELD_ORDER):
        expected = np.asarray(final[key], np.float32).reshape(record.data.shape)
        np.testing.assert_array_equal(record.data, expected)
    qa["binary_roundtrip_exact_float32"] = True
    qa["model_references"] = {key: {"filename": Path(value).name, "sha256": sha256(value)}
                               for key, value in config["model"].items() if key in ("topography", "sigma", "template")}
    qa["climatology"] = {"start_year": period["start_year"], "end_year": period["end_year"],
                          "months": months, "minimum_valid_years": period["minimum_valid_years"],
                          "year_label": "last month", "weighting": "equal calendar months"}
    qa["runtime"] = {"package": __version__, "python": platform.python_version(),
                      "numpy": np.__version__, "scipy": scipy.__version__, "netCDF4": netCDF4.__version__}
    write_json(output / "quality_report.json", qa)
    if config["diagnostics"]["plot"]:
        plot_checks(output / "background_diagnostics.png", final, sigma, lat, lon, pressure_fields, plev, title)
    files = {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in sorted(output.iterdir())
             if p.is_file() and p.name not in ("build_status.json", "output_manifest.json")}
    write_json(output / "output_manifest.json", files)
    return {"state": "complete", "completed_utc": now(), "source_files": len(inventory),
            "background": stem, "quality_report": "quality_report.json",
            "binary_roundtrip_exact_float32": True}
