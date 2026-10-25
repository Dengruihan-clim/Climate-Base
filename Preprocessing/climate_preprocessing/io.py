"""Atomic, non-overwriting export of normalized seasonal series."""
from __future__ import annotations
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def write_seasonal(path, result, *, pressure_units=None, inventory=(), audit=None):
    """Write a new NetCDF file; pressure units must be explicitly supplied.

    The field uses (season_year, [pressure], latitude, longitude).  Provenance
    should contain privacy-safe relative filenames, as returned by an archive.
    A final output, including a dangling symlink, is never overwritten.
    """
    return write_seasonal_stream(path, iter((result,)), pressure_units=pressure_units, inventory=inventory, audit=audit)


def write_seasonal_stream(path, records, *, pressure_units=None, inventory=(), audit=None):
    """Stream SeasonalField records to one new, atomically published file.

    ``inventory`` and ``audit`` may be callables so provenance can be collected
    after a lazy input iterator has finished.  On any failure the temporary
    file is removed and no final output is published.
    """
    from netCDF4 import Dataset
    destination = Path(path).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Output exists; choose a new NetCDF filename")
    iterator = iter(records)
    try:
        result = next(iterator)
    except StopIteration as error:
        raise ValueError("At least one seasonal record is required") from error
    if result.pressure is not None and pressure_units not in {"Pa", "hPa"}:
        raise ValueError("Supply pressure_units='Pa' or 'hPa' for pressure-level output")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, filename = tempfile.mkstemp(prefix="." + destination.name + ".", suffix=".partial", dir=destination.parent)
    os.close(fd)
    temporary = Path(filename)
    try:
        with Dataset(temporary, "w", format="NETCDF4") as ds:
            ds.createDimension("season_year", None)
            years_var = ds.createVariable("season_year", "i4", ("season_year",))
            years_var.units = "1"
            axes = [("latitude", result.latitude, "degrees_north"),
                    ("longitude", result.longitude, "degrees_east")]
            if result.pressure is not None:
                axes.append(("pressure", result.pressure, pressure_units))
            for name, values, units in axes:
                ds.createDimension(name, len(values))
                var = ds.createVariable(name, "i4" if name == "season_year" else "f8", (name,))
                var[:] = values
                var.units = units
            ds["season_year"].long_name = "Season year, labelled by year of the final configured month"
            if result.pressure is not None:
                ds["pressure"].positive = "down"
                ds["pressure"].standard_name = "air_pressure"
            dims = ("season_year",) + (("pressure",) if result.pressure is not None else ()) + ("latitude", "longitude")
            var = ds.createVariable("seasonal_value", "f8", dims, zlib=True, complevel=4, fill_value=-9999.)
            var.units = result.units
            position, last_year = 0, None

            def append(record):
                nonlocal position, last_year
                if (record.calendar != result.calendar or record.units != result.units
                        or record.months != result.months or record.weighting != result.weighting
                        or not np.array_equal(record.latitude, result.latitude)
                        or not np.array_equal(record.longitude, result.longitude)
                        or (record.pressure is None) != (result.pressure is None)
                        or (record.pressure is not None and not np.array_equal(record.pressure, result.pressure))):
                    raise ValueError("Seasonal stream metadata/grid changed")
                years = np.asarray(record.years)
                if years.ndim != 1 or years.size == 0 or not np.issubdtype(years.dtype, np.integer):
                    raise ValueError("Season years must be a nonempty integer vector")
                expected = (len(years),) + ((len(result.pressure),) if result.pressure is not None else ()) + (len(result.latitude), len(result.longitude))
                if (np.any(np.diff(years) <= 0) or (last_year is not None and years[0] <= last_year)
                        or record.values.shape != expected):
                    raise ValueError("Seasonal stream years or value dimensions are invalid")
                if np.iscomplexobj(record.values):
                    raise ValueError("Seasonal output values must be real")
                stop = position + len(years)
                ds["season_year"][position:stop] = years
                ds["seasonal_value"][position:stop] = np.ma.masked_invalid(record.values)
                position, last_year = stop, int(years[-1])

            append(result)
            for record in iterator:
                append(record)
            final_inventory = inventory() if callable(inventory) else inventory
            final_audit = audit() if callable(audit) else audit
            ds.setncatts({"title": "Complete calendar-aware seasonal field", "authors": "RH_Deng; Codex (OpenAI)",
                          "created_utc": datetime.now(timezone.utc).isoformat(), "calendar": result.calendar,
                          "season_months": json.dumps(list(result.months)), "weighting": result.weighting,
                          "missing_policy": "All season months must be finite; no partial-month renormalization or imputation",
                          "input_inventory": json.dumps(list(final_inventory), ensure_ascii=False),
                          "preprocessing_audit": json.dumps(final_audit or {}, ensure_ascii=False)})
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
