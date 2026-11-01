"""Explicit lazy preparation and compatible legacy monthly commands.

References: original Climate-Base CLI at d0ad09e (see SOURCE_NOTES.md).
Climate-Base improvements: versioned prepare/inspect configuration is separate
from the legacy check/seasonal contract; preparation computes only on output.
"""
import argparse
import json
from pathlib import Path
from .archive import MonthlyArchive
from .seasons import season_dates, iter_seasons
from .io import write_seasonal_stream


def load_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {"data_root", "fields", "archive", "analysis", "output"}
    if set(config) - allowed:
        raise ValueError("Unknown configuration keys: " + ", ".join(sorted(set(config) - allowed)))
    root = Path(config.get("data_root", ".")).expanduser()
    if not root.is_absolute():
        root = path.parent / root
    analysis = config["analysis"]
    if set(analysis) - {"field", "start_year", "end_year", "months", "weighting"}:
        raise ValueError("Unknown analysis configuration keys")
    if analysis.get("weighting", "days") not in {"days", "equal"}:
        raise ValueError("analysis.weighting must be days or equal")
    dates = season_dates(analysis["start_year"], analysis["end_year"], analysis.get("months", (12, 1, 2)))
    archive = MonthlyArchive(config["fields"], data_root=root, **config.get("archive", {}))
    if analysis["field"] not in archive.fields:
        raise ValueError("analysis.field is not a configured field")
    output = Path(config["output"]).expanduser() if "output" in config else None
    if output is not None and not output.is_absolute():
        output = path.parent / output
    return archive, analysis, dates, output


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare local CF NetCDF with xarray/Dask or run compatible monthly commands."
    )
    parser.add_argument("action", choices=("check", "seasonal", "inspect", "prepare"))
    parser.add_argument("config", type=Path)
    args = parser.parse_args(argv)
    if args.action in {"inspect", "prepare"}:
        return _prepare_command(args.action, args.config, parser)
    archive, analysis, dates, output = load_config(args.config)
    if args.action == "check":
        records = [archive.inspect(name, year, month) for name in archive.fields for _, year, month in dates]
        print(json.dumps({"state": "checked", "records": len(records), "calendar": archive.calendar,
                          "fields": list(archive.fields), "audit": archive.audit, "inventory": archive.inventory,
                          "field_values_read": False, "outputs_written": False}, ensure_ascii=False, indent=2))
        return 0
    if output is None:
        parser.error("seasonal requires a new output NetCDF filename in the configuration")
    if output.exists() or output.is_symlink():
        raise FileExistsError("Output exists; choose a new filename")
    records = iter_seasons(archive, analysis["field"], analysis["start_year"], analysis["end_year"],
                           months=analysis.get("months", (12, 1, 2)), weighting=analysis.get("weighting", "days"))
    write_seasonal_stream(output, records, pressure_units=archive.pressure_units,
                          inventory=lambda: archive.inventory, audit=lambda: archive.audit)
    from netCDF4 import Dataset
    with Dataset(output) as ds:
        summary = {"state": "complete", "output": str(output), "years": ds["season_year"][:].tolist(),
                   "shape": list(ds["seasonal_value"].shape), "calendar": ds.calendar, "units": ds["seasonal_value"].units}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _prepare_command(action, path, parser):
    from .config import load_prepare_config
    from .pipeline import prepare_dataset
    from .io import write_dataset

    config = load_prepare_config(path)
    if action == "prepare":
        if config.output is None:
            parser.error("prepare requires output in the configuration")
        if config.output.exists() or config.output.is_symlink():
            raise FileExistsError("Output exists; choose a new NetCDF filename")
    with prepare_dataset(config.inputs, **config.options) as dataset:
        summary = {
            "state": "inspected" if action == "inspect" else "prepared",
            "dimensions": dict(dataset.sizes),
            "variables": {
                name: {"dimensions": list(variable.dims), "units": variable.attrs.get("units"),
                       "chunks": variable.chunks}
                for name, variable in dataset.data_vars.items()
            },
            "field_values_read": action == "prepare",
            "outputs_written": action == "prepare",
        }
        if action == "prepare":
            summary["output"] = str(write_dataset(dataset, config.output))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    main()
