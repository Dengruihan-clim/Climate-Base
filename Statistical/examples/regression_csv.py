"""Run configurable field regression on annual CSV columns.

Authors: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from climate_statistics import benjamini_hochberg, linear_regression


def _path(value, parent):
    if not isinstance(value, str) or not value or "<" in value or ">" in value:
        raise ValueError("Replace all input/output path placeholders")
    path = Path(value).expanduser()
    return path if path.is_absolute() else parent / path


def _plain(value):
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def run(config_path):
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    required = {"input_csv", "year_column", "predictor_column", "response_columns", "output_json"}
    allowed = required | {"control_columns", "include_trend", "inference", "min_samples", "fdr_alpha"}
    if not isinstance(config, dict) or required - config.keys() or config.keys() - allowed:
        raise ValueError("Configuration has missing or unknown keys")
    for key in ("year_column", "predictor_column"):
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(key + " must be a nonempty column name")
    responses = config["response_columns"]
    controls = config.get("control_columns", [])
    if (not isinstance(responses, list) or not responses or not isinstance(controls, list)
            or any(not isinstance(name, str) or not name for name in responses + controls)):
        raise ValueError("response_columns and control_columns must contain column names")
    names = [config["year_column"], config["predictor_column"], *controls, *responses]
    if len(set(names)) != len(names):
        raise ValueError("Year, predictor, control and response names must be distinct")
    include_trend = config.get("include_trend", True)
    if not isinstance(include_trend, bool):
        raise ValueError("include_trend must be true or false")
    output = _path(config["output_json"], config_path.parent)
    if output.exists():
        raise FileExistsError("Output already exists; choose a new filename")
    columns = {name: [] for name in names}
    with _path(config["input_csv"], config_path.parent).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV must have unique column headers")
        if set(names) - set(reader.fieldnames):
            raise ValueError("Configured columns were not found in the CSV")
        for number, row in enumerate(reader, start=2):
            if None in row or any(row[name] is None for name in names):
                raise ValueError(f"CSV row {number} has the wrong number of cells")
            for name in names:
                token = row[name].strip()
                try:
                    value = np.nan if token.lower() in {"", "na", "nan", "null"} else float(token)
                except ValueError as error:
                    raise ValueError(f"Non-numeric value at row {number}, column {name}") from error
                columns[name].append(value)
    arrays = {name: np.asarray(values, dtype=float) for name, values in columns.items()}
    field = np.column_stack([arrays[name] for name in responses])
    control = np.column_stack([arrays[name] for name in controls]) if controls else None
    result = linear_regression(arrays[config["predictor_column"]], field,
                               arrays[config["year_column"]], controls=control,
                               include_trend=include_trend, inference=config.get("inference", "ols"),
                               min_samples=config.get("min_samples"))
    fdr = benjamini_hochberg(result["p"], alpha=config.get("fdr_alpha", .05))
    payload = {"response_columns": responses, "predictor_column": config["predictor_column"],
               "control_columns": controls, "include_trend": include_trend,
               "coefficient_units": "response units per supplied predictor unit",
               "regression": result, "fdr_across_response_columns": fdr}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(_plain(payload), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    arguments = parser.parse_args()
    payload = run(arguments.config)
    print(json.dumps({"state": "written", "responses": payload["response_columns"]}))


if __name__ == "__main__":
    main()
