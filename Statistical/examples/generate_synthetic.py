"""Create synthetic annual CSV inputs and a runnable regression config.

No observed data are included. Authors: RH_Deng; Codex (OpenAI).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(42)
    years = np.arange(1981, 2021)
    predictor = rng.normal(size=len(years))
    control = rng.normal(size=len(years))
    a = 2.5 * predictor + .8 * control + .03 * (years - years.mean()) + rng.normal(0, .2, len(years))
    b = -.7 * predictor + .2 * control + rng.normal(0, .3, len(years))
    with (destination / "annual_series.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["year", "predictor", "control", "response_a", "response_b"])
        writer.writerows(zip(years, predictor, control, a, b))
    config = {"input_csv": "annual_series.csv", "year_column": "year", "predictor_column": "predictor",
              "response_columns": ["response_a", "response_b"], "control_columns": ["control"],
              "include_trend": True, "inference": "ols", "min_samples": 10,
              "output_json": "regression.json"}
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Synthetic inputs and config.json created.")


if __name__ == "__main__":
    main()
