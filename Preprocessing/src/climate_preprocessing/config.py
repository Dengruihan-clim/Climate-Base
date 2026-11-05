"""Versioned configuration for the lazy NetCDF preparation workflow.

References
----------
Meteostat v2.1.5 ``api/config.py`` (configuration separation only):
https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/api/config.py

Climate-Base improvements
-------------------------
Independent implementation: explicit per-run configuration, strict unknown-key
rejection, paths relative to the configuration, and no global environment state.
The legacy monthly configuration remains separate and unchanged.
"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class PrepareConfig:
    """Resolved local paths and declared options; not a new array container."""

    inputs: tuple[Path, ...]
    output: Path | None
    options: dict
    output_format: str = "netcdf"


def load_prepare_config(path):
    """Read schema version 1 without opening any meteorological input files."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Preparation config must be a JSON object")
    allowed = {
        "schema_version", "input", "output", "chunks", "coordinates", "time",
        "longitude_convention", "latitude_order", "region", "units", "metadata_policy",
        "pressure", "expected_time", "time_range",
        "input_format", "input_options", "output_format",
        "regrid", "vertical", "adapter",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown preparation configuration keys: {sorted(set(config) - allowed)}")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Preparation config requires schema_version=1")
    input_format = config.get("input_format", "netcdf")
    if not isinstance(input_format, str) or input_format not in {"netcdf", "grib", "zarr"}:
        raise ValueError("input_format must be netcdf, grib or zarr")
    output_format = config.get("output_format", "netcdf")
    if not isinstance(output_format, str) or output_format not in {"netcdf", "zarr"}:
        raise ValueError("output_format must be netcdf or zarr")
    options = config.get("input_options")
    supported = {"netcdf": set(), "grib": {"filter_by_keys"}, "zarr": {"require_complete"}}[input_format]
    if options is not None and (not isinstance(options, dict) or set(options) - supported):
        raise ValueError(f"Invalid input_options for {input_format}; supported keys are {sorted(supported)}")
    policy = config.get("metadata_policy", "strict")
    if not isinstance(policy, str) or policy not in ("strict", "preserve_sources"):
        raise ValueError("metadata_policy must be 'strict' or 'preserve_sources'")
    for key, keys in (("pressure", {"pressure", "target_units", "order"}),
                      ("time_range", {"start", "end"})):
        value = config.get(key)
        if value is not None and (not isinstance(value, dict) or set(value) - keys):
            raise ValueError(f"Invalid {key}; supported keys are {sorted(keys)}")
    expected = config.get("expected_time")
    if expected is not None and (
        not isinstance(expected, list) or not expected
        or any(not isinstance(value, str) or not value for value in expected)
    ):
        raise ValueError("expected_time must be a nonempty list of complete ISO date strings")
    sources = config.get("input")
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources):
        raise ValueError("input must be a filename or a nonempty list of filenames")

    def resolve(value):
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    inputs = tuple(resolve(source) for source in sources)
    if config.get("regrid") is not None:
        from .regrid.api import validate_regrid_options

        validate_regrid_options(config["regrid"])
        regrid = dict(config["regrid"])
        weights_dir = regrid.get("weights_dir")
        if weights_dir is not None:
            if not isinstance(weights_dir, str) or not weights_dir.strip():
                raise ValueError("weights_dir must be a nonempty local directory")
            regrid["weights_dir"] = resolve(weights_dir)
        config["regrid"] = regrid
    if config.get("vertical") is not None:
        from .vertical.interpolate import validate_vertical_options

        validate_vertical_options(config["vertical"])
    if config.get("adapter") is not None:
        from .adapters.api import validate_adapter_options

        validate_adapter_options(config["adapter"])
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input filenames are not allowed")
    output = config.get("output")
    if output is not None and (not isinstance(output, str) or not output):
        raise ValueError("output must be a nonempty filename")
    # Do not resolve the final output component: doing so follows a dangling
    # symlink and would defeat the writer's explicit non-overwrite guarantee.
    if output is not None:
        output = Path(output).expanduser()
        if not output.is_absolute():
            output = path.parent / output
        output = output.absolute()
    options = {key: value for key, value in config.items()
               if key not in {"schema_version", "input", "output", "output_format"}}
    return PrepareConfig(inputs, output, options, output_format)
