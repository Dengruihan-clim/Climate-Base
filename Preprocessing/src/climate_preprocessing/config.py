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


def load_prepare_config(path):
    """Read schema version 1 without opening any meteorological input files."""
    path = Path(path).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict):
        raise ValueError("Preparation config must be a JSON object")
    allowed = {
        "schema_version", "input", "output", "chunks", "coordinates", "time",
        "longitude_convention", "latitude_order", "region", "units", "metadata_policy",
    }
    if set(config) - allowed:
        raise ValueError(f"Unknown preparation configuration keys: {sorted(set(config) - allowed)}")
    if type(config.get("schema_version")) is not int or config["schema_version"] != 1:
        raise ValueError("Preparation config requires schema_version=1")
    policy = config.get("metadata_policy", "strict")
    if not isinstance(policy, str) or policy not in ("strict", "preserve_sources"):
        raise ValueError("metadata_policy must be 'strict' or 'preserve_sources'")
    sources = config.get("input")
    if isinstance(sources, str):
        sources = [sources]
    if not isinstance(sources, list) or not sources or any(not isinstance(s, str) or not s for s in sources):
        raise ValueError("input must be a filename or a nonempty list of filenames")

    def resolve(value):
        candidate = Path(value).expanduser()
        return (path.parent / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()

    inputs = tuple(resolve(source) for source in sources)
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
    options = {key: value for key, value in config.items() if key not in {"schema_version", "input", "output"}}
    return PrepareConfig(inputs, output, options)
