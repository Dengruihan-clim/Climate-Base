"""Structured, privacy-aware records for explicitly requested operations.

References
----------
CF 1.12, section 2.6.2 (description of file contents):
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html

Climate-Base improvements
-------------------------
Independently implemented JSON operation records supplement, rather than
overwrite, input history. Records include runtime versions and input basenames;
new records never require private absolute paths or reading field values.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path

import xarray as xr

PROVENANCE_ATTRIBUTE = "climate_preprocessing_provenance"


def runtime_versions() -> dict[str, str]:
    """Return versions relevant to reproducibility without importing backends."""
    from ._version import __version__

    # The imported source may differ from another installed distribution's
    # metadata, for example during an editable or wheel verification run.
    versions = {"climate-preprocessing": __version__}
    for distribution in ("xarray", "dask", "numpy", "netCDF4", "cftime"):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            continue
    return versions


def input_record(path: str | Path) -> dict:
    """Describe a local input using its basename, size and modification stamp.

    This is an identity hint, not a cryptographic snapshot: no SHA256 or field
    scan is performed. Input ordering disambiguates repeated basenames.
    """
    path = Path(path)
    stat = path.stat()
    return {"name": path.name, "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def record_operation(
    dataset: xr.Dataset,
    operation: str,
    parameters: dict | None = None,
    *,
    inputs: list[dict] | None = None,
) -> xr.Dataset:
    """Return a shallow copy with one JSON provenance event and shared close.

    ``parameters`` and ``inputs`` must contain JSON-compatible, public metadata.
    Callers must not put secrets or private filesystem paths in these objects.
    The caller still owns the source lifetime; closing either result or source
    releases the shared backend. No field data are computed.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Expected an xarray.Dataset")
    if not isinstance(operation, str) or not operation.strip():
        raise ValueError("An operation name is required")
    raw = dataset.attrs.get(PROVENANCE_ATTRIBUTE)
    if raw is None:
        provenance = {"schema_version": 1, "operations": []}
    else:
        try:
            provenance = json.loads(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("Existing preprocessing provenance is not valid JSON") from error
        if (not isinstance(provenance, dict) or provenance.get("schema_version") != 1
                or not isinstance(provenance.get("operations"), list)):
            raise ValueError("Unsupported preprocessing provenance schema")
    event = {
        "operation": operation,
        "parameters": {} if parameters is None else parameters,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": runtime_versions(),
    }
    if inputs is not None:
        event["inputs"] = inputs
    provenance["operations"].append(event)
    serialized = json.dumps(provenance, ensure_ascii=False, allow_nan=False, sort_keys=True)
    result = dataset.copy(deep=False)
    result.attrs = dict(dataset.attrs)
    result.attrs[PROVENANCE_ATTRIBUTE] = serialized
    result.set_close(dataset.close)
    return result
