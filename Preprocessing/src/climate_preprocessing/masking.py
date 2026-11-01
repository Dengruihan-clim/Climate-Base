"""Explicit lazy masking; no physical-range guesses or missing-value filling.

Reference: xarray's labelled conditional operation:
https://docs.xarray.dev/en/stable/generated/xarray.DataArray.where.html

Climate-Base improvements: preserve valid zero and negative values, reject
complex fields, retain QC/source variables, and leave field evaluation to Dask.
This is independently implemented using public xarray/Dask operations.
"""

import numpy as np

from .provenance import record_operation
from .validation import ensure_lazy, require_dataset


def mask_invalid(dataset, *, variables=None):
    """Replace floating NaN/Inf by NaN lazily for explicit data variables.

    By default floating data variables are selected; integer flags and string
    identifiers are preserved. Explicit integer variables remain unchanged.
    Coordinates are never masked. Encoded fill values should first be decoded
    by the reader; this function does not guess undocumented sentinel values.
    """
    require_dataset(dataset)
    if variables is None:
        names = [name for name, value in dataset.data_vars.items() if value.dtype.kind in "fc"]
    else:
        names = [variables] if isinstance(variables, str) else list(variables)
    if len(names) != len(set(names)):
        raise ValueError("variables must not contain duplicate names")
    result = ensure_lazy(dataset)
    for name in names:
        if name not in result.data_vars:
            raise ValueError(f"Expected a data variable, got {name!r}")
        variable = result[name]
        if variable.dtype.kind == "c":
            raise ValueError(f"Complex field {name!r} is unsupported")
        if variable.dtype.kind not in "iufb":
            raise ValueError(f"Variable {name!r} is not numeric")
        if variable.dtype.kind == "f":
            masked = variable.where(np.isfinite(variable))
            masked.attrs = dict(variable.attrs)
            masked.encoding = dict(variable.encoding)
            result[name] = masked
    result.set_close(dataset.close)
    return record_operation(result, "mask_invalid", {"variables": names})
