"""Small shared contracts for labelled, lazily evaluated data.

References
----------
Xarray public chunking API:
https://docs.xarray.dev/en/stable/generated/xarray.Dataset.chunk.html

Climate-Base improvements
-------------------------
The Dask-backed field contract is explicit; coordinate inspection is allowed,
but validation never coerces a field to NumPy or executes its task graph.
This module is an independent implementation using public xarray APIs.
"""

from collections.abc import Mapping

import xarray as xr


def require_dataset(dataset):
    """Require a Dataset; callers must explicitly name a standalone DataArray."""
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Expected an xarray.Dataset")


def ensure_lazy(dataset, chunks="auto"):
    """Return a Dataset with Dask-backed data variables without evaluating them.

    Parameters
    ----------
    dataset : xarray.Dataset
        Field data are never read by this function. Dimension coordinates may
        already be in memory, as required by xarray indexes.
    chunks : str or mapping, default 'auto'
        Passed to Dataset.chunk. ``None`` is rejected: eager field execution
        must be an explicit caller action such as ``result.compute()``.

    Returns
    -------
    xarray.Dataset
        A new container retaining the source close callback. If all variables
        are already chunked and chunks is 'auto', existing chunks are kept.
    """
    require_dataset(dataset)
    if chunks is None or not (chunks == "auto" or isinstance(chunks, Mapping)):
        raise ValueError("chunks must be 'auto' or a dimension-to-chunk mapping")
    if chunks == "auto":
        result = dataset.copy(deep=False)
        for name, variable in dataset.data_vars.items():
            if variable.chunks is None:
                # Chunk only new eager fields. Dataset.chunk('auto') would also
                # repartition existing Dask fields, discarding a user's chosen
                # memory/task-size policy when even one eager field was added.
                result[name] = variable.chunk("auto")
    else:
        result = dataset.chunk(chunks)
    result.set_close(dataset.close)
    return result
