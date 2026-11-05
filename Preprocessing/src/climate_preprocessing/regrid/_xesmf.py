"""The sole, optional xESMF/ESMPy boundary for horizontal regridding.

References
----------
xESMF frontend and sparse weight application (upstream owns the numerical
implementation), inspected at b30d1f12b085cc162af9cde359925ab51bdc1331:
https://github.com/pangeo-data/xESMF/blob/b30d1f12b085cc162af9cde359925ab51bdc1331/xesmf/frontend.py
https://xesmf.readthedocs.io/en/stable/user_api.html

Climate-Base improvements
-------------------------
An independently written adapter imports native dependencies only on demand,
never invents cell bounds or default cache paths, uses explicitly named
horizontal axes, and keeps meteorological field applications Dask-backed.
Grid geometry/weight construction is eager; field computation is deferred.
NaN source values propagate through participating weights (skipna=False),
unmapped destinations become NaN, and no extrapolation is requested. Source
units, calendars and remaining dimensions are not numerically transformed.
"""

from importlib import import_module, metadata
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr

METHODS = frozenset({"bilinear", "nearest_s2d", "nearest_d2s", "conservative", "conservative_normed"})


def _load_backend():
    try:
        xe = import_module("xesmf")
        try:
            esmpy = import_module("esmpy")
        except ImportError:
            esmpy = import_module("ESMF")
    except (ImportError, OSError) as error:
        raise ImportError(
            "Horizontal regridding requires xESMF and a working ESMPy/ESMF native "
            "installation. Install the regrid extra and ESMPy in a separate supported "
            "environment (typically conda-forge on Linux); pip xesmf alone is insufficient."
        ) from error
    return xe, esmpy


def backend_versions():
    """Import and identify the real backend used to generate cache weights.

    ESMPy's runtime version is used rather than guessing a native ESMF version
    from package availability. Dependency versions relevant to sparse weight
    application are included, so reuse remains deliberately conservative.
    """
    xe, esmpy = _load_backend()
    versions = {"xesmf": str(xe.__version__), "esmpy": str(esmpy.__version__)}
    for name in ("numpy", "xarray", "dask", "sparse", "scipy", "numba", "cf-xarray", "shapely"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            # SciPy, for example, is not required by xESMF 0.9.2's sparse
            # application path. Successful native imports above establish
            # availability; missing optional distribution metadata must not
            # masquerade as an unavailable native backend. Record the absence
            # explicitly so a later environment change cannot reuse its key.
            versions[name] = "distribution-not-installed"
    # Some ESMPy builds expose the linked ESMF version separately; otherwise the
    # wrapper's version remains the recorded evidence, not a guessed value.
    native = getattr(esmpy, "__ESMF_VERSION__", None)
    if native is not None:
        versions["esmf"] = str(native)
    return versions


def build_regridder(source_grid, target_grid, *, method, periodic=False, filename=None, reuse_weights=False):
    """Build or load real weights for already validated native ``lat/lon`` grids.

    The caller owns geometry, mask and cache validation. With ``filename=None``
    no weight file is written. Passing a new filename deliberately asks xESMF
    to write there; production caching should instead call ``save_weights`` on
    its private staging filename. Conservative methods have their own spherical
    cell closure and reject the separate bilinear periodic flag.
    """
    if method not in METHODS:
        raise ValueError(f"Unsupported horizontal regridding method: {method!r}")
    if type(periodic) is not bool or type(reuse_weights) is not bool:
        raise TypeError("periodic and reuse_weights must be boolean")
    if periodic and method.startswith("conservative"):
        raise ValueError("conservative methods require periodic=False and explicit cell bounds")
    if method.startswith("conservative"):
        for grid in (source_grid, target_grid):
            if "lat_b" not in grid or "lon_b" not in grid:
                raise ValueError("Conservative regridding requires explicit source and target cell bounds")
    if reuse_weights and filename is None:
        raise ValueError("Reusing weights requires an explicitly validated cache filename")
    xe, _ = _load_backend()
    return xe.Regridder(
        source_grid, target_grid, method,
        periodic=periodic, filename=None if filename is None else str(filename),
        reuse_weights=reuse_weights, unmapped_to_nan=True, ignore_degenerate=False,
        input_dims=("lat", "lon"), output_dims=("lat", "lon"),
    )


def save_weights(regridder, path):
    """Write weights to a caller-owned staging path, refusing an existing path.

    NaN entries deliberately mark unmapped destination rows; they are valid
    xESMF weights and survive the NetCDF roundtrip. Cache publication/checksums
    are managed separately, rather than relying on xESMF's shape-based name.
    """
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Weight staging path already exists: {path.name}")
    regridder.to_netcdf(str(path))
    return path


def apply_regridder(regridder, field, *, output_latitude, output_longitude):
    """Return a lazy field with original dimension order on the target grid.

    Only the two horizontal sizes change. Nonhorizontal coordinate metadata
    and chunks are preserved; source grid-dependent auxiliary coordinates are
    excluded. The public API handles CF relationship and attribute rewriting.
    """
    if not isinstance(field, xr.DataArray):
        raise TypeError("A regridded field must be an xarray.DataArray")
    horizontal = ("latitude", "longitude")
    if not set(horizontal).issubset(field.dims):
        raise ValueError("Regridded fields require latitude and longitude dimensions")
    if field.dtype.kind not in "iuf":
        raise ValueError("Regridded fields must contain real numeric values")
    original_dims = field.dims
    extra_dims = tuple(dim for dim in field.dims if dim not in horizontal)
    source = field.transpose(*extra_dims, *horizontal)
    if not isinstance(source.data, da.Array):
        source = source.chunk("auto")
    # xESMF's ndarray path accepts Dask arrays and avoids attaching native grid
    # coordinates with potentially colliding names to the source DataArray.
    target_lat = np.asarray(output_latitude, dtype=np.float64)
    target_lon = np.asarray(output_longitude, dtype=np.float64)
    chunks = tuple(min(size, max(source.chunks[-2 + index]))
                   for index, size in enumerate((target_lat.size, target_lon.size)))
    transformed = regridder(source.data, skipna=False, output_chunks=chunks)
    if not isinstance(transformed, da.Array):
        raise RuntimeError("The xESMF backend did not preserve the lazy Dask field contract")
    coords = {
        name: coordinate for name, coordinate in source.coords.items()
        if not set(horizontal).intersection(coordinate.dims)
    }
    coords.update(latitude=("latitude", target_lat), longitude=("longitude", target_lon))
    result = xr.DataArray(transformed, dims=source.dims, coords=coords, name=field.name, attrs=dict(field.attrs))
    result.encoding = {}
    return result.transpose(*original_dims)
