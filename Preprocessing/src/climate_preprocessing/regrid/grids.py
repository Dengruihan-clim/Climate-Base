"""Explicit rectilinear geometry for the optional xESMF adapter.

References
----------
CF 1.12 sections 4.1, 4.2 and 7.1 (angular axes and cell boundaries):
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
xESMF grid/boundary and mask conventions:
https://xesmf.readthedocs.io/en/stable/user_api.html

Climate-Base improvements
-------------------------
Independently implemented geometry validation, separate from field evaluation.
Bounds are explicit, contiguous, and never inferred. Masks are static, eager,
boolean, and exactly labelled (True means valid). Detached coordinate/mask
copies and canonical hashes identify actual grids rather than their shapes.
No meteorological field is read; only the declared geometry is inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import xarray as xr

from ..coordinates import identify_coordinate
from ..exceptions import CoordinateError
from ..metadata import _attribute
from ..validation import require_dataset


def _numbers(values, label):
    """Copy declared geometry (including lazy CF bounds), never field arrays."""
    if isinstance(values, xr.DataArray):
        values = values.data
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise CoordinateError(f"{label} must contain real numeric values")
    output = np.array(array, dtype=np.float64, copy=True, order="C")
    if not np.isfinite(output).all():
        raise CoordinateError(f"{label} must contain finite values")
    # All angles fit comfortably in float64; reject exotic integer inputs that
    # would acquire a false coordinate identity after conversion.
    if array.dtype.kind in "iu" and np.any(np.abs(output) > 2**53):
        raise CoordinateError(f"{label} integer values exceed exact float64 precision")
    output[output == 0] = 0.0  # -0.0 and +0.0 describe the same geometry.
    return output


def _centers(values, role):
    output = _numbers(values, role)
    if output.ndim != 1 or output.size < 2:
        raise CoordinateError(f"{role} requires at least two one-dimensional centers")
    if np.any(np.diff(output) <= 0):
        raise CoordinateError(f"{role} centers must be strictly ascending; normalize explicitly first")
    if role == "latitude":
        if np.any(np.abs(output) > 90):
            raise CoordinateError("latitude centers must lie in [-90, 90]")
    else:
        if output[-1] - output[0] >= 360:
            raise CoordinateError("longitude must span less than 360 degrees without duplicate cyclic endpoints")
        cyclic = np.sort(np.mod(output, 360.0))
        if np.any(np.diff(cyclic) <= 1e-10):
            raise CoordinateError("longitude contains duplicate cyclic centers")
        if np.any(np.diff(output) >= 180):
            raise CoordinateError("longitude gaps of 180 degrees or more are ambiguous")
    return output


def _bounds(values, centers, role):
    output = _numbers(values, f"{role} bounds")
    if output.ndim == 1 and output.shape == (centers.size + 1,):
        output = np.column_stack((output[:-1], output[1:]))
    if output.shape != (centers.size, 2):
        raise CoordinateError(f"{role} bounds must be n+1 edges or an (n, 2) array")
    if np.any(output[:, 1] <= output[:, 0]):
        raise CoordinateError(f"{role} bounds must have increasing, positive-width cells")
    if np.any(centers <= output[:, 0]) or np.any(centers >= output[:, 1]):
        raise CoordinateError(f"{role} centers must be strictly inside their bounds")
    # Approximate equality would silently alter cell areas in the generated
    # edge array. The caller must explicitly reconcile imperfect boundaries.
    if not np.array_equal(output[:-1, 1], output[1:, 0]):
        raise CoordinateError(f"{role} bounds must be exactly contiguous without gaps or overlaps")
    if role == "latitude" and np.any(np.abs(output) > 90):
        raise CoordinateError("latitude bounds must lie in [-90, 90]")
    if role == "longitude":
        if np.any(output[:, 1] - output[:, 0] >= 180):
            raise CoordinateError("longitude cell width must be less than 180 degrees")
        if output[-1, 1] - output[0, 0] > 360:
            raise CoordinateError("longitude bounds cannot cover more than 360 degrees")
    return output


def rectilinear_grid(latitude, longitude, *, latitude_bounds=None, longitude_bounds=None):
    """Create a CF grid from explicit degree centers and optional cell edges.

    ``latitude`` and ``longitude`` are ascending one-dimensional degree arrays.
    Longitude may be unwrapped across the dateline, e.g. 170, 180, 190. Bounds
    accept n+1 edges or n-by-2 lower/upper pairs; no bounds are inferred. This
    constructor does not generate a spatial resolution or read field values.
    """
    lat = _centers(latitude, "latitude")
    lon = _centers(longitude, "longitude")
    dataset = xr.Dataset(coords={
        "latitude": ("latitude", lat, {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"}),
        "longitude": ("longitude", lon, {"standard_name": "longitude", "units": "degrees_east", "axis": "X"}),
    })
    for role, values, centers in (("latitude", latitude_bounds, lat), ("longitude", longitude_bounds, lon)):
        if values is not None:
            name = f"{role}_bounds"
            dataset[name] = ((role, "bounds"), _bounds(values, centers, role), {"units": dataset[role].attrs["units"]})
            dataset[role].attrs["bounds"] = name
    return dataset


def _digest(array):
    if array is None:
        return None
    canonical = np.asarray(array, dtype="u1" if array.dtype.kind == "b" else "<f8", order="C")
    return {"shape": list(canonical.shape), "dtype": canonical.dtype.str,
            "sha256": hashlib.sha256(canonical.tobytes(order="C")).hexdigest()}


@dataclass(frozen=True)
class GridSpec:
    """Detached validated geometry; numeric arrays are read-only snapshots."""

    latitude: np.ndarray
    longitude: np.ndarray
    latitude_bounds: np.ndarray | None
    longitude_bounds: np.ndarray | None
    mask: np.ndarray | None

    @property
    def fingerprint_payload(self):
        return {name: _digest(getattr(self, name)) for name in
                ("latitude", "longitude", "latitude_bounds", "longitude_bounds", "mask")}

    @property
    def grid(self):
        """Return an independent xESMF-native geometry Dataset."""
        output = xr.Dataset(coords={"lat": ("lat", self.latitude.copy()), "lon": ("lon", self.longitude.copy())})
        for short, values in (("lat", self.latitude_bounds), ("lon", self.longitude_bounds)):
            if values is not None:
                edges = np.concatenate((values[:, 0], values[-1:, 1]))
                output = output.assign_coords({f"{short}_b": (f"{short}_b", edges)})
        if self.mask is not None:
            output["mask"] = (("lat", "lon"), self.mask.astype(np.int32))
        return output


def validate_grid(dataset, *, mask=None, require_bounds=False, periodic=False):
    """Inspect canonical CF geometry without accessing meteorological fields.

    Inputs require ``latitude(latitude)`` and ``longitude(longitude)`` with
    explicit angular units. Static masks require eager boolean values and exact
    dimension coordinates. Lazy masks must be explicitly computed by callers;
    deriving a mask from a field is therefore never an implicit field scan.
    Coordinates and their explicitly referenced bounds may be read eagerly to
    build weights, including CF bounds returned lazily by an input backend.
    ``periodic`` requires explicit longitude bounds spanning exactly 360 degrees
    (within 1e-10 degrees for the total span only). Conservative methods request
    ``require_bounds=True``. Two-dimensional and projected grids are rejected.
    """
    require_dataset(dataset)
    if type(require_bounds) is not bool or type(periodic) is not bool:
        raise TypeError("require_bounds and periodic must be booleans")
    arrays = {}
    for role in ("latitude", "longitude"):
        identify_coordinate(dataset, role, explicit=role)
        variable = dataset[role]
        if role not in dataset.coords or variable.dims != (role,):
            raise CoordinateError(f"{role} must be a canonical one-dimensional dimension coordinate")
        if variable.attrs.get("standard_name", role) != role:
            raise CoordinateError(f"{role} has conflicting standard_name metadata")
        if str(variable.attrs.get("axis", "Y" if role == "latitude" else "X")).upper() != (
            "Y" if role == "latitude" else "X"
        ):
            raise CoordinateError(f"{role} has conflicting axis metadata")
        arrays[role] = _centers(variable, role)
        bound_name = _attribute(variable, "bounds")
        arrays[f"{role}_bounds"] = None
        if bound_name is None:
            if require_bounds or (periodic and role == "longitude"):
                raise CoordinateError(f"Explicit {role} bounds are required for this regridding operation")
            continue
        if bound_name not in dataset.variables:
            raise CoordinateError(f"Missing {role} bounds variable {bound_name!r}")
        bound = dataset[bound_name]
        other = [dimension for dimension in bound.dims if dimension != role]
        if bound.ndim != 2 or role not in bound.dims or len(other) != 1 or bound.sizes[other[0]] != 2:
            raise CoordinateError(f"{role} bounds require the axis dimension and a size-2 vertex dimension")
        if "units" in bound.attrs and bound.attrs["units"] != variable.attrs["units"]:
            raise CoordinateError(f"{role} bounds units must match coordinate units")
        arrays[f"{role}_bounds"] = _bounds(bound.transpose(role, other[0]), arrays[role], role)
    if periodic:
        edges = arrays["longitude_bounds"]
        if not np.isclose(edges[-1, 1] - edges[0, 0], 360.0, rtol=0, atol=1e-10):
            raise CoordinateError("periodic=True requires longitude bounds covering the full 360-degree domain")
    arrays["mask"] = None
    if mask is not None:
        if not isinstance(mask, xr.DataArray) or mask.dtype.kind != "b":
            raise TypeError("A regridding mask must be a boolean xarray.DataArray (True means valid)")
        if mask.chunks is not None:
            raise CoordinateError("A static regridding mask must be eager; compute it explicitly")
        if set(mask.dims) != {"latitude", "longitude"} or mask.ndim != 2:
            raise CoordinateError("A static mask must have exactly latitude and longitude dimensions")
        for role in ("latitude", "longitude"):
            if role not in mask.coords or mask[role].dims != (role,):
                raise CoordinateError(f"Mask requires explicit {role} coordinate labels")
            if not np.array_equal(_numbers(mask[role], f"mask {role}"), arrays[role]):
                raise CoordinateError(f"Mask {role} coordinate labels must exactly match the grid")
        if set(mask.coords) - {"latitude", "longitude"}:
            raise CoordinateError("Static masks cannot carry unverified auxiliary coordinates")
        arrays["mask"] = np.array(mask.transpose("latitude", "longitude").data, dtype=bool, copy=True)
        if not arrays["mask"].any():
            raise CoordinateError("A regridding mask must retain at least one valid cell")
    for array in arrays.values():
        if array is not None:
            array.setflags(write=False)
    return GridSpec(**arrays)
