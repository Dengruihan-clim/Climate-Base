"""Physical geometry and cache identity contracts, without optional backends."""

from dask import delayed
import dask.array as da
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.regrid.grids import rectilinear_grid, validate_grid
from climate_preprocessing.regrid.weights import weight_key, weight_metadata


def grid(**kwargs):
    options = {"latitude_bounds": [-20, 0, 20], "longitude_bounds": [160, 180, 200]}
    options.update(kwargs)
    return rectilinear_grid([-10, 10], [170, 190], **options)


def test_cf_bounds_native_edges_and_dateline_geometry():
    source = grid()
    specification = validate_grid(source, require_bounds=True)
    assert source.latitude.attrs["bounds"] == "latitude_bounds"
    assert source.longitude.attrs["units"] == "degrees_east"
    np.testing.assert_array_equal(specification.grid.lon_b, [160, 180, 200])
    np.testing.assert_array_equal(specification.grid.lat_b, [-20, 0, 20])
    assert specification.grid.lon.dims == ("lon",)
    assert specification.latitude_bounds.shape == (2, 2)


def test_geometry_snapshot_does_not_share_mutable_input_arrays():
    source = grid()
    specification = validate_grid(source)
    original = specification.fingerprint_payload
    # Recent pandas/xarray index views are read-only; replace their labels via
    # public xarray APIs and mutate non-index bounds to exercise copy isolation.
    source = source.assign_coords(latitude=[-11, 10])
    source.latitude_bounds.data[0, 0] = -21
    native = specification.grid
    native = native.assign_coords(lon=[175, 190])
    assert specification.fingerprint_payload == original
    np.testing.assert_array_equal(specification.longitude, [170, 190])
    with pytest.raises(ValueError, match="read-only"):
        specification.latitude[0] = -12


def test_geometry_validation_does_not_compute_lazy_fields():
    calls = []

    @delayed
    def field():
        calls.append("field")
        return np.ones((2, 2))

    source = grid()
    source["temperature"] = (("latitude", "longitude"), da.from_delayed(field(), shape=(2, 2), dtype=float))
    source["latitude_bounds"] = source.latitude_bounds.chunk()
    source["longitude_bounds"] = source.longitude_bounds.chunk()
    validate_grid(source, require_bounds=True)
    assert calls == []


@pytest.mark.parametrize("latitude,longitude,match", [
    ([0], [0, 1], "at least two"),
    ([10, -10], [0, 1], "ascending"),
    ([0, 0], [0, 1], "ascending"),
    ([-91, 10], [0, 1], "latitude centers"),
    ([-10, 10], [0, 360], "cyclic"),
    ([-10, 10], [0, 180], "ambiguous"),
    ([-10, 10], [0, np.nan], "finite"),
    ([-10, 10], [0, 0], "ascending"),
    ([True, False], [0, 1], "real numeric"),
])
def test_bad_coordinate_physics_are_rejected(latitude, longitude, match):
    with pytest.raises(ValueError, match=match):
        rectilinear_grid(latitude, longitude)


@pytest.mark.parametrize("bounds,match", [
    ([[-20, 0], [1, 20]], "contiguous"),
    ([[-20, 1], [0, 20]], "contiguous"),
    ([[-20, -10], [0, 20]], "strictly inside"),
    ([-95, 0, 20], "latitude bounds"),
    ([[-20, 0, 1], [0, 20, 30]], "n, 2"),
    ([0, -20, 20], "positive-width"),
])
def test_invalid_bounds_rejected(bounds, match):
    with pytest.raises(ValueError, match=match):
        grid(latitude_bounds=bounds)


def test_conservative_requires_both_explicit_bounds():
    with pytest.raises(ValueError, match="longitude bounds"):
        validate_grid(grid(longitude_bounds=None), require_bounds=True)


@pytest.mark.parametrize("mutation,match", [
    (lambda source: source.latitude.attrs.update(units="m"), "degrees_north"),
    (lambda source: source.latitude.attrs.update(standard_name="grid_latitude"), "conflicting"),
    (lambda source: source.latitude.attrs.update(axis="X"), "conflicting axis"),
    (lambda source: source.latitude.attrs.update(bounds="absent"), "Missing latitude bounds"),
    (lambda source: source.latitude_bounds.attrs.update(units="radian"), "bounds units"),
])
def test_cf_evidence_is_checked(mutation, match):
    source = grid()
    mutation(source)
    with pytest.raises(ValueError, match=match):
        validate_grid(source)


def test_curvilinear_and_noncanonical_axis_are_rejected():
    source = grid().rename({"latitude": "y"})
    with pytest.raises(ValueError, match="Missing latitude"):
        validate_grid(source)
    source = grid().drop_vars(["latitude", "latitude_bounds"])
    source = source.assign_coords(latitude=(("y", "longitude"), [[-10, -10], [10, 10]],
                                           {"units": "degrees_north"}))
    with pytest.raises(ValueError, match="one-dimensional"):
        validate_grid(source)


def test_transposed_cf_bounds_and_mask_are_explicitly_labelled():
    source = grid()
    source["latitude_bounds"] = source.latitude_bounds.transpose("bounds", "latitude")
    mask = xr.DataArray([[True, False], [False, True]], dims=("longitude", "latitude"),
                        coords={"latitude": source.latitude, "longitude": source.longitude})
    specification = validate_grid(source, mask=mask)
    np.testing.assert_array_equal(specification.grid["mask"], [[1, 0], [0, 1]])
    assert specification.grid["mask"].dims == ("lat", "lon")


@pytest.mark.parametrize("kind", ["mismatched", "unlabelled", "integer", "all_false", "extra_coord", "lazy"])
def test_mask_rejection_contract(kind):
    source = grid()
    mask = xr.DataArray(np.ones((2, 2), dtype=bool), dims=("latitude", "longitude"),
                        coords={"latitude": source.latitude, "longitude": source.longitude})
    if kind == "mismatched":
        mask = mask.assign_coords(longitude=[170, 191])
    elif kind == "unlabelled":
        mask = mask.drop_vars("longitude")
    elif kind == "integer":
        mask = mask.astype(int)
    elif kind == "all_false":
        mask = ~mask
    elif kind == "extra_coord":
        mask = mask.assign_coords(time=0)
    else:
        mask = mask.chunk()
    with pytest.raises((ValueError, TypeError)):
        validate_grid(source, mask=mask)


def test_periodicity_requires_full_explicit_longitude_coverage():
    global_grid = rectilinear_grid([-10, 10], [60, 180, 300], longitude_bounds=[0, 120, 240, 360])
    validate_grid(global_grid, periodic=True)
    with pytest.raises(ValueError, match="full 360"):
        validate_grid(grid(), periodic=True)
    with pytest.raises(ValueError, match="Explicit longitude bounds"):
        validate_grid(grid(longitude_bounds=None), periodic=True)


def test_weight_identity_changes_for_geometry_mask_method_periodicity_and_backend():
    original = validate_grid(grid())
    target = validate_grid(grid())
    options = {"method": "bilinear", "periodic": False, "backend_versions": {"xesmf": "0.8.10", "esmpy": "8.7"}}
    first = weight_key(original, target, **options)
    assert len(first) == 64
    assert weight_key(validate_grid(grid()), target, **options) == first
    changed_center = grid()
    changed_center = changed_center.assign_coords(longitude=("longitude", [171, 191], changed_center.longitude.attrs))
    changed_bounds = grid(longitude_bounds=[159, 180, 201])
    mask = xr.DataArray([[True, False], [True, True]], dims=("latitude", "longitude"), coords={
        "latitude": grid().latitude, "longitude": grid().longitude})
    for source in (validate_grid(changed_center), validate_grid(changed_bounds), validate_grid(grid(), mask=mask)):
        assert weight_key(source, target, **options) != first
    for name, value in (("method", "nearest_s2d"), ("periodic", True),
                        ("backend_versions", {"xesmf": "0.8.11", "esmpy": "8.7"})):
        assert weight_key(original, target, **{**options, name: value}) != first
    metadata = weight_metadata(original, target, **options)
    assert metadata["options"]["skipna"] is False
