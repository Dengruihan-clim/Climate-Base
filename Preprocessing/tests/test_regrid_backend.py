"""Real ESMF integration: analytical fields, spherical areas and lazy graphs.

These cases skip when the native optional backend is unavailable; no mock
weights are substituted for evidence about the numerical backend. Great-circle
polygon areas are computed independently of xESMF, using spherical triangles.
"""

import json

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing import rectilinear_grid, regrid_dataset, validate_metadata, write_dataset, open_dataset
from climate_preprocessing.regrid import _xesmf

pytestmark = pytest.mark.regrid


@pytest.fixture(scope="module", autouse=True)
def real_backend():
    try:
        return _xesmf.backend_versions()
    except ImportError as error:
        pytest.skip(f"Real optional xESMF/ESMPy backend unavailable: {error}")


def field(grid, values, *, name="temperature", dims=("latitude", "longitude")):
    result = grid.copy(deep=True)
    values = np.asarray(values, dtype=float)
    result[name] = xr.DataArray(da.from_array(values, chunks=tuple(min(2, n) for n in values.shape)),
                                dims=dims, attrs={"units": "K", "long_name": "synthetic scalar"})
    return result


def unit_vector(latitude, longitude):
    lat, lon = np.deg2rad([latitude, longitude])
    return np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def triangle_area(a, b, c):
    # Solid angle of a spherical triangle; independent of ESMF weight/area code.
    determinant = abs(float(np.dot(a, np.cross(b, c))))
    denominator = 1 + float(np.dot(a, b) + np.dot(b, c) + np.dot(c, a))
    return 2 * np.arctan2(determinant, denominator)


def polygon_areas(latitude_edges, longitude_edges):
    result = np.empty((len(latitude_edges) - 1, len(longitude_edges) - 1))
    for j, (south, north) in enumerate(zip(latitude_edges[:-1], latitude_edges[1:])):
        for i, (west, east) in enumerate(zip(longitude_edges[:-1], longitude_edges[1:])):
            a, b, c, d = [unit_vector(lat, lon) for lat, lon in
                          ((south, west), (south, east), (north, east), (north, west))]
            result[j, i] = triangle_area(a, b, c) + triangle_area(a, c, d)
    return result


def conservative_grids():
    # The source subdivides only the meridional edges, which are great circles.
    # Source and target therefore have exactly the same spherical outer domain.
    source = rectilinear_grid([0.5, 1.5, 2.5, 3.5], [0.5, 1.5],
                              latitude_bounds=[0, 1, 2, 3, 4], longitude_bounds=[0, 1, 2])
    target = rectilinear_grid([1, 3], [0.5, 1.5],
                              latitude_bounds=[0, 2, 4], longitude_bounds=[0, 1, 2])
    return source, target


@pytest.mark.parametrize("method", ["bilinear", "nearest_s2d", "conservative", "conservative_normed"])
def test_constant_field_for_supported_preserving_methods(method):
    source = rectilinear_grid([0, 1, 2], [0, 1, 2],
                              latitude_bounds=[-0.5, 0.5, 1.5, 2.5],
                              longitude_bounds=[-0.5, 0.5, 1.5, 2.5])
    target = rectilinear_grid([0.5, 1.5], [0.5, 1.5],
                              latitude_bounds=[0, 1, 2], longitude_bounds=[0, 1, 2])
    result = regrid_dataset(field(source, np.full((3, 3), 7.25)), target,
                            variables=["temperature"], method=method)
    np.testing.assert_allclose(result.temperature.compute(), 7.25, atol=3e-6)
    assert isinstance(result.temperature.data, da.Array)
    assert result.temperature.attrs["units"] == "K"


def test_bilinear_small_angle_affine_field():
    source = rectilinear_grid([0, 0.02, 0.04], [0, 0.02, 0.04])
    target = rectilinear_grid([0.005, 0.015, 0.025, 0.035], [0.005, 0.015, 0.025, 0.035])
    values = 5 + 2 * source.latitude.to_numpy()[:, None] + 3 * source.longitude.to_numpy()[None, :]
    expected = 5 + 2 * target.latitude.to_numpy()[:, None] + 3 * target.longitude.to_numpy()[None, :]
    result = regrid_dataset(field(source, values), target, variables="temperature", method="bilinear")
    # ESMF interpolates on a sphere, so a lat/lon affine field is only locally
    # linear. The stated tolerance bounds that small-angle geometric difference.
    np.testing.assert_allclose(result.temperature.compute(), expected, rtol=0, atol=2e-8)


@pytest.mark.parametrize("method, expected", [
    ("nearest_s2d", [[1, 3], [4, 6]]),
    ("nearest_d2s", [[3, 3], [9, 6]]),
])
def test_nearest_directions_have_different_meanings(method, expected):
    source = rectilinear_grid([0, 2], [0, 2, 4])
    target = rectilinear_grid([0.2, 2.2], [0.2, 4.2])
    result = regrid_dataset(field(source, [[1, 2, 3], [4, 5, 6]]), target,
                            variables="temperature", method=method)
    np.testing.assert_allclose(result.temperature.compute(), expected)


def test_conservative_matches_independent_spherical_area_integral():
    source, target = conservative_grids()
    values = np.array([[2, 3], [5, 7], [11, 13], [17, 19]], dtype=float)
    result = regrid_dataset(field(source, values), target, variables="temperature", method="conservative")
    source_area = polygon_areas([0, 1, 2, 3, 4], [0, 1, 2])
    target_area = polygon_areas([0, 2, 4], [0, 1, 2])
    source_integral = float(np.sum(values * source_area))
    target_integral = float(np.sum(result.temperature.compute().to_numpy() * target_area))
    assert target_integral == pytest.approx(source_integral, rel=3e-11, abs=1e-14)


@pytest.mark.parametrize("method", ["conservative", "conservative_normed"])
def test_masked_partial_coverage_normalization(method):
    source, target = conservative_grids()
    mask = xr.DataArray(np.array([[False, False], [True, True], [True, True], [True, True]]),
                        dims=("latitude", "longitude"), coords={
                            "latitude": source.latitude, "longitude": source.longitude})
    result = regrid_dataset(field(source, np.full((4, 2), 2.0)), target,
                            variables="temperature", method=method, source_mask=mask)
    if method == "conservative":
        area = polygon_areas([0, 1, 2, 3, 4], [0, 1, 2])
        target_area = polygon_areas([0, 2, 4], [0, 1, 2])
        expected = np.vstack([2 * area[1] / target_area[0], [2, 2]])
    else:
        expected = np.full((2, 2), 2.0)
    np.testing.assert_allclose(result.temperature.compute(), expected, rtol=3e-10, atol=1e-12)


def test_periodic_bilinear_maps_across_longitude_seam():
    source = rectilinear_grid([-1, 1], [45, 135, 225, 315],
                              latitude_bounds=[-2, 0, 2], longitude_bounds=[0, 90, 180, 270, 360])
    target = rectilinear_grid([-0.5, 0.5], [-1, 1])
    result = regrid_dataset(field(source, np.ones((2, 4))), target,
                            variables="temperature", method="bilinear", periodic=True)
    np.testing.assert_allclose(result.temperature.compute(), 1.0)


def test_polar_conservative_bounds_retain_constant():
    grid = rectilinear_grid([75, 85], [45, 135, 225, 315],
                            latitude_bounds=[70, 80, 90], longitude_bounds=[0, 90, 180, 270, 360])
    result = regrid_dataset(field(grid, np.ones((2, 4))), grid,
                            variables="temperature", method="conservative")
    np.testing.assert_allclose(result.temperature.compute(), 1.0, rtol=1e-12)


def test_unmapped_destination_is_nan_and_valid_zero_survives():
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.5, 1.5], [0.5, 3.5])
    result = regrid_dataset(field(source, np.zeros((3, 3))), target,
                            variables="temperature", method="bilinear")
    actual = result.temperature.compute().to_numpy()
    np.testing.assert_array_equal(actual[:, 0], [0, 0])
    assert np.isnan(actual[:, 1]).all()


def test_nan_sources_are_not_dynamically_renormalized():
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.25, 1.75], [0.25, 1.75])
    values = np.ones((3, 3))
    values[0, 0] = np.nan
    result = regrid_dataset(field(source, values), target, variables="temperature", method="bilinear")
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[0, 0])
    np.testing.assert_allclose(actual[[0, 1, 1], [1, 0, 1]], 1.0)


def test_explicit_target_mask_produces_nan():
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.5, 1.5], [0.5, 1.5])
    mask = xr.DataArray([[True, False], [True, True]], dims=("latitude", "longitude"),
                        coords={"latitude": target.latitude, "longitude": target.longitude})
    result = regrid_dataset(field(source, np.ones((3, 3))), target,
                            variables="temperature", method="bilinear", target_mask=mask)
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[0, 1])
    np.testing.assert_allclose(actual[[0, 1, 1], [0, 0, 1]], 1.0)


def test_nearest_d2s_respects_both_masks_without_renormalizing_sums():
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.25, 1.75], [0.25, 1.75])
    source_valid = np.ones((3, 3), dtype=bool)
    source_valid[1, 1] = False
    source_mask = xr.DataArray(source_valid, dims=("latitude", "longitude"),
                               coords={"latitude": source.latitude, "longitude": source.longitude})
    target_valid = np.array([[False, True], [True, True]])
    target_mask = xr.DataArray(target_valid, dims=("latitude", "longitude"),
                               coords={"latitude": target.latitude, "longitude": target.longitude})
    result = regrid_dataset(field(source, np.ones((3, 3))), target,
                            variables="temperature", method="nearest_d2s",
                            source_mask=source_mask, target_mask=target_mask)
    actual = result.temperature.compute().to_numpy()
    assert np.isnan(actual[0, 0])
    assert np.isfinite(actual[target_valid]).all()
    # Each valid source contributes once to a valid destination; averaging the
    # contributors would instead produce three unit-valued destinations.
    assert actual[target_valid].sum() == pytest.approx(float(source_valid.sum()))


def test_field_construction_executes_zero_tasks_and_preserves_dimensions():
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.5, 1.5], [0.5, 1.5])
    source = source.assign_coords(time=[0, 1])
    data = da.from_delayed(delayed(lambda: np.full((3, 2, 3), 4.0))(), shape=(3, 2, 3), dtype=float)
    source["temperature"] = xr.DataArray(data.rechunk((2, 1, 2)), dims=("longitude", "time", "latitude"),
                                         attrs={"units": "K"})
    before = source.copy(deep=False)
    tasks = []
    with Callback(pretask=lambda key, graph, state: tasks.append(key)):
        result = regrid_dataset(source, target, variables="temperature", method="bilinear")
    assert tasks == []
    assert result.temperature.dims == source.temperature.dims
    assert result.temperature.chunks[1] == (1, 1)
    assert result.temperature.chunks[0] == (2,)
    assert source.identical(before)
    np.testing.assert_allclose(result.temperature.compute(), 4.0)


def test_weight_cache_reuse_preserves_nan_sentinels(tmp_path):
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.5, 1.5], [0.5, 3.5])
    values = field(source, np.ones((3, 3)))
    cache = tmp_path / "weights"
    first = regrid_dataset(values, target, variables="temperature", method="bilinear", weights_dir=cache)
    second = regrid_dataset(values, target, variables="temperature", method="bilinear", weights_dir=cache)
    xr.testing.assert_equal(first.temperature.compute(), second.temperature.compute())
    assert np.isnan(second.temperature.compute().to_numpy()[:, 1]).all()
    event = json.loads(second.attrs["climate_preprocessing_provenance"])["operations"][-1]
    assert event["parameters"]["weights_reused"] is True
    assert str(tmp_path) not in json.dumps(event)
    assert len(list(cache.glob("*.nc"))) == 1


def test_regridded_field_cf_cleanup_and_netcdf_roundtrip(tmp_path):
    source = rectilinear_grid([0, 1, 2], [0, 1, 2])
    target = rectilinear_grid([0.5, 1.5], [0.5, 1.5],
                              latitude_bounds=[0, 1, 2], longitude_bounds=[0, 1, 2])
    dataset = field(source, np.ones((3, 3)))
    dataset = dataset.expand_dims(time=np.array(["2001-01-01"], dtype="datetime64[ns]"))
    dataset["qc"] = xr.zeros_like(dataset.temperature, dtype=np.int8)
    dataset.qc.attrs = {"flag_values": [0], "flag_meanings": "good"}
    dataset["cell_area"] = xr.ones_like(dataset.temperature)
    dataset["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    dataset.temperature.attrs.update(ancillary_variables="qc", cell_measures="area: cell_area",
                                     grid_mapping="crs", cell_methods="time: mean", actual_range=[1, 1])
    dataset.temperature.encoding.update(scale_factor=0.1, add_offset=250, dtype="int16")
    result = regrid_dataset(dataset, target, variables="temperature", method="bilinear")
    assert not {"qc", "cell_area", "crs"}.intersection(result.variables)
    assert not {"ancillary_variables", "grid_mapping", "cell_measures", "actual_range"}.intersection(
        result.temperature.attrs)
    assert result.temperature.attrs["cell_methods"] == "time: mean"
    assert result.temperature.encoding == {}
    validate_metadata(result)
    path = tmp_path / "regridded.nc"
    write_dataset(result, path)
    with open_dataset(path) as reopened:
        np.testing.assert_allclose(reopened.temperature.compute(), 1.0)
        np.testing.assert_array_equal(reopened.time, result.time)
        bounds = reopened.latitude.attrs.get("bounds", reopened.latitude.encoding.get("bounds"))
        assert bounds in reopened.variables


def test_no_implicit_weight_file_without_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    source = rectilinear_grid([0, 1], [0, 1])
    target = rectilinear_grid([0.25, 0.75], [0.25, 0.75])
    result = regrid_dataset(field(source, np.ones((2, 2))), target, variables="temperature", method="bilinear")
    result.temperature.compute()
    assert list(tmp_path.iterdir()) == []
