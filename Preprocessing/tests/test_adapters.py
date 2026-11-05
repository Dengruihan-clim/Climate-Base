"""Independent product contracts: exact labels, physical evidence and CF links."""

import json

import dask.array as da
from dask import delayed
from dask.callbacks import Callback
import numpy as np
import pytest
import xarray as xr

from climate_preprocessing.adapters import adapt_product, adapter_source_time, validate_adapter_options
from climate_preprocessing import open_dataset, write_dataset, validate_metadata

ERA = "era5-monthly-pressure-levels"
MERRA = "merra2-m2imnpasm-5.12.4"
JRA = "jra3q-monthly-isobaric-analysis"
DEFAULTS = {ERA: ("time", "level", "latitude", "longitude"),
            MERRA: ("time", "lev", "lat", "lon"), JRA: ("time", "pressure_level", "lat", "lon")}


def source(product=ERA, *, members=False):
    time, pressure, latitude, longitude = DEFAULTS[product]
    result = xr.Dataset(coords={
        time: np.array(["2001-01-15", "2001-03-31"], dtype="datetime64[ns]"),
        pressure: (pressure, [850., 500.], {"units": "hPa"}),
        latitude: (latitude, [30., 10.], {"units": "degrees_north"}),
        longitude: (longitude, [20., 40.], {"units": "degrees_east"}),
    })
    dims = (time, pressure, latitude, longitude)
    values = np.arange(16, dtype=np.float32).reshape(2, 2, 2, 2)
    if members:
        result = result.assign_coords(expver=[5, 1], number=[3, 7])
        values = values[None, None] + np.array([5, 1])[:, None, None, None, None, None] * 1000
        values = values + np.array([3, 7])[None, :, None, None, None, None] * 100
        dims = ("expver", "number", *dims)
    result["V"] = (dims, da.from_array(values, chunks=tuple(1 for _ in dims)), {"units": "m s-1"})
    result["utc_date"] = (time, np.array([20010115, 20010331], dtype=np.int32))
    return result


def adapt(dataset, product=ERA, **kwargs):
    return adapt_product(dataset, product=product, variables={"northward_wind": "V"}, **kwargs)


@pytest.mark.parametrize("product", [ERA, MERRA, JRA])
def test_three_profiles_rename_only_with_no_inferred_units_order_or_statistics(product):
    original = source(product)
    result = adapt(original, product)
    assert result.northward_wind.dims == ("time", "pressure", "latitude", "longitude")
    np.testing.assert_array_equal(result.pressure, [850, 500])
    np.testing.assert_array_equal(result.latitude, [30, 10])
    np.testing.assert_array_equal(result.northward_wind.compute(), original.V.compute())
    np.testing.assert_array_equal(result.utc_date.compute(), original.utc_date)
    assert result.northward_wind.dtype == original.V.dtype
    assert result.northward_wind.attrs["units"] == "m s-1"
    assert result.pressure.attrs["units"] == "hPa"
    assert result.northward_wind.attrs["standard_name"] == "northward_wind"
    assert "cell_methods" not in result.northward_wind.attrs
    assert "bounds" not in result.time.attrs
    event = json.loads(result.attrs["climate_preprocessing_provenance"])["operations"][-1]
    assert event["parameters"]["identity"]["provider_authenticated"] is False
    assert event["parameters"]["identity"]["unverified_metadata"]


def test_exact_member_expver_labels_preserve_scalars_and_only_remove_selected_ranges():
    original = source(members=True)
    original.V.attrs.update(actual_range=[0, 9999], valid_range=[-10000, 10000], minimum_value=0)
    original.V.encoding.update(chunksizes=(1, 1, 1, 2, 2, 2), chunks=(1, 1, 1, 2, 2, 2), original_shape=original.V.shape,
                               scale_factor=0.25, dtype="int16")
    before = original.copy(deep=False)
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = adapt(original, selectors={"expver": 5, "number": 7})
    assert tasks == []
    assert result.expver.ndim == 0 and result.expver.item() == 5
    assert result.number.ndim == 0 and result.number.item() == 7
    np.testing.assert_array_equal(result.northward_wind.compute(), np.arange(16).reshape(2, 2, 2, 2) + 5700)
    assert "actual_range" not in result.northward_wind.attrs
    assert "minimum_value" not in result.northward_wind.attrs
    assert result.northward_wind.attrs["valid_range"] == [-10000, 10000]
    assert "chunksizes" not in result.northward_wind.encoding and "original_shape" not in result.northward_wind.encoding
    assert "chunks" not in result.northward_wind.encoding
    assert result.northward_wind.encoding["scale_factor"] == 0.25
    assert original.identical(before)
    assert original.V.encoding == before.V.encoding


@pytest.mark.parametrize("name", ["expver", "member", "number", "realization"])
@pytest.mark.parametrize("scalar", [False, True])
def test_even_singleton_and_scalar_known_axes_require_explicit_selection(name, scalar):
    dataset = source()
    dataset = dataset.assign_coords({name: 7}) if scalar else dataset.expand_dims({name: [7]})
    with pytest.raises(ValueError, match="selectors"):
        adapt(dataset)
    result = adapt(dataset, selectors={name: 7})
    assert result[name].ndim == 0 and result[name].item() == 7


def test_unknown_extra_mapped_field_dimension_also_requires_a_label_selector():
    dataset = source().expand_dims(extra=[11, 19])
    with pytest.raises(ValueError, match="selectors"):
        adapt(dataset)
    result = adapt(dataset, selectors={"extra": 19})
    assert result.extra.item() == 19


@pytest.mark.parametrize("value", [True, None, [5], slice(None), np.nan, np.inf, {"value": 5}])
def test_selectors_accept_only_exact_json_scalars(value):
    with pytest.raises(ValueError, match="selectors"):
        adapt(source(members=True), selectors={"expver": value, "number": 7})


def test_selector_string_ids_do_not_coerce_or_drop_leading_zeroes():
    dataset = source().expand_dims(member=["0001", "0002"])
    with pytest.raises(ValueError, match="exact"):
        adapt(dataset, selectors={"member": 1})
    result = adapt(dataset, selectors={"member": "0001"})
    assert result.member.item() == "0001"


@pytest.mark.parametrize("kind", ["duplicate", "missing_label", "unindexed", "time_varying", "unused"])
def test_ambiguous_or_unavailable_selectors_fail(kind):
    dataset = source()
    if kind == "duplicate":
        dataset = dataset.expand_dims(expver=[1, 1])
        selectors = {"expver": 1}
    elif kind == "missing_label":
        dataset = dataset.expand_dims(expver=[1, 5])
        selectors = {"expver": 2}
    elif kind == "unindexed":
        dataset = dataset.expand_dims(expver=2)
        selectors = {"expver": 1}
    elif kind == "time_varying":
        dataset = dataset.assign_coords(expver=("time", [1, 5]))
        selectors = {"expver": 1}
    else:
        selectors = {"not_present": 1}
    with pytest.raises(ValueError):
        adapt(dataset, selectors=selectors)


def test_selectors_may_not_collapse_physical_axes():
    with pytest.raises(ValueError, match="physical axes"):
        adapt(source(), selectors={"level": 500})


@pytest.mark.parametrize("role, units", [
    ("air_temperature", "degree_Celsius"), ("eastward_wind", "m/s"), ("northward_wind", "m s**-1"),
    ("specific_humidity", "g/kg"), ("geopotential", "m**2 s**-2"),
    ("geopotential_height", "gpm"), ("lagrangian_tendency_of_air_pressure", "hPa/s"),
])
def test_explicit_roles_validate_dimensions_without_converting_values_or_unit_labels(role, units):
    dataset = source(MERRA)
    dataset.V.attrs["units"] = units
    if role == "geopotential_height":
        dataset.V.attrs["standard_name"] = "geopotential_height"
    result = adapt_product(dataset, product=MERRA, variables={role: "V"})
    assert result[role].attrs["units"] == units
    np.testing.assert_array_equal(result[role].compute(), dataset.V.compute())


@pytest.mark.parametrize("units", [None, "K", "m", "unknown", "degree_east"])
def test_wind_field_name_and_product_never_supply_missing_or_wrong_units(units):
    dataset = source()
    if units is None:
        dataset.V.attrs.pop("units")
    else:
        dataset.V.attrs["units"] = units
    with pytest.raises(ValueError, match="unit|dimension"):
        adapt(dataset)


@pytest.mark.parametrize("standard", ["eastward_wind", "grid_northward_wind", "vertical_pressure_velocity"])
def test_conflicting_or_unverified_ad_hoc_standard_name_is_not_overwritten(standard):
    dataset = source(MERRA)
    dataset.V.attrs["standard_name"] = standard
    with pytest.raises(ValueError, match="standard_name"):
        adapt(dataset, MERRA)


def test_merra_layer_height_cannot_be_asserted_as_geopotential_height():
    dataset = source(MERRA)
    dataset.V.attrs.update(units="m", standard_name="mid_layer_heights")
    with pytest.raises(ValueError, match="standard_name"):
        adapt_product(dataset, product=MERRA, variables={"geopotential_height": "V"})


def test_merra_height_units_alone_cannot_supply_geopotential_height_semantics():
    dataset = source(MERRA).rename(V="H")
    dataset.H.attrs["units"] = "m"
    with pytest.raises(ValueError, match="existing explicit standard_name"):
        adapt_product(dataset, product=MERRA, variables={"geopotential_height": "H"})


def test_merra_explicit_existing_geopotential_height_semantics_preserve_values():
    dataset = source(MERRA).rename(V="H")
    dataset.H.attrs.update(units="m", standard_name="geopotential_height")
    result = adapt_product(dataset, product=MERRA, variables={"geopotential_height": "H"})
    np.testing.assert_array_equal(result.geopotential_height.compute(), dataset.H.compute())
    assert result.geopotential_height.attrs["units"] == "m"


@pytest.mark.parametrize("change", ["missing_angle_units", "wrong_angle_standard", "pressure_model", "pressure_nonmonotone",
                                    "pressure_nonpositive", "cyclic_duplicate", "auxiliary_dimension", "undecoded_time"])
def test_axis_geometry_requires_physical_evidence(change):
    dataset = source()
    if change == "missing_angle_units":
        dataset.latitude.attrs.pop("units")
    elif change == "wrong_angle_standard":
        dataset.latitude.attrs["standard_name"] = "grid_latitude"
    elif change == "pressure_model":
        dataset.level.attrs["units"] = "1"
    elif change == "pressure_nonmonotone":
        dataset = xr.concat([dataset.isel(level=[0]), dataset.isel(level=[1]), dataset.isel(level=[0])], dim="level")
        dataset = dataset.assign_coords(level=("level", [850., 500., 700.], {"units": "hPa"}))
    elif change == "pressure_nonpositive":
        dataset = dataset.assign_coords(level=("level", [0., 500.], {"units": "hPa"}))
    elif change == "cyclic_duplicate":
        dataset = dataset.assign_coords(longitude=("longitude", [0., 360.], {"units": "degrees_east"}))
    elif change == "auxiliary_dimension":
        dataset = dataset.rename_dims(latitude="y")
    else:
        dataset = dataset.assign_coords(time=[0, 1])
    with pytest.raises(ValueError):
        adapt(dataset)


def test_modern_time_name_requires_explicit_mapping_and_no_calendar_conversion():
    dataset = source().rename(time="valid_time")
    with pytest.raises(ValueError, match="time"):
        adapt(dataset)
    result = adapt(dataset, coordinates={"time": "valid_time"})
    np.testing.assert_array_equal(result.time, dataset.valid_time)
    options = {"product": ERA, "variables": {"northward_wind": "V"}, "coordinates": {"time": "valid_time"}}
    assert adapter_source_time(options) == "valid_time"


def test_monthly_profile_rejects_two_records_in_same_month_but_not_missing_months():
    dataset = source().assign_coords(time=np.array(["2001-01-01", "2001-01-15"], dtype="datetime64[ns]"))
    with pytest.raises(ValueError, match="month"):
        adapt(dataset)
    result = adapt(source())
    assert result.sizes["time"] == 2


@pytest.mark.parametrize("method", ["sum", "maximum", "minimum", "standard_deviation", "point"])
def test_monthly_mean_rejects_conflicting_time_statistics(method):
    dataset = source()
    dataset.V.attrs["cell_methods"] = f"latitude: mean time: {method}"
    with pytest.raises(ValueError, match="mean"):
        adapt(dataset)


@pytest.mark.parametrize("container", ["attrs", "encoding"])
def test_cf_links_follow_renaming_and_preserve_selected_scalar_references_and_comments(container):
    dataset = source(MERRA, members=True)
    dataset["lev_bounds"] = (("lev", "bounds"), [[800, 900], [450, 550]])
    dataset.lev.attrs["bounds"] = "lev_bounds"
    dataset["qc"] = xr.zeros_like(dataset.V, dtype=np.int8)
    dataset["cell_area"] = (("lat", "lon"), np.ones((2, 2)))
    dataset["crs"] = xr.DataArray(0, attrs={"grid_mapping_name": "latitude_longitude"})
    links = {"coordinates": "time lev lat lon expver number", "ancillary_variables": "qc",
             "grid_mapping": "crs: lat lon", "cell_measures": "area: cell_area",
             "cell_methods": "time: mean lev: mean (comment: lev: keep original text)"}
    getattr(dataset.V, container).update(links)
    result = adapt(dataset, MERRA, selectors={"expver": 5, "number": 7})
    actual = getattr(result.northward_wind, container)
    assert actual["coordinates"] == "time pressure latitude longitude expver number"
    assert actual["grid_mapping"] == "crs: latitude longitude"
    assert actual["cell_methods"] == "time: mean pressure: mean (comment: lev: keep original text)"
    assert actual["cell_measures"] == "area: cell_area"
    assert result.pressure.attrs["bounds"] == "lev_bounds"
    assert result.lev_bounds.dims == ("pressure", "bounds")
    validate_metadata(result)
    assert getattr(dataset.V, container) == links | ({"units": "m s-1"} if container == "attrs" else {})


@pytest.mark.parametrize("product, attribute, value", [
    (ERA, "rda_dataset", "other"), (ERA, "stream", "mnth"), (ERA, "type", "fc"),
    (ERA, "levtype", "sfc"), (ERA, "product_type", "monthly_averaged_reanalysis_by_hour_of_day"),
    (MERRA, "ShortName", "M2TMNPASM"), (MERRA, "VersionID", "5.12.3"),
    (MERRA, "identifier_product_doi", "https://doi.org/10.5067/OTHER"),
    (JRA, "rda_dataset", "d640004"), (JRA, "field_group", "fcst_p"),
    (JRA, "group", "fcst_p125"),
    (JRA, "data_type", "forecast"), (JRA, "statistical_processing", "Standard deviation(6 hour analysis)"),
    (JRA, "original_time_step", "3-hour"), (JRA, "level_code", 101),
])
def test_available_product_identity_and_statistical_conflicts_fail(product, attribute, value):
    dataset = source(product)
    dataset.attrs[attribute] = value
    with pytest.raises(ValueError, match="identity/statistic"):
        adapt(dataset, product)


@pytest.mark.parametrize("product, evidence", [
    (ERA, {"rda_dataset": "ds633.1", "stream": "moda", "type": "an", "levtype": "pl"}),
    (MERRA, {"ShortName": "M2IMNPASM", "VersionID": "5.12.4", "identifier_product_doi": "10.5067/2E096JV59PK7"}),
    (JRA, {"rda_dataset": "d640002", "group": "anl_p125", "data_type": "analysis",
           "statistical_processing": "Mean(6 hour analysis)", "original_time_step": "6-hour", "level_code": 100}),
])
def test_available_official_metadata_is_checked_and_recorded_without_provider_certification(product, evidence):
    dataset = source(product)
    dataset.V.attrs.update(evidence)
    if product == ERA:
        dataset.V.attrs.update(ecmwf_parameter=132, ecmwf_local_table=128, short_name="v",
                               rda_dataset_group="ERA5 monthly mean atmospheric pressure level analysis [netCDF4]")
    result = adapt(dataset, product)
    identity = json.loads(result.attrs["climate_preprocessing_provenance"])["operations"][-1]["parameters"]["identity"]
    observed = {item["attribute"] for item in identity["verified_metadata"]}
    assert set(evidence) <= observed
    assert identity["provider_authenticated"] is False


@pytest.mark.parametrize("doi", ["https://doi.org/10.5067/2E096JV59PK7",
                                "http://dx.doi.org/10.5067/2e096jv59pk7", "DOI: 10.5067/2E096JV59PK7"])
def test_merra_collection_doi_accepts_standard_identifier_representations(doi):
    dataset = source(MERRA)
    dataset.attrs["identifier_product_doi"] = doi
    adapt(dataset, MERRA)


@pytest.mark.parametrize("attribute, value", [("ecmwf_parameter", 131), ("ecmwf_local_table", 129),
                                               ("short_name", "u"), ("GRIB_paramId", 133)])
def test_era5_parameter_metadata_cannot_conflict_with_explicit_role(attribute, value):
    dataset = source()
    dataset.V.attrs[attribute] = value
    with pytest.raises(ValueError, match="parameter"):
        adapt(dataset)


@pytest.mark.parametrize("value, valid", [(0, True), (48, True), (8, False), (56, False),
    ("resolved u- and v-components of vector quantities relative to easterly and northerly directions", True),
    ("grid relative", False)])
def test_wind_orientation_uses_grib_bit5_mask8_not_j_increment_bit(value, valid):
    dataset = source(JRA)
    dataset.V.attrs["resolution_and_component_flags"] = value
    if valid:
        adapt(dataset, JRA)
    else:
        with pytest.raises(ValueError, match="Grid-relative"):
            adapt(dataset, JRA)


@pytest.mark.parametrize("kind", ["field", "axis", "duplicate_source", "unknown_role", "wrf"])
def test_mapping_and_profile_ambiguity_fails_without_overwrite(kind):
    dataset = source()
    options = {"product": ERA, "variables": {"northward_wind": "V"}}
    if kind == "field":
        dataset["northward_wind"] = xr.zeros_like(dataset.V)
    elif kind == "axis":
        dataset["pressure"] = xr.DataArray(1.0)
    elif kind == "duplicate_source":
        options["variables"]["eastward_wind"] = "V"
    elif kind == "unknown_role":
        options["variables"] = {"rainfall": "V"}
    else:
        options["product"] = "wrf"
    with pytest.raises(ValueError):
        adapt_product(dataset, **options)


def test_lazy_field_graph_and_source_lifetime_survive_adaptation_and_errors():
    dataset = source()
    dataset["V"] = dataset.V.copy(data=da.from_delayed(
        delayed(lambda: np.full((2, 2, 2, 2), 9., dtype=np.float32))(), shape=(2, 2, 2, 2), dtype=np.float32))
    closed = []
    dataset.set_close(lambda: closed.append(True))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args)):
        result = adapt(dataset)
    assert tasks == [] and closed == []
    invalid = dataset.copy(deep=False)
    invalid.V.attrs["units"] = "K"
    invalid.set_close(dataset.close)
    with pytest.raises(ValueError):
        adapt(invalid)
    assert closed == []
    np.testing.assert_array_equal(result.northward_wind.compute(), 9)
    result.close()
    assert closed == [True]


def test_decoded_time_encoding_stays_in_encoding_through_real_netcdf_roundtrip(tmp_path):
    path = tmp_path / "source.nc"
    source().to_netcdf(path)
    with open_dataset(path) as dataset:
        original_units = dataset.time.encoding["units"]
        result = adapt(dataset)
        assert "units" not in result.time.attrs
        assert result.time.encoding["units"] == original_units
        write_dataset(result, tmp_path / "adapted.nc")
    with open_dataset(tmp_path / "adapted.nc") as reopened:
        np.testing.assert_array_equal(reopened.time, source().time)
        np.testing.assert_array_equal(reopened.northward_wind.compute(), source().V.compute())


def test_unknown_options_are_rejected_before_reading_data():
    with pytest.raises(ValueError, match="options"):
        validate_adapter_options({"product": ERA, "variables": {"northward_wind": "V"}, "merge_expver": True})
