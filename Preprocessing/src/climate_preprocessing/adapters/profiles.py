"""Narrow, versioned product representations; no field-name or unit guesses.

References
----------
ERA5 monthly pressure-level documentation:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-pressure-levels-monthly-means
MERRA-2 File Specification and M2IMNPASM 5.12.4 collection:
https://gmao.gsfc.nasa.gov/pubs/docs/Bosilovich785.pdf
https://doi.org/10.5067/2E096JV59PK7
JRA-3Q GDEX monthly isobaric analysis collection and GRIB2 vector flag:
https://gdex.ucar.edu/datasets/d640002/
https://codes.ecmwf.int/grib/format/grib2/ftables/3/3/

Climate-Base improvements
-------------------------
Explicit representation IDs, coordinate defaults, and conflict checks separate
caller declarations from verified metadata. These tables do not authenticate
source provenance, infer units, merge members, or convert statistical products.
"""

PROFILES = {
    "era5-monthly-pressure-levels": {
        "coordinates": {"time": "time", "pressure": "level", "latitude": "latitude", "longitude": "longitude"},
        "identity": {
            "rda_dataset": {"ds633.1", "d633001"},
            "product_type": {"monthly_averaged_reanalysis"},
            "stream": {"moda"}, "type": {"an"}, "levtype": {"pl"},
            "rda_dataset_group": {"ERA5 monthly mean atmospheric pressure level analysis [netCDF4]"},
        },
    },
    "merra2-m2imnpasm-5.12.4": {
        "coordinates": {"time": "time", "pressure": "lev", "latitude": "lat", "longitude": "lon"},
        "identity": {"ShortName": {"M2IMNPASM"}, "Shortname": {"M2IMNPASM"}, "VersionID": {"5.12.4"},
                     "identifier_product_doi": {"10.5067/2E096JV59PK7"}},
    },
    "jra3q-monthly-isobaric-analysis": {
        "coordinates": {"time": "time", "pressure": "pressure_level", "latitude": "lat", "longitude": "lon"},
        "identity": {
            "rda_dataset": {"d640002"}, "group": {"anl_p", "anl_p125"}, "field_group": {"anl_p", "anl_p125"},
            "data_type": {"analysis"}, "statistical_processing": {"Mean(6 hour analysis)"},
            "original_time_step": {"6-hour"}, "level_code": {"100"},
        },
    },
}

# Base-unit labels identify dimensions using the existing strict unit table.
# Geopotential has its own explicit vocabulary because gravity conversion is
# intentionally outside the unit-conversion API.
ROLE_UNITS = {
    "air_temperature": "K", "eastward_wind": "m s-1", "northward_wind": "m s-1",
    "specific_humidity": "kg kg-1", "geopotential": None,
    "geopotential_height": "m", "lagrangian_tendency_of_air_pressure": "Pa s-1",
}

ERA5_PARAMETERS = {
    "air_temperature": (130, "t"), "eastward_wind": (131, "u"), "northward_wind": (132, "v"),
    "specific_humidity": (133, "q"), "geopotential": (129, "z"),
    "geopotential_height": (156, "gh"), "lagrangian_tendency_of_air_pressure": (135, "w"),
}

KNOWN_SELECTION_AXES = frozenset({"expver", "member", "number", "realization"})
