"""Invented monthly fields and model references for offline software checks.

The generated files contain no real reanalysis or model reference data.
They must not be used as scientific forcings or as production model inputs.
"""
import csv
import json
from pathlib import Path
import struct

import cftime
from netCDF4 import Dataset
import numpy as np


def _write_record(stream, payload):
    marker = struct.pack('>i', len(payload))
    stream.write(marker + payload + marker)


def _write_gtool_field(stream, item, units, values, axes):
    values = np.asarray(values, dtype=np.float32)
    nz, ny, nx = values.shape
    header = [''] * 64
    for index, value in {
            0: '9010', 1: 'SYNTHETIC QA', 2: item, 13: 'NOT REAL DATA',
            15: units, 24: '0', 25: 'HOUR', 26: '00010101 000000',
            28: axes[0], 29: '1', 30: str(nx),
            31: axes[1], 32: '1', 33: str(ny),
            34: axes[2], 35: '1', 36: str(nz),
            37: 'UR4', 38: '-999', 63: str(values.size)}.items():
        header[index] = value
    _write_record(stream, ''.join(value.ljust(16) for value in header).encode('ascii'))
    _write_record(stream, values.astype('>f4').tobytes(order='C'))


def _model_references(directory):
    directory.mkdir()
    sigma = np.geomspace(.995, .0083, 20).astype(np.float32)
    background = directory / 'synthetic_background.t42l20'
    with background.open('wb') as stream:
        for key, item, units in (
                ('U', 'GAU', 'm/s'), ('V', 'GAV', 'm/s'), ('T', 'GAT', 'K'),
                ('PS', 'GAPS', 'hPa'), ('Q', 'GAQ', 'kg/kg')):
            levels = 1 if key == 'PS' else 20
            value = {'T': 270., 'PS': 1000., 'Q': .001}.get(key, 0.)
            _write_gtool_field(stream, item, units,
                               np.full((levels, 64, 128), value, np.float32),
                               ('GLON128', 'GGLA64', 'SFC1' if key == 'PS' else 'CSIG20'))
    sigma_path = directory / 'GTAXLOC.CSIG20'
    with sigma_path.open('wb') as stream:
        _write_gtool_field(stream, 'CSIG20', 'ND', sigma.reshape(1, 1, 20),
                           ('CSIG20', 'NUMBER', 'SFC1'))
    return background, sigma_path


def _month_midpoint_days(year, month, calendar):
    following_year = year + (month == 12)
    following_month = 1 if month == 12 else month + 1
    boundaries = [cftime.datetime(year, month, 1, calendar=calendar),
                  cftime.datetime(following_year, following_month, 1, calendar=calendar)]
    return float(np.mean(cftime.date2num(
        boundaries, 'days since 2000-01-01 00:00:00', calendar=calendar)))


def create_demo(directory, regression=False, calendar='standard'):
    """Create a new self-contained synthetic case and return two config paths.

    With zero winds and pressure velocity, temperature is linear in actual
    calendar-month midpoint time. Its analytic Q1/Cp is therefore the chosen
    seasonal amplitude times cos(latitude)*cos(longitude), in K/day.
    """
    if calendar not in ('standard', '360_day'):
        raise ValueError('Demo calendar must be standard or 360_day')
    if not isinstance(regression, bool):
        raise ValueError('Demo regression must be true or false')
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    data = directory / 'data'
    data.mkdir()
    background, sigma = _model_references(directory / 'model_reference')
    years = np.arange(2001, 2013 if regression else 2003)
    if regression:
        rng = np.random.default_rng(73192)
        predictor = rng.normal(size=len(years)) * 2.5
        control = rng.normal(size=len(years)) * 1.7
        amplitudes = (.2 + .075 * predictor + .04 * control
                      + .003 * (years - 2006.5))
        with (directory / 'predictors.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['year', 'predictor', 'control'])
            writer.writerows(zip(years.tolist(), predictor.tolist(), control.tolist()))
    else:
        amplitudes = .2 + .2 * np.arange(len(years))
    lat = np.linspace(-90., 90., 19)
    lon = np.arange(0., 360., 10.)
    pressure = np.array([10000., 20000., 50000., 90000.])
    pattern = np.cos(np.deg2rad(lat))[:, None] * np.cos(np.deg2rad(lon))[None, :]
    profile = 250. + .0003 * pressure
    records = []
    for year, amplitude in zip(years, amplitudes):
        anchor = _month_midpoint_days(int(year), 1, calendar)
        for cy, month in ((int(year) - 1, 11), (int(year) - 1, 12),
                          (int(year), 1), (int(year), 2), (int(year), 3)):
            middle = _month_midpoint_days(cy, month, calendar)
            records.append((middle, float(amplitude), middle - anchor))
    units = {'T': 'K', 'U': 'm s-1', 'V': 'm s-1', 'OMEGA': 'Pa s-1', 'PS': 'Pa'}
    with Dataset(data / 'monthly.nc', 'w', format='NETCDF4') as ds:
        for name, size in (('time', len(records)), ('lev', len(pressure)),
                           ('lat', len(lat)), ('lon', len(lon))):
            ds.createDimension(name, size)
        for name, values, unit, standard in (
                ('lat', lat, 'degrees_north', 'latitude'),
                ('lon', lon, 'degrees_east', 'longitude'),
                ('lev', pressure, 'Pa', 'air_pressure')):
            variable = ds.createVariable(name, 'f8', (name,))
            variable[:] = values
            variable.units = unit
            variable.standard_name = standard
        ds['lev'].positive = 'down'
        time = ds.createVariable('time', 'f8', ('time',))
        time.units = 'days since 2000-01-01 00:00:00'
        time.calendar = calendar
        time[:] = [record[0] for record in records]
        for key, unit in units.items():
            dimensions = ('time', 'lat', 'lon') if key == 'PS' else ('time', 'lev', 'lat', 'lon')
            variable = ds.createVariable(key, 'f8', dimensions,
                                         zlib=True, complevel=1, fill_value=np.nan)
            variable.units = unit
            variable.cell_methods = 'time: mean'
            for index, (_, amplitude, elapsed_days) in enumerate(records):
                if key == 'T':
                    variable[index] = profile[:, None, None] + amplitude * elapsed_days * pattern[None]
                else:
                    variable[index] = 100000. if key == 'PS' else 0.
        ds.title = 'SYNTHETIC SOFTWARE TEST; NOT REAL REANALYSIS'
        ds.synthetic_only = 1
        ds.temperature_definition = 'T=profile+seasonal_amplitude*elapsed_calendar_midpoint_days*cos(lat)*cos(lon); U=V=OMEGA=0'
    diagnose = {
        'name': 'synthetic_heating', 'source_product': 'SYNTHETIC SOFTWARE TEST ONLY',
        'data_root': 'data', 'output': 'diagnosis_output',
        'period': {'start_year': int(years[0]), 'end_year': int(years[-1]), 'months': [12, 1, 2]},
        'grid': {'step_degrees': 10.},
        'coordinates': {'time': 'time', 'latitude': 'lat', 'longitude': 'lon', 'pressure': 'lev'},
        'fields': {key: {'files': '{data_root}/monthly.nc', 'variable': key} for key in units},
        'integrity': {'hash_inputs': True, 'require_receipts': False},
        'regression': ({'file': 'predictors.csv', 'year_column': 'year',
                        'predictor_column': 'predictor', 'control_column': 'control',
                        'predictor_units': 'synthetic original input unit',
                        'min_years': 8, 'max_lag': 0} if regression else None)}
    convert = {
        'name': 'synthetic_thermal', 'source_product': 'SYNTHETIC SOFTWARE TEST ONLY',
        'output': 'forcing_output',
        'input': {'file': 'diagnosis_output/diagnosed_heating.nc', 'variable': 'heating_k_day',
                  'coordinates': {'pressure': 'pressure_hpa', 'latitude': 'lat', 'longitude': 'lon'}},
        'model': {'background': background.relative_to(directory).as_posix(),
                  'sigma': sigma.relative_to(directory).as_posix()},
        'region': {'south': -45., 'north': 15., 'west': 30., 'east': 120., 'width': 5.},
        'amplitude_multiplier': 1.}
    paths = {'diagnose': directory / 'diagnose.json', 'convert': directory / 'convert.json'}
    for key, value in (('diagnose', diagnose), ('convert', convert)):
        paths[key].write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (directory / 'SYNTHETIC_ONLY.txt').write_text(
        'Invented software-test inputs and model references. Not a scientific forcing or production model input.\n',
        encoding='utf-8')
    return paths
