"""Spatial taper and strict T42L20 LBM thermal-forcing file I/O.

The binary layout matches the LBM Fortran sequential convention: longitude
varies fastest, latitude is north to south, sigma is surface to model top.
Every record contains one horizontal plane of big-endian IEEE float32 values,
enclosed by four-byte big-endian record-length markers.  The 61 records are
20 vorticity, 20 divergence, 20 temperature, and one surface-pressure plane.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
from typing import Callable

import numpy as np


NLON = 128
NLAT = 64
NLEV = 20
NRECORDS = 3 * NLEV + 1
RECORD_BYTES = NLAT * NLON * 4
FILE_BYTES = NRECORDS * (RECORD_BYTES + 8)
_MARKER = struct.pack(">i", RECORD_BYTES)


def _vector(values, name: str, size: int | None = None) -> np.ndarray:
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} contains masked values")
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real")
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if size is not None and result.size != size:
        raise ValueError(f"{name} must contain {size} values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def region_taper(lat, lon, south=-45.0, north=15.0, west=30.0,
                 east=120.0, width=5.0) -> np.ndarray:
    """Return a separable raised-cosine window, shaped ``(lat, lon)``.

    ``lat`` and ``lon`` are one-dimensional coordinate vectors in any order.
    Longitude is periodic: the region runs eastward from ``west`` to ``east``.
    The default longitude interval is 30..120 degrees east. Other cyclic
    intervals, including intervals crossing Greenwich, are also supported.
    All four boundaries and all exterior points are exactly zero.  Each edge
    rises from zero to one over ``width`` degrees *inside* the region; corner
    weights are the product of their two edge weights.  No spectral filtering
    is applied, so exterior zero values remain exact.
    """
    latitude = _vector(lat, "lat")
    longitude = _vector(lon, "lon")
    bounds = np.asarray([south, north, west, east, width], dtype=np.float64)
    if not np.all(np.isfinite(bounds)):
        raise ValueError("region bounds and width must be finite")
    south, north, west, east, width = map(float, bounds)
    if not -90.0 <= south < north <= 90.0:
        raise ValueError("latitude bounds must satisfy -90 <= south < north <= 90")
    if np.any(np.abs(latitude) > 90.0):
        raise ValueError("lat must be between -90 and 90 degrees")
    span = (east - west) % 360.0
    if span == 0.0:
        raise ValueError("longitude bounds must describe a nonzero arc below 360 degrees")
    if width <= 0.0 or 2.0 * width > min(north - south, span):
        raise ValueError("width must be positive and at most half either region span")

    def edge_weight(distance: np.ndarray, inside: np.ndarray) -> np.ndarray:
        weight = np.zeros(distance.shape, dtype=np.float64)
        core = inside & (distance >= width)
        ramp = inside & (distance > 0.0) & (distance < width)
        weight[core] = 1.0
        weight[ramp] = 0.5 * (1.0 - np.cos(np.pi * distance[ramp] / width))
        return weight

    lat_inside = (latitude > south) & (latitude < north)
    lat_distance = np.minimum(latitude - south, north - latitude)
    offset = np.mod(longitude - west, 360.0)
    lon_inside = (offset > 0.0) & (offset < span)
    lon_distance = np.minimum(offset, span - offset)
    lat_weight = edge_weight(lat_distance, lat_inside)
    lon_weight = edge_weight(lon_distance, lon_inside)
    return lat_weight[:, None] * lon_weight[None, :]


def _atomic_write(path, writer: Callable) -> Path:
    """Publish a complete sibling temporary file without replacing a target."""
    # Keep a final symlink visible to the existence check, including dangling
    # symlinks; resolve() here could otherwise create their referent instead.
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {target}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp",
                                     dir=target.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        # link() publishes the entire inode and fails if a concurrent writer
        # created the destination; unlike POSIX rename(), it cannot overwrite.
        os.link(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)
    return target


def write_forcing_grd(path, heating_k_s) -> Path:
    """Write thermal-only T42L20 forcing and return its absolute path.

    ``heating_k_s`` must be finite, real, and shaped ``(20, 64, 128)`` in
    descending-sigma / north-to-south-latitude / eastward-longitude order.
    Temperature values are K/s and retain their supplied signs and amplitudes.
    Vorticity, divergence, and surface-pressure forcing are exactly zero.
    Missing/masked values and float32 overflow are errors.  The write is atomic
    and an existing destination is never replaced.
    """
    if np.ma.isMaskedArray(heating_k_s) and np.any(np.ma.getmaskarray(heating_k_s)):
        raise ValueError("heating_k_s contains masked values")
    if np.iscomplexobj(heating_k_s):
        raise ValueError("heating_k_s must be real")
    heating = np.asarray(heating_k_s, dtype=np.float64)
    if heating.shape != (NLEV, NLAT, NLON):
        raise ValueError(f"heating_k_s must have shape {(NLEV, NLAT, NLON)}")
    if not np.all(np.isfinite(heating)):
        raise ValueError("heating_k_s contains nonfinite values")
    with np.errstate(over="ignore", invalid="ignore"):
        thermal = heating.astype(">f4")
    if not np.all(np.isfinite(thermal)):
        raise ValueError("heating_k_s exceeds finite float32 range")
    zero_payload = bytes(RECORD_BYTES)

    def write(stream):
        for _ in range(2 * NLEV):
            stream.write(_MARKER)
            stream.write(zero_payload)
            stream.write(_MARKER)
        for plane in thermal:
            stream.write(_MARKER)
            stream.write(plane.tobytes(order="C"))
            stream.write(_MARKER)
        stream.write(_MARKER)
        stream.write(zero_payload)
        stream.write(_MARKER)

    return _atomic_write(path, write)


def read_forcing_grd(path) -> dict[str, np.ndarray]:
    """Strictly read this thermal-only binary format into independent arrays.

    Returns ``vor``, ``div``, ``tmp`` with shape ``(20,64,128)`` and ``ps``
    with shape ``(64,128)``.  Values retain their on-disk north-to-south order.
    Incorrect lengths/markers, extra bytes, nonfinite values, or nonzero
    nonthermal forcing raise ``ValueError`` rather than being silently read.
    """
    source = Path(path)
    if source.stat().st_size != FILE_BYTES:
        raise ValueError(f"Forcing must contain exactly {FILE_BYTES} bytes")
    planes = []
    with source.open("rb") as stream:
        for record in range(NRECORDS):
            start = stream.read(4)
            if start != _MARKER:
                raise ValueError(f"Incorrect leading record marker at record {record + 1}")
            payload = stream.read(RECORD_BYTES)
            if len(payload) != RECORD_BYTES:
                raise ValueError(f"Truncated payload at record {record + 1}")
            if stream.read(4) != _MARKER:
                raise ValueError(f"Incorrect trailing record marker at record {record + 1}")
            plane = np.frombuffer(payload, dtype=">f4").astype(np.float32).reshape(NLAT, NLON)
            if not np.all(np.isfinite(plane)):
                raise ValueError(f"Nonfinite payload at record {record + 1}")
            if (record < 2 * NLEV or record == 3 * NLEV) and np.any(plane != 0):
                raise ValueError(f"Nonthermal forcing must be zero at record {record + 1}")
            planes.append(plane)
        if stream.read(1):
            raise ValueError("Unexpected trailing bytes")
    return {"vor": np.stack(planes[:NLEV]),
            "div": np.stack(planes[NLEV:2 * NLEV]),
            "tmp": np.stack(planes[2 * NLEV:3 * NLEV]),
            "ps": planes[-1]}


def write_forcing_ctl(path, dataset, sigma, lat, lon,
                      title="Custom LBM thermal forcing") -> Path:
    """Write a matching GrADS descriptor with ascending YDEF plus YREV.

    ``sigma`` must contain 20 strictly descending values in (0,1], ``lat``
    the 64 Gaussian latitudes from north to south, and ``lon`` the 128 T42
    longitudes 0, 2.8125, ..., 357.1875.  A relative ``dataset`` is interpreted
    relative to the descriptor directory.  DSET is always written relative to
    that directory.  As for the binary writer, an existing path is an error.
    """
    if not isinstance(title, str) or not title or len(title) > 128:
        raise ValueError("title must contain 1..128 ASCII characters")
    if "\n" in title or "\r" in title:
        raise ValueError("title must occupy one line")
    try:
        title.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("title must contain ASCII characters") from error
    levels = _vector(sigma, "sigma", NLEV)
    latitude = _vector(lat, "lat", NLAT)
    longitude = _vector(lon, "lon", NLON)
    if not np.all((levels > 0.0) & (levels <= 1.0)) or not np.all(np.diff(levels) < 0.0):
        raise ValueError("sigma must be strictly descending within (0,1]")
    gaussian = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(NLAT)[0]))[::-1]
    if not np.all(np.diff(latitude) < 0.0) or not np.allclose(latitude, gaussian, rtol=0, atol=1e-4):
        raise ValueError("lat must be the 64 Gaussian latitudes ordered north to south")
    if not np.allclose(longitude, np.arange(NLON) * (360.0 / NLON), rtol=0, atol=1e-8):
        raise ValueError("lon must be the 128 T42 longitudes from 0 to 357.1875")
    target = Path(path).absolute()
    source = Path(dataset)
    if not source.is_absolute():
        source = target.parent / source
    relative = Path(os.path.relpath(source.resolve(), target.parent)).as_posix()
    if "\n" in relative or "\r" in relative:
        raise ValueError("dataset path must not contain line breaks")

    def coordinates(values):
        return "\n".join(" ".join(f"{value:.12g}" for value in values[start:start + 8])
                         for start in range(0, len(values), 8))

    descriptor = (
        f"DSET ^{relative}\n"
        f"TITLE {title}\n"
        "UNDEF -999.\n"
        "OPTIONS SEQUENTIAL YREV BIG_ENDIAN\n"
        f"XDEF {NLON} LINEAR 0. {360.0 / NLON:.12g}\n"
        f"YDEF {NLAT} LEVELS\n{coordinates(latitude[::-1])}\n"
        f"ZDEF {NLEV} LEVELS\n{coordinates(levels)}\n"
        "TDEF 1 LINEAR 00Z01JAN0001 1dy\n"
        "VARS 4\n"
        f"vor {NLEV} 99 vorticity forcing [s-2]\n"
        f"div {NLEV} 99 divergence forcing [s-2]\n"
        f"tmp {NLEV} 99 heating [K/s]; multiply by 86400 for K/day\n"
        "ps 0 99 surface pressure forcing\n"
        "ENDVARS\n"
    )
    return _atomic_write(target, lambda stream: stream.write(descriptor.encode("ascii")))
