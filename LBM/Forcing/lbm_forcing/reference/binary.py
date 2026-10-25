"""Strict binary I/O for the LBM T42L20 background-state format.

Arrays are C-order (sigma, latitude, longitude). Latitude MUST run north to
south; longitude MUST be 0, 2.8125, ..., 357.1875. Sigma runs surface to top.
Surface pressure is in hPa, not Pa. No spatial interpolation is done here.

The native ecmsbs.f/ncepsbs.f writers emit U, V, T, PS, Q in that order.
Each Gtool field consists of a 1024-byte ASCII header (64 * 16 chars), then
one float32 payload, each inside 4-byte Fortran sequential record markers.
The verified erawin.t42l20 template is big endian. Gtool data are Fortran
(longitude, latitude, sigma), equivalent to C-order (sigma, latitude, lon).
GrADS output has one sequential record per horizontal plane, same ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO, Mapping, Sequence
import os
import struct
import tempfile

import numpy as np


FIELD_ORDER = ("U", "V", "T", "PS", "Q")
GTOOL_ITEMS = ("GAU", "GAV", "GAT", "GAPS", "GAQ")
FIELD_UNITS = ("m/s", "m/s", "K", "hPa", "kg/kg")
NX, NY, NZ = 128, 64, 20
MISSING = -999.0


@dataclass
class GtoolRecord:
    """One Gtool header/data pair; data always have shape (nz, ny, nx)."""

    header: tuple[str, ...]
    data: np.ndarray
    endian: str = ">"

    @property
    def item(self) -> str:
        return self.header[2].strip()

    @property
    def units(self) -> str:
        return self.header[15].strip()


def _read_record(handle: BinaryIO, endian: str, *, eof_ok: bool = False) -> bytes | None:
    leading = handle.read(4)
    if not leading and eof_ok:
        return None
    if len(leading) != 4:
        raise ValueError("Truncated Fortran record marker")
    size = struct.unpack(endian + "i", leading)[0]
    if size < 0:
        raise ValueError("Segmented/negative Fortran records are unsupported")
    payload = handle.read(size)
    trailing = handle.read(4)
    if len(payload) != size or trailing != leading:
        raise ValueError("Truncated record or mismatching Fortran record markers")
    return payload


def _write_record(handle: BinaryIO, payload: bytes, endian: str) -> None:
    marker = struct.pack(endian + "i", len(payload))
    handle.write(marker)
    handle.write(payload)
    handle.write(marker)


def _dimensions(header: Sequence[str]) -> tuple[int, int, int]:
    try:
        xyz = tuple(int(header[end]) - int(header[start]) + 1
                    for start, end in ((29, 30), (32, 33), (35, 36)))
    except (ValueError, IndexError) as exc:
        raise ValueError("Invalid Gtool axis start/end fields") from exc
    if min(xyz) < 1:
        raise ValueError(f"Invalid Gtool dimensions: {xyz}")
    return xyz[2], xyz[1], xyz[0]


def read_gtool(path: str | Path) -> list[GtoolRecord]:
    """Read and validate all UR4 Gtool records, detecting byte order.

    The nominal SIZE header is not trusted: the original ERA40 background
    retains a stale T21 size. Payload size is checked against axis extents.
    Header text, including padding, is retained for template-based writing.
    """
    result: list[GtoolRecord] = []
    with Path(path).open("rb") as handle:
        first = handle.read(4)
        if first == struct.pack(">i", 1024):
            endian = ">"
        elif first == struct.pack("<i", 1024):
            endian = "<"
        else:
            raise ValueError("Expected a 1024-byte Gtool header record")
        handle.seek(0)
        while True:
            raw_header = _read_record(handle, endian, eof_ok=True)
            if raw_header is None:
                break
            if len(raw_header) != 1024:
                raise ValueError("Every Gtool header must be 1024 bytes")
            header = tuple(raw_header[k:k + 16].decode("ascii")
                           for k in range(0, 1024, 16))
            if header[0].strip() != "9010" or header[37].strip() != "UR4":
                raise ValueError("Only Gtool 9010 / UR4 float32 is supported")
            shape = _dimensions(header)
            payload = _read_record(handle, endian)
            if len(payload) != int(np.prod(shape)) * 4:
                raise ValueError(f"Payload size disagrees with axis dimensions {shape}")
            data = np.frombuffer(payload, dtype=endian + "f4").astype(np.float32)
            result.append(GtoolRecord(header, data.reshape(shape), endian))
    return result


def _atomic_write(path: str | Path, write_contents) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                    dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            write_contents(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, destination)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def _fields(fields: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(fields) != set(FIELD_ORDER):
        raise ValueError(f"Expected exactly {FIELD_ORDER}; got {tuple(fields)}")
    normalized = {}
    for name in FIELD_ORDER:
        if np.ma.isMaskedArray(fields[name]) and np.ma.getmaskarray(fields[name]).any():
            raise ValueError(f"{name} contains masked values")
        arr = np.asarray(fields[name])
        if name == "PS" and arr.shape == (NY, NX):
            arr = arr[None, :, :]
        expected = (1 if name == "PS" else NZ, NY, NX)
        if arr.shape != expected:
            raise ValueError(f"{name} shape {arr.shape}; expected {expected}")
        if not np.isfinite(arr).all() or np.any(arr == MISSING):
            raise ValueError(f"{name} contains missing/non-finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            arr = np.asarray(arr, dtype=np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} cannot be represented as float32")
        normalized[name] = arr
    if np.any(normalized["PS"] <= 0) or np.any(normalized["PS"] > 2000):
        raise ValueError("PS must be positive surface pressure in hPa, not Pa")
    return normalized


def _text(value: str, width: int = 16) -> str:
    value.encode("ascii")
    if len(value) > width:
        raise ValueError(f"Header value exceeds {width} ASCII characters")
    return value.ljust(width)


def _validate_template(records: Sequence[GtoolRecord]) -> None:
    if tuple(rec.item for rec in records) != GTOOL_ITEMS:
        raise ValueError(f"Template items must be ordered {GTOOL_ITEMS}")
    if any(rec.endian != ">" for rec in records):
        raise ValueError("The LBM template must use the verified big-endian format")
    for name, units, rec in zip(FIELD_ORDER, FIELD_UNITS, records):
        expected = (1 if name == "PS" else NZ, NY, NX)
        axes = tuple(rec.header[k].strip() for k in (28, 31, 34))
        expected_axes = ("GLON128", "GGLA64", "SFC1" if name == "PS" else "CSIG20")
        if rec.data.shape != expected or axes != expected_axes or rec.units != units:
            raise ValueError(f"{name} template has inconsistent axes, dimensions or units")


def write_background(template: str | Path, destination: str | Path,
                     fields: Mapping[str, np.ndarray],
                     title: str = "Custom LBM basic state", year: int = 1) -> Path:
    """Write big-endian native LBM U/V/T/PS/Q using verified template metadata.

    ``fields`` uses uppercase U,V,T,PS,Q keys and the ordering documented at
    module level. ``title`` (up to 128 ASCII chars) is stored across the eight
    EDIT history fields; its first 16 chars also identify the dataset. Each
    variable's original title and units, all axis names, and missing value
    conventions are retained. Model date is Jan 1 of ``year`` at 00 UTC;
    this is a stationary background, not a dated 1991-2020 model integration.
    Header SIZE is corrected, and creation/modification stamps identify the
    newly generated file. Writing is atomic; no partial final file is left.
    """
    records = read_gtool(template)
    _validate_template(records)
    arrays = _fields(fields)
    if not isinstance(year, int) or not 1 <= year <= 9999:
        raise ValueError("year must be an integer in 1..9999")
    if not title or len(title) > 128:
        raise ValueError("title must contain 1..128 ASCII characters")
    title.encode("ascii")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d %H%M%S")

    def emit(handle):
        for name, rec in zip(FIELD_ORDER, records):
            header = list(rec.header)
            header[1] = _text(title[:16])
            for k in range(8):
                header[16 + k] = _text(title[16 * k:16 * (k + 1)])
            header[24], header[25] = f"{0:16d}", _text("HOUR")
            header[26], header[27] = _text(f"{year:04d}0101 000000"), f"{0:16d}"
            header[59], header[60] = _text(timestamp), _text("PYLBM1.0")
            header[61], header[62] = _text(timestamp), _text("CUSTOM")
            header[63] = f"{arrays[name].size:16d}"
            _write_record(handle, "".join(header).encode("ascii"), ">")
            _write_record(handle, arrays[name].astype(">f4").tobytes(order="C"), ">")

    _atomic_write(destination, emit)
    return Path(destination)


def write_grads_sequential(destination: str | Path,
                           fields: Mapping[str, np.ndarray]) -> Path:
    """Write the matching 81-plane, big-endian sequential GrADS dataset."""
    arrays = _fields(fields)

    def emit(handle):
        for name in FIELD_ORDER:
            for plane in arrays[name]:
                _write_record(handle, plane.astype(">f4").tobytes(order="C"), ">")

    _atomic_write(destination, emit)
    return Path(destination)


def t42_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Return longitude (eastward) and Gaussian latitude (north to south)."""
    latitude = np.rad2deg(np.arcsin(np.polynomial.legendre.leggauss(NY)[0]))[::-1]
    return np.arange(NX, dtype=float) * (360.0 / NX), latitude


def read_sigma(path: str | Path) -> np.ndarray:
    """Load exact float32 CSIG20 centers from GTAXLOC.CSIG20 as float64."""
    records = read_gtool(path)
    if len(records) != 1 or records[0].item != "CSIG20" or records[0].data.size != NZ:
        raise ValueError("Expected the twenty-level CSIG20 axis file")
    sigma = records[0].data.ravel().astype(np.float64)
    if not (np.all((sigma > 0) & (sigma < 1)) and np.all(np.diff(sigma) < 0)):
        raise ValueError("CSIG20 must decrease from surface to model top")
    return sigma


def write_grads_ctl(destination: str | Path, dataset: str | Path,
                    sigma: np.ndarray, title: str = "Custom LBM basic state",
                    year: int = 1) -> Path:
    """Describe the matching sequential dataset, including big_endian/yrev."""
    sigma = np.asarray(sigma)
    if sigma.shape != (NZ,) or not np.isfinite(sigma).all() or not np.all(np.diff(sigma) < 0):
        raise ValueError("Expected twenty decreasing sigma values")
    if not 1 <= year <= 9999:
        raise ValueError("Invalid year")
    title.encode("ascii")
    if "\n" in title or "\r" in title:
        raise ValueError("Title must occupy one line")
    _, north_to_south = t42_coordinates()
    target = Path(destination)
    dset = Path(dataset)
    if dset.is_absolute():
        dset = Path(os.path.relpath(dset, target.parent))
    lines = [f"DSET ^{dset.as_posix()}", "OPTIONS sequential big_endian yrev",
             f"TITLE {title}", "UNDEF -999.", "XDEF 128 LINEAR 0 2.8125", "YDEF 64 LEVELS"]
    for values in (north_to_south[::-1], sigma):
        if values is sigma:
            lines.append("ZDEF 20 LEVELS")
        for start in range(0, len(values), 8):
            lines.append(" ".join(f"{value:.10g}" for value in values[start:start + 8]))
    lines.extend([f"TDEF 1 LINEAR 00Z01JAN{year:04d} 1mo", "VARS 5",
                  "u 20 99 zonal wind [m/s]", "v 20 99 meridional wind [m/s]",
                  "t 20 99 temperature [K]", "p 0 99 surface pressure [hPa]",
                  "q 20 99 specific humidity [kg/kg]", "ENDVARS", ""])
    _atomic_write(target, lambda handle: handle.write("\n".join(lines).encode("ascii")))
    return target
