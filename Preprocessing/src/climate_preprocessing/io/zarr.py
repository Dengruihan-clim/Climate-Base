"""Local Zarr v2 I/O with lazy fields and an explicit completion protocol.

References
----------
Zarr 2 storage specification and xarray's public Zarr/CF APIs:
https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html
https://zarr.readthedocs.io/en/v2.18.7/api/storage.html
https://docs.xarray.dev/en/stable/generated/xarray.Dataset.to_zarr.html

Climate-Base improvements
-------------------------
Independent adapters preserve CF relationships and Dask execution, require an
explicit local format, and distinguish completed output from a partial store.
Writers stage all chunks, reserve a new directory exclusively, link files without
replacement, then publish a versioned completion manifest last. Readers verify
metadata hashes and accessed file size/mtime, rejecting missing expected chunks
instead of silently receiving Zarr fill values. No cloud transaction, append,
power-loss durability, full-field hash or statistical operation is implied.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import hashlib
import importlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile

import numpy as np
import xarray as xr

from .netcdf import _checked_chunks
from ._time_roundtrip import validate_time_roundtrip
from .write import _prepare
from ..metadata import validate_metadata
from ..provenance import record_operation
from ..time import validate_time
from ..validation import ensure_lazy

_COMPLETE = ".climate-preprocessing-complete.json"
_METADATA = frozenset({".zgroup", ".zattrs", ".zarray", ".zmetadata"})


def _backend():
    try:
        backend = importlib.import_module("zarr")
        importlib.import_module("numcodecs")
    except (ImportError, OSError) as error:
        raise ImportError(
            "Zarr support requires the optional backend; install climate-preprocessing[zarr] "
            "(zarr>=2.18.7,<3 and numcodecs>=0.15.1,<0.16) in the selected environment"
        ) from error
    if str(backend.__version__).split(".")[0] != "2":
        raise ImportError("This adapter supports Zarr v2 only; install climate-preprocessing[zarr] with zarr<3")
    return backend


def _linked(path):
    details = path.lstat()
    return stat.S_ISLNK(details.st_mode) or bool(getattr(details, "st_file_attributes", 0) & 0x400)


def _local_path(value):
    value = os.fspath(value)
    if not isinstance(value, str) or not value or "://" in value or "::" in value:
        raise ValueError("Expected a local Zarr directory path, not a URL or remote/zip store")
    candidate = Path(value).expanduser().absolute()
    # Do not resolve first: that would hide a symbolic link or Windows junction.
    for component in (candidate, *candidate.parents):
        if component.exists() or component.is_symlink():
            if _linked(component):
                raise ValueError("Symbolic links and reparse points are not supported for Zarr stores")
    return candidate


def _identity(path):
    details = path.lstat()
    if _linked(path) or not stat.S_ISDIR(details.st_mode):
        raise ValueError("A regular local directory is required for the Zarr store")
    return details.st_dev, details.st_ino


def _stamp(path, *, metadata=False):
    details = path.lstat()
    if _linked(path) or not stat.S_ISREG(details.st_mode):
        raise ValueError("Zarr entries must be regular files; linked/special entries are unsupported")
    result = {"size": details.st_size, "mtime_ns": details.st_mtime_ns}
    if metadata:
        result["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _snapshot(path):
    """List filesystem metadata and hash only Zarr JSON metadata, never chunks."""
    _identity(path)
    files = {}
    for directory, subdirectories, names in os.walk(path, followlinks=False):
        current = Path(directory)
        for name in subdirectories:
            _identity(current / name)
        for name in names:
            if current == path and name == _COMPLETE:
                continue
            item = current / name
            files[item.relative_to(path).as_posix()] = _stamp(item, metadata=name in _METADATA)
    return files


def _json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeError) as error:
        raise ValueError(f"Invalid or unavailable Zarr metadata: {path.name}") from error


def _format(path, files):
    if (path / "zarr.json").exists():
        raise ValueError("Zarr v3 is unsupported; supply a consolidated local Zarr v2 group")
    group = _json(path / ".zgroup")
    consolidated = _json(path / ".zmetadata")
    if not isinstance(group, dict) or group.get("zarr_format") != 2:
        raise ValueError("Expected a Zarr v2 root group")
    if (not isinstance(consolidated, dict) or consolidated.get("zarr_consolidated_format") != 1
            or not isinstance(consolidated.get("metadata"), dict)):
        raise ValueError("A valid consolidated Zarr v2 metadata document is required")
    metadata_keys = {key for key in files if PurePosixPath(key).name in _METADATA and key != ".zmetadata"}
    if metadata_keys != set(consolidated["metadata"]):
        raise ValueError("Consolidated Zarr metadata omits or adds individual metadata files")
    if any(PurePosixPath(key).name == ".zgroup" and key != ".zgroup" for key in metadata_keys):
        raise ValueError("Nested Zarr groups are unsupported; supply a single root Dataset")
    # A stale consolidated view must not hide conflicting CF attributes/schema.
    for name, metadata in consolidated["metadata"].items():
        key = _safe_key(name)
        if PurePosixPath(key).name not in _METADATA or _json(path / key) != metadata:
            raise ValueError("Consolidated Zarr metadata differs from individual metadata files")


def _safe_key(key):
    if (not isinstance(key, str) or not key or "\\" in key or ":" in key
            or key.startswith("/") or any(part in {"", ".", ".."} for part in key.split("/"))):
        raise ValueError("Invalid relative file key in the Zarr manifest")
    return key


def _manifest(path, require_complete):
    actual = _snapshot(path)
    marker = path / _COMPLETE
    if marker.is_symlink():
        raise ValueError("Completion markers cannot be symbolic links")
    if not marker.exists():
        if require_complete:
            raise ValueError(
                "Zarr store has no completion marker; it is incomplete or externally managed. "
                "For an independently validated external v2 store explicitly use require_complete=False"
            )
        return actual, None
    marker_stamp = _stamp(marker, metadata=True)
    document = _json(marker)
    if (not isinstance(document, dict) or document.get("schema_version") != 1
            or document.get("zarr_format") != 2 or not isinstance(document.get("files"), dict)):
        raise ValueError("Invalid or unsupported Zarr completion manifest")
    expected = document["files"]
    for name, entry in expected.items():
        _safe_key(name)
        if (not isinstance(entry, dict) or type(entry.get("size")) is not int
                or type(entry.get("mtime_ns")) is not int or entry["size"] < 0):
            raise ValueError("Invalid Zarr completion manifest file information")
    if actual != expected:
        raise ValueError("Zarr store files changed or are missing since completion")
    return actual, marker_stamp


class _ReadOnlyStore(MutableMapping):
    """Zarr-compatible mapping which checks the bytes actually requested."""

    def __init__(self, path, files, marker_stamp):
        self.path = Path(path)
        self.files = files
        self.identity = _identity(self.path)
        self.marker_stamp = marker_stamp

    def _check(self, key):
        try:
            if _identity(self.path) != self.identity:
                raise ValueError("Zarr store directory changed after opening")
            if self.marker_stamp is not None and _stamp(self.path / _COMPLETE, metadata=True) != self.marker_stamp:
                raise ValueError("Zarr completion marker changed after opening")
            # xarray caches decoded schema/CF attributes. Check the metadata
            # snapshot on field access too, not just the requested chunk file.
            for name, expected in self.files.items():
                if "sha256" in expected and _stamp(self.path / name, metadata=True) != expected:
                    raise ValueError("Zarr metadata changed after opening; reopen the Dataset")
            path = self.path / key
            for parent in path.parents:
                if parent == self.path:
                    break
                _identity(parent)
            stamp = _stamp(path, metadata=PurePosixPath(key).name in _METADATA)
        except OSError as error:
            raise ValueError("Zarr input file became unavailable after opening") from error
        if stamp != self.files[key]:
            raise ValueError("Zarr input file changed after opening; reopen the Dataset")

    def __getitem__(self, key):
        if key not in self.files:
            raise KeyError(key)
        self._check(key)
        value = (self.path / key).read_bytes()
        self._check(key)
        return value

    def __contains__(self, key):
        if key not in self.files:
            return False
        self._check(key)
        return True

    def __iter__(self):
        return iter(self.files)

    def __len__(self):
        return len(self.files)

    def __setitem__(self, key, value):
        raise TypeError("Input Zarr stores are read-only")

    def __delitem__(self, key):
        raise TypeError("Input Zarr stores are read-only")


def _open(source, chunks, require_complete):
    if (source / "zarr.json").exists():
        raise ValueError("Zarr v3 is unsupported; supply a consolidated local Zarr v2 group")
    files, marker = _manifest(source, require_complete)
    _format(source, files)
    store = _ReadOnlyStore(source, files, marker)
    raw = xr.open_zarr(store, chunks=chunks, consolidated=True, decode_cf=False, zarr_version=2)
    try:
        if isinstance(chunks, Mapping) and set(chunks) - set(raw.dims):
            raise ValueError(f"Unknown chunk dimensions: {sorted(set(chunks) - set(raw.dims))}")
        validate_metadata(raw)
        result = xr.decode_cf(raw, mask_and_scale=True, decode_times=True, decode_coords="all")
        result.set_close(raw.close)
        validate_metadata(result)
        if "time" in result.coords and result.time.ndim == 1:
            validate_time(result)
        result.encoding.pop("source", None)
        for variable in result.variables.values():
            variable.encoding.pop("source", None)
        result = ensure_lazy(result)
    except BaseException:
        raw.close()
        raise
    return result, files, marker


def open_zarr(path, *, chunks="auto", require_complete=True):
    """Read a local, consolidated Zarr v2 group with lazy Dask fields.

    Completion markers are required by default. Set ``require_complete=False``
    only for a separately validated external store: an unmarked store's absent
    chunks may intentionally mean fill values, so its prior completeness cannot
    be proven. Marked stores are always checked, even with this option disabled.
    Small coordinates are decoded for indexing; opening reads metadata and file
    stats but not meteorological chunks. Accessed files are checked for size,
    mtime and directory identity; JSON metadata are additionally hashed. This is
    not a cryptographic snapshot of field values. Keep the store unchanged until
    computation finishes and close the returned Dataset afterwards.
    Copying a managed store must preserve relative keys, sizes and nanosecond
    modification stamps; otherwise regenerate it with this writer from a
    separately validated source. A manifest is not a portable checksum archive.
    """
    chunks = _checked_chunks(chunks)
    if type(require_complete) is not bool:
        raise TypeError("require_complete must be a boolean")
    source = _local_path(path)
    _identity(source)
    backend = _backend()
    dataset, files, marker = _open(source, chunks, require_complete)
    try:
        result = record_operation(dataset, "open_zarr", {
            "chunks": chunks, "zarr_format": 2, "zarr_version": backend.__version__,
            "require_complete": require_complete, "completion_verified": marker is not None,
            "integrity": "metadata SHA256 and accessed file size/mtime; no field hash",
        }, inputs=[{"name": source.name, "file_count": len(files),
                    "bytes": sum(item["size"] for item in files.values())}])
    except BaseException:
        dataset.close()
        raise
    return result


def _cleanup_owned(path, parent, identity, expected=None):
    """Remove only this writer's verified directory, never a replaced target."""
    if not path.exists():
        return
    try:
        owned = path.parent.resolve() == parent and _identity(path) == identity
    except (OSError, ValueError):
        return
    if not owned:
        return
    if expected is not None:
        try:
            actual = _snapshot(path)
        except (OSError, ValueError):
            return
        if any(key not in expected or entry != expected[key] for key, entry in actual.items()):
            return
        allowed_directories = {str(parent) for key in expected for parent in PurePosixPath(key).parents
                               if str(parent) != "."}
        for directory, subdirectories, _ in os.walk(path, followlinks=False):
            for name in subdirectories:
                relative = (Path(directory) / name).relative_to(path).as_posix()
                if relative not in allowed_directories:
                    return
        if (path / _COMPLETE).exists():
            # Published output is complete and must not be removed by cleanup.
            return
    # A second path/identity check immediately precedes recursive removal.
    if path.parent.resolve() == parent and _identity(path) == identity:
        shutil.rmtree(path)


def _publish(stage, destination, files):
    _local_path(destination)
    destination.mkdir(exist_ok=False)
    identity = _identity(destination)
    owned_directories = {destination: identity}

    def check_reservation():
        _local_path(destination)
        if any(_identity(directory) != stamp for directory, stamp in owned_directories.items()):
            raise ValueError("Zarr publication target changed after its exclusive reservation")

    try:
        directories = {parent for key in files for parent in PurePosixPath(key).parents if str(parent) != "."}
        for directory in sorted(directories, key=lambda value: (len(value.parts), str(value))):
            check_reservation()
            child = destination / str(directory)
            child.mkdir(exist_ok=False)
            owned_directories[child] = _identity(child)
        for key in sorted(files):
            check_reservation()
            os.link(stage / key, destination / key)
        check_reservation()
        os.link(stage / _COMPLETE, destination / _COMPLETE)
        check_reservation()
    except BaseException:
        _cleanup_owned(destination, destination.parent.resolve(), identity, files)
        raise


def write_zarr(dataset, path, *, encoding=None):
    """Execute Dask and publish a new local Zarr v2 directory without overwrite.

    All chunks (including all-fill chunks) and consolidated metadata are written
    to a sibling temporary directory with xarray ``safe_chunks=True``. Choose
    compatible Dask/encoding chunk boundaries explicitly; overlapping writes
    are rejected instead of silently rechunked. Source packing is discarded;
    use ``encoding`` for new Zarr compression/chunks/packing. Floating fields
    packed as integers require an explicit finite integer ``_FillValue`` within
    the target dtype; callers must choose a sentinel that does not collide with
    their encoded valid values. No value range or sentinel is inferred.

    A new target directory is reserved using exclusive mkdir. Files and the
    completion manifest are linked without replacement, with the manifest last.
    Readers may observe an unmarked directory during publication and reject it.
    This protocol needs a local filesystem supporting hard links; it is not an
    atomic whole-directory rename or a cloud/power-loss transaction. Existing
    paths, symlinks and concurrent competing targets are never overwritten.
    Failure cleanup only removes verified writer-owned directories. Process
    termination may leave an unmarked store requiring explicit operator cleanup.

    Writing does not close caller-owned source Datasets. No append, region
    update, multi-group layout, ZIP store or Zarr v3 output is supported.
    """
    destination = _local_path(path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Output exists; choose a new Zarr directory")
    backend = _backend()
    prepared, output_encoding = _prepare(dataset, encoding)
    if any(not isinstance(name, str) or not name or "/" in name or "\\" in name
           or name.startswith(".") for name in prepared.variables):
        raise ValueError("Zarr variable names must be nonempty single components without leading dots")
    for name in prepared.variables:
        settings = output_encoding.setdefault(name, {})
        target_dtype = np.dtype(settings.get("dtype", prepared[name].dtype))
        if prepared[name].dtype.kind == "f" and target_dtype.kind in "iu":
            fill = settings.get("_FillValue")
            if fill is None or np.ndim(fill) != 0:
                raise ValueError("Integer packing of floating Zarr values requires an explicit _FillValue")
            value = np.asarray(fill).item()
            limits = np.iinfo(target_dtype)
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not np.isfinite(value) or int(value) != value
                    or not limits.min <= int(value) <= limits.max):
                raise ValueError("Integer Zarr packing _FillValue must be finite, integral and within target dtype")
        if settings.get("write_empty_chunks", True) is not True:
            raise ValueError("Completed output requires write_empty_chunks=True for every variable")
        # The per-variable setting is supported by both the minimum xarray
        # backend and newer releases; the top-level argument is newer.
        settings["write_empty_chunks"] = True
    prepared = record_operation(prepared, "write_zarr", {
        "target": destination.name, "zarr_format": 2, "zarr_version": backend.__version__,
        "publication": "exclusive directory, non-replacing hard links, completion marker last",
    })
    destination.parent.mkdir(parents=True, exist_ok=True)
    _local_path(destination)
    parent = destination.parent.resolve()
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", suffix=".partial", dir=parent))
    identity = _identity(stage)
    try:
        prepared.to_zarr(stage, mode="w-", consolidated=True, compute=True, encoding=output_encoding,
                         safe_chunks=True, zarr_version=2)
        check, files, _ = _open(stage, {}, False)
        try:
            if dict(check.sizes) != dict(prepared.sizes) or set(check.variables) != set(prepared.variables):
                raise ValueError("Written Zarr variables/dimensions differ from the prepared Dataset")
            validate_time_roundtrip(dataset, check)
        finally:
            check.close()
        manifest = {"schema_version": 1, "zarr_format": 2, "files": files}
        with (stage / _COMPLETE).open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, sort_keys=True, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _publish(stage, destination, files)
    finally:
        _cleanup_owned(stage, parent, identity)
    return destination
