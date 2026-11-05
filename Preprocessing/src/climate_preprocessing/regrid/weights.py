"""Content-addressed local weight files with explicit integrity manifests.

References
----------
xESMF weight reuse and sparse weight files:
https://xesmf.readthedocs.io/en/stable/user_api.html

Climate-Base improvements
-------------------------
Independently implemented cache identity includes exact coordinate, bounds and
mask hashes, method, periodicity, numerical options and backend versions. Cache
hits verify a checksum manifest; incomplete or incompatible entries fail rather
than being silently reused or overwritten. Exclusive per-key creation and
hard-link publication protect completed local files. This is a local filesystem
protocol, not a distributed/object-store transaction or an untrusted-cache API.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile

from .grids import GridSpec

_SCHEMA = 1
_METHODS = frozenset({"bilinear", "nearest_s2d", "nearest_d2s", "conservative", "conservative_normed"})


def _serialize(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def weight_metadata(source, target, *, method, periodic, backend_versions):
    """Return public, path-free metadata defining a reusable weight matrix."""
    if not isinstance(source, GridSpec) or not isinstance(target, GridSpec):
        raise TypeError("Weight fingerprints require validated GridSpec objects")
    if not isinstance(method, str) or method not in _METHODS:
        raise ValueError(f"Unsupported regridding method: {method!r}")
    if type(periodic) is not bool:
        raise TypeError("periodic must be boolean")
    if not isinstance(backend_versions, Mapping) or not backend_versions or any(
        not isinstance(key, str) or not key or not isinstance(value, str) or not value
        for key, value in backend_versions.items()
    ):
        raise ValueError("backend_versions must contain nonempty software/version strings")
    return {
        "schema_version": _SCHEMA,
        "source": source.fingerprint_payload, "target": target.fingerprint_payload,
        "method": method, "periodic": periodic,
        "backend_versions": dict(backend_versions),
        "options": {"unmapped_to_nan": True, "extrap_method": None, "ignore_degenerate": False,
                    "skipna": False},
    }


def weight_key(source, target, *, method, periodic, backend_versions):
    """Hash the complete weight contract, not merely source/target shapes."""
    return hashlib.sha256(_serialize(weight_metadata(
        source, target, method=method, periodic=periodic, backend_versions=backend_versions,
    ))).hexdigest()


def _regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or getattr(info, "st_file_attributes", 0) & 1024:
        raise ValueError(f"Cache entry must be a regular local file: {path.name}")
    return info


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _checksum(path):
    before = _regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("Weight cache file changed while opening")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if _identity(os.fstat(handle.fileno())) != _identity(before):
            raise ValueError("Weight cache file changed while checksumming")
    if _identity(_regular(path)) != _identity(before):
        raise ValueError("Weight cache file changed while checksumming")
    return digest.hexdigest(), before.st_size


def _verify(path, manifest, key, metadata):
    info = _regular(manifest)
    if info.st_size > 1_000_000:
        raise ValueError("Weight cache manifest is unexpectedly large")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as error:
        raise ValueError("Weight cache manifest is invalid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"key", "metadata", "bytes", "sha256"}:
        raise ValueError("Weight cache manifest has an unsupported schema")
    if payload["key"] != key or payload["metadata"] != metadata:
        raise ValueError("Weight cache metadata does not match the requested grid/backend contract")
    checksum, size = _checksum(path)
    if payload["sha256"] != checksum or type(payload["bytes"]) is not int or payload["bytes"] != size:
        raise ValueError("Weight cache checksum does not match; regenerate in a new cache location")
    if _identity(_regular(manifest)) != _identity(info):
        raise ValueError("Weight cache manifest changed while verifying")


def validate_cached_weights(cache_dir, key, metadata):
    """Recheck an existing cache entry, for example after backend matrix reads.

    This does not create directories, acquire a build lock, or repair entries.
    Opening and rechecking around a backend read detects ordinary concurrent
    cache replacement; this local cache is not designed for hostile writers.
    """
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("Weight key must be a lowercase SHA256 digest")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA:
        raise ValueError("Unsupported weight metadata schema")
    if hashlib.sha256(_serialize(metadata)).hexdigest() != key:
        raise ValueError("Weight key does not match metadata")
    directory = Path(cache_dir).expanduser().absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Weight cache requires a regular local directory")
    if getattr(directory.stat(), "st_file_attributes", 0) & 1024:
        raise ValueError("Weight cache directory cannot be a reparse point")
    path = directory / f"{key}.nc"
    _verify(path, directory / f"{key}.json", key, metadata)
    return path


def build_or_load_weights(cache_dir, key, metadata, builder):
    """Return ``(weight_path, reused)`` after validating or publishing weights.

    ``metadata`` must be the result of :func:`weight_metadata` and ``key`` its
    SHA256. ``builder(path)`` writes a complete NetCDF weight file to a new
    private temporary path. Callers own validating the numerical matrix and
    opening it immediately after this function. A corrupt, partial, locked, or
    incompatible cache raises; nothing is overwritten and stale locks are not
    automatically removed. Cached weights are independent of source field data.
    """
    if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
        raise ValueError("Weight key must be a lowercase SHA256 digest")
    if not isinstance(metadata, dict) or metadata.get("schema_version") != _SCHEMA:
        raise ValueError("Unsupported weight metadata schema")
    if hashlib.sha256(_serialize(metadata)).hexdigest() != key:
        raise ValueError("Weight key does not match metadata")
    # A caller or builder mutating its dictionary must not change the contract
    # associated with an in-flight content-addressed entry.
    metadata = json.loads(_serialize(metadata))
    if not callable(builder):
        raise TypeError("builder must be callable")
    directory = Path(cache_dir).expanduser().absolute()
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Weight cache requires a regular local directory")
    root_info = directory.stat()
    if getattr(root_info, "st_file_attributes", 0) & 1024:
        raise ValueError("Weight cache directory cannot be a reparse point")
    path, manifest, lock = (directory / f"{key}{suffix}" for suffix in (".nc", ".json", ".lock"))
    try:
        lock_handle = lock.open("xb")
    except FileExistsError as error:
        raise FileExistsError("Weight cache entry is locked; retry after its creator finishes") from error
    with lock_handle:
        lock_identity = _identity(os.fstat(lock_handle.fileno()))
    published = []
    try:
        if path.exists() or path.is_symlink() or manifest.exists() or manifest.is_symlink():
            if not path.exists() or not manifest.exists():
                raise ValueError("Incomplete weight cache entry; weights and manifest must both exist")
            _verify(path, manifest, key, metadata)
            return path, True
        with tempfile.TemporaryDirectory(prefix=f".{key[:12]}-", dir=directory) as temporary:
            stage = Path(temporary)
            temporary_weights = stage / "weights.nc"
            builder(temporary_weights)
            checksum, size = _checksum(temporary_weights)
            if size == 0:
                raise ValueError("Weight builder produced an empty file")
            temporary_manifest = stage / "manifest.json"
            temporary_manifest.write_bytes(_serialize({"key": key, "metadata": metadata,
                                                       "bytes": size, "sha256": checksum}) + b"\n")
            current_root = directory.stat()
            if (current_root.st_dev, current_root.st_ino) != (root_info.st_dev, root_info.st_ino):
                raise ValueError("Weight cache directory changed during construction")
            for source, destination in ((temporary_weights, path), (temporary_manifest, manifest)):
                os.link(source, destination)
                published.append((destination, _identity(_regular(destination))))
            _verify(path, manifest, key, metadata)
        return path, False
    except BaseException:
        # Remove only this invocation's own links, never a replacement file.
        for destination, identity in reversed(published):
            try:
                if _identity(_regular(destination)) == identity:
                    destination.unlink()
            except (FileNotFoundError, ValueError):
                pass
        raise
    finally:
        try:
            if _identity(_regular(lock)) == lock_identity:
                lock.unlink()
        except (FileNotFoundError, ValueError):
            pass
