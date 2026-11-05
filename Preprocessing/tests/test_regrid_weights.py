"""Cache integrity/transaction behavior, not numerical backend integration."""

import json
from pathlib import Path

import pytest

from climate_preprocessing.regrid.grids import rectilinear_grid, validate_grid
from climate_preprocessing.regrid.weights import (
    build_or_load_weights, validate_cached_weights, weight_key, weight_metadata,
)


def contract():
    geometry = validate_grid(rectilinear_grid([-10, 10], [0, 20]))
    options = {"method": "bilinear", "periodic": False, "backend_versions": {"xesmf": "test-fixture"}}
    return weight_key(geometry, geometry, **options), weight_metadata(geometry, geometry, **options)


def builder(path):
    # Cache-only fixtures intentionally do not stand in for ESMF weight tests.
    Path(path).write_bytes(b"cache-integrity-fixture")


def test_cache_miss_and_verified_hit_skip_reconstruction(tmp_path):
    key, metadata = contract()
    calls = []

    def build(path):
        calls.append(path)
        builder(path)

    path, reused = build_or_load_weights(tmp_path, key, metadata, build)
    assert reused is False
    assert path.read_bytes() == b"cache-integrity-fixture"
    again, reused = build_or_load_weights(tmp_path, key, metadata, build)
    assert (again, reused) == (path, True)
    assert len(calls) == 1
    assert validate_cached_weights(tmp_path, key, metadata) == path
    manifest = json.loads((tmp_path / f"{key}.json").read_text())
    assert manifest["metadata"] == metadata
    assert str(tmp_path) not in json.dumps(manifest)
    assert not (tmp_path / f"{key}.lock").exists()


@pytest.mark.parametrize("corruption", ["bytes", "length", "metadata", "json", "weight_missing", "manifest_missing"])
def test_corrupt_or_partial_cache_is_not_reused_or_repaired(tmp_path, corruption):
    key, metadata = contract()
    path, _ = build_or_load_weights(tmp_path, key, metadata, builder)
    manifest = tmp_path / f"{key}.json"
    if corruption == "bytes":
        path.write_bytes(b"Cache-integrity-fixture")
    elif corruption == "length":
        path.write_bytes(b"short")
    elif corruption == "metadata":
        payload = json.loads(manifest.read_text())
        payload["metadata"]["method"] = "conservative"
        manifest.write_text(json.dumps(payload))
    elif corruption == "json":
        manifest.write_text("{")
    elif corruption == "weight_missing":
        path.unlink()
    else:
        manifest.unlink()
    before = {child.name: child.read_bytes() for child in tmp_path.iterdir()}
    with pytest.raises(ValueError):
        build_or_load_weights(tmp_path, key, metadata, builder)
    assert {child.name: child.read_bytes() for child in tmp_path.iterdir()} == before


def test_locked_cache_is_not_overwritten(tmp_path):
    key, metadata = contract()
    lock = tmp_path / f"{key}.lock"
    lock.write_text("another writer")
    with pytest.raises(FileExistsError, match="locked"):
        build_or_load_weights(tmp_path, key, metadata, builder)
    assert lock.read_text() == "another writer"
    assert len(list(tmp_path.iterdir())) == 1


def test_failed_builder_cleans_only_its_temporary_files(tmp_path):
    key, metadata = contract()
    unrelated = tmp_path / "unrelated.nc"
    unrelated.write_bytes(b"keep")

    def fail(path):
        builder(path)
        raise RuntimeError("deliberate interruption")

    with pytest.raises(RuntimeError, match="deliberate"):
        build_or_load_weights(tmp_path, key, metadata, fail)
    assert list(tmp_path.iterdir()) == [unrelated]
    assert unrelated.read_bytes() == b"keep"


def test_nonoverwrite_if_target_appears_during_build(tmp_path):
    key, metadata = contract()
    target = tmp_path / f"{key}.nc"

    def concurrent(path):
        builder(path)
        target.write_bytes(b"other creator")

    with pytest.raises(FileExistsError):
        build_or_load_weights(tmp_path, key, metadata, concurrent)
    assert target.read_bytes() == b"other creator"
    assert not (tmp_path / f"{key}.json").exists()


def test_manifest_publication_failure_removes_owned_weight_only(tmp_path):
    key, metadata = contract()
    manifest = tmp_path / f"{key}.json"

    def concurrent(path):
        builder(path)
        manifest.write_bytes(b"other manifest")

    with pytest.raises(FileExistsError):
        build_or_load_weights(tmp_path, key, metadata, concurrent)
    assert manifest.read_bytes() == b"other manifest"
    assert not (tmp_path / f"{key}.nc").exists()


def test_post_backend_read_checksum_detects_replaced_weight(tmp_path):
    key, metadata = contract()
    path, _ = build_or_load_weights(tmp_path, key, metadata, builder)
    path.write_bytes(b"changed since the adapter opened this file")
    with pytest.raises(ValueError, match="checksum"):
        validate_cached_weights(tmp_path, key, metadata)


@pytest.mark.parametrize("bad_key", ["../outside", "A" * 64, "a" * 63])
def test_cache_keys_cannot_be_paths(tmp_path, bad_key):
    _, metadata = contract()
    with pytest.raises(ValueError, match="SHA256"):
        build_or_load_weights(tmp_path, bad_key, metadata, builder)
    assert not list(tmp_path.iterdir())


def test_mismatched_fingerprint_fails_before_any_file_write(tmp_path):
    key, metadata = contract()
    metadata["periodic"] = True
    with pytest.raises(ValueError, match="does not match metadata"):
        build_or_load_weights(tmp_path, key, metadata, builder)
    assert not list(tmp_path.iterdir())


def test_empty_weight_file_is_not_published(tmp_path):
    key, metadata = contract()
    with pytest.raises(ValueError, match="empty"):
        build_or_load_weights(tmp_path, key, metadata, lambda path: Path(path).touch())
    assert not list(tmp_path.iterdir())
