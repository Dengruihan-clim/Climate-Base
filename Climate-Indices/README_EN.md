# climate_indices upstream source snapshot

[中文](README.md) · [Provenance and licensing](reference/README.md) · [Machine-readable provenance](reference/UPSTREAM.json)

This directory contains a pinned source snapshot of [monocongo/climate_indices](https://github.com/monocongo/climate_indices) for reference and reuse alongside the other tools in this repository. The upstream project provides SPI, SPEI, PET, the Palmer family, and other climate indices. Its algorithms, code, and original documentation remain attributed to the upstream authors and contributors. The additions here are the integration and provenance guides.

## Layout

| Path | Contents |
| --- | --- |
| [`upstream/`](upstream/) | Complete, unmodified snapshot of the pinned commit's Git-tracked files |
| [`upstream/README.md`](upstream/README.md) | Original installation and usage guide |
| [`upstream/LICENSE`](upstream/LICENSE) | Full license, historical statements, and third-party notices |
| [`upstream/CITATION.cff`](upstream/CITATION.cff) | Original software citation metadata |
| [`upstream/DISCLAIMER`](upstream/DISCLAIMER) | Original disclaimer |
| [`reference/README.md`](reference/README.md) | Bilingual provenance, attribution, and maintenance guide |
| [`reference/UPSTREAM.json`](reference/UPSTREAM.json) | Machine-readable record of the pinned source and import scope |
| [`reference/SHA256SUMS.txt`](reference/SHA256SUMS.txt) | Per-file SHA256 checksums for the upstream snapshot |

## Pinned source

| Field | Value |
| --- | --- |
| Repository | <https://github.com/monocongo/climate_indices> |
| Source branch | `main`, identifying the branch sampled for this import |
| Commit | [`e6b4a5098a3f6218e933ccf587e4032122e45e4c`](https://github.com/monocongo/climate_indices/commit/e6b4a5098a3f6218e933ccf587e4032122e45e4c) |
| Git tree | `fa30400b32d1ddcab1ece3fa2885c29df8d9dd51` |
| Upstream commit time | `2026-10-26T05:58:04Z` |
| Import date | `2026-10-26`, Asia/Shanghai |
| Declared version | `3.1.0`, from this commit's project and citation metadata |
| Snapshot scope | 6,113 Git-tracked files, 97,829,440 bytes |

This is a development snapshot pinned to a commit, not a claim that it is a stable release. The pinned upstream README says that `3.1.0` has not yet been published on PyPI. Installing a package from PyPI therefore does not identify this source commit. Git history is not included. This directory is neither an automatically synchronized fork nor a Git submodule.

## Installation and use

The snapshot declares Python `>=3.10,<3.15`. In a compatible, separate environment, run from this directory:

```bash
python -m pip install ./upstream
python -m climate_indices --help
```

Consult [`upstream/pyproject.toml`](upstream/pyproject.toml) for dependencies. This commit uses `hatchling.build` for the regular Python package. The Rust acceleration extension is optional, and ordinary source installation does not require a Rust toolchain. Follow the pinned architecture documentation if you explicitly want to build that extension.

Choose input variables, units, time axes, calibration periods, and missing-data handling according to the [bundled upstream documentation](upstream/docs/) and the applicable scientific methods. The online documentation at <https://climate-indices.readthedocs.io/en/latest/> may change; consult the bundled documentation when reproducing this snapshot.

This import checks source provenance and snapshot integrity. The complete upstream test suite has not been run here, and the scientific results of the individual indices have not been independently revalidated. [`upstream/VALIDATION.md`](upstream/VALIDATION.md) remains an upstream document, not a new execution report from this repository.

## Attribution, citation, and updates

The upstream license is BSD-3-Clause. Its complete copyright notices, terms, disclaimers, and third-party notices are retained in [`upstream/LICENSE`](upstream/LICENSE). Cite the upstream software using [`upstream/CITATION.cff`](upstream/CITATION.cff), and record the exact commit used. The maintainers of this repository do not replace the upstream authors in software attribution.

Updates require explicitly selecting another upstream commit, importing and verifying its files, and updating the provenance record, checksum manifest, and version table. No automatic tracking of upstream `main` is configured. Record any future local code changes separately and identify their upstream base; a modified directory should no longer be described as an unmodified snapshot.
