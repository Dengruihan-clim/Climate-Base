# Developing climate-preprocessing

Use the `src/` package through an installed editable package or wheel. Tests do not modify `sys.path`. On the current Windows workstation use `C:\ProgramData\anaconda3\python.exe` for Python and pip; use an isolated dependency location when additional packages are needed.

```text
python -m pip install -e ".[dev]"
python -m ruff check src tests examples tools
python -m pytest tests -q
python -m build
python tools/check_wheel.py dist
python tools/update_manifest.py
```

`check_wheel.py` installs the built wheel into a temporary directory without changing the base environment, checks its import path, and reruns the complete tests from outside the source tree using the same interpreter. Dependencies must already be available. The [CI template](../docs/preprocessing/preprocessing-ci.yml.example) defines Linux/Windows Python 3.11/3.12 jobs and a conservative dependency combination. It is inactive: this update does not enable GitHub Actions, and local/HPC passes are not remote CI results. To enable automation, copy the template to the repository's `.github/workflows/preprocessing.yml` and commit it.

Package text uses LF line endings, enforced by `.gitattributes`, so Git source
archives and local builds have matching bytes. Regenerate `SHA256SUMS.txt` only
after documentation and validation records are final. Before publishing, verify
the manifest against the staged Git objects as well as the sdist; checking only
working-tree bytes can miss Git's line-ending normalization.

Field operations must use xarray/Dask without implicit `.values`, `.load()` or `.compute()`. Small coordinate validation and NumPy inside scheduled array kernels are allowed. Preserve existing chunks when adding lazy fields. Compute only at documented output/legacy-return boundaries. Keep source Dataset resources alive through execution and close them on failure.

Each changed numerical module must include `References` and `Climate-Base improvements`, distinguish original author code from independently written adapters, and state the input units/dimensions/calendar/missing-value contract. Update [THIRD_PARTY.md](THIRD_PARTY.md), examples and both READMEs when capabilities change. Do not label metadata recognition as complete CF certification or unmeasured Dask execution as a guaranteed speedup.

Tests should prove meaningful behavior: analytical values, CF relationships, calendar semantics, rejection of ambiguous inputs, zero Dask field tasks before explicit execution, resource closure, and failure-safe output. Default tests are offline synthetic data; remote providers and heavy backends need separate future integration suites. Preserve the legacy 42-case baseline and add tests for new behavior.

The package does not select a new software license in this refactor. Preserve existing provenance and all third-party licensing; the maintainer decides the license of author-owned code before distribution.
