# Source Provenance and Extraction Scope

[简体中文](SOURCE_PROVENANCE.md)

**Authors: RH_Deng; Codex (OpenAI, AI-assisted refactoring)**  
Verification date: 2026-10-25

## Source relationships

This package contains code for preparing **three-dimensional basic states for the LBM T42L20 dry model**. Its sources are the original HPC basic-state preparation programs and their reference copies: the original DJF builder, seasonal extensions, horizontal and vertical numerical modules, a binary module and the original unit tests. The seasonal extensions support MAM, AMJ, MJJ and JJA; they served as references for removing fixed experiment periods and separating seasonal configuration from the processing code.

The remote sources were inspected through read-only operations, and the required programs and existing tests were extracted. No remote files were modified, jobs submitted or model integrations run. SHA256 checks matched for all 10 existing local source copies and their corresponding remote files. Only generic source descriptions, source filenames and fingerprints are recorded below; internal project names, directories, server details and accounts are omitted.

## Key source-file fingerprints

| Logical source | File | SHA256 |
| --- | --- | --- |
| Original DJF builder | `make_merra2_background.py` | `ad3de26cfe59737634872a61f8574363521761e6af4d2e8ea838d4413fcfcde5` |
| Seasonal extensions | `make_merra2_background.py` | `2a2106b030ef9da26a57a15dbf434fdaff9be56122d27f57d3115f9e6e7b7da5` |
| Shared numerical modules in both builders and their reference copies | `lbm_horizontal.py` | `f3c07d691dcaf0f6d15611a5f3996a66fce4ecf4b8b5078dfe0b1ccb4fe2307c` |
| Same as above | `lbm_vertical.py` | `355474a66d5e3428709190200d03cca9c1324568bd1fed577bb8e005086aaec7` |
| Same as above | `lbm_binary.py` | `4ff7b00bb48ccf83a151a85ceb2a7f3590864616e037a6ac142f9b51ed492f2f` |
| Original unit tests for the seasonal extensions | `test_seasonal_builder.py` | `f5c3f93b421324c8284439fd4f3ef148aec3197fed83789607b9b480473b381c` |
| Same as above | `test_lbm_binary.py` | `8bb3d5c3a0d03967dfd63560f9cf920f912be2db326eba00fbd304816d9d2bde` |
| Same as above | `test_lbm_horizontal.py` | `89b6b5d630566eff27d22ff033d2ed4d344809fc0a31696b21fbbce505900d63` |
| Same as above | `test_lbm_vertical.py` | `6a1e1e2cb8a62fe92a964b9b2ecb3223ea88e8fadff11317fafda85c76d83138` |

The three shared modules have identical source fingerprints across the original DJF builder, seasonal extensions and reference copies. These fingerprints identify the **original files before refactoring**. Modules in this package may have different hashes after refactoring or interface adaptation. Use the checksum manifest supplied with this distribution to identify the packaged files.

## Preserved methods and refactoring

The retained scientific processing sequence includes monthly climatologies and equally weighted seasonal monthly means, the exact Gaussian grid, periodic bilinear interpolation with missing-value handling, the `ecmsbs` terrain/virtual-temperature surface-pressure correction, natural cubic splines in `ln(p)` with constant endpoint extension, T42 scalar projection on every σ level, and the Gtool3/GrADS file conventions.

The reusable interface moves data-file selection, CF time and coordinates, unit normalization, season years, model reference files and output directories into configuration. It no longer requires the original HPC directory layout, task state or mandatory downloader verification receipts. Fixed product filenames, fixed period labels and installation steps in the original code have been separated from the processing sequence. Historical production-run validation results in the source documentation have not been adopted as validation results for this package.

The model topography `grz.t42`, σ axis `GTAXLOC.CSIG20` and format template `erawin.t42l20` were used only for integration testing against actual model-file formats in the private verification environment. They are not distributed with the code. That test used synthetic reanalysis inputs; it therefore does not establish scientific validation of a newly generated background from real reanalysis, or stability of a model integration.

## Reference documentation

The original builders cite the following LBM user guide:

[Linear Baroclinic Model user guide](https://github.com/HanXiang0721/Note-Numerical-Methods/blob/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

This reference documents the methodological background of the original programs. It does not imply that the third-party authors participated in this refactoring. Original authorship and terms of use for the model, data and documentation remain independent; this package's author credits do not replace their provenance.
