# 来源与归属 / Provenance and attribution

本子目录说明 [`../upstream/`](../upstream/) 中 `climate_indices` 源码的来源、固定版本和引用方式。上游代码归原作者及贡献者所有；本仓库仅收录一个固定快照，并新增来源说明及完整性记录。

This directory records the origin, pinned revision, and citation of the `climate_indices` sources in [`../upstream/`](../upstream/). The software remains the work of its upstream authors and contributors. This repository adds a pinned copy, provenance documentation, and integrity records.

## 固定来源 / Pinned origin

| 项目 / Field | 记录 / Value |
| --- | --- |
| 项目 / Project | `climate_indices` |
| 上游 / Repository | <https://github.com/monocongo/climate_indices> |
| 原作者 / Upstream author in citation metadata | James Adams; upstream contributors retain their respective contributions |
| 来源分支 / Source branch | `main` |
| 固定提交 / Pinned commit | [`e6b4a5098a3f6218e933ccf587e4032122e45e4c`](https://github.com/monocongo/climate_indices/commit/e6b4a5098a3f6218e933ccf587e4032122e45e4c) |
| Git tree | `fa30400b32d1ddcab1ece3fa2885c29df8d9dd51` |
| 上游提交时间 / Commit time | `2026-10-26T05:58:04Z` |
| 导入日期 / Import date | `2026-10-26`, Asia/Shanghai |
| 源码版本字段 / Declared source version | `3.1.0`; identifies the snapshot's metadata, not a stable-release guarantee |
| 导入范围 / Imported scope | 6,113 Git-tracked files, 97,829,440 bytes |
| 本地快照 / Bundled snapshot | [`../upstream/`](../upstream/) |
| 上游在线文档 / Online documentation | <https://climate-indices.readthedocs.io/en/latest/> |

固定提交是判断本次收录内容的依据；分支名和在线文档可能在以后变化。机器记录见 [`UPSTREAM.json`](UPSTREAM.json)，逐文件校验值见 [`SHA256SUMS.txt`](SHA256SUMS.txt)。

The commit identifies the imported content. Branch tips and online documentation may move independently. See [`UPSTREAM.json`](UPSTREAM.json) for machine-readable provenance and [`SHA256SUMS.txt`](SHA256SUMS.txt) for per-file checksums.

## 许可与原始声明 / License and original notices

[`../upstream/LICENSE`](../upstream/LICENSE) 原样保留完整 BSD-3-Clause 许可证、历史 NOAA public-domain 声明，以及所包含的 Cephes/SciPy、Boost、NumPy 等第三方 notices。使用和再分发时应遵守其中适用的条款，不以本仓库的收录说明替代原许可证。

[`../upstream/LICENSE`](../upstream/LICENSE) preserves the full BSD-3-Clause license, historical NOAA public-domain statement, and included third-party notices such as Cephes/SciPy, Boost, and NumPy. Follow the applicable terms for use and redistribution; this provenance guide does not replace the license.

上游 [`CITATION.cff`](../upstream/CITATION.cff)、[`DISCLAIMER`](../upstream/DISCLAIMER)、版权头和公开作者信息均保留。原作者身份、软件版权和对原机构的声明不会因收录至本仓库而转移，也不表示原作者或机构为本仓库背书。

The original [`CITATION.cff`](../upstream/CITATION.cff), [`DISCLAIMER`](../upstream/DISCLAIMER), copyright headers, and public author metadata remain intact. Copying the sources does not transfer authorship or imply endorsement by their authors or institutions.

## 引用 / Citation

研究中使用该软件时，请优先依据所用快照的 [`../upstream/CITATION.cff`](../upstream/CITATION.cff) 引用 `climate_indices` 及其原作者，并在方法或补充材料中报告上述完整提交 SHA。引用具体指数时，还应引用其方法所依据的原始研究。

For research use, cite `climate_indices` and its original authors using the bundled [`CITATION.cff`](../upstream/CITATION.cff), and report the full commit SHA above. Also cite the primary methodological literature for the indices actually used.

该提交的引用文件没有登记软件 DOI；本说明不补造 DOI。若今后上游提供持久标识符，应核对它所对应的版本再使用。引用本仓库的来源说明不能代替引用上游软件。

The pinned citation metadata does not register a software DOI, and none is invented here. If upstream later supplies a persistent identifier, verify which version it represents before citing it. Citing this integration guide does not replace citation of the upstream software.

## 导入边界与维护 / Import scope and maintenance

- `upstream/` 保存固定提交的完整、未修改 Git 跟踪文件快照，包括上游源码、测试、文档和随库示例；不包含 `.git` 历史目录。
- 本仓库新增说明放在 `Climate-Indices/` 与本 `reference/` 子目录，不改写上游 README、许可证或作者署名。
- 这是源码快照，不是 Git submodule、实时镜像或自动同步的 fork。
- 后续更新须明确选择新提交，重新核对来源与文件，更新元数据、校验清单和版本说明。若作本地代码修改，应单独说明，不再标注为未修改快照。
- 本次仅核对导入完整性；未运行完整上游测试套件，也未为其全部科学结果新增独立验证声明。

The bundled tree is an unmodified snapshot of the pinned commit's tracked files, including code, tests, documentation, and upstream examples; Git history is omitted. The integration guides are kept outside that tree. This is not a submodule, live mirror, or automatically synchronized fork. Future updates must select an explicit commit and refresh provenance and checksums. Record any future local modifications separately. Import integrity checks are not a claim that the complete upstream tests were run or that every scientific result was independently validated.

新增说明不记录本机数据路径、登录凭据或个人计算结果。上游已经公开的版权与引用信息保留原样，以保持来源可追溯。

The added guides contain no local data paths, login credentials, or personal analysis results. Upstream's existing public copyright and citation information is preserved for traceability.
