# Climate-Base

面向气候研究的可配置 Python 前处理、统计与动力诊断工具集。提供再分析月资料读取、完整季节平均、统计回归和趋势分析、波活动通量与环流诊断，以及 **LBM T42L20 干模式背景态与热源强迫制作**。各模块包含示例、测试和中英文说明。

Configurable Python preprocessing, statistics, and dynamical diagnostics for climate research: monthly reanalysis reading, seasonal means, regression, trends, wave activity flux, circulation diagnostics, and **basic states and thermal forcing for dry T42L20 LBM experiments**. Modules include examples, tests, and bilingual documentation.

## Modules / 模块

| Directory / 目录 | Purpose / 用途 | Documentation / 说明 |
| --- | --- | --- |
| [Preprocessing](Preprocessing/) | Lazy xarray/Dask NetCDF preparation; compatible monthly/seasonal APIs / xarray/Dask 惰性前处理及兼容月资料/季节接口 | [中文](Preprocessing/README.md) · [English](Preprocessing/README_EN.md) |
| [Statistical](Statistical/) | Year alignment, controlled regression, trends, bootstrap and FDR / 年份配对、控制回归、趋势、bootstrap 与 FDR | [中文](Statistical/README.md) · [English](Statistical/README_EN.md) |
| [Dynamics](Dynamics/) | TN wave flux, Rossby wave source, spherical winds and mass streamfunction / TN 波通量、Rossby 波源、球谐风场与质量流函数 | [中文](Dynamics/README.md) · [English](Dynamics/README_EN.md) |
| [LBM/Basic](LBM/Basic/) | Reanalysis climatology → model basic state / 再分析气候场 → 模式背景态 | [中文](LBM/Basic/README.md) · [English](LBM/Basic/README_EN.md) |
| [LBM/Forcing](LBM/Forcing/) | Apparent-heating diagnosis, optional regression, and thermal-forcing conversion / 视热源诊断、可选回归与热强迫转换 | [中文](LBM/Forcing/README.md) · [English](LBM/Forcing/README_EN.md) |

See the [LBM module overview / LBM 模块总览](LBM/README.md) for the directory structure.

基础前处理当前版本为 **0.3.0**。已补齐气压坐标规范化、明确预期时间检查/切片和用户布尔掩膜。xarray/Dask NetCDF 闭环包括惰性读取、坐标整理、裁剪、掩膜、单位转换和带来源记录的写出。跨年文件可显式采用 `metadata_policy="preserve_sources"`，分别保留各文件的处理历史与描述范围，同时严格校验单位、网格和历法。结构与进度见 [架构与代码树](docs/preprocessing/architecture.md)、[官方库调研（含 Meteostat）](docs/preprocessing/upstream-references.md) 和 [逐项实施计划](docs/preprocessing/implementation-plan.md)。后续后端继续分期建设；新增范围不包含统计聚合或诊断，现有 API 保持兼容。

Preprocessing is now at **0.3.0**, adding physical pressure normalization, exact expected-time checks/selection and explicit boolean masks, alongside lazy xarray/Dask NetCDF reading, coordinate normalization, subsetting, masking, unit conversion and provenance-aware output. Explicit `metadata_policy="preserve_sources"` reconciles per-file histories and descriptive ranges while retaining strict unit, grid and calendar checks. The [architecture](docs/preprocessing/architecture.md), [upstream research including Meteostat](docs/preprocessing/upstream-references.md), and [implementation plan](docs/preprocessing/implementation-plan.md) track development. Later backends remain planned; no new statistical aggregation is introduced, and legacy APIs remain compatible.

0.3.0 已通过本地源码、树外安装包和 Linux HPC 环境的各 283 项测试。压力/时间/掩膜与原拼接实测合计 12 项通过、5 项预期拒绝，无失败。真实 ERA5 V 的 2003、2004 年原始文件已验证可惰性拼接为 24 个月；两个小范围月份样本与参考值完全一致，原文件保持不变。完整全场处理和性能加速比未作验证，详见 [验证记录](Preprocessing/validation.json) 与 [变更说明](Preprocessing/CHANGELOG.md)。

Version 0.3.0 passed 283 tests each for the local source, an installed wheel outside the source tree, and the Linux HPC installation. Pressure/time/mask and merge checks recorded 12 passes and 5 expected rejections, with no failures. Original ERA5 V files for 2003 and 2004 formed a lazy 24-month dataset; two small monthly samples matched reference values exactly, with sources unchanged. Full-field processing and speedup are not benchmarked; see the [validation record](Preprocessing/validation.json) and [changelog](Preprocessing/CHANGELOG.md).

本轮提供 [CI 工作流模板](docs/preprocessing/preprocessing-ci.yml.example)，尚未启用 GitHub Actions，也没有远端 CI 运行结果。需要自动检查时，将模板复制到 `.github/workflows/preprocessing.yml` 后提交。

A [CI workflow template](docs/preprocessing/preprocessing-ci.yml.example) is provided; GitHub Actions is not enabled by this update, and no remote CI result is claimed. To enable automated checks, copy the template to `.github/workflows/preprocessing.yml` and commit it.

## Getting Started / 使用

进入所需模块，按说明安装依赖，修改配置并提供自己的数据。基本环境为 Python 3.11+；统计模块使用 NumPy/SciPy，前处理模块使用 xarray/Dask/NumPy/netCDF4/cftime。球谐风场另需 windspharm 及其编译后端；LBM 工具还需相应模式参考文件。

Enter the desired module, install its dependencies, and configure your own data following its guide. Python 3.11+ is required; statistics uses NumPy/SciPy, and preprocessing uses xarray/Dask/NumPy/netCDF4/cftime. Spherical winds additionally require windspharm and its compiled backend; LBM tools need model reference files.

本仓库提供可复用代码，不分发真实数据或模型参考文件，也不自动运行 LBM 积分。

The repository provides reusable code. Real datasets and model reference files are not bundled, and the tools do not automatically run LBM integrations.

## Credits and References / 署名与引用

**Tool organization and refactoring / 工具整理与重构：RH_Deng；Codex（OpenAI）。**

原 LBM 模式作者、方法论文及 HanXiang0721 社区使用指南的归属与引用，见 [References and Contributions / 参考与贡献说明](LBM/REFERENCES.md)。

Credit the original LBM developers and the community guide separately from these preparation tools; see the linked references for details.

波活动通量、波源、球谐计算与质量流函数的原始论文和软件归属见 [Dynamics References / 动力诊断参考文献](Dynamics/REFERENCES.md)；统计方法参考见 [Statistical](Statistical/README.md)。计算代码和动力诊断输出元数据同样保留相应方法引用。

Original papers and software credits for wave flux, wave sources, spherical computations, and mass streamfunctions are listed in Dynamics References. Statistical method references are included in its module guide; citations are also retained in calculation code and dynamics output metadata.
