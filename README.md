# Climate-Base

面向气候研究的可配置 Python 前处理与统计工具集。提供再分析月资料读取、完整季节平均、统计回归和趋势分析，以及 **LBM T42L20 干模式背景态与热源强迫制作**。各模块包含示例、测试和中英文说明。

Configurable Python preprocessing and statistical tools for climate research: monthly reanalysis reading, complete seasonal means, field regression, trend analysis, and **basic states and thermal forcing for dry T42L20 LBM experiments**. Modules include examples, tests, and bilingual documentation.

## Modules / 模块

| Directory / 目录 | Purpose / 用途 | Documentation / 说明 |
| --- | --- | --- |
| [Preprocessing](Preprocessing/) | CF monthly readers, coordinate/unit checks, calendar-aware seasonal means / CF 月资料读取、坐标与单位检查、日历感知季节平均 | [中文](Preprocessing/README.md) · [English](Preprocessing/README_EN.md) |
| [Statistical](Statistical/) | Year alignment, controlled regression, trends, bootstrap and FDR / 年份配对、控制回归、趋势、bootstrap 与 FDR | [中文](Statistical/README.md) · [English](Statistical/README_EN.md) |
| [LBM/Basic](LBM/Basic/) | Reanalysis climatology → model basic state / 再分析气候场 → 模式背景态 | [中文](LBM/Basic/README.md) · [English](LBM/Basic/README_EN.md) |
| [LBM/Forcing](LBM/Forcing/) | Apparent-heating diagnosis, optional regression, and thermal-forcing conversion / 视热源诊断、可选回归与热强迫转换 | [中文](LBM/Forcing/README.md) · [English](LBM/Forcing/README_EN.md) |

See the [LBM module overview / LBM 模块总览](LBM/README.md) for the directory structure.

## Getting Started / 使用

进入所需模块，按说明安装依赖，修改配置并提供自己的数据。基本环境为 Python 3.11+；统计模块使用 NumPy/SciPy，前处理模块使用 NumPy/netCDF4/cftime。LBM 工具还需提供相应模式参考文件。

Enter the desired module, install its dependencies, and configure your own data following its guide. Python 3.11+ is required; statistics uses NumPy/SciPy, and preprocessing uses NumPy/netCDF4/cftime. LBM tools also require model reference files.

本仓库提供可复用代码，不分发真实数据或模型参考文件，也不自动运行 LBM 积分。

The repository provides reusable code. Real datasets and model reference files are not bundled, and the tools do not automatically run LBM integrations.

## Credits and References / 署名与引用

**Tool organization and refactoring / 工具整理与重构：RH_Deng；Codex（OpenAI）。**

原 LBM 模式作者、方法论文及 HanXiang0721 社区使用指南的归属与引用，见 [References and Contributions / 参考与贡献说明](LBM/REFERENCES.md)。

Credit the original LBM developers and the community guide separately from these preparation tools; see the linked references for details.
