# Climate-Base

面向气候研究的可配置 Python 数据准备工具集。目前提供基于再分析资料的 **LBM T42L20 干模式背景态与热源强迫制作**，包含配置示例、测试和中英文说明。

Configurable Python data-preparation tools for climate research. Current modules prepare **basic states and thermal forcing for dry T42L20 LBM experiments** from reanalysis data, with configuration examples, tests, and bilingual documentation.

## Modules / 模块

| Directory / 目录 | Purpose / 用途 | Documentation / 说明 |
| --- | --- | --- |
| [LBM/Basic](LBM/Basic/) | Reanalysis climatology → model basic state / 再分析气候场 → 模式背景态 | [中文](LBM/Basic/README.md) · [English](LBM/Basic/README_EN.md) |
| [LBM/Forcing](LBM/Forcing/) | Apparent-heating diagnosis, optional regression, and thermal-forcing conversion / 视热源诊断、可选回归与热强迫转换 | [中文](LBM/Forcing/README.md) · [English](LBM/Forcing/README_EN.md) |

See the [LBM module overview / LBM 模块总览](LBM/README.md) for the directory structure.

## Getting Started / 使用

进入所需模块，安装其 `requirements.txt` 中的依赖，并按说明修改配置、提供再分析资料及相应模型参考文件。基本环境为 Python 3.11+、NumPy、SciPy 和 netCDF4。

Enter the desired module, install its requirements, and configure your own data and model references following the module guide. The core environment requires Python 3.11+, NumPy, SciPy, and netCDF4.

本仓库提供输入准备工具，不分发真实数据或模型参考文件，也不自动运行 LBM 积分。

This repository provides input-preparation tools. Real datasets and model reference files are not bundled, and the tools do not automatically run LBM integrations.

## Credits and References / 署名与引用

**Tool organization and refactoring / 工具整理与重构：RH_Deng；Codex（OpenAI）。**

原 LBM 模式作者、方法论文及 HanXiang0721 社区使用指南的归属与引用，见 [References and Contributions / 参考与贡献说明](LBM/REFERENCES.md)。

Credit the original LBM developers and the community guide separately from these preparation tools; see the linked references for details.
