# LBM

**Repository tool organization and refactoring / 本仓库工具整理与重构作者：RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring)**

Tools and documentation for the Linear Baroclinic Model (LBM).

线性斜压模式（LBM）工具与说明。`LBM` 为一级目录，各模块放在其下的二级目录中。

## Model attribution and references / 模型归属与引用

Credit the original LBM development and formulation with **Watanabe and Kimoto (2000, 2001)**. The model is based on the CCSR/NIES AGCM 5.4g dynamical core. RH_Deng and Codex are credited for this repository's tool organization and refactoring. The community workflow guide in `HanXiang0721/Note-Numerical-Methods` is a separate source.

原 LBM 模式方法应引用 **Watanabe 与 Kimoto（2000，2001）**，并按实际使用情况致谢社区流程指南与再分析资料。详见 [统一参考与贡献说明 / References and contributions](REFERENCES.md)，其中分别列明原模式、社区指南和本仓库工具的贡献归属。

## Modules / 模块

| Directory / 目录 | Purpose / 用途 | Documentation / 说明 |
| --- | --- | --- |
| [Basic](Basic/) | Build T42L20 basic states from reanalysis data / 利用再分析资料制作 T42L20 背景态 | [English](Basic/README_EN.md) · [简体中文](Basic/README.md) |
| [Forcing](Forcing/) | Diagnose reanalysis heating and build T42L20 thermal forcing / 诊断再分析加热并制作 T42L20 热源强迫 | [English](Forcing/README_EN.md) · [简体中文](Forcing/README.md) |

## Directory structure / 目录结构

```text
LBM/
├── README.md
├── REFERENCES.md
├── Basic/
│   ├── README.md
│   ├── README_EN.md
│   ├── lbm_background/
│   ├── examples/
│   ├── tests/
│   └── tools/
└── Forcing/
    ├── README.md
    ├── README_EN.md
    ├── lbm_forcing/
    ├── examples/
    ├── tests/
    └── tools/
```
