# climate_indices 上游源码快照

[English](README_EN.md) · [来源与许可说明](reference/README.md) · [机器可读来源记录](reference/UPSTREAM.json)

本目录收录 [monocongo/climate_indices](https://github.com/monocongo/climate_indices) 的一个固定源码快照，便于在同一仓库查阅和复用气候指数工具。上游提供 SPI、SPEI、PET、Palmer 系列等计算，以及其他干旱、火险、湿润极端和能源相关指标。算法、代码和原始文档归上游作者及贡献者所有；本仓库新增的是收录与来源说明。

## 目录

| 路径 | 内容 |
| --- | --- |
| [`upstream/`](upstream/) | 固定提交的完整、未修改 Git 跟踪文件快照 |
| [`upstream/README.md`](upstream/README.md) | 上游安装、功能和使用文档 |
| [`upstream/LICENSE`](upstream/LICENSE) | 完整许可证、历史声明及第三方 notices |
| [`upstream/CITATION.cff`](upstream/CITATION.cff) | 上游软件引用元数据 |
| [`upstream/DISCLAIMER`](upstream/DISCLAIMER) | 上游免责声明 |
| [`reference/README.md`](reference/README.md) | 中英文来源、归属和维护说明 |
| [`reference/UPSTREAM.json`](reference/UPSTREAM.json) | 固定提交和导入范围的机器可读记录 |
| [`reference/SHA256SUMS.txt`](reference/SHA256SUMS.txt) | 上游快照逐文件 SHA256 校验清单 |

## 固定版本

| 项目 | 记录 |
| --- | --- |
| 上游仓库 | <https://github.com/monocongo/climate_indices> |
| 来源分支 | `main`，仅表示本次取样来源 |
| 提交 | [`e6b4a5098a3f6218e933ccf587e4032122e45e4c`](https://github.com/monocongo/climate_indices/commit/e6b4a5098a3f6218e933ccf587e4032122e45e4c) |
| Git tree | `fa30400b32d1ddcab1ece3fa2885c29df8d9dd51` |
| 上游提交时间 | `2026-10-26T05:58:04Z` |
| 导入日期 | `2026-10-26`，Asia/Shanghai |
| 上游版本字段 | `3.1.0`，取自该提交的项目配置与引用文件 |
| 快照范围 | 6,113 个 Git 跟踪文件，97,829,440 字节 |

这是固定提交的开发快照，不是对某个稳定发行版的承诺。该提交的上游 README 明确说明其 `3.1.0` 尚未发布到 PyPI；直接安装 PyPI 软件包不等于安装这里的固定提交。此处不包含 Git 历史，也不是会自动同步的 fork 或 Git submodule。

## 安装与使用

该快照声明支持 Python `>=3.10,<3.15`。在适配的独立环境中，进入本目录后安装本地源码：

```bash
python -m pip install ./upstream
python -m climate_indices --help
```

依赖以 [`upstream/pyproject.toml`](upstream/pyproject.toml) 为准。这个提交使用 `hatchling.build` 构建常规 Python 包，Rust 加速扩展是可选项；普通源码安装不要求 Rust 工具链。需要主动构建可选扩展时，按该快照的上游架构文档配置，不要将其与普通安装混为一谈。

具体输入变量、单位、时间轴、校准期和缺测处理应依据 [上游文档](upstream/docs/) 与相应方法要求设置。在线文档地址为 <https://climate-indices.readthedocs.io/en/latest/>，其内容可能随上游更新；复现实验时优先核对本快照中的文档。

本次收录核对源码来源和快照完整性，未运行上游完整测试套件，也未重新验证各指数的科学结果。[`upstream/VALIDATION.md`](upstream/VALIDATION.md) 是保留的上游验证说明，不是本仓库新增的运行报告。

## 归属、引用和更新

上游使用 BSD-3-Clause 许可证，完整的版权、许可条件、免责声明和第三方 notices 均保留在 [`upstream/LICENSE`](upstream/LICENSE)。研究中使用本软件时，应按 [`upstream/CITATION.cff`](upstream/CITATION.cff) 引用上游，并记录实际使用的固定提交；本仓库维护者不替代上游软件作者。

今后更新需要显式选择新的上游提交，重新导入并核对文件，更新 `reference/UPSTREAM.json`、校验清单和本页版本记录。此目录不会自动跟踪上游 `main`。若以后修改上游代码，应单独记录修改内容和所依据的上游提交，不能继续将修改后的目录称为未修改快照。
