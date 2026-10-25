# LBM 参考与贡献说明 / References and Contributions

## 原模式与方法 / Original model and formulation

原 LBM 模式与其线性方程体系应归功于原开发工作，并引用 **M. Watanabe 与 M. Kimoto（2000，2001）**。[官方开发说明](https://ccsr.aori.u-tokyo.ac.jp/~lbm/sub/lbm_2.html)介绍，模式基于东京大学气候系统研究中心（CCSR）与日本国立环境研究所（NIES）共同开发的 **CCSR/NIES AGCM 5.4g** 动力核，在给定基本态附近对 σ 坐标原始方程显式线性化。

Credit the original LBM development and linear-equation formulation described by **M. Watanabe and M. Kimoto (2000, 2001)**. The official development page traces the model to the **CCSR/NIES AGCM version 5.4g** dynamical core, jointly developed at CCSR, University of Tokyo, and NIES. The sigma-coordinate primitive equations are explicitly linearized about a basic state.

建议使用 LBM 开展科研实验时同时引用以下方法论文及其勘误，并另行引用实际使用的再分析资料。/ For LBM research, cite the methods paper and its corrigendum below, together with the datasets actually used.

1. Watanabe, M., and M. Kimoto, 2000: **Atmosphere–ocean thermal coupling in the North Atlantic: A positive feedback.** *Quarterly Journal of the Royal Meteorological Society*, **126**(570), 3343–3369. [https://doi.org/10.1002/qj.49712657017](https://doi.org/10.1002/qj.49712657017).
2. Watanabe, M., and M. Kimoto, 2001: **Corrigendum.** *Quarterly Journal of the Royal Meteorological Society*, **127**(572), 733–734. [https://doi.org/10.1002/qj.49712757223](https://doi.org/10.1002/qj.49712757223).

官方资料 / Official resources:

- [LBM website / 官方主页](https://ccsr.aori.u-tokyo.ac.jp/~lbm/sub/lbm.html)
- [Development / 开发说明](https://ccsr.aori.u-tokyo.ac.jp/~lbm/sub/lbm_2.html)
- [Publications using LBM / 官方参考列表](https://ccsr.aori.u-tokyo.ac.jp/~lbm/sub/lbm_5.html)

这些引用对应原模式及方法，不将本仓库的干模式输入准备工具解释为湿 LBM 实现。实际采用湿模式扩展时，应核对相应模型版本并补充该扩展的原始方法引用。/ These references credit the original model and formulation. The dry-model input preparation tools in this repository do not constitute a moist LBM implementation. If a moist extension is actually used, verify its version and cite its own original methods.

## 社区流程指南 / Community workflow guide

**HanXiang0721（GitHub 账号 / GitHub handle）**，`HanXiang0721/Note-Numerical-Methods` 仓库，**Linear-Baroclinic-Model使用指南.md**。该资源提供社区使用与自制背景场流程参考；它与原 LBM 模式开发工作分别归属。/ This community resource describes model use and custom basic-state preparation. Credit its contributors separately from the original LBM developers.

- [Repository / 仓库](https://github.com/HanXiang0721/Note-Numerical-Methods)
- [Guide content / 指南文件内容](https://github.com/HanXiang0721/Note-Numerical-Methods/blob/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)
- [User-specified commit history / 用户指定的提交历史](https://github.com/HanXiang0721/Note-Numerical-Methods/commits/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

以上文件链接指向可变的 `master` 分支。复现实验时应从提交历史记录所用修订；此处不虚构固定提交号、真实姓名或出版年份。/ The file links point to the mutable `master` branch. For reproducibility, record the revision used from its commit history. No unverified commit, personal name, or publication year is assigned here.

本次于 2026-10-25 经 GitHub 文件历史 API 核对，指南文件最近一次修改对应修订 [`7321a5fcf8b48b726402485213ca1f19b8f50fb6`](https://github.com/HanXiang0721/Note-Numerical-Methods/commit/7321a5fcf8b48b726402485213ca1f19b8f50fb6)，提交时间为 2018-09-25。/ The file-history API was checked on 2026-10-25: the guide's latest file-changing revision was `7321a5fcf8b48b726402485213ca1f19b8f50fb6`, committed on 2018-09-25.

## 本仓库工具 / Repository tools

**工具整理与重构作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。** 工作范围为本仓库 Python 数据准备工具的整理、配置接口、重构、测试和中英文说明。该署名不表示对原 LBM 模式、社区指南或再分析资料的作者身份。

**Tool organization and refactoring: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).** This credit covers the repository's Python preparation tools, configuration interfaces, refactoring, tests, and bilingual documentation. It does not establish authorship of the original LBM, the community guide, or reanalysis products.

如需说明工具来源，可引用本仓库 [Climate-Base](https://github.com/Dengruihan-clim/Climate-Base) 及实际使用的提交号，并在方法部分保留上述模型与资料引用。本说明不新增、替代或改变原模型、社区指南、资料及其他第三方代码的许可或使用条件。

To identify the preparation tools, cite the [Climate-Base repository](https://github.com/Dengruihan-clim/Climate-Base) and the revision actually used, while retaining the model and dataset references in the methods. This note does not create, replace, or change any license or usage conditions for the model, guide, datasets, or other third-party code.
