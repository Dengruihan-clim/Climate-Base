# 来源与署名 / Source Provenance and Credits

**工具整理与重构 / Tool organization and refactoring：RH_Deng；Codex（OpenAI，AI辅助整理与重构 / AI-assisted refactoring）**
核对日期 / Verified：2026-10-25

## 提取范围

本包来源于已核对的 HPC 热源制作代码，包含逐气压层视热源诊断与回归、月资料和地面气压读取、模型网格/σ 层映射、区域过渡窗及热强迫文件编码。此次只读提取了 20 份原程序、依赖、测试及说明，现有本地副本与远端源文件 SHA256 全部一致；没有执行原入口、修改远端、提交任务或运行模式。

以下记录通用逻辑来源、原文件名和整理前指纹，不包含内部项目、账户、服务器或原目录结构。包内文件经重构后可有不同哈希，应以本次分发的校验清单核对。

## Scope of extraction

The sources are verified HPC programs for pressure-resolved apparent-heating diagnosis and regression, monthly atmospheric and surface-pressure input, model-grid/sigma remapping, geographic tapering, and thermal-forcing file encoding. Twenty original programs, dependencies, tests, and notes were extracted through read-only operations. All existing local copies matched their remote source SHA256 fingerprints. No original entry point was executed, remote file modified, job submitted, or model integration run.

The table records generic source roles, original filenames, and fingerprints before refactoring. Internal projects, accounts, servers, and original directory structures are omitted. Refactored files may have different hashes; use the checksum manifest supplied with this distribution for the packaged files.

## 关键原文件 / Key original files

| 逻辑来源 / Source role | 原文件 / Original file | SHA256 |
| --- | --- | --- |
| 逐层热源诊断入口 / Pressure-resolved diagnosis entry | `stage4_DJF_lbm_forcing.py` | `d2fa18b0759b538d3b0d039da7b51a15fe77075c24fdbc89248893f3222759e5` |
| 诊断和回归内核 / Diagnosis and regression kernels | `lbm_forcing_math.py` | `76ae4af6aab30da7dd1f0afb8289f0cf5dc9f5ca1c7c24c4d0e2445fd2303008` |
| 原月场读取模块 / Original monthly-field reader | `lbm_forcing_io.py` | `769aca7604d07a7031b15cd7f29bf1fae388885263dd7f9988db382990ed7e69` |
| 模型热强迫转换入口 / Model thermal-forcing conversion entry | `stage4_DJF_Q1toLBMforcing.py` | `5cbd7c04721582f81310a523f7789c153a1a931a32f22a039ebd4ae7c166d30a` |
| 过渡窗和编码内核 / Taper and encoding kernels | `q1_to_lbm_math.py` | `855fa0a24c9705fc3723f0aa430baf4ea551b44f34c2b6babc807e4e981da483` |
| 原热力学数值支持 / Original thermodynamic numerical support | `heating_math.py` | `0dd29bb22df1783714e867fc7a668f988e4e2ca6ce7dd69e010b4143af4f2aba` |
| 原诊断内核测试 / Original diagnosis-kernel tests | `test_lbm_forcing_math.py` | `e064aedead6848dc3429ec0c99caca27bc4bed3007c0ee09b4bebc972efc984d` |
| 原转换与格式测试 / Original conversion and format tests | `test_q1_to_lbm_math.py` | `c98903a6bcffd9d43846feb14f9f9391e4ab8aa823fdf972d16635d127147ab4` |

## 保留方法与重构边界

保留的方法包括：逐月 Q1 热力方程、各月独立地面气压掩膜、实际 CF 日历月中点的不等间隔时间导数、连续差分模板、等月权季节平均、可选原尺度预测量回归、实际年距 Bartlett 推断、缺测感知水平映射、自然 `ln(p)` 样条与常值端点、最终区域过渡窗，以及 61 记录的大端热强迫格式。

新接口将输入变量、资料路径、季节时段、可选 CSV、输出和模式背景选择移入配置。原固定预测量、时段、区域和运行状态不绑定新入口；区域可配置，默认振幅倍数为 1。输入资料和模型背景需要使用者提供，实际数据不随包分发。

仅用于私有集成检查的真实模式背景与 σ 轴通过指纹核对；它们不属于代码分发内容。合成场测试、原程序数值对比或二进制回读只能验证软件及相应计算行为，不能替代真实资料的科学验收、模式读入或积分稳定性检查。原程序的历史生产结果不作为本包验收结果。

## Preserved methods and refactoring boundaries

Retained methods include the monthly Q1 thermodynamic equation, independent surface-pressure masks for each month, unequal-interval time derivatives at actual CF-calendar month midpoints, contiguous finite-difference stencils, equal-month seasonal means, optional regression in the original predictor scale, actual-year-lag Bartlett inference, missing-aware horizontal mapping, natural splines in `ln(p)` with constant endpoint extension, a final geographic taper, and the 61-record big-endian thermal-forcing format.

The new interface configures input variables, data paths, season periods, an optional CSV, outputs, and the model background. It is not tied to the original predictor, period, region, or run state. The region is configurable, and the default amplitude multiplier is 1. Users supply the input data and model background; real data are not distributed.

The real model background and sigma axis used for private integration checks were verified by fingerprints and are not included in the code distribution. Synthetic-field tests, numerical comparisons against the original programs, and binary round trips check software and the corresponding calculations. They do not replace scientific validation with real data, model read-in, or integration-stability checks. Historical production results from the original programs are not validation results for this package.

## 模型与社区资料 / Model and community references

原 LBM 模式及社区说明的作者归属、参考文献和使用条件见 [统一参考与署名说明](../../REFERENCES.md)。本包作者署名描述代码提取、接口整理及重构，不表示 LBM 模式由本包作者原创。

See the [shared references and credits](../../REFERENCES.md) for the original LBM model, community documentation, attribution, and terms of use. The authorship here describes source extraction, interface organization, and refactoring; it does not claim authorship of the LBM model itself.
