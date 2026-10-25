# Statistical：统计回归与趋势分析

[English](README_EN.md)

面向年序列和季节场的独立 Python 模块。提供真实年份配对、逐格点控制回归、趋势估计、循环残差块 bootstrap、lag-1 有效样本量诊断及 Benjamini–Hochberg FDR。仅依赖 NumPy 和 SciPy；输入资料、研究区域和预测量由使用者提供。

**作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。**

## 安装与运行

需要 Python 3.11+。在本目录运行：

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
```

无需真实资料的完整例子：

```bash
python examples/generate_synthetic.py --output-dir ./demo_input
python examples/regression_csv.py --config ./demo_input/config.json
```

第一步创建合成 CSV 和可运行配置；第二步输出 JSON。目标目录和输出文件均拒绝覆盖。自己的数据可参照 [配置模板](examples/regression.json)，将 `<DATA_ROOT>`、`<OUTPUT_ROOT>` 替换为自己的路径。CSV 的首行为列名，年份必须是无重复、严格递增的整数；空白、`NA`、`NaN`、`null` 作为缺测。FDR 的检验族是配置中列出的全部响应列。

## Python 接口

```python
from climate_statistics import align_years, linear_regression, benjamini_hochberg

# predictor、field、control 均为 time-first；三组年份可不同。
years, (x, y, z) = align_years(
    (predictor_years, predictor),
    (field_years, field),
    (control_years, control),
)
result = linear_regression(x, y, years, controls=z,
                           include_trend=True, inference="ols", min_samples=15)
fdr = benjamini_hochberg(result["p"], alpha=0.05)
beta = result["beta"]
q_values = fdr["q_values"]
```

`field` 形状为 `(time, *space)`，也支持单序列 `(time,)`。`predictor` 为 `(time,)`；控制量可为 `(time,)` 或 `(time, n_controls)`。各变量应先按科学定义形成同一频率的年或季节序列；可用 [Preprocessing](../Preprocessing/README.md) 生成完整季节场。

| 函数 | 用途及主要返回值 |
| --- | --- |
| `align_years(*series)` | 每项为 `(years, values)`；返回共同年份及按这些年份选取的数组，不填补缺年 |
| `linear_regression(x, field, years, controls=None, ...)` | 含截距、可选真实年趋势和控制量的 OLS；`beta`、`control_beta`、`trend_beta`、偏相关 `r`、`p`、`p_nominal`、`n`、`n_eff`、`df` |
| `lag1_bartlett(x, y, years, ...)` | 对已定义的两条序列计算实际相邻年的 lag-1 相关、近似有效样本量及 t 检验 |
| `benjamini_hochberg(p_values, alpha=.05)` | 在全部有限 p 值上校正；返回 `q_values`、`reject`、`n_tests` |
| `ols_trend(field, years, ...)` | 逐格点 OLS 趋势、名义标准误与置信区间 |
| `sen_trend(field, years, ...)` | Theil–Sen 斜率、SciPy 斜率区间与名义 Kendall 检验 |
| `circular_block_bootstrap(field, years, ...)` | 保持固定有效空间支撑，配对重采样趋势残差的循环块，返回斜率分布和百分位区间 |

趋势使用示例：

```python
from climate_statistics import ols_trend, sen_trend, circular_block_bootstrap

ols = ols_trend(field, years, min_samples=15)
sen = sen_trend(field, years, min_samples=15, confidence=0.95)
boot = circular_block_bootstrap(
    field, years, method="sen", n_resamples=1000,
    block_length=3, seed=42, confidence=0.90,
)
```

## 科学约定

- 回归在每个格点使用预测量、响应、控制量共同有限的年份。设计矩阵秩不足或样本不足时保留计数，估计量为 NaN。`align_years` 只取年份交集，不替代逐格点缺测筛选。
- 系数保留**输入预测量的尺度**，不会对子样本重新标准化。若输入是先在完整参考期标准化的指数，系数就是每一个该参考期标准差的响应。
- 年趋势按真实年距离拟合。所有趋势的 `slope`、区间和 bootstrap 分布以“输入量单位/年”为单位；`slope_per_decade` 为其十倍。名义 OLS 区间、Sen 区间和 Kendall p 均不自动修正时间相关。Sen 区间的 `confidence` 限于 `[0.5, 1)`，以避免 SciPy 对低于 0.5 的 `alpha` 作对称解释；OLS 和 bootstrap 的区间水平可在 `(0, 1)` 内选择。
- `inference="ols"` 是名义线性模型推断。回归可显式选择 `"lag1_bartlett"`：先移除截距、控制量和可选趋势，再用残差序列的 lag-1 近似调整有效自由度。只配对**年份相差一年的样本**，不会把缺年两侧当相邻；诊断不足时返回 NaN。它是近似诊断，未实现 HAC 标准误。
- `stderr_nominal` 属于名义 OLS；`stderr` 在 lag-1 选项下按有效自由度作近似缩放，未定义的诊断不回退到名义值。趋势函数仅接受 `inference="ols"`；不能把确定性的年份预测量直接套入平稳序列的 lag-1 校正。
- 块 bootstrap 仅接受完整连续整数年轴；有年份缺口或输入中存在缺测时拒绝运行。使用前须显式选择全时段完整的固定空间支撑，各格点共用残差抽样索引。`method` 可选 `"sen"` 或 `"ols"`，块长、随机种子、重复次数与区间水平应明确报告。
- FDR 只校正调用时提供的有限 p 值，不会替代正确的单点推断；标准 BH 的控制结论需要独立性或适当正依赖条件。检验族应在分析前定义。

## 验证与范围

测试包含独立 OLS/SciPy 对照、偏回归等价性、缺年和局部缺测、常数与秩亏、lag-1 年份配对、FDR 以及块重采样检查。合成例子用于验证接口；不包含真实资料或研究结果，也没有验证任何具体气候机制。

本模块从已有研究代码中的通用统计内核整理而来，移除了固定研究量、路径和调度逻辑。模型输入准备仍见 [LBM](../LBM/README.md)。使用时请引用 [Climate-Base](https://github.com/Dengruihan-clim/Climate-Base) 及实际提交号，并分别引用所用资料和统计方法。本说明不新增或改变第三方库、资料与方法的许可条件。

## 方法参考

- Benjamini, Y., and Y. Hochberg, 1995: Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society, Series B*, **57**, 289–300. [DOI](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)。
- Sen, P. K., 1968: Estimates of the regression coefficient based on Kendall's tau. *Journal of the American Statistical Association*, **63**, 1379–1389. [DOI](https://doi.org/10.1080/01621459.1968.10480934)。
- [SciPy `theilslopes`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.theilslopes.html) 与 [Kendall 检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)：具体估计和名义区间的实现说明。
