# 再分析热源诊断与 LBM 热强迫制作

[English](README_EN.md)

**本仓库工具整理与重构：RH_Deng；Codex（OpenAI，AI辅助整理与重构）**
整理日期：2026-10-25

本工具提供两个可独立使用的入口：`diagnose` 从再分析月资料诊断逐气压层视热源 Q1，并取季节气候平均或作可选预测量回归；`convert` 将已有的逐气压层加热场映射到 **LBM T42L20**，生成模型热强迫输入。已有合适加热场时可直接使用 `convert`。

本包制作输入文件，不运行 LBM、不提交 Slurm 作业。代码自包含，不需安装相邻的背景制作包；转换时仍须提供实际实验使用的模型背景与 σ 轴。原模型和社区资料的归属见 [参考与署名](../REFERENCES.md)，原代码来源见 [来源说明](docs/SOURCE_PROVENANCE.md)。LBM 模式不是本包作者原创。

## 1. 环境与合成示例

需要 Python 3.11+、`numpy`、`scipy`、`netCDF4`。在本目录使用自己的解释器：

```bash
cd LBM/Forcing
python -m pip install -r requirements.txt
python -m lbm_forcing --help
```

不需要模型可执行程序、Fortran 编译器、NCL、CDO 或调度器。路径中的 `python` 应按自己的环境替换；本地运行应遵守指定解释器的约定。

先生成完全合成的示例，按顺序诊断再转换：

```bash
python tools/make_demo.py --directory /path/to/new_demo
python -m lbm_forcing diagnose --config /path/to/new_demo/diagnose.json --check
python -m lbm_forcing diagnose --config /path/to/new_demo/diagnose.json
python -m lbm_forcing convert --config /path/to/new_demo/convert.json --check
python -m lbm_forcing convert --config /path/to/new_demo/convert.json
```

示例目录和正式输出目录均应全新。**合成资料、合成模型背景及生成的强迫只用于软件测试，不能作为科学背景或正式模式试验输入。**

## 2. 从月资料诊断热源

复制 [MERRA-2 诊断配置](examples/diagnose_merra2.json)，替换 `/path/to/...` 路径和实际变量名：

```bash
python -m lbm_forcing diagnose --config examples/diagnose_merra2.json --check
python -m lbm_forcing diagnose --config examples/diagnose_merra2.json
```

### 输入与配置

| 配置项 | 含义 |
| --- | --- |
| `name`、`source_product` | 安全文件名前缀及真实资料产品标识 |
| `data_root` | 资料根目录，用于 `{data_root}` 文件模板 |
| `output` | 全新输出目录；相对路径以配置文件所在目录为基准 |
| `period` | `start_year`、`end_year` 与按顺序连续的 `months` |
| `grid.step_degrees` | 全球规则诊断网格间隔，示例为 2.5° |
| `coordinates` | `time`、`latitude`、`longitude`、`pressure` 对应的源坐标名 |
| `fields` | T、U、V、OMEGA、PS 各自的 `files` 模板与 `variable` |
| `integrity` | 默认计算输入 SHA256；`require_receipts: false` 不强制下载器收据，已有收据仍校验 |
| `regression` | `null` 为季节 Q1 气候平均；对象配置启用 CSV 预测量回归 |

T 为 K 或可识别摄氏度单位；U/V 为 m s⁻¹；OMEGA 为 **dp/dt**，使用 Pa s⁻¹ 或 hPa s⁻¹，正值向下，不能用几何垂直速度 m s⁻¹ 替代；PS 和气压轴可为 Pa 或 hPa。程序将诊断输入统一为 K、m s⁻¹、Pa s⁻¹ 和 Pa。PS 必须来自同一再分析产品的同时期大气地面气压，不能用另一类陆面产品代替。

输入是全球、可用一维经纬坐标表示的气压层 CF NetCDF，可按月存放，也可使用年度多记录文件。程序依据实际 CF 时间选择月份，不根据文件名推断时间；缺月、重复目标记录、坐标或单位不兼容会报错。气压层字段需使用一致的物理气压层，至少有三层以支持差分；不直接读取 hybrid/model-level、曲线网格或 GRIB。经纬度单位须为程序识别的 degrees_north/degrees_east 等价写法。

时间先按各文件原 CF 日历解码。对于选中年份不早于 1583 年的记录，`standard`、`gregorian`、`proleptic_gregorian` 按等价的现代格里高利日历兼容处理；更早的不同日历不混用。`360_day`、`noleap`、`all_leap` 等日历保持各自定义，不兼容的混用会报错，质量报告保留源日历标签。

文件模板支持 `{data_root}`、`{year}`、`{month}`、`{yyyymm}`；可用 `{month:02d}` 格式化月份。MERRA-2 示例中 OMEGA 位于 `Omega` 目录，PS 位于 `SP` 目录；实际命名不同时修改模板。

### 季节年与相邻月份

季节按**最后一个月份所在年份**标记。DJF 1991 是 1990 年 12 月加 1991 年 1、2 月。诊断每个月的时间导数还需要前后月 T，且这些相邻 T 各自需要同月 PS。因此每个 DJF 的温度和 PS 通常需要 **11–3 月**，U/V/OMEGA 则需要所选的 12–2 月。`--check` 会列出所需月份。

先用完整月场诊断每月 Q1，再要求季节各月有效并作**等月权平均**，之后进行跨季节气候平均或回归。不是先取多年气候场再代入非线性项，也不是天数加权。时间差分使用源 CF 日历中实际月中点的秒间隔，包含闰年或相应日历的月长差异。

### 热力方程与掩膜

```text
Q1/Cp = dT/dt + u/(a cosφ) dT/dλ + v/a dT/dφ
        + omega [dT/dp - (Rd/Cp) T/p]
Cp = 1004 J kg-1 K-1; Rd = 287.05 J kg-1 K-1; a = 6371000 m
```

Q1 的能量单位为 W kg⁻¹，正值表示加热；乘以 `86400/Cp` 得到 K day⁻¹。保持气压层分辨率，**不作垂直积分或层平均**。

每个月先将其自身 PS 映射到原大气格点并掩去地下值，然后严格双线性插值到诊断网格。任一正权重角点缺测则结果缺测，不对缺角权重重新归一。相邻月 T 使用各自月份的 PS。空间和气压差分使用连续的三点模板，不跨缺测邻点跳跃；经度周期连接，极点缺测。各热源项共用有效掩膜，避免用不同样本拼出 Q1。

月均资料诊断的 Q1 不能恢复未解析的次月尺度瞬变热输送协方差。输出的 `ps_mean` 为所选季节月份的 PS 均值，`ps_min` 为包含相邻温度月份在内的 PS 最小值，用于地下排除和资料记录；二者**不能代替转换阶段模型背景的 PS**。

## 3. 可选预测量回归

将 `regression: null` 替换为如下对象：

```json
{
  "file": "/path/to/predictor.csv",
  "year_column": "year",
  "predictor_column": "predictor",
  "control_column": "control",
  "predictor_units": "original supplied unit",
  "min_years": 10,
  "max_lag": 3
}
```

CSV 每个配置季节年一行，包含整数年份和原尺度预测量；使用控制量时还需对应控制列。`control_column: null` 可关闭控制量。表中的年份必须与季节年定义一致；按程序要求提供完整、有限的预测量/控制记录。

各格点在自身有限配对年份上拟合：

```text
heating = intercept + actual_year [+ control] + beta × predictor
```

默认最少 10 个有效季节。预测量残差**不重新标准化**，beta 表示每 **1 个输入预测量单位**的响应。若希望“每一个标准差”的响应，应自行提供明确标准化的预测量并准确标注单位；程序不自动归一振幅。

统计包含原样本量及有效样本量，Bartlett 自相关按实际年距 `1..max_lag` 配对，默认 1–3 年；名义和有效自由度随是否含控制量分别扣除 3 或 4。常值响应的真实零系数和很小的系数保留，相关或 p 值仍可能无定义。p 值没有多重比较校正，也不会用来筛选或清零强迫。

## 4. 转换为 LBM 热强迫

复制 [压力层转换配置](examples/convert_pressure.json)：

```bash
python -m lbm_forcing convert --config examples/convert_pressure.json --check
python -m lbm_forcing convert --config examples/convert_pressure.json
```

| 配置项 | 含义 |
| --- | --- |
| `name`、`output` | 安全文件名前缀与全新输出目录 |
| `input.file`、`variable` | 一个静态三维压力层加热场的文件和显式变量名 |
| `input.coordinates` | 压力、纬度、经度坐标名；示例为 pressure_hpa/lat/lon |
| `input.units_override` | 可选的显式单位说明；仅在准确了解源场单位时使用 |
| `model.background` | 实际实验将使用的 T42L20 原生背景文件，提供其 PS |
| `model.sigma` | 同一套模型的精确 `GTAXLOC.CSIG20` 轴 |
| `region` | south/north/west/east 边界及向区域内部延伸的过渡宽度 width，单位为度 |
| `amplitude_multiplier` | 显式振幅倍数，默认 1，不作额外峰值归一 |

输入须为静态 `pressure × latitude × longitude` 场，压力正值、单位 Pa/hPa，升降序会归一；水平为全球一维经纬网格。接受 K day⁻¹、K s⁻¹ 或 W kg⁻¹，使用 Cp=1004 统一换算。压力积分标记、列积分通量或不兼容单位会被拒绝；不能把 W m⁻² 直接当成逐层加热。预测量尺度以独立元数据保留，变量名本身不决定物理单位。

`diagnosed_heating.nc` 中的 `heating_k_day` 是诊断所选的季节气候场或回归 beta，可直接转换。输入变量不会自动猜测；有额外时间维时，应先明确选择要制作的一张静态场。

模型背景必须是**实际积分所用的那套背景态**，目标压力为 `sigma × model_background_PS`。可使用背景制作工具的输出；编码模板不是已制作的模型背景，不能随意拿模板替代。转换不需要模型地形文件或真实热强迫模板，也不修改背景文件。

转换阶段先在源压力层上作缺测感知的周期双线性映射，这一步允许对有效角点权重重新归一；随后每根廓线排除缺测层，在 `ln(p)` 中作自然三次样条，范围外使用最近有效端点的常值。它与诊断阶段严格缺角掩膜的规则不同。

最后在 σ 网格乘以区域过渡窗并转换为 K s⁻¹。默认区域为 45°S–15°N、30°E–120°E，边界向内 5° 为余弦过渡；边界和区域外为精确零。保留加热和冷却符号，不作垂直积分、显著性筛选或默认振幅归一；**过渡窗后不作全球谱滤波**，以保持区域外的精确零。

## 5. 两种检查模式

| 命令 | 读取内容 | 写输出 |
| --- | --- | --- |
| `diagnose --check` | 配置并计算所需月份，不读取原再分析场 | 否 |
| `convert --check` | 输入加热场、模型背景与 σ 轴，检查区域及兼容性 | 否 |

检查通过仍不代表真实热源已完成科学验收、模型已成功读入或积分稳定。正式构建不覆盖现有输出目录；输入错误和数值检查失败应根据报告处理。

## 6. 输出与验证边界

`diagnose` 生成 `seasonal/<year>.nc`、`diagnosed_heating.nc`、质量/输入/输出清单和状态记录。压力层总文件保存所选 `heating_k_day`、Q1 的 K/day 与能量形式、气候场及可用的各项统计；配置回归后还保存相应 beta 和统计结果。请按字段元数据区分气候平均和每输入预测量单位的响应。

`convert` 生成 `<name>.grd`、`<name>.ctl`、`forcing.nc`、质量报告、输入/输出清单和状态记录。二进制为大端 float32，横向经度变化最快、纬度北到南、σ 层从近地面到模式顶，共 **61 个 Fortran 顺序记录**：20 层涡度、20 层散度、20 层温度、1 层地面气压。热强迫只写温度记录，其他记录为零；温度强迫单位为 **K s⁻¹**。

可运行离线测试：

```bash
python -B -m unittest discover -s tests -p 'test_*.py'
```

随包测试和合成示例检查软件、单位、掩膜、插值或文件格式；实际检查结果以分发的验证报告为准。没有将真实模式参考文件、原生产验收或合成场比较等同于新的真实资料科学验收。本工具不运行模式；正式使用仍需核对输入物理意义、端点延伸、所选背景 PS、模式读入和积分结果。

## 7. 共享与隐私

所有示例使用占位路径，不包含真实资料、模型背景、内部项目名称、账户、SSH 设置或原任务日志。运行生成的报告可能记录所提供的资料名称和位置；共享实际结果前需另行检查。作者署名描述本次代码整理与重构，不替代原 LBM 和社区资料的归属及使用条件。
