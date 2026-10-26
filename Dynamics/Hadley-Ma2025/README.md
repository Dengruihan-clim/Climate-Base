# Hadley-Ma2025：区域 Hadley 环流诊断与指数响应

[English](README_EN.md) · [参考文献](REFERENCES.bib) · [输入与运行示例](examples/README.md)

一个可独立使用的 Python 工作流，参考 Ma et al. (2025) 的区域 Hadley 环流诊断，计算局地三维质量流函数、环流水平强度、区域纬度—气压结构，以及它们对指定气候指数的线性响应。

**整理与实现：RH_Deng；Codex（OpenAI，辅助组织和重构）。** 科学方法和上游软件归原作者所有，完整引用见 [REFERENCES.bib](REFERENCES.bib)。

本目录提供通用计算代码，不附带观测数据、私有计算结果或服务器连接设置。用户需要自行准备风场与指数，并引用实际使用的数据集。

## 安装

使用 Python 3.10 或更高版本，在兼容的环境中安装科学计算与文件读写依赖：

```bash
python -m pip install -r requirements.txt
```

输入已计算好的辐散经向风 `vd` 时，无须安装球谐变换后端。输入总风 `u/v` 时，还需要真实的全球 Helmholtz 分解后端：

```bash
python -m pip install -r requirements-spectral.txt
```

可选后端为 `windspharm` 或兼容的 `skyborn.windspharm`。`windspharm` 依赖编译后的谱变换库；某些平台需要 Fortran 工具链，不能保证所有操作系统和 Python 版本都有预编译安装包。也可使用提供兼容后端的环境，再在同一环境安装本目录的基础依赖。未安装真实后端时，不能把总风直接当作辐散风。

另一种安装选择是 `python -m pip install skyborn`；本目录的真实后端验算使用 skyborn 0.4.5。运行 `python tools/validate_spectral_backend.py` 核对实际安装，验算记录见 [VALIDATION.md](VALIDATION.md)。

安装后可以用解析合成风场检验实际谱后端：

```bash
python tools/validate_spectral_backend.py
```

后端缺失或误差超出阈值时，该命令会明确失败。跳过、替代或模拟谱变换不能算作实际后端验算。

## 快速运行

使用总风进行全球 Helmholtz 分解，再计算区域环流：

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/winds.nc" --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley"
```

使用已经独立验证的辐散经向风：

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/divergent_winds.nc" --input-kind divergent --v-var vd --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley"
```

已聚合为每年一个目标季节样本的输入使用 `--frequency seasonal`；只保存计算结果可添加 `--no-plots`。查看完整参数：

```bash
python hadley_ma2025.py --help
```

将 `<DATA_ROOT>` 和 `<OUTPUT_ROOT>` 替换为自己的目录。输入与坐标要求见 [examples/README.md](examples/README.md)。

程序先选择指定风变量与气压层，再将所选时空数据读入内存；大范围、长时段输入需要足够内存。只需要某些年份时，可提前准备相应输入文件。

常用参数：

| 参数 | 用途 / 默认值 |
| --- | --- |
| `--winds` | 一个或多个 NetCDF 风场文件；多文件按坐标合并 |
| `--index-column` | 指定指数 CSV 的数值列 |
| `--input-kind total\|divergent` | 默认 `total`；`divergent` 直接使用辐散经向风 |
| `--u-var`、`--v-var` | 默认 `u`、`v`；使用 `vd` 时显式指定 `--v-var vd` |
| `--frequency monthly\|seasonal` | 默认 `monthly`；月资料按日数加权，要求完整三个月 |
| `--season` | 默认 `SON` |
| `--start-year`、`--end-year` | 限定分析年份 |
| `--baseline-start`、`--baseline-end` | 默认 `1981`、`2010` |
| `--pressure-levels` | 依次列出目标气压层，单位 hPa |
| `--pressure-units` | 显式声明输入气压单位；一般从坐标属性读取 |
| `--wind-units` | 显式声明输入风速单位，例如 `"m s-1"`；一般从变量属性读取 |
| `--regional-reduction mean\|sum` | 默认 `mean` |
| `--main-lon-min`、`--main-lon-max` | 默认 `60`、`120` |
| `--sensitivity-lon-min`、`--sensitivity-lon-max` | 默认 `70`、`110` |
| `--no-detrend-fields`、`--no-detrend-index` | 分别关闭场或指数的线性去趋势 |
| `--alpha` | 显著性阈值，默认 `0.1` |
| `--truncation` | 可选球谐三角截断阶数 |
| `--edge-target-level` | 边界诊断层，默认 `500` hPa |
| `--edge-core-lat-min`、`--edge-core-lat-max` | 环流核心搜索区间，默认 `0`、`25`°N |
| `--edge-search-lat-min`、`--edge-search-lat-max` | 核心以北的边界搜索窗口，默认 `20`、`40`°N |
| `--no-smooth-edge` | 关闭边界诊断前默认的三点纬度平滑 |
| `--no-plots` | 只保存数值结果 |
| `--overwrite` | 显式允许替换已有输出 |

## 合成示例

生成解析风场和指数，再直接使用其已知辐散风运行完整流程：

```bash
python examples/generate_synthetic.py --output-dir demo_input
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind divergent --v-var vd --index demo_input/index.csv --index-column nino34 --output-dir demo_output/divergent
```

示例生成器要求目标目录尚不存在。生成的 `winds.nc` 同时包含总风 `u/v` 和解析辐散经向风 `vd`，`index.csv` 为逐年指数。安装真实谱后端后可对同一合成场运行总风路径：

```bash
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind total --index demo_input/index.csv --index-column nino34 --output-dir demo_output/total
```

## 计算流程

1. 读取风场和指定指数，聚合目标季节并对齐共同年份。
2. 对全球总风进行 Helmholtz 分解，得到辐散经向风；也可直接读取预先计算的 `vd`。
3. 对辐散经向风作气压积分，得到局地三维质量流函数。
4. 先从每年原始流函数计算水平强度，再求强度异常并回归到指数。
5. 沿指定经度区间对原始流函数求平均或求和，获得纬度—气压剖面；计算气候态、异常及指数回归。
6. 在指定气压层和纬度搜索窗口内诊断环流边界的零交叉，并输出有效样本统计。

默认目标季节为 SON，气压层为 100、150、200、250、300、400、500、600、700、850、1000 hPa；主区间为 60°–120°E，敏感性区间为 70°–110°E，区域归约默认为经度平均。气候基准期默认 1981–2010 年。回归使用共同样本期，场和指数按设置去趋势，指数在共同样本上使用样本标准差（`ddof=1`）重新标准化，因此回归系数对应指数增加 1 个样本标准差。

## 物理定义与解释

Ma et al. (2025) 式 (3) 的理想上边界为气压 0。对于均匀全球经度网格，本实现采用

\[
\psi(t,p,\phi,\lambda)
=\frac{R\cos\phi\,\Delta\lambda}{g}
\int_{p_{\mathrm{top}}}^{p}v_d(t,p',\phi,\lambda)\,\mathrm{d}p',
\qquad \Delta\lambda=\frac{2\pi}{n_x}.
\]

气压积分使用 Pa，并用相邻层的梯形积分；输出单位为 kg s⁻¹。`p_top` 是实际提供的最高层，默认 100 hPa，流函数在该层设为零。**这表示相对于实际最高层的有限层积分，没有恢复 0–100 hPa 的未知输送。** 不施加底部残差闭合修正。

- **水平结构：** 每个经纬度、每年计算 `HC_intensity = max_level(abs(psi))`，再对其异常作回归。色标正值表示指数升高时该处强度增强，负值表示减弱。最大值和绝对值是非线性运算，必须区分这一结果与“先对流函数回归，再取最大绝对值”。
- **垂直结构：** 对局地流函数沿选定经度区间归约后，保留纬度与气压维度。填色表示指数回归系数；背景等值线表示气候态，叠加的偏移零线可用于查看气候态加 1 标准差指数响应后的零交叉变化。
- **区域 `mean` 与 `sum`：** 局地流函数已经包含单个经度格点宽度 `Delta lambda`。`sum` 累加选区内各经度格点的质量输送，更接近论文 Figure 2 垂直结构所采用的经度积分流程；`mean` 是本工具保留的单格输送平均选项，不代表论文默认定义。两者均以 kg s⁻¹ 表示，但数值尺度与物理含义不同；`mean` 也随经度分辨率改变。与论文幅值比较时还需统一经度区间、资料、时段和积分层。二者都不自动构成质量闭合的二维区域翻转环流，不能直接与传统全纬圈平均流函数混用。
- **边界诊断：** 在指定层中寻找核心区以北、给定纬度窗口内的第一个零交叉。窗口内没有零交叉时保留缺测；若全部年份都不满足条件，有效样本数可以为 0，不能将其解释为边界没有响应。边界统计只使用边界与指数均有限的年份，并在该子样本上按设置去趋势、重新标准化指数，因此样本和指数尺度可能与完整共同期的格点回归不同。
- **显著性：** 使用普通相关/线性回归的双侧 t 检验，自由度为 `n-2`。默认阈值为 `p < 0.1`；未进行时间自相关自由度修正或空间多重检验校正。图中打点表示逐格点检验通过，不是区域整体显著性的证明。

## 与参考论文的关系

参考文献为 Ma, W., Wang, Y., Miao, Y., and Li, X. (2025), *Effect of Regional Hadley Circulation–Subtropical Jet Interaction: The Amplification of Central-Pacific ENSO-Induced Teleconnection Pattern*, *Geophysical Research Letters*, 52(9), e2025GL115344，DOI：[10.1029/2025GL115344](https://doi.org/10.1029/2025GL115344)。

本工具采用其区域环流诊断思路，并提供通用指数响应分析。默认季节、时段、压力上边界、区域和指数均属于本工具的配置，不代表论文全部实验设置。它不完整复现论文中的中部型/东部型 ENSO 分类、副热带急流相互作用或 PNA 遥相关机制分析。

## 输出与复用

主要结果文件：

| 文件 | 内容 |
| --- | --- |
| `mass_streamfunction.nc` | 局地质量流函数及相关元数据 |
| `horizontal_intensity.nc` | 水平强度、气候态、异常及指数回归 |
| `regional_sections.nc` | 两个经度区间的纬度—气压结构与回归 |
| `predictor.csv` | 共同年份及实际用于回归的指数 |
| `summary.csv` | 运行设置与区域摘要 |
| `edge_timeseries.csv` | 各年份区域环流核心与边界纬度 |
| `edge_statistics.csv` | 边界与指数的统计及有效样本数 |
| `horizontal_response.png` | 水平强度响应图；启用绘图时保存 |
| `vertical_response.png` | 区域垂直结构响应图；启用绘图时保存 |

复用时记录代码版本、实际输入数据、共同样本期、积分上边界、区域归约方式、指数定义和去趋势设置。具体字段以保存文件的元数据及当前 `--help` 为准。

合成数据示例用于核对输入格式及数值流程，不能替代真实资料的质量检查，也不能证明参考论文的所有科学结论。
