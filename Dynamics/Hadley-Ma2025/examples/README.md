# 输入与命令示例 / Inputs and commands

以下命令使用占位目录；运行前替换 `<DATA_ROOT>` 和 `<OUTPUT_ROOT>`。命令应在包含 `hadley_ma2025.py` 的目录执行。所有风场和指数由使用者自行准备。

Replace `<DATA_ROOT>` and `<OUTPUT_ROOT>` before running. Execute commands from the directory containing `hadley_ma2025.py`. Users supply their own winds and climate index.

## 无观测资料的合成示例 / Synthetic example

```bash
python examples/generate_synthetic.py --output-dir demo_input
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind divergent --v-var vd --index demo_input/index.csv --index-column nino34 --output-dir demo_output/divergent
```

生成器只创建解析合成场，要求 `demo_input` 尚不存在。安装可选谱后端后，可运行实际后端验算，并使用相同风场的 `u/v` 路径：

```bash
python tools/validate_spectral_backend.py
python hadley_ma2025.py --winds demo_input/winds.nc --input-kind total --index demo_input/index.csv --index-column nino34 --output-dir demo_output/total
```

The generator creates analytic synthetic fields and requires a new directory. The divergent-wind path uses only base requirements. The validation command and total-wind run require a real optional spectral backend.

## 总风输入 / Total winds

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/winds.nc" --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley_total"
```

总风路径需要全球经纬度网格的 `u/v`，单位为 m s⁻¹，并安装可用的球谐变换后端。必须在全球场上完成 Helmholtz 分解，再选择区域；不能把区域裁剪后的风场送入该分解。

Total-wind input requires global `u/v` in m s⁻¹ and a working spectral backend. Helmholtz decomposition precedes regional selection.

## 预计算辐散风 / Precomputed divergent wind

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/divergent_winds.nc" --input-kind divergent --v-var vd --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley_divergent"
```

`vd` 必须是真实的辐散经向风分量，单位为 m s⁻¹。此路径不执行 Helmholtz 分解，因此不需要可选谱依赖。使用者负责记录其来源、全球分解方法和截断设置。

`vd` must be the actual divergent meridional component in m s⁻¹. This path bypasses decomposition and the optional spectral dependency. Record its provenance, decomposition method, and truncation.

## 已聚合季节样本 / Seasonal samples

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/son_vd.nc" --input-kind divergent --v-var vd --frequency seasonal --index "<DATA_ROOT>/son_index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley_son" --no-plots
```

季节输入每年只能对应一个目标季节样本；场与指数必须表示同一季节。可使用日期 `time` 或整年 `year` 维度/坐标。`--no-plots` 保留数值计算和结果保存。

Seasonal inputs must contain one target-season sample per year. Winds and the index must describe the same season. A date-based `time` or integer `year` dimension/coordinate identifies the sample year. `--no-plots` retains numerical calculation and output saving.

## 指数 CSV / Index CSV

每年一个目标季节指数可以使用：

```csv
year,nino34
1981,-0.3
1982,1.2
1983,0.5
```

也可使用 `time,nino34`。若包含 `season` 列，使用目标季节记录；若同一年有多个月的日期记录，则按目标季节完整三个月聚合。以下仅示意三个完整月份，不构成足够的回归样本：

```csv
time,nino34
1981-09-01,-0.3
1981-10-01,-0.4
1981-11-01,-0.2
```

Use `year,<index-column>` for one seasonal value per year, or `time,<index-column>`. A `season` column may identify season-specific records. Multiple monthly records in one year are aggregated using the complete three-month target season. The short examples illustrate the schema and do not provide enough years for regression.

## 多文件与坐标 / Multiple files and coordinates

```bash
python hadley_ma2025.py --winds "<DATA_ROOT>/winds_early.nc" "<DATA_ROOT>/winds_late.nc" --index "<DATA_ROOT>/index.csv" --index-column nino34 --output-dir "<OUTPUT_ROOT>/hadley_multifile"
```

风场按坐标合并。常用坐标别名 `latitude/longitude` 会归一为 `lat/lon`；`plev`、`pressure`、`lev`、`isobaricInhPa`、`isobaricInPa` 会归一为 `level`。压力坐标必须有明确单位属性，支持 Pa、hPa、mbar、mb、kPa；缺失时可通过 `--pressure-units` 显式声明。程序不会仅根据压力数值大小猜测单位。

输入风场变量必须明确声明为 m s⁻¹（接受 `m s-1`、`m/s` 等常见等价写法）。缺失风速单位时，只有已确认数据本身以 m s⁻¹ 表示，才可使用 `--wind-units "m s-1"` 声明；该参数不执行风速转换。以节等其他单位存储的数据应先作实际数值转换。

Wind files are combined by coordinates. Common coordinate aliases are normalized: `latitude/longitude` to `lat/lon`, and `plev`, `pressure`, `lev`, `isobaricInhPa`, or `isobaricInPa` to `level`. Pressure coordinate attributes must declare Pa, hPa, mbar, mb, or kPa, unless explicitly supplied with `--pressure-units`. Units are not inferred from magnitudes.

Wind variables must explicitly declare m s⁻¹; common equivalent strings such as `m s-1` or `m/s` are accepted. If metadata are missing, use `--wind-units "m s-1"` only after confirming that the values already have those units. This option declares units without converting wind values. Convert other units, such as knots, numerically before use.

GRIB 扩展名 `.grib`、`.grb`、`.grib2` 还需要可选 `cfgrib` 及其可用的 ecCodes 环境；基础依赖默认不包含这些额外读取组件。

GRIB inputs (`.grib`, `.grb`, `.grib2`) additionally need optional `cfgrib` and a working ecCodes environment; these readers are not part of the base requirements.

## 输入准备检查 / Preparation

- NetCDF 风场应具有时间、气压、纬度和经度坐标；变量和坐标名称的可用配置见 `python hadley_ma2025.py --help`。
- 命令行输入使用完整、均匀的全球经度网格，不重复 0°/360° 周期端点；规则纬度网格包含两极。纬度和风场维度必须一致。
- 气压应具有明确单位；本工具积分时统一为 Pa。按工具的坐标及单位约定准备数据，勿仅修改单位标签。
- 指数 CSV 包含 `year` 或可解析的 `time` 列与数值指数列。`--index-column` 指定数值列。
- 月输入需有完整目标季节；季节输入需已使用一致的聚合定义。对齐、去趋势和标准化应使用共同样本期。
- 最高输入气压层决定积分参考层；默认 100 hPa 不等价于参考论文中从气压 0 起算的理想积分。
- 输入缺失值、地下压力层和不完整全球场需要在科学上恰当处理，不能为通过运行而随意填零。

The NetCDF input needs consistent time, pressure, latitude, and longitude coordinates; the index CSV needs a `year` or parseable `time` column and a numerical index column. Use complete seasonal samples and a consistent seasonal definition. CLI input requires a uniform complete global longitude grid without a duplicate cyclic endpoint and a regular latitude grid containing both poles. Consult `--help` for supported coordinate and unit options. The supplied pressure top is the integration reference. Do not relabel units or fill missing/underground winds with arbitrary zeros.

## 参数帮助 / Full interface

```bash
python hadley_ma2025.py --help
```

区域、季节、积分层、基准期、去趋势和输出设置以当前脚本的参数说明为准。应用结果时同时报告所用设置。

Use the script's current argument help for sectors, season, levels, baseline, detrending, and output settings, and report those settings with applications.
