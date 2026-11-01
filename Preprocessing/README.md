# 基础气象前处理：xarray 与 Dask

[English](README_EN.md)

**作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。**

## 0.3.0：压力、预期时间与用户掩膜

0.3.0 已通过本地源码、树外 wheel 与 Linux HPC 各 **283 项测试**。原跨年拼接实测 5 项通过、1 项预期拒绝；新增压力/时间/掩膜实测 7 项通过、4 项预期拒绝，均无失败或未运行项。2004 年 1 月两层、16×32 格点与原有参考逐值一致，构图零场任务，掩膜与写回一致，原文件大小和修改时间不变。预期拒绝验证缺记录、空时间选择、掩膜错位、覆盖输出及原严格属性策略。不计算完整全场，不声称已测得加速比。详见 [validation.json](validation.json)。

按代码树补齐 P2/P3 的下一批能力，所有场变换继续通过 xarray/Dask 惰性执行：

- `normalize_pressure(ds, pressure="level", target_units="hPa", order="ascending")`：依据明确单位和 CF 证据识别一维气压坐标，转换 Pa/hPa、排序，并同步数据、bounds 与 CF 引用。输出坐标名为 `pressure`；不将任意模式层改成气压，不进行垂直插值。
- `validate_time(ds, expected=...)`：与用户给出的完整预期时间索引精确比较，检查中间和首尾缺记录。支持解码时间或完整 ISO 日期字符串，保留源历法；不推断频率、不补值、不转换历法。
- `subset_time(ds, start=..., end=...)`：按精确时刻作闭区间选择，保留时间 bounds 和辅助变量。`YYYY-MM-DD` 表示当日午夜；不接受仅年份/月的部分日期，空结果报错。
- `apply_mask(ds, condition, variables=[...], reason="...")`：布尔条件 True 保留、False 置缺测；必须显式选择字段并注明理由。共享维标签须完全同序匹配，允许按维广播，不自动交集或重新对齐。原 QC 与 CF 结构变量受保护，不自动将 QC 解释为有效性。

`prepare_dataset` 及 schema-v1 JSON 配置增加可选 `pressure`、`expected_time`、`time_range`；先检查完整输入时间，再切片。默认不要求地面资料含气压轴。任意布尔掩膜通过 Python API 在准备之后组合，JSON 不执行表达式。

```python
from climate_preprocessing import apply_mask, prepare_dataset, write_dataset

with prepare_dataset(
    "monthly.nc", chunks={"time": 1},
    pressure={"pressure": "level", "target_units": "hPa", "order": "ascending"},
    expected_time=["2000-01-01", "2000-02-01", "2000-03-01"],
    time_range={"start": "2000-02-01", "end": "2000-03-01"},
) as ds:
    selected = apply_mask(ds, ds.quality == 0, variables=["temperature"],
                          reason="按该产品定义保留 quality == 0 的记录")
    write_dataset(selected, "new_masked.nc")
```

示例中的变量和 QC 含义需与实际产品一致。独立掩膜文件必须由调用者保持打开直至计算结束；结果关闭共享源 Dataset，不取得外部掩膜文件的关闭所有权。可运行纯合成示例：`python examples/pressure_time_mask.py /path/to/new_demo`。它验证压力换算与重排、时间选择、QC 保留和缺测写回；不下载真实数据。

当前压力层中心和 bounds 端点均须大于零，零压顶界留待层界面流程处理。整数场掩膜需要浮点数表示 NaN；不能被浮点数精确表示的保留整数会在对应 Dask 块执行时报错，避免悄悄改变数值。

参考与独立改进逐模块列明，见 [THIRD_PARTY.md](THIRD_PARTY.md)。历法转换、GRIB/Zarr、重网格、混合层、产品适配与 Meteostat provider 仍按后续阶段实施。

## 0.2.1：已实现的基础前处理闭环

新通用接口以 **xarray Dataset/DataArray + Dask 分块任务图** 处理场数据。读取、坐标重排、裁剪、掩膜和单位转换保持惰性；`write_dataset` 或使用者显式 `.compute()` 时执行。小型坐标/时间/边界检查可以读取内存数组，块内部数值核可用 NumPy；不先加载整个场再假装分块。Dask 支持分块并行，但实际速度取决于数据规模、块大小与磁盘带宽，本轮验证的是正确性和惰性契约。

已实现：本地 NetCDF 单/多文件读取、严格 CF 关联检查、时间与历法检查、一维经纬度规范化、跨经度接缝裁剪、非有限值掩膜、显式单位转换、来源审计、拒覆盖写出，以及 `inspect/prepare` CLI。新流程不做月/季平均、距平或其他统计。

```python
from climate_preprocessing import prepare_dataset, write_dataset

with prepare_dataset(
    ["input_2000.nc", "input_2001.nc"],
    coordinates={"latitude": "lat", "longitude": "lon"},
    chunks={"time": 12, "lat": 90, "lon": 180},
    longitude_convention="0_360",
    region={"latitude_bounds": [-30, 60], "longitude_bounds": [270, 90]},
    units={"temperature": "K"},
) as ds:
    # ds 中的场仍是 Dask 数组；源文件须保持可用且不变。
    write_dataset(ds, "new_prepared.nc")  # 显式计算与写出边界
```

第一次体验可运行 `python examples/netcdf_prepare.py /path/to/new_demo`；示例只创建合成数据。其生成的 `prepare.json` 使用 `schema_version: 1`，可独立运行：

```bash
python -m climate_preprocessing inspect /path/to/new_demo/prepare.json
python -m climate_preprocessing prepare /path/to/new_demo/prepare.json
```

示例已经写出的 `prepared.nc` 不可覆盖，再次 `prepare` 请先在配置中选择新输出名。所有相对输入/输出路径相对于配置文件；输入为显式文件列表。`chunks` 使用源维度名，未知维度报错，不能设为 `None`。`inspect` 读取元数据和坐标，但不执行场值任务。

当前支持范围是规则一维经纬网；经度 bounds 需为包含中心、宽度小于 180° 的局部未跨缝区间。通用单位接口沿用现有明确换算表，尚未引入 Pint/cf-xarray 的全套词汇。时间支持已解码日期/顺序检查、显式预期索引核验和精确闭区间选择，不推断频率、自动补缺时段或转换日历。GRIB/Zarr、重网格、混合层、提供方专用适配和 Meteostat 获取接口尚待后续阶段。

多文件按实际时间排序；重复时间、不同历法、变量/单位冲突或非拼接维网格变化会报错。重复出现的静态 data variable 暂须单独读取，不能为比较大场而隐式计算；CF 静态坐标可作有限大小检查。

默认 `metadata_policy="strict"` 要求所有属性一致。0.2.1 起，对于同类逐年资料，可在 `open_mfdataset`、`prepare_dataset` 或 JSON 配置中显式选择 `metadata_policy="preserve_sources"`：

```python
from climate_preprocessing import open_mfdataset

with open_mfdataset(
    ["V_2003.nc", "V_2004.nc"],
    chunks={"time": 1},
    metadata_policy="preserve_sources",
) as ds:
    # 按实际时间拼接，场仍为 Dask 数组；在这里选取需要的时空范围。
    print(ds.sizes)
```

此策略只允许全局 `history`、`CONVERSION_DATE`、`date_created`、`date_modified`、`creation_date`，以及变量 `actual_range`、`minimum_value`、`maximum_value` 存在差异。它们从多文件合并结果的顶层属性移入每个来源的 `source_metadata` 审计记录，不丢弃各文件记录，也不把单年范围误作整个时段的范围。没有扫描场数据重算极值；单文件属性保持不变。单位、历法、气压层/网格、CF 引用、`valid_*` 限制与所有未列出的属性仍严格检查。此选项仅协调文件说明信息，不执行重网格或单位转换。使用前仍需核实输入属于可拼接的同类产品。

配置文件采用相同选项。例如，下面的 `inspect` 配置只检查并构建惰性数据集；路径相对于配置文件：

```json
{
  "schema_version": 1,
  "input": ["V_2003.nc", "V_2004.nc"],
  "metadata_policy": "preserve_sources",
  "chunks": {"time": 1},
  "coordinates": {"latitude": "latitude", "longitude": "longitude"}
}
```

保存为 `inspect.json` 后运行 `python -m climate_preprocessing inspect inspect.json`。执行 `prepare` 时还需配置一个不存在的 `output` 文件名，并按需要设置 `region` 与 `units`。

输出保留 CF 引用，重新构造磁盘 encoding，避免原始 packed dtype/scale 被重复应用。单位/经度变换会移除过期范围属性并记录；变换后的实际范围不隐式扫场重算。JSON 来源记录保存软件版本、步骤、参数、输入文件名/大小/mtime；通用读取在构图及每个执行块检查大小/mtime 变化，但不计算 SHA256，不能把它当作加密快照。已有源属性不会自动清除私人文字，发布前应检查自己的源元数据。

源码在 [src/climate_preprocessing](src/climate_preprocessing/)，模块内 `References` 与 `Climate-Base improvements` 区分设计参考、原有方法与新增改进。参见 [第三方参考](THIRD_PARTY.md)、[开发说明](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)。

## 0.2.1 验证记录

2026-11-01，本地 Windows 源码、源码树外安装的 wheel，以及 Linux HPC 独立环境分别通过 **147 项测试**。依赖版本和验证范围见 [validation.json](validation.json)。本轮仅提供 [CI 模板](../docs/preprocessing/preprocessing-ci.yml.example)，未启用 GitHub Actions，没有远端 CI 执行结果；需要启用时，将模板复制到仓库的 `.github/workflows/preprocessing.yml` 后提交。

真实资料验证直接使用 ERA5 月平均气压层 V 风场的 2003、2004 年原始文件。显式 `preserve_sources` 策略形成 `time=24, level=37, latitude=721, longitude=1440` 的惰性数据集，构图期间执行场任务数为 0。随后从合并后的 Dask 图计算两个时次、两层、16×32 格点，场值、坐标和时间标识与已验证的小样本完全一致；写出读回保留单位、历法和逐来源记录，原始文件大小与修改时间不变。

此次专项检查为 **5 项通过、1 项预期拒绝、0 项失败**。预期拒绝用于确认默认 `strict` 模式仍拒绝描述属性差异；它不表示新策略合并失败。没有计算完整 24 个月全场，也没有验证完整资料质量或 Dask 加速比。真实数据、账号路径和服务器运行日志不随公共源码分发。

## 保留的月均/季节兼容接口

下文描述原有月均接口。`MonthlyArchive.read()`、`seasonal_mean()` 的返回类型保持不变，内部场计算已迁移到 xarray/Dask，在旧接口返回处计算。新增 `MonthlyArchive.read_lazy(..., chunks="auto")` 返回 DataArray；调用者在计算/写出后负责 `.close()`。它是月场兼容视图，不是完整 CF Dataset 往返接口。

原接口将作者既有的月度资料读入、完整记录检查和季节归约方法整理为独立 Python 包。使用者提供本地 NetCDF 数据、变量名及坐标配置，即可读取规范化月场并生成完整季节序列，供后续回归、趋势和推断使用。

适用于**已经转换为月均 CF NetCDF** 的 ERA5、MERRA-2、JRA-3Q 等资料。变量名和坐标名必须匹配实际文件元数据；程序不假定所有提供方或 JRA-3Q 产品使用同一组名称，不直接读取原始 GRIB，也不自动适配混合模式层。

## 安装与快速体验

要求 Python ≥3.11，依赖 NumPy、xarray、Dask、netCDF4、cftime；测试另需 pytest。本项目 Windows 工作站使用 `C:\ProgramData\anaconda3\python.exe`，以下 `python` 代表使用者选定的解释器：

```bash
python -m pip install -e ".[test]"
python examples/netcdf_prepare.py /path/to/new_demo
python -m climate_preprocessing inspect /path/to/new_demo/prepare.json
python -m pytest tests -q
```

以上命令在 `Preprocessing` 目录执行。生成器只创建全合成数据，要求目标目录尚不存在，并演示写出；再次写出须选择新文件名。源码目录仅包含代码、配置示例和测试，不附带真实资料。示例中的 `/path/to/...` 均需替换。旧月资料/季节示例仍由 `examples/generate_synthetic.py` 生成，其 `config.json` 对应 `check`、`seasonal` 命令。

## 输入契约与配置

参见 [配置示例](examples/config.example.json)。`files` 可为文件名、glob 或字符串列表；相对路径以 `data_root` 为基准。支持 `{data_root}`、`{year}`、`{month:02d}` 和 `{yyyymm}` 模板，既可读逐月文件，也可从年度多记录文件选择月份。

- 每个文件必须有非空、有限、严格递增的一维 CF 时间坐标及时间单位。日期用文件原始 calendar 解码，绝不从文件名推断。每个日历月最多一条记录；请求月份在全部候选文件中必须恰好出现一次。
- 输入应已是所需月均场。程序不能仅凭一条月记录证明其来自完整日/小时样本，使用者须核实产品定义。程序不把逐日或逐小时资料自动聚合成月平均。
- 经纬度及可选气压坐标均须为一维；纬度单位明确为北向角度、经度为东向角度。支持 CF 常见等价写法，拒绝 `degrees_south`/`degrees_west`。不支持曲线网格、约化经度网格、混合模式层或非单值 ensemble/expver 等额外维；需先明确选择这些维度。
- 变量维度可转置，额外单值维可挤除。默认纬度升序、经度 `[0,360)` 升序、气压 hPa 升序；可显式改成纬度降序、`[-180,180)` 经度、Pa 或降序气压。重复坐标及重复循环端点报错；不自动删列。默认允许区域网格；`require_global_longitude=true` 会检查经度循环覆盖，仍不进行插值或补齐。
- 同一字段跨月须保持规范化后的格点、气压层和输出单位完全一致。不同字段默认允许各自原生网格；`shared_grid=true` 额外要求共同水平网格及所有三维字段的共同气压层，允许同网格二维地面字段。不会自动取层交集或重网格。
- `hash_inputs`、`shared_grid`、`require_global_longitude` 必须是真正的布尔值；字符串 `"false"` 不会当作布尔配置。年月必须是 Python/NumPy 整数，拒绝布尔值及浮点截断。

## Python API

```python
from climate_preprocessing import Coordinates, FieldSpec, MonthlyArchive, iter_seasons
from climate_preprocessing import write_seasonal_stream

archive = MonthlyArchive(
    {"temperature": FieldSpec(
        files="temperature_{year}.nc",
        variable="temperature",
        coordinates=Coordinates(time="time", latitude="latitude",
                                longitude="longitude", pressure="pressure_level"),
        expected_units=("K", "degree_Celsius"),
        target_units="K",
    )},
    data_root="/path/to/data",
    calendar_policy="modern_gregorian",
    pressure_units="hPa",
)
month = archive.read("temperature", 2000, 1)
# month.values: (pressure, latitude, longitude); 二维场没有 pressure 维。
metadata = archive.inspect("temperature", 2000, 2)

records = iter_seasons(archive, "temperature", 2000, 2020,
                       months=(12, 1, 2), weighting="days")
write_seasonal_stream("/path/to/output/seasonal.nc", records,
    pressure_units=archive.pressure_units,
    inventory=lambda: archive.inventory, audit=lambda: archive.audit)
```

`MonthlyField` 返回数值、纬度、经度、可选气压、输出单位、原始/有效 calendar 和变换记录。`seasonal_mean(...)` 返回内存中的 `SeasonalField`；大数据使用 `iter_seasons(...)` 与流式写出。`seasonal` CLI 逐季写入同一个临时 NetCDF，不把多年完整季节场同时放入内存；每次返回一季的完整数组；执行时内存还取决于 Dask 分块、并发及归约缓冲区。

安装独立的统计包后，按同一季节年份对齐自己的预测量，即可连接回归接口；本包不预先改变预测量尺度：

```python
from climate_preprocessing import seasonal_mean
from climate_statistics import linear_regression
seasons = seasonal_mean(archive, "temperature", 2000, 2020, months=(12, 1, 2))
# predictor 与 seasons.years 已按年份对齐；可另提供相同年份的 controls。
fit = linear_regression(predictor, seasons.values, seasons.years, controls=None)
```

## 日历与完整季节

季节月份必须是连续、不重复的 1–12 个月序列，以**最后一个月所在年份**标记。DJF 2000 对应 1999-12、2000-01、2000-02；配置的起年就是首个季节标签，不会自动少算第一个冬季，缺前一年 12 月会报错。

`weighting="days"` 按每条资料的实际 CF 日历月长度加权，支持 Gregorian 闰年、`noleap`、`all_leap`、`360_day` 等。`weighting="equal"` 明确选择等月平均，例如研究中定义的月末状态平均。不会猜测某字段应求和；月累计量若需季节累计，须在其他明确方法中处理。

每个所需月必须存在，每格点各月均有限才输出该季节数值；缺测、掩膜或原始非有限值保留为 NaN，不补零、不缩短季节、不对剩余月份重归一。零值和正负号均保留。

默认 `calendar_policy="strict"`：只合并 CF 同义名，如 standard/gregorian、noleap/365_day。显式选择 `modern_gregorian` 后，先按源 calendar 解码，**仅对选中年 ≥1583** 的 standard/gregorian/proleptic_gregorian 统一为 proleptic_gregorian，便于现代资料跨来源匹配。古历及其他日历不混合；使用该策略的分析期应完全位于现代时段。原始 calendar 标签保留在审计中。

## 单位与物理含义

未设置 `target_units` 时保留输入单位，不按变量名猜物理量。`expected_units` 显式限制允许的源单位；`units_override` 是使用者对源属性的明确纠正，原属性、覆盖值与全部变换均记录。

已知兼容换算包括 K/摄氏度、Pa/hPa、kg/kg 与 g/kg、m/mm、Pa/s 与 hPa/s、K/s 与 K/day，支持常见等价单位写法。米每秒垂直速度不能换成 Pa/s。位势到位势高度、能量到温度倾向、累计降水到速率等需要物理假设，不自动处理；确有依据时可显式声明 `scale`、`offset` 和 `target_units`。例如位势转位势高度须由使用者明确选取重力常数：

```python
from climate_preprocessing import convert_units
height = convert_units(geopotential, "m2 s-2", "m", scale=1/9.80665)
```

未知单位可原样保留，不能未经声明跨维度换算。复数拒绝；有限输入因换算产生 Inf/NaN 时报错，不把溢出当作源缺测。惰性接口在实际计算/写出时执行数值溢出检查。旧 MonthlyArchive 不判断变量的物理范围、不裁剪、不做地形/地下掩膜、不重新标准化，也不做水平或垂直插值。

## 输出、检查和验证边界

NetCDF `seasonal_value` 使用 `(season_year, [pressure], latitude, longitude)`，保存 calendar、单位、月份、权重方法和审计。来源清单保存相对 `data_root` 的文件名、大小、可选 SHA256 和选用月记录；外部文件使用匿名目录标签。输入不修改；已有目标文件（含符号链接）不覆盖。新文件在同目录完成后通过硬链接原子发布，文件系统须支持该操作；失败时删除临时文件，不发布部分结果。

`check` 验证全部配置字段和所需年月的时间、维度、网格、单位，**不解码场变量数值**。默认 `hash_inputs=true` 仍会为 SHA256 读取源文件全部字节，因此不是完全不读文件内容的元数据检查；关闭哈希可减少这部分 I/O。`check` 不等于全量场值或科学质量验收。源文件在索引后发生变化会报错。

默认自动化测试使用全合成 CF 文件，覆盖年月/日历、跨年及完整季节、坐标与维度转置、严格单位、缺月/重复月、格点/气压变化、非有限值、来源隐私、流式写出及拒覆盖。不附带真实数据或远程连接逻辑，不安装或调用已有 HPC 科学工程入口。本模块提供前处理，不执行下游统计推断。

### 有限资源的真实资料逐项验证

在已有依赖的环境中，可用源码附带的独立工具检查自己的实际 NetCDF；输出目录必须尚不存在：

```bash
python tools/validate_real_data.py \
  --input /path/to/year_a.nc --input /path/to/year_b.nc \
  --output-dir /path/to/new_validation_report \
  --variable V --latitude latitude --longitude longitude --time time --pressure level \
  --monthly-product
```

`--monthly-product` 只用于已经核实产品定义的月资料；小时/日资料应省略该选项，工具不会推算月均。最多输入两个文件，每个文件仅读取一个时次、前两个气压层、中部最多 16 个纬度和前 32 个连续经度；额外非单值维需明确指定 `--select DIM=INDEX`。场变换使用 xarray/Dask，Dask 最多两个线程，BLAS/OpenMP 各一个线程，不做统计归约或全文件散列。

小切片仍可能解压整个原生 NetCDF 压缩块。工具先记录其估计解码大小，默认单块上限 192 MiB（可通过 `--allow-native-chunk-mib` 明确调整），每个原文件只在后端连续切片边界读取一次场值。原始文件用于元数据、严格多文件契约与惰性构图检查；坐标整理、区域裁剪、掩膜、单位保留、写回、组合流程及旧接口实际计算，使用这些真实切片派生的小文件。派生文件为拼接测试统一属性，报告明确区分 `original_source` 与 `derived_real_subset`，不代表完整原始资料可直接拼接或已通过全场质量验收。

每项结果立即写入 `results.jsonl`，最终汇总在 `report.json`。状态为 `PASS`、`FAIL`、`NOT_APPLICABLE` 或 `EXPECTED_REJECTION`；有 `FAIL` 时返回非零退出码。已独立确认的原始属性冲突，以及拒绝覆盖已有输出，记录为预期拒绝；资源超限或未经确认的月产品不会伪报通过。报告保留软件版本、选取范围、来源文件名/大小/mtime 和检查依据，输入文件保持原样。
