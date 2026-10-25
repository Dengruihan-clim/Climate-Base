# 月均气候资料前处理

[English](README_EN.md)

**作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。**

`climate_preprocessing` 将作者既有的月度资料读入、完整记录检查和季节归约方法整理为独立 Python 包。使用者提供本地 NetCDF 数据、变量名及坐标配置，即可读取规范化月场并生成完整季节序列，供后续回归、趋势和推断使用。

适用于**已经转换为月均 CF NetCDF** 的 ERA5、MERRA-2、JRA-3Q 等资料。变量名和坐标名必须匹配实际文件元数据；程序不假定所有提供方或 JRA-3Q 产品使用同一组名称，不直接读取原始 GRIB，也不自动适配混合模式层。

## 安装与快速体验

要求 Python ≥3.11，依赖 NumPy、netCDF4、cftime；测试另需 pytest。请在自己选定的环境中执行：

```bash
python -m pip install -e ".[test]"
python examples/generate_synthetic.py /path/to/new_demo
python -m climate_preprocessing check /path/to/new_demo/config.json
python -m climate_preprocessing seasonal /path/to/new_demo/config.json
python -m pytest tests -q
```

生成器只创建全合成数据，要求目标目录尚不存在。发布目录仅包含源码、配置示例和测试，不附带真实资料。示例中的 `/path/to/...` 均需替换。

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

`MonthlyField` 返回数值、纬度、经度、可选气压、输出单位、原始/有效 calendar 和变换记录。`seasonal_mean(...)` 返回内存中的 `SeasonalField`；大数据使用 `iter_seasons(...)` 与流式写出。`seasonal` CLI 逐季写入同一个临时 NetCDF，不把多年完整季节场同时放入内存；内存仍需容纳一条月场与归约缓冲区。

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

未知单位可原样保留，不能未经声明跨维度换算。复数拒绝；有限输入因换算产生 Inf/NaN 时直接报错，不把溢出当作源缺测。库不判断变量的物理范围，不裁剪、不做地形/地下掩膜，不重新标准化，也不做水平或垂直插值。

## 输出、检查和验证边界

NetCDF `seasonal_value` 使用 `(season_year, [pressure], latitude, longitude)`，保存 calendar、单位、月份、权重方法和审计。来源清单保存相对 `data_root` 的文件名、大小、可选 SHA256 和选用月记录；外部文件使用匿名目录标签。输入不修改；已有目标文件（含符号链接）不覆盖。新文件在同目录完成后通过硬链接原子发布，文件系统须支持该操作；失败时删除临时文件，不发布部分结果。

`check` 验证全部配置字段和所需年月的时间、维度、网格、单位，**不解码场变量数值**。默认 `hash_inputs=true` 仍会为 SHA256 读取源文件全部字节，因此不是完全不读文件内容的元数据检查；关闭哈希可减少这部分 I/O。`check` 不等于全量场值或科学质量验收。源文件在索引后发生变化会报错。

测试使用全合成 CF 文件，覆盖年月/日历、跨年及完整季节、坐标与维度转置、严格单位、缺月/重复月、格点/气压变化、非有限值、来源隐私、流式写出及拒覆盖。不附带真实数据或远程连接逻辑，不安装或调用 HPC 工程入口。本模块提供前处理，不执行下游统计推断。
