# 基础气象前处理架构与代码树

状态：整体设计持续分期实施；2026-11-01 已开始实现首批 xarray/Dask NetCDF 闭环。实际可用 API 以 [当前 README](../../Preprocessing/README.md) 为准，目标树仍含后续未实现模块。

重构基线：[Climate-Base d0ad09e](https://github.com/Dengruihan-clim/Climate-Base/tree/d0ad09e0ca1f9cc589469a0fe08a30f8e7be6601)，调研时已核对本地 HEAD 与远端 main。后续用户要求所有场运算采用 xarray/Dask，故 Dask 由原计划的可选依赖改为核心必需；保留旧公开返回类型，在其边界计算。

阅读顺序：[上游调研](upstream-references.md) → 本文代码树 → [逐项实施计划](implementation-plan.md) → [开发与管理约定](development.md)。

## 1. 目标与边界

将现有“严格月均 NetCDF 读取器”逐步扩展为可复用的基础气象数据前处理包，继续使用发行名 `climate-preprocessing`、导入名 `climate_preprocessing`，保留仓库各子包独立安装的结构。

本次所称前处理是：读取、解码、检查、选择、规范化、坐标/单位变换、网格转换和可追溯输出。插值、去交错、保守重网格虽然包含数值运算，目的仍是改变数据表示或网格位置，属于范围内。

| 范围内 | 本轮范围外 |
| --- | --- |
| NetCDF/Zarr 读写、GRIB 读取，多文件严格拼接，压缩与缺测编码 | 月/季/年平均、累计量的时间聚合、区域平均 |
| CF 轴、单位、日历、边界、投影和变量关联的识别与检查 | 气候态、距平、标准化、去趋势、滤波 |
| 时间排序/切片/缺重复记录检查，显式选择成员和预报时效 | 相关、回归、EOF、显著性检验、气候指数 |
| 经纬度整理、区域裁剪、掩膜、垂直插值、水平重网格 | 动力/热力诊断和物理量经验推算 |
| 站点目录、数据源适配、缓存、质量标记和来源记录 | 默认补值、站点融合估计、模式资料自动替代观测 |

已有 `seasonal_mean`、`iter_seasons`、季节输出和 `seasonal` 命令继续兼容；不在本轮扩展，也不在拆分目录时直接移入 `Statistical`。后续若调整其归属，单独制定迁移方案。

用户所写 “NACR” 暂按 **NCAR** 理解。NCAR 是机构；本次学习 GeoCAT 和 WRF-Python，NCL 仅作为历史方法与对照来源。Meteostat 按用户指定的 `meteostat/meteostat` 仓库研究，不以旧版教程替代。

## 2. 重构前结构与保留资产

```text
Preprocessing/
├── climate_preprocessing/
│   ├── __init__.py, __main__.py
│   ├── archive.py       # CF 月记录、坐标、单位、来源、读取集中于此
│   ├── units.py
│   ├── seasons.py       # 已有季节聚合
│   ├── io.py            # 已有季节输出
│   └── _validation.py
├── tests/               # 合成 CF 数据与错误边界测试
├── examples/
├── pyproject.toml
├── README.md, README_EN.md, SOURCE_NOTES.md
└── validation.json, SHA256SUMS.txt
```

已有基础是严格且有价值的：按真实 CF 时间选月；拒绝缺月/重复月；显式坐标和单位；不隐式删重复经度、不取层交集、不猜成员；保留缺测；文件变化检测、可选 SHA256 与隐私友好清单；流式季节写出与拒覆盖发布。

基线限制是 `archive.py` 职责集中、数据为 eager NumPy、只支持一维经纬度及气压轴、输出专用于季节结果。当前 [archive.py](../../Preprocessing/src/climate_preprocessing/archive.py) 已加入实际惰性读取；原季节写出迁至 [io/_seasonal.py](../../Preprocessing/src/climate_preprocessing/io/_seasonal.py)，通用读写单独实现。不是只给旧 eager 实现加一个 `chunks` 参数。

## 3. 目标代码树

**以下是逐阶段建设的目标，不表示这些文件已经存在。** 先实现有验收用例的模块，不预建一批仅含 `pass` 的接口。小模块先用单文件，功能增长后再拆子包。

其中 `.github/workflows/preprocessing.yml` 仍是待启用的目标路径。当前只提供 [工作流模板](preprocessing-ci.yml.example)，实际位于 `docs/preprocessing/preprocessing-ci.yml.example`，未启用 GitHub Actions；需要自动检查时复制到目标路径后提交。

```text
Climate-Base/
├── .github/                         # P1：仓库级自动检查
│   ├── workflows/preprocessing.yml
│   └── pull_request_template.md
├── docs/preprocessing/              # P0：本轮已落地
│   ├── architecture.md
│   ├── upstream-references.md
│   ├── implementation-plan.md
│   └── development.md
├── Preprocessing/
│   ├── pyproject.toml               # 保留独立构建；依赖与工具统一管理
│   ├── README.md, README_EN.md
│   ├── CONTRIBUTING.md, CHANGELOG.md
│   ├── SOURCE_NOTES.md              # 原有作者来源记录
│   ├── THIRD_PARTY.md               # 实际复用的上游/版本/许可/改动
│   ├── .pre-commit-config.yaml
│   ├── src/climate_preprocessing/
│   │   ├── __init__.py              # 少量稳定公开 API；不导入重后端
│   │   ├── __main__.py              # 仅调用 cli.main
│   │   ├── _version.py              # 统一版本来源
│   │   ├── _validation.py           # 保留旧基础校验的兼容路径
│   │   ├── exceptions.py            # 可区分格式、坐标、单位、依赖错误
│   │   ├── specs.py                 # 输入/网格/输出规格，不另造数据容器
│   │   ├── validation.py            # 结构、坐标、声明范围、质量标记检查
│   │   ├── metadata.py              # CF 关系与 attrs/encoding 分离
│   │   ├── coordinates.py           # 轴识别、重命名、经度域、排序、bounds
│   │   ├── units.py                 # 旧 convert_units 保留；新具名数组接口
│   │   ├── time.py                  # CF 解码、历法、排序/切片/重复检查
│   │   ├── subset.py                # 时间、层、变量、成员和跨日界线裁剪
│   │   ├── masking.py               # 明确的有效性/地形/用户掩膜，不补值
│   │   ├── provenance.py            # 输入标识、版本、参数和变换记录
│   │   ├── io/
│   │   │   ├── __init__.py          # 同时保留旧 write_seasonal* 导出
│   │   │   ├── netcdf.py            # xarray + 显式 NetCDF engine
│   │   │   ├── multifile.py         # 时间索引与严格合并策略
│   │   │   ├── grib.py              # 可选 cfgrib/ecCodes，不自写解码器
│   │   │   ├── zarr.py              # 可选 Zarr，单独约定发布语义
│   │   │   └── write.py             # encoding、输出检查、拒覆盖发布
│   │   ├── regrid/
│   │   │   ├── api.py              # 方法、网格、掩膜与输出契约
│   │   │   ├── weights.py          # 构建/应用分开，缓存指纹与清单
│   │   │   └── _xesmf.py           # 唯一 xESMF/ESMPy 适配边界
│   │   ├── vertical/
│   │   │   ├── pressure.py         # CF 混合坐标及系数/单位约定
│   │   │   ├── interpolate.py      # 线性/对数压力插值公开接口
│   │   │   └── _geocat.py          # 可选 GeoCAT；按需复用 MetPy
│   │   ├── adapters/
│   │   │   ├── registry.py         # 显式产品/版本注册
│   │   │   ├── era5.py, merra2.py, jra3q.py
│   │   │   └── wrf.py              # WRF 抽取、去交错、坐标适配
│   │   ├── catalog/                # 借鉴 Meteostat，后期扩展
│   │   │   ├── files.py            # 本地文件和变量/时段清单
│   │   │   └── stations.py         # 站点标识、坐标、海拔、可用期
│   │   ├── providers/
│   │   │   ├── base.py             # 获取契约；不放数据变换算法
│   │   │   └── meteostat.py        # 可选站点数据获取与来源保留
│   │   ├── cache.py                # 请求缓存；与重网格权重缓存分开
│   │   ├── stations.py             # 站点记录/单位/时区/QC 规范化
│   │   ├── config.py               # 有版本的声明式配置
│   │   ├── pipeline.py             # 编排已公开函数，不自建计算调度器
│   │   ├── cli.py                  # inspect/prepare 与旧命令兼容
│   │   ├── archive.py              # 旧 MonthlyArchive 等导入门面
│   │   ├── seasons.py              # 旧季节 API 导入门面
│   │   └── _legacy/
│   │       ├── archive.py
│   │       ├── seasons.py
│   │       └── seasonal_io.py      # 保持旧返回类型与默认行为
│   ├── tests/
│   │   ├── conftest.py             # 合成数据工厂，不修改 sys.path
│   │   ├── unit/                   # 按 metadata/coordinates/time 等对应
│   │   ├── integration/            # 读→变换→写；可选后端分标记
│   │   ├── compatibility/          # 旧 API、CLI、默认值回归
│   │   └── data/README.md          # 小型公开夹具来源与校验和
│   ├── examples/
│   │   ├── netcdf_prepare.py
│   │   ├── regrid.py
│   │   ├── hybrid_to_pressure.py
│   │   ├── station_prepare.py
│   │   └── configs/
│   ├── docs/                       # 实现后生成的 API 与用户手册
│   └── benchmarks/                 # 后期检查惰性和内存；不用真实私有数据
├── Statistical/                    # 统计计算保持独立
├── Dynamics/                       # 动力诊断保持独立
└── ...                             # LBM、Climate-Indices 等既有目录
```

这里选择 MetPy 常用的 `src/ + tests/ + docs/ + examples/` 布局；并非声称所有参考库都用 `src/`。核心按数据操作分层，产品名只出现在适配器。先有单个后端时不建设动态插件框架。

## 4. 数据与依赖契约

### 4.1 统一数据对象

新增通用 API 使用 `xarray.Dataset/DataArray`，场数据必须由 Dask 分块承载；小型坐标检查和块内部内核可用 NumPy。输入的内存场转为 Dask 后处理，既有分块尽量保留。计划中的 `specs.py` 仅保存选项和校验条件。`MonthlyArchive.read()` 仍返回 `MonthlyField`，在此兼容边界计算；新增 `read_lazy()` 返回 DataArray。

规则经纬网使用一维 `latitude/longitude`；曲线网格保留 `y/x` 维和二维纬经坐标；垂直坐标区分 pressure、height 与 model level。不能把任意 `level` 改名为 pressure。第一阶段只承诺现有规则网格，其他网格随对应阶段验证后开放。

站点资料是另一种几何：保存 `station_id`、时间、经纬度、海拔、变量、质量标记与来源。规则记录可用 `(time, station)`，稀疏记录保留 observation 维；不为套格点接口制造稠密矩阵。Meteostat 的 pandas 对象在适配边界转换，核心不同时维护第二套通用计算 API。

Meteostat 标准接口提供的是规范化产品：关闭 `clean` 不能恢复已舍入的精度或已丢弃的原始 flags。其适配器必须声明这个限制；只有取得原始资料的专门路径才能承诺保留原始 QC/精度。`include_model_data` 开关、provider 白名单和逐变量来源检查共同构成来源约束，不能只靠一个全局开关。具体源码依据见 [Meteostat 调研](upstream-references.md#meteostat)。读取已发布的日/月产品属于 I/O；自行计算日/月产品属于本轮排除的聚合。

### 4.2 依赖策略

| 分组（拟定） | 依赖方向 | 原则 |
| --- | --- | --- |
| core | NumPy、xarray、Dask array、netCDF4、cftime | 当前首批闭环的必需依赖；不等同于分布式集群部署 |
| 后续 CF/单位扩展 | cf-xarray、Pint | 当前先采用明确 CF 属性与原有单位表；扩大范围前另行验证，不宣称已经接入 |
| grib | cfgrib + ecCodes | 只在调用 GRIB 功能时加载 |
| zarr | Zarr 及必要 I/O 依赖 | 明确支持的格式版本；不隐式接入云服务 |
| regrid | xESMF + ESMPy/ESMF 环境 | Python extra 不保证系统后端可用；独立安装说明与集成环境 |
| vertical | GeoCAT-comp、MetPy（按实际调用） | GeoCAT 本身已复用 MetPy，不重复移植算法 |
| wrf | WRF-Python 与编译后端 | 专用环境、独立测试；不作为默认 Windows 安装要求 |
| stations | Meteostat | 可选数据源；网络/缓存操作必须显式 |
| test / docs / dev | pytest / Sphinx / Ruff、pre-commit、build 等 | 与运行依赖分开 |

除已落地 core/test/dev 外，其余是目标分组。实际配置见 [pyproject.toml](../../Preprocessing/pyproject.toml)。保留 Python ≥3.11；不能把上游 main 的最新依赖要求原样复制过来。发现新功能必须提高下限时，应在该功能实施前明确记录版本变化。

GeoCAT 的运行依赖包含统计相关包，但本项目只调用已列明的前处理 API；将其放入可选组可避免核心读写安装这些依赖。MetPy、WRF-Python 也不整体导入其诊断接口。Iris 只作元数据与治理对照，不并列成为核心数据模型。

## 5. 必须提前约定的行为

1. **识别与修正分离。** CF 属性优先，显式 mapping 覆盖需审计；歧义时报错。校验报告和重命名/单位转换是不同操作。基础校验不是完整 CF 合规认证。
2. **坐标变换带动数据。** 纬度翻转、经度取模、排序同步作用于数据、bounds 与关联变量；默认重复点报错。二维经纬网不能套一维排序；跨日界线裁剪显式处理。选择操作默认不插值。
3. **多文件合并从严。** 使用实际 CF 时间、显式 concat 轴和非拼接维精确匹配；静态变量/属性冲突要检测。不能默认 outer join、`compat='override'` 或忽略错误。0.2.1 提供显式 `metadata_policy="preserve_sources"`，只把白名单中的文件说明及描述范围移入逐来源审计；单位、网格、历法、CF 引用、有效值限制和未知属性仍从严。缺时段检查依据用户声明的频率/预期索引，而非对所有资料强套逐月规则。
4. **保留日历与预报语义。** 支持 cftime，不默默转 pandas；不自动删闰日或混合 360_day 与 Gregorian。历法转换需指定对齐/缺日策略。`forecast_reference_time`、`step`、`valid_time` 分开，按有效时间重复的预报不自动去重。
5. **单位转换真实改变数值。** 不只改 `units` 属性；区分摄氏温度与温差。未知单位可原样保留但不可自动换算；位势到高度、累计到速率等物理语义转换不混同单位换算。
6. **缺测不变有效零。** `_FillValue`、mask 和 NaN 统一时保留来源；QC 原标记作为关联变量保留。按声明范围筛选需显式选择；不做统计异常识别或自动补洞。
7. **元数据经过变换后重新核验。** 保护 `bounds`、`grid_mapping`、`formula_terms`、`cell_measures`、`ancillary_variables` 的引用完整性；更换网格时重新生成或明确移除已失效关联，并记录原因。输出重新构造 encoding，避免旧 scale/offset 重复打包。基准参考固定 CF 1.12，而非网页主分支草案。
8. **惰性是一项可验证行为。** 通用变换不隐式调用 `.values`、`.load()` 或对场 `np.asarray`；读取小型坐标与扫描场值分开。对会强制计算的后端明确告知、分块或拒绝，不能仅因接受 xarray 就宣称惰性。源文件在计算/输出完成前保持可用并管理关闭责任。
9. **重网格方法显式。** bilinear、nearest、conservative 各有含义；不按变量名猜方法。保守法要求合法单元边界和面积/掩膜约定，区分 intensive 与 extensive 量、conservative 与 conservative_normed。权重指纹包含真实网格、bounds、mask、方法、periodic、选项和后端版本，不只网格 shape。
10. **垂直层语义先于插值。** 区分 `p=a*p0+b*ps` 与 `p=ap+b*ps`，以及层中心/层界面；系数、ps 与目标压力的单位和维度均检查。默认不外推，不跨缺测层桥接；地下层策略显式。去交错与风矢量旋转独立于标量插值。
11. **输出与来源同时设计。** 已有输出默认拒覆盖；本地单文件先写临时文件、验收后发布。不能把硬链接方案宣称适用于 Zarr/对象存储；多文件存储另定义完整性标记与提交过程。JSON 审计记录输入、参数、软件版本、关键变换和可选 hash，不暴露凭据或私人绝对路径。
12. **站点数据来源透明。** 本地文件不需要 provider；联网获取才进入 providers。观测、模式、插值估计和聚合产品保持可区分；站点时区到 UTC 的转换明确处理 DST/歧义时刻。请求缓存记录 provider、参数、格式/包版本、有效期和校验，不与可长期复用的重网格权重缓存混用。

## 6. 依赖方向与兼容迁移

调用顺序为 `CLI/配置 → 流程编排 → 公开前处理函数 → xarray/可选后端`。`providers` 获取来源产品/记录并声明其处理级别，原始资料另走 raw-provider 路径；`adapters` 解释产品特有字段。二者复用基础层，基础层不反向依赖某个产品。

迁移 `src/` 时先只移动源码并修正构建与导入测试，再拆职责。`archive.py`、`seasons.py` 保留旧路径；原来的 `io.py` 变成 `io/` 包时，在 `io/__init__.py` 重导出 `write_seasonal` 和 `write_seasonal_stream`。旧 CLI `check/seasonal` 及无 schema_version 的配置维持旧语义；新增配置使用独立版本及 `inspect/prepare` 命令，避免把旧 `check` 的月资料假设传播到通用接口。

新函数命名与参数在各阶段落地前确认，不把本设计中的文件名当成已发布 API。兼容代码只做转发或冻结旧实现；不维护两套会持续分叉的基础算法。每次合并须同时通过旧接口回归和新增契约验收。

旧异常类型及触发条件也属于兼容契约；新增专门异常需兼容已有 `ValueError` 等捕获方式。修改错误文本若影响 CLI 或现有测试，也须作为明确变更记录。
