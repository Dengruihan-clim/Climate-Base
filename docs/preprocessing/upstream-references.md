# 官方库源码调研与选型依据

调研日期：2026-11-01。目标是学习基础气象前处理与代码管理，支撑 [目标代码树](architecture.md) 和 [实施计划](implementation-plan.md)。本轮没有安装这些库、执行其代码、复制其算法或下载真实气象资料。

“官方”指机构维护或项目自身的官方仓库/文档，不表示所有项目均由政府机构开发。“常用”按气象 Python 生态集成关系与公开项目定位选择，不将 GitHub stars 当作科学正确性或使用人数证明。

## 1. 结论与复用关系

| 项目 | 维护来源/定位 | 重点学习与复用 | 在 Climate-Base 的角色 |
| --- | --- | --- | --- |
| [MetPy](https://github.com/Unidata/MetPy) | Unidata，气象 Python 工具 | 单位/CF 适配、插值接口，src/tests/docs/examples 管理 | 设计参照；专门插值按需调用；不引入全部 calc |
| [GeoCAT-comp](https://github.com/NCAR/geocat-comp) | NCAR GeoCAT | 混合层到气压层、维度与元数据包装、可信参考值验证 | 可选 vertical 后端 |
| [WRF-Python](https://github.com/NCAR/wrf-python) | NCAR/GeoCAT，WRF 专用 | 抽取、去交错、模式坐标与风矢量适配 | 可选 WRF 适配器 |
| [xarray](https://github.com/pydata/xarray) | PyData 社区，带标签数组 | Dataset/DataArray、I/O backend、CF/cftime、Dask | 新通用 API 的基础数据模型 |
| [cf-xarray](https://github.com/xarray-contrib/cf-xarray) | xarray-contrib 社区 | CF 语义访问器、坐标与关联变量识别 | 元数据识别基础 |
| [cfgrib](https://github.com/ecmwf/cfgrib) | ECMWF | ecCodes 到 xarray 的 GRIB 读取适配 | 可选 grib 后端 |
| [xESMF](https://github.com/pangeo-data/xESMF) | Pangeo 社区，ESMF Python 接口层 | 水平重网格、后端分层、权重复用 | 可选 regrid 后端 |
| [Meteostat](https://github.com/meteostat/meteostat) | Meteostat 开源项目，站点天气资料 | 站点目录、provider、配置/缓存、来源与测试分层 | 管理参照；后期可选站点 provider |
| [Iris](https://github.com/SciTools/iris) | SciTools，气象/地球科学数据处理 | 数据模型、元数据、文档与发布组织 | 对照参考；不新增第二套核心数据模型 |

核心思路是组合这些库的公开 API，自己负责严格输入契约、操作编排、来源记录与稳定接口。不是把每个库复制一份到仓库，也不是把所有依赖装进基础环境。

## 2. 可追溯源码基线

以下 SHA 由官方远端 `git ls-remote` 核验。tag 是研究基线，HEAD 快照只是固定本次阅读内容；**都不是已完成兼容测试的依赖锁文件**。用户文档的 stable/latest URL 可能随时间更新，实施时以所选发行版复核。

| 项目 | 研究 ref | 完整提交 SHA |
| --- | --- | --- |
| Climate-Base | main / 本地 HEAD | `d0ad09e0ca1f9cc589469a0fe08a30f8e7be6601` |
| MetPy | v1.7.1 | `433bdd18cc807efc2507e91094776403edee5973` |
| GeoCAT-comp | v2026.04.0 | `d3f0764279938701f0abfd1c81fed0fa915441c5` |
| WRF-Python | v1.4.2 | `124a8336529af6397fe150e14bd436d923122cdd` |
| Meteostat | v2.1.5；本次与 main/HEAD 一致 | `c7f1ae457da9f4a0b6872e5dbbdfd53c4f6dd48d` |
| xarray | HEAD 快照 | `2846938bbb820c5108bec6e9c80c785c0c8bf2b1` |
| cf-xarray | HEAD 快照 | `ee13cb860717299ca8f66d35461e5e50910f3d47` |
| cfgrib | HEAD 快照 | `1d11c7b919e9e87ebae50282fec62da441a1f76b` |
| xESMF | HEAD 快照 | `b30d1f12b085cc162af9cde359925ab51bdc1331` |

Iris 与 GeoCAT-examples 仅阅读项目概览，不将其未固定的具体实现作为本轮算法依据。规范参考使用已发布的 [CF 1.12](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html)；本次默认主分支文档显示 1.14-draft，不以草案作为本项目合规承诺。

## 3. MetPy 与 NCAR：学习接口、组合与验证

### MetPy

实际组织包括 `src/metpy/{calc,interpolate,io,plots}`、独立 `tests/docs/examples/tutorials/ci`。本项目学习其职责划分，不沿用超出范围的诊断与绘图目录。贡献指南强调一个逻辑变更配套测试、文档；测试资料使用可追溯缓存。参见固定版本 [CONTRIBUTING.md](https://github.com/Unidata/MetPy/blob/v1.7.1/CONTRIBUTING.md) 与 [pyproject.toml](https://github.com/Unidata/MetPy/blob/v1.7.1/pyproject.toml)。

[`interpolate/one_dimension.py`](https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/interpolate/one_dimension.py) 将线性/对数插值组织成独立 API；[`tests/interpolate/test_one_dimension.py`](https://github.com/Unidata/MetPy/blob/v1.7.1/tests/interpolate/test_one_dimension.py) 覆盖单位、轴、逆序、多维与边界。借鉴的是“输入适配—内核—输出”分工，实际算法优先调用公开接口。

[`xarray.py`](https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/xarray.py) 处理单位、CF 轴和 CRS；其中明确说明某些非 Dask 懒数据在单位量化/转换时会进入内存。因此 Climate-Base 需要自己验证惰性行为，不能把支持 xarray 当作保证。

### GeoCAT-comp / NCL

NCAR 已有 [Pivot to Python](https://www.ncl.ucar.edu/Document/Pivot_to_Python/) 路线；NCL 本身不是 Python 包。GeoCAT-comp 的 [`interpolation.py`](https://github.com/NCAR/geocat-comp/blob/v2026.04.0/geocat/comp/interpolation.py) **已经调用 MetPy 插值内核**，再补充混合气压、维度、DataArray 与分块处理；默认不启用地下外推。这支持组合依赖而非重复移植公式的设计。

[`test_interpolation.py`](https://github.com/NCAR/geocat-comp/blob/v2026.04.0/test/test_interpolation.py) 使用 NCL 参考输出及不同输入形态作验证；[pre-commit 配置](https://github.com/NCAR/geocat-comp/blob/v2026.04.0/.pre-commit-config.yaml) 固定检查工具版本。我们采用解析真值、上游对照、元数据与错误边界并行的测试方法。

[`pyproject.toml`](https://github.com/NCAR/geocat-comp/blob/v2026.04.0/pyproject.toml) 包含 `eofs/xskillscore` 等依赖，因此 GeoCAT 放在 vertical 可选组。GeoCAT-examples 的 [Gallery 与文档组织](https://github.com/NCAR/geocat-examples) 可作为示例管理参考，不成为运行依赖。

### WRF-Python

[`destag.py`](https://github.com/NCAR/wrf-python/blob/v1.4.2/src/wrf/destag.py) 把去交错与元数据包装分开；相邻格点求中点是网格位置变换，可纳入前处理。WRF 变量抽取、去交错、插值与风坐标转换放专用适配层，CAPE/涡度等诊断留在范围外。

当前核验版 [`pyproject.toml`](https://github.com/NCAR/wrf-python/blob/v1.4.2/pyproject.toml) 使用 scikit-build-core/CMake 与 Fortran，Python ≥3.10，维护者为 GeoCAT；[`ci.yml`](https://github.com/NCAR/wrf-python/blob/v1.4.2/.github/workflows/ci.yml) 包含构建 wheel 后安装测试。因此不能沿用“已停更/只能用旧 Python”的笼统结论，也不能从其元数据推断所有 Windows 路径都已持续验证。

## 4. xarray 生态：数据模型与专门后端

### xarray / cf-xarray

xarray 使用 `xarray/` 源码与 `doc/ci/asv_bench/design_notes` 等组织方式，并不与 MetPy 的 `src/` 完全一致。我们学习其公共数据模型、backend 和依赖分组，保留本项目更小的结构。[固定依赖配置](https://github.com/pydata/xarray/blob/2846938bbb820c5108bec6e9c80c785c0c8bf2b1/pyproject.toml) 的 Python 下限为 3.12；这不能直接覆盖 Climate-Base 现有 ≥3.11 的安装承诺。

官方 [I/O 文档](https://docs.xarray.dev/en/stable/user-guide/io.html) 支持显式 engine 和多文件 `preprocess`。Climate-Base 应额外约束 concat 轴、非拼接维、静态变量、时间重复和属性冲突，不复制示例中的宽松覆盖设置。非索引操作可能移除 encoding，输出需要重建打包/压缩设置。

[气候数据文档](https://docs.xarray.dev/en/stable/user-guide/weather-climate.html) 说明 cftime 与 CF 关联变量；它们是前处理参考，文档中的聚合/统计能力不是本轮目标。[cf-xarray 官方文档](https://cf-xarray.readthedocs.io/en/latest/) 提供语义坐标访问；其 [固定 CI](https://github.com/xarray-contrib/cf-xarray/blob/ee13cb860717299ca8f66d35461e5e50910f3d47/.github/workflows/ci.yaml) 分设 minimum/no-optional 环境，值得直接借鉴。

### cfgrib

ECMWF 的 [官方说明](https://github.com/ecmwf/cfgrib/tree/1d11c7b919e9e87ebae50282fec62da441a1f76b) 与 [`xarray_plugin.py`](https://github.com/ecmwf/cfgrib/blob/1d11c7b919e9e87ebae50282fec62da441a1f76b/cfgrib/xarray_plugin.py) 显示：读取建立在 ecCodes/xarray 上，参数包括 `filter_by_keys`、`indexpath`、`cache_geo_coords`。异构层类型/step 需要显式分组，不能强合为单一超立方体。

学习 `cfgrib/` 与坐标翻译 `cf2cdm/` 的分工。我们的包装控制只读原始目录、错误策略、缓存和来源；不自写 GRIB 解码器，也不在本轮承诺 GRIB 写出。

### xESMF

实际分层是 [`frontend.py`](https://github.com/pangeo-data/xESMF/blob/b30d1f12b085cc162af9cde359925ab51bdc1331/xesmf/frontend.py)（用户 API）、[`backend.py`](https://github.com/pangeo-data/xESMF/blob/b30d1f12b085cc162af9cde359925ab51bdc1331/xesmf/backend.py)（ESMPy 包装）、[`smm.py`](https://github.com/pangeo-data/xESMF/blob/b30d1f12b085cc162af9cde359925ab51bdc1331/xesmf/smm.py)（稀疏权重应用）。[用户 API](https://xesmf.readthedocs.io/en/stable/user_api.html) 支持权重复用；保守法需要边界，缺二维 bounds 不能按一维方式自动推导。

我们的新增设计是为权重建立包含真实坐标、bounds、mask、方法、选项和后端版本的指纹。仅依赖 shape 形成的文件名不能判断两张网格相同。ESMPy/ESMF 采用独立环境，参见 [安装说明](https://xesmf.readthedocs.io/en/stable/installation.html)。

<a id="meteostat"></a>

## 5. Meteostat：站点、数据源、缓存与管理

用户指定的 [meteostat/meteostat v2.1.5](https://github.com/meteostat/meteostat/tree/c7f1ae457da9f4a0b6872e5dbbdfd53c4f6dd48d) 使用 `import meteostat as ms`、`ms.stations.nearby`、`ms.hourly/daily`、`TimeSeries.fetch`，不能用旧版 `Hourly(...)`、`Stations()` 教程代替当前接口。入口见 [`__init__.py`](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/__init__.py) 与 [官方文档](https://dev.meteostat.net/python)。

### 实际代码组织

```text
meteostat/
├── api/             # 用户入口、配置、站点/库存、TimeSeries
├── core/            # cache/network/data/schema/validator/providers
├── providers/       # 声明式注册表 + meteostat/noaa/dwd/eccc/... 解析器
├── interpolation/   # 邻站估计与高程修正，独立于读取
├── typing.py        # Station/ProviderSpec/Request/License 等契约
└── enumerations.py, parameters.py, py.typed
tests/{unit,integration,provider,fixtures}/
examples/
.github/workflows/{tests,provider-tests,lint,typing,publish}.yml
pyproject.toml, poetry.lock, CONTRIBUTING.md, CITATION.cff, LICENSE
```

| 固定版本入口 | 已读到的设计 | 迁移到我们项目的方式 |
| --- | --- | --- |
| [api/stations.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/api/stations.py) | SQLite 目录、邻站/元信息/可用期查询分开 | `catalog/stations.py` 与真正的数据读取分离 |
| [typing.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/typing.py) | 来源、参数、请求、许可等结构化规格 | 使用少量明确规格对象，避免松散 dict 贯穿所有层 |
| [providers/index.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/providers/index.py) | priority、grade、license、参数、范围与实现模块登记 | provider 注册与解析分离，显式选择来源 |
| [api/timeseries.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/api/timeseries.py) | 稳定用户接口、来源与许可输出 | 统一外部接口；来源信息随输出保留 |
| [api/config.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/api/config.py) | 集中配置、环境变量类型转换 | 本项目使用显式配置快照，便于审计和复现 |
| [core/cache.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/core/cache.py) | 缓存与业务分开、TTL、清理 | 缓存服务与原始资料归档分开管理 |

### 不能照搬的默认行为

1. `include_model_data=True` 是默认配置。适配器须显式限制来源，并核验输出，不将模式补充视作观测。其 [过滤实现](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/providers/meteostat/shared.py) 作用于相应 provider 路径，不能假设开关对任意直接 provider 都是全局保证。
2. `fetch` 默认 `squash=True, fill=False, sources=False, clean=True`。按源优先级合并可以让同一行不同变量来自不同源；合并时使用 `sources=True` 保留逐变量 `*_source`。`squash=False` 的来源保留在 `source` 索引，不会因 `sources=True` 自动生成上述列。`fill=True` 是按日期索引 reindex，新行保持缺测，**不是数值插补**。证据：[utils/data.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/utils/data.py)。
3. 默认 clean 会将不符合规则的值置空；更早的数据服务已进行类型转换与浮点一位小数舍入，`clean=False` 不能恢复原精度。证据：[core/schema.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/core/schema.py)、[core/data.py](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/core/data.py)。
4. provider 的 RECORD/OBSERVATION/ANALYSIS/FORECAST 是来源级类别，不是逐值质量旗标。[GHCND 解析器](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/providers/noaa/ghcnd.py) 读取过 flags，但最终列筛选并不保留它们。因此标准输出应标为规范化站点产品；要求原始 flags/精度时另行获取 raw 数据。
5. [`api/interpolate.py`](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/api/interpolate.py) 与 [`lapserate.py`](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/interpolation/lapserate.py) 包含邻站估值和高程修正，不放进观测读取默认路径。normals、派生日/月聚合、自动模型补充都不纳入此次基础功能。将来做站点插值应单列接口并标为估计值。

### 缓存与测试管理

缓存键当前由函数与参数构成，TTL 还受全局配置影响；借鉴职责分离即可，不据此宣称缓存天然可复现。我们计划记录数据产品/解析器版本与有效配置，并将请求缓存、固定原始快照、重网格权重分开。

[`core/network.py`](https://github.com/meteostat/meteostat/blob/v2.1.5/meteostat/core/network.py) 有超时和重试；部分 provider 仍直接读 URL，故不能声称所有请求都经过统一网络层。Climate-Base 应明确 provider 的网络行为，并在普通测试中隔离它。

[`tests.yml`](https://github.com/meteostat/meteostat/blob/v2.1.5/.github/workflows/tests.yml) 的常规测试与 [`provider-tests.yml`](https://github.com/meteostat/meteostat/blob/v2.1.5/.github/workflows/provider-tests.yml) 的外部数据源测试分开；[`lint.yml`](https://github.com/meteostat/meteostat/blob/v2.1.5/.github/workflows/lint.yml)、[`typing.yml`](https://github.com/meteostat/meteostat/blob/v2.1.5/.github/workflows/typing.yml) 分别负责静态检查和类型检查。我们学习测试隔离，不自动复制其联网开 issue 等管理动作。

[`pyproject.toml`](https://github.com/meteostat/meteostat/blob/v2.1.5/pyproject.toml) 使用 Poetry，Python `^3.11`、pandas `>=2.3,<4`；这是放在 stations 可选组的另一个理由，不需要为此迁移现有 setuptools。

## 6. 许可与引用边界

MetPy 核验版为 BSD-3-Clause；GeoCAT-comp、WRF-Python 为 Apache-2.0；Meteostat 代码为 MIT，见各版本 pyproject/LICENSE。**软件许可与数据许可分别记录**。Meteostat 再分发数据与直接 provider 数据的条款不能混同，按 [官方许可说明](https://dev.meteostat.net/license) 和实际来源核对 attribution。

本轮只学习组织方式与公开接口。后续若调用库，登记依赖版本；若复制/改编代码，核验具体文件许可、保留版权、注明源 commit 和修改。其他参考项目在实际引入前补齐具体版本许可，不凭这张研究表替代许可清单。本项目自有代码的许可另由维护者确定。

## 7. 本轮验证范围

已完成官方仓库/文档与关键源码阅读、远端 ref 核验、本地现有结构审计和文档检查。未运行上游包、未做真实数据兼容或数值性能测试，未将所有候选依赖安装为工作环境；这些属于后续对应实施阶段。
