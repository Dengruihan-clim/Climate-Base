# 基础气象前处理：xarray 与 Dask

[English](README_EN.md)

**作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。**

## 0.7.0 仓库版本：具名月均气压层产品适配

P7 第一阶段已完成实现与验收，增加 `adapt_product`：调用者显式选择产品、变量角色、四轴坐标及额外维标签，输出统一名称的 xarray/Dask Dataset。本次 2026-11-05 GitHub 仓库更新采用 **0.7.0**，验证日期为 **2026-11-01**；0.3.0 为历史仓库版本，0.4.0–0.6.0 开发阶段未单独发布。WRF 不在本项目当前实施范围；下一阶段为 P8 站点扩展。

| `product` | 本轮限定的产品表示 |
| --- | --- |
| `era5-monthly-pressure-levels` | ERA5 整月平均气压层再分析，`monthly_averaged_reanalysis` / `moda`；已解码的一维经纬网表示 |
| `merra2-m2imnpasm-5.12.4` | MERRA-2 `M2IMNPASM.5.12.4` / `instM_3d_asm_Np`，瞬时同化场的月平均 |
| `jra3q-monthly-isobaric-analysis` | GDEX `d640002` 的 `anl_p` / `anl_p125` 月平均等压面分析场 CF NetCDF |

JRA-3Q 的 `anl_p` 是 GDEX 将原生约化网格转换后的规则 Gaussian 经度/纬度轴，纬度不等距；`anl_p125` 为 1.25° 经纬网。允许已裁剪子集，不要求完整全球尺寸。上述范围不包含逐小时/月内逐时平均、标准差、预报场、地表产品或原始混合模式层。产品定义和直接核验的官方元数据见 [上游证据](../docs/preprocessing/upstream-references.md#p7-具名产品证据)。

`variables={"northward_wind": "v"}` 表示明确的角色到源变量映射，不按提供方名或短变量名猜测物理量。角色限定为 `air_temperature`、`eastward_wind`、`northward_wind`、`specific_humidity`、`geopotential`、`geopotential_height` 和 `lagrangian_tendency_of_air_pressure`。`coordinates` 可覆盖 `time/pressure/latitude/longitude` 的 profile 默认名称：ERA5 为 `time/level/latitude/longitude`，MERRA-2 为 `time/lev/lat/lon`，JRA-3Q 为 `time/pressure_level/lat/lon`；其他表示必须显式映射。

已解码时间必须有序、同年同月最多一条记录；固定气压轴须保持原始严格单调次序。映射物理场经选择后必须恰好包含四轴。其额外维度及已知的 `expver/member/number/realization` 坐标均须用 `selectors` 指定精确标签，包括单值维与已知标量坐标；选中坐标及来源保留。`expver` 不合并、不取首个非缺测值，也不进行成员平均。只有显式映射的场被声明为已适配，其他字段随共享选择/坐标引用同步保留。

适配校验单位及现有元数据、规范名称并同步 CF 引用；一般缺少 `standard_name` 时按明确角色补写并审计；MERRA-2 的 `geopotential_height` 是例外，源字段必须已明确声明同名 `standard_name`，即使单位为 `m` 或 `gpm`，缺该声明也会拒绝。它不排序坐标、隐式换单位、计算统计量、补写 `time: mean` 或构造时间 bounds。ERA5 位势 `m2 s-2` 与 JRA 位势高度 `gpm` 分开处理；不能把 MERRA-2 `H` 的 ad hoc 高度声明当作自动位势高度转换。未证实的 ad hoc `standard_name` 会拒绝，须由调用者先完成明确的科学解释。已有产品/统计/物理量声明冲突时报错；缺少证据时记录调用者声明，不能把通过结构检查当作完整产品认证或完整月采样证明。原有单位转换与垂直/水平插值仍为独立显式步骤。

```python
from climate_preprocessing import adapt_product, open_dataset, write_dataset

with open_dataset("monthly.nc", chunks={"valid_time": 1}) as source:
    result = adapt_product(
        source,
        product="era5-monthly-pressure-levels",
        variables={"northward_wind": "v"},
        coordinates={"time": "valid_time", "pressure": "pressure_level",
                     "latitude": "latitude", "longitude": "longitude"},
        selectors={"expver": 1},  # 仅当源文件确有此维；按实际标签填写
    )
    write_dataset(result, "adapted.nc")
```

`prepare_dataset(..., adapter={...})` 和 schema-v1 JSON 的 `adapter` 接受同样的四个配置项，仅适用于 NetCDF 或派生的本地 Zarr 表示；与 `input_format="grib"` 组合会拒绝。独立 `adapt_product` 也要求已解码的固定气压月资料，不证明原始 GRIB 的 time/step 语义。多文件先逐输入适配，再按原严格策略拼接；显式 `preserve_sources` 先保存每个源文件选择前的白名单范围属性，再清除选择后失效的范围，保持来源可追溯；源 expver 维长可以不同，但选中标量、网格、单位和历法仍须一致。`chunks` 和 `adapter.coordinates` 使用源名称；顶层 `time` 保持 `time`，后续裁剪、单位、垂直/水平步骤使用适配后的规范名称。完整合成示例见 [product_adapters.py](examples/product_adapters.py)；示例不代表真实产品字段认证。

0.7.0 Windows 源码和树外实际安装的候选 wheel 各 **714 通过、26 跳过**；仅核心依赖环境 **602 通过、138 跳过**；HPC 实际安装候选包 **740 项全通过、0 跳过**（105 条警告）。38 个运行模块及共 77 个 Python 文件逐字节核对一致，详见 [validation.json](validation.json)。

真实 ERA5 P7 小样本 **10 通过、4 项预期拒绝**，原值/API/CLI 对照最大误差均为 0，原文件大小与修改时间不变。三个 profile 的合成示例及各自 CLI 全部通过；MERRA-2/JRA-3Q 原始样本尚未验收，合成检查只证明契约。历史 P4–P6 数字分别保留，不累计为本轮结果。完整全场和性能加速比未测，公开记录不包含私人账户或原始资料绝对路径。

## 0.6.0 历史阶段：固定气压层与混合层垂直插值

P6 首批固定/混合层中心插值已完成实现与验收，提供 `hybrid_pressure` 与 `interpolate_pressure`，继续保持 xarray/Dask 场运算。以下结果属于 0.6.0 历史阶段，不替代当前 P7 验收。

0.6.0 Windows 源码与树外实际安装 wheel 各 **593 通过、26 跳过**；仅核心依赖环境 **483 通过、136 项可选后端跳过**；HPC 实际安装候选包 **619 项全部通过、0 跳过**。35 个运行时模块逐字节一致，4 项真实 GeoCAT 对照通过。跳过不代表对应后端已验收，详细环境与范围见 [validation.json](validation.json)。

真实 ERA5 小样本验收 **12 通过、2 项预期拒绝**，linear/log 对照最大误差均为 0。混合层用例使用合成 CF 系数，不等同于原始模式层产品验收。完整全场与性能加速比未测，公开记录不包含私人账户或原始资料绝对路径。

`hybrid_pressure` 不需要可选数值后端；插值按需安装 `.[vertical]`（MetPy 1.7.1 系列）。GeoCAT 同样调用 MetPy 的公开一维插值内核，因此本包直接封装该内核，GeoCAT 作为参考及可选对照，不增加未使用的强制依赖。安装与测试仍使用指定 Anaconda Python 及隔离依赖。

```python
from climate_preprocessing import open_dataset, hybrid_pressure, interpolate_pressure, write_dataset

with open_dataset("hybrid.nc", chunks={"time": 1, "lev": 16}) as source:
    pressure_pa = hybrid_pressure(source, coordinate="lev")  # 构建惰性 Pa 压力场
    result = interpolate_pressure(
        source, [850., 500., 200.], variables=["temperature"],
        method="log", coordinate="lev", target_units="hPa",
    )
    write_dataset(result, "new_pressure.nc")
```

输入可为有明确 Pa/hPa 单位证据的固定气压坐标，或 CF `atmosphere_hybrid_sigma_pressure_coordinate`。后者只接受完整且无重复项的两类 `formula_terms`：

| 公式 | 必要单位与维度 |
| --- | --- |
| `p = a*p0 + b*ps` | `a`、`b` 为同层中心轴上的一维系数，单位 `1`；`p0` 为正的标量参考压力，`ps` 为地面气压 |
| `p = ap + b*ps` | `ap` 为同层中心轴上的一维压力系数，`b` 单位 `1`；`ps` 为地面气压 |

`p0/ap/ps` 均需明确压力单位，支持既有 Pa/hPa 与毫巴别名换算。源至少两个层中心；系数必须与源层维和标签精确对应，不将 n+1 层界面或 bounds 当作中心。压力轴、系数和标量参考值可作为几何读取；ps 和气象场不在构图阶段执行。独立返回的压力 DataArray 使用通用源层标签，移除会悬空的混合公式声明，并用非 CF JSON 属性保留公式与源变量名；原 Dataset 元数据不变。

`method="linear"` 在压力上插值，`method="log"` 在对数压力上插值；方法必须显式选择。目标层使用 `target_units`（Pa/hPa，默认 hPa），必须正、有限、唯一，多个目标须单调并保留指定升/降序，输出维名为 `pressure`。仅选显式浮点物理场，不处理整数类别；结果为层中心点值，不生成目标 bounds，也不代表层平均或守恒垂直重映射。

不外推，不删除缺测层后跨缺口插值。源 NaN/inf 使相邻插值区间无效；恰好命中有限源层的目标仍保留该值。每个有效柱的压力须严格单调，不自动排序混合层。固定层可显式传 `surface_pressure="ps"` 屏蔽地下源/目标；省略时地形未知，仅按源层覆盖范围处理。混合层始终使用公式的 CF ps，不接受其他替代场。缺 ps 的整柱为 NaN；非正/无限 ps、非正/非有限派生压力或非单调柱在相应 Dask 任务执行时报错。目标即使在地面以上，只要低于最低有效源层，仍返回 NaN。

垂直核会把完整源柱合为一个垂直块，并保留非垂直分块，请据此安排时间/水平块的内存。输出保留单位、时间/历法、水平元数据、地面气压及独立变量，移除并审计旧层系数、源层 QC、bounds 与失效引用。源资源保持打开直至计算/写出；返回结果共享源关闭回调。

`prepare_dataset` / schema-v1 配置增加 `vertical`，如 `"vertical": {"target_levels": [850, 500], "variables": ["temperature"], "method": "log", "coordinate": "lev", "target_units": "hPa"}`。顺序为源规范化/裁剪和单位转换 → 垂直插值 → 非有限值掩膜 → 可选水平重网格，避免先把非法 ps 的无穷值改成缺测。`inspect` 只构图，不写结果或执行气象场；读取器可以解码时间/bounds 标签，不能把这些坐标任务误报为场计算。可运行合成示例见 [vertical_pressure.py](examples/vertical_pressure.py)。

## 0.5.0 历史阶段：水平重网格与权重缓存

0.5.0 完成 P5 首批规则经纬网支持与验收，该阶段改动与 0.4.0 一并保留在本地、未单独发布，现纳入 0.7.0 仓库版本。数值内核使用 xESMF/ESMPy/ESMF，字段由 xarray/Dask 惰性处理，不增加统计计算。二维曲线网格、矢量旋转等 P5 扩展另行验收；以下测试数字属于 0.5.0 历史阶段。

0.5.0 Windows 源码和树外候选 wheel 各 **473 通过、25 跳过**：24 项因缺原生 xESMF/ESMPy，1 项因符号链接权限；仅核心依赖环境 **406 通过、92 项可选后端跳过**。Linux HPC 实际安装的候选 wheel **498 项全部通过**，使用独立 Python 3.12.16 环境中的 **xESMF 0.9.2、ESMPy/ESMF 8.9.1**。31 个运行源码模块与本地、候选包和 HPC 逐字节一致。

真实 ERA5 P5 安装包复验为 **8 通过、2 项预期拒绝、0 失败**：构图零场任务，同网格恒等、已知索引最近邻与 NetCDF 往返逐值一致，原文件大小和修改时间不变。0.5.0 与 0.6.0 均未重跑 P4 的旧真实资料套件，不将其计入对应阶段通过。完整全场和性能加速比未测。该历史阶段的最终文档包经运行代码、测试、示例与工具逐字节核对后保留在本地，未重复部署 HPC。详见 [validation.json](validation.json)。

核心安装不导入重网格后端。`.[regrid]` 提供 xESMF Python 依赖，**不会安装可用的原生 ESMF**；请先在独立的受支持环境中安装 ESMPy/ESMF，再安装本包。Linux/conda-forge 安装方式见 [xESMF 官方说明](https://xesmf.readthedocs.io/en/stable/installation.html)。本机脚本仍使用指定 Anaconda Python，不为此更改现有科学计算环境。

```python
from climate_preprocessing import prepare_dataset, rectilinear_grid, regrid_dataset, write_dataset

target = rectilinear_grid(
    latitude=[-10, 10], longitude=[170, 190],
    latitude_bounds=[-20, 0, 20], longitude_bounds=[160, 180, 200],
)
with prepare_dataset("input.nc", chunks={"time": 1}) as source:
    result = regrid_dataset(
        source, target, variables=["temperature"], method="bilinear",
        weights_dir="weights",  # 显式请求本地权重缓存；省略时不写权重文件
    )
    write_dataset(result, "new_regridded.nc")
```

源和目标均要求升序的一维 `latitude` / `longitude` 维坐标与明确的度单位，每轴至少两个中心。经度可用 170、180、190 这样的连续展开形式跨日期变更线；不接受重复循环端点、二维曲线网格、投影网格或有歧义的跨缝边界。保守方法必须提供两轴连续 CF bounds，不自动推断单元边界。输入字段须显式选择且为浮点标量；整数类别、格点总量等 extensive 量，以及网格相对风矢量旋转不在本批范围内。

| 显式方法 | 数值含义 |
| --- | --- |
| `bilinear` | 双线性插值；无覆盖位置为 NaN，不开启外推 |
| `nearest_s2d` | 每个目标点取最近源点，采用后端的全局最近邻搜索域 |
| `nearest_d2s` | 各源点选择最近目标点，同一目标收到的值相加；一般不保持常数或 intensive 幅值 |
| `conservative` | 以完整目标单元面积归一；仅在覆盖与 ESMF 球面多边形几何一致时讨论面积积分守恒 |
| `conservative_normed` | 以有效覆盖面积归一；部分覆盖时与 `conservative` 不同，不承诺完整目标面积下的积分守恒 |

面积遵循 ESMF 球面大圆边界多边形的定义，不以普通点平均或任意 `cos(latitude)` 权重作为守恒证据。`periodic=True` 仅用于非保守方法的全球源网格，要求显式经度 bounds 证明完整 360° 覆盖；保守方法自行处理闭合并拒绝该标志。

`source_mask` / `target_mask` 由 Python API 接受静态、已计算的布尔 DataArray，标签必须与对应网格精确同序匹配，True 表示有效；不从 QC 或缺测场自动生成掩膜。坐标、声明的 bounds、掩膜及权重构建为明确的几何计算，气象场图保持惰性。参与权重的源 NaN 传播，不开启 `skipna` 动态归一或填补；无映射及屏蔽目标为 NaN。非水平分块保留，水平输出分块由源块大小与目标尺寸限定；速度与内存仍取决于分块和资料规模。

权重键包含坐标、bounds、mask、方法、周期设置、固定数值选项和后端版本，shape 相同不足以复用。缓存使用独占锁、临时文件、拒覆盖发布和 SHA256 清单；损坏、不完整或版本不匹配条目报错，不静默重建。锁文件不自动清除。此协议仅适用于支持硬链接的本地文件系统；记录中保存摘要而非私人绝对路径。

输出保留选中场及不依赖水平网格的变量，移除并审计旧空间场、QC/面积/CRS 引用、范围和失效打包信息；单位、时间/历法及仍有效的 CF 关系保留。复杂的失效混合层公式依赖会拒绝。源 Dataset 必须保持打开至计算结束，返回结果共享源关闭回调。

`prepare_dataset(..., regrid={...})` 和 schema-v1 配置接受 `regrid`；先完成源坐标规范化、裁剪、单位与非有限值处理，再重网格。JSON 不执行掩膜表达式。`weights_dir` 相对于配置文件解析：

```json
{
  "schema_version": 1,
  "input": "input.nc",
  "chunks": {"time": 1},
  "regrid": {
    "target_grid": {"latitude": [-10, 10], "longitude": [170, 190]},
    "variables": ["temperature"],
    "method": "bilinear",
    "weights_dir": "weights"
  },
  "output": "new_regridded.nc"
}
```

`inspect` 也需要可用的重网格后端，会在内存中构建几何权重，但不写缓存/结果、不执行气象场。`prepare` 才按配置持久化权重并写结果。[horizontal_regrid.py](examples/horizontal_regrid.py) 及其生成配置的 CLI inspect/prepare 已在 HPC 通过；来源与独立改进见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 0.4.0 历史阶段：可选 GRIB / Zarr

0.4.0 实现代码树 P4，当时仅保留本地及测试部署、未单独发布，现纳入 0.7.0 仓库版本。该阶段接口独立组合官方 cfgrib/ecCodes 与 xarray/Zarr，不复制解码器，不增加统计计算。

安装按需选择的后端，核心安装不会导入它们：

```bash
python -m pip install -e ".[grib,zarr]"
```

0.4.0 历史验收：Windows 源码与树外安装 wheel 各 **393 项通过、1 项平台限制跳过**；仅核心依赖环境 **326 项通过、68 项可选后端跳过**；Linux HPC **394 项全部通过**。当时真实 ERA5 三组检查合计 **18 项通过、8 项预期拒绝、0 失败**，原始文件不变。GRIB 检查使用真实 NetCDF 小样本派生的 GRIB2，尚非原始 GRIB 档案验收。这三组真实资料检查未在 0.5.0 或 0.6.0 阶段重跑，不能作为本轮验证结果；完整全场和性能加速比当时也未测。

0.4.0 当时后端配置为 cfgrib 0.9.15.1、Zarr 2.18.7、numcodecs 0.15.1。Windows 使用 ecCodes Python 绑定 2.50.0 / 原生库 2.50.0；Linux HPC 使用绑定 2.50.0 / 原有原生库 2.44.0，均通过该阶段格式测试。绑定版本不等于实际加载的原生库版本，两者分别记录。当时本机使用指定 Anaconda Python 及工作区隔离依赖，HPC 使用版本环境复用既有科学计算栈；后续阶段的独立 Linux 环境见对应验证记录。

```python
from climate_preprocessing import inspect_grib_groups, open_grib, open_zarr, write_zarr

print(inspect_grib_groups("input.grib2"))  # 分组信息，不解码气象场
with open_grib("input.grib2", filter_by_keys={"typeOfLevel": "isobaricInhPa"}) as ds:
    # time=起报时间，step=时效；valid_time 可为二维，原样保留。
    write_zarr(ds, "new_output.zarr")  # 明确计算边界；拒绝已有目标
with open_zarr("new_output.zarr") as ds:
    print(ds.sizes)  # 场仍为 Dask 数组
```

GRIB 要求本地纯串联、单字段物理消息；异构层型/网格/stepType 等需显式筛选，重复槽、稀疏消息组合、多字段 GRIB2 和非零起报秒会拒绝，避免上游静默丢字段或截断时间。读取保留统计窗口到来源记录，不把累计/平均转成瞬时值。头键检查可能读取完整压缩消息字节，但不请求或解码气象场值；不写 `.idx`，禁用 cfgrib 地理缓存。返回 float64 保留 ecCodes 解码精度。

Zarr 首批只支持本地、consolidated **v2 根组**和支持硬链接的文件系统。写入先完成兄弟暂存目录，再独占创建目标并发布文件，最后发布有版本的完成标记；这不是跨平台整目录原子替换或掉电持久性承诺。默认读取要求标记，检查元数据摘要与文件清单，读取块时检查源状态；缺 chunk、元数据变化和目标冲突会明确失败。受管目录迁移需保留文件修改时间，或从已验证源重新写出。只对外部独立验证的标准 v2 数据显式使用 `require_complete=False`：它没有本库完成保证，原先缺失的 chunk 仍按 Zarr fill 语义解释。

不支持 Zarr v3、云/对象存储、嵌套组、追加/区域覆盖或 GRIB 写出。气象浮点场打包为整数时必须给出合法 `_FillValue`；字段的新打包编码和精度由调用者明确指定。原数据保持只读，结果和输入须按 Dataset 生命周期关闭。

NetCDF 与 Zarr 写出共用日期编码政策：`time` 及其 CF `bounds` / `climatology` 关联变量使用一致的 `units`、`dtype` 和兼容历法；显式冲突会拒绝。默认使用 `int64`，按关联日期组所需的精度选择时间单位；纳秒数组中的日期若均为整微秒，就使用微秒以保留 cftime/旧接口的读回兼容性，只有实际存在亚微秒标签才使用纳秒。覆盖日期编码时须成对指定 `units` 与 `dtype`，不能把整数存储本身当作精度保证。

在最终文件或 Zarr 完成标记发布前，写出器回读已识别的解码日期变量（包括 `valid_time`、时间 bounds / climatology），逐标签比较精度、缺失位置和历法语义。每个日期变量最多 **1,000,000 个标签**，二维 bounds 的端点也计入；超过上限会拒绝发布。旧 xarray 对“纳秒日期与 NaT 混合”或远年 cftime 微秒日期若发生舍入，同样拒绝发布，而非接受时间被改变的输出。日期编码选择和回验都属于显式写出边界；不会为此额外全量回读气象数值场，也不进行历法转换或统计计算。

`prepare_dataset` / schema-v1 配置新增 `input_format`（默认 `netcdf`）、`input_options`。GRIB 仅接受 `filter_by_keys`，Zarr 仅接受 `require_complete`；两个后端均限一个明确输入，不按后缀猜格式。CLI 配置新增 `output_format`（`netcdf` 或 `zarr`），默认行为保持不变。GRIB 的 `expected_time` 默认检查起报轴，不会把有效时间自动摊平成分析序列。

```json
{
  "schema_version": 1,
  "input": "input.grib2",
  "input_format": "grib",
  "input_options": {"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
  "pressure": {"pressure": "isobaricInhPa", "target_units": "hPa"},
  "output": "new_prepared.zarr",
  "output_format": "zarr"
}
```

运行 `python examples/optional_formats.py new_demo` 可生成纯合成 NetCDF→Zarr 示例及 inspect 配置。参考、改进及限制见 [THIRD_PARTY.md](THIRD_PARTY.md)，本轮验证记录见 [validation.json](validation.json)。

## 0.3.0 历史交付：压力、预期时间与用户掩膜

0.3.0 已通过本地源码、树外 wheel 与 Linux HPC 各 **283 项测试**。原跨年拼接实测 5 项通过、1 项预期拒绝；新增压力/时间/掩膜实测 7 项通过、4 项预期拒绝，均无失败或未运行项。2004 年 1 月两层、16×32 格点与原有参考逐值一致，构图零场任务，掩膜与写回一致，原文件大小和修改时间不变。预期拒绝验证缺记录、空时间选择、掩膜错位、覆盖输出及原严格属性策略。不计算完整全场，不声称已测得加速比。详见 [0.3.0 历史验证记录](https://github.com/Dengruihan-clim/Climate-Base/blob/1e48f285ee5760ac42253a6c6ab01d91b29f880f/Preprocessing/validation.json)。

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

示例中的变量和 QC 含义需与实际产品一致。独立掩膜文件必须由调用者保持打开直至计算结束；结果关闭共享源 Dataset，不取得外部掩膜文件的关闭所有权。可运行纯合成示例：`python examples/pressure_time_mask.py new_demo`。它验证压力换算与重排、时间选择、QC 保留和缺测写回；不下载真实数据。

当前压力层中心和 bounds 端点均须大于零，零压顶界留待层界面流程处理。整数场掩膜需要浮点数表示 NaN；不能被浮点数精确表示的保留整数会在对应 Dask 块执行时报错，避免悄悄改变数值。

参考与独立改进逐模块列明，见 [THIRD_PARTY.md](THIRD_PARTY.md)。GRIB/Zarr v2 自本地 0.4.0 起提供，0.5.0 增加上述水平重网格。0.6.0 增加上述固定/混合层中心插值；0.7.0 增加三类具名月均气压层产品适配；历法转换与 Meteostat provider 仍待实施。

## NetCDF 基础前处理闭环（自 0.2.1 起）

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

第一次体验可运行 `python examples/netcdf_prepare.py new_demo`；示例只创建合成数据。其生成的 `prepare.json` 使用 `schema_version: 1`，可独立运行：

```bash
python -m climate_preprocessing inspect new_demo/prepare.json
python -m climate_preprocessing prepare new_demo/prepare.json
```

示例已经写出的 `prepared.nc` 不可覆盖，再次 `prepare` 请先在配置中选择新输出名。所有相对输入/输出路径相对于配置文件；输入为显式文件列表。`chunks` 使用源维度名，未知维度报错，不能设为 `None`。`inspect` 读取元数据和坐标，但不执行场值任务。

当前支持范围是规则一维经纬网；经度 bounds 需为包含中心、宽度小于 180° 的局部未跨缝区间。通用单位接口沿用现有明确换算表，尚未引入 Pint/cf-xarray 的全套词汇。时间支持已解码日期/顺序检查、显式预期索引核验和精确闭区间选择，不推断频率、自动补缺时段或转换日历。GRIB/Zarr v2、水平重网格与固定/混合层中心插值按上文独立契约使用；三类产品适配遵循 0.7.0 显式契约；Meteostat 获取接口尚待后续阶段。

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

输出保留 CF 引用，重新构造磁盘 encoding，避免原始 packed dtype/scale 被重复应用。单位/经度变换会移除过期范围属性并记录；变换后的实际范围不隐式扫场重算。JSON 来源记录保存软件版本、步骤、参数、输入文件名/大小/mtime；NetCDF/GRIB 读取在构图及场块执行时检查源文件大小/mtime。Zarr 读取核验 JSON 元数据 SHA256 和文件清单，并检查块访问时的源状态；这些操作都不哈希整个气象场，不能当作场值的加密快照。已有源属性不会自动清除私人文字，发布前应检查自己的源元数据。

源码在 [src/climate_preprocessing](src/climate_preprocessing/)，模块内 `References` 与 `Climate-Base improvements` 区分设计参考、原有方法与新增改进。参见 [第三方参考](THIRD_PARTY.md)、[开发说明](CONTRIBUTING.md)、[变更记录](CHANGELOG.md)。

## 0.2.1 历史验证记录

2026-11-01，本地 Windows 源码、源码树外安装的 wheel，以及 Linux HPC 独立环境分别通过 **147 项测试**。该版本依赖和验证范围见 [0.2.1 历史记录](https://github.com/Dengruihan-clim/Climate-Base/blob/14378910e5ef88eb025b83a3c04429f1c48ce5e5/Preprocessing/validation.json)。该阶段仅提供 [CI 模板](../docs/preprocessing/preprocessing-ci.yml.example)，未启用 GitHub Actions，没有远端 CI 执行结果；需要启用时，将模板复制到仓库的 `.github/workflows/preprocessing.yml` 后提交。

真实资料验证直接使用 ERA5 月平均气压层 V 风场的 2003、2004 年原始文件。显式 `preserve_sources` 策略形成 `time=24, level=37, latitude=721, longitude=1440` 的惰性数据集，构图期间执行场任务数为 0。随后从合并后的 Dask 图计算两个时次、两层、16×32 格点，场值、坐标和时间标识与已验证的小样本完全一致；写出读回保留单位、历法和逐来源记录，原始文件大小与修改时间不变。

此次专项检查为 **5 项通过、1 项预期拒绝、0 项失败**。预期拒绝用于确认默认 `strict` 模式仍拒绝描述属性差异；它不表示新策略合并失败。没有计算完整 24 个月全场，也没有验证完整资料质量或 Dask 加速比。真实数据、账号路径和服务器运行日志不随公共源码分发。

## 保留的月均/季节兼容接口

下文描述原有月均接口。`MonthlyArchive.read()`、`seasonal_mean()` 的返回类型保持不变，内部场计算已迁移到 xarray/Dask，在旧接口返回处计算。新增 `MonthlyArchive.read_lazy(..., chunks="auto")` 返回 DataArray；调用者在计算/写出后负责 `.close()`。它是月场兼容视图，不是完整 CF Dataset 往返接口。

原接口将作者既有的月度资料读入、完整记录检查和季节归约方法整理为独立 Python 包。使用者提供本地 NetCDF 数据、变量名及坐标配置，即可读取规范化月场并生成完整季节序列，供后续回归、趋势和推断使用。

适用于**已经转换为月均 CF NetCDF** 的 ERA5、MERRA-2、JRA-3Q 等资料。变量名和坐标名必须匹配实际文件元数据；程序不假定所有提供方或 JRA-3Q 产品使用同一组名称，不直接读取原始 GRIB，也不自动适配混合模式层。

## 安装与快速体验

要求 Python ≥3.11，依赖 NumPy、xarray、Dask、netCDF4、cftime；测试另需 pytest。本项目 Windows 工作站使用用户指定的 Anaconda 解释器，以下 `python` 代表使用者选定的解释器：

```bash
python -m pip install -e ".[test]"
python examples/netcdf_prepare.py new_demo
python -m climate_preprocessing inspect new_demo/prepare.json
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
