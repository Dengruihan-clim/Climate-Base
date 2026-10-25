# Dynamics：大气动力诊断

[English](README_EN.md) · [原始方法与软件引用](REFERENCES.md)

独立的 Python 计算模块，提供 Takaya–Nakamura 静止水平波活动通量、完整及线性 Rossby 波源、球谐风场与 Helmholtz 分解，以及经向质量流函数。输入为使用者提供的物理场；不附带真实资料、研究结果或服务器连接逻辑。

**作者：RH_Deng；Codex（OpenAI，AI 辅助整理与重构）。** 原始方法与软件作者分别列于 [REFERENCES.md](REFERENCES.md)，计算函数的 docstring 和 [机器可读引用](climate_dynamics/citations.py) 同时保留出处。

## 安装

需要 Python 3.11+。通量、波源代数装配与质量流函数仅依赖 NumPy：

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
```

**球谐风场还需要 windspharm 及其 pyspharm/SPHEREPACK 编译后端。** 已配置兼容编译环境时可安装 `python -m pip install ".[spectral]"`；该命令仍可能需要 Fortran 编译链，不保证所有 Windows/Python 组合都有现成 wheel。也可在平台支持时使用 conda-forge 的 `windspharm`，再在该环境安装本包。核心 NumPy 方法不受这个可选依赖影响。

配置后运行真实后端验算：

```bash
python tools/validate_winds_backend.py
```

该工具只用解析合成场；缺后端时明确失败，不把跳过或 mock 测试当作数值验证。

## 合成示例与配置

```bash
python examples/generate_synthetic.py --output-dir ./demo_input
python -m climate_dynamics --config ./demo_input/tn_waf.json
python -m climate_dynamics --config ./demo_input/rossby_wave_source.json
python -m climate_dynamics --config ./demo_input/linear_rws.json
python -m climate_dynamics --config ./demo_input/mass_streamfunction.json
# 需要真实球谐后端：
python -m climate_dynamics --config ./demo_input/wind.json
```

生成器创建全合成的数值 NPZ 文件和五份配置，要求目标目录不存在。自己的资料可参考 [配置模板](examples/config.example.json)，替换 `<DATA_ROOT>`、`<OUTPUT_ROOT>`。CLI 使用 `allow_pickle=False`，只接受实数数值数组，并要求明确声明物理单位；**不会隐式转换单位**。输入为已经选好时间、层次、网格和物理定义的诊断场，直接读取原始 GRIB、季节合成和回归不属于这个 CLI 的步骤；可分别使用 [Preprocessing](../Preprocessing/README.md) 和 [Statistical](../Statistical/README.md)。

| `method` | NPZ 必需数组 | 输出及单位 |
| --- | --- | --- |
| `tn_waf` | `lat, lon, psi, u_background, v_background` | `waf_x, waf_y`，m² s⁻²；单个配置压力层 |
| `rossby_wave_source` | `lat, lon` 和下述六个分量 | `stretch, advection, rws`，s⁻² |
| `linear_rws` | `lat, lon`，六个 `background_*` 和六个 `perturbation_*` 分量 | 四项一阶贡献与 `rws`，保留输入扰动/回归系数尺度 |
| `wind` | `lat, lon, u, v` | 流函数、速度势、涡度、散度、分量风及绝对涡度梯度 |
| `mass_streamfunction` | `lat, lon, pressure, v`；CLI 的 `v=(sample,pressure,lat,lon)` | 固定支撑经度均值、支撑诊断、相对质量流函数 kg s⁻¹ |

所有波源分量的末两维为纬度、经度，六个分量为 `eta`（绝对涡度，s⁻¹）、`div`（散度，s⁻¹）、`udiv/vdiv`（辐散风，m s⁻¹）、`eta_x/eta_y`（物理水平梯度，s⁻¹ m⁻¹）。输入经纬度单位为度，压力积分使用 **Pa**；TN 配置的 `pressure_hpa` 使用 **hPa**。配置的 `units` 使用示例中的明确字符串，不能把 hPa 仅改标签为 Pa。

`linear_rws` 的 `units` 声明物理基准单位；可另用 `predictor_unit` 声明扰动系数的分母单位（缺省 `"1"`，即无量纲幅度）。例如 `predictor_unit="K"` 表示六个扰动分量均为每 K 预测量的系数，输出相应记录为 s⁻²/K；数值不会重新标准化。背景分量仍为物理基准单位。

输出 NPZ 的 `_metadata_json` 是可用 `json.loads(str(archive["_metadata_json"]))` 读取的文本，记录文献、参数、单位、输入 SHA256 和方法限制，不写入私人输入绝对路径。输出完成后在同目录用硬链接原子发布；文件系统须支持该操作，已有文件或符号链接均不覆盖。若计算失败不发布部分结果。

## Python 接口与衔接

```python
from climate_dynamics import diagnose_wind, tn_waf, rossby_wave_source, linear_rws

# u/v 为完整全球场，形状 (..., lat, lon)。返回纬度一律升序。
background = diagnose_wind(u_mean, v_mean, lat, lon, truncation=42)
# 单独诊断扰动风时必须排除固定的 f 及其梯度：
perturbation = diagnose_wind(u_pert, v_pert, lat, lon, truncation=42, omega=0.)
source_mean = rossby_wave_source(background)
source_linear = linear_rws(background, perturbation)
flux = tn_waf(perturbation["psi"], background["u_total"], background["v_total"],
              background["lat"], background["lon"], pressure_hpa=200.)
```

这段示例将背景风也截断到指定阶数；`u_total/v_total` 是保留谱分量之和，可能不同于原风。两个风场诊断必须采用相同网格、坐标、半径与截断，不能把升序诊断结果与未重排的降序风场混用。也可以向 TN 显式提供按研究定义准备的背景总风。

质量流函数接口：

```python
from climate_dynamics import fixed_longitude_mean, mass_streamfunction

means, fraction, mask, averaging = fixed_longitude_mean(
    {"v": v, "omega": omega}, longitude=lon,
    longitude_bounds=(170., -170.), cyclic=True, min_fraction=0.9,
)
psi_mass, integration = mass_streamfunction(
    means["v"], pressure_pa, lat, reference_pressure_pa=pressure_pa[0],
)
```

Python 均值接口支持任意前导样本维，用 `sample_axes` 指定需共同有效的轴；缺省为轴 0，经度缺省末轴。质量积分缺省 `(..., pressure, latitude)`，可指定 `pressure_axis/latitude_axis`。各数组及坐标须先对齐；`longitude_cell_weights` 还可接实际 `cell_bounds`，避免从中心点推断边界。

## 科学约定

- **TN：** [Takaya 与 Nakamura（2001）](REFERENCES.md) 的静止水平形式。按弧度差分，周期经度要求完整均匀全球 `[0,360)` 网格；纬度升序，可非均匀。默认排除两极、`|lat|<=10°`、总风速小于 5 m/s 及非西风格点。缺测不填补，连续差分模板不跳过缺测格点。`psi` 是给定旋转流函数扰动；代码不把位势除以科氏参数。通量对幅度是**二次量**；多个压力层须显式按层循环，不能将同一个压力参数无声套给不同层。
- **波源：** [Sardeshmukh 与 Hoskins（1988）](REFERENCES.md) 的 `S=-eta*div-Vchi·grad(eta)`。完整源为非线性量，一阶源省略扰动乘扰动项。背景 `eta` 含行星涡度；扰动 `eta` 及其梯度不再加入固定行星项。输入分量回归系数须使用同一预测量尺度。每个结果的各分项与总和共用有限掩膜，不做显著性筛选。
- **球谐：** 使用 [Dawson（2016）及 SPHEREPACK 方法](REFERENCES.md)。只接受完整有限全球风，不补地下或缺测风。规则纬度包含真正的南北极；高斯纬度匹配完整 Legendre 节点。经度须均匀递增且无循环重复端点。统一谱截断用于全部诊断，输出保持前导维和经度顺序、纬度变为升序。谱势函数的零阶规范及东西/南北符号由后端保留。
- **质量流函数：** 参照 [Oort 与 Yienger（1996）、Adam 等（2018）](REFERENCES.md) 的纬向质量环流背景及连续方程定义，计算 `2*pi*a*cos(phi)/g * integral[p_ref,p](v_mean dp)`。首输入层设零；若该层缺测则零参考也无效。压力向下递增，梯形积分遇缺层后向下全为缺测。没有假定参考层以上风为零，不作上边界外推，也不自动补地表以下层。
- **经度支撑：** 用单元交宽加权，在所有指定字段与样本中选固定共同支撑；覆盖率不足的整行输出缺测，真实零值保留。有限区域或不完整经度支撑的流函数是**整圈等效相对量**，不是区域实际总质量输送，也不能据此自动声称闭合的二维翻转环流。

默认半径 6,371,000 m、角速度 7.292115×10⁻⁵ s⁻¹、重力加速度 9.80665 m s⁻²，接口允许显式配置。源码原算法对照、解析场测试与真实后端验算见 [validation.json](validation.json)。这些合成检查不替代具体资料的科学质量评估。

## 引用与贡献

使用时请引用实际采用的原始方法论文、上游软件和再分析资料，并注明本仓库提交号。完整条目和实现与原方法的边界见 [REFERENCES.md](REFERENCES.md)。本文与计算代码区分原始方法作者、软件开发者和本仓库整理者；不新增或替代第三方许可条件。
