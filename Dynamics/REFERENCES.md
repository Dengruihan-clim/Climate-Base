# 原始方法、软件与贡献 / Methods, Software, and Contributions

**本仓库工具整理与重构 / Tool organization and refactoring: RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring).**

下面的作者应分别获得原始物理方法与软件的归属。本仓库署名对应配置接口、代码整理、测试和说明，不表示提出这些原始方法。计算文件的 docstring 和 `climate_dynamics/citations.py` 同样保留引用；命令行输出包含实际方法对应的文献元数据。

The references below credit the physical methods and upstream software separately. Repository authorship covers interfaces, organization, tests, and documentation. Citations also appear in kernel docstrings, `climate_dynamics/citations.py`, and the metadata of command-line outputs.

## 波活动通量 / Wave activity flux

**Takaya, K., and H. Nakamura, 2001:** A Formulation of a Phase-Independent Wave-Activity Flux for Stationary and Migratory Quasigeostrophic Eddies on a Zonally Varying Basic Flow. *Journal of the Atmospheric Sciences*, **58**, 608–627. [DOI: 10.1175/1520-0469(2001)058<0608:AFOAPI>2.0.CO;2](https://doi.org/10.1175/1520-0469%282001%29058%3C0608%3AAFOAPI%3E2.0.CO%3B2).

`tn_waf` 使用该框架的**静止扰动水平通量形式**；没有实现完整三维或移动波通量。输入为给定旋转流函数扰动及背景总风，调用者须判断准地转近似和背景流条件是否适合。经纬导数按弧度差分；压力归一化为 `pressure_hpa/1000`。通量对扰动幅度是二次量；从回归流函数型算出的通量不能标成线性回归系数。

`tn_waf` implements the **stationary horizontal** form. It does not implement the full three-dimensional or migratory-wave formulation. Users must assess the quasi-geostrophic and background-flow assumptions for the supplied rotational streamfunction pattern. Derivatives use radians and pressure is normalized by 1000 hPa. Flux is quadratic in perturbation amplitude, including when its input is a regression pattern.

## Rossby 波源 / Rossby wave source

**Sardeshmukh, P. D., and B. J. Hoskins, 1988:** The Generation of Global Rotational Flow by Steady Idealized Tropical Divergence. *Journal of the Atmospheric Sciences*, **45**, 1228–1251. [DOI: 10.1175/1520-0469(1988)045<1228:TGOGRF>2.0.CO;2](https://doi.org/10.1175/1520-0469%281988%29045%3C1228%3ATGOGRF%3E2.0.CO%3B2).

`rossby_wave_source` 计算 `S = -eta*div - Vchi·grad(eta)`；`eta` 是绝对涡度，梯度为每米的物理梯度。`linear_rws` 为同一表达式的直接一阶变分，保留两项伸展与两项平流贡献；没有将该代数展开作为另一项原创方法。扰动绝对涡度的变化不再包含固定的行星涡度，因而单独诊断扰动风时须使用 `omega=0`。线性源不等于有限非线性场差，也不等于对瞬时非线性波源的回归。

The full source is `S = -eta*div - Vchi·grad(eta)`, using absolute vorticity and physical gradients per metre. The four-term `linear_rws` is its algebraic first variation. A separately diagnosed wind perturbation must use `omega=0` to exclude the fixed planetary term. This first variation differs from a finite nonlinear difference and from regression of instantaneous nonlinear sources.

## 球谐风场与上游计算 / Spherical winds and upstream computations

**Dawson, A., 2016:** Windspharm: A High-Level Library for Global Wind Field Computations Using Spherical Harmonics. *Journal of Open Research Software*, **4**(1), e31. [DOI: 10.5334/jors.129](https://doi.org/10.5334/jors.129).

**Adams, J. C., and P. N. Swarztrauber, 1999:** SPHEREPACK 3.0: A Model Development Facility. *Monthly Weather Review*, **127**, 1872–1878. [DOI: 10.1175/1520-0493(1999)127<1872:SAMDF>2.0.CO;2](https://doi.org/10.1175/1520-0493%281999%29127%3C1872%3ASAMDF%3E2.0.CO%3B2).

`diagnose_wind` 是 `windspharm.standard.VectorWind` 的输入验证、轴变换和元数据封装。实际球谐变换、Helmholtz 分解和谱导数由 **windspharm → pyspharm → SPHEREPACK** 完成，应分别保留这些软件的作者与许可说明。本仓库不分发其源码或编译程序，也不以有限差分模拟缺失后端。

The adapter validates and rearranges inputs around `windspharm.standard.VectorWind`. The **windspharm → pyspharm → SPHEREPACK** stack performs the transforms, decomposition, and spectral derivatives. Retain upstream attribution and licensing. Upstream sources and binaries are not bundled, and no finite-difference substitute is used.

- [windspharm documentation](https://ajdawson.github.io/windspharm/)
- [VectorWind API](https://ajdawson.github.io/windspharm/api/windspharm.standard.html)
- [windspharm source](https://github.com/ajdawson/windspharm)
- [pyspharm source](https://github.com/jswhit/pyspharm)

## 经向质量流函数 / Meridional mass streamfunction

**Oort, A. H., and J. J. Yienger, 1996:** Observed Interannual Variability in the Hadley Circulation and Its Connection to ENSO. *Journal of Climate*, **9**, 2751–2767. [DOI: 10.1175/1520-0442(1996)009<2751:OIVITH>2.0.CO;2](https://doi.org/10.1175/1520-0442%281996%29009%3C2751%3AOIVITH%3E2.0.CO%3B2).

该文是以月平均纬向平均经向风计算质量流函数的经典观测应用；此处并不把本模块的参考压力或区域掩膜归于该文。/ This is an earlier observational application using monthly zonal-mean meridional wind. The reference-pressure and regional-mask choices in this module are not attributed to that paper.

**Adam, O., K. M. Grise, P. Staten, I. R. Simpson, S. M. Davis, N. A. Davis, D. W. Waugh, T. Birner, and A. Ming, 2018:** The TropD software package (v1): standardized methods for calculating tropical-width diagnostics. *Geoscientific Model Development*, **11**, 4339–4357. [DOI: 10.5194/gmd-11-4339-2018](https://doi.org/10.5194/gmd-11-4339-2018).

引用该文讨论的质量连续方程与标准质量流函数定义；没有复制 TropD 实现，也没有实现其热带边界检出工具。这里计算 `Psi = 2*pi*a*cos(phi)/g * integral[p_ref,p](v_mean dp)`，以首个输入层 `p_ref` 为零参考并用相邻层梯形积分。缺层后不重新起算，未知参考层以上的积分不补零、不外推。区域经度平均风产生的是**整圈等效的相对流函数**，不等同于该区域的实际总质量输送；有完整全球经度支撑时才可作相应全球纬向平均解释。

This is a reference for the continuity-based streamfunction definition, not copied TropD code or its tropical-edge detection. The present calculation uses a zero reference at the first supplied level and adjacent-level trapezoids. Missing layers terminate deeper valid integration; no unknown upper-layer contribution is added. Regional means yield a **full-circumference-equivalent relative streamfunction**, rather than actual band-integrated transport. A global zonal interpretation requires complete global longitude support.

经度单元交宽和共同有限支撑是本仓库的数值实现约定，不作为上述文章新增的方法归属。具体配置、压力参考层、区域、截断阶数、地球参数与实际软件版本应随研究报告。

Longitude-cell intersection widths and fixed common support are implementation choices here. Report the chosen pressure reference, domain, truncation, planetary constants, and actual software versions in applications.

## 使用时引用 / Citation in applications

引用实际使用的方法论文、再分析资料和软件，并标明 [Climate-Base](https://github.com/Dengruihan-clim/Climate-Base) 的提交号。本说明不替代原始资料与软件的许可条件。

Cite the methods, datasets, and software actually used, together with the [Climate-Base](https://github.com/Dengruihan-clim/Climate-Base) revision. This note does not replace their licensing conditions.
