# 原代码来源与提取边界

[English](SOURCE_PROVENANCE_EN.md)

**作者：RH_Deng；Codex（OpenAI，AI辅助整理与重构）**  
核对日期：2026-10-25

## 来源关系

本包仅整理 **LBM T42L20 干模式三维背景态** 的制作代码。来源为原 HPC 背景态制作程序及其参考副本，包括原 DJF 制作程序、季节扩展程序、水平与垂直数值模块、二进制模块及原单元测试。季节扩展程序支持 MAM、AMJ、MJJ、JJA，用作移除固定实验时段、分离季节配置的参考。

本次通过只读操作核对远端源代码，提取必要程序和原有测试；没有修改远端文件、提交作业或运行模型。10 项现有本地来源副本与对应远端文件逐字节 SHA256 一致。以下仅记录通用逻辑来源、源文件名及指纹，不公布内部项目名称、目录、服务器或账户。

## 关键源文件指纹

| 逻辑来源 | 文件 | SHA256 |
| --- | --- | --- |
| 原 DJF 制作程序 | `make_merra2_background.py` | `ad3de26cfe59737634872a61f8574363521761e6af4d2e8ea838d4413fcfcde5` |
| 季节扩展程序 | `make_merra2_background.py` | `2a2106b030ef9da26a57a15dbf434fdaff9be56122d27f57d3115f9e6e7b7da5` |
| 两套制作程序及参考副本中的通用数值模块 | `lbm_horizontal.py` | `f3c07d691dcaf0f6d15611a5f3996a66fce4ecf4b8b5078dfe0b1ccb4fe2307c` |
| 同上 | `lbm_vertical.py` | `355474a66d5e3428709190200d03cca9c1324568bd1fed577bb8e005086aaec7` |
| 同上 | `lbm_binary.py` | `4ff7b00bb48ccf83a151a85ceb2a7f3590864616e037a6ac142f9b51ed492f2f` |
| 季节扩展程序的原单元测试 | `test_seasonal_builder.py` | `f5c3f93b421324c8284439fd4f3ef148aec3197fed83789607b9b480473b381c` |
| 同上 | `test_lbm_binary.py` | `8bb3d5c3a0d03967dfd63560f9cf920f912be2db326eba00fbd304816d9d2bde` |
| 同上 | `test_lbm_horizontal.py` | `89b6b5d630566eff27d22ff033d2ed4d344809fc0a31696b21fbbce505900d63` |
| 同上 | `test_lbm_vertical.py` | `6a1e1e2cb8a62fe92a964b9b2ecb3223ea88e8fadff11317fafda85c76d83138` |

三个通用模块在原 DJF 制作程序、季节扩展程序及参考副本中的源文件指纹一致。这些指纹标识的是**整理前的原文件**；包内模块经重构或接口适配后，不应要求其哈希仍相同。包内文件身份以本次分发的校验清单为准。

## 保留与重构

保留的科学路线包括月气候态和等月权季节平均、精确 Gaussian 网格、周期缺测感知双线性插值、`ecmsbs` 地形/虚温地面气压订正、`ln(p)` 自然三次样条与常值端点延伸、全部 σ 层的 T42 标量投影及 Gtool3/GrADS 文件规则。

新的可复用接口将数据定位、CF 时间和坐标、单位归一、季节年份、模型参考文件和输出目录移入配置；不再要求原 HPC 的目录、任务状态或强制下载器收据。原代码中的固定产品文件名、固定时段标签和安装步骤被解耦；源说明中的历史生产验收数据没有沿用为本包验收结论。

模型地形 `grz.t42`、σ 轴 `GTAXLOC.CSIG20` 和格式模板 `erawin.t42l20` 仅用于本次私有环境的真实格式集成测试，不随代码分发。该测试使用合成再分析输入，因此不证明新的真实再分析背景已获科学验收，也不证明模式积分稳定。

## 参考说明

原制作程序引用的 LBM 使用说明：

[Linear Baroclinic Model 使用指南](https://github.com/HanXiang0721/Note-Numerical-Methods/blob/master/Linear-Baroclinic-Model%E4%BD%BF%E7%94%A8%E6%8C%87%E5%8D%97.md)

参考链接用于交代原程序采用的处理背景，不表示第三方作者参与本次重构。模型、数据与说明的原有作者归属和使用条件保持独立；本包作者署名不替代其来源说明。
