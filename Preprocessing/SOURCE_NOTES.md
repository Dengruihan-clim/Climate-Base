# Algorithm and provenance notes / 算法与来源说明

Authors / 作者：RH_Deng; Codex (OpenAI, AI-assisted organization and refactoring / AI 辅助整理与重构)

The reusable implementation was refactored from author-owned CF monthly-input and complete-season reduction kernels. The selected original readers/reducers were checked against their deployed copies by read-only SHA256 comparison before refactoring. Internal project names, server locations and connection details are excluded from this distribution; the private comparison record is maintained separately.

本实现从作者既有 CF 月度读入与完整季节归约内核重构；重构前对选定原源与部署副本进行只读 SHA256 对照。公开目录不保留内部项目名、服务器位置或连接资料，私人对照记录另存。

Preserved methods / 保留的方法：

- Decode actual CF time, reject missing/duplicate requested months, and verify variable dimensions/units and native coordinates. 按实际 CF 时间选月，拒绝缺月/重复月，并检查维度、单位与原生坐标。
- Keep missing values and require every season month at each output cell. 保留缺测，每个季节格点均要求各月齐全。
- Define cross-year seasons explicitly and use declared day/equal-month weighting. 明确跨年季节与日加权/等月平均。
- Normalize coordinates by reindexing rather than interpolating. 以重排索引规范化坐标，不插值。

Generalizations / 泛化：

- Configurable CF coordinate/variable names, annual multi-record files and monthly templates. 可配置 CF 名称、年度多记录文件及逐月模板。
- Native rectilinear grids, optional shared-grid validation, Pa/hPa pressure axes, actual CF calendars and explicit affine unit transformations. 原生直角经纬网格、可选共同网格核验、Pa/hPa 气压、真实 CF 日历及显式仿射单位变换。
- Metadata-only field inspection, relative-path/hash provenance, streaming seasonal export and non-overwrite publication. 不解码场值的元数据检查、相对路径/哈希追溯、流式季节写出与拒覆盖发布。

No third-party model source, private data, derived research results or credentials are included. NumPy, netCDF4, cftime and pytest remain external dependencies with their own licenses. This note records provenance and does not create or change licensing terms for code, data or third-party software.

未附第三方模式源码、私人数据、研究结果或凭据；所列 Python 库作为外部依赖使用，其许可分别由发布方提供。本说明仅记录来源，不为代码、资料或第三方软件新增或改变许可条款。

Format reference / 格式参考：[CF Metadata Conventions](https://cfconventions.org/).
