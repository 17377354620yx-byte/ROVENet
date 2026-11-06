# 历史分支与本次基准的差异

参考分支 `backup/p2i-before-bf16-20260914` 在本地 Git 对象中解析为 `cc68d8786186966f6834aec695fa435f65f29e04`，与指定源工作树的 `HEAD` 完全相同。GitHub 的该分支页面未能直接读取，因此使用本地同一提交的 Git 对象逐文件核对了 `model.py`、`config.py`、`loss.py`、`dataset.py`、`trainval.py`、`test.py`、RTOR、A3、coarse/fine matching 与 registration。当前 ROVENet 初始版本还包含源工作树的未提交修改和新增文件，以下是相对于 `cc68` 的差异。

三个已跟踪 P2P 文件的完整逐行差异见 [HISTORICAL_SOURCE_DIFF.md](HISTORICAL_SOURCE_DIFF.md)。本地核对直接读取了该分支的 12 个源码文件，而非仅比较文件名或提交说明。

| 文件/区域 | 差异 | 分类 | 对算法的影响 |
| --- | --- | --- | --- |
| `experiments/geotransformer.p2p_liver/config.py` | 新增 `P2P_TEST_DATA_ROOT`，使 test root、list、statistics 可单独指定；cooperative 校验不再检查 `architecture` 必须为 `rtor_a3`，仍约束单编码器与原 LGR。 | 新功能、工程接口 | 完整 RTOR+A3 的数值参数及计算路径不变。 |
| `experiments/geotransformer.p2p_liver/trainval.py` | checkpoint 协议记录 `geometry_weight`；恢复时核对 architecture、geometry weight、seed；CLI 增加 seed 与 geometry weight。 | 工程可追溯性、新功能 | 默认值不变；显式传入非默认 geometry weight 会改变 A3 几何偏置，属于用户主动的算法参数变化。 |
| `experiments/geotransformer.p2p_liver/test.py` | 从 checkpoint 恢复 geometry weight；增补 RTE、点 RMSE、Euler 分量误差、SR@5、严格 RR 等指标及数据根路径。 | 评估新功能 | 模型推理和旧指标公式不变，增加的指标只是报告字段。 |
| `experiments/geotransformer.ircadb_v3/` | 新的 3Dircadb 数据实验。 | 新功能 | 不覆盖 P2P 模型参数及数据协议。 |
| `experiments/Liver/`、`experiments/geotransformer.p2i_lreg/`、消融脚本 | 源工作树新增或保留的旁系实验与结构消融。 | 历史遗留代码 | 本次移至 `archive/`，不参与正式 ROVENet 构建。 |
| `geotransformer/modules/liver/{registration_model,topology_overlap,fine_local_refiner,cooperative_matching,overlap_selection}.py` | 源工作树相对 `cc68` 无未提交源码差异；本次只移动模块、重命名顶层类及删除关闭 RTOR/A3 的旁路。 | 明确的工程重构 | 完整 RTOR+A3 路径保持数值等价。 |

**真实算法差异：** 在指定历史提交与本次重构前的完整 RTOR+A3 核心模块之间，未发现源码差异。唯一可通过用户参数引入的数值变化是训练 CLI 的 `--geometry_weight`；默认值仍为 `1.0`。新增的 3Dircadb 实验有不同的数据尺度及实验配置，属于单独任务，不是 P2P 网络重设计。

历史研究结果与图表中的旧路径作为原始证据保留；它们没有被新路径的实验结果替换。完整源代码可在本仓库初始提交 `0d259cd` 与本地 Git 历史提交 `cc68d878` 中复核。
