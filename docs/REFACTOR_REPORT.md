# ROVENet 正式架构重构报告

## 范围与保护

本次仅修改独立仓库 `/home/yangx/code/new_deform/ai_worker/ROVENet` 的 `refactor/rovenet-fixed-architecture` 分支。开始时 Git 工作区干净，基线为初始提交 `0d259cd`。未改动 `/home/yangx/code/new_deform/ai_worker/RTORv2`，未删除任何源工作树的 checkpoint、训练输出、评估结果或数据。旧消融代码与旁系实验保存在本仓库 `archive/`；已有研究文档与图表保留。

## 结构变化

| 以前 | 现在 | 处理 |
| --- | --- | --- |
| `geotransformer/modules/liver/registration_model.py` | `rovenet/models/registration.py` | 顶层类 `GeoTransformer` 改为 `ROVENet`；固定组装完整网络。 |
| `geotransformer/modules/liver/topology_overlap.py` | `rovenet/modules/topology_overlap.py` | RTOR 实现原样移动，包含局部拓扑、Poincaré 距离、双侧交互、overlap logits。 |
| `geotransformer/modules/liver/fine_local_refiner.py` | `rovenet/modules/fine_refinement.py` | A3 实现原样移动，保留双向几何注意力、residual gate 和参数名。 |
| `geotransformer/modules/liver/{overlap_selection,cooperative_matching}.py` | `rovenet/modules/` | 保留 RTOR overlap 筛选和训练 proposal 混合。 |
| P2I-LReg、其他 Liver 实验与模块消融脚本 | `archive/` | 历史材料保留，退出正式安装包与训练入口。 |
| `geotransformer/` 公共组件 | 原路径 | KPConv、GeometricTransformer、Sinkhorn、LGR、C++ 算子保持不变。 |

新增 `pyproject.toml`、`LICENSE`、架构文档和回归测试。`requirements.txt` 的 Open3D 版本改为本机已验证的 `0.16.0`，修正旧清单 `0.11.2` 与当前 Python 3.10 环境不一致的问题。正式的 P2P Liver 与 3Dircadb 实验仍在原目录，训练和测试脚本路径不变。

## 消融清理与保留的算法

- 删除 `cfg.ablation`、`self.rtor_enabled`、`self.a3_enabled`，及 RTOR/A3 条件构建、关闭时的 overlap 常量旁路、A3 跳过分支和关闭 RTOR 时跳过 overlap loss 的分支。
- `self.topology_overlap_refiner`、`self.fine_local_refiner` 始终构建并参与前向，原有属性名与 `state_dict` 前缀保留。
- P2P 与 3Dircadb 的 `--architecture` 参数被显式拒绝，旧消融脚本移至 `archive/`；新 checkpoint 记录 `architecture='ROVENet'`，旧 checkpoint 中的 `architecture='rtor_a3'` 继续识别。
- 保留 `legacy`、`cooperative`、`soft_overlap` profile，因为它们控制 proposal schedule/overlap region 策略，不决定模块是否存在。保留训练/推理判断、有效性检查及 registration profile 参数。双编码器实验路径随后从正式网络移除，旧单编码器 checkpoint 仍可严格加载。
- RTOR 的局部几何、双曲距离、双侧图交互、overlap 权重与 focal loss；A3 的 patch 几何签名、双向 attention、residual 和细匹配路径均未重写。
- 未调整网络维度、损失权重、采样、匹配阈值、Sinkhorn 迭代次数或 LGR 参数。

## 历史版本核对

指定分支 `backup/p2i-before-bf16-20260914` 在本地解析为 `cc68d878`，与源工作树 HEAD 相同。逐文件核对及相对未提交修改的分类见 [HISTORICAL_DIFF.md](HISTORICAL_DIFF.md)。官方 [GeoTransformer](https://github.com/qinzheng93/GeoTransformer) 的仓库结构和安装入口用于组织参考；其算法没有覆盖 ROVENet 模块。

## Checkpoint 与数值等价性

基准为重构前 `GeoTransformer` 的完整 `rtor_a3` + `cooperative` 配置，使用原源工作树的 `rtor_a3_cooperative_cc68_seed35/snapshots/best.pth.tar`、P2P in-silico 真实首个样本、相同输入/权重/环境及固定 Python、NumPy、PyTorch 随机种子 `42`。新旧都在同一台机器上使用 `geo_v2` 环境与 GPU。先将旧模型的全部张量输出保存至 `/tmp/rovenet_before.pt`，再运行新模型逐项比较。

| 项目 | 结果 |
| --- | --- |
| 模型参数总量 | 新旧均为 `10,621,188`。 |
| `state_dict` 键与 tensor shape | 307 个键的名称、顺序及形状完全相同。 |
| 旧 checkpoint 加载 | `model.load_state_dict(checkpoint['model'], strict=True)` 成功。 |
| 张量输出 | 40 个键一致；包含 coarse/fine correspondences、overlap logits、`matching_scores`、最终 `estimated_transform`。最大绝对误差 `0.0`。 |
| Loss | `loss`、`c_loss`、`f_loss`、`o_loss` 四项最大绝对误差 `0.0`。 |
| 评估指标 | PIR、IR、RRE、RTE、RMSE、RR 六项最大绝对误差 `0.0`。 |
| 比较容差 | 浮点张量 `atol=1e-5`、`rtol=1e-5`，非浮点张量要求完全相同；本次实测浮点结果逐项精确相同。 |

单样本 `test.py` 在旧 checkpoint 上严格加载并导出 JSON/CSV，RMS-TRE 为 `4.3403 mm`。3Dircadb 的旧完整 checkpoint 也单样本验证（见下方测试记录）。这些 smoke test 用于验证运行链路，不表示整套测试集性能。

## 测试记录

- 当前 `tests/`：29 个测试全部通过（设置 `ROVENET_LEGACY_CHECKPOINT` 后运行）；回归测试对完整旧 checkpoint 的严格加载已通过。重构前的 55 个测试与 2 个子测试曾全部通过，旧 P2I/消融相关测试随代码归档，不属于固定架构的当前测试集。
- P2P Liver：2 个训练样本 + 2 个验证样本、1 epoch 的训练 smoke test 完成，生成训练侧 `best.pth.tar`；`c_loss`、`f_loss`、`o_loss` 均参与计算。
- P2P Liver：旧 checkpoint 单样本 in-silico 测试通过，结果导出至 `/tmp/rovenet_smoke_eval.json`。
- 3Dircadb：旧完整 checkpoint 严格加载且单样本测试通过，RMS-TRE 为 `3.267 mm`，结果导出至 `/tmp/rovenet_ircadb_smoke.json`。
- 包构建：`python -m pip wheel --no-deps --no-build-isolation .` 成功生成 `rovenet-0.1.0-cp310-cp310-linux_x86_64.whl`；自定义扩展由原源码重新编译。
- 退役 CLI：`trainval.py --architecture rtor_only` 以退出码 `2` 明确报错，避免旧参数静默失效。

局限：本次没有从头完成 150 epoch 训练，也没有重新跑完整测试集；重构报告不对模型性能作新结论。`archive/` 下的历史消融代码仅保留供追溯，不保证能在固定架构 API 上运行。

## 后续清理：双编码器实验路径

正式 ROVENet 仅使用一次 `KPConvFPN` 前向，同时提取 coarse 与 fine 特征。删除了 `self.fine_backbone` 的复制、条件前向与 `cfg.model.dual_encoder` 配置。P2P Liver 和 3Dircadb 的训练、测试入口显式拒绝旧 `--dual_encoder` 参数；新 checkpoint 不再写入此字段。旧 checkpoint 若记录 `dual_encoder=false` 或没有该字段，仍可严格加载；记录 `true` 的 checkpoint 明确报错。历史权重组合脚本保存在 `archive/diagnostics/compose_encoders.py`。

在相同旧单编码器 checkpoint 与固定 P2P 样本上再次核对全部参数形状、模型张量输出、loss 和指标，确认与重构前基准一致。此清理未触及 RTOR、A3、matching 或 registration 的计算。
