# ROVENet 模型组件与数据流

ROVENet 是单刚体 `R,t` 的 complete-to-partial 点云配准网络。顶层类为 [`rovenet.models.registration.ROVENet`](../rovenet/models/registration.py)，P2P Liver 与 3Dircadb 实验通过各自的 `model.py` 创建同一网络。`geotransformer/` 提供公共实现；`rovenet/` 放置 RTOR、A3 和随它们训练的 proposal/overlap 逻辑。`GeometricTransformer` 是内部几何注意力模块，没有改名。

## 前向传播调用位置

| 顺序 | 模块与代码 | 输入、输出及作用 |
| --- | --- | --- |
| 1 | `ROVENet.forward` 的 `point_to_node_partition` | 将细点分配到粗 superpoint 的局部 patch；若有监督刚体变换，计算 GT 节点对应与 overlap。 |
| 2 | `self.backbone` (`KPConvFPN`) | 单次前向同时提取 fine 和 coarse 特征。 |
| 3 | `self.transformer` (`GeometricTransformer`) | 在 reference/source 粗节点上编码距离和角度上下文，并 L2 归一化。 |
| 4 | `self.topology_overlap_refiner` (`TopologyOverlapRefiner`, RTOR) | **每次前向都执行**。分别建立两侧局部 kNN 图，提取归一化距离、曲率、线性度、平面度及 Poincaré 距离，进行边门控与图更新，再做双侧跨云交互。输出两侧残差细化的粗特征与 `ref_overlap_logits` / `src_overlap_logits`。 |
| 5 | `_apply_rtor`、`select_overlap_region`、`self.coarse_matching` | RTOR logits 经 sigmoid 转为两侧 overlap 概率，并通过保留原值的 floor/power 映射形成粗匹配权重。推理时仅在配置启用时筛选完整 source 的节点；partial reference 保持完整。粗匹配使用两侧权重。RTOR 始终运行，overlap 筛选是 profile 内部策略。 |
| 6 | `self.coarse_target`、`mix_coarse_proposals` | 训练时以 GT 节点对应提供 fine patch 监督；cooperative profile 按原 schedule 混入预测 coarse candidates，保留原始 GT 预算和去重规则。 |
| 7 | `self.fine_local_refiner` (`GeometryAwareFineRefiner`, A3) | **每次前向都执行**。从成对 fine patch 的局部坐标计算中心化、尺度归一化的径向、距离均值和方差特征，形成几何注意力偏置；reference/source 双向使用更新前特征，经过 attention、前馈与可学习 residual gate 输出细化特征。掩码及参数名 `layers.0.*` 保持不变。 |
| 8 | `self.optimal_transport` | 细特征点积、通道缩放与 Sinkhorn，输出 `matching_scores`。 |
| 9 | `self.fine_matching` (`LocalGlobalRegistration`) | 按原配置选出 fine correspondences，执行局部到全局的刚体估计，输出 `ref_corr_points`、`src_corr_points`、`corr_scores`、`estimated_transform`。 |

完整网络始终保留 `self.topology_overlap_refiner` 和 `self.fine_local_refiner` 两个属性名及其 state_dict 前缀。RTOR 的 Poincaré 计算在当前实现中固定执行，没有 `use_poincare` 开关；`poincare_curvature` 参数保留。`rtor_residual_scale` 的原值 `1.0` 未修改。`fine_refiner.geometry_sigma`、`geometry_weight`、`residual_init`、注意力头数与 dropout 未调整。

## 损失与评估

P2P Liver 的 `OverallLoss` 始终计算 coarse circle loss、fine matching loss 与双侧 RTOR overlap focal loss，合成为 `weight_coarse * c_loss + weight_fine * f_loss + weight_overlap * o_loss`。权重、正样本半径、Sinkhorn 迭代次数和 registration 参数均沿用重构前配置。A3 经 fine matching loss 接受梯度；RTOR 经 coarse loss 及 overlap loss 接受梯度。

训练时 `trainval.py` 仍按独立 deformation group 验证，保存训练侧最优 `best.pth.tar`。测试时 `test.py` 严格加载 checkpoint；保留 in-silico 与 in-vitro 的数据预处理、marker RMS-TRE 毫米单位、RRE/RTE、成功率和逐样本导出。`legacy`、`cooperative`、`soft_overlap` 是原有 proposal 与 overlap 筛选策略，均使用同一完整 RTOR+A3 网络。
