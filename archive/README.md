# 历史实验归档

此目录保留独立项目整理前的实验代码与研究记录，供追溯使用，不属于正式 ROVENet 训练和测试入口。

- `p2i_lreg/` 与 `tests/`：旧 P2I-LReg 架构对比实验，含关闭 A3/RTOR 的消融路线；其配置与正式固定网络不兼容。
- `Liver/`：原工作树中的其他肝脏实验实现。
- `run_module_ablation.sh`、`run_three_arm_ablation.sh` 和 `MODULE_ABLATION_PLAN.md`：历史模块消融流程。
- `diagnostics/`、`build_sci_evidence.py`：依赖旧消融接口的研究诊断与证据生成流程，连同原始报告保留。
- `diagnostics/compose_encoders.py`：旧双编码器权重组合实验，正式 ROVENet 不再构建第二套 backbone。

这些文件没有用新模型重写，历史实验结果和数值定义保持原样。正式入口位于 `experiments/geotransformer.p2p_liver/` 和 `experiments/geotransformer.ircadb_v3/`。
