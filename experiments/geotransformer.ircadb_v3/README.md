# 3Dircadb dataset_v3：完整到局部肝脏刚性初始化配准

本目录把 `liver_deform_output/dataset_v3` 直接用于 RTOR+A3。模型只预测 `R,t`，目标点云保留 FEM 非刚性残差。坐标和标签均为米，列向量记法为 `target ≈ R @ source + t`，行向量记法为 `target ≈ source @ R.T + t`。训练输入为完整术前表面 4096 点和局部术中表面 1024 点，独立采样，没有共享点索引。`evaluation.npz` 的内部 128 个体点只用于验证和测试的 RMS-TRE，不作为模型输入。

## 数据划分与参数

- 训练：模型 5、6、7、8、9、10、13、14、15、16，共 9300 对。
- 验证：模型 17，共 930 对。按其内部点 RMS-TRE 选 `best.pth.tar`。
- 测试：模型 1、3，共 1860 对。测试集不用于调参和检查点选择。
- 相对 P2P 基础配置，体素大小改为 4 mm，邻域半径为 10 mm，粗匹配真值半径为 14 mm，细匹配正样本半径为 10 mm。这些是**本实验的工程设定**，不是论文直接参数，也未经消融验证。
- 固定使用完整 ROVENet（RTOR+A3），默认 `interaction_profile=legacy`，从头训练。已有 P2P 权重受不同坐标尺度和任务影响，不自动加载。旧 `--architecture` 消融参数不再支持。

## 完整训练与测试

在本项目根目录运行：

```bash
CUDA_VISIBLE_DEVICES=0 IRCADB_V3_RUN_NAME=full_v1 \
  bash experiments/geotransformer.ircadb_v3/run_full.sh
```

`run_full.sh` 固定使用 `/home/yangx/miniconda3/envs/geo_v2/bin/python`，默认 30 个完整 epoch。每轮使用全部 9300 个训练对和全部 930 个验证对。完成后自动用验证集最佳权重测试全部 1860 个测试对。可用 `IRCADB_V3_EPOCHS=60` 延长训练；已有 `snapshot.pth.tar` 时自动续训。仅保留最新 epoch、resume 和 best 检查点，避免存储全部 epoch。

默认数据根目录是 `/home/yangx/code/two_work/Liver_surface_data_process-main/liver_deform_output/dataset_v3`，可通过 `IRCADB_V3_DATA_ROOT` 指定同格式数据。结果写入 `output/geotransformer.ircadb_v3.<run_name>/`，其中 `test_summary.json` 汇总总体和逐模型 RMS-TRE、RRE、RTE 与 SR@5/10/20 mm，`test_summary.csv` 为逐对结果。

## 快速检查

```bash
CUDA_VISIBLE_DEVICES=0 IRCADB_V3_RUN_NAME=smoke_v1 \
  /home/yangx/miniconda3/envs/geo_v2/bin/python \
  experiments/geotransformer.ircadb_v3/trainval.py \
  --train_limit 2 --validation_size 2 --max_epoch 1 --log_steps 1

CUDA_VISIBLE_DEVICES=0 IRCADB_V3_RUN_NAME=smoke_v1 \
  /home/yangx/miniconda3/envs/geo_v2/bin/python \
  experiments/geotransformer.ircadb_v3/test.py \
  --snapshot output/geotransformer.ircadb_v3.smoke_v1/snapshots/best.pth.tar \
  --test_limit 2 --output output/geotransformer.ircadb_v3.smoke_v1/smoke_test.json
```

两样本检查已经通过，只验证运行链路，不报告为论文性能。真实术中泛化仍需独立真实数据验证。
