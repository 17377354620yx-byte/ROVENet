# RTOR 与 A3 模块消融方案

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-11-02
- Verification Status: 代码路径与参数已核对；完整训练结果待验证
- Version Label: module_ablation_cc68_v1

## 研究问题

在同一 `cooperative` 训练协议下，RTOR、A3 以及 A3 的几何偏置分别对配准精度有何贡献？双模块组合是否优于各单模块，是否存在正向交互？

这里的 A3 是块内双向交叉注意力；`geometry_weight=0` 保留该注意力及其参数，只关闭几何偏置。因而 `joint` 与 `geometry_off` 的差异专门检验几何信息的贡献。

## 实验矩阵

### 当前指定的三组子实验

若只运行 RTOR、A3 和“保留 A3 注意力但关闭几何偏置”，第三组必须与 `a3` 一样关闭 RTOR，才能隔离几何偏置。使用：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_three_arm_ablation.sh 1041
```

该命令只运行 `rtor`、`a3`、`a3_geometry_off` 三组，种子由位置参数指定，三组分别训练 150 epoch 并测试第 150 轮。`a3 − a3_geometry_off` 是几何偏置效应；`rtor` 与 `a3` 是两种单模块性能比较。这个三组子实验不能估计 RTOR 与 A3 的交互效应。原有的 `geometry_off` 名称仍指双模块（RTOR+A3）关闭几何偏置，不能与 `a3_geometry_off` 混用。

### 完整五组方案

所有组均使用相同数据、单编码器、原 LGR 参数、`cooperative` 训练协议、150 epoch、学习率 `1e-4`，并在各组使用完全相同的随机种子。`cooperative` 中 predicted coarse proposals 按第 5 至 20 轮逐步引入，最大占比 0.25；source 硬筛选关闭。每组只测试 `epoch-150.pth.tar`。

| 组 | `--architecture` | RTOR | A3 注意力 | 几何偏置 | 作用 |
|---|---|---:|---:|---:|---|
| `baseline` | `geotransformer` | 关 | 关 | 关 | 共同对照 |
| `rtor` | `rtor_only` | 开 | 关 | 关 | RTOR 单模块 |
| `a3` | `a3_only` | 关 | 开 | 开 | A3 单模块 |
| `joint` | `rtor_a3` | 开 | 开 | 开 | 两模块组合 |
| `geometry_off` | `rtor_a3` | 开 | 开 | 关 | 隔离几何偏置 |

这五组是主实验。原 `legacy` profile 的模型可作为历史参考，但不能替代本矩阵中的对照，因为它的训练 proposals 和推理筛选条件不同。

## 执行

工作目录：`/home/yangx/code/new_deform/ai_worker/ROVENet`；环境：`geo_v2`。先用一个种子验证流程：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh baseline 1041
CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh rtor 1041
CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh a3 1041
CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh joint 1041
CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh geometry_off 1041
```

正式实验预先固定种子 `1041, 1042, 1043, 1044, 1045`，不要根据测试结果增删种子或只展示最佳种子：

```bash
for seed in 1041 1042 1043 1044 1045; do
  for arm in baseline rtor a3 joint geometry_off; do
    CUDA_VISIBLE_DEVICES=0 bash scripts/run_module_ablation.sh "$arm" "$seed"
  done
done
```

脚本每组训练完后运行 in-silico `none/2/4` 与 in-vitro 四组完整测试。训练中断后重跑相同命令，会从已有 `snapshot.pth.tar` 续训；已有 epoch 150 权重不会重新训练。结果放在 `output/geotransformer.p2p_liver.ablate_coop_<arm>_seed<seed>_epoch150/`，其中 JSON 的 `method.mean_rms_tre_mm`、`method.success_rate_20mm` 和逐样本 `samples` 用于统计。

## 预先规定的分析

主指标是四组测试的平均 RMS-TRE 均值（四组等权，越低越好）。同时完整报告每个测试集的 RMS-TRE、SR@20mm、标准差与每个种子的结果，不以某一组数据表现最好为由改换主指标。

对同一种子、同一测试集先计算以下差值（负值代表左侧组更好），再跨预定种子汇总：

1. A3 在 RTOR 上的增益：`joint − rtor`。
2. 几何偏置在双模块中的增益：`joint − geometry_off`。
3. RTOR 在 A3 上的增益：`joint − a3`。
4. 模块交互：`joint − rtor − a3 + baseline`。负值表明组合改善超过两个单模块效应之和。

以**种子**作为独立重复单位，报告五个配对差值、跨种子均值和 95% t 置信区间。逐样本配对分析可用于定位低可见度或高噪声问题，但不能把同一模型的 6315 个样本当作 6315 次独立训练。主结论需同时检查 `joint − rtor` 与 `joint − geometry_off`；若任一方向不稳定，就如实报告，不把单个优胜种子作为模块有效性的证据。

## 复现与边界

- 记录每组 `run_manifest.json`、Git 提交 `cc68d87` 加本工作区修改、种子、`epoch-150.pth.tar` 和四个测试 JSON。
- 测试集只用于最终评估；不得依据测试结果调种子、超参或选择 epoch。此前按测试分数搜索种子的实验应与这组预定种子的消融结果分开报告。
- `geometry_weight=0` 隔离的是几何偏置，不等于移除整个 A3；`joint − rtor` 才检验完整 A3 模块在 RTOR 存在时的边际效应。
