# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

本项目面向 1024 阵元阵列天线的方向图综合，采用“物理硬指标闭合 + AI 快速修正 + 工程非理想评估”的分层路线。最新 upstream v3 主线提供了 1024 阵元平面/曲面 AI 管线、真实 Ascend 910 基准和 4096 阵元计算规模测试；本仓库同时保留一组独立的物理验收、鲁棒性和固定曲面任务样本效率证据。不同证据层使用的 geometry、评估器、模型和任务并不完全相同，本文不把它们合并成一个单一 benchmark。

## 证据结构

| 层次 | 冻结任务 | 主要结论 |
|---|---|---|
| Track P | 32×32、1024 阵元 planar physics track | Sum/Difference、理想 adaptive null 和大扫描范围指标闭合；并评估非理想退化与已知失效重构 |
| Track AI-v3 | upstream v3 的 1024 阵元平面/曲面 AI 管线 | 平面增广改善了固定 v3 模型的平面分布外表现；提供 Ascend 910 实测推理与吞吐数据 |
| Track AI-fixed | 一个固定的 32×32、1024 阵元 reconstructed curved geometry | coordinate-aware Taylor 上的 held-out residual improvement、sample efficiency 和 CPU 在线耗时 |
| Engineering boundary | 8×8 HFSS 数据与 4096 规模测试 | 分别用于工程链路边界和计算规模验证，不外推为 64×64 全波结论 |

## 核心结果

### Track P：1024 阵元物理硬指标

固定 official evaluator 1.0.0 的补充物理验收结果为：

- Sum SLL：273/273 通过，最差 `-35.156576 dBc`；
- Azimuth Difference SLL：273/273 通过，最差 `-21.754833 dBc`；
- Elevation Difference SLL：273/273 通过，最差 `-21.638634 dBc`；
- 理想 adaptive null：Sum strict null 在 73/73 cases 通过，Difference strict null 在 273/273 cases 通过；
- 理想 intrinsic null 达到模型的数值精度下限，展示时应写作低于 `-100 dBc` 的 numerical floor，而不是硬件性能。

这里的硬指标结果属于 Track P 的 Taylor/Bayliss + LCMV physics pipeline，不是 DeepSets 单独输出。

### Track AI-v3：最新 upstream 平面/曲面管线

upstream v3 在 1024 阵元任务上报告了以下当前结果：

- 平面增广后，服务器复现的 40 个方向平面退化最大值为 `+0.34 dB`；
- 同一 v3 实验的曲面测试恢复率为 `120.2%`；
- 平面验收中，v3 direct arm 为标准方向 `70/73`、随机方向 `199/200`，v3+LCMV arm 为 `69/73`、`194/200`；这些是 arm-specific results，不应改写为 DeepSets direct output 在所有方向上 100% 通过；
- NPU benchmark 使用真实 v3 权重，在 Ascend 910 上得到纯推理 mean `0.499 ms`、端到端 mean `0.632 ms`、纯推理 P99 `0.532 ms`，batch=64 吞吐 `27,649 samples/s`；
- 4096 阵元结果是 DeepSets 计算规模测试，不是 4096 阵元 HFSS 全波验证。

v3 的 NPU 数字属于 upstream v3 的 256-dim 模型和对应硬件环境；不能与下述固定曲面 CPU 模型数字直接横向排名。

### Track AI-fixed：固定曲面任务的样本效率

在一个固定 reconstructed curved geometry 上，Stage 5A 使用 independent teacher parents `2/4/6/8`，每个 parent 的 D4 rows 是 correlated augmentation，不是独立 teacher samples：

| Independent parents | D4 rows | Held-out improvement | Mean gain |
|---:|---:|---:|---:|
| 2 | 16 | 3/4 nominal | approximately 0 dB |
| 4 | 32 | 4/4 | approximately 0.837 dB |
| 6 | 48 | 3/4 | approximately 1.351 dB |
| 8 | 64 | 4/4 | approximately 1.326 dB |

因此可支持的保守结论是：从 4 个 independent teacher parents 开始出现具有实际意义的 held-out improvement，6–8 个 parents 的平均改善约为 1.3 dB。该 study 不替代、也不直接 benchmark upstream v3 的 mixed-geometry/NPU workflow。

该固定曲面模型的 held-out test 为 4/4 parents 改善 Taylor，平均增益 `1.1876 dB`；CPU inference-only mean 为 `9.157 ms`，end-to-end mean 为 `9.756 ms`。模型复杂度为 `597,250 parameters`、`405,733,376 MACs`、`811,466,752 FLOPs`，仅指 neural forward。

### Robustness 与已知失效重构

补充 Track P robustness evidence 使用固定 frozen cases：

- `Δx, Δy ~ Uniform[-0.05,+0.05]λ`：平均 Sum-SLL degradation `+2.186 dB`；
- `0.5 dB` amplitude model + `6-bit / 5.625°` phase：平均 degradation `+0.860 dB`；
- 5%/10%/20% element failure 分别对应 51/102/204 个失效阵元，平均 degradation `+4.212/+6.405/+9.967 dB`；
- 频率 sweep 为 `0.90–1.10`，common-joint compliance 为 `76/336`，完整 ±10% band 为 `3/16` cases。

对已知 failure mask 的 B2 active-set reconstruction 平均恢复 `2.684/3.011/2.094 dB`（5%/10%/20%）。这是 limited recovery；全部 960 个 mask 的 common-joint hard-spec pass 仅为 `2/960`。Azimuth Difference 已运行，Elevation Difference failure reconstruction 未纳入正式 benchmark。

## 方法与复现入口

### AI 方法

DeepSets 使用逐阵元共享编码、mean/max pooling 和逐阵元 residual head。v3 主线的训练、平面增广和 NPU 基准入口分别为：

```bash
cd project
python run_planar_generalization.py
python run_planar_fix_train.py
python run_acceptance_v3_ai.py
python run_benchmark_v3.py
```

teacher generation、训练和模型文件属于离线或环境相关步骤。默认的 v3 评估器是 `project/mylib/evaluation.py` 的 legacy uv-domain path；上述 Track P 补充证据使用 official evaluator 1.0.0，二者不应无说明地混用。

### 基础测试

```bash
pip install -r requirements.txt
cd project
python -m pytest tests/ -v
```

无 NPU 环境仍可运行测试和解析基线；NPU 数据仅能在相应 Ascend 910 环境复现。完整模型、teacher labels 和实验输出不作为普通代码入口的隐式依赖。

## 证据边界与局限

1. AI fixed-task sample-efficiency evidence 只适用于一个固定的 32×32 reconstructed curved geometry 和冻结的 parent split，不证明跨 geometry 的 unseen-task generalization。
2. upstream v3 NPU evidence 是 v3 模型的真实 Ascend 910 benchmark；它不自动覆盖 Track AI-fixed 模型或 Track P physics evaluator。
3. 固定曲面任务的严格 `0.5 dB` teacher-proximity 条件在完整 validation set 上未达到；收敛轨迹和 runtime 仍已记录。
4. failure-aware reconstruction 展示的是已知 mask 下的部分 Sum-SLL 恢复，不是所有失效条件下的 full hard-spec recovery。
5. 8×8 HFSS 验证用于端口映射、EEP 重组和有源匹配链路的工程边界；不能由 8×8 外推 64×64 全波性能。
6. SOCP 统一称为 `SOCP/cutting-plane optimization reference`；它是固定约束、网格和切平面设置下的 reference，不作为普适最优性 claim。

## 证据定位

最新 v3 machine-readable outputs 位于 `project/outputs/`，包括 `acceptance_v3_ai.json`、`benchmark_v3.json`、`planar_fix_v3.json` 和 `random200_taylor_3bw.json`。Track P、robustness、failure reconstruction 和 fixed-task sample-efficiency 的完整冻结证据属于相应研究证据包；README 只保留其可审查摘要，不要求普通 clone 携带全部中间结果。
