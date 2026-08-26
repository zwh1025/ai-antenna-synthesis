# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

基于 LSTM 编码器-解码器结构和可分离 Taylor/Bayliss 解析基线，在 32×32（1024 阵元）2D 平面阵上实现 ±60° 圆锥扫描。修正 LCMV 自适应置零在保持 -35 dBc 副瓣的同时放置 4 个零陷。全流程在 Ascend 910 NPU 上训练与部署。

**当前状态：解析基线达标，AI 失效补偿研发中。**

## 当前实测指标

### 理想条件（3×3dB_BW 口径，73 独立方向）

| 指标 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 阵列规模 | ≥ 1000 阵元 | 1024（32×32） | ✓ |
| 和波束 SLL | ≤ -35 dBc | -35.2 dB worst | ✓ 100% |
| 差波束 SLL | ≤ -20 dBc | -21.0 dB worst | ✓ 100% |
| 差波束零深 | ≤ -30 dBc | ≤ -38 dB | ✓ 100% |
| 自适应置零 | ≤ -30 dBc, ≥4 点 | ≤ -38 dB × 4 | ✓ 100% |
| 指向精度 | ≤ 1/30·3dB BW | RMS 0.028° (≤0.195°) | ✓ 100% |
| LCMV 后 SLL | 保持 Taylor | -35.1 dB (Δ‖w‖=0.003) | ✓ 100% |

### 非理想条件（固定激励，20 种子 × 3 方向）

| 条件 | SLL worst | 退化 | 状态 |
|---|---|---|---|
| 理想 | -35.2 dB | 0 | ✓ |
| 0.5dB+6bit 量化 | -32.1 dB | +3.1 | △ |
| ±λ/20 位置扰动 | -30.4 dB | +4.7 | △ |
| 5% 阵元失效 | -29.4 dB | +5.8 | ✗ 需 AI |
| 10% 阵元失效 | -26.0 dB | +9.1 | ✗ 需 AI |
| 20% 阵元失效 | -23.1 dB | +12.1 | ✗ 需 AI |
| ±10% 频偏 | -0.9 dB | +34.3 | ✗ 波束偏斜 |

### 已知差距

- **严格第一零点口径**：86% 达标（8 方向 -31~-34 dB），因可分离 Taylor 对角过渡区乘积，非测量误差
- **LCMV 置零后**：严格口径 84%，3×3dB_BW 口径 100%
- **AI 失效补偿**：CNN 框架已建，标签生成需改为梯度下降最优权值（闭式差值标签为零）
- **频偏退化**：固定移相器下 ±10% 频偏产生波束偏斜，可能需要真时延
- **NPU 限制**：不支持 float64 的 logsumexp/masked_fill

## 关键修正记录

| 修正项 | 影响 |
|---|---|
| 相位公式 `linspace→k·cos(θ)·pos` | 波束指向从 60°→120° |
| accuracy `dim=2→dim=32` | 虚假 97% → 真实 5% |
| masked_mse_loss `.any→.sum==(0)` | loss 从 0 → 正确值 |
| LCMV 主瓣约束 `f[0]=1→归一化w_ref` | SLL 从 -11 → -35 dB |
| 主瓣排除 `2×3dB→3×3dB` | 60° 扫描从 -24 → -35 dB |
| 位置编码 `动态→固定[-1,1]` | N=15 和 N=48 编码一致 |
| 推理堆叠 `axis=0→axis=-1` | 自回归推理通道顺序正确 |

## 技术栈

- **PyTorch 2.7** + **torch_npu 2.7.1**（Ascend NPU）
- **Ascend 910_9362**（2× 64 GB HBM）
- Python 3.11 / NumPy 1.26 / SciPy 1.17

## 目录结构

```
├── project/                        # 项目代码
│   ├── mylib/                      # 核心库
│   │   ├── antenna_calc.py             # 物理计算（激励/方向图/SLL/2D任意位置）
│   │   ├── embedding.py                # 词嵌入（numpy + torch 可微）
│   │   ├── models.py                   # Encoder/Decoder/Seq2SeqModel
│   │   ├── dataset.py                  # 数据集生成（含 SLL 输入, 固定编码范围）
│   │   ├── train.py                    # 训练（NPU/val_loss/padding mask/checkpoint）
│   │   ├── synthesis_2d.py             # 2D 方向图综合
│   │   ├── sum_diff.py                 # 和差波束 + LCMV 置零（修正后）
│   │   └── evaluation.py               # 方向自适应第一零点评估
│   ├── tests/                      # 单元测试（48/48 通过）
│   ├── run_acceptance_v2.py        # 73 方向正式验收
│   ├── run_random_validation.py    # 200 随机方向验证
│   ├── run_param_sweep.py          # Taylor 参数扫描
│   ├── run_failure_benchmark.py    # 失效补偿基准
│   ├── run_failure_cnn.py          # CNN 失效补偿
│   ├── run_diagnose_scan.py        # 60° 扫描退化诊断
│   ├── run_nonideal_v2.py          # 非理想实验（修正后）
│   └── ...                         # 其他脚本
├── 技术报告.md
├── 算法报告.md
├── PPT大纲.md
├── 项目计划方案.md
└── README.md
```

## 快速开始

```bash
cd project
# 运行测试（48/48 通过）
python tests/test_antenna_calc.py
python tests/test_embedding.py
python tests/test_models.py
python tests/test_antenna_calc_2d.py
python tests/test_sum_diff.py
python tests/test_train.py

# 73 方向正式验收
python run_acceptance_v2.py

# 200 随机方向验证
python run_random_validation.py

# 失效补偿基准
python run_failure_benchmark.py
```

## 协作

- 分支：`main` 为稳定分支
- 提交前运行 `cd project && python tests/test_*.py`
- NPU 训练：`get_device()` 自动选择 NPU > GPU > CPU
