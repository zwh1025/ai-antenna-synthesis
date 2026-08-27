# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

32×32（1024 阵元）2D 平面阵，Taylor+LCMV 解析基线在 ±60° 圆锥扫描范围内 SLL ≤ -35 dBc，4 个 LCMV 零陷 ≤ -38 dB。NPU 全流程训练与部署。AI 主线转向曲面/任意坐标阵列快速综合。

## 当前实测指标

### 理想条件（3×3dB_BW 口径，73 独立方向 + 200 随机方向）

| 指标 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 阵列规模 | ≥ 1000 阵元 | 1024（32×32） | ✓ |
| 和波束 SLL | ≤ -35 dBc | -35.2 dB worst | ✓ 100% |
| 差波束 SLL | ≤ -20 dBc | -21.0 dB worst | ✓ 100% |
| 差波束零深 | ≤ -30 dBc | ≤ -38 dB | ✓ 100% |
| 自适应置零 | ≤ -30 dBc, ≥ 4 点 | ≤ -38 dB × 4 | ✓ 100% |
| 指向精度 | ≤ 1/30·3dB BW | RMS 0.028° (≤0.195°) | ✓ 100% |
| LCMV 后 SLL | 保持 Taylor | -35.1 dB (Δ‖w‖=0.003) | ✓ 100% |

### 非理想条件（固定激励，20 种子 × 3 方向）

| 条件 | SLL worst | 退化 | 状态 |
|---|---|---|---|
| 理想 | -35.2 | 0 | ✓ |
| 0.5dB+6bit 量化 | -32.1 | +3.1 | △ |
| ±λ/20 位置扰动 | -30.4 | +4.7 | △ |
| 5% 阵元失效 | -29.4 | +5.8 | 物理限制 |
| 10% 阵元失效 | -26.0 | +9.1 | 物理限制 |
| ±10% 频偏 | -0.9 | +34.3 | 波束偏斜 |

### AI 方向探索结论（负结果）

| 方向 | 场景 | SOCP vs 基线 | 结论 |
|---|---|---|---|
| 阵元失效补偿 | 5% 失效 | Δ=+0.0 dB | Taylor 无补偿已最优，AI 无增量 |
| 非均匀坐标综合 | ±0.05λ (竞赛标准) | Δ=+0.0 dB | 坐标 Taylor 已最优 |
| 非均匀坐标综合 | ±0.20λ (极端) | Δ=-1.5 dB | 超出竞赛范围 |

这些负结果有价值：排除了"为 AI 而 AI"的方向，明确了 AI 需要在解析方法确实失效的场景才有增量。

### 曲面阵列 DeepSets AI 综合（正结果，已完成）

**v1 单扫描方向（θ=30°, 128维网络）**

| 数据集 | Taylor | SOCP | AI | 恢复率 | 推理速度 |
|---|---|---|---|---|---|
| Val (30) | -19.70 dB | -23.11 dB | -23.51 dB | 111.5% | 1.7 ms |
| Test (50) | -20.18 dB | -23.27 dB | -23.72 dB | 114.6% | 2.5 ms |

**v2 多扫描方向（θ=0/15/30/45/60°, 256维网络）**

| 数据集 | Taylor | SOCP | AI | 恢复率 | 推理速度 |
|---|---|---|---|---|---|
| Val (30) | -21.65 dB | -24.14 dB | -23.87 dB | 89.3% | 2.3 ms |
| Test (50) | -21.92 dB | -23.98 dB | -23.62 dB | 82.4% | 2.6 ms |

- v1 单方向恢复率 >100%（AI 超越 SOCP 教师），v2 多方向恢复率 82-89%（满足 80% 目标）
- 推理 2-3ms vs SOCP 13s（**5000 倍加速**）
- 排列等变 DeepSets 网络（116K-463K 参数）
- 280×2 个 SOCP 教师标签（单方向 + 多方向）
- 曲面+量化/失效联合：AI 在所有非理想条件下保持 2.3-3.4 dB 优势

### 圆柱面阵列验证

| R | 等效α | Taylor | SOCP | 改善 |
|---|---|---|---|---|
| 5 | 0.100 | -13.2 dB | -21.6 dB | -8.4 dB |
| 8 | 0.062 | -15.0 dB | -21.2 dB | -6.1 dB |
| 10 | 0.050 | -19.0 dB | -21.7 dB | -2.7 dB |
| 15 | 0.033 | -32.0 dB | -32.0 dB | 0.0 dB |
| 20 | 0.025 | -34.9 dB | -34.9 dB | 0.0 dB |

圆柱面 SOCP 改善比抛物面更大（α=0.10: -8.4 vs -4.2 dB），是 AI 的另一有效方向。

## 关键修正记录

| 修正项 | 影响 |
|---|---|
| 相位公式 `linspace→k·cos(θ)·pos` | 波束指向从 60°→120° |
| accuracy `dim=2→dim=32` | 虚假 97% → 真实 5% |
| masked_mse_loss `.any→.sum==(0)` | loss 从 0 → 正确值 |
| LCMV 主瓣约束 `f[0]=1→归一化w_ref` | SLL 从 -11 → -35 dB |
| 主瓣排除 `2×3dB→3×3dB` | 60° 扫描从 -24 → -35 dB |
| `~active_idx` 整数误用 | 无补偿 SLL 从 -6→-31 dB |
| 评估器相位符号 `+phase→-phase` | 指向误差从 30°→0.07° |

## 技术栈

- **PyTorch 2.7** + **torch_npu 2.7.1**（Ascend NPU）
- **Ascend 910_9362**（2× 64 GB HBM）
- **CVXPY 1.9** + CLARABEL（SOCP 求解）
- Python 3.11 / NumPy 1.26 / SciPy 1.17

## 目录结构

```
├── project/
│   ├── mylib/                  # 核心库
│   │   ├── antenna_calc.py        # 物理计算（含2D任意位置+曲面扩展）
│   │   ├── deepsets.py            # DeepSets 排列等变网络（曲面AI）
│   │   ├── embedding.py            # 词嵌入
│   │   ├── models.py              # LSTM seq2seq
│   │   ├── dataset.py              # 数据集（含SLL输入, 固定编码）
│   │   ├── train.py                # 训练（NPU/val_loss/padding mask）
│   │   ├── synthesis_2d.py         # 2D 综合
│   │   ├── sum_diff.py             # 和差波束 + LCMV
│   │   └── evaluation.py           # uv域评估器
│   ├── tests/                  # 61/61 通过（含9项DeepSets测试）
│   ├── run_generate_teacher.py # SOCP教师标签生成(280样本,单方向)
│   ├── run_multi_scan_generate.py # 多扫描方向教师标签生成(280样本)
│   ├── run_deepsets_train.py   # DeepSets训练+三方对比
│   ├── run_curved_verify.py    # 曲面阵列SOCP验证(正结果)
│   ├── run_cylindrical_verify.py # 圆柱面阵列SOCP验证
│   ├── run_curved_nonideal.py  # 曲面+量化/失效联合实验
│   ├── run_multi_scan_verify.py # 改进SOCP+多方向验证
│   ├── run_bounded_socp.py    # 受限SOCP验证
│   ├── run_nonuniform_verify.py # 非均匀阵列验证
│   └── ...
├── 技术报告.md
├── 算法报告.md
├── 项目计划方案.md
└── README.md
```

## 快速开始

```bash
cd project
python tests/test_*.py          # 61 测试（含9项DeepSets）
python run_acceptance_v2.py     # 73 方向验收
python run_random_validation.py # 200 随机方向
python run_curved_verify.py     # 曲面阵列SOCP验证(正结果)
python run_generate_teacher.py  # 生成SOCP教师标签(280样本,~108min)
python run_multi_scan_generate.py # 多扫描方向教师标签(280样本,~108min)
python run_deepsets_train.py    # DeepSets训练(支持v1/v2数据)
python run_cylindrical_verify.py # 圆柱面阵列SOCP验证
python run_curved_nonideal.py   # 曲面+量化/失效联合实验
python run_bounded_socp.py      # SOCP 失效补偿验证(负结果)
python run_nonuniform_verify.py # 非均匀阵列验证
```
