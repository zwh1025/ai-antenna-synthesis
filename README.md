# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

32×32（1024 阵元）阵列天线，Taylor+LCMV 解析基线在 ±60° 圆锥扫描范围内 SLL ≤ -35 dBc，4 个 LCMV 零陷 ≤ -38 dB。曲面阵列（抛物面/圆柱面）场景下，DeepSets 排列等变网络学习坐标→权值映射，NPU 推理 0.499 ms 替代 SOCP 23 秒求解。v3 模型经平面样本增广后单一网络同时覆盖平面与曲面阵列。全流程在华为昇腾 910 NPU 上实现训练与部署，4096 阵元完成计算规模扩展测试。另以 8×8 曲面阵列已求解 HFSS 数据（64 复数 EEP + 64 端口 S 参数）离线重组，验证权值到真实电磁模型的工程闭环。

## 实测指标

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

> **口径说明**：副瓣评估采用 3×3dB_BW 排除口径（竞赛方未明确规定具体倍数）。同一方向图在 2×3dB_BW 下 SLL 约 -24 dB，3×3dB_BW 下约 -35 dB。本研究认为 3×3dB_BW 在 60° 大扫描角下更合理地排除了主瓣裙边。

> **结果边界（如实披露）**：
> 1. -35 dBc 理想平面阵结果来自 Taylor+LCMV 系统，不是 DeepSets 单独输出；
> 2. 固定权值下的 ±λ/20 位置扰动、阵元失效、量化、频偏测试存在明显退化，多项未达 -35 dB，应作为局限看待（见 `run_nonideal_v2.py` / `run_curved_nonideal.py`）；
> 3. 4096 阵元完成的是 DeepSets 计算规模测试，不是 HFSS 全波验证。

### 曲面阵列 DeepSets AI 综合

| 数据集 | Taylor | SOCP | AI | 恢复率 |
|---|---|---|---|---|
| v1 验证集(30) | -19.70 dB | -23.11 dB | -23.51 dB | 111.5% |
| v1 测试集(50) | -20.18 dB | -23.27 dB | -23.72 dB | 114.6% |
| v2 验证集(30) | -21.65 dB | -24.14 dB | -23.87 dB | 89.3% |
| v2 测试集(50) | -21.92 dB | -23.98 dB | -23.62 dB | 82.4% |
| **v3 测试集(50，平面增广后)** | -21.92 dB | -23.98 dB | **-24.56 dB** | **128.0%** |

> v1 为单扫描方向（θ=30°），v2 为多扫描方向（θ=0/15/30/45/60°）。SOCP 教师仅 5 轮切平面+15×15 粗网格，非全局最优，故 AI 恢复率 >100% 表示超过弱 SOCP 基线。

### 平面/曲面统一覆盖（v3，2026-09-02）

v2 模型存在**捷径学习**：训练集中扫描角与修正需求完全相关，导致平面阵 θ≥30° 分布外推理退化 10.2–15.7 dB。加入 120 个平面样本（α=0，目标 ΔW=0，零成本生成）重训练 v3 后：

| 验证项 | v2（修复前） | v3（本地） | v3（服务器复现） |
|---|---|---|---|
| 平面阵 40 方向最差退化 | +18.15 dB | **+0.51 dB** | **+0.34 dB** |
| 曲面测试集恢复率 | 82.4% | **128%** | **120%** |

单一 v3 模型现可同时覆盖平面与曲面阵列。复现：`run_planar_generalization.py`（发现问题）→ `run_planar_fix_train.py`（增广修复）。

### NPU 基准（昇腾910 vs 鲲鹏CPU，同机A/B测试）

v3（256 维，真实权重，`benchmark_v3.json`）：

| 指标 | NPU | CPU | 加速比 |
|---|---|---|---|
| 推理 mean (1000轮) | 0.499 ms | 2.762 ms | 5.5x |
| 推理 P50 (1000轮) | 0.497 ms | 1.674 ms | 3.4x |
| 推理 P99 (1000轮) | 0.532 ms | 12.296 ms | 23.1x |
| 端到端延迟 | 0.632 ms | — | — |
| 连续吞吐量 (batch=1) | 2003/s | 362/s | 5.5x |
| 批量吞吐量 (bs8/16/64) | 13699/s / 21608/s / 27649/s | — | — |
| 精度一致性(真实权重) | max_err=2.47×10⁻⁶ | — | cos_sim=1.00000000 |
| vs SOCP 23秒 | 0.499ms | — | 46,080x |

v2（128 维）对照：纯推理 0.504 ms、端到端 0.658 ms、吞吐 1985/s——**网络加宽一倍后 NPU 推理延迟持平**。训练加速（128维 bs16）15.7x、（512维 bs64）86.1x。

> 服务器配备 2 颗 Ascend 910_9362（各 64GB HBM），本次仅使用单卡。模型仅 0.47-1.9MB，单卡算力远超需求。vs SOCP 的加速比是"AI 前向推理"与"迭代优化求解"的算法+硬件综合差距，不是同类计算的硬件加速比，不能表述为"NPU 硬件单独带来 46,080 倍加速"。

### 8×8 HFSS 全波工程验证

复用实验室已求解的 8×8 曲面阵列 HFSS 数据（12.5 GHz），通过 64 个复数 EEP 线性重组 + 64 端口 S 参数有源匹配，离线验证 AI 权值能否进入真实多端口电磁模型。模型权值与 HFSS 入射波采用 `a_HFSS = conj(w_model)` 相位约定。

| 项目 | 结果 |
|---|---|
| EEP 数量 / 每个采样点 | 64 / 32580 |
| EEP 重组交叉复算误差（vs HFSS 直接激励） | ~10⁻⁹ dB |
| AI 总接受功率 | 81.5% – 91.6%（1 W 归一化） |
| 控制方向（degraded/fallback/unsupported） | 无误导出 |
| 阵因子层安全导出（8×8 D4 归档） | 536/544 方向 |
| 全波严格对比（6 个方向） | **0/6 通过，AI 副瓣较 Taylor 差 0.79 – 2.80 dB** |

**正确解读**：该实验证明端口映射、相位约定、EEP 重组和有源匹配链路成立，是工程可实现性证据；但 AI 在小阵列真实互耦下未优于 Taylor，阵因子层 98.5% 安全导出率不等同于全波通过率，且不能由 8×8 外推 64×64 电磁性能。8×8 模型与安全归档见根目录 `deepsets_8x8_d4_v2_1.tar.gz`（冻结模型 + 544 方向预测 + 支持域查找表 + 安全导出器），完整结果见 `HFSS_8x8曲面阵列验证/returned_results/`。

## 技术栈

- **PyTorch 2.7.1** + **torch_npu 2.7.1**（昇腾 NPU）
- **Ascend 910_9362**（64GB HBM）+ 鲲鹏 CPU（40 核 3.0GHz）
- **CVXPY** + CLARABEL（SOCP 求解）
- Python 3.11 / NumPy 1.26 / SciPy 1.17

## 目录结构

```
├── project/
│   ├── mylib/                       # 核心库（9 模块）
│   │   ├── antenna_calc.py           # 物理计算（激励/方向图/SLL/2D任意位置）
│   │   ├── deepsets.py               # DeepSets 排列等变网络
│   │   ├── embedding.py              # 词嵌入（numpy + torch 可微）
│   │   ├── models.py                 # LSTM seq2seq（旧主线，已弃用）
│   │   ├── dataset.py                # LSTM 数据生成（旧）
│   │   ├── train.py                   # 训练工具（设备选择/早停/masked loss）
│   │   ├── synthesis_2d.py           # 2D 梯度综合（备用）
│   │   ├── sum_diff.py               # 和差波束 + LCMV 置零
│   │   └── evaluation.py             # uv 域评估器（-30dB 连通域 + 3×3dB_BW）
│   ├── tests/                       # 61 项测试（8 文件）
│   │   ├── test_antenna_calc.py      # 12 项（物理层）
│   │   ├── test_antenna_calc_2d.py  # 8 项（2D 分离激励）
│   │   ├── test_deepsets.py          # 9 项（含排列等变性）
│   │   ├── test_embedding.py         # 7 项
│   │   ├── test_models.py            # 7 项（LSTM）
│   │   ├── test_train.py             # 8 项（训练工具）
│   │   ├── test_sum_diff.py          # 6 项（和差波束+LCMV）
│   │   └── test_failure_mask.py      # 4 项（失效索引）
│   ├── run_acceptance_v2.py          # 73 方向正式验收（~6min）
│   ├── run_random_validation.py     # 200 随机方向（~30s）
│   ├── run_2d_sum_diff.py            # 48×48 和差波束（~30s）
│   ├── run_nonideal_v2.py            # 非理想实验：量化/失效/频偏（~1min）
│   ├── run_curved_verify.py          # 曲面阵列 SOCP 验证（正结果，~11min）
│   ├── run_cylindrical_verify.py     # 圆柱面阵列 SOCP 验证（~8min）
│   ├── run_curved_nonideal.py        # 曲面+量化/失效联合实验（~5min）
│   ├── run_bounded_socp.py           # SOCP 失效补偿验证（负结果，~1min）
│   ├── run_nonuniform_verify.py      # 非均匀坐标验证（负结果）
│   ├── run_failure_benchmark.py     # 失效补偿基准
│   ├── run_ga_pso_compare.py        # Taylor/GA/PSO/SOCP/AI 五方法对比
│   ├── run_generate_teacher.py      # SOCP 教师标签生成 v1（280样本，~108min）
│   ├── run_multi_scan_generate.py   # 多方向教师标签 v2（280样本，~108min）
│   ├── run_deepsets_train.py         # DeepSets 训练 + 三方对比（~2min NPU）
│   ├── run_benchmark.py              # CPU/NPU 标准基准（7项，~10min）
│   ├── run_benchmark_supplement.py   # 端到端延迟 + 1000轮 P99（~5min）
│   ├── run_demo.py                   # 30 秒演示全流程（录屏用）
│   ├── make_charts.py                # 生成数据图表
│   └── make_ppt.py                   # 基于模板生成 PPT
├── 技术报告.md                        # 6 章 + 15 参考文献
├── 算法报告.md                        # 算例验证 + 数据集 + 代码结构
├── 性能对比报告.md                     # Taylor/GA/PSO/SOCP/AI 五方法对比
├── 基准测试数据汇总.md                 # 8 章节完整 NPU/CPU 对比数据
├── 项目计划方案.md                    # 技术路线 + 阶段目标
├── PPT大纲_v2.md                     # 答辩 PPT 大纲（13 页，NPU 为主线）
├── deepsets_8x8_d4_v2_1.tar.gz        # 8×8 DeepSets D4 安全归档（冻结模型+544方向预测+安全导出器）
├── CONTRIBUTING.md                   # 环境配置和运行方式
├── requirements.txt                  # 依赖列表
└── README.md
```

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/zwh1025/ai-antenna-synthesis.git
cd ai-antenna-synthesis

# 2. 安装依赖
pip install -r requirements.txt

# 3. 运行测试验证环境
cd project
python -m pytest tests/ -v           # 61 项测试

# 4. 核心实验（按需运行）
python run_acceptance_v2.py          # 73 方向验收（~6min）
python run_curved_verify.py          # 曲面 SOCP 验证（~11min）
python run_generate_teacher.py       # 生成教师标签（~108min，一次性）
python run_deepsets_train.py         # DeepSets 训练（~2min NPU / ~1.5min CPU）
python run_deepsets_train.py --data_path outputs/teacher_labels_v2.npz --hidden_dim 256 --save_name deepsets_model_v2_256.pt  # v2 多方向
python run_demo.py                   # 30 秒演示（需教师标签+模型文件）
python run_benchmark.py              # NPU/CPU 基准（需 NPU 环境）
python run_benchmark_supplement.py   # 端到端延迟+1000轮 P99（需 NPU）
python run_planar_generalization.py # 平面阵泛化测试（CPU，~3min）
python run_planar_fix_train.py      # v3 增广修复训练+双场景验证（CPU，~10min）

# 5. 扩展实验
python run_ga_pso_compare.py        # Taylor/GA/PSO/SOCP/AI 五方法对比
python run_cylindrical_verify.py     # 圆柱面阵列
python run_curved_nonideal.py        # 曲面+量化/失效联合
python run_nonideal_v2.py            # 平面阵非理想条件
python run_bounded_socp.py           # SOCP 失效补偿（负结果）
python run_nonuniform_verify.py      # 非均匀坐标（负结果）
```

> 无 NPU 环境也能运行测试和解析基线，自动使用 CPU。NPU 基准和 AI 推理速度数据需 NPU 环境复现。教师标签和模型文件在 `outputs/` 下（.gitignore 排除），需先运行生成脚本。

## 关键修正记录

| 修正项 | 影响 |
|---|---|
| 相位公式 `linspace→k·cos(θ)·pos` | 波束指向从 60°→正确 |
| accuracy `dim=2→dim=32` | 虚假 97% → 真实 5% |
| masked_mse_loss `.any→.sum==0` | loss 恒 0 → 正确值 |
| LCMV 主瓣约束 `f[0]=1→归一化w_ref` | SLL -11→-35 dB |
| 主瓣排除 `2×3dB→3×3dB` | 60° 扫描 -24→-35 dB |
| `~active_idx` 整数误用 | 无补偿 SLL -6→-31 dB |
| 评估器相位符号 `+phase→-phase` | 指向误差 30°→0.07° |
| v3 平面样本增广 | 平面阵 θ≥30° 退化 +10~16 dB → +0.51 dB（捷径学习修复） |
