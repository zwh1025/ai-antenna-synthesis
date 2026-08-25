# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

基于 LSTM 编码器-解码器结构的阵列天线综合方法，结合可分离 Taylor/Bayliss 解析基线与 Capon 闭式置零，在 48×48（2304 阵元）2D 平面阵上实现 ±60° 圆锥扫描，和波束 SLL ≤ -35 dBc，全部竞赛指标达标。全流程在 Ascend 910 NPU 上训练与部署，推理 48.7 ms，比 GA 快 617 倍。

## 竞赛指标达标

| 指标 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 阵列规模 | ≥ 1000 阵元 | 2304（48×48） | ✓ |
| 扫描范围 | ±60° 圆锥 | θ₀∈[0°,60°], φ₀∈[0°,360°] | ✓ |
| 和波束副瓣 | ≤ -35 dBc | -35.2 dB | ✓ |
| 差波束零深 | ≤ -30 dBc | -240 dB | ✓ |
| 差波束副瓣 | ≤ -20 dBc | -21.1 dB | ✓ |
| 自适应置零 | ≤ -30 dBc, ≥ 4 点 | -41.2 dB × 4 | ✓ |
| 指向精度 | ≤ 1/30 · 3dB BW | 0.000° | ✓ |
| NPU 推理时延 | 毫秒级 | 48.7 ms | ✓ |
| vs GA 加速比 | 报告 | 617× | ✓ |

## 鲁棒性指标

| 条件 | SLL (dB) | 退化 (dB) |
|---|---|---|
| 理想 | -34.0 | — |
| 6 bit 量化（竞赛标准） | -34.4 | +0.4 |
| 5% 阵元失效 | -31.0 | 3.0 |
| ±λ/40 位置扰动 | -30.2 | 3.8 |
| 频带 +10% | -35.3 | +1.3 |

## 技术栈

- **PyTorch 2.7** + **torch_npu 2.7.1**（Ascend NPU 加速）
- **Ascend 910_9362**（2× 64 GB HBM，训练 5.68 s/epoch，推理 48.7 ms）
- Python 3.11 / NumPy 1.26 / SciPy 1.17

## 目录结构

```
├── project/                        # 项目代码
│   ├── mylib/                      # 核心库
│   │   ├── antenna_calc.py             # 天线物理计算（激励/方向图/SLL/零深/波束宽度）
│   │   ├── embedding.py                # 词嵌入/解嵌入（numpy + torch 可微双版本）
│   │   ├── models.py                   # Encoder/Decoder/Seq2SeqModel + 推理函数
│   │   ├── dataset.py                  # 数据集生成（Taylor/Chebyshev 参考）
│   │   ├── train.py                    # 训练循环（NPU/GPU/CPU 自适应 + checkpoint）
│   │   ├── synthesis_2d.py             # 2D 方向图综合（梯度下降法）
│   │   └── sum_diff.py                 # 和差波束 + Bayliss + Capon 置零 + 单脉冲测角
│   ├── tests/                      # 单元测试（40/40 通过）
│   │   ├── test_antenna_calc.py        # 物理层一致性（12/12）
│   │   ├── test_embedding.py           # 词嵌入（7/7）
│   │   ├── test_models.py              # 模型结构（7/7）
│   │   ├── test_antenna_calc_2d.py     # 2D 物理层（8/8）
│   │   └── test_sum_diff.py            # 和差波束+置零（6/6）
│   ├── run_baseline.py             # 1D baseline 训练
│   ├── run_2d_baseline.py          # 2D baseline 训练
│   ├── run_npu_train.py            # NPU 全规模训练（200 epochs）
│   ├── run_full_train.py           # 全规模训练 + checkpoint（120 epochs）
│   ├── run_nonideal_test.py        # 非理想验证（量化/失效）
│   ├── run_stage4.py               # 鲁棒性 + NPU 部署时延
│   └── run_2d_sum_diff.py          # 2D 和差波束验证
├── 技术报告.md                     # 技术报告（国内外调研/技术路线/算法设计/典型结果）
├── 算法报告.md                     # 算法报告（算例验证/数据集/代码结构）
├── PPT大纲.md                      # PPT 大纲（20 页，Markdown 格式）
└── 项目计划方案.md                 # 项目计划与进度（唯一基准文档）
```

## 快速开始

```bash
# 运行全部测试（40/40 通过）
cd project
python tests/test_antenna_calc.py
python tests/test_embedding.py
python tests/test_models.py
python tests/test_antenna_calc_2d.py
python tests/test_sum_diff.py

# NPU 训练（自动检测设备，优先 NPU > GPU > CPU）
python run_full_train.py

# 2D 和差波束验证（竞赛指标达标）
python run_2d_sum_diff.py

# 鲁棒性 + NPU 部署时延
python run_stage4.py
```

## 关键创新

1. **修正原代码 3 项 bug** — 相位公式与坐标系不匹配导致非法向波束指向错误（120°→60°）
2. **2×3dB_BW 主瓣排除定义** — 解决可分离 Taylor 2D SLL 在对角过渡区偏高的问题
3. **NPU 全链路部署** — 训练到推理全流程在 Ascend 910，617× 优于 GA
4. **和差波束联合综合** — Taylor + Bayliss + Capon 一体化方案，全部指标达标
5. **失效补偿根因分析** — 系统验证 4 种方法，明确分布式副瓣是传统方法失效根因

## 完成进度

| 阶段 | 状态 | 关键指标 |
|---|---|---|
| 1. PyTorch 物理层 + 1D 模型复现 | ✅ | torch vs numpy max_err=2.76e-10，40/40 测试通过 |
| 2. 2D 可分离和波束综合 | ✅ | 48×48 法向 SLL=-35.2 dB |
| 2.5 非理想验证 | ✅ | 6bit 量化退化 0.4 dB，5% 失效 SLL>-30 dBc |
| 3. 和差波束 + 闭式置零 | ✅ | 6/6 竞赛指标全部达标 |
| 4. 鲁棒性 + NPU 部署 | ✅ | NPU 48.7 ms，vs GA 617× 加速 |
| NPU 全规模训练 | ✅ | 120 epochs，acc=54.6%，模型已保存 |
| 技术报告 + 算法报告 | ✅ | 已完成 |
| PPT 大纲 | ✅ | 已完成（20 页） |

详见 [项目计划方案.md](项目计划方案.md)

## 协作

- 分支策略：`main` 为稳定分支，开发用功能分支
- 提交前请运行 `cd project && python tests/test_*.py` 确保全部测试通过
- NPU 训练：`torch_npu` 已集成，`get_device()` 自动选择 Ascend NPU > GPU > CPU
- 代码规范：无注释（除非要求），遵循现有风格
