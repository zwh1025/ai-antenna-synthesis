# 基于人工智能的阵列天线技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604
> 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

基于 LSTM 编码器-解码器结构的阵列天线综合方法，使用 AI 赋能大规模阵列天线的波束综合。目标实现 ≥1000 阵元 2D 平面阵的 ±60° 圆锥扫描，和波束 SLL ≤ -35 dBc。

## 技术栈

- **PyTorch 2.7** + **torch_npu**（Ascend NPU 加速）
- **Ascend 910 NPU**（2× 64GB HBM，0.6s/epoch，比 CPU 快 16 倍）
- Python 3.11 / NumPy / SciPy

## 目录结构

```
├── project/                    # 项目代码
│   ├── mylib/                  # 核心库
│   │   ├── antenna_calc.py         # 天线物理计算（激励/方向图/SLL/零深/波束宽度）
│   │   ├── embedding.py            # 词嵌入/解嵌入（numpy 线性插值 + torch softmax 可微）
│   │   ├── models.py               # Encoder/Decoder/Seq2SeqModel
│   │   ├── dataset.py              # 数据集生成（Taylor/Chebyshev 参考）
│   │   ├── train.py                # 训练循环（NPU/GPU/CPU 自适应设备选择）
│   │   └── synthesis_2d.py         # 2D 方向图综合
│   ├── tests/                 # 单元测试（26 项全部通过）
│   │   ├── test_antenna_calc.py        # 物理层一致性（12/12）
│   │   ├── test_embedding.py           # 词嵌入（7/7）
│   │   ├── test_models.py              # 模型结构（7/7）
│   │   └── test_antenna_calc_2d.py     # 2D 物理层（8/8）
│   ├── run_baseline.py        # 1D baseline 训练
│   └── run_2d_baseline.py     # 2D baseline 训练
└── 项目计划方案.md             # 项目计划与进度（唯一基准文档）
```

## 快速开始

```bash
# 运行全部测试
cd project
python tests/test_antenna_calc.py
python tests/test_embedding.py
python tests/test_models.py
python tests/test_antenna_calc_2d.py

# NPU 训练（自动检测设备，优先 NPU > GPU > CPU）
python run_baseline.py
python run_2d_baseline.py
```

## 进度

| 阶段 | 状态 | 关键指标 |
|---|---|---|
| 1. PyTorch 物理层 + 1D 模型复现 | ✅ 完成 | torch vs numpy max_err=2.76e-10 |
| 2. 2D 可分离和波束综合 | ✅ 完成 | 32×32 法向 SLL=-34dB, 48×48=-35.2dB |
| 2.5 轻度非理想验证 | ⏳ 待开始 | |
| 3. 和差波束 + 闭式置零 | ⏳ 待开始 | |
| 4. 强鲁棒性与 NPU 部署 | ⏳ 待开始 | |

详见 [项目计划方案.md](项目计划方案.md)

## 协作

- 分支策略：`master` 为稳定分支，开发用 `dev` 或功能分支
- 提交前请运行 `python tests/` 下全部测试
- NPU 训练：`torch_npu` 已集成，`get_device()` 自动选择 Ascend NPU
