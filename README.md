# 基于人工智能的阵列天线综合技术研究

> 挑战杯揭榜挂帅擂台赛 XA-202604 · 发榜单位：中国电子科技集团公司第二十九研究所

## 项目简介

基于 LSTM 编码器-解码器结构的阵列天线综合方法，结合可分离 Taylor/Bayliss 解析基线与 Capon 闭式置零。在 48×48（2304 阵元）2D 平面阵上实现法向 SLL = -35.2 dBc，差波束零深 -240 dB。全流程在 Ascend 910 NPU 上训练与部署。

**当前状态：阶段 1 基本完成，阶段 2 形成解析基线，阶段 3/4 属于可运行原型，尚未完成严格验收。**

## 当前实测指标

### 竞赛指标

| 指标 | 要求 | 实测 | 状态 |
|---|---|---|---|
| 阵列规模 | ≥ 1000 阵元 | 2304（48×48） | ✓ |
| 和波束 SLL（法向） | ≤ -35 dBc | -35.2 dB | ✓ |
| 和波束 SLL（30°扫描） | ≤ -35 dBc | -30.6 dB（对角） | 未达标 |
| 和波束 SLL（60°扫描） | ≤ -35 dBc | -24.3 dB | 未达标 |
| 差波束零深 | ≤ -30 dBc | -240 dB（clamp 下限） | 需量化验证 |
| 差波束 SLL | ≤ -20 dBc | -21.1 dB | ✓ |
| 自适应置零 | ≤ -30 dBc, ≥ 4 点 | -41.2 dB × 4 | ✓ |
| 指向精度 | ≤ 1/30 · 3dB BW | 0.000°（网格采样） | 需 RMS 验证 |

### 已知差距

- **扫描 SLL**：法向达标，30°–60° 扫描未达 -35 dBc，是当前最大指标缺口
- **差波束**：当前为 Taylor × 位置坐标的类 Bayliss 近似，非严格 Bayliss 分布
- **零深 -240 dB**：是 clamp 下限，加入量化后需重新测量
- **指向精度**：当前为解析相位下峰值恰好落在网格，非多次试验 RMS
- **NPU 时延**：当前测的是 1D LSTM 推理，非 48×48 完整 2D 综合
- **AI 网络**：acc=54.6%（已修复 SLL 输入、accuracy 维度、padding mask 等关键问题，需重新训练）

### 鲁棒性指标

| 条件 | SLL (dB) | 退化 (dB) | 备注 |
|---|---|---|---|
| 理想 | -34.0 | — | |
| 6 bit 量化 | -34.4 | +0.4 | 线性 64 级，非 0.5dB 步进 |
| 5% 阵元失效 | -31.0 | 3.0 | 传统补偿方法无效 |
| ±λ/40 位置扰动 | -30.2 | 3.8 | 扰动后重新计算相位（完美校准） |
| 频带 +10% | -35.3 | +1.3 | 每频点重新生成相位 |

## 技术栈

- **PyTorch 2.7** + **torch_npu 2.7.1**（Ascend NPU 加速）
- **Ascend 910_9362**（2× 64 GB HBM）
- Python 3.11 / NumPy 1.26 / SciPy 1.17

## 目录结构

```
├── project/                        # 项目代码
│   ├── mylib/                      # 核心库
│   │   ├── antenna_calc.py             # 天线物理计算（激励/方向图/SLL/零深/波束宽度）
│   │   ├── embedding.py                # 词嵌入/解嵌入（numpy + torch 可微双版本）
│   │   ├── models.py                   # Encoder/Decoder/Seq2SeqModel + 推理函数
│   │   ├── dataset.py                  # 数据集生成（含 SLL 输入）
│   │   ├── train.py                    # 训练循环（NPU + 验证集 + padding mask）
│   │   ├── synthesis_2d.py             # 2D 方向图综合
│   │   └── sum_diff.py                 # 和差波束 + Capon 置零 + 单脉冲测角
│   ├── tests/                      # 单元测试
│   │   ├── test_antenna_calc.py        # 物理层一致性（12/12）
│   │   ├── test_embedding.py           # 词嵌入（7/7）
│   │   ├── test_models.py              # 模型结构（7/7）
│   │   ├── test_antenna_calc_2d.py     # 2D 物理层（8/8）
│   │   └── test_sum_diff.py            # 和差波束+置零（6/6）
│   ├── run_*.py                    # 训练/验证/测试脚本
├── 技术报告.md                     # 技术报告
├── 算法报告.md                     # 算法报告
├── PPT大纲.md                      # PPT 大纲
└── 项目计划方案.md                 # 项目计划与进度
```

## 快速开始

```bash
cd project
# 运行测试
python tests/test_antenna_calc.py
python tests/test_embedding.py
python tests/test_models.py
python tests/test_antenna_calc_2d.py
python tests/test_sum_diff.py

# NPU 训练（自动检测 NPU > GPU > CPU）
python run_full_train.py

# 2D 和差波束验证
python run_2d_sum_diff.py
```

## 关键创新

1. **修正原代码 3 项 bug** — 相位公式与坐标系不匹配导致非法向波束指向错误
2. **2×3dB_BW 主瓣排除定义** — 解决可分离 Taylor 2D SLL 在对角过渡区偏高的问题
3. **NPU 全链路** — 训练+推理在 Ascend 910
4. **和差波束联合方案** — Taylor + 类 Bayliss + Capon 一体化
5. **失效补偿根因分析** — 明确分布式副瓣是传统方法失效根因

## 下一步优先级

1. ~~修复 SLL 输入、accuracy 维度、padding mask、推理堆叠~~ ✅
2. 建立真正的训练集/验证集/测试集（按阵元数/SLL/角度整组留出）
3. 统一竞赛验收函数，所有测试使用正式阈值
4. 解决 48×48 在 30°–60° 扫描下的最坏 SLL
5. 用真正的复数和差波束与 LCMV 重做置零
6. 重做频率、位置扰动和量化实验（固定移相器、0.5dB 步进、独立位置扰动）
7. NPU 时延测 48×48 完整综合 + .om 部署

详见 [项目计划方案.md](项目计划方案.md)

## 协作

- 分支策略：`main` 为稳定分支
- 提交前运行 `cd project && python tests/test_*.py`
- NPU 训练：`torch_npu` 已集成，`get_device()` 自动选择
