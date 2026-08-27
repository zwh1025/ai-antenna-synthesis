# 协作指南

## 项目背景

挑战杯揭榜挂帅擂台赛 XA-202604，基于 AI 的阵列天线综合技术研究。

## 环境配置

```bash
# 1. 克隆仓库
git clone https://github.com/zwh1025/ai-antenna-synthesis.git
cd ai-antenna-synthesis

# 2. 创建 conda 环境（需管理员权限运行 Anaconda Prompt）
conda create -n antenna python=3.11
conda activate antenna

# 3. 安装依赖
conda install -c conda-forge cvxpy -y
pip install numpy scipy torch matplotlib

# 4. 运行测试验证
cd project
python tests/test_antenna_calc.py
```

## 快速运行

| 脚本 | 耗时(CPU) | 作用 |
|---|---|---|
| `python tests/test_*.py` | <1min | 52项单元测试 |
| `python run_2d_sum_diff.py` | ~30s | 48×48和差波束竞赛指标 |
| `python run_acceptance_v2.py` | ~6min | 73方向正式验收 |
| `python run_nonideal_v2.py` | ~1min | 非理想实验(量化/失效/频偏) |
| `python run_curved_verify.py` | ~11min | 曲面阵列SOCP验证 |
| `python run_bounded_socp.py` | ~1min | SOCP失效补偿验证(负结果) |

## 当前项目状态

### 已完成
- 32×32(1024阵元) Taylor基线: SLL ≤ -35 dBc, 100%达标
- LCMV置零: 4个零陷 ≤ -38 dB, SLL保持
- 73方向+200随机方向验证
- 非理想实验(量化/失效/扰动/频偏)
- SOCP负结果研究: 均匀平面阵下Taylor已接近最优

### 进行中
- 曲面阵列AI可行性验证(已确认SOCP改善4.2-4.6dB)
- 下一步: DeepSets网络训练

### 负结果(有价值)
- 阵元失效补偿: SOCP无法改善(物理限制)
- 非均匀坐标(±λ/20): SOCP无改善(Taylor已最优)

## 分支策略

- `main`: 稳定分支，只合并已验证的代码
- 开发新功能时创建分支: `git checkout -b feature/curved-ai`
- 提交前必须通过测试: `python tests/test_*.py`

## 代码结构

```
project/
├── mylib/              # 核心库(6模块)
│   ├── antenna_calc.py # 物理计算
│   ├── evaluation.py   # uv域评估器
│   ├── sum_diff.py     # 和差波束+LCMV
│   ├── train.py        # 训练(NPU/GPU/CPU)
│   ├── dataset.py      # 数据生成
│   └── models.py       # LSTM seq2seq
├── tests/              # 52项测试
├── run_*.py            # 实验脚本
└── outputs/            # 结果JSON
```

## 关键技术决策

1. 相位统一弧度, psi = k*x - phase
2. 主瓣排除: 3×3dB_BW口径(竞赛标准)
3. LCMV: 先归一化w_ref使主瓣响应=1
4. 失效mask: 用布尔索引w[fmask]=0, 禁用~active_idx
5. SOCP solver: CLARABEL(SCS在大规模不收敛)
6. 评估器: uv均匀网格 + -30dB连通域

## 联系方式

GitHub Issues: https://github.com/zwh1025/ai-antenna-synthesis/issues
