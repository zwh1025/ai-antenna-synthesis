# Stage 1 — Metric / Evaluator Freeze

状态：**STAGE_1_GO**
版本：`OFFICIAL_EVALUATOR_VERSION = "1.0.0"`
日期：2026-09-01

本文冻结后续正式实验使用的唯一 Official Evaluator。它只定义测量规则，不修改 Taylor、Bayliss、LCMV、SOCP、GA、PSO 或 DeepSets 的综合算法、训练数据、loss、checkpoint 和既有 headline number。

实现入口：`project/mylib/official_evaluator.py`
完整单案例入口：`evaluate_official_case(...)`
73-direction acceptance 入口已切换到该 API：`project/run_acceptance_v2.py`。

## 1. Scope

本阶段冻结：

- 坐标系、上半空间和阵因子符号；
- dBc 归一化；
- 和波束 SLL、差波束 SLL；
- null center、null window；
- 和波束 3 dB beamwidth；
- 差波束零交叉 pointing error 和 dataset RMSE；
- inference-only、end-to-end synthesis、optimizer runtime 三类 latency；
- versioned result schema、legacy mapping 和 deterministic unit tests。

本阶段不生成正式 73+200 结果，不重跑大规模 benchmark，不重新训练，不更新 README/技术报告/算法报告中的历史数字。

## 2. Coordinate conventions

阵列位于 x-y 平面，+z 是 broadside 法向。公开 API 的角度单位为 degree，内部计算转换为 radian。

| Quantity | Frozen definition |
|---|---|
| `theta` | 从 +z broadside 量起的俯仰/离轴角，范围 `[0°, 90°]` |
| `phi` | x-y 平面内从 +x 朝 +y 的方位角，按 `[0°, 360°)` 归一化 |
| boresight | `(theta, phi) = (0°, 0°)`；theta=0 时 phi 物理等价 |
| upper hemisphere | `theta ∈ [0°, 90°]`，等价于 `u²+v²≤1` 且 `w=+sqrt(1-u²-v²)` |
| direction cosines | `u=sin(theta)cos(phi)`，`v=sin(theta)sin(phi)`，`w=cos(theta)` |
| ±60° cone | `theta=0°` 一个方向，加上 `theta∈{10°,20°,...,60°}` 和 `phi∈{0°,30°,...,330°}`，共 `1+6×12=73` 个独立方向 |
| wavelength | 坐标按 wavelength 归一化；`k=2π/lambda` |

阵因子沿用仓库已有物理定义：

```text
w_n = amp_n exp(j phase_n)
F(u,v) = Σ conj(w_n) exp(j k (x_n u + y_n v))
```

这与 `mylib.antenna_calc.calculate_2d_pattern` 的 `psi=k(xu+yv)-phase` 一致。相同的 `(theta, phi)`、坐标和权值必须得到相同的 steering response；不再用角度差的平面近似代替球面误差。

## 3. Pattern normalization

方向图先在 official visible domain 求峰值：

```text
P_norm(theta, phi) = |F(theta, phi)| / max_visible |F|
pattern_dBc = 20 log10(max(P_norm, 10^(-300/20)))
```

这等价于功率比的 `10 log10(|F|² / max|F|²)`，因此不是 `10 log10(amplitude)`。`-300 dBc` 是数值显示下限，不代表实验设备动态范围。

和波束与差波束分别以自己的 visible-domain maximum 归一化；差波束中心零点不会因为差波束峰值结构而被归一化为另一个波束的数值。

## 4. Sum-beam SLL definition

### 4.1 Candidate definitions audited

| Method | Definition | Advantages | Problems | Existing usage |
|---|---|---|---|---|
| 3×3 dB BW exclusion | 以目标附近经验排除角 `3×(0.886×2/Nx)/cos(theta0)` 外的最大 dBc 为 SLL | 快、实现简单 | 倍数是经验量；对 2×/3× 和离散网格敏感；不是真正 first-null | `evaluate_uv()['sll_3bw']`、旧 robustness 脚本、README/报告 headline |
| First-null boundary | 沿主瓣截面找峰值之后的第一局部最小值，主瓣外取最大值 | 物理解释直接，与传统 SLL 更接近 | 2D、扫描和非对称阵列需要明确边界和 fallback | 旧脚本声称使用，但 `sll_first_null` 实际只是 connected alias |
| Connected main-lobe | 从峰值出发，以固定 dB threshold 的 8 连通域作为主瓣 | 对任意 2D 网格易实现 | threshold 不是 first-null；可能吞入或切断第一副瓣；对网格/连通性敏感 | 旧 `evaluate_uv()['sll_connected']`，并被错误别名为 `sll_first_null` |

### 4.2 Official choice

Official 字段为：

```text
sum.sll_db = max(pattern_dBc on visible & outside official first-null uv envelope)
```

实现为 `sum_sll_from_pattern_db()` / `evaluate_sum_beam()`：

1. 使用均匀 `u,v ∈ [-1,1]` 网格，默认 `201×201`；圆盘外点不参与评估。
2. 找 visible-domain 的离散峰值 `(u_peak,v_peak)`。
3. 分别沿阵列坐标轴 `u` 和 `v`，在穿过该峰值的 profile 上用 2001 个采样点和双线性插值寻找第一 null。
4. 第一 null 的操作定义是：从 profile 峰值向外，遇到“至少比峰值低 3 dB”的第一个局部最小值。平坦肩部必须结束后不再继续下降才算局部最小值。
5. 若该 profile 没有采到局部最小值，按顺序使用第一个 `-3 dB` crossing，再退化到 visible boundary；使用的 fallback 会写入 `sum.main_lobe.boundary_methods`。
6. 由正/负 u、正/负 v 四个边界组成轴对齐的 first-null uv envelope。其内部为 main-lobe region；边界归入 main-lobe，边界外才是 sidelobe。
7. 在整个 visible upper hemisphere 的 mask 外取最大 dBc。SLL 不使用主平面 cut，也不把 `theta/phi` 矩形外的不可见点算进去。

唯一通过字段和阈值为：

```text
sum.sll_db <= SUM_SLL_THRESHOLD_DB = -35 dBc
```

该选择保留了 first-null 的物理含义，同时把 2D 边界、网格、插值和 fallback 全部显式化。它不以旧报告数字为目标；如果冻结后旧结果变差，必须如实记录。

## 5. Difference-beam SLL definition

差波束不能复用“sum peak 周围一个圆形排除区”：差波束在目标方向有中心零点，并有两个相对的主瓣。

Official 字段为：

```text
difference.sll_db = max(pattern_dBc on visible & outside both difference-lobe envelopes)
```

定义如下：

1. 差波束相对于自己的 visible-domain 最大场幅归一化。
2. 调用方必须声明差轴 `difference_axis_phi_deg`。该角度是 uv 平面从 +u 轴朝 +v 轴的方向；默认 `0°` 对应当前 x 方向 Bayliss 差波束。
3. 在目标 `(u0,v0)` 通过差轴的 profile 上，分别寻找 `q>0` 和 `q<0` 两侧的 lobe peak。
4. 从每个 lobe peak 向外搜索第一 null；再在各自 lobe peak 的横向 profile 上搜索两侧第一 null。
5. 两个 lobe 的 main-lobe region 是各自的差轴区间与横向区间的矩形 envelope 的并集。目标中心零点位于两个 lobe 之间，不会被当成 sidelobe peak。
6. 在整个 visible upper hemisphere、两个 envelope 之外取最大 dBc。已知副瓣值应由该最大值直接返回，而不是取窗口内最深点。

唯一通过字段和阈值为：

```text
difference.sll_db <= DIFFERENCE_SLL_THRESHOLD_DB = -20 dBc
```

若两侧 profile 不可见或无法识别两个 lobe，evaluator 报错而不是自动给出好看的 SLL；这类任务必须先修正输入或明确为不适用。

## 6. Main-lobe exclusion and legacy SLL roles

主瓣排除的 official 规则只有上一节定义的 first-null uv envelope。connected 和 3×BW 仍可作为诊断字段，用来解释历史数字，但不能进入任何 PASS/FAIL、pass-rate 或 headline。

| Historical name | Stage 1 role |
|---|---|
| `sll_first_null` from old `mylib.evaluation.evaluate_uv` | **deprecated legacy alias**；它实际等于 connected SLL，不是真 first-null；不得作为 official 字段 |
| `sll_connected` | legacy diagnostic；固定 `pattern>-30 dB`、8 连通域 |
| `sll_3bw` | diagnostic only；保留旧的经验 3×3dB BW exclusion 以便解释历史结果 |
| `sum.sll_db` | 唯一 official sum SLL，只有它可以判 `≤-35 dBc` |

`evaluate_sum_beam()['diagnostic']` 会同时保存 `sll_connected_db`、`sll_3bw_db` 和 grid size，但不会把它们写成 official pass 字段。

## 7. Null center definition

每个 null 目标保存精确的 `theta_deg`、`phi_deg`。`null_center_db` 由连续 steering response 直接计算：

```text
center_db = 20 log10(|F(theta_null, phi_null)| / max_visible |F|)
```

禁止用 official 粗网格最近点替代目标坐标。LCMV 的零约束若在目标处成立，center response 可以接近机器精度并显示为 `-300 dBc` floor。

阈值映射冻结为：

```text
baseline adaptive-null: center_db <= -30 dBc
strict sum null:        center_db <= -65 dBc
strict difference null: center_db <= -50 dBc
```

四个或更多 null 的 formal result 必须逐点保存 center 值，不能只保存一个平均值或最深值。

## 8. Null window definition

`null_window_worst_db` 是目标周围球面 cap 内的最大响应，即最差抑制度：

- 半径：默认 `3°`；
- 采样：`theta/phi` offset 的 `0.25°` 网格；只保留 upper hemisphere 且球面距离不超过 `3°` 的点；
- 计算：连续 array factor 在这些采样点上的最大 dBc；
- window 包含 exact center，但 center 单独由连续 target response 计算并单独保存。

```text
window_worst_db = max_{angular_distance <= 3°} pattern_dBc
```

window 是 robustness diagnostic，不替代 center：

```text
null_center_db             -> baseline / strict target-point compliance
null_window_worst_db       -> local robustness diagnostic
```

这保留了赛题文字中“点零陷”和仓库旧的 `±3°` 抑制度两种信息，并明确记录 competition wording ambiguity：如果未来赛题明确要求窗口最坏值也必须满足 strict `-65/-50 dBc`，只需改变 threshold mapping，不应改变字段含义。

## 9. Sum 3 dB beamwidth

唯一字段为 `sum.beamwidth_3db_deg`，含义是和波束 3 dB 全宽，不是差波束宽度。

定义：

1. 在固定 `phi0` 的主扫描 great-circle 平面取 signed angle `alpha∈[-90°,90°]`；`alpha<0` 用 `(theta=-alpha, phi=phi0+180°)`，`alpha>=0` 用 `(theta=alpha, phi=phi0)`。
2. 采样步长固定为 `0.05°`，在目标 `theta0±15°` 内找该 cut 的局部峰值。
3. threshold 是该 cut 峰值减 3 dB；向两侧找 crossing。
4. 每个 crossing 使用相邻两点的 dB 线性插值；返回两 crossing 的角度差。
5. 无法找到任一 crossing 返回 `NaN` 并在 metadata 中说明，不用数组边界伪造 BW。

因此离轴扫描的 BW 是目标扫描平面内的 principal great-circle cut；不是全方向平均，也不是固定 `0.886×2/Nx` 的估计值。

## 10. Difference pointing error / RMS

通用 per-case API 是：

```text
pointing_error_deg = arccos(u_target · u_estimate) in degrees
```

它是球面夹角，自动处理 phi seam，例如 `179°` 与 `-179°` 不会被算成 `358°`。

对当前差波束定义，`evaluate_difference_beam()` 把估计方向定义为：目标处沿声明差轴的连续 uv line 上，复响应幅度的最近零交叉；使用 1001 个 `q∈[-0.25,0.25]` 采样并对局部幅度平方做抛物线插值。返回：

```text
difference.pointing_error_deg
difference.zero_crossing_direction
```

数据集级别唯一 RMSE 为：

```text
pointing_rmse_deg = sqrt(mean(pointing_error_deg(case)^2))
```

唯一阈值来源为该案例的和波束 BW：

```text
pointing_threshold_deg = sum.beamwidth_3db_deg / 30
```

固定 `0.15°`、`0.2°` 等数字只有在明确标注为某个案例的 `BW/30` 时才可出现。旧脚本的 coarse-grid peak error、parabolic sum-peak error、mean、worst 和 RMS 不再互相冒充；`0.028° / 0.24° / 0.46°` 分别只能作为旧 provenance 中各自定义下的历史数字，不能视为同一 official metric。

## 11. Latency definitions

Stage 1 只冻结口径，不执行大规模 timing benchmark。统计工具 `summarize_latency(samples_ms)` 返回：`n`、mean、std、P50、P95、P99、min、max。

### 11.1 `inference_only_latency`

只包括已准备好的模型输入到 `model.forward()` 或等价 NPU forward。正式记录必须注明：设备、batch、warm-up 次数、重复次数、同步方式、输入是否已在目标设备、以及统计样本。不得把 preprocessing、权值重建或 pattern evaluation 偷含进来。

### 11.2 `end_to_end_synthesis_latency`

从阵元坐标与综合需求开始，到最终可用复激励权值结束，至少包含：

```text
feature construction
normalization
device transfer when required
model inference
denormalization
residual reconstruction
final complex-weight generation
```

若方向图评估不属于综合过程，应排除并在 result metadata 写明；正式 benchmark 必须保存端到端各阶段或至少完整路径的 raw timings。当前仓库的 supplement 只能证明简化 transfer-inclusive / forward 链路，不能自动升级为完整 synthesis latency。

### 11.3 `optimizer_runtime`

GA、PSO、SOCP 等记录完整 optimization runtime，并明确 solver、网格、cutting iterations、seed、case 数及是否包含 evaluator。纯 neural forward 不得与“optimizer + 多次 evaluation”直接做无条件 speedup 比较。

## 12. Official result schema

`evaluate_official_case()` 返回以下结构；数值结果文件必须额外记录输入/权值/模型的 hash 和生成配置：

```python
{
    "metric_version": "1.0.0",
    "task": {
        "theta0_deg": 0.0,
        "phi0_deg": 0.0,
        "upper_hemisphere": True,
        "wavelength": 1.0,
    },
    "sum": {
        "sll_db": -35.0,
        "sll_threshold_db": -35.0,
        "beamwidth_3db_deg": 4.5,
        "peak_direction": {"theta_deg": 0.0, "phi_deg": 0.0},
        "pointing_error_deg": 0.0,
        "pointing_threshold_deg": 0.15,
        "main_lobe": {"...": "first-null envelope metadata"},
        "diagnostic": {
            "sll_connected_db": -30.0,
            "sll_3bw_db": -35.0,
            "grid_size": 201,
        },
    },
    "difference": None,
    "adaptive_null": {
        "sum": {
            "targets": [],
            "center_db": [],
            "window_worst_db": [],
            "strict_field": "center_db",
        },
        "difference": None,
    },
    "latency": {
        "inference_only": None,
        "end_to_end_synthesis": None,
        "optimizer_runtime": None,
    },
}
```

正式 benchmark 不能依赖 Markdown 人工复制数字；每个 result file 必须包含 `metric_version`，并在 provenance 中记录脚本、配置、seed、样本/方向、权值或模型 hash。

## 13. Evaluator version

当前冻结版本为：

```text
OFFICIAL_EVALUATOR_VERSION = "1.0.0"
```

改变坐标、归一化、main-lobe mask、null window、BW、pointing 或 latency 口径时，必须递增版本并重新生成受影响的正式结果；不得用新 evaluator 覆盖旧结果而不标版本。

## 14. Legacy metric mapping

| Consumer | Old metric / implementation | Canonical replacement | Migrated? |
|---|---|---|---|
| `run_acceptance_v2.py` | `mylib.evaluation.evaluate_uv`; pass-rate 使用 `sll_first_null` 和 `max_3deg` | `evaluate_official_case`; `sum.sll_db`、`adaptive_null.sum.center_db`；3×BW/window 仅 diagnostic | **Yes** |
| `run_random_validation.py` | old `sll_first_null`、`max_3deg`、custom fine peak/BW | 后续正式随机 benchmark 使用 `evaluate_official_case`、official BW、center null 和 spherical pointing RMSE | No；保留以复现历史脚本 |
| `run_2d_sum_diff.py` | `get_2d_sll`、主瓣圆形排除、`min` null、旧 1D BW | `evaluate_official_case` 的 sum/difference API | No；48×48 单场景历史验证 |
| `run_nonideal_v2.py` | 自定义 3×3BW SLL | 新 robustness 结果使用 `evaluate_sum_beam`；旧输出若恢复必须标 legacy | No；避免改变旧实验运行语义 |
| `run_failure_benchmark.py` | 自定义 peak exclusion 3×BW | 新 failure benchmark 使用 canonical sum evaluator | No；历史脚本保留 |
| `mylib/synthesis_2d.py` | `get_2d_sll` 作为 gradient objective/diagnostic | 不迁移；改变它会改变综合算法与历史训练语义 | **Intentionally no** |
| `mylib/antenna_calc.py` 1D helpers | fixed-width `get_sll_1d`、min-null、first crossing BW | 保留为 1D legacy/unit-test helpers；2D formal result 不调用 | **Intentionally no** |
| `mylib/evaluation.py` | `evaluate_uv` connected/3BW 混合返回 | 保留 legacy module，新增 benchmark 不得使用 `sll_first_null` 别名 | **Intentionally no** |

保留 legacy 是为了让旧实验入口仍可复现，不表示它们的输出可以进入 Stage 1 official pass/fail。新的正式 benchmark 必须显式调用 `official_evaluator` 并保存 `metric_version`。

## 15. Ambiguities remaining

以下问题已被字段和 mapping 隔离，但不是本阶段擅自修改的算法问题：

1. 赛题“严格深零”是否要求 center 还是 window worst；本版本采用 center 作为 target-point compliance，window 单独 diagnostic，并保留阈值映射。
2. 当前 difference 物理实现主要是 x 方向 Bayliss；对任意旋转差轴，调用方必须声明 `difference_axis_phi_deg`，不能默认把差波束当 sum beam。
3. first-null 在噪声/强非对称 pattern 中可能没有清晰局部最小值；本版本固定 fallback 并写入 metadata，正式结果若大量触发 fallback 应单独报告。
4. 当前 `run_random_validation.py` 等历史脚本的旧数字仍不具备 official provenance；Stage 2 以后必须重新生成或明确标记为 legacy。
5. AI 平面硬基准与曲面泛化任务的 requirement boundary 仍属于 Stage 0 的 R6.2 任务定义问题，本阶段只保证它们可以使用同一个 schema 记录，未宣称 AI 已达标。

## 16. Unit-test evidence

新增 `project/tests/test_official_evaluator.py`，包含 6 个 deterministic tests：

- field/power normalization scale invariance；
- 已知 sum main lobe 与 sidelobe 的 first-null exclusion；
- 两个 difference lobes、中心 null 和已知 sidelobe；
- exact continuous null center 与 window maximum；
- pointing RMSE 与 phi seam 的球面误差；
- latency statistics API。

额外验证了 canonical module 与 acceptance entry point 可编译；未执行正式 73-direction benchmark。

## 17. Stage 1 freeze decision

| Gate | Decision | Evidence |
|---|---|---|
| A — official sum SLL | PASS | `sum_sll_from_pattern_db` / `evaluate_sum_beam` |
| B — official difference SLL | PASS | `difference_sll_from_pattern_db` / `evaluate_difference_beam` |
| C — center/window null split | PASS | `evaluate_nulls` 输出 `center_db` 和 `window_worst_db` |
| D — unique 3 dB BW | PASS | `_sum_beamwidth`，固定 great-circle、step 和线性插值 |
| E — pointing RMS and BW/30 | PASS | `pointing_error_deg`、`pointing_rmse_deg`、`pointing_threshold_deg` |
| F — latency split | PASS | schema 三类字段和 `summarize_latency` |
| G — synthetic tests | PASS | 新增 6 项 deterministic tests |
| H — new benchmark API | PASS | `run_acceptance_v2.py` 已迁移到 `evaluate_official_case` |
| I — legacy mapping | PASS | 本文第 14 节；旧 evaluator 保留并标 deprecated/legacy |

因此 Stage 1 判定为：

```text
STAGE_1_GO
```

这只表示 measurement ruler 已冻结，不表示现有算法或历史 headline number 已经通过冻结后的正式验收。

GA、PSO、SOCP 等记录完整 optimization runtime，并明确 solver、网格、cutting iterations、seed、case 数及是否包含 evaluator。纯 neural forward 不得与“optimizer + 多次 evaluation”直接做无条件 speedup 比较。
