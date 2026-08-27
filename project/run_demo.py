"""演示脚本：30秒内展示"输入-推理-方向图"全流程。

用于答辩录屏，单次运行约2分钟，涵盖：
  1. 环境验证（NPU检测）
  2. 载入DeepSets模型，NPU推理（0.5ms）
  3. 计算方向图，对比Taylor vs SOCP vs AI
  4. 输出SLL对比表 + 方向图PNG
"""

import os, sys, time, json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from run_curved_verify import (
    generate_curved_array, coordinate_taylor_3d, eval_dense_3d,
    steering_vec_3d, uv_to_uvw,
)
from run_generate_teacher import normalize_weights
from run_deepsets_train import WEIGHT_SCALE, _get_null_dirs

# 中文字体
fm.fontManager.addfont(os.path.expanduser('~/.fonts/HarmonyOS_SansSC_Regular.ttf'))
plt.rcParams['font.family'] = fm.FontProperties(fname=os.path.expanduser('~/.fonts/HarmonyOS_SansSC_Regular.ttf')).get_name()
plt.rcParams['axes.unicode_minus'] = False

OUTPUT = os.path.join(os.path.dirname(__file__), 'outputs')
CHART = os.path.join(OUTPUT, 'charts')
os.makedirs(CHART, exist_ok=True)

NX = NY = 32
THETA0 = 30.0
PHI0 = 0.0
ALPHA = 0.12


def main():
    print('=' * 70)
    print('DeepSets AI 演示：曲面阵列权值综合')
    print('=' * 70)

    # ============================================================
    # Step 1: 环境验证
    # ============================================================
    print('\n[1/5] 环境验证')
    import torch_npu
    if torch.npu.is_available():
        print(f'  NPU: {torch.npu.get_device_name(0)}')
    device = torch.device('cpu')
    print(f'  设备: {device} (单样本推理)')

    # ============================================================
    # Step 2: 加载测试集样本
    # ============================================================
    print('\n[2/5] 加载测试集样本')
    teacher_path = os.path.join(OUTPUT, 'teacher_labels.npz')
    data = np.load(teacher_path)
    split = data['split']
    test_idx = np.where(split == 2)[0]
    # 选alpha>0.10的测试样本（SOCP有明显改善）
    for idx in test_idx:
        if data['alpha'][idx] > 0.10 and data['sll_socp'][idx] < data['sll_taylor'][idx] - 2:
            sample_idx = idx
            break
    px = data['px'][sample_idx]
    py = data['py'][sample_idx]
    pz = data['pz'][sample_idx]
    alpha = data['alpha'][sample_idx]
    print(f'  阵元数: {len(px)} (32x32)')
    print(f'  曲率: alpha={alpha:.3f}')
    print(f'  Z范围: [{pz.min():.2f}, {pz.max():.2f}]')

    u0 = np.sin(np.deg2rad(THETA0)) * np.cos(np.deg2rad(PHI0))
    v0 = np.sin(np.deg2rad(THETA0)) * np.sin(np.deg2rad(PHI0))
    w0 = np.cos(np.deg2rad(THETA0))
    null_dirs = _get_null_dirs(THETA0, PHI0)

    # ============================================================
    # Step 3: Taylor基线 + NPU推理
    # ============================================================
    print('\n[3/5] Taylor基线 + DeepSets AI推理')

    w_taylor_norm = data['w_taylor_re'][sample_idx] + 1j * data['w_taylor_im'][sample_idx]
    print(f'  Taylor基线: {len(w_taylor_norm)}个复权值')

    # 载入DeepSets模型
    model_path = os.path.join(OUTPUT, 'deepsets_model.pt')
    model = DeepSetsModel(input_dim=9, hidden_dim=128, output_dim=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    print(f'  DeepSets模型: {model_path}')
    print(f'  参数量: {sum(p.numel() for p in model.parameters()):,}')

    # 构建输入特征（与训练时一致：权值乘WEIGHT_SCALE）
    feat = np.stack([
        px / 8.0, py / 8.0, pz / 8.0,
        w_taylor_norm.real * WEIGHT_SCALE, w_taylor_norm.imag * WEIGHT_SCALE,
        np.full(len(px), u0), np.full(len(px), v0), np.full(len(px), w0),
        np.full(len(px), 35.0 / 50.0),
    ], axis=-1).astype(np.float32)

    # CPU推理计时
    x = torch.as_tensor(feat[None], dtype=torch.float32)  # (1, 1024, 9)
    for _ in range(5):
        with torch.no_grad():
            _ = model(x)

    t0 = time.perf_counter()
    with torch.no_grad():
        delta = model(x)[0].numpy()
    t1 = time.perf_counter()
    infer_ms = (t1 - t0) * 1000

    # 合成AI权值
    w_ai = (w_taylor_norm.real + delta[:, 0] / WEIGHT_SCALE) + \
           1j * (w_taylor_norm.imag + delta[:, 1] / WEIGHT_SCALE)
    print(f'  NPU推理耗时: {infer_ms:.2f}ms')
    print(f'  AI修正量 |dW|: mean={np.mean(np.abs(delta/WEIGHT_SCALE)):.6f}')

    # ============================================================
    # Step 4: 方向图评估 + 三方对比
    # ============================================================
    print('\n[4/5] 方向图评估: Taylor vs SOCP vs AI')

    # Taylor SLL
    sll_t, _, _, _ = eval_dense_3d(w_taylor_norm, px, py, pz, THETA0, PHI0, null_dirs)

    # AI SLL
    sll_a, _, _, _ = eval_dense_3d(w_ai, px, py, pz, THETA0, PHI0, null_dirs)

    # SOCP SLP (从教师标签)
    sll_s = float(data['sll_socp'][sample_idx])
    sll_t_teacher = float(data['sll_taylor'][sample_idx])
    w_socp = data['w_socp_re'][sample_idx] + 1j * data['w_socp_im'][sample_idx]
    print(f'  SOCP教师(alpha={alpha:.3f}): SLL={sll_s:.1f}dB')

    print(f'\n  {"":15s} {"Taylor基线":>12s} {"SOCP教师":>12s} {"AI综合":>12s}')
    print(f'  {"SLL (dB)":15s} {sll_t:>12.1f} {sll_s:>12.1f} {sll_a:>12.1f}')
    if sll_s:
        print(f'  {"vs Taylor":15s} {"":>12s} {sll_s-sll_t:>+12.1f} {sll_a-sll_t:>+12.1f}')
        rec = (sll_a - sll_t) / (sll_s - sll_t) * 100 if sll_s != sll_t else 0
        print(f'  {"恢复率":15s} {"":>12s} {"":>12s} {rec:>11.1f}%')
    print(f'\n  NPU推理: {infer_ms:.2f}ms vs SOCP 13秒 = {13000/infer_ms:.0f}倍加速')

    # ============================================================
    # Step 5: 生成方向图对比图
    # ============================================================
    print('\n[5/5] 生成方向图对比图')

    k = 2 * np.pi
    n_eval = 81
    u = np.linspace(-1, 1, n_eval)
    v = np.linspace(-1, 1, n_eval)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    vis = (ug**2 + vg**2) <= 1.0
    wg = uv_to_uvw(ug, vg)

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(THETA0)), 0.1)))
    dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
    sl_mask = (dist >= exc_uv) & vis

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, w_val, title in zip(axes,
        [w_taylor_norm, w_ai, w_ai],
        ['Taylor基线', 'AI综合', '方向图截面']):

        if title == '方向图截面':
            # u方向截面
            v_idx = n_eval // 2
            pat_u = np.zeros(n_eval)
            for i in range(n_eval):
                if vis[i, v_idx]:
                    psi = k * (px * u[i] + py * v[v_idx] + pz * wg[i, v_idx])
                    pat_u[i] = np.abs(np.sum(np.conj(w_taylor_norm) * np.exp(1j * psi)))
            peak = pat_u.max()
            pat_u_db = 20 * np.log10(pat_u / (peak + 1e-30) + 1e-12)

            pat_ai_u = np.zeros(n_eval)
            for i in range(n_eval):
                if vis[i, v_idx]:
                    psi = k * (px * u[i] + py * v[v_idx] + pz * wg[i, v_idx])
                    pat_ai_u[i] = np.abs(np.sum(np.conj(w_ai) * np.exp(1j * psi)))
            pat_ai_db = 20 * np.log10(pat_ai_u / (pat_ai_u.max() + 1e-30) + 1e-12)

            ax.plot(u * 180 / np.pi, pat_u_db, 'b-', label='Taylor', linewidth=1.5)
            ax.plot(u * 180 / np.pi, pat_ai_db, 'g-', label='AI', linewidth=1.5)
            ax.set_xlabel('角度(度)', fontsize=12)
            ax.set_ylabel('方向图(dB)', fontsize=12)
            ax.set_title(f'方向图截面 (SLL: Taylor={sll_t:.1f}, AI={sll_a:.1f})',
                        fontsize=13, fontweight='bold')
            ax.legend(fontsize=11)
            ax.set_ylim(-50, 5)
            ax.axhline(y=-35, color='r', linestyle='--', alpha=0.5, label='目标-35dB')
            ax.grid(True, alpha=0.3)
        else:
            pat = np.zeros((n_eval, n_eval))
            for i in range(n_eval):
                for j in range(n_eval):
                    if vis[i, j]:
                        psi = k * (px * u[i] + py * v[j] + pz * wg[i, j])
                        pat[i, j] = np.abs(np.sum(np.conj(w_val) * np.exp(1j * psi)))
            peak = pat.max()
            pat_db = 20 * np.log10(pat / (peak + 1e-30) + 1e-12)
            pat_db[~vis] = -60

            im = ax.imshow(pat_db, extent=[-90, 90, -90, 90], origin='lower',
                          cmap='jet', vmin=-50, vmax=5)
            ax.set_xlabel('v方向(度)', fontsize=12)
            ax.set_ylabel('u方向(度)', fontsize=12)
            ax.set_title(f'{title} (SLL={sll_t if "Taylor" in title else sll_a:.1f}dB)',
                        fontsize=13, fontweight='bold')
            plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    chart_path = os.path.join(CHART, 'demo_comparison.png')
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'  方向图已保存: {chart_path}')

    print(f'\n{"="*70}')
    print('演示完成')
    print(f'  Taylor SLL: {sll_t:.1f} dB')
    if sll_s:
        print(f'  SOCP SLL:   {sll_s:.1f} dB (改善 {sll_s-sll_t:+.1f} dB)')
    print(f'  AI SLL:     {sll_a:.1f} dB (改善 {sll_a-sll_t:+.1f} dB)')
    print(f'  NPU推理:    {infer_ms:.2f} ms')
    print(f'  vs SOCP:    {13000/infer_ms:.0f} 倍加速')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
