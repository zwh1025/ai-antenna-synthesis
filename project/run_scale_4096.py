"""64×64（4096阵元）规模泛化质量验证：v3 模型跨规模能力实测。

背景：此前 4096 阵元仅有推理计时基准（NPU 1.23 ms，随机输入），从未验证
方向图质量——"结构支持规模迁移"只是计时口径的主张。本脚本补齐质量证据：
在 64×64 平面阵与 64×64 曲面阵（α=0.12）上，用 v3 模型（仅在 32×32
即 1024 阵元上训练）直接推理，与 Taylor 坐标基线对比副瓣质量。

规模外推的两个分布偏移（如实测）：
  1. 坐标特征：训练 |x/8|≤0.97，64×64 达 ±1.97（超出 2 倍）；
  2. 权值特征：主瓣归一化权值 ~1/N，训练 N=1024 时 |w|·1024=O(1)，
     4096 阵元时字面复用 1024 会使特征缩小 4 倍。

因此测试两种推理变体（都合法，报告两者）：
  A 字面口径：特征缩放沿用训练常数 WEIGHT_SCALE=1024；
  B 语义口径：特征缩放用 N=4096——训练时常数 1024 恰等于 N_train，
    故"×N"才是"相对幅度"语义的自然推广，使特征量级回到训练分布。

评价：eval_dense_3d（81×81 密集网格，与曲面训练/教师对比同口径）。
曲面侧另跑 SOCP 切平面（θ=30）一个方向，计算 4096 阵元的 AI 恢复率。

输出: outputs/scale_4096.json
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from mylib.deepsets import DeepSetsModel
from run_curved_verify import (
    generate_curved_array, coordinate_taylor_3d, eval_dense_3d,
)
from run_generate_teacher import normalize_weights
from run_deepsets_train import _get_null_dirs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'deepsets_model_v3_256.pt')
ASSETS = os.path.join(OUTPUT_DIR, 'ascend_return_20260831', 'github_991916e7_assets')

NX = NY = 64
N_ELEM = NX * NY
SLL = 35
COORD_NORM = 8.0
SLL_NORM = 50.0
HIDDEN = 256

PLANAR_DIRS = [(0, 0), (0, 45), (30, 0), (30, 45), (60, 0), (60, 45)]
CURVED_DIRS = [(0, 0), (30, 0), (60, 0)]
CURVED_ALPHA = 0.12
SOCP_DIR = (30, 0)


def _normalize_weights_torch(w, px, py, pz, u0, v0, w0):
    """normalize_weights 的 torch 等价实现（绕过本机 OMP 冲突）。"""
    k = 2 * np.pi
    a_main = np.exp(1j * k * (px * u0 + py * v0 + pz * w0))
    a_t = torch.as_tensor(a_main, dtype=torch.complex128)
    w_t = torch.as_tensor(w, dtype=torch.complex128)
    resp = (a_t.conj() @ w_t).item()
    if abs(resp) < 1e-12:
        return w
    return w / resp


def model_predict(model, px, py, pz, amp_x, amp_y, theta0, phi0, scale):
    """v3 前向。scale: 特征/残差缩放常数（A=1024 字面 / B=N 语义）。"""
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    w_t = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0)

    n = len(px)
    feat = np.stack([
        px / COORD_NORM, py / COORD_NORM, pz / COORD_NORM,
        w_t.real * scale, w_t.imag * scale,
        np.full(n, u0), np.full(n, v0), np.full(n, w0),
        np.full(n, SLL / SLL_NORM),
    ], axis=-1).astype(np.float32)

    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    w_ai = w_t + (delta[:, 0] + 1j * delta[:, 1]) / scale
    ratio = float(np.sqrt(np.sum(np.abs(delta) ** 2)) /
                  (np.sqrt(np.sum(np.abs(w_t) ** 2)) * scale + 1e-30))
    return w_ai, w_t, ratio


def eval_case(model, px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs):
    out = {}
    sll_t, pt_t, nd_t, _ = eval_dense_3d(
        coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0),
        px, py, pz, theta0, phi0, null_dirs)
    out['taylor'] = {'sll': float(sll_t), 'pt': float(pt_t)}

    for tag, scale in [('A_literal_1024', 1024.0), ('B_semantic_N', float(N_ELEM))]:
        w_ai, w_t, ratio = model_predict(model, px, py, pz, amp_x, amp_y,
                                         theta0, phi0, scale)
        sll_a, pt_a, nd_a, _ = eval_dense_3d(w_ai, px, py, pz,
                                             theta0, phi0, null_dirs)
        out[tag] = {
            'sll': float(sll_a) if not np.isnan(sll_a) else None,
            'pt': float(pt_a),
            'delta_ratio': ratio,
            'vs_taylor_db': (float(sll_a) - float(sll_t)
                             if not np.isnan(sll_a) else None),
        }
    return out


def run_socp_4096(px, py, pz, theta0, phi0, null_dirs, sll_taylor, w_taylor_n):
    """SOCP 切平面（复用训练管线），4096 变量。"""
    from run_multi_scan_generate import run_socp_cutting
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    t0 = time.time()
    w_socp, sll_socp = run_socp_cutting(
        px, py, pz, u0, v0, w0, theta0, phi0, null_dirs,
        sll_taylor, w_taylor_n)
    return w_socp, float(sll_socp), time.time() - t0


def main():
    print('=' * 78, flush=True)
    print('Scale Generalization: v3 (trained on 32x32=1024) on 64x64=4096',
          flush=True)
    print('=' * 78, flush=True)

    model = DeepSetsModel(input_dim=9, hidden_dim=HIDDEN, output_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu',
                                     weights_only=True))
    model.eval()
    print(f'model: {MODEL_PATH} (trained only on 1024-element arrays)',
          flush=True)

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    results = {'description': 'v3 scale generalization 1024->4096 quality test',
               'array': f'{NX}x{NY}={N_ELEM}',
               'variants': {'A_literal_1024': 'feature scale 1024 (training constant)',
                            'B_semantic_N': 'feature scale N=4096 (relative-amplitude semantics)'}

               }

    # ---------- 平面 64×64 ----------
    print('\n[1] planar 64x64 (pz=0)', flush=True)
    px = np.tile(posx[:, None], (1, NY)).ravel()
    py = np.tile(posy[None, :], (NX, 1)).ravel()
    pz = np.zeros(N_ELEM)
    planar = []
    for theta0, phi0 in PLANAR_DIRS:
        null_dirs = _get_null_dirs(theta0, phi0)
        r = eval_case(model, px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs)
        r['dir'] = [theta0, phi0]
        planar.append(r)
        print('  (th=%2d,ph=%3d): Taylor=%6.2f | A=%6.2f (%+.2f) | B=%6.2f (%+.2f)'
              % (theta0, phi0, r['taylor']['sll'],
                 r['A_literal_1024']['sll'], r['A_literal_1024']['vs_taylor_db'],
                 r['B_semantic_N']['sll'], r['B_semantic_N']['vs_taylor_db']),
              flush=True)
    results['planar'] = planar

    # ---------- 曲面 64×64 ----------
    # 两种口径:
    #  matched: alpha=0.03 —— 与训练 alpha=0.12 的 32x32 是同一物理抛物面
    #           (口径×2 则 alpha/4, 下垂 z_max=14.9λ 落在训练范围内),
    #           这是真正的"同一表面、更多阵元"规模测试;
    #  aggressive: alpha=0.12 —— 下垂 59.5λ, z/8 特征达 7.4, 远超训练
    #           范围(≤2.25), 属曲率外推极限测试, 如实报告。
    print('\n[2] curved 64x64 matched-alpha 0.03 (same physical paraboloid '
          'as training 0.12@32x32)', flush=True)
    rng = np.random.RandomState(42)
    pxc = np.tile(posx[:, None], (1, NY))
    pyc = np.tile(posy[None, :], (NX, 1))
    pxc = pxc + rng.uniform(-0.02, 0.02, (NX, NY))
    pyc = pyc + rng.uniform(-0.02, 0.02, (NX, NY))
    pzc = 0.03 * (pxc ** 2 + pyc ** 2)
    pxc, pyc, pzc = pxc.ravel(), pyc.ravel(), pzc.ravel()
    curved = []
    w_taylor_cache = {}
    for theta0, phi0 in CURVED_DIRS:
        null_dirs = _get_null_dirs(theta0, phi0)
        r = eval_case(model, pxc, pyc, pzc, amp_x, amp_y, theta0, phi0, null_dirs)
        r['dir'] = [theta0, phi0]
        curved.append(r)
        print('  (th=%2d,ph=%3d): Taylor=%6.2f | A=%6.2f (%+.2f) | B=%6.2f (%+.2f)'
              % (theta0, phi0, r['taylor']['sll'],
                 r['A_literal_1024']['sll'], r['A_literal_1024']['vs_taylor_db'],
                 r['B_semantic_N']['sll'], r['B_semantic_N']['vs_taylor_db']),
              flush=True)
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0 = np.cos(np.deg2rad(theta0))
        w_t = coordinate_taylor_3d(pxc, pyc, pzc, amp_x, amp_y, theta0, phi0)
        w_taylor_cache[(theta0, phi0)] = _normalize_weights_torch(
            w_t, pxc, pyc, pzc, u0, v0, w0)
    results['curved_matched_alpha0.03'] = curved

    # aggressive: alpha=0.12 在 64x64 (下垂59.5λ, 训练外曲率外推)
    print('\n[2b] curved 64x64 aggressive-alpha 0.12 (sagitta 59.5λ, '
          'beyond training range — curvature extrapolation limit)',
          flush=True)
    rng2 = np.random.RandomState(42)
    pxa = np.tile(posx[:, None], (1, NY))
    pya = np.tile(posy[None, :], (NX, 1))
    pxa = pxa + rng2.uniform(-0.02, 0.02, (NX, NY))
    pya = pya + rng2.uniform(-0.02, 0.02, (NX, NY))
    pza = 0.12 * (pxa ** 2 + pya ** 2)
    pxa, pya, pza = pxa.ravel(), pya.ravel(), pza.ravel()
    curved_agg = []
    for theta0, phi0 in CURVED_DIRS:
        null_dirs = _get_null_dirs(theta0, phi0)
        r = eval_case(model, pxa, pya, pza, amp_x, amp_y, theta0, phi0, null_dirs)
        r['dir'] = [theta0, phi0]
        curved_agg.append(r)
        print('  (th=%2d,ph=%3d): Taylor=%6.2f | A=%6.2f (%+.2f) | B=%6.2f (%+.2f)'
              % (theta0, phi0, r['taylor']['sll'],
                 r['A_literal_1024']['sll'], r['A_literal_1024']['vs_taylor_db'],
                 r['B_semantic_N']['sll'], r['B_semantic_N']['vs_taylor_db']),
              flush=True)
    results['curved_aggressive_alpha0.12'] = curved_agg

    # ---------- SOCP 参照（matched 曲面 θ=30）----------
    print(f'\n[3] SOCP cutting-plane on 4096 vars (matched alpha=0.03, '
          f'dir {SOCP_DIR}) ...', flush=True)
    try:
        th0, ph0 = SOCP_DIR
        sll_t_ref = [c for c in curved if c['dir'] == [th0, ph0]][0]['taylor']['sll']
        w_socp, sll_socp, t_socp = run_socp_4096(
            pxc, pyc, pzc, th0, ph0, _get_null_dirs(th0, ph0),
            sll_t_ref, w_taylor_cache[(th0, ph0)])
        sll_ai_a = [c for c in curved if c['dir'] == [th0, ph0]][0]['A_literal_1024']['sll']
        sll_ai_b = [c for c in curved if c['dir'] == [th0, ph0]][0]['B_semantic_N']['sll']
        rec_a = ((sll_ai_a - sll_t_ref) / (sll_socp - sll_t_ref) * 100
                 if abs(sll_socp - sll_t_ref) > 0.01 else None)
        rec_b = ((sll_ai_b - sll_t_ref) / (sll_socp - sll_t_ref) * 100
                 if abs(sll_socp - sll_t_ref) > 0.01 else None)
        results['socp_reference'] = {
            'dir': list(SOCP_DIR), 'sll_taylor': sll_t_ref,
            'sll_socp': sll_socp, 'time_s': t_socp,
            'sll_ai_A': sll_ai_a, 'sll_ai_B': sll_ai_b,
            'recovery_A_pct': rec_a, 'recovery_B_pct': rec_b,
        }
        print('  Taylor=%.2f SOCP=%.2f (%.1fs) | AI_A=%.2f (恢复%.0f%%) '
              'AI_B=%.2f (恢复%.0f%%)'
              % (sll_t_ref, sll_socp, t_socp, sll_ai_a, rec_a or -1,
                 sll_ai_b, rec_b or -1), flush=True)
    except Exception as e:
        results['socp_reference'] = {'error': str(e)}
        print(f'  SOCP 失败: {e}', flush=True)

    # 计时参照（引用既有冻结基准）
    bench = os.path.join(ASSETS, 'benchmark_report.json')
    if os.path.exists(bench):
        b = json.load(open(bench, encoding='utf-8'))
        inf = b.get('infer', {})
        results['timing_reference'] = {
            'note': '随机输入计时基准（既有冻结数据）',
            'ne4096_bs1_NPU_mean_ms': inf.get('ne4096_bs1_NPU', {}).get('mean'),
            'ne4096_bs1_CPU_mean_ms': inf.get('ne4096_bs1_CPU', {}).get('mean'),
        }

    out_path = os.path.join(OUTPUT_DIR, 'scale_4096.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f'\nSaved: {out_path}')
    print('=' * 78, flush=True)


if __name__ == '__main__':
    main()
