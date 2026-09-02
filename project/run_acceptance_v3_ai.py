"""DeepSets v3 模型平面阵竞赛验收测试。

问题：32×32 理想平面阵的竞赛指标（73 标准方向 + 200 随机方向）此前仅由
Taylor+LCMV 解析系统验证，缺少模型实测。本脚本用 v3 模型在完全相同的
验收口径下补测。

两条测试臂：
  ai_direct : v3 模型直接输出权值（无 LCMV 后处理）
  ai_lcmv   : v3 模型输出 + LCMV 自适应置零（比赛合规的完整 AI 管线，
              与已通过的 Taylor+LCMV 同口径对比）

注意：v3 的平面训练样本目标 ΔW=0（平面阵 Taylor 已最优），因此模型在
平面阵上是"Taylor 复现器"，自身不产生零陷——零陷能力由 LCMV 提供，
这正是混合系统设计的预期行为。

评估口径与 run_acceptance_v2 / run_random_validation 完全一致：
  - evaluate_uv（201×201 uv 网格，-30dB 连通域 + 3×3dB_BW 双口径）
  - 73 标准方向零陷：acceptance 约定（30° 基准 + 角距 ≥15°）
  - 200 随机方向：θ∈[0,60], φ∈[0,360)，随机 4 零陷，抛物线插值指向

环境注意：本机 Anaconda base 的 libiomp 与 torch 的 libomp 冲突，
numpy BLAS（matmul/lstsq）在 torch 加载后会段错误（0xc06d007f）。
因此方向图计算与 LCMV 最小二乘全部改用 torch 线性代数实现，
先与原 numpy 实现做数值一致性校验（阈值 1e-6 dB / 1e-8 权值）。

输出: outputs/acceptance_v3_ai.json
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from mylib import evaluation as ev
from mylib.deepsets import DeepSetsModel
from run_curved_verify import coordinate_taylor_3d
from run_generate_teacher import normalize_weights
from run_acceptance_v2 import get_scan_directions, get_null_dirs
from run_random_validation import random_null_dirs, fine_peak_search

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
MODEL_PATH = os.path.join(OUTPUT_DIR, 'deepsets_model_v3_256.pt')

NX = NY = 32
SLL = 35
N_RANDOM = 200
COORD_NORM = 8.0
SLL_NORM = 50.0
WEIGHT_SCALE = 1024.0

_orig_pattern = ev._pattern_on_uv_grid


def _capon_nulling_2d_torch(posx, posy, amp_2d, phase_2d, theta0, phi0,
                            null_directions, lamb=1.0):
    """capon_nulling_2d 的 torch 实现（数学等价，规避 numpy BLAS）。

    R_inv = I，故 CR = C^H C；其余步骤与 mylib/sum_diff.py 逐行对应。
    """
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = len(posx), len(posy)
    k = 2 * np.pi / lamb

    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    w_ref = (amp_2d * np.exp(1j * phase_2d)).ravel()

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    a_main = np.exp(1j * k * (posx_2d * u0 + posy_2d * v0)).ravel()

    main_resp = np.vdot(a_main, w_ref)
    w_ref = w_ref / main_resp

    cols = [a_main]
    for tn, pn in null_directions:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        cols.append(np.exp(1j * k * (posx_2d * un + posy_2d * vn)).ravel())

    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0

    Ct = torch.as_tensor(C, dtype=torch.complex128)
    wt = torch.as_tensor(w_ref, dtype=torch.complex128)
    ft = torch.as_tensor(f, dtype=torch.complex128)

    CR = Ct.conj().T @ Ct
    residual = ft - Ct.conj().T @ wt
    x = None
    for driver in ('gelsd', 'gelsy'):
        try:
            x = torch.linalg.lstsq(CR, residual.unsqueeze(-1), driver=driver)[0]
            break
        except Exception:
            continue
    if x is None:
        x = torch.linalg.solve(CR + 1e-10 * torch.eye(CR.shape[0],
                                                     dtype=torch.complex128),
                               residual.unsqueeze(-1))
    w_opt = wt + Ct @ x.squeeze(-1)
    w_opt = w_opt.numpy()

    w_mat = w_opt.reshape(Nx, Ny)
    new_amp = np.abs(w_mat)
    if new_amp.max() > 0:
        new_amp = new_amp / new_amp.max()
    new_phase = np.angle(w_mat) % (2 * np.pi)
    return new_amp, new_phase


def _torch_lstsq(a, b, rcond=None):
    """np.linalg.lstsq 的 torch 替代（兜底，正常路径不再触达）。"""
    a = np.asarray(a)
    b = np.asarray(b)
    dt = torch.complex128 if np.iscomplexobj(a) or np.iscomplexobj(b) else torch.float64
    A = torch.as_tensor(a, dtype=dt)
    B = torch.as_tensor(b, dtype=dt)
    b_vec = (B.dim() == 1)
    if b_vec:
        B = B.unsqueeze(-1)
    X = None
    for driver in ('gelsd', 'gelsy'):
        try:
            X = torch.linalg.lstsq(A, B, driver=driver)[0]
            break
        except Exception:
            continue
    if X is None:
        X = torch.linalg.solve(A + 1e-10 * torch.eye(A.shape[0], dtype=dt), B)
    x = X.squeeze(-1).numpy() if b_vec else X.numpy()
    return x, None, None, None


def _pattern_on_uv_grid_torch(amp, phase, posx, posy, n_uv=201, lamb=1.0):
    """torch 实现的 _pattern_on_uv_grid（数值与原 numpy 实现一致）。

    F(u,v) = |sum_n conj(w_n) exp(j k (x_n u + y_n v))|
    用欧拉公式拆成 cos/sin 实数矩阵乘，避免 numpy BLAS。
    """
    k = 2 * np.pi / lamb
    amp = np.asarray(amp, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = amp.shape

    u = np.linspace(-1, 1, n_uv)
    v = np.linspace(-1, 1, n_uv)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    visible = (u_grid**2 + v_grid**2) <= 1.0

    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    w = amp * np.exp(1j * phase)
    w_conj = np.conj(w.ravel())
    wc_re = torch.as_tensor(w_conj.real)
    wc_im = torch.as_tensor(w_conj.imag)
    pos_t = torch.as_tensor(np.stack(
        [posx_2d.ravel(), posy_2d.ravel()], axis=1))  # (N, 2)

    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    visible_flat = visible.ravel()
    idx_vis = np.where(visible_flat)[0]
    uv_t = torch.as_tensor(np.stack(
        [u_flat[idx_vis], v_flat[idx_vis]], axis=1))  # (P, 2)

    P = len(idx_vis)
    vals = torch.empty(P, dtype=torch.float64)
    CH = 2048
    for s in range(0, P, CH):
        psi = k * (uv_t[s:s + CH] @ pos_t.T)  # (CH, N)
        c, sn = torch.cos(psi), torch.sin(psi)
        re = c @ wc_re - sn @ wc_im
        im = c @ wc_im + sn @ wc_re
        vals[s:s + CH] = torch.sqrt(re * re + im * im)

    pattern = np.full(n_uv * n_uv, -300.0)
    pattern[idx_vis] = vals.numpy()

    peak = pattern[idx_vis].max()
    pattern_db = np.where(pattern > 0,
                          20 * np.log10(pattern / (peak + 1e-30) + 1e-12), -300)
    pattern_db = pattern_db.reshape(n_uv, n_uv)
    return pattern_db, u_grid, v_grid, visible


def apply_patches():
    """替换 BLAS 相关实现并做数值一致性校验。"""
    ev._pattern_on_uv_grid = _pattern_on_uv_grid_torch
    np.linalg.lstsq = _torch_lstsq


def verify_patches(posx, posy, amp_x, amp_y, theta0, phi0, model):
    """torch 实现与原 numpy 实现的一致性校验。

    pattern：与原 numpy 实现逐点比对（<1e-6 dB）。
    capon：numpy BLAS 版在 torch 加载后必崩（OMP 冲突），改为功能性
    校验——torch 版 capon 输出须满足零陷约束（目标点响应 ≤ −35 dB）。
    """
    from mylib.antenna_calc import beam_steering_phase_2d, combine_2d_excitation

    px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp2d, ph2d = combine_2d_excitation(amp_x, amp_y, px, py)

    p_orig, _, _, _ = _orig_pattern(amp2d, ph2d, posx, posy, n_uv=101)
    p_new, _, _, _ = _pattern_on_uv_grid_torch(amp2d, ph2d, posx, posy, n_uv=101)
    diff = np.max(np.abs(p_orig - p_new))
    print(f"  pattern check: max|diff| = {diff:.2e} dB")
    assert diff < 1e-6, f"pattern mismatch: {diff}"

    null_dirs = get_null_dirs(theta0, phi0)
    amp_l, ph_l = _capon_nulling_2d_torch(posx, posy, amp2d, ph2d, theta0,
                                          phi0, null_dirs)
    r = ev.evaluate_uv(amp_l, ph_l, posx, posy, theta0, phi0,
                       null_dirs=null_dirs)
    nulls = [nr['target_response'] for nr in r['null_results']]
    worst = max(nulls)
    print(f"  capon functional check: worst null target response = "
          f"{worst:.1f} dB (expect <= -35)")
    assert worst <= -35.0, f"capon nulling failed: {worst:.1f} dB"


def model_predict(model, px, py, pz, amp_x, amp_y, theta0, phi0):
    """v3 模型推理：返回复权值（Taylor 基线 + ΔW）。"""
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))
    w_t = normalize_weights(w_t, px, py, pz, u0, v0, w0)

    n = len(px)
    feat = np.stack([
        px / COORD_NORM, py / COORD_NORM, pz / COORD_NORM,
        w_t.real * WEIGHT_SCALE, w_t.imag * WEIGHT_SCALE,
        np.full(n, u0), np.full(n, v0), np.full(n, w0),
        np.full(n, SLL / SLL_NORM),
    ], axis=-1).astype(np.float32)

    with torch.no_grad():
        delta = model(torch.as_tensor(feat[None]))[0].numpy()
    return w_t + (delta[:, 0] + 1j * delta[:, 1]) / WEIGHT_SCALE


def to_amp_phase(w_flat):
    return np.abs(w_flat).reshape(NX, NY), np.angle(w_flat).reshape(NX, NY)


_pos_1d = (None, None)


def eval_one(model, posx_flat, posy_flat, pz, amp_x, amp_y, theta0, phi0, null_dirs):
    """一条方向的两个测试臂。返回 dict。"""
    posx, posy = _pos_1d
    w_ai = model_predict(model, posx_flat, posy_flat, pz, amp_x, amp_y,
                         theta0, phi0)
    amp_a, ph_a = to_amp_phase(w_ai)

    r_dir = ev.evaluate_uv(amp_a, ph_a, posx, posy, theta0, phi0,
                           null_dirs=null_dirs)

    amp_l, ph_l = _capon_nulling_2d_torch(posx, posy, amp_a, ph_a, theta0,
                                          phi0, null_dirs)
    r_l = ev.evaluate_uv(amp_l, ph_l, posx, posy, theta0, phi0,
                         null_dirs=null_dirs)
    nulls_l = [nr['max_3deg'] for nr in r_l['null_results']]
    nulls_d = [nr['max_3deg'] for nr in r_dir['null_results']]

    return {
        'ai_direct': {
            'sll_3bw': r_dir['sll_3bw'], 'sll_fn': r_dir['sll_first_null'],
            'pointing_err': r_dir['pointing_err'], 'bw_3db': r_dir['bw_3db'],
            'worst_null': max(nulls_d) if nulls_d else None,
        },
        'ai_lcmv': {
            'sll_3bw': r_l['sll_3bw'], 'sll_fn': r_l['sll_first_null'],
            'pointing_err': r_l['pointing_err'], 'bw_3db': r_l['bw_3db'],
            'worst_null': max(nulls_l) if nulls_l else None,
        },
    }


def summarize(results, arm, label):
    sll = np.array([r[arm]['sll_3bw'] for r in results])
    nulls = np.array([r[arm]['worst_null'] for r in results])
    pts = np.array([r[arm]['pointing_err'] for r in results])
    bws = np.array([r[arm]['bw_3db'] for r in results])
    rms = float(np.sqrt(np.mean(pts ** 2)))
    bw_mean = float(np.mean(bws))
    print(f"\n  [{label}] {arm} ({len(results)} directions)")
    print(f"    SLL(3x3dB_BW): mean={sll.mean():.2f} worst={sll.max():.2f} dB"
          f"  pass<=-35: {np.sum(sll <= -35)}/{len(sll)}")
    print(f"    worst null(3deg): mean={nulls.mean():.2f} worst={nulls.max():.2f} dB"
          f"  pass<=-30: {np.sum(nulls <= -30)}/{len(nulls)}")
    print(f"    pointing: RMS={rms:.3f} deg (target BW/30={bw_mean/30:.3f})"
          f"  pass: {np.sum(pts <= bw_mean/30)}/{len(pts)}")
    return {
        'sll_3bw_mean': float(sll.mean()), 'sll_3bw_worst': float(sll.max()),
        'sll_pass': int(np.sum(sll <= -35)),
        'null_worst': float(nulls.max()), 'null_pass': int(np.sum(nulls <= -30)),
        'pointing_rms': rms, 'pointing_target': bw_mean / 30,
        'pointing_pass': int(np.sum(pts <= bw_mean / 30)),
    }


def main():
    print("=" * 78, flush=True)
    print("DeepSets v3 Planar Acceptance Test (73 standard + 200 random)", flush=True)
    print("=" * 78, flush=True)

    model = DeepSetsModel(input_dim=9, hidden_dim=256, output_dim=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu',
                                      weights_only=True))
    model.eval()
    print(f"Model loaded: {MODEL_PATH}", flush=True)

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    px = np.tile(posx[:, None], (1, NY)).ravel()
    py = np.tile(posy[None, :], (NX, 1)).ravel()
    pz = np.zeros(NX * NY)

    apply_patches()
    global _pos_1d
    _pos_1d = (posx, posy)
    verify_patches(posx, posy, amp_x, amp_y, 30.0, 45.0, model)
    print("  patches verified", flush=True)

    # ---------- 73 标准方向 ----------
    dirs = get_scan_directions()
    print(f"\nPart 1: {len(dirs)} standard directions", flush=True)
    t0 = time.time()
    std_results = []
    for i, (theta0, phi0) in enumerate(dirs):
        null_dirs = get_null_dirs(theta0, phi0)
        r = eval_one(model, px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs)
        r['theta0'], r['phi0'] = theta0, phi0
        std_results.append(r)
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  {i+1:3d}/{len(dirs)} ({time.time()-t0:.0f}s) "
                  f"th={theta0:.0f} ph={phi0:.0f} "
                  f"dir_sll={r['ai_direct']['sll_3bw']:.2f} "
                  f"lcmv_sll={r['ai_lcmv']['sll_3bw']:.2f} "
                  f"null={r['ai_lcmv']['worst_null']:.1f}", flush=True)

    # ---------- 200 随机方向 ----------
    print(f"\nPart 2: {N_RANDOM} random directions (seed 42)", flush=True)
    rng = np.random.RandomState(42)
    t0 = time.time()
    rnd_results = []
    for i in range(N_RANDOM):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        null_dirs = random_null_dirs(theta0, phi0, rng)
        r = eval_one(model, px, py, pz, amp_x, amp_y, theta0, phi0, null_dirs)
        r['theta0'], r['phi0'] = theta0, phi0

        w_ai = model_predict(model, px, py, pz, amp_x, amp_y, theta0, phi0)
        amp_a, ph_a = to_amp_phase(w_ai)
        amp_l, ph_l = _capon_nulling_2d_torch(posx, posy, amp_a, ph_a, theta0,
                                              phi0, null_dirs)
        pe, _, _ = fine_peak_search(amp_l, ph_l, posx, posy, theta0, phi0)
        r['ai_lcmv']['pointing_err_fine'] = float(pe)
        rnd_results.append(r)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1:3d}/{N_RANDOM} ({time.time()-t0:.0f}s) "
                  f"th={theta0:.1f} ph={phi0:.1f} "
                  f"dir_sll={r['ai_direct']['sll_3bw']:.2f} "
                  f"lcmv_sll={r['ai_lcmv']['sll_3bw']:.2f} "
                  f"pt_fine={pe:.3f}", flush=True)

    # ---------- 汇总 ----------
    print("\n" + "=" * 78)
    print("SUMMARY vs competition targets")
    print("=" * 78)
    summary = {}
    for label, results in [('standard_73', std_results), ('random_200', rnd_results)]:
        summary[label] = {}
        for arm in ['ai_direct', 'ai_lcmv']:
            summary[label][arm] = summarize(results, arm, label)

    pe_fine = np.array([r['ai_lcmv']['pointing_err_fine'] for r in rnd_results])
    rms_fine = float(np.sqrt(np.mean(pe_fine ** 2)))
    print(f"\n  [random_200] ai_lcmv fine pointing RMS = {rms_fine:.4f} deg"
          f" (Taylor+LCMV reference: 0.028 deg)")
    summary['random_200']['ai_lcmv']['pointing_rms_fine'] = rms_fine

    out = {
        'description': 'DeepSets v3 model planar acceptance test, '
                       'same caliber as run_acceptance_v2 + run_random_validation',
        'model': os.path.basename(MODEL_PATH),
        'arms': ['ai_direct (model output only)',
                 'ai_lcmv (model output + LCMV nulling)'],
        'summary': summary,
        'standard_73': std_results,
        'random_200': rnd_results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'acceptance_v3_ai.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}", flush=True)
    print("=" * 78, flush=True)


if __name__ == '__main__':
    main()
