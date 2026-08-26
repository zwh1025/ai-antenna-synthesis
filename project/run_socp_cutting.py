"""SOCP 切平面迭代：提高 SLL 从 -20dB 到 -30~-35dB。

流程:
  1. 粗网格(11×11)求解 SOCP
  2. 密集网格(81×81)验证，找最差副瓣点
  3. 加入最差点到约束，重新求解
  4. best-so-far 保存正式 SLL 最优
  5. 最终用 101×101 独立网格评估

关键: 求解网格和验证网格不同，防止网格作弊。
"""

import os, sys, time, json
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35


def steering_vec(posx, posy, u, v, active_idx):
    k = 2 * np.pi
    px = np.tile(posx[:, None], (1, NY)).ravel()[active_idx]
    py = np.tile(posy[None, :], (NX, 1)).ravel()[active_idx]
    return np.exp(1j * k * (px * u + py * v))


def solve_socp(posx, posy, active_idx, u0, v0,
               sl_u, sl_v, null_uv, eps_null=0.0316):
    """SOCP 求解。返回 w_opt 或 None。"""
    n_active = len(active_idx)
    w = cp.Variable(n_active, complex=True)
    t = cp.Variable()

    a_main = steering_vec(posx, posy, u0, v0, active_idx)
    constraints = [a_main.conj() @ w == 1.0 + 0j]

    for u_s, v_s in zip(sl_u, sl_v):
        a_sl = steering_vec(posx, posy, u_s, v_s, active_idx)
        constraints.append(cp.norm(a_sl.conj() @ w, 2) <= t)

    for u_n, v_n in null_uv:
        a_n = steering_vec(posx, posy, u_n, v_n, active_idx)
        constraints.append(cp.norm(a_n.conj() @ w, 2) <= eps_null)

    prob = cp.Problem(cp.Minimize(t), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None
    return w.value


def eval_pattern_fast(w_full, posx, posy, u_arr, v_arr, vis_mask):
    """快速方向图评估（线性值）。"""
    k = 2 * np.pi
    px2d = np.tile(posx[:, None], (1, NY)).ravel()
    py2d = np.tile(posy[None, :], (NX, 1)).ravel()
    pat = np.zeros(len(u_arr))
    for i in range(len(u_arr)):
        if not vis_mask[i]:
            pat[i] = 0
            continue
        psi = k * (px2d * u_arr[i] + py2d * v_arr[i])
        pat[i] = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))
    return pat


def cutting_plane_socp(posx, posy, active_idx, theta0, phi0, null_dirs,
                       n_coarse=11, n_fine=81, n_eval=101,
                       n_iters=4, n_add_per_iter=5, eps_null=0.0316):
    """切平面迭代 SOCP。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))

    null_uv = [(np.sin(np.deg2rad(tn))*np.cos(np.deg2rad(pn)),
                np.sin(np.deg2rad(tn))*np.sin(np.deg2rad(pn)))
               for tn, pn in null_dirs]

    # 初始粗网格约束点
    u_c = np.linspace(-1, 1, n_coarse)
    v_c = np.linspace(-1, 1, n_coarse)
    ug_c, vg_c = np.meshgrid(u_c, v_c, indexing='ij')
    vis_c = (ug_c**2 + vg_c**2) <= 1.0
    dist_c = np.sqrt((ug_c - u0)**2 + (vg_c - v0)**2)
    sl_mask_c = (dist_c >= exc_uv) & vis_c
    sl_u = list(ug_c[sl_mask_c])
    sl_v = list(vg_c[sl_mask_c])

    # 密集验证网格
    u_f = np.linspace(-1, 1, n_fine)
    v_f = np.linspace(-1, 1, n_fine)
    ug_f, vg_f = np.meshgrid(u_f, v_f, indexing='ij')
    vis_f = (ug_f**2 + vg_f**2) <= 1.0
    dist_f = np.sqrt((ug_f - u0)**2 + (vg_f - v0)**2)
    sl_mask_f = (dist_f >= exc_uv) & vis_f
    u_f_flat = ug_f[sl_mask_f]
    v_f_flat = vg_f[sl_mask_f]
    vis_f_flat = vis_f.ravel()

    px2d = np.tile(posx[:, None], (1, NY)).ravel()
    py2d = np.tile(posy[None, :], (NX, 1)).ravel()

    best_sll = float('inf')
    best_w = None

    for it in range(n_iters):
        # 求解 SOCP
        w_opt = solve_socp(posx, posy, active_idx, u0, v0,
                           sl_u, sl_v, null_uv, eps_null)
        if w_opt is None:
            break

        # 构建完整权值
        w_full = np.zeros(NX * NY, dtype=complex)
        w_full[active_idx] = w_opt

        # 密集网格验证
        pat = eval_pattern_fast(w_full, posx, posy, u_f_flat, v_f_flat,
                                np.ones(len(u_f_flat), dtype=bool))

        # 主瓣峰值
        psi_main = k * (px2d * u0 + py2d * v0)
        peak = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi_main)))

        if peak < 1e-10:
            break

        sll = 20 * np.log10(np.max(pat) / (peak + 1e-30) + 1e-12)

        # best-so-far
        if sll < best_sll:
            best_sll = sll
            best_w = w_opt.copy()

        # 找最差副瓣点
        pat_norm = pat / (peak + 1e-12)
        sorted_idx = np.argsort(pat_norm)[::-1]

        n_added = 0
        for idx in sorted_idx:
            if n_added >= n_add_per_iter:
                break
            u_new = u_f_flat[idx]
            v_new = v_f_flat[idx]
            # 检查是否已存在
            already = False
            for su, sv in zip(sl_u, sl_v):
                if abs(su - u_new) < 0.005 and abs(sv - v_new) < 0.005:
                    already = True
                    break
            if not already:
                sl_u.append(u_new)
                sl_v.append(v_new)
                n_added += 1

    return best_w, best_sll, len(sl_u)


def eval_final(w_active, active_idx, posx, posy, theta0, phi0, null_dirs,
               n_eval=101):
    """最终独立评估（101×101网格）。"""
    k = 2 * np.pi
    w_full = np.zeros(NX * NY, dtype=complex)
    w_full[active_idx] = w_active

    px2d = np.tile(posx[:, None], (1, NY)).ravel()
    py2d = np.tile(posy[None, :], (NX, 1)).ravel()

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    u = np.linspace(-1, 1, n_eval)
    v = np.linspace(-1, 1, n_eval)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    vis = (ug**2 + vg**2) <= 1.0

    pat = np.zeros(n_eval * n_eval)
    for idx in range(n_eval * n_eval):
        if not vis.ravel()[idx]:
            pat[idx] = 0
            continue
        uu = ug.ravel()[idx]
        vv = vg.ravel()[idx]
        psi = k * (px2d * uu + py2d * vv)
        pat[idx] = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))

    pat = pat.reshape(n_eval, n_eval)
    peak = pat[vis].max()

    # 峰值位置
    idx_peak = np.unravel_index(np.argmax(np.where(vis, pat, -1)), pat.shape)
    peak_u = ug[idx_peak]
    peak_v = vg[idx_peak]
    peak_theta = np.degrees(np.arcsin(np.clip(np.sqrt(peak_u**2 + peak_v**2), 0, 1)))
    peak_phi = np.degrees(np.arctan2(peak_v, peak_u)) % 360
    pt_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    # 3×3dB_BW 排除
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_deg = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
    sl_mask = (dist >= np.sin(np.deg2rad(exc_deg))) & vis
    sll = float(20 * np.log10(np.max(pat[sl_mask]) / (peak + 1e-30) + 1e-12)) if np.any(sl_mask) else float('nan')

    # 零陷
    null_depths = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        dist_n = np.sqrt((ug - un)**2 + (vg - vn)**2)
        idx_n = np.unravel_index(np.argmin(dist_n), pat.shape)
        null_resp = 20 * np.log10(pat[idx_n] / (peak + 1e-30) + 1e-12)
        null_depths.append(float(null_resp))

    # PAPR
    amp = np.abs(w_active)
    papr = float(amp.max() / (np.mean(amp) + 1e-12))

    return sll, pt_err, max(null_depths), papr


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)

    print("="*70)
    print(f"SOCP Cutting Plane ({NX}×{NY})")
    print(f"Coarse: 11×11, Fine: 81×81, Eval: 101×101")
    print(f"Iterations: 4, Add 5 points/iter")
    print("="*70)

    results = {}
    t_total = time.time()

    for rate in [0.05, 0.10, 0.20]:
        rate_str = f"{int(rate*100)}%"
        slls_no = []; slls_socp = []; times = []; nulls = []; paprs = []
        n_tests = 10

        for i in range(n_tests):
            theta0 = rng.uniform(0, 60)
            phi0 = rng.uniform(0, 360)
            null_dirs = [(30,(phi0+90)%360),(30,(phi0+180)%360),
                         (30,(phi0+270)%360),(55,(phi0+45)%360)]

            n_fail = int(NX * NY * rate)
            fmask = np.zeros(NX * NY, dtype=bool)
            fmask[rng.choice(NX * NY, n_fail, replace=False)] = True
            active_idx = np.where(~fmask)[0]

            px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)
            al, pl = capon_nulling_2d(posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)

            # 无补偿
            w_no = (al * np.exp(1j * pl)).ravel()
            w_no[fmask] = 0
            w_no_active = w_no[active_idx].copy()
            sll_no, pt_no, nd_no, papr_no = eval_final(
                w_no_active, active_idx, posx, posy, theta0, phi0, null_dirs)
            slls_no.append(sll_no)

            # SOCP 切平面
            t_s = time.time()
            w_opt, sll_fast, n_constraints = cutting_plane_socp(
                posx, posy, active_idx, theta0, phi0, null_dirs,
                n_coarse=11, n_fine=81, n_eval=101,
                n_iters=4, n_add_per_iter=5, eps_null=0.0316)
            t_e = time.time()

            if w_opt is not None:
                sll_opt, pt_opt, nd_opt, papr_opt = eval_final(
                    w_opt, active_idx, posx, posy, theta0, phi0, null_dirs)
                slls_socp.append(sll_opt)
                nulls.append(nd_opt)
                paprs.append(papr_opt)
            else:
                slls_socp.append(float('nan'))
                nulls.append(float('nan'))
                paprs.append(float('nan'))
            times.append(t_e - t_s)

            if (i+1) % 5 == 0:
                print(f"  {rate_str} {i+1}/{n_tests}: no={sll_no:.1f} "
                      f"socp={slls_socp[-1]:.1f} t={t_e-t_s:.1f}s "
                      f"constr={n_constraints}")

        slls_no = np.array(slls_no)
        slls_socp = np.array(slls_socp)
        times = np.array(times)
        nulls = np.array(nulls)
        paprs = np.array(paprs)
        valid = ~np.isnan(slls_socp)

        if np.any(valid):
            improve = np.nanmean(slls_socp) - np.mean(slls_no)
            print(f"\n  {rate_str}: no={np.mean(slls_no):.1f} "
                  f"socp={np.nanmean(slls_socp):.1f} Δ={improve:+.1f} "
                  f"valid={np.sum(valid)}/{n_tests} t={np.nanmean(times):.1f}s "
                  f"null={np.nanmean(nulls):.1f} papr={np.nanmean(paprs):.2f}")
            results[rate_str] = {
                'sll_no_mean': float(np.mean(slls_no)),
                'sll_no_worst': float(np.max(slls_no)),
                'sll_socp_mean': float(np.nanmean(slls_socp)),
                'sll_socp_worst': float(np.nanmax(slls_socp)),
                'improve': float(improve),
                'valid': int(np.sum(valid)),
                'time': float(np.nanmean(times)),
                'null_mean': float(np.nanmean(nulls)),
                'papr_mean': float(np.nanmean(paprs)),
            }

    t_end = time.time()
    print(f"\n{'='*70}")
    print("SOCP CUTTING PLANE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Rate':>5} {'No_comp':>8} {'SOCP':>8} {'Δ':>6} {'Worst':>8} {'Null':>6} {'PAPR':>6} {'Time':>6} {'Valid':>6}")
    for rate_str, r in results.items():
        print(f"  {rate_str:>5} {r['sll_no_mean']:>8.1f} {r['sll_socp_mean']:>8.1f} "
              f"{r['improve']:>+6.1f} {r['sll_socp_worst']:>8.1f} "
              f"{r['null_mean']:>6.1f} {r['papr_mean']:>6.2f} "
              f"{r['time']:>5.1f}s {r['valid']:>5d}/{n_tests}")
    print(f"\n  Total time: {t_end-t_total:.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'socp_cutting_plane.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
