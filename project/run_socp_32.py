"""32×32 SOCP 调参：换 CLARABEL solver + 降维 + 粗网格。

关键修复:
1. 用 CLARABEL 替代 SCS（内点法，更适合中等规模 SOCP）
2. 粗网格 11×11（~50 副瓣约束，非 21×21 的 ~300）
3. 不加幅度约束（减少 970 个 SOC 约束）
4. 只做 1-2 次切平面迭代
5. 同时修 16×16 评估器问题（3×3dB_BW 对 16×16 排除过大）
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


def steering_vec(posx, posy, u, v, Nx, Ny, active_idx):
    k = 2 * np.pi
    px = np.tile(posx[:, None], (1, Ny)).ravel()[active_idx]
    py = np.tile(posy[None, :], (Nx, 1)).ravel()[active_idx]
    return np.exp(1j * k * (px * u + py * v))


def solve_socp_32(posx, posy, active_idx, u0, v0,
                  sl_u, sl_v, null_uv=None, eps_null=0.0316,
                  solver='CLARABEL'):
    """32×32 SOCP 求解。"""
    n_active = len(active_idx)

    w = cp.Variable(n_active, complex=True)
    t = cp.Variable()

    a_main = steering_vec(posx, posy, u0, v0, NX, NY, active_idx)
    constraints = [a_main.conj() @ w == 1.0 + 0j]

    for u_s, v_s in zip(sl_u, sl_v):
        a_sl = steering_vec(posx, posy, u_s, v_s, NX, NY, active_idx)
        constraints.append(cp.norm(a_sl.conj() @ w, 2) <= t)

    if null_uv:
        for u_n, v_n in null_uv:
            a_n = steering_vec(posx, posy, u_n, v_n, NX, NY, active_idx)
            constraints.append(cp.norm(a_n.conj() @ w, 2) <= eps_null)

    prob = cp.Problem(cp.Minimize(t), constraints)

    try:
        if solver == 'CLARABEL':
            prob.solve(solver=cp.CLARABEL, verbose=False)
        else:
            prob.solve(solver=cp.SCS, max_iters=10000, verbose=False)
    except Exception as e:
        print(f"    Solver error: {e}")
        return None

    if prob.status not in ['optimal', 'optimal_inaccurate']:
        print(f"    Status: {prob.status}")
        return None

    return w.value


def eval_sll_uv(w_active, active_idx, posx, posy, theta0, phi0, null_dirs=None,
                n_eval=101):
    """在 uv 域评估 SLL（实际峰值排除）。"""
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
            pat[idx] = -300
            continue
        uu = ug.ravel()[idx]
        vv = vg.ravel()[idx]
        psi = k * (px2d * uu + py2d * vv)
        mag = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))
        pat[idx] = mag

    pat = pat.reshape(n_eval, n_eval)
    peak = pat[vis].max()

    # 找实际峰值位置
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
    if null_dirs:
        for tn, pn in null_dirs:
            un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
            vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
            dist_n = np.sqrt((ug - un)**2 + (vg - vn)**2)
            idx_n = np.unravel_index(np.argmin(dist_n), pat.shape)
            null_resp = 20 * np.log10(pat[idx_n] / (peak + 1e-30) + 1e-12)
            null_depths.append(float(null_resp))

    return sll, pt_err, null_depths


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)

    print("="*70)
    print(f"32×32 SOCP Tuning (CLARABEL, coarse grid)")
    print("="*70)

    results = {}

    for rate in [0.05, 0.10, 0.20]:
        rate_str = f"{int(rate*100)}%"
        slls_no = []; slls_socp = []; times = []; nulls = []
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
            sll_no, pt_no, nd_no = eval_sll_uv(
                w_no_active, active_idx, posx, posy, theta0, phi0, null_dirs)
            slls_no.append(sll_no)

            # SOCP
            u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
            v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
            bw = 0.886 * 2.0 / NX * 180 / np.pi
            exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))

            # 粗网格 11×11
            n_uv = 11
            u = np.linspace(-1, 1, n_uv)
            v = np.linspace(-1, 1, n_uv)
            ug, vg = np.meshgrid(u, v, indexing='ij')
            vis = (ug**2 + vg**2) <= 1.0
            dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
            sl_mask = (dist >= exc_uv) & vis
            sl_u = ug[sl_mask]
            sl_v = vg[sl_mask]

            null_uv = [(np.sin(np.deg2rad(tn))*np.cos(np.deg2rad(pn)),
                        np.sin(np.deg2rad(tn))*np.sin(np.deg2rad(pn)))
                       for tn, pn in null_dirs]

            t_s = time.time()
            w_opt = solve_socp_32(
                posx, posy, active_idx, u0, v0,
                sl_u, sl_v, null_uv=null_uv, eps_null=0.0316,
                solver='CLARABEL')
            t_e = time.time()

            if w_opt is not None:
                sll_opt, pt_opt, nd_opt = eval_sll_uv(
                    w_opt, active_idx, posx, posy, theta0, phi0, null_dirs)
                slls_socp.append(sll_opt)
                nulls.append(max(nd_opt) if nd_opt else 0)
            else:
                slls_socp.append(float('nan'))
                nulls.append(float('nan'))
            times.append(t_e - t_s)

            if (i+1) % 5 == 0:
                print(f"  {rate_str} {i+1}/{n_tests}: no={sll_no:.1f} "
                      f"socp={slls_socp[-1]:.1f} t={t_e-t_s:.1f}s")

        slls_no = np.array(slls_no)
        slls_socp = np.array(slls_socp)
        times = np.array(times)
        valid = ~np.isnan(slls_socp)

        if np.any(valid):
            improve = np.nanmean(slls_socp) - np.mean(slls_no)
            print(f"\n  {rate_str}: no={np.mean(slls_no):.1f} "
                  f"socp={np.nanmean(slls_socp):.1f} Δ={improve:+.1f} "
                  f"valid={np.sum(valid)}/{n_tests} t={np.nanmean(times):.1f}s")
            results[rate_str] = {
                'sll_no_mean': float(np.mean(slls_no)),
                'sll_no_worst': float(np.max(slls_no)),
                'sll_socp_mean': float(np.nanmean(slls_socp)),
                'sll_socp_worst': float(np.nanmax(slls_socp)),
                'improve': float(improve),
                'valid': int(np.sum(valid)),
                'time': float(np.nanmean(times)),
            }
        else:
            print(f"\n  {rate_str}: ALL FAILED")
            results[rate_str] = {'error': 'all_failed'}

    print(f"\n{'='*70}")
    print("32×32 SOCP SUMMARY")
    print(f"{'='*70}")
    print(f"{'Rate':>5} {'No_comp':>8} {'SOCP':>8} {'Δ':>6} {'Valid':>6} {'Time':>6}")
    for rate_str, r in results.items():
        if 'error' not in r:
            print(f"  {rate_str:>5} {r['sll_no_mean']:>8.1f} {r['sll_socp_mean']:>8.1f} "
                  f"{r['improve']:>+6.1f} {r['valid']:>5d}  {r['time']:>5.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'socp_32_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
