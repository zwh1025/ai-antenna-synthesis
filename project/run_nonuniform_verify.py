"""非均匀阵列 SOCP vs 坐标Taylor 可行性验证。

生成 30 个非均匀 32×32 阵列（位置扰动 ±0.05λ/±0.10λ/±0.20λ）
比较：
  1. 坐标Taylor: 理想Taylor幅度 + 实际坐标扫描相位
  2. SOCP: 最小化最大副瓣, 约束主瓣=1 + 4零陷

关键: 扫描相位必须用实际坐标计算, 不能用理想坐标
"""

import os, sys, time, json
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase, beam_steering_phase_2d,
    combine_2d_excitation, angular_distance_deg,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35


def generate_nonuniform_array(posx_ideal, posy_ideal, perturb, rng):
    """在规则32×32阵列上加入独立2D位置扰动。"""
    px = posx_ideal[:, None] + rng.uniform(-perturb, perturb, (NX, NY))
    py = posy_ideal[None, :] + rng.uniform(-perturb, perturb, (NX, NY))
    return px, py  # (Nx, Ny) 2D arrays


def steering_vec_nonuniform(px_flat, py_flat, u, v):
    """非均匀阵列导向向量。"""
    k = 2 * np.pi
    return np.exp(1j * k * (px_flat * u + py_flat * v))


def coordinate_taylor(posx_ideal, posy_ideal, px, py, amp_x, amp_y,
                      theta0, phi0):
    """坐标Taylor: 理想Taylor幅度 + 实际坐标扫描相位。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    # Taylor 幅度（理想阵列, 不变）
    amp_2d = np.outer(amp_x, amp_y)  # (Nx, Ny)

    # 扫描相位: 用实际坐标
    phase_2d = k * (px * u0 + py * v0)  # (Nx, Ny)

    return amp_2d, phase_2d


def solve_socp_nonuniform(px_flat, py_flat, u0, v0,
                          sl_u, sl_v, null_uv, eps_null=0.0316):
    """非均匀阵列 SOCP。"""
    n = len(px_flat)
    w = cp.Variable(n, complex=True)
    t = cp.Variable()

    a_main = steering_vec_nonuniform(px_flat, py_flat, u0, v0)
    constraints = [a_main.conj() @ w == 1.0 + 0j]

    for u_s, v_s in zip(sl_u, sl_v):
        a_sl = steering_vec_nonuniform(px_flat, py_flat, u_s, v_s)
        constraints.append(cp.norm(a_sl.conj() @ w, 2) <= t)

    for u_n, v_n in null_uv:
        a_n = steering_vec_nonuniform(px_flat, py_flat, u_n, v_n)
        constraints.append(cp.norm(a_n.conj() @ w, 2) <= eps_null)

    prob = cp.Problem(cp.Minimize(t), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None
    return w.value


def eval_dense_nonuniform(w, px_flat, py_flat, theta0, phi0, null_dirs,
                          n_eval=101):
    """密集uv网格评估。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    u = np.linspace(-1, 1, n_eval)
    v = np.linspace(-1, 1, n_eval)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    vis = (ug**2 + vg**2) <= 1.0

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
    sl_mask = (dist >= exc_uv) & vis

    u_flat = ug[sl_mask]
    v_flat = vg[sl_mask]

    pat = np.zeros(len(u_flat))
    for i in range(len(u_flat)):
        psi = k * (px_flat * u_flat[i] + py_flat * v_flat[i])
        pat[i] = np.abs(np.sum(np.conj(w) * np.exp(1j * psi)))

    psi_main = k * (px_flat * u0 + py_flat * v0)
    main_resp = np.abs(np.sum(np.conj(w) * np.exp(1j * psi_main)))
    if main_resp < 1e-10:
        return float('nan'), 0, []

    sll = 20 * np.log10(np.max(pat) / (main_resp + 1e-30))

    # 指向误差：在全可见域找主瓣峰值
    vis_flat = vis.ravel()
    ug_flat = ug.ravel()
    vg_flat = vg.ravel()
    pat_all = np.zeros(len(ug_flat))
    for i in range(len(ug_flat)):
        if vis_flat[i]:
            psi = k * (px_flat * ug_flat[i] + py_flat * vg_flat[i])
            pat_all[i] = np.abs(np.sum(np.conj(w) * np.exp(1j * psi)))
    peak_all_idx = np.argmax(pat_all)
    peak_u = ug_flat[peak_all_idx]
    peak_v = vg_flat[peak_all_idx]
    peak_theta = np.degrees(np.arcsin(np.clip(np.sqrt(peak_u**2 + peak_v**2), 0, 1)))
    peak_phi = np.degrees(np.arctan2(peak_v, peak_u)) % 360
    pt_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    # 零陷
    null_depths = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        psi_n = k * (px_flat * un + py_flat * vn)
        nd = 20 * np.log10(np.abs(np.sum(np.conj(w) * np.exp(1j * psi_n))) / (main_resp + 1e-30))
        null_depths.append(float(nd))

    # 最差10个副瓣点
    sorted_idx = np.argsort(pat)[::-1][:10]
    worst = [(u_flat[i], v_flat[i]) for i in sorted_idx]

    return sll, pt_err, null_depths, worst


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)
    perturbs = [0.05, 0.10, 0.20]
    n_per_perturb = 10  # 每种扰动10个阵列
    theta0 = 30.0
    phi0 = 0.0
    null_dirs = [(30, 90), (30, 180), (30, 270), (55, 45)]

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    null_uv = [(np.sin(np.deg2rad(tn))*np.cos(np.deg2rad(pn)),
                np.sin(np.deg2rad(tn))*np.sin(np.deg2rad(pn)))
               for tn, pn in null_dirs]

    print("="*70)
    print(f"Non-uniform Array SOCP vs Coordinate Taylor ({NX}×{NY})")
    print(f"Perturbs: {perturbs}, {n_per_perturb} arrays each")
    print(f"Scan: θ={theta0}° φ={phi0}°, 4 nulls")
    print("="*70)

    results = []
    t_total = time.time()

    for perturb in perturbs:
        slls_taylor = []; slls_socp = []; times = []
        nulls_taylor = []; nulls_socp = []

        for i in range(n_per_perturb):
            px, py = generate_nonuniform_array(posx_ideal, posy_ideal, perturb, rng)
            px_flat = px.ravel()
            py_flat = py.ravel()

            # 坐标Taylor基线
            amp_t, phase_t = coordinate_taylor(
                posx_ideal, posy_ideal, px, py, amp_x, amp_y, theta0, phi0)
            w_taylor = (amp_t * np.exp(1j * phase_t)).ravel()

            sll_t, pt_t, nd_t, worst_t = eval_dense_nonuniform(
                w_taylor, px_flat, py_flat, theta0, phi0, null_dirs)
            slls_taylor.append(sll_t)
            nulls_taylor.append(max(nd_t) if nd_t else 0)

            # SOCP
            # 初始粗网格约束
            n_coarse = 15
            u_c = np.linspace(-1, 1, n_coarse)
            v_c = np.linspace(-1, 1, n_coarse)
            ug_c, vg_c = np.meshgrid(u_c, v_c, indexing='ij')
            vis_c = (ug_c**2 + vg_c**2) <= 1.0
            bw = 0.886 * 2.0 / NX * 180 / np.pi
            exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
            dist_c = np.sqrt((ug_c - u0)**2 + (vg_c - v0)**2)
            sl_mask_c = (dist_c >= exc_uv) & vis_c
            sl_u = list(ug_c[sl_mask_c])
            sl_v = list(vg_c[sl_mask_c])

            best_sll = sll_t  # best-so-far = 坐标Taylor
            best_w = w_taylor.copy()

            t_s = time.time()
            for it in range(5):  # 5轮切平面
                w_opt = solve_socp_nonuniform(
                    px_flat, py_flat, u0, v0, sl_u, sl_v, null_uv)
                if w_opt is None:
                    break

                sll_s, pt_s, nd_s, worst_s = eval_dense_nonuniform(
                    w_opt, px_flat, py_flat, theta0, phi0, null_dirs)

                if sll_s < best_sll - 0.05:
                    best_sll = sll_s
                    best_w = w_opt.copy()

                # 加入最差点
                for u_w, v_w in worst_s:
                    already = any(abs(su-u_w)<0.01 and abs(sv-v_w)<0.01
                                   for su, sv in zip(sl_u, sl_v))
                    if not already:
                        sl_u.append(u_w)
                        sl_v.append(v_w)
            t_e = time.time()

            slls_socp.append(best_sll)
            times.append(t_e - t_s)
            if best_sll < sll_t:
                nulls_socp.append(max(nd_s) if nd_s else 0)
            else:
                nulls_socp.append(nulls_taylor[-1])

            if (i+1) % 5 == 0:
                print(f"  ±{perturb}λ {i+1}/{n_per_perturb}: "
                      f"taylor={sll_t:.1f} socp={best_sll:.1f} "
                      f"Δ={best_sll-sll_t:+.1f} t={t_e-t_s:.1f}s")

        slls_taylor = np.array(slls_taylor)
        slls_socp = np.array(slls_socp)
        times = np.array(times)
        improve = np.mean(slls_socp) - np.mean(slls_taylor)

        print(f"\n  ±{perturb}λ: taylor={np.mean(slls_taylor):.1f} "
              f"socp={np.mean(slls_socp):.1f} Δ={improve:+.1f} "
              f"t={np.mean(times):.1f}s")

        results.append({
            'perturb': perturb,
            'taylor_mean': float(np.mean(slls_taylor)),
            'taylor_worst': float(np.max(slls_taylor)),
            'socp_mean': float(np.mean(slls_socp)),
            'socp_worst': float(np.max(slls_socp)),
            'improve': float(improve),
            'time': float(np.mean(times)),
            'null_taylor': float(np.mean(nulls_taylor)),
            'null_socp': float(np.mean(nulls_socp)),
        })

    t_end = time.time()
    print(f"\n{'='*70}")
    print("NON-UNIFORM ARRAY VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Perturb':>8} {'Taylor':>8} {'SOCP':>8} {'Δ':>6} {'Worst_T':>8} {'Worst_S':>8} {'Time':>6}")
    for r in results:
        print(f"  ±{r['perturb']}λ {r['taylor_mean']:>8.1f} {r['socp_mean']:>8.1f} "
              f"{r['improve']:>+6.1f} {r['taylor_worst']:>8.1f} {r['socp_worst']:>8.1f} "
              f"{r['time']:>5.1f}s")

    # 决策
    all_improve = np.mean([r['improve'] for r in results])
    print(f"\n  Mean improve: {all_improve:+.2f} dB")
    if all_improve < -0.5:
        print("  → SOCP effective on non-uniform arrays. Proceed to AI.")
    else:
        print("  → SOCP not effective. Need larger perturbation or different task.")

    with open(os.path.join(OUTPUT_DIR, 'nonuniform_socp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Total: {t_end-t_total:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
