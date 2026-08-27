"""曲面阵列 SOCP 可行性验证。

32×32=1024阵元, z = α(x²+y²) 抛物面
阵因子: F = sum conj(w) * exp(j*k*(x*u + y*v + z*w_dir))
基线: 平面Taylor幅度 + 实际3D坐标扫描相位
教师: SOCP最小化最大副瓣

退出条件: SOCP改善<1dB → 停止AI路线
"""

import os, sys, time, json
import numpy as np
import cvxpy as cp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    angular_distance_deg,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35


def generate_curved_array(posx_ideal, posy_ideal, curvature, rng):
    """生成抛物面阵列: z = α(x²+y²) + 小扰动。"""
    px = np.tile(posx_ideal[:, None], (1, NY))
    py = np.tile(posy_ideal[None, :], (NX, 1))
    pz = curvature * (px**2 + py**2)
    # 加小扰动增加多样性
    px = px + rng.uniform(-0.02, 0.02, (NX, NY))
    py = py + rng.uniform(-0.02, 0.02, (NX, NY))
    pz = curvature * (px**2 + py**2)
    return px.ravel(), py.ravel(), pz.ravel()


def steering_vec_3d(px, py, pz, u, v, w_dir):
    """3D导向向量。F = sum conj(w_n) * exp(j*k*(x_n*u + y_n*v + z_n*w))"""
    k = 2 * np.pi
    return np.exp(1j * k * (px * u + py * v + pz * w_dir))


def coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0):
    """坐标Taylor: 理想Taylor幅度 + 实际3D坐标扫描相位。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))  # z方向分量

    amp = np.outer(amp_x, amp_y).ravel()
    phase = k * (px * u0 + py * v0 + pz * w0)
    return amp * np.exp(1j * phase)


def solve_socp_3d(px, py, pz, u0, v0, w0,
                   sl_u, sl_v, sl_w, null_uv, null_w, eps_null=0.0316):
    """3D SOCP。"""
    n = len(px)
    w = cp.Variable(n, complex=True)
    t = cp.Variable()

    a_main = steering_vec_3d(px, py, pz, u0, v0, w0)
    constraints = [a_main.conj() @ w == 1.0 + 0j]

    for u_s, v_s, w_s in zip(sl_u, sl_v, sl_w):
        a_sl = steering_vec_3d(px, py, pz, u_s, v_s, w_s)
        constraints.append(cp.norm(a_sl.conj() @ w, 2) <= t)

    for u_n, v_n, w_n in zip(null_uv[0], null_uv[1], null_w):
        a_n = steering_vec_3d(px, py, pz, u_n, v_n, w_n)
        constraints.append(cp.norm(a_n.conj() @ w, 2) <= eps_null)

    prob = cp.Problem(cp.Minimize(t), constraints)
    try:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        return None
    if prob.status not in ['optimal', 'optimal_inaccurate']:
        return None
    return w.value


def uv_to_uvw(u, v):
    """uv方向余弦→3D方向向量。w = sqrt(1-u²-v²)。"""
    w2 = 1.0 - u**2 - v**2
    w = np.sqrt(np.maximum(w2, 0))
    return w


def eval_dense_3d(w, px, py, pz, theta0, phi0, null_dirs, n_eval=81):
    """密集uv网格评估(3D)。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))

    u = np.linspace(-1, 1, n_eval)
    v = np.linspace(-1, 1, n_eval)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    vis = (ug**2 + vg**2) <= 1.0
    wg = uv_to_uvw(ug, vg)

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0)**2 + (vg - v0)**2)
    sl_mask = (dist >= exc_uv) & vis

    # 一次性计算全可见域方向图（避免两次循环）
    vis_flat = vis.ravel()
    ug_flat = ug.ravel()
    vg_flat = vg.ravel()
    wg_flat = wg.ravel()
    n_vis = int(vis_flat.sum())
    pat_all = np.zeros(len(ug_flat))
    for i in range(len(ug_flat)):
        if vis_flat[i]:
            psi = k * (px * ug_flat[i] + py * vg_flat[i] + pz * wg_flat[i])
            pat_all[i] = np.abs(np.sum(np.conj(w) * np.exp(1j * psi)))

    psi_main = k * (px * u0 + py * v0 + pz * w0)
    main_resp = np.abs(np.sum(np.conj(w) * np.exp(1j * psi_main)))
    if main_resp < 1e-10:
        return float('nan'), 0, [], []

    # 从全方向图提取副瓣
    sl_flat_mask = sl_mask.ravel()
    pat_sl = pat_all[sl_flat_mask]
    sll = 20 * np.log10(np.max(pat_sl) / (main_resp + 1e-30))

    # 指向误差：全可见域峰值
    peak_idx_all = np.argmax(pat_all)
    peak_u = ug_flat[peak_idx_all]
    peak_v = vg_flat[peak_idx_all]
    peak_theta = np.degrees(np.arcsin(np.clip(np.sqrt(peak_u**2 + peak_v**2), 0, 1)))
    peak_phi = np.degrees(np.arctan2(peak_v, peak_u)) % 360
    pt_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    null_depths = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        wn = np.cos(np.deg2rad(tn))
        psi_n = k * (px * un + py * vn + pz * wn)
        nd = 20 * np.log10(np.abs(np.sum(np.conj(w) * np.exp(1j * psi_n))) / (main_resp + 1e-30))
        null_depths.append(float(nd))

    sl_u_flat = ug_flat[sl_flat_mask]
    sl_v_flat = vg_flat[sl_flat_mask]
    sl_w_flat = wg_flat[sl_flat_mask]
    sorted_idx = np.argsort(pat_sl)[::-1][:10]
    worst = [(sl_u_flat[i], sl_v_flat[i], sl_w_flat[i]) for i in sorted_idx]

    return sll, pt_err, null_depths, worst


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)
    curvatures = [0.0, 0.02, 0.05, 0.10, 0.15]  # α参数
    n_per_curve = 6
    theta0 = 30.0
    phi0 = 0.0
    null_dirs = [(30, 90), (30, 180), (30, 270), (55, 45)]

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))

    null_u = [np.sin(np.deg2rad(tn))*np.cos(np.deg2rad(pn)) for tn, pn in null_dirs]
    null_v = [np.sin(np.deg2rad(tn))*np.sin(np.deg2rad(pn)) for tn, pn in null_dirs]
    null_w = [np.cos(np.deg2rad(tn)) for tn, pn in null_dirs]

    print("="*70)
    print(f"Curved Array SOCP Verification ({NX}×{NY})")
    print(f"Curvatures: {curvatures}, {n_per_curve} arrays each")
    print(f"Scan: θ={theta0}° φ={phi0}°, 4 nulls")
    print("="*70)

    results = []
    t_total = time.time()

    for curve in curvatures:
        slls_taylor = []; slls_socp = []; times = []
        nulls_t = []; nulls_s = []

        for i in range(n_per_curve):
            px, py, pz = generate_curved_array(posx_ideal, posy_ideal, curve, rng)

            # 坐标Taylor基线
            w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
            sll_t, pt_t, nd_t, worst_t = eval_dense_3d(
                w_taylor, px, py, pz, theta0, phi0, null_dirs)
            slls_taylor.append(sll_t)
            nulls_t.append(max(nd_t) if nd_t else 0)

            # SOCP
            n_coarse = 15
            u_c = np.linspace(-1, 1, n_coarse)
            v_c = np.linspace(-1, 1, n_coarse)
            ug_c, vg_c = np.meshgrid(u_c, v_c, indexing='ij')
            vis_c = (ug_c**2 + vg_c**2) <= 1.0
            wg_c = uv_to_uvw(ug_c, vg_c)
            bw = 0.886 * 2.0 / NX * 180 / np.pi
            exc_uv = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
            dist_c = np.sqrt((ug_c - u0)**2 + (vg_c - v0)**2)
            sl_mask_c = (dist_c >= exc_uv) & vis_c
            sl_u = list(ug_c[sl_mask_c])
            sl_v = list(vg_c[sl_mask_c])
            sl_w = list(wg_c[sl_mask_c])

            best_sll = sll_t
            best_w = w_taylor.copy()

            t_s = time.time()
            for it in range(5):
                w_opt = solve_socp_3d(
                    px, py, pz, u0, v0, w0,
                    sl_u, sl_v, sl_w,
                    (null_u, null_v), null_w)
                if w_opt is None:
                    break

                sll_s, pt_s, nd_s, worst_s = eval_dense_3d(
                    w_opt, px, py, pz, theta0, phi0, null_dirs)

                if sll_s < best_sll - 0.05:
                    best_sll = sll_s
                    best_w = w_opt.copy()

                for u_w, v_w, w_w in worst_s:
                    already = any(abs(su-u_w)<0.01 and abs(sv-v_w)<0.01
                                   for su, sv in zip(sl_u, sl_v))
                    if not already:
                        sl_u.append(u_w)
                        sl_v.append(v_w)
                        sl_w.append(w_w)
            t_e = time.time()

            slls_socp.append(best_sll)
            times.append(t_e - t_s)
            nulls_s.append(max(nd_s) if nd_s and best_sll < sll_t else nulls_t[-1])

            if (i+1) % 3 == 0:
                print(f"  α={curve:.2f} {i+1}/{n_per_curve}: "
                      f"taylor={sll_t:.1f} socp={best_sll:.1f} "
                      f"Δ={best_sll-sll_t:+.1f} t={t_e-t_s:.1f}s")

        slls_taylor = np.array(slls_taylor)
        slls_socp = np.array(slls_socp)
        times = np.array(times)
        improve = np.mean(slls_socp) - np.mean(slls_taylor)

        print(f"  α={curve:.2f}: taylor={np.mean(slls_taylor):.1f} "
              f"socp={np.mean(slls_socp):.1f} Δ={improve:+.1f} "
              f"t={np.mean(times):.1f}s\n")

        results.append({
            'curvature': curve,
            'taylor_mean': float(np.mean(slls_taylor)),
            'taylor_worst': float(np.max(slls_taylor)),
            'socp_mean': float(np.mean(slls_socp)),
            'socp_worst': float(np.max(slls_socp)),
            'improve': float(improve),
            'time': float(np.mean(times)),
        })

    t_end = time.time()
    print(f"{'='*70}")
    print("CURVED ARRAY VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Curve':>6} {'Taylor':>8} {'SOCP':>8} {'Δ':>6} {'Worst_T':>8} {'Worst_S':>8} {'Time':>6}")
    for r in results:
        print(f"  α={r['curvature']:.2f} {r['taylor_mean']:>8.1f} {r['socp_mean']:>8.1f} "
              f"{r['improve']:>+6.1f} {r['taylor_worst']:>8.1f} {r['socp_worst']:>8.1f} "
              f"{r['time']:>5.1f}s")

    all_improve = np.mean([r['improve'] for r in results if r['curvature'] > 0])
    print(f"\n  Mean improve (curved only): {all_improve:+.2f} dB")
    if all_improve < -1.0:
        print("  → SOCP effective on curved arrays. Proceed to AI.")
    else:
        print("  → SOCP not effective. Stop curved array AI route.")

    with open(os.path.join(OUTPUT_DIR, 'curved_socp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Total: {t_end-t_total:.1f}s")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
