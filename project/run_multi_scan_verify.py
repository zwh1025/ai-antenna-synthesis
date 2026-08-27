"""提升切平面 + 多扫描方向验证。

对比旧 SOCP（15×15, 5轮）vs 新 SOCP（25×25, 10轮）在多个扫描方向下的表现。

扫描方向: theta=0/15/30/45/60度, phi=0/45/90/135/180/225/270/315度
零陷方向: 相对于主瓣的固定角偏移
  - 3个零陷: (theta0, phi0+90/180/270)
  - 1个零陷: (min(theta0+25, 85), phi0+45)

输出: 各方向下 Taylor vs SOCP_old vs SOCP_new 的 SLL 对比
"""

import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from run_curved_verify import (
    generate_curved_array, steering_vec_3d, coordinate_taylor_3d,
    solve_socp_3d, uv_to_uvw, eval_dense_3d,
)
from run_generate_teacher import normalize_weights

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35
ALPHA_TEST = [0.10, 0.15]
THETA_LIST = [0.0, 30.0, 60.0]


def get_null_dirs(theta0, phi0):
    """相对主瓣的零陷方向。"""
    return [
        (theta0, (phi0 + 90) % 360),
        (theta0, (phi0 + 180) % 360),
        (theta0, (phi0 + 270) % 360),
        (min(theta0 + 25, 85), (phi0 + 45) % 360),
    ]


def run_socp_cutting(px, py, pz, u0, v0, w0, theta0, phi0,
                     null_dirs, sll_taylor, w_taylor_norm,
                     n_coarse, n_iters):
    """SOCP 切平面迭代（参数化版本）。"""
    null_u = [np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
              for tn, pn in null_dirs]
    null_v = [np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
              for tn, pn in null_dirs]
    null_w = [np.cos(np.deg2rad(tn)) for tn, pn in null_dirs]

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

    best_sll = sll_taylor
    best_w = w_taylor_norm.copy()

    for it in range(n_iters):
        w_opt = solve_socp_3d(
            px, py, pz, u0, v0, w0,
            sl_u, sl_v, sl_w,
            (null_u, null_v), null_w)
        if w_opt is None:
            break
        w_opt = normalize_weights(w_opt, px, py, pz, u0, v0, w0)

        sll_s, pt_s, nd_s, worst_s = eval_dense_3d(
            w_opt, px, py, pz, theta0, phi0, null_dirs)

        if not np.isnan(sll_s) and sll_s < best_sll - 0.05:
            best_sll = sll_s
            best_w = w_opt.copy()

        for u_w, v_w, w_w in worst_s:
            already = any(abs(su - u_w) < 0.01 and abs(sv - v_w) < 0.01
                         for su, sv in zip(sl_u, sl_v))
            if not already:
                sl_u.append(u_w)
                sl_v.append(v_w)
                sl_w.append(w_w)

    return best_w, best_sll


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    rng = np.random.RandomState(42)

    print("=" * 75)
    print("Improved SOCP + Multi-Scan Direction Verification")
    print(f"  Curvatures: {ALPHA_TEST}")
    print(f"  Scan angles: theta={THETA_LIST}, phi=0")
    print(f"  Old SOCP: 15x15 grid, 5 iterations")
    print(f"  New SOCP: 25x25 grid, 10 iterations")
    print("=" * 75)

    results = []
    t_total = time.time()

    for theta0 in THETA_LIST:
        phi0 = 0.0
        null_dirs = get_null_dirs(theta0, phi0)
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0 = np.cos(np.deg2rad(theta0))

        for alpha in ALPHA_TEST:
            px, py, pz = generate_curved_array(
                posx_ideal, posy_ideal, alpha, rng)

            w_taylor = coordinate_taylor_3d(
                px, py, pz, amp_x, amp_y, theta0, phi0)
            w_taylor_norm = normalize_weights(
                w_taylor, px, py, pz, u0, v0, w0)
            sll_t, _, _, _ = eval_dense_3d(
                w_taylor_norm, px, py, pz, theta0, phi0, null_dirs)

            t0 = time.time()
            _, sll_old = run_socp_cutting(
                px, py, pz, u0, v0, w0, theta0, phi0, null_dirs,
                sll_t, w_taylor_norm, n_coarse=15, n_iters=5)
            t_old = time.time() - t0

            t0 = time.time()
            _, sll_new = run_socp_cutting(
                px, py, pz, u0, v0, w0, theta0, phi0, null_dirs,
                sll_t, w_taylor_norm, n_coarse=25, n_iters=10)
            t_new = time.time() - t0

            d_old = sll_old - sll_t
            d_new = sll_new - sll_t
            d_imp = sll_new - sll_old

            print(f"  theta={theta0:4.0f} alpha={alpha:.2f}: "
                  f"taylor={sll_t:.1f} old={sll_old:.1f}({d_old:+.1f}) "
                  f"new={sll_new:.1f}({d_new:+.1f}) "
                  f"new-old={d_imp:+.1f} "
                  f"t_old={t_old:.1f}s t_new={t_new:.1f}s")

            results.append({
                'theta0': theta0, 'phi0': phi0, 'alpha': alpha,
                'sll_taylor': float(sll_t),
                'sll_old': float(sll_old),
                'sll_new': float(sll_new),
                'delta_old': float(d_old),
                'delta_new': float(d_new),
                'improve_new_vs_old': float(d_imp),
                'time_old': float(t_old),
                'time_new': float(t_new),
            })

    t_end = time.time()
    print(f"\n  Total: {t_end - t_total:.1f}s")

    print(f"\n{'='*75}")
    print("SUMMARY: Old SOCP (15x15,5) vs New SOCP (25x25,10)")
    print(f"{'='*75}")
    print(f"{'Theta':>6} {'Alpha':>6} {'Taylor':>8} {'Old':>8} {'New':>8} "
          f"{'D_old':>7} {'D_new':>7} {'New-Old':>8} {'T_old':>6} {'T_new':>6}")
    for r in results:
        print(f"{r['theta0']:>5.0f} {r['alpha']:>6.2f} "
              f"{r['sll_taylor']:>8.1f} {r['sll_old']:>8.1f} {r['sll_new']:>8.1f} "
              f"{r['delta_old']:>+7.1f} {r['delta_new']:>+7.1f} "
              f"{r['improve_new_vs_old']:>+8.1f} "
              f"{r['time_old']:>5.1f}s {r['time_new']:>5.1f}s")

    old_deltas = [r['delta_old'] for r in results]
    new_deltas = [r['delta_new'] for r in results]
    print(f"\n  Mean delta: old={np.mean(old_deltas):+.2f} "
          f"new={np.mean(new_deltas):+.2f} "
          f"improvement={np.mean(new_deltas)-np.mean(old_deltas):+.2f} dB")
    print(f"  Mean time:  old={np.mean([r['time_old'] for r in results]):.1f}s "
          f"new={np.mean([r['time_new'] for r in results]):.1f}s "
          f"ratio={np.mean([r['time_new'] for r in results])/np.mean([r['time_old'] for r in results]):.1f}x")

    import json
    with open(os.path.join(OUTPUT_DIR, 'improved_socp_verify.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{'='*75}")


if __name__ == '__main__':
    main()
