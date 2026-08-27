"""圆柱面阵列 SOCP 验证。

圆柱面: z = R * (1 - cos(x/R)), 沿 y 轴方向平直。
与抛物面 z = alpha*(x²+y²) 对比: 圆柱只在 x 方向弯曲。

R 值: 5, 8, 10, 15, 20
对应等效抛物面曲率: alpha ≈ 1/(2R)
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from run_curved_verify import (
    steering_vec_3d, coordinate_taylor_3d,
    solve_socp_3d, uv_to_uvw, eval_dense_3d,
)
from run_generate_teacher import normalize_weights
from run_multi_scan_generate import run_socp_cutting, get_null_dirs

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35

RADII = [5, 8, 10, 15, 20]
N_PER_RADIUS = 4
THETA0 = 30.0
PHI0 = 0.0


def generate_cylindrical_array(posx_ideal, posy_ideal, R, rng):
    """生成圆柱面阵列: z = R*(1-cos(x/R)), y 方向平直。"""
    px = np.tile(posx_ideal[:, None], (1, NY))
    py = np.tile(posy_ideal[None, :], (NX, 1))
    pz = R * (1.0 - np.cos(px / R))
    px = px + rng.uniform(-0.02, 0.02, (NX, NY))
    py = py + rng.uniform(-0.02, 0.02, (NX, NY))
    pz = R * (1.0 - np.cos(px / R))
    return px.ravel(), py.ravel(), pz.ravel()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    null_dirs = get_null_dirs(THETA0, PHI0)
    u0 = np.sin(np.deg2rad(THETA0)) * np.cos(np.deg2rad(PHI0))
    v0 = np.sin(np.deg2rad(THETA0)) * np.sin(np.deg2rad(PHI0))
    w0 = np.cos(np.deg2rad(THETA0))

    rng = np.random.RandomState(42)

    print("=" * 70)
    print(f"Cylindrical Array SOCP Verification ({NX}x{NY})")
    print(f"Radii: {RADII}, {N_PER_RADIUS} arrays each")
    print(f"Scan: theta={THETA0} phi={PHI0}")
    print("=" * 70)

    results = []
    t_total = time.time()

    for R in RADII:
        alpha_equiv = 1.0 / (2 * R)
        slls_taylor = []
        slls_socp = []
        times = []

        for i in range(N_PER_RADIUS):
            px, py, pz = generate_cylindrical_array(
                posx_ideal, posy_ideal, R, rng)

            w_taylor = coordinate_taylor_3d(
                px, py, pz, amp_x, amp_y, THETA0, PHI0)
            w_taylor_norm = normalize_weights(
                w_taylor, px, py, pz, u0, v0, w0)
            sll_t, _, _, _ = eval_dense_3d(
                w_taylor_norm, px, py, pz, THETA0, PHI0, null_dirs)
            slls_taylor.append(sll_t)

            t0 = time.time()
            w_socp, sll_s = run_socp_cutting(
                px, py, pz, u0, v0, w0, THETA0, PHI0, null_dirs,
                sll_t, w_taylor_norm, n_coarse=15, n_iters=5)
            dt = time.time() - t0
            times.append(dt)
            slls_socp.append(sll_s)

            if (i + 1) % 2 == 0:
                print(f"  R={R:4.0f} {i+1}/{N_PER_RADIUS}: "
                      f"taylor={sll_t:.1f} socp={sll_s:.1f} "
                      f"delta={sll_s-sll_t:+.1f} t={dt:.1f}s")

        slls_taylor = np.array(slls_taylor)
        slls_socp = np.array(slls_socp)
        improve = np.mean(slls_socp) - np.mean(slls_taylor)

        print(f"  R={R:4.0f} (alpha~{alpha_equiv:.3f}): "
              f"taylor={np.mean(slls_taylor):.1f} "
              f"socp={np.mean(slls_socp):.1f} "
              f"delta={improve:+.1f} "
              f"t={np.mean(times):.1f}s\n")

        results.append({
            'R': R,
            'alpha_equiv': float(alpha_equiv),
            'taylor_mean': float(np.mean(slls_taylor)),
            'taylor_worst': float(np.max(slls_taylor)),
            'socp_mean': float(np.mean(slls_socp)),
            'socp_worst': float(np.max(slls_socp)),
            'improve': float(improve),
            'time': float(np.mean(times)),
        })

    t_end = time.time()
    print(f"{'='*70}")
    print("CYLINDRICAL ARRAY VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'R':>6} {'alpha~':>8} {'Taylor':>8} {'SOCP':>8} "
          f"{'Delta':>6} {'Worst_T':>8} {'Worst_S':>8}")
    for r in results:
        print(f"  R={r['R']:>3.0f} {r['alpha_equiv']:>8.3f} "
              f"{r['taylor_mean']:>8.1f} {r['socp_mean']:>8.1f} "
              f"{r['improve']:>+6.1f} "
              f"{r['taylor_worst']:>8.1f} {r['socp_worst']:>8.1f}")

    all_improve = np.mean([r['improve'] for r in results])
    print(f"\n  Mean improve: {all_improve:+.2f} dB")
    if all_improve < -1.0:
        print("  → SOCP effective on cylindrical arrays.")
    else:
        print("  → SOCP not effective on cylindrical arrays.")

    print(f"\n  Total: {t_end - t_total:.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'cylindrical_socp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
