"""生成 SOCP 教师标签：曲面阵列坐标 + Taylor 基线 + SOCP 最优权值。

生成 200(训练) + 30(验证) + 50(测试) = 280 个曲面阵列样本，
每个样本包含：
  - 阵元坐标 (px, py, pz)
  - Taylor 基线权值（归一化到 a_main^H w = 1）
  - SOCP 切平面最优权值
  - SLL 指标
  - 扫描参数

曲率 α ~ Uniform(0.08, 0.15)，扫描方向 θ=30° φ=0° 固定。
复用 run_curved_verify.py 中的 SOCP 和评估函数。
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

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35
N_TRAIN = 200
N_VAL = 30
N_TEST = 50
ALPHA_MIN = 0.08
ALPHA_MAX = 0.15

THETA0 = 30.0
PHI0 = 0.0
NULL_DIRS = [(30, 90), (30, 180), (30, 270), (55, 45)]
N_CUTTING_ITERS = 5
N_COARSE_GRID = 15


def normalize_weights(w, px, py, pz, u0, v0, w0):
    """归一化权值使 a_main^H w = 1。"""
    a_main = steering_vec_3d(px, py, pz, u0, v0, w0)
    resp = np.conj(a_main) @ w
    if np.abs(resp) < 1e-12:
        return w
    return w / resp


def run_socp_cutting_plane(px, py, pz, u0, v0, w0, theta0, phi0,
                           sll_taylor, w_taylor_norm):
    """SOCP 切平面迭代，返回最优权值和 SLL。"""
    null_u = [np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
              for tn, pn in NULL_DIRS]
    null_v = [np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
              for tn, pn in NULL_DIRS]
    null_w = [np.cos(np.deg2rad(tn)) for tn, pn in NULL_DIRS]

    u_c = np.linspace(-1, 1, N_COARSE_GRID)
    v_c = np.linspace(-1, 1, N_COARSE_GRID)
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

    for it in range(N_CUTTING_ITERS):
        w_opt = solve_socp_3d(
            px, py, pz, u0, v0, w0,
            sl_u, sl_v, sl_w,
            (null_u, null_v), null_w)
        if w_opt is None:
            break
        w_opt = normalize_weights(w_opt, px, py, pz, u0, v0, w0)

        sll_s, pt_s, nd_s, worst_s = eval_dense_3d(
            w_opt, px, py, pz, theta0, phi0, NULL_DIRS)

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


def generate_one_sample(rng, posx_ideal, posy_ideal, amp_x, amp_y):
    """生成一个曲面阵列样本并求解 SOCP。"""
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)
    px, py, pz = generate_curved_array(posx_ideal, posy_ideal, alpha, rng)

    u0 = np.sin(np.deg2rad(THETA0)) * np.cos(np.deg2rad(PHI0))
    v0 = np.sin(np.deg2rad(THETA0)) * np.sin(np.deg2rad(PHI0))
    w0 = np.cos(np.deg2rad(THETA0))

    w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, THETA0, PHI0)
    w_taylor_norm = normalize_weights(w_taylor, px, py, pz, u0, v0, w0)

    sll_t, _, _, _ = eval_dense_3d(
        w_taylor_norm, px, py, pz, THETA0, PHI0, NULL_DIRS)

    w_socp, sll_s = run_socp_cutting_plane(
        px, py, pz, u0, v0, w0, THETA0, PHI0, sll_t, w_taylor_norm)

    return {
        'px': px, 'py': py, 'pz': pz,
        'w_taylor_re': w_taylor_norm.real,
        'w_taylor_im': w_taylor_norm.imag,
        'w_socp_re': w_socp.real,
        'w_socp_im': w_socp.imag,
        'sll_taylor': float(sll_t),
        'sll_socp': float(sll_s),
        'alpha': float(alpha),
        'theta0': THETA0, 'phi0': PHI0,
        'u0': float(u0), 'v0': float(v0), 'w0': float(w0),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    n_total = N_TRAIN + N_VAL + N_TEST
    rng = np.random.RandomState(42)

    print("=" * 70)
    print(f"Generating SOCP teacher labels: {n_total} samples")
    print(f"  Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")
    print(f"  Alpha: [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"  Scan: theta={THETA0} phi={PHI0}, {len(NULL_DIRS)} nulls")
    print(f"  SOCP: {N_CUTTING_ITERS} cutting plane iters")
    print("=" * 70)

    all_data = []
    t_start = time.time()

    for i in range(n_total):
        t0 = time.time()
        sample = generate_one_sample(rng, posx_ideal, posy_ideal, amp_x, amp_y)
        dt = time.time() - t0
        all_data.append(sample)

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t_start
            remaining = (n_total - i - 1) * (elapsed / (i + 1))
            print(f"  [{i+1:3d}/{n_total}] alpha={sample['alpha']:.3f} "
                  f"sll_t={sample['sll_taylor']:.1f} "
                  f"sll_s={sample['sll_socp']:.1f} "
                  f"delta={sample['sll_socp']-sample['sll_taylor']:+.1f} "
                  f"t={dt:.1f}s ETA={remaining:.0f}s")

    t_end = time.time()
    print(f"\nGeneration complete: {t_end - t_start:.1f}s "
          f"({(t_end - t_start)/n_total:.1f}s/sample)")

    split_idx = np.zeros(n_total, dtype=np.int32)
    split_idx[:N_TRAIN] = 0
    split_idx[N_TRAIN:N_TRAIN + N_VAL] = 1
    split_idx[N_TRAIN + N_VAL:] = 2

    save_dict = {
        'split': split_idx,
        'n_elements': NX * NY,
        'n_train': N_TRAIN,
        'n_val': N_VAL,
        'n_test': N_TEST,
    }
    for key in ['px', 'py', 'pz', 'w_taylor_re', 'w_taylor_im',
                'w_socp_re', 'w_socp_im']:
        save_dict[key] = np.array([s[key] for s in all_data])

    for key in ['sll_taylor', 'sll_socp', 'alpha',
                'theta0', 'phi0', 'u0', 'v0', 'w0']:
        save_dict[key] = np.array([s[key] for s in all_data])

    save_path = os.path.join(OUTPUT_DIR, 'teacher_labels.npz')
    np.savez(save_path, **save_dict)
    print(f"\nSaved: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    train_alphas = save_dict['alpha'][:N_TRAIN]
    val_alphas = save_dict['alpha'][N_TRAIN:N_TRAIN+N_VAL]
    test_alphas = save_dict['alpha'][N_TRAIN+N_VAL:]

    print("\nSummary:")
    print(f"  Train: alpha=[{train_alphas.min():.3f}, {train_alphas.max():.3f}] "
          f"SLL taylor={np.mean(save_dict['sll_taylor'][:N_TRAIN]):.1f} "
          f"socp={np.mean(save_dict['sll_socp'][:N_TRAIN]):.1f} "
          f"delta={np.mean(save_dict['sll_socp'][:N_TRAIN]-save_dict['sll_taylor'][:N_TRAIN]):+.1f}")
    print(f"  Val:   alpha=[{val_alphas.min():.3f}, {val_alphas.max():.3f}] "
          f"SLL taylor={np.mean(save_dict['sll_taylor'][N_TRAIN:N_TRAIN+N_VAL]):.1f} "
          f"socp={np.mean(save_dict['sll_socp'][N_TRAIN:N_TRAIN+N_VAL]):.1f} "
          f"delta={np.mean(save_dict['sll_socp'][N_TRAIN:N_TRAIN+N_VAL]-save_dict['sll_taylor'][N_TRAIN:N_TRAIN+N_VAL]):+.1f}")
    print(f"  Test:  alpha=[{test_alphas.min():.3f}, {test_alphas.max():.3f}] "
          f"SLL taylor={np.mean(save_dict['sll_taylor'][N_TRAIN+N_VAL:]):.1f} "
          f"socp={np.mean(save_dict['sll_socp'][N_TRAIN+N_VAL:]):.1f} "
          f"delta={np.mean(save_dict['sll_socp'][N_TRAIN+N_VAL:]-save_dict['sll_taylor'][N_TRAIN+N_VAL:]):+.1f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
