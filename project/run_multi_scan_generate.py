"""多扫描方向 SOCP 教师标签生成。

在多个扫描方向下生成 SOCP 教师标签，训练 DeepSets 泛化到任意方向。

扫描方向: theta=0/15/30/45/60度, phi=0/45/90/.../315度 随机组合
SOCP: 15x15 粗网格, 5 轮切平面（与 v1 一致, ~23s/样本）
零陷方向: 相对主瓣的固定角偏移（theta<10°时用固定 30° 零陷）

输出: outputs/teacher_labels_v2.npz
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
N_TRAIN = 200
N_VAL = 30
N_TEST = 50
ALPHA_MIN = 0.08
ALPHA_MAX = 0.15

THETA_CHOICES = [0.0, 15.0, 30.0, 45.0, 60.0]
PHI_CHOICES = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]

N_CUTTING_ITERS = 5
N_COARSE_GRID = 15


def get_null_dirs(theta0, phi0):
    """相对主瓣的零陷方向。theta<10°时用固定 30° 零陷。"""
    if theta0 < 10:
        return [(30, 0), (30, 90), (30, 180), (30, 270)]
    return [
        (theta0, (phi0 + 90) % 360),
        (theta0, (phi0 + 180) % 360),
        (theta0, (phi0 + 270) % 360),
        (min(theta0 + 25, 85), (phi0 + 45) % 360),
    ]


def run_socp_cutting(px, py, pz, u0, v0, w0, theta0, phi0,
                     null_dirs, sll_taylor, w_taylor_norm):
    """SOCP 切平面迭代。"""
    null_u = [np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
              for tn, pn in null_dirs]
    null_v = [np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
              for tn, pn in null_dirs]
    null_w = [np.cos(np.deg2rad(tn)) for tn, pn in null_dirs]

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


def generate_one_sample(rng, posx_ideal, posy_ideal, amp_x, amp_y):
    """生成一个多扫描方向曲面阵列样本。"""
    alpha = rng.uniform(ALPHA_MIN, ALPHA_MAX)
    theta0 = rng.choice(THETA_CHOICES)
    phi0 = rng.choice(PHI_CHOICES)
    null_dirs = get_null_dirs(theta0, phi0)

    px, py, pz = generate_curved_array(posx_ideal, posy_ideal, alpha, rng)

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0 = np.cos(np.deg2rad(theta0))

    w_taylor = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    w_taylor_norm = normalize_weights(w_taylor, px, py, pz, u0, v0, w0)

    sll_t, _, _, _ = eval_dense_3d(
        w_taylor_norm, px, py, pz, theta0, phi0, null_dirs)

    w_socp, sll_s = run_socp_cutting(
        px, py, pz, u0, v0, w0, theta0, phi0, null_dirs, sll_t, w_taylor_norm)

    return {
        'px': px, 'py': py, 'pz': pz,
        'w_taylor_re': w_taylor_norm.real,
        'w_taylor_im': w_taylor_norm.imag,
        'w_socp_re': w_socp.real,
        'w_socp_im': w_socp.imag,
        'sll_taylor': float(sll_t),
        'sll_socp': float(sll_s),
        'alpha': float(alpha),
        'theta0': float(theta0), 'phi0': float(phi0),
        'u0': float(u0), 'v0': float(v0), 'w0': float(w0),
        'null_dirs': null_dirs,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx_ideal = uniform_linear_array_pos(NX)
    posy_ideal = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    n_total = N_TRAIN + N_VAL + N_TEST
    rng = np.random.RandomState(123)

    print("=" * 70)
    print(f"Multi-Scan Teacher Label Generation v2: {n_total} samples")
    print(f"  Train: {N_TRAIN}, Val: {N_VAL}, Test: {N_TEST}")
    print(f"  Alpha: [{ALPHA_MIN}, {ALPHA_MAX}]")
    print(f"  Theta: {THETA_CHOICES}")
    print(f"  Phi: {PHI_CHOICES}")
    print(f"  SOCP: {N_COARSE_GRID}x{N_COARSE_GRID} grid, {N_CUTTING_ITERS} iters")
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
            print(f"  [{i+1:3d}/{n_total}] th={sample['theta0']:.0f} "
                  f"ph={sample['phi0']:.0f} a={sample['alpha']:.3f} "
                  f"t={sample['sll_taylor']:.1f} s={sample['sll_socp']:.1f} "
                  f"d={sample['sll_socp']-sample['sll_taylor']:+.1f} "
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
        'version': 'v2_multi_scan',
    }
    for key in ['px', 'py', 'pz', 'w_taylor_re', 'w_taylor_im',
                'w_socp_re', 'w_socp_im']:
        save_dict[key] = np.array([s[key] for s in all_data])

    for key in ['sll_taylor', 'sll_socp', 'alpha',
                'theta0', 'phi0', 'u0', 'v0', 'w0']:
        save_dict[key] = np.array([s[key] for s in all_data])

    save_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v2.npz')
    np.savez(save_path, **save_dict)
    print(f"\nSaved: {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")

    # Summary by scan direction
    thetas = save_dict['theta0']
    print("\nSummary by scan direction:")
    for th in THETA_LIST if 'THETA_LIST' in dir() else THETA_CHOICES:
        mask = thetas == th
        n = np.sum(mask)
        if n > 0:
            dt = np.mean(save_dict['sll_taylor'][mask])
            ds = np.mean(save_dict['sll_socp'][mask])
            print(f"  theta={th:.0f}: n={n} taylor={dt:.1f} socp={ds:.1f} "
                  f"delta={ds-dt:+.1f}")

    for split_name, sl, el in [('Train', 0, N_TRAIN),
                                ('Val', N_TRAIN, N_TRAIN+N_VAL),
                                ('Test', N_TRAIN+N_VAL, n_total)]:
        t_m = np.mean(save_dict['sll_taylor'][sl:el])
        s_m = np.mean(save_dict['sll_socp'][sl:el])
        a_m = save_dict['alpha'][sl:el]
        print(f"  {split_name}: n={el-sl} alpha=[{a_m.min():.3f},{a_m.max():.3f}] "
              f"taylor={t_m:.1f} socp={s_m:.1f} delta={s_m-t_m:+.1f}")
    print("=" * 70)


if __name__ == '__main__':
    main()
