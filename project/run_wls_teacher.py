"""WLS/IRLS 教师优化：加权最小二乘拟合理想方向图。

核心思路:
  1. 理想方向图 d = A_full @ w_taylor（完整阵列）
  2. 失效后只优化活跃阵元: min ||Q(A_active @ w_active - d)||² + λ||w_active - w_init||²
  3. Q 对不同方向加权（主瓣低权、副瓣高权、最高副瓣最高权）
  4. IRLS: 每轮找出最高副瓣方向，加大权重，重新求解
  5. best-so-far 保护: 保存正式 SLL 最优的解，无改善返回初始解

约束:
  - 主瓣复响应 = 初始值
  - 4个零陷 = 0
  - 失效阵元 = 0
  - 幅度 ≤ 1
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    angular_distance_deg, calculate_2d_pattern, get_2d_sll,
)
from mylib.sum_diff import capon_nulling_2d
from mylib.evaluation import evaluate_uv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL_DESIGN = 35


def build_steering_matrix(posx, posy, u_list, v_list, lamb=1.0, active=None):
    """构建活跃阵元导向矩阵 A (n_points, n_active)。

    A[p, n] = exp(j * k * (x_n * u_p + y_n * v_p))
    F(u,v) = conj(w) · A = sum conj(w_n) * exp(j*k*(x_n*u + y_n*v))
    所以 A[p,n] = exp(j*k*(x_n*u_p + y_n*v_p))
    方向图: F = A @ conj(w_active)  (注意共轭)
    """
    k = 2 * np.pi / lamb
    posx_2d = np.tile(posx[:, None], (1, NY))
    posy_2d = np.tile(posy[None, :], (NX, 1))

    if active is not None:
        px = posx_2d.ravel()[active]
        py = posy_2d.ravel()[active]
    else:
        px = posx_2d.ravel()
        py = posy_2d.ravel()

    n_active = len(px)
    n_pts = len(u_list)
    A = np.zeros((n_pts, n_active), dtype=complex)
    for p in range(n_pts):
        A[p] = np.exp(1j * k * (px * u_list[p] + py * v_list[p]))
    return A


def wls_optimize(posx, posy, amp_lcmv, phase_lcmv, failure_mask,
                 theta0, phi0, null_dirs, n_uv=51, n_irls=10, lamb_reg=0.01):
    """WLS/IRLS 失效补偿。

    返回 (amp, phase, sll_history)
    """
    k = 2 * np.pi
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)

    active_flat = (~failure_mask).ravel()
    n_active = np.sum(active_flat)

    # 初始权值
    w_init = (amp_lcmv * np.exp(1j * phase_lcmv)).ravel()
    w_active = w_init[active_flat].copy()

    # 理想方向图（完整阵列 Taylor+LCMV）
    w_full = w_init.copy()  # 完整阵列权值

    # uv 网格
    u = np.linspace(-1, 1, n_uv)
    v = np.linspace(-1, 1, n_uv)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    visible = (ug**2 + vg**2) <= 1.0

    u_flat = ug.ravel()[visible.ravel()]
    v_flat = vg.ravel()[visible.ravel()]
    n_pts = len(u_flat)

    # 主瓣方向
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    # 排除区域
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc_deg = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    exc_uv = np.sin(np.deg2rad(exc_deg))
    dist = np.sqrt((u_flat - u0)**2 + (v_flat - v0)**2)
    is_sl = dist >= exc_uv

    # 导向矩阵
    A_full = build_steering_matrix(posx, posy, u_flat, v_flat)
    A_active = build_steering_matrix(posx, posy, u_flat, v_flat, active=active_flat)

    # 理想方向图（完整阵列）
    d_ideal = A_full @ np.conj(w_full)

    # 约束矩阵（主瓣 + 零陷）
    u_main = [u0]
    v_main = [v0]
    u_null = []
    v_null = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        u_null.append(un)
        v_null.append(un)

    # 约束导向向量（活跃阵元）
    A_constr = build_steering_matrix(
        posx, posy,
        u_main + [np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn)) for tn, pn in null_dirs],
        v_main + [np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn)) for tn, pn in null_dirs],
        active=active_flat)

    # 约束值: 主瓣 = 初始响应, 零陷 = 0
    main_resp = A_constr[0] @ np.conj(w_active)
    f_constr = np.zeros(len(A_constr), dtype=complex)
    f_constr[0] = main_resp  # 保持主瓣响应

    # 权重初始化: 主瓣低权, 副瓣单位权
    Q = np.ones(n_pts)
    Q[~is_sl] = 0.01  # 主瓣区域低权
    Q[is_sl] = 1.0    # 副瓣区域

    best_sll = float('inf')
    best_w = w_active.copy()
    sll_history = []

    for irls_iter in range(n_irls):
        # 加权最小二乘 + 约束
        # min ||Q(A @ conj(w) - d)||² + λ||w - w_init||²  s.t. C @ conj(w) = f
        # 令 z = conj(w), 则 w = conj(z)
        # min ||Q(A @ z - d)||² + λ||conj(z) - w_init||²  s.t. C @ z = f
        # min ||Q(A @ z - d)||² + λ||z - conj(w_init)||²  s.t. C @ z = f

        z_init = np.conj(w_active)

        # 构建加权系统
        Wq = np.diag(Q)
        Aw = Wq @ A_active  # (n_pts, n_active)
        dw = Wq @ d_ideal   # (n_pts,)

        # 正则化
        Reg = np.sqrt(lamb_reg) * np.eye(n_active, dtype=complex)
        Ar = np.vstack([Aw, Reg])
        dr = np.concatenate([dw, np.sqrt(lamb_reg) * z_init])

        # 约束: C @ z = f
        C = A_constr  # (n_constraints, n_active)
        f = f_constr

        # KKT 系统: [Ar^H Ar   C^H ] [z]   [Ar^H dr]
        #          [C       0   ] [λ] = [f      ]
        A_kkt = np.block([
            [Ar.conj().T @ Ar, C.conj().T],
            [C, np.zeros((C.shape[0], C.shape[0]), dtype=complex)]
        ])
        b_kkt = np.concatenate([Ar.conj().T @ dr, f])

        try:
            sol = np.linalg.lstsq(A_kkt, b_kkt, rcond=1e-10)[0]
            z_opt = sol[:n_active]
        except np.linalg.LinAlgError:
            break

        w_opt = np.conj(z_opt)

        # 幅度限制
        amp_opt = np.abs(w_opt)
        clip = amp_opt > 1.0
        if np.any(clip):
            w_opt[clip] = w_opt[clip] / amp_opt[clip]

        # 重建完整权值
        w_full_opt = np.zeros(NX * NY, dtype=complex)
        w_full_opt[active_flat] = w_opt

        # 评估
        amp = np.abs(w_full_opt).reshape(NX, NY)
        if amp.max() > 0:
            amp = amp / amp.max()
        phase = np.angle(w_full_opt).reshape(NX, NY) % (2 * np.pi)

        r = evaluate_uv(amp, phase, posx, posy, theta0, phi0,
                        null_dirs, n_uv=n_uv)
        sll = r['sll_3bw']
        sll_history.append(sll)

        if sll < best_sll:
            best_sll = sll
            best_w = w_opt.copy()

        # IRLS: 找最高副瓣方向，加大权重
        # 在 uv 网格上找最高副瓣
        pat = calculate_2d_pattern(amp.astype(np.float32), phase.astype(np.float32),
                                   posx, posy,
                                   np.linspace(0, 90, 91),
                                   np.linspace(0, 360, 181)).numpy()
        # 找最高副瓣点
        sl_max = -float('inf')
        sl_idx = -1
        for p in range(n_pts):
            if is_sl[p]:
                # 计算该点方向图
                u_p = u_flat[p]
                v_p = v_flat[p]
                psi = k * (posx[:, None] * u_p + posy[None, :] * v_p) - phase
                mag = np.abs(np.sum(amp * np.exp(1j * psi)))
                if mag > sl_max:
                    sl_max = mag
                    sl_idx = p

        if sl_idx >= 0:
            Q[sl_idx] *= 2.0  # 加大最高副瓣方向权重

    # 用 best-so-far
    w_final = np.zeros(NX * NY, dtype=complex)
    w_final[active_flat] = best_w

    amp = np.abs(w_final).reshape(NX, NY)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = np.angle(w_final).reshape(NX, NY) % (2 * np.pi)

    return amp, phase, sll_history


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)

    rng = np.random.RandomState(42)
    n_scenes = 10  # 先10个验证

    print("="*70)
    print(f"WLS/IRLS Teacher Optimization ({n_scenes} scenes)")
    print("="*70)

    results = []
    t0 = time.time()

    for i in range(n_scenes):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        rate = rng.choice([0.05, 0.10, 0.20])

        null_dirs = []
        for _ in range(4):
            for _ in range(20):
                tn = rng.uniform(10, 85)
                pn = rng.uniform(0, 360)
                if angular_distance_deg(tn, pn, theta0, phi0) >= 15:
                    null_dirs.append((float(tn), float(pn)))
                    break
        while len(null_dirs) < 4:
            null_dirs.append((85.0, float(rng.uniform(0, 360))))

        n_fail = int(NX * NY * rate)
        mask_flat = np.zeros(NX * NY, dtype=bool)
        mask_flat[rng.choice(NX * NY, n_fail, replace=False)] = True
        failure_mask = mask_flat.reshape(NX, NY)

        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)
        amp_lcmv, phase_lcmv = capon_nulling_2d(
            posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)

        # 无补偿
        amp_no = amp_lcmv * (~failure_mask)
        r_no = evaluate_uv(amp_no, phase_lcmv, posx, posy, theta0, phi0,
                            null_dirs, n_uv=101)

        # WLS/IRLS
        t_s = time.time()
        amp_opt, phase_opt, sll_hist = wls_optimize(
            posx, posy, amp_lcmv, phase_lcmv,
            failure_mask, theta0, phi0, null_dirs,
            n_uv=51, n_irls=10, lamb_reg=0.01)
        t_e = time.time()

        r_opt = evaluate_uv(amp_opt, phase_opt, posx, posy, theta0, phi0,
                            null_dirs, n_uv=101)

        sll_no = r_no['sll_3bw']
        sll_opt = r_opt['sll_3bw']
        null_no = max(nr['max_3deg'] for nr in r_no['null_results'])
        null_opt = max(nr['max_3deg'] for nr in r_opt['null_results'])
        improve = sll_opt - sll_no

        results.append({
            'i': i, 'rate': float(rate), 'theta0': theta0, 'phi0': phi0,
            'sll_no': sll_no, 'sll_opt': sll_opt, 'improve': improve,
            'null_no': null_no, 'null_opt': null_opt,
            'pt_no': r_no['pointing_err'], 'pt_opt': r_opt['pointing_err'],
            'time': t_e - t_s, 'sll_history': sll_hist,
        })

        status = "✓" if improve < -1 else "✗"
        print(f"  {i+1:2d}/{n_scenes}: rate={rate:.0%} θ={theta0:.1f}° "
              f"no={sll_no:.1f} opt={sll_opt:.1f} Δ={improve:+.1f}{status} "
              f"null_no={null_no:.1f} null_opt={null_opt:.1f} "
              f"t={t_e-t_s:.1f}s")

    t1 = time.time()
    sll_no_arr = np.array([r['sll_no'] for r in results])
    sll_opt_arr = np.array([r['sll_opt'] for r in results])
    improve_arr = np.array([r['improve'] for r in results])

    print(f"\n{'='*70}")
    print("WLS/IRLS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':>20} {'No comp':>8} {'WLS/IRLS':>10} {'Improve':>8}")
    print(f"  {'SLL mean':>20}: {np.mean(sll_no_arr):>8.1f} {np.mean(sll_opt_arr):>10.1f} "
          f"{np.mean(improve_arr):>+8.1f}")
    print(f"  {'SLL worst':>20}: {np.max(sll_no_arr):>8.1f} {np.max(sll_opt_arr):>10.1f} "
          f"{np.max(improve_arr):>+8.1f}")
    print(f"  {'Time':>20}: {t1-t0:.1f}s ({(t1-t0)/n_scenes:.1f}s/scene)")

    with open(os.path.join(OUTPUT_DIR, 'wls_validation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    if np.mean(improve_arr) < -1:
        print("\n  → WLS/IRLS effective! Proceed to CNN.")
    else:
        print("\n  → WLS/IRLS insufficient. Need diagnosis.")

    print(f"\n{'='*70}")

if __name__ == '__main__':
    main()
