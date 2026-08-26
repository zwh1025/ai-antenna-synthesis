"""SOCP 失效补偿可行性验证。

直接最小化最大副瓣 (minimax)，而非 L2 范数。
这是判断"L2 方法失败是算法局限还是物理极限"的最干净方法。

SOCP 形式:
  minimize  t
  s.t.  a_main^H w = 1          (主瓣复响应)
        |a_sl^H w| ≤ t           (副瓣采样点)
        |a_null^H w| ≤ ε         (4个零陷)
        失效阵元 = 0
        可选 |w_n| ≤ 1           (幅度上限)

切平面迭代: 粗网格求解 → 密集网格验证 → 加入最差点 → 重新求解

四层实验:
  1. 无失效无零陷: 确认SOCP不破坏Taylor
  2. 有失效无零陷: 判断单纯失效能否补偿
  3. 有失效4零陷: 判断零陷约束代价
  4. 有失效零陷幅度上限: 硬件约束极限
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
from mylib.evaluation import evaluate_uv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')


def build_uv_points(n_uv=21):
    """均匀 uv 网格。"""
    u = np.linspace(-1, 1, n_uv)
    v = np.linspace(-1, 1, n_uv)
    ug, vg = np.meshgrid(u, v, indexing='ij')
    visible = (ug**2 + vg**2) <= 1.0
    return ug[visible], vg[visible]


def steering_vec(posx, posy, u, v, Nx, Ny, active_idx=None):
    """构建活跃阵元导向向量 (复数)。

    F(u,v) = sum_n conj(w_n) * exp(j*k*(x_n*u + y_n*v))
    所以 a_n = exp(j*k*(x_n*u + y_n*v)), F = a^H w (注意: cvxpy 中 a^H w)
    """
    k = 2 * np.pi
    px2d = np.tile(posx[:, None], (1, Ny))
    py2d = np.tile(posy[None, :], (Nx, 1))
    px = px2d.ravel()
    py = py2d.ravel()
    if active_idx is not None:
        px = px[active_idx]
        py = py[active_idx]
    return np.exp(1j * k * (px * u + py * v))


def solve_socp(posx, posy, Nx, Ny, active_idx,
               u_main, v_main, sl_u, sl_v,
               null_uv=None, eps_null=0.0316,  # -30dB
               amp_limit=None, w_init=None, lam_reg=0.0):
    """求解 SOCP。

    返回 w_opt (complex array, n_active)
    """
    n_active = len(active_idx)

    # 变量
    w = cp.Variable(n_active, complex=True)
    t = cp.Variable()

    # 主瓣约束: a_main^H w = 1
    a_main = steering_vec(posx, posy, u_main, v_main, Nx, Ny, active_idx)
    constraints = [cp.reshape(a_main.conj() @ w, (1,)) == 1.0]

    # 副瓣约束: |a_sl^H w| <= t
    for u_s, v_s in zip(sl_u, sl_v):
        a_sl = steering_vec(posx, posy, u_s, v_s, Nx, Ny, active_idx)
        constraints.append(cp.norm(a_sl.conj() @ w, 2) <= t)

    # 零陷约束
    if null_uv:
        for u_n, v_n in null_uv:
            a_n = steering_vec(posx, posy, u_n, v_n, Nx, Ny, active_idx)
            constraints.append(cp.norm(a_n.conj() @ w, 2) <= eps_null)

    # 幅度上限
    if amp_limit is not None:
        constraints.append(cp.norm(w, 2) <= amp_limit * np.sqrt(n_active))
        for i in range(n_active):
            constraints.append(cp.norm(w[i], 2) <= amp_limit)

    # 目标
    objective = cp.Minimize(t + lam_reg * cp.norm(w - w_init, 2)**2 if w_init is not None
                            else cp.Minimize(t))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, max_iters=5000, verbose=False)

    if prob.status != 'optimal':
        return None

    return w.value


def cutting_plane_socp(posx, posy, Nx, Ny, active_idx,
                       theta0, phi0, null_dirs=None,
                       n_coarse=21, n_fine=81,
                       n_iters=5, eps_null=0.0316,
                       amp_limit=None, w_init=None, lam_reg=0.0):
    """切平面迭代 SOCP。"""
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    # 副瓣区域排除
    bw = 0.886 * 2.0 / Nx * 180 / np.pi
    exc_deg = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    exc_uv = np.sin(np.deg2rad(exc_deg))

    # 粗网格副瓣采样
    sl_u, sl_v = build_uv_points(n_coarse)
    dist = np.sqrt((sl_u - u0)**2 + (sl_v - v0)**2)
    sl_mask = dist >= exc_uv
    sl_u = sl_u[sl_mask]
    sl_v = sl_v[sl_mask]

    # 零陷
    null_uv = None
    if null_dirs:
        null_uv = []
        for tn, pn in null_dirs:
            un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
            vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
            null_uv.append((un, vn))

    best_sll = float('inf')
    best_w = None

    for it in range(n_iters):
        w_opt = solve_socp(
            posx, posy, Nx, Ny, active_idx,
            u0, v0, sl_u, sl_v,
            null_uv=null_uv, eps_null=eps_null,
            amp_limit=amp_limit, w_init=w_init, lam_reg=lam_reg)

        if w_opt is None:
            break

        # 密集网格验证
        fine_u, fine_v = build_uv_points(n_fine)
        dist_fine = np.sqrt((fine_u - u0)**2 + (fine_v - v0)**2)
        sl_mask_fine = dist_fine >= exc_uv

        w_full = np.zeros(Nx * Ny, dtype=complex)
        w_full[active_idx] = w_opt

        k = 2 * np.pi
        px2d = np.tile(posx[:, None], (1, Ny)).ravel()
        py2d = np.tile(posy[None, :], (Nx, 1)).ravel()

        worst_sll = 0
        worst_u = 0
        worst_v = 0
        for u_f, v_f in zip(fine_u[sl_mask_fine], fine_v[sl_mask_fine]):
            psi = k * (px2d * u_f + py2d * v_f)
            mag = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))
            if mag > worst_sll:
                worst_sll = mag
                worst_u = u_f
                worst_v = v_f

        peak_main = np.abs(np.sum(np.conj(w_full) * np.exp(1j * k * (px2d * u0 + py2d * v0))))
        sll_db = 20 * np.log10(worst_sll / (peak_main + 1e-30))

        if sll_db < best_sll:
            best_sll = sll_db
            best_w = w_opt.copy()

        # 加入最差点到约束
        sl_u = np.append(sl_u, worst_u)
        sl_v = np.append(sl_v, worst_v)

    return best_w, best_sll


def evaluate_w(w_active, active_idx, posx, posy, Nx, Ny, theta0, phi0, null_dirs):
    """评估权值。"""
    w_full = np.zeros(Nx * Ny, dtype=complex)
    w_full[active_idx] = w_active

    amp = np.abs(w_full).reshape(Nx, Ny)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = np.angle(w_full).reshape(Nx, Ny) % (2 * np.pi)

    r = evaluate_uv(amp, phase, posx, posy, theta0, phi0, null_dirs, n_uv=101)

    # 主瓣增益
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    px2d = np.tile(posx[:, None], (1, Ny)).ravel()
    py2d = np.tile(posy[None, :], (Nx, 1)).ravel()
    main_gain = np.abs(np.sum(np.conj(w_full) * np.exp(1j * k * (px2d * u0 + py2d * v0))))

    # 幅度动态范围
    amp_active = np.abs(w_active)
    amp_papr = float(amp_active.max() / (np.mean(amp_active) + 1e-12))

    return {
        'sll': r['sll_3bw'],
        'pointing': r['pointing_err'],
        'main_gain': float(main_gain),
        'papr': amp_papr,
        'null_max': max(nr['max_3deg'] for nr in r['null_results']) if null_dirs else None,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 先 16×16 验证
    Nx = Ny = 16
    SLL = 35
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x, amp_y = taylor_2d_separable(Nx, Ny, SLL)

    print("="*70)
    print(f"SOCP Verification ({Nx}×{Ny} = {Nx*Ny} elements)")
    print("="*70)

    rng = np.random.RandomState(42)
    results = {}

    # 四层实验
    conditions = [
        ("no_fail_no_null", False, False, False),
        ("fail_no_null", True, False, False),
        ("fail_null", True, True, False),
        ("fail_null_amp", True, True, True),
    ]

    for cond_name, has_fail, has_null, has_amp in conditions:
        print(f"\n--- {cond_name} ---")
        rates = [0.05, 0.10, 0.20] if has_fail else [0.0]
        cond_results = []

        for rate in rates:
            slls_no = []; slls_socp = []; times = []; paprs = []
            n_tests = 10 if has_fail else 1

            for test_i in range(n_tests):
                theta0 = rng.uniform(0, 60)
                phi0 = rng.uniform(0, 360)

                # 失效
                if has_fail:
                    n_fail = int(Nx * Ny * rate)
                    fmask = np.zeros(Nx * Ny, dtype=bool)
                    fmask[rng.choice(Nx * Ny, n_fail, replace=False)] = True
                else:
                    fmask = np.zeros(Nx * Ny, dtype=bool)

                active_idx = np.where(~fmask)[0]

                # Taylor 基线
                px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
                amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)

                # 零陷
                null_dirs = None
                if has_null:
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

                    al, pl = capon_nulling_2d(
                        posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)
                    amp_ref, phase_ref = al, pl

                # 无补偿
                w_no = (amp_ref * np.exp(1j * phase_ref)).ravel()
                w_no[fmask] = 0
                w_no_active = w_no[active_idx]
                r_no = evaluate_w(w_no_active, active_idx, posx, posy, Nx, Ny,
                                  theta0, phi0, null_dirs)
                slls_no.append(r_no['sll'])

                # SOCP
                t_s = time.time()
                w_init_active = w_no[active_idx].copy()

                w_socp, sll_fast = cutting_plane_socp(
                    posx, posy, Nx, Ny, active_idx,
                    theta0, phi0, null_dirs=null_dirs,
                    n_coarse=21, n_fine=51, n_iters=3,
                    eps_null=0.0316,  # -30dB
                    amp_limit=1.0 if has_amp else None,
                    w_init=w_init_active, lam_reg=0.01)
                t_e = time.time()

                if w_socp is not None:
                    r_socp = evaluate_w(w_socp, active_idx, posx, posy, Nx, Ny,
                                       theta0, phi0, null_dirs)
                    slls_socp.append(r_socp['sll'])
                    paprs.append(r_socp['papr'])
                else:
                    slls_socp.append(float('nan'))
                    paprs.append(float('nan'))
                times.append(t_e - t_s)

            slls_no = np.array(slls_no)
            slls_socp = np.array(slls_socp)
            times = np.array(times)

            improve = np.nanmean(slls_socp) - np.nanmean(slls_no)
            rate_str = f"{int(rate*100)}%" if has_fail else "0%"
            print(f"  {rate_str:>4}: no_comp={np.nanmean(slls_no):.1f} "
                  f"SOCP={np.nanmean(slls_socp):.1f} Δ={improve:+.1f} "
                  f"time={np.nanmean(times):.1f}s papr={np.nanmean(paprs):.2f}")

            cond_results.append({
                'rate': float(rate),
                'sll_no_mean': float(np.nanmean(slls_no)),
                'sll_no_worst': float(np.nanmax(slls_no)),
                'sll_socp_mean': float(np.nanmean(slls_socp)),
                'sll_socp_worst': float(np.nanmax(slls_socp)),
                'improve': float(improve),
                'time_mean': float(np.nanmean(times)),
                'papr': float(np.nanmean(paprs)),
            })

        results[cond_name] = cond_results

    # 32×32 单场景验证
    print(f"\n--- 32×32 scale test ---")
    Nx32 = Ny32 = 32
    posx32 = uniform_linear_array_pos(Nx32)
    posy32 = uniform_linear_array_pos(Ny32)
    amp_x32, amp_y32 = taylor_2d_separable(Nx32, Ny32, SLL)

    for rate in [0.05, 0.10, 0.20]:
        theta0 = 30.0; phi0 = 0.0
        n_fail = int(Nx32 * Ny32 * rate)
        fmask = np.zeros(Nx32 * Ny32, dtype=bool)
        fmask[rng.choice(Nx32 * Ny32, n_fail, replace=False)] = True
        active_idx = np.where(~fmask)[0]

        px, py = beam_steering_phase_2d(posx32, posy32, theta0, phi0)
        amp_ref, phase_ref = combine_2d_excitation(amp_x32, amp_y32, px, py)
        null_dirs = [(30,90),(30,180),(30,270),(55,45)]
        al, pl = capon_nulling_2d(posx32, posy32, amp_ref, phase_ref, theta0, phi0, null_dirs)

        w_no = (al * np.exp(1j * pl)).ravel()
        w_no[fmask] = 0
        w_no_active = w_no[active_idx]
        r_no = evaluate_w(w_no_active, active_idx, posx32, posy32, Nx32, Ny32,
                          theta0, phi0, null_dirs)

        t_s = time.time()
        w_socp, _ = cutting_plane_socp(
            posx32, posy32, Nx32, Ny32, active_idx,
            theta0, phi0, null_dirs=null_dirs,
            n_coarse=11, n_fine=41, n_iters=3,
            eps_null=0.0316,
            w_init=w_no_active, lam_reg=0.01)
        t_e = time.time()

        if w_socp is not None:
            r_socp = evaluate_w(w_socp, active_idx, posx32, posy32, Nx32, Ny32,
                                theta0, phi0, null_dirs)
            print(f"  {int(rate*100):>2}%: no={r_no['sll']:.1f} SOCP={r_socp['sll']:.1f} "
                  f"Δ={r_socp['sll']-r_no['sll']:+.1f} t={t_e-t_s:.1f}s")
        else:
            print(f"  {int(rate*100):>2}%: SOCP failed")

    # 汇总
    print(f"\n{'='*70}")
    print("SOCP VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"{'Condition':>20} {'Rate':>5} {'No_comp':>8} {'SOCP':>8} {'Δ':>6} {'Time':>6}")
    for cond_name, cond_results in results.items():
        for r in cond_results:
            print(f"  {cond_name:>20} {int(r['rate']*100):>4}% "
                  f"{r['sll_no_mean']:>8.1f} {r['sll_socp_mean']:>8.1f} "
                  f"{r['improve']:>+6.1f} {r['time_mean']:>5.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'socp_verification.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
