"""受限SOCP切平面验证：32×32, 5%失效, 5固定场景。

流程:
  1. 无补偿Taylor作为初始best_solution
  2. SOCP粗网格(15×15)求解
  3. 独立密集网格(101×101)验证, 找最高10~20副瓣点
  4. 加入约束, 重新求解
  5. 只有密集SLL优于best才更新
  6. 最多10轮, 或连续3轮<0.1dB, 或约束>1000
  7. 最终结果不得差于初始Taylor

决策标准: 改善≥0.5dB → 继续CNN; 否则停止失效补偿
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


def eval_dense(w_full, posx, posy, theta0, phi0, null_dirs, n_eval=101):
    """密集网格评估，返回 SLL, 指向误差, 零陷, 最差副瓣位置列表。"""
    k = 2 * np.pi
    px2d = np.tile(posx[:, None], (1, NY)).ravel()
    py2d = np.tile(posy[None, :], (NX, 1)).ravel()
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

    # 计算所有副瓣点的方向图
    pat_sl = np.zeros(len(u_flat))
    for i in range(len(u_flat)):
        psi = k * (px2d * u_flat[i] + py2d * v_flat[i])
        pat_sl[i] = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))

    # 主瓣响应(目标方向)
    psi_main = k * (px2d * u0 + py2d * v0)
    main_resp = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi_main)))

    if main_resp < 1e-10:
        return float('nan'), 0, [], []

    sll = 20 * np.log10(np.max(pat_sl) / (main_resp + 1e-30))

    # 指向误差：在全可见域找主瓣峰值
    vis_flat = vis.ravel()
    ug_flat = ug.ravel()
    vg_flat = vg.ravel()
    pat_all = np.zeros(len(ug_flat))
    for i in range(len(ug_flat)):
        if vis_flat[i]:
            psi = k * (px2d.ravel()[i] * ug_flat[i] + py2d.ravel()[i] * vg_flat[i])
            pat_all[i] = np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi)))
    peak_all_idx = np.argmax(pat_all)
    pt_err = angular_distance_deg(
        np.degrees(np.arcsin(np.clip(np.sqrt(ug_flat[peak_all_idx]**2 + vg_flat[peak_all_idx]**2), 0, 1))),
        np.degrees(np.arctan2(vg_flat[peak_all_idx], ug_flat[peak_all_idx])) % 360,
        theta0, phi0)

    # 零陷
    null_depths = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        psi_n = k * (px2d * un + py2d * vn)
        nd = 20 * np.log10(np.abs(np.sum(np.conj(w_full) * np.exp(1j * psi_n))) / (main_resp + 1e-30))
        null_depths.append(float(nd))

    # 最差10个副瓣点(用于切平面)
    sorted_idx = np.argsort(pat_sl)[::-1][:10]
    worst_points = [(u_flat[i], v_flat[i], pat_sl[i] / (main_resp + 1e-30))
                    for i in sorted_idx]

    return sll, pt_err, null_depths, worst_points


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)
    n_scenes = 5
    rate = 0.05

    print("="*70)
    print(f"Bounded SOCP Cutting Plane ({NX}×{NY}, {rate:.0%} failure)")
    print(f"Scenes: {n_scenes}, Max iters: 10, Add 10 pts/iter")
    print(f"Stop: 3 rounds <0.1dB, or >1000 constraints")
    print(f"Decision: improve ≥0.5dB → continue CNN")
    print("="*70)

    results = []

    for scene in range(n_scenes):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        null_dirs = [(30,(phi0+90)%360),(30,(phi0+180)%360),
                     (30,(phi0+270)%360),(min(theta0+25,85),(phi0+45)%360)]

        n_fail = int(NX * NY * rate)
        fmask = np.zeros(NX * NY, dtype=bool)
        fmask[rng.choice(NX * NY, n_fail, replace=False)] = True
        active_idx = np.where(~fmask)[0]

        # Taylor + LCMV 基线
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)
        al, pl = capon_nulling_2d(posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)

        # 无补偿 (Taylor + LCMV, 失效阵元置零)
        w_no = (al * np.exp(1j * pl)).ravel().copy()
        w_no[fmask] = 0  # 正确: 布尔mask索引

        sll_no, pt_no, nd_no, _ = eval_dense(w_no, posx, posy, theta0, phi0, null_dirs)

        # best-so-far 初始 = 无补偿
        best_sll = sll_no
        best_w = w_no.copy()

        # SOCP 参数
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        null_uv = [(np.sin(np.deg2rad(tn))*np.cos(np.deg2rad(pn)),
                    np.sin(np.deg2rad(tn))*np.sin(np.deg2rad(pn)))
                   for tn, pn in null_dirs]

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

        no_improve_rounds = 0
        t_start = time.time()

        for it in range(10):
            # 求解 SOCP
            w_opt = solve_socp(posx, posy, active_idx, u0, v0,
                               sl_u, sl_v, null_uv, eps_null=0.0316)
            if w_opt is None:
                break

            # 重建完整权值
            w_full = np.zeros(NX * NY, dtype=complex)
            w_full[active_idx] = w_opt

            # 密集网格评估
            sll_dense, pt_dense, nd_dense, worst_pts = eval_dense(
                w_full, posx, posy, theta0, phi0, null_dirs)

            # best-so-far
            if sll_dense < best_sll - 0.05:
                best_sll = sll_dense
                best_w = w_full.copy()
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            # 加入最差副瓣点
            for u_w, v_w, _ in worst_pts:
                already = False
                for su, sv in zip(sl_u, sl_v):
                    if abs(su - u_w) < 0.005 and abs(sv - v_w) < 0.005:
                        already = True
                        break
                if not already:
                    sl_u.append(u_w)
                    sl_v.append(v_w)

            # 停止条件
            if no_improve_rounds >= 3:
                break
            if len(sl_u) > 1000:
                break

        t_end = time.time()
        improve = best_sll - sll_no

        # 最终评估
        sll_final, pt_final, nd_final, _ = eval_dense(
            best_w, posx, posy, theta0, phi0, null_dirs)

        results.append({
            'scene': scene, 'theta0': theta0, 'phi0': phi0,
            'sll_no': sll_no, 'sll_socp': sll_final,
            'improve': improve, 'time': t_end - t_start,
            'n_constraints': len(sl_u),
            'pt_no': pt_no, 'pt_socp': pt_final,
            'null_no': max(nd_no), 'null_socp': max(nd_final),
        })

        status = "✓" if improve < -0.5 else ("=" if abs(improve) < 0.5 else "✗")
        print(f"  Scene {scene+1}: θ={theta0:.1f}° no={sll_no:.1f} "
              f"socp={sll_final:.1f} Δ={improve:+.1f}{status} "
              f"null_no={max(nd_no):.1f} null_socp={max(nd_final):.1f} "
              f"t={t_end-t_start:.1f}s constr={len(sl_u)}")

    # 汇总
    sll_no_arr = np.array([r['sll_no'] for r in results])
    sll_socp_arr = np.array([r['sll_socp'] for r in results])
    improve_arr = np.array([r['improve'] for r in results])

    print(f"\n{'='*70}")
    print("BOUNDED SOCP VERIFICATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':>15} {'No_comp':>8} {'SOCP':>8} {'Improve':>8}")
    print(f"  {'SLL mean':>15}: {np.mean(sll_no_arr):>8.1f} {np.mean(sll_socp_arr):>8.1f} "
          f"{np.mean(improve_arr):>+8.1f}")
    print(f"  {'SLL worst':>15}: {np.max(sll_no_arr):>8.1f} {np.max(sll_socp_arr):>8.1f} "
          f"{np.max(improve_arr):>+8.1f}")
    print(f"  {'Mean time':>15}: {np.mean([r['time'] for r in results]):>8.1f}s")

    print(f"\n  {'Scene':>5} {'θ0':>5} {'No':>8} {'SOCP':>8} {'Δ':>6} {'Null':>6} {'Time':>6}")
    for r in results:
        print(f"  {r['scene']+1:>5} {r['theta0']:>5.1f}° {r['sll_no']:>8.1f} "
              f"{r['sll_socp']:>8.1f} {r['improve']:>+6.1f} "
              f"{r['null_socp']:>6.1f} {r['time']:>5.1f}s")

    # 决策
    mean_improve = np.mean(improve_arr)
    print(f"\n  Decision: mean improve = {mean_improve:+.2f} dB")
    if mean_improve < -0.5:
        print("  → SOCP effective (≥0.5dB). Proceed to CNN teacher labels.")
    else:
        print("  → SOCP not effective (<0.5dB). Stop failure compensation.")
        print("  → Pivot to: non-uniform array coordinate → weight synthesis.")

    with open(os.path.join(OUTPUT_DIR, 'bounded_socp.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
