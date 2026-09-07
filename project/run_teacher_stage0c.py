"""阶段 0C：深曲率教师极限探索——SOCP 配置矩阵实测。

问题：阶段 0A 的密集 SOCP（rho=1.4, 零陷-35）在深曲率仅 -21.9~-26.9 dB，
教师质量成为 AI 上限瓶颈。但该配置只测了一个点，且零陷用了比比赛要求
（<=-30 dBc）更严的 -35，范数约束只测了保守的 rho=1.4。

本脚本在 6 个深曲率难例上测试配置矩阵，回答"改进 SOCP 配置能否逼近 -35，
还是需要算法级创新，抑或物理不可行"：

  配置变体：
    A: rho=1.7, 零陷-30（比赛口径）   —— 范数适度放宽
    B: rho=2.2, 零陷-30               —— 范数大幅放宽
    C: 无范数约束, 零陷-30            —— 可行域上界（权值必然激进，只看上限）
    D: rho=1.7, 零陷-35               —— 分离零陷与范数的影响

  难例（阶段 0A 中最难的 6 个）：
    alpha=0.15 x theta{0,30,45,60} + alpha=0.10 x theta{45,60}

  优化提速：切平面迭代内用 81x81 快速评价（阶段 0A 每轮完整独立验收
  导致单场景 14~23 min），最终结果才做独立验收（201x201+错位+随机+
  峰值细化），单场景压至 ~3 min。

输出: outputs/teacher_stage0c.json
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from run_curved_verify import coordinate_taylor_3d, uv_to_uvw
from run_deepsets_train import _get_null_dirs
from run_scale_fix_v4 import _normalize_weights_torch
from run_teacher_stage0a import (
    independent_eval, _pattern_torch, _recheck_add,
)

try:
    import cvxpy as cp
except ImportError:
    cp = None

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
OUT_PATH = os.path.join(OUTPUT_DIR, 'teacher_stage0c.json')

NX = NY = 32
SLL_DESIGN = 35

CASES = [(0.15, 0.0, 0.0), (0.15, 30.0, 0.0), (0.15, 45.0, 45.0),
         (0.15, 60.0, 0.0), (0.10, 45.0, 45.0), (0.10, 60.0, 0.0)]

CONFIGS = [
    ('A_rho1.7_null30', 1.7, 10 ** (-30 / 20)),
    ('B_rho2.2_null30', 2.2, 10 ** (-30 / 20)),
    ('C_norho_null30', None, 10 ** (-30 / 20)),
    ('D_rho1.7_null35', 1.7, 10 ** (-35 / 20)),
]

N_INIT_GRID = 41
N_RECHECK_GRID = 81
N_CUT_ROUNDS = 15


def fast_sll_81(w, px, py, pz, theta0, u0, v0):
    """81x81 快速副瓣评价（切平面迭代内用）。"""
    n = 81
    u = np.linspace(-1, 1, n)
    ug, vg = np.meshgrid(u, u, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    wg = uv_to_uvw(ug, vg)
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = (dist >= exc) & vis
    pat = _pattern_torch(w, px, py, pz, ug[m], vg[m])
    main = _pattern_torch(w, px, py, pz,
                          np.array([u0]), np.array([v0]))[0]
    return 20 * np.log10(np.max(pat) / (main + 1e-30))


def solve_dense_socp_fast(px, py, pz, theta0, phi0, null_dirs, sll_taylor,
                          w_taylor_n, eps_null, rho):
    """密集切平面 SOCP：迭代内快速评价, 返回最优权值。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))

    null_u = [np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p))
              for t, p in null_dirs]
    null_v = [np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p))
              for t, p in null_dirs]
    null_w = [np.cos(np.deg2rad(t)) for t, p in null_dirs]

    u = np.linspace(-1, 1, N_INIT_GRID)
    ug, vg = np.meshgrid(u, u, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    wg = uv_to_uvw(ug, vg)
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = (dist >= exc) & vis
    sl_u, sl_v, sl_w = list(ug[m]), list(vg[m]), list(wg[m])

    def a_vec(a, b, c):
        return np.exp(1j * k * (px * a + py * b + pz * c))

    norm_bound = (rho * float(np.sqrt(np.sum(np.abs(w_taylor_n) ** 2)))
                  if rho is not None else None)

    best_w, best_sll = w_taylor_n.copy(), sll_taylor
    n = len(px)
    for it in range(N_CUT_ROUNDS):
        wv = cp.Variable(n, complex=True)
        t = cp.Variable()
        cons = [a_vec(u0, v0, w0c).conj() @ wv == 1.0 + 0j]
        for su, sv, sw in zip(sl_u, sl_v, sl_w):
            cons.append(cp.norm(a_vec(su, sv, sw).conj() @ wv, 2) <= t)
        for un, vn, wn_ in zip(null_u, null_v, null_w):
            cons.append(cp.norm(a_vec(un, vn, wn_).conj() @ wv, 2) <= eps_null)
        if norm_bound is not None:
            cons.append(cp.norm(wv, 2) <= norm_bound)
        prob = cp.Problem(cp.Minimize(t), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            break
        if prob.status not in ['optimal', 'optimal_inaccurate'] or \
                wv.value is None:
            break
        w_cur = np.asarray(wv.value)
        resp = np.abs(np.sum(np.conj(a_vec(u0, v0, w0c)) * w_cur))
        if resp > 1e-12:
            w_cur = w_cur / resp
        sll_fast = fast_sll_81(w_cur, px, py, pz, theta0, u0, v0)
        if sll_fast < best_sll - 0.02:
            best_sll = sll_fast
            best_w = w_cur.copy()
        _recheck_add(sl_u, sl_v, sl_w, w_cur, px, py, pz, theta0, u0, v0,
                     N_RECHECK_GRID)
    return best_w


def main():
    results = {}
    if os.path.exists(OUT_PATH):
        results = json.load(open(OUT_PATH, encoding='utf-8'))

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)

    print('=' * 80, flush=True)
    print('Stage 0C: deep-curvature teacher limit sweep '
          '(6 hard cases x 4 configs)', flush=True)
    print('=' * 80, flush=True)

    total = len(CASES) * len(CONFIGS)
    done = 0
    for ci, (alpha, theta0, phi0) in enumerate(CASES):
        cid = 'a%.2f_t%02d_p%03d' % (alpha, theta0, phi0)
        if cid in results and len(results[cid]) >= len(CONFIGS) + 1:
            print(f'[{ci+1}/{len(CASES)}] {cid} done, skip', flush=True)
            continue
        rng = np.random.RandomState(1000 + ci)
        px = np.tile(posx[:, None], (1, NY))
        py = np.tile(posy[None, :], (NX, 1))
        px = (px + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        py = (py + rng.uniform(-0.02, 0.02, (NX, NY))).ravel()
        pz = alpha * (px ** 2 + py ** 2)
        null_dirs = _get_null_dirs(theta0, phi0)

        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0c = np.cos(np.deg2rad(theta0))
        w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
        w_tn = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0c)
        r_t = independent_eval(w_tn, px, py, pz, theta0, phi0, null_dirs)

        case = {'alpha': alpha, 'theta0': theta0, 'phi0': phi0,
                'taylor': r_t['sll_db']}
        print(f'\n[{ci+1}/{len(CASES)}] alpha={alpha} theta={theta0} '
              f'Taylor={r_t["sll_db"]:.2f}', flush=True)
        results[cid] = case

        for name, rho, eps in CONFIGS:
            done += 1
            t0 = time.time()
            w_best = solve_dense_socp_fast(px, py, pz, theta0, phi0,
                                           null_dirs, r_t['sll_db'], w_tn,
                                           eps, rho)
            r = independent_eval(w_best, px, py, pz, theta0, phi0, null_dirs,
                                 w_taylor=w_tn)
            case[name] = {
                'sll_db': r['sll_db'],
                'worst_null_neighborhood_db': r['worst_null_neighborhood_db'],
                'norm_ratio': r.get('norm_ratio_vs_taylor'),
                'amp_dynamic_range_db': r['amp_dynamic_range_db'],
                'time_s': time.time() - t0,
            }
            print(f'  {name}: SLL={r["sll_db"]:.2f} '
                  f'null={r["worst_null_neighborhood_db"]:.1f} '
                  f'norm={r.get("norm_ratio_vs_taylor", 0):.2f} '
                  f'dyn={r["amp_dynamic_range_db"]:.0f}dB '
                  f'({time.time()-t0:.0f}s)', flush=True)
            results[cid] = case
            json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'),
                      indent=1, ensure_ascii=False)

    # ---------- 汇总 ----------
    print('\n' + '=' * 80)
    print('汇总（独立验收器口径；对照: 阶段0A rho=1.4/零陷-35）')
    print('=' * 80)
    hdr = '%-16s %7s' % ('case', 'taylor')
    for name, _, _ in CONFIGS:
        hdr += ' %9s' % name[:9]
    print(hdr)
    for cid, c in results.items():
        if 'taylor' not in c:
            continue
        row = '%-16s %7.2f' % (cid, c['taylor'])
        for name, _, _ in CONFIGS:
            v = c.get(name)
            row += ' %9s' % ('%.1f' % v['sll_db'] if v else '—')
        print(row)
    print('\n范数比 (norm/norm_taylor):')
    for cid, c in results.items():
        if 'taylor' not in c:
            continue
        row = '%-16s' % cid
        for name, _, _ in CONFIGS:
            v = c.get(name)
            row += ' %9s' % ('%.2f' % v['norm_ratio'] if v else '—')
        print(row)
    print('\n零陷邻域最坏 (dB):')
    for cid, c in results.items():
        if 'taylor' not in c:
            continue
        row = '%-16s' % cid
        for name, _, _ in CONFIGS:
            v = c.get(name)
            row += ' %9s' % ('%.0f' % v['worst_null_neighborhood_db']
                             if v else '—')
        print(row)
    print(f'\nSaved: {OUT_PATH}')


if __name__ == '__main__':
    main()
