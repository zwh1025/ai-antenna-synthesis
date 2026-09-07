"""Stage 0D v2：官方口径验收的密集 SOCP（修正网格间伪影）。

0D v1 诊断结论：solve_dense_socp 的切平面（31/41 网格加点 + independent
口径 best 选择）在扰动孔径上产生网格间副瓣伪影（官方 201 网格口径 -10dB，
independent 口径 -24dB），教师上限被伪影污染。

v2 修正（不改动 0A 冻结代码，复制改造）：
  1) recheck 加点网格 81 -> 201（与官方评估器同密度）
  2) best 选择改用官方评估器（evaluate_official_case, 201 网格）
  3) 起点 best = 坐标 Taylor 的官方口径 SLL

用法（服务器/本地）：
  python run_teacher_stage0d_v2.py            # 全部场景
  python run_teacher_stage0d_v2.py --tag fail20 --case regular_000   # 单场景

输出：outputs/teacher_stage0d_v2.json
"""

import os, sys, time, json, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('RAYON_NUM_THREADS', '1')

import cvxpy as cp
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from mylib.official_evaluator import evaluate_official_case
from run_curved_verify import uv_to_uvw, coordinate_taylor_3d
from run_stage4a_robustness_degradation import (
    apply_position_error, quantize_amplitude, quantize_phase,
)
import run_teacher_stage0d_pilot as v1

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'outputs', 'teacher_stage0d_v2.json')
RHO = 1.4
EPS_NULL_DB = -35
N_INIT = 31
ROUNDS = 25
GRID = 201


def official_sll(w_flat, case, posx_ax, posy_ax, lamb=1.0):
    amp = np.abs(w_flat).reshape(v1.NX, v1.NY)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = (np.angle(w_flat) % (2 * np.pi)).reshape(v1.NX, v1.NY)
    r = evaluate_official_case(amp, phase, posx_ax, posy_ax,
                               case['theta0_deg'], case['phi0_deg'],
                               null_dirs=case['null_dirs'], n_uv=GRID,
                               lamb=lamb)
    return float(r['sum']['sll_db'])


def _grid_uvw(n_grid):
    u = np.linspace(-1, 1, n_grid)
    ug, vg = np.meshgrid(u, u, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    return ug[vis], vg[vis], uv_to_uvw(ug[vis], vg[vis])


EXC_BW = 3.0   # 初始粗排除区（后续每轮由官方掩膜自动修正）


def compute_fixed_mask(w_ref, posx_ax, posy_ax):
    """固定掩膜：参考权值（A0 失效版）的官方第一零点包络格点布尔阵。"""
    from mylib.official_evaluator import _pattern_grid, _sum_main_lobe_mask
    amp = np.abs(w_ref).reshape(v1.NX, v1.NY)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = (np.angle(w_ref) % (2 * np.pi)).reshape(v1.NX, v1.NY)
    pattern_db, ug, vg, vis, _ = _pattern_grid(
        amp, phase, posx_ax, posy_ax, n_uv=GRID)
    mask, details, _ = _sum_main_lobe_mask(pattern_db, ug, vg, vis)
    return mask, details


def fixed_mask_recheck(sl_u, sl_v, sl_w, w_full, case, posx_ax, posy_ax,
                       fixed_mask, topk=30):
    """固定掩膜版约束集修正：掩膜外最坏点加入；掩膜内历史点剔除。
    掩膜固定（A0 基线包络）→ 凸问题无追逐。返回 (added, removed, sll_fixed)。"""
    from mylib.official_evaluator import _pattern_grid
    amp = np.abs(w_full).reshape(v1.NX, v1.NY)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = (np.angle(w_full) % (2 * np.pi)).reshape(v1.NX, v1.NY)
    pattern_db, ug, vg, vis, _ = _pattern_grid(
        amp, phase, posx_ax, posy_ax, n_uv=GRID)

    n = ug.shape[0]
    step = 2.0 / (n - 1)

    def in_mask(u, v):
        i = int(round((u + 1.0) / step))
        j = int(round((v + 1.0) / step))
        if not (0 <= i < n and 0 <= j < n):
            return False
        return bool(fixed_mask[i, j])

    keep = [k for k in range(len(sl_u)) if not in_mask(sl_u[k], sl_v[k])]
    removed = len(sl_u) - len(keep)
    sl_u[:] = [sl_u[k] for k in keep]
    sl_v[:] = [sl_v[k] for k in keep]
    sl_w[:] = [sl_w[k] for k in keep]

    side_db = np.where(vis & ~fixed_mask, pattern_db, -3000.0)
    order = np.argsort(side_db.ravel())[::-1][:topk]
    added = 0
    for idx in order:
        i, j = idx // n, idx % n
        uu, vv = float(ug[i, j]), float(vg[i, j])
        if not any(abs(su - uu) < 1e-9 and abs(sv - vv) < 1e-9
                   for su, sv in zip(sl_u, sl_v)):
            sl_u.append(uu)
            sl_v.append(vv)
            sl_w.append(float(uv_to_uvw(np.array([uu]), np.array([vv]))[0]))
            added += 1
    sll = float(np.max(side_db)) if np.any(vis & ~fixed_mask) else float('nan')
    return added, removed, sll


def _exclusion(theta0, u0, v0, n_grid, nx=32):
    ug, vg, wg = _grid_uvw(n_grid)
    bw = 0.886 * 2.0 / nx * 180 / np.pi
    exc = np.sin(np.deg2rad(EXC_BW * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = dist >= exc
    return list(ug[m]), list(vg[m]), list(wg[m])


def pattern_npy(w, px, py, pz, uf, vf, chunk=4096):
    """numpy 向量化方向图（幅度），分块防爆内存。"""
    out = np.empty(len(uf))
    for s in range(0, len(uf), chunk):
        e = min(s + chunk, len(uf))
        wf = np.sqrt(np.maximum(1.0 - uf[s:e] ** 2 - vf[s:e] ** 2, 0.0))
        ph = 2 * np.pi * (px[None, :] * uf[s:e, None]
                          + py[None, :] * vf[s:e, None]
                          + pz[None, :] * wf[:, None])
        out[s:e] = np.abs(np.exp(1j * ph) @ w)
    return out


def solve_socp_official(px, py, pz, case, w_taylor, null_dirs,
                        posx_ax, posy_ax, lamb=1.0, fail_mask=None,
                        w_ref=None,
                        rho=RHO, eps_db=EPS_NULL_DB,
                        n_init=N_INIT, rounds=ROUNDS):
    """固定掩膜口径的密集切平面 SOCP（全阵变量，失效元等式置零）。

    掩膜固定为 w_ref（A0 基线）的官方第一零点包络：凸问题、无追逐。
    best 同时记录固定掩膜口径（sll_fixed, 收敛指标）与官方自适应口径
    （official_sll, 报告口径）。"""
    fixed_mask, mask_details = compute_fixed_mask(
        w_ref if w_ref is not None else w_taylor, posx_ax, posy_ax)
    theta0, phi0 = case['theta0_deg'], case['phi0_deg']
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    k = 2 * np.pi
    eps_null = 10 ** (eps_db / 20)

    null_u = [np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p))
              for t, p in null_dirs]
    null_v = [np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p))
              for t, p in null_dirs]
    null_w = [np.cos(np.deg2rad(t)) for t, p in null_dirs]

    sl_u, sl_v, sl_w = _exclusion(theta0, u0, v0, n_init)

    def a_vec(u, v, w):
        return np.exp(1j * k * (px * u + py * v + pz * w))

    norm_bound = (rho * float(np.sqrt(np.sum(np.abs(w_taylor) ** 2)))
                  if rho is not None else None)

    best_w = w_taylor.copy()
    best_sll = official_sll(best_w, case, posx_ax, posy_ax, lamb)
    _, _, sll_fixed0 = fixed_mask_recheck(
        [], [], [], best_w, case, posx_ax, posy_ax, fixed_mask, topk=0)
    best_fixed = sll_fixed0
    log = [{'round': 0, 'event': 'start', 'official_sll': best_sll,
            'fixed_sll': sll_fixed0}]

    n = len(px)
    for it in range(rounds):
        wv = cp.Variable(n, complex=True)
        t = cp.Variable()
        cons = [a_vec(u0, v0, w0c).conj() @ wv == 1.0 + 0j]
        if fail_mask is not None and fail_mask.any():
            cons.append(wv[fail_mask] == 0.0 + 0j)
        for su, sv, sw in zip(sl_u, sl_v, sl_w):
            cons.append(cp.norm(a_vec(su, sv, sw).conj() @ wv, 2) <= t)
        for un, vn, wn_ in zip(null_u, null_v, null_w):
            cons.append(cp.norm(a_vec(un, vn, wn_).conj() @ wv, 2) <= eps_null)
        if norm_bound is not None:
            cons.append(cp.norm(wv, 2) <= norm_bound)
        prob = cp.Problem(cp.Minimize(t), cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception as e:
            log.append({'round': it + 1, 'event': 'exception', 'err': str(e)})
            break
        if prob.status not in ['optimal', 'optimal_inaccurate'] or \
                wv.value is None:
            log.append({'round': it + 1, 'event': 'bad_status',
                        'status': str(prob.status)})
            break
        w_cur = np.asarray(wv.value)
        resp = np.abs(np.sum(np.conj(a_vec(u0, v0, w0c)) * w_cur))
        if resp > 1e-12:
            w_cur = w_cur / resp
        sll_cur = official_sll(w_cur, case, posx_ax, posy_ax, lamb)
        added, removed, sll_fixed = fixed_mask_recheck(
            sl_u, sl_v, sl_w, w_cur, case, posx_ax, posy_ax, fixed_mask)
        if it >= 3 and added == 0 and removed == 0:
            log.append({'round': it + 1, 'event': 'converged'})
            break
        if sll_fixed < best_fixed - 0.01:
            best_fixed = sll_fixed
            best_w = w_cur.copy()
            best_sll = sll_cur
        log.append({'round': it + 1, 'event': 'solved',
                    't': float(t.value), 'official_sll': sll_cur,
                    'fixed_sll': sll_fixed, 'best_fixed': best_fixed,
                    'best_official': best_sll,
                    'added': added, 'removed': removed})
    return best_w, best_sll, log


# ---------------- 场景构建（复用 v1 逻辑） ----------------

def build_scenarios():
    manifest = json.load(open(os.path.join(
        v1.ROOT, 'results', 'stage2_strict_closure', 'baseline',
        'case_manifest.json'), encoding='utf-8'))
    by_id = {c['case_id']: c for c in manifest['regular'] + manifest['random']}
    cases = [by_id[i] for i in v1.PILOT_REGULAR + v1.PILOT_RANDOM]

    posx_ax = uniform_linear_array_pos(32)
    posy_ax = uniform_linear_array_pos(32)
    px = np.tile(posx_ax[:, None], (1, 32)).ravel()
    py = np.tile(posy_ax[None, :], (32, 1)).ravel()
    pz = np.zeros(1024)

    fj = json.load(open(os.path.join(v1.STAGE4A, 'failure_cases.json')))
    out = []
    for rate in v1.FAIL_RATES:
        for case in cases:
            r0 = [r for r in fj['realizations']
                  if r['case_id'] == case['case_id']
                  and abs(r['failure_rate'] - rate) < 1e-9
                  and r['seed'] == v1.SEED][0]
            mask = np.zeros(1024, dtype=bool)
            mask[np.array(r0['failed_indices'])] = True
            out.append((case, {'kind': 'failure', 'rate': rate, 'mask': mask,
                               'tag': f'fail{int(rate*100):02d}'}))
    for case in cases:
        ppx, ppy, _, _ = apply_position_error(posx_ax, posy_ax, v1.SEED)
        out.append((case, {'kind': 'position', 'px': ppx, 'py': ppy,
                           'tag': 'pos'}))
    return out, posx_ax, posy_ax, px, py, pz


def run_one(args):
    import torch
    torch.set_num_threads(1)
    (idx, cid, tag, theta0, phi0, null_dirs, px_full, py_full, pz_full,
     w_taylor_full, fail_mask, w_ref_full, posx_ax, posy_ax) = args
    case = _G['by_id'][cid]
    t0 = time.time()
    best_w, best_sll, log = solve_socp_official(
        px_full, py_full, pz_full, case, w_taylor_full, null_dirs,
        posx_ax, posy_ax, fail_mask=fail_mask, w_ref=w_ref_full)
    return {'idx': idx, 'cid': cid, 'tag': tag, 'a2_sll': best_sll,
            'a2_time_s': time.time() - t0, 'log': log}


_G = {}


def main():
    import multiprocessing as mp
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', type=str, default=None)
    ap.add_argument('--case', type=str, default=None)
    args = ap.parse_args()

    scenarios, posx_ax, posy_ax, px, py, pz = build_scenarios()
    if args.tag:
        scenarios = [s for s in scenarios if s[1]['tag'] == args.tag]
    if args.case:
        scenarios = [s for s in scenarios if s[0]['case_id'] == args.case]
    _G['by_id'] = {c['case_id']: c for c, _ in scenarios}
    print('Rebuilding A0 weights (mask reference) ...', flush=True)
    _G['a0_weights'] = v1.rebuild_stage2_weights(
        [c for c, _ in scenarios])
    print('  done', flush=True)

    tasks = []
    for i, (case, perturb) in enumerate(scenarios):
        cid = case['case_id']
        theta0, phi0 = case['theta0_deg'], case['phi0_deg']
        null_dirs = [tuple(nd) for nd in case['null_dirs']]
        if perturb['kind'] == 'position':
            pxs = perturb['px'].reshape(-1).copy()
            pys = perturb['py'].reshape(-1).copy()
            fail_mask = None
        else:
            pxs, pys = px, py
            fail_mask = perturb['mask']
        w_a1, w_t = v1.arm_a1(pxs, pys, pz, theta0, phi0, null_dirs,
                              np.ones(1024, dtype=bool) if fail_mask is None
                              else ~fail_mask)
        wr = _G['a0_weights'][cid]
        w_ref = (wr['amp'] * np.exp(1j * wr['phase'])).reshape(-1).copy()
        if fail_mask is not None:
            w_ref[fail_mask] = 0.0
        tasks.append((i, cid, perturb['tag'], theta0, phi0, null_dirs,
                      pxs, pys, pz, w_t, fail_mask, w_ref,
                      posx_ax, posy_ax))

    n_proc = min(len(tasks), max(1, (os.cpu_count() or 8) - 2))
    print(f'Stage0D v2: {len(tasks)} SOCP (official-caliber recheck+best, '
          f'rho={RHO}, eps={EPS_NULL_DB}dB, {ROUNDS} rounds) '
          f'on {n_proc} procs', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(n_proc) as pool:
        results = pool.map(run_one, tasks)
    print(f'done in {time.time()-t0:.0f}s', flush=True)

    payload = {'stage': 'Stage 0D v2 — official-caliber dense SOCP',
               'fix': 'recheck grid 81->201 (official density), '
                      'best selection by official evaluator',
               'config': {'rho': RHO, 'eps_null_db': EPS_NULL_DB,
                          'rounds': ROUNDS, 'init_grid': N_INIT,
                          'recheck_grid': GRID},
               'rows': results}
    for r in results:
        print(f"  {r['tag']:8s} {r['cid']:12s} A2_official={r['a2_sll']:7.2f} "
              f"({r['a2_time_s']:.0f}s)", flush=True)
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved: {OUTPUT}', flush=True)


if __name__ == '__main__':
    main()
