"""教师求解器验真（阶段 0A，按修正后的方案执行）。

目的：在批量生成任何教师标签之前，先用 12 个代表算例回答——
  密集切平面 SOCP 的教师质量上限到底在哪里？权值是否失控？
  AP 的 -34.97 dB 能否在独立网格上复现？

方法矩阵（同坐标、同种子、同独立验收器）：
  taylor     : 坐标 Taylor 基线
  weak_socp  : 现有教师配方（15×15 初始、5 轮、零陷-30、无范数约束）
  ap         : 交替投影（81×81 设计网格、零陷掩模-40）
  dense_socp : 密集切平面（41×41 初始、15 轮、复查 81×81、零陷-35、
               范数约束 ||w||2 <= rho*||w_taylor||2, rho=1.4）
  dense_warm : 同 dense_socp 但以 AP 解为切平面初始 best（仅前 4 算例，
               只比较耗时——凸问题热启动不应改变最终质量，若质量不同
               说明求解器未充分收敛）

独立验收器（GPT 修正要求，不再用设计网格自评）：
  - 201×201 可见域网格 + 半格距错位网格 + 2000 随机角度
  - 副瓣候选局部峰值细化（11×11）
  - 零陷中心连续响应 + ±1.5° 球冠邻域最坏值
  - 统一主瓣归一化；记录权值范数/最大幅度/动态范围
  - 最差零陷 = max()（修复 run_ap_compare 的 min() bug）

另生成 8×8 HFSS 几何上的强 SOCP 权值（4 方向，秒级），供后续 EEP
重组止损检查使用（阶段 0B，不在本脚本内做重组）。

输出: outputs/teacher_stage0a.json（逐算例增量落盘）
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from run_curved_verify import coordinate_taylor_3d, solve_socp_3d, uv_to_uvw
from run_deepsets_train import _get_null_dirs

try:
    import cvxpy as cp
except ImportError:
    cp = None

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
OUT_PATH = os.path.join(OUTPUT_DIR, 'teacher_stage0a.json')

NX = NY = 32
SLL_DESIGN = 35
LAMB = 1.0

# 12 算例: alpha x theta, phi 轴向/对角混合, 固定种子
CASES = []
for alpha in [0.05, 0.10, 0.15]:
    for theta, phi in [(0.0, 0.0), (30.0, 0.0), (45.0, 45.0), (60.0, 0.0)]:
        CASES.append((alpha, theta, phi))

RHO = 1.4                  # 范数约束倍数
EPS_NULL_DENSE = 10 ** (-35 / 20)   # 密集 SOCP 零陷 -35 dB
N_INIT_GRID = 41
N_RECHECK_GRID = 81
N_CUT_ROUNDS = 15
WARM_CASES = 4


# ---------------- 独立验收器 ----------------

def _pattern_torch(w, px, py, pz, u_arr, v_arr):
    """连续阵因子 |F| 在任意 (u,v) 采样点上的幅度（torch 分块）。
    w = sqrt(1-u^2-v^2)（上半空间可见域约定）。"""
    k = 2 * np.pi
    w_arr = np.sqrt(np.maximum(1.0 - u_arr ** 2 - v_arr ** 2, 0.0))
    pos_t = torch.as_tensor(np.stack([px, py, pz], axis=1))
    wt = torch.as_tensor(np.conj(w), dtype=torch.complex128)
    out = np.empty(len(u_arr))
    CH = 4096
    for s in range(0, len(u_arr), CH):
        e = min(s + CH, len(u_arr))
        uvw_t = torch.as_tensor(np.stack(
            [u_arr[s:e], v_arr[s:e], w_arr[s:e]], axis=1))
        psi = k * (uvw_t @ pos_t.T)
        out[s:e] = (torch.exp(1j * psi) @ wt).abs().numpy()
    return out


def independent_eval(w, px, py, pz, theta0, phi0, null_dirs,
                     w_taylor=None, nx=NX):
    """独立验收: 错位网格+随机采样+峰值细化; 主瓣连续归一化。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))

    # 主瓣响应(连续)归一化
    a0 = np.exp(1j * k * (px * u0 + py * v0 + pz * w0c))
    resp = np.abs(np.sum(np.conj(a0) * w))
    if resp < 1e-12:
        return {'error': 'zero main lobe'}
    wn = w / resp

    # 采样点: 201x201 + 错位 + 随机
    n = 201
    axis = np.linspace(-1, 1, n)
    step = axis[1] - axis[0]
    ug, vg = np.meshgrid(axis, axis, indexing='ij')
    axis2 = axis + step / 2
    axis2 = np.clip(axis2, -1, 1)
    ug2, vg2 = np.meshgrid(axis2, axis2, indexing='ij')
    rng = np.random.RandomState(2026)
    ur = rng.uniform(-1, 1, 2000)
    vr = rng.uniform(-1, 1, 2000)
    keep = ur ** 2 + vr ** 2 <= 1.0
    ur, vr = ur[keep], vr[keep]
    U = np.concatenate([ug.ravel(), ug2.ravel(), ur])
    V = np.concatenate([vg.ravel(), vg2.ravel(), vr])
    vis = U ** 2 + V ** 2 <= 1.0
    U, V = U[vis], V[vis]
    pat = _pattern_torch(wn, px, py, pz, U, V) / np.abs(
        np.sum(np.conj(a0) * wn))

    # 3x3dB_BW 排除（与项目主口径一致, nx 阵元口径）
    bw = 0.886 * 2.0 / nx * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((U - u0) ** 2 + (V - v0) ** 2)
    sl = dist >= exc
    sll_db = 20 * np.log10(np.max(pat[sl]) + 1e-30)

    # 峰值细化: top-10 副瓣候选, 11x11 局部网格
    idx = np.argsort(pat[sl])[::-1][:10]
    sl_u, sl_v = U[sl], V[sl]
    best = np.max(pat[sl])
    for i in idx:
        cu, cv = sl_u[i], sl_v[i]
        du = np.linspace(-step, step, 11)
        gu, gv = np.meshgrid(du, du)
        uu = np.clip(cu + gu, -1, 1).ravel()
        vv = np.clip(cv + gv, -1, 1).ravel()
        ok = uu ** 2 + vv ** 2 <= 1.0
        uu, vv = uu[ok], vv[ok]
        dd = np.sqrt((uu - u0) ** 2 + (vv - v0) ** 2)
        m = dd >= exc
        if np.any(m):
            pv = np.max(_pattern_torch(wn, px, py, pz, uu[m], vv[m]))
            best = max(best, pv)
    sll_db = 20 * np.log10(best + 1e-30)

    # 指向: 全采样峰值
    ip = np.argmax(pat)
    pu, pv = U[ip], V[ip]
    pt = np.degrees(np.arcsin(np.clip(np.sqrt(pu ** 2 + pv ** 2), 0, 1)))
    pp = np.degrees(np.arctan2(pv, pu)) % 360
    from mylib.antenna_calc import angular_distance_deg
    pt_err = angular_distance_deg(pt, pp, theta0, phi0)

    # 零陷: 中心连续 + ±1.5° 邻域最坏
    nulls = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        wn_dirc = np.cos(np.deg2rad(tn))
        center = 20 * np.log10(np.abs(np.sum(np.conj(
            np.exp(1j * k * (px * un + py * vn + pz * wn_dirc))) * wn)) + 1e-30)
        dth = np.deg2rad(np.linspace(-1.5, 1.5, 13))
        dph = np.deg2rad(np.linspace(-1.5, 1.5, 13))
        gu, gv = np.meshgrid(dth, dph)
        tu, tv = np.deg2rad(tn) + gu.ravel(), np.deg2rad(pn) + gv.ravel()
        us = np.sin(tu) * np.cos(tv)
        vs = np.sin(tu) * np.sin(tv)
        ok = us ** 2 + vs ** 2 <= 1.0
        vals = _pattern_torch(wn, px, py, pz, us[ok], vs[ok])
        worst_n = 20 * np.log10(np.max(vals) + 1e-30)
        nulls.append({'center_db': float(center),
                      'neighborhood_worst_db': float(worst_n)})

    # 权值度量（主瓣归一化下）
    amp = np.abs(wn)
    nz = amp[amp > 1e-15]
    out = {
        'sll_db': float(sll_db),
        'pointing_err_deg': float(pt_err),
        'worst_null_center_db': float(max(n['center_db'] for n in nulls)),
        'worst_null_neighborhood_db': float(
            max(n['neighborhood_worst_db'] for n in nulls)),
        'nulls': nulls,
        'w_norm2': float(np.sqrt(np.sum(amp ** 2))),
        'w_max_amp': float(amp.max()),
        'w_min_nonzero_amp': float(nz.min()) if len(nz) else 0.0,
        'amp_dynamic_range_db': float(20 * np.log10(amp.max() / nz.min()))
        if len(nz) else None,
    }
    if w_taylor is not None:
        at = np.abs(w_taylor)
        out['delta_w_ratio'] = float(
            np.sqrt(np.sum(np.abs(wn - w_taylor) ** 2)) /
            np.sqrt(np.sum(at ** 2)))
        out['norm_ratio_vs_taylor'] = out['w_norm2'] / float(
            np.sqrt(np.sum(at ** 2)))
    return out


# ---------------- 求解器 ----------------

def _exclusion_uv(theta0, u0, v0, n_grid, nx=NX):
    u = np.linspace(-1, 1, n_grid)
    ug, vg = np.meshgrid(u, u, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    wg = uv_to_uvw(ug, vg)
    bw = 0.886 * 2.0 / nx * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = (dist >= exc) & vis
    return list(ug[m]), list(vg[m]), list(wg[m])


def _recheck_add(sl_u, sl_v, sl_w, w, px, py, pz, theta0, u0, v0, n_grid,
                 topk=20):
    """在 n_grid 密集网格复查, 把最坏副瓣点加入约束集（torch 向量化）。"""
    u = np.linspace(-1, 1, n_grid)
    ug, vg = np.meshgrid(u, u, indexing='ij')
    vis = (ug ** 2 + vg ** 2) <= 1.0
    wg = uv_to_uvw(ug, vg)
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = np.sin(np.deg2rad(3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)))
    dist = np.sqrt((ug - u0) ** 2 + (vg - v0) ** 2)
    m = (dist >= exc) & vis
    uf, vf, wf = ug[m], vg[m], wg[m]
    pat = _pattern_torch(w, px, py, pz, uf, vf)
    order = np.argsort(pat)[::-1][:topk]
    added = 0
    for j in order:
        if not any(abs(su - uf[j]) < 0.005 and abs(sv - vf[j]) < 0.005
                   for su, sv in zip(sl_u, sl_v)):
            sl_u.append(float(uf[j]))
            sl_v.append(float(vf[j]))
            sl_w.append(float(wf[j]))
            added += 1
    return added, float(np.max(pat))


def solve_dense_socp(px, py, pz, theta0, phi0, null_dirs, sll_taylor,
                     w_taylor_n, eps_null=EPS_NULL_DENSE, rho=RHO,
                     n_init=N_INIT_GRID, rounds=N_CUT_ROUNDS, w_init=None):
    """密集切平面 SOCP: 主瓣=1, 副瓣<=t, 零陷<=eps, 可选范数约束。
    w_init: AP 解(热启动)——只作为初始 best 与初始越界点来源。"""
    k = 2 * np.pi
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))

    null_u = [np.sin(np.deg2rad(t)) * np.cos(np.deg2rad(p))
              for t, p in null_dirs]
    null_v = [np.sin(np.deg2rad(t)) * np.sin(np.deg2rad(p))
              for t, p in null_dirs]
    null_w = [np.cos(np.deg2rad(t)) for t, p in null_dirs]

    sl_u, sl_v, sl_w = _exclusion_uv(theta0, u0, v0, n_init)

    def a_vec(u, v, w):
        return np.exp(1j * k * (px * u + py * v + pz * w))

    def evaluate(wv):
        r = independent_eval(wv, px, py, pz, theta0, phi0, null_dirs)
        return r

    norm_bound = (rho * float(np.sqrt(np.sum(np.abs(w_taylor_n) ** 2)))
                  if rho is not None else None)

    if w_init is not None:
        best_w = w_init.copy()
        r0 = evaluate(best_w)
        best_sll = r0['sll_db']
        _recheck_add(sl_u, sl_v, sl_w, best_w, px, py, pz, theta0, u0, v0,
                     N_RECHECK_GRID, topk=40)
    else:
        best_w = w_taylor_n.copy()
        best_sll = sll_taylor

    n = len(px)
    for it in range(rounds):
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
        r = evaluate(w_cur)
        if r['sll_db'] < best_sll - 0.02:
            best_sll = r['sll_db']
            best_w = w_cur.copy()
        _recheck_add(sl_u, sl_v, sl_w, w_cur, px, py, pz, theta0, u0, v0,
                     N_RECHECK_GRID)
    return best_w, best_sll


def run_ap(px, py, pz, theta0, phi0, null_dirs, w_init):
    from run_ap_compare import run_alternating_projection
    w, sll_design, iters, conv = run_alternating_projection(
        px, py, pz, theta0, phi0, null_dirs, w_init)
    return w, sll_design, iters, conv


def run_weak_socp(px, py, pz, theta0, phi0, null_dirs, sll_taylor, w_taylor_n):
    from run_multi_scan_generate import run_socp_cutting
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    w, sll = run_socp_cutting(px, py, pz, u0, v0, w0c, theta0, phi0,
                              null_dirs, sll_taylor, w_taylor_n)
    return w, sll


# ---------------- 主流程 ----------------

def main():
    results = {}
    if os.path.exists(OUT_PATH):
        results = json.load(open(OUT_PATH, encoding='utf-8'))

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)

    print('=' * 80, flush=True)
    print('Stage 0A: teacher solver verification (12 cases, independent eval)',
          flush=True)
    print('=' * 80, flush=True)

    for ci, (alpha, theta0, phi0) in enumerate(CASES):
        cid = 'a%.2f_t%02d_p%03d' % (alpha, theta0, phi0)
        if cid in results and 'dense_socp' in results[cid]:
            print(f'[{ci+1}/12] {cid} 已完成, 跳过', flush=True)
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
        k = 2 * np.pi
        a0 = np.exp(1j * k * (px * u0 + py * v0 + pz * w0c))
        w_tn = w_t / np.sum(np.conj(a0) * w_t)
        r_t = independent_eval(w_tn, px, py, pz, theta0, phi0, null_dirs)

        case = {'alpha': alpha, 'theta0': theta0, 'phi0': phi0,
                'taylor': r_t}
        print(f'\n[{ci+1}/12] alpha={alpha} theta={theta0} phi={phi0}',
              flush=True)
        print(f'  taylor: SLL={r_t["sll_db"]:.2f}', flush=True)

        # weak SOCP
        t0 = time.time()
        w_weak, _ = run_weak_socp(px, py, pz, theta0, phi0, null_dirs,
                                  r_t['sll_db'], w_tn)
        t_weak = time.time() - t0
        r_weak = independent_eval(w_weak, px, py, pz, theta0, phi0, null_dirs,
                                  w_taylor=w_tn)
        case['weak_socp'] = {**r_weak, 'time_s': t_weak}
        print(f'  weak_socp: SLL={r_weak["sll_db"]:.2f} '
              f'null_worst={r_weak["worst_null_neighborhood_db"]:.1f} '
              f'norm_ratio={r_weak.get("norm_ratio_vs_taylor", 0):.2f} '
              f'({t_weak:.0f}s)', flush=True)
        results[cid] = case
        json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'),
                  indent=1, ensure_ascii=False)

        # AP
        t0 = time.time()
        w_ap, sll_design, iters, conv = run_ap(px, py, pz, theta0, phi0,
                                               null_dirs, w_tn)
        t_ap = time.time() - t0
        r_ap = independent_eval(w_ap, px, py, pz, theta0, phi0, null_dirs,
                                w_taylor=w_tn)
        case['ap'] = {**r_ap, 'time_s': t_ap, 'design_grid_sll': sll_design,
                      'iterations': iters, 'converged': conv}
        print(f'  ap: SLL(indep)={r_ap["sll_db"]:.2f} '
              f'(design={sll_design:.2f}) '
              f'null_worst={r_ap["worst_null_neighborhood_db"]:.1f} '
              f'norm_ratio={r_ap.get("norm_ratio_vs_taylor", 0):.2f} '
              f'({t_ap:.0f}s)', flush=True)
        results[cid] = case
        json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'),
                  indent=1, ensure_ascii=False)

        # dense SOCP cold
        t0 = time.time()
        w_dc, sll_dc = solve_dense_socp(px, py, pz, theta0, phi0, null_dirs,
                                        r_t['sll_db'], w_tn)
        t_dc = time.time() - t0
        r_dc = independent_eval(w_dc, px, py, pz, theta0, phi0, null_dirs,
                                w_taylor=w_tn)
        case['dense_socp'] = {**r_dc, 'time_s': t_dc}
        print(f'  dense_socp: SLL(indep)={r_dc["sll_db"]:.2f} '
              f'null_worst={r_dc["worst_null_neighborhood_db"]:.1f} '
              f'norm_ratio={r_dc.get("norm_ratio_vs_taylor", 0):.2f} '
              f'dyn_range={r_dc["amp_dynamic_range_db"]:.0f}dB '
              f'({t_dc:.0f}s)', flush=True)
        results[cid] = case
        json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'),
                  indent=1, ensure_ascii=False)

        # dense SOCP warm (前 4 例, 只比时间)
        if ci < WARM_CASES:
            t0 = time.time()
            w_dw, sll_dw = solve_dense_socp(px, py, pz, theta0, phi0,
                                            null_dirs, r_t['sll_db'], w_tn,
                                            w_init=w_ap)
            t_dw = time.time() - t0
            r_dw = independent_eval(w_dw, px, py, pz, theta0, phi0, null_dirs,
                                    w_taylor=w_tn)
            case['dense_warm'] = {**r_dw, 'time_s': t_dw}
            print(f'  dense_warm: SLL(indep)={r_dw["sll_db"]:.2f} '
                  f'({t_dw:.0f}s vs cold {t_dc:.0f}s)', flush=True)
            results[cid] = case
            json.dump(results, open(OUT_PATH, 'w', encoding='utf-8'),
                      indent=1, ensure_ascii=False)

    # ---------- 汇总 ----------
    print('\n' + '=' * 80)
    print('汇总（全部为独立验收器口径）')
    print('=' * 80)
    print(f'{"case":<16} {"taylor":>7} {"weak":>7} {"AP":>7} '
          f'{"dense":>7} {"d-null":>7} {"d-norm":>6} {"d-dyn":>6}')
    for cid, c in results.items():
        if 'dense_socp' not in c:
            continue
        print('%-16s %7.2f %7.2f %7.2f %7.2f %7.1f %6.2f %6.0f' % (
            cid, c['taylor']['sll_db'], c['weak_socp']['sll_db'],
            c['ap']['sll_db'], c['dense_socp']['sll_db'],
            c['dense_socp']['worst_null_neighborhood_db'],
            c['dense_socp'].get('norm_ratio_vs_taylor', 0),
            c['dense_socp']['amp_dynamic_range_db']))
    if any('dense_warm' in c for c in results.values()):
        print('\n热启动耗时对比:')
        for cid, c in results.items():
            if 'dense_warm' in c:
                print('  %s: cold %.0fs -> warm %.0fs (SLL %.2f vs %.2f)' % (
                    cid, c['dense_socp']['time_s'], c['dense_warm']['time_s'],
                    c['dense_socp']['sll_db'], c['dense_warm']['sll_db']))

    # ---------- 8×8 强权值预生成（供阶段 0B EEP 止损）----------
    print('\n[8x8] 生成 HFSS 几何上的强 SOCP 权值（4 方向）...', flush=True)
    try:
        m = np.genfromtxt(
            r'D:\学习\tiaozhansai\HFSS_8x8曲面阵列验证\returned_results'
            r'\conformal_8x8\server_return_20260830\raw_extracted\input'
            r'\mapping\element_port_map.csv', delimiter=',', names=True)
        px8 = m['x_mm'] / 24.0
        py8 = m['y_mm'] / 24.0
        pz8 = m['z_mm'] / 24.0
        w8 = {}
        for (t8, p8) in [(20.0, 20.0), (55.0, 35.0), (2.0, 30.0), (1.0, 30.0)]:
            nd8 = _get_null_dirs(t8, p8)
            u08 = np.sin(np.deg2rad(t8)) * np.cos(np.deg2rad(p8))
            v08 = np.sin(np.deg2rad(t8)) * np.sin(np.deg2rad(p8))
            w08 = np.cos(np.deg2rad(t8))
            # 8x8 用自身 Taylor 基线
            nx8 = 8
            # 近似: 幅度用 8 元 Taylor, 相位按坐标
            ax8 = np.ones(nx8)
            amp_grid = np.outer(np.ones(nx8), np.ones(nx8))
            # 用均匀幅度+坐标扫描相位作为基线起点
            w_tb = np.exp(1j * 2 * np.pi * (px8 * u08 + py8 * v08 + pz8 * w08))
            r8 = independent_eval(w_tb, px8, py8, pz8, t8, p8, nd8, nx=8)
            w_best, sll8 = solve_dense_socp(
                px8, py8, pz8, t8, p8, nd8, r8['sll_db'], w_tb,
                eps_null=10 ** (-40 / 20))
            r_best = independent_eval(w_best, px8, py8, pz8, t8, p8, nd8,
                                      w_taylor=w_tb, nx=8)
            w8['th%02d_ph%03d' % (t8, p8)] = w_best
            print('  (th=%2d,ph=%3d): taylor=%6.2f dense=%6.2f '
                  'null=%6.1f norm_ratio=%.2f' % (
                      t8, p8, r8['sll_db'], r_best['sll_db'],
                      r_best['worst_null_neighborhood_db'],
                      r_best.get('norm_ratio_vs_taylor', 0)), flush=True)
        np.savez(os.path.join(OUTPUT_DIR, 'strong_socp_weights_8x8.npz'), **w8)
        print('  saved: outputs/strong_socp_weights_8x8.npz', flush=True)
    except Exception as e:
        print(f'  8x8 生成失败: {e}', flush=True)

    print('\nDone:', OUT_PATH)


if __name__ == '__main__':
    main()
