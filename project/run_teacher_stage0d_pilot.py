"""Stage 0D 试点：扰动孔径教师上限验证（AI 取代 Taylor 论证第一步）。

问题：条件 AI 重综合的天花板由教师决定。SOCP 教师在扰动孔径上能恢复多少？
三臂对比（全部官方评估器 v1.0.0，201 网格，与 Stage4A 同口径）：
  A0 固定权值    Stage4A 冻结口径重建（Taylor-LCMV 权值不动，孔径/量化/频率扰动）
  A1 坐标Taylor+LCMV  扰动孔径解析重综合（= 条件网络 v5 的 w0 基线 + 置零后处理）
  A2 密集SOCP    范数约束教师上限（rho=1.4，阶段0A 配置）

场景（Stage4A 冻结协议，seed=0）：
  ideal        6 cases（sanity：A0 对齐 Stage2/Stage4A 冻结数字）
  失效 5/10/20% 6 cases（mask 精确复用 failure_cases.json failed_indices）
  位置 ±0.05λ   6 cases（RandomState 协议复刻）
  量化 0.5dB/6bit 4 cases（确定性，各臂输出过同一量化器）
  频偏 ±10%     2 cases × 5 频点（坐标 × ratio 等效，评估 lamb=1/ratio）

输出：outputs/teacher_stage0d_pilot.json
判定：A2 官方口径 SLL 相对 A0 的恢复量 = 条件 AI 的可达上限参考。
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
)
from mylib.official_evaluator import evaluate_official_case
from run_curved_verify import coordinate_taylor_3d, uv_to_uvw
from run_teacher_stage0a import solve_dense_socp, independent_eval
from run_stage4a_robustness_degradation import (
    apply_position_error, quantize_amplitude, quantize_phase,
)

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('RAYON_NUM_THREADS', '1')

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'outputs', 'teacher_stage0d_pilot.json')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGE4A = os.path.join(ROOT, 'results', 'stage4a_robustness_degradation')

NX = NY = 32
N_ELEMENTS = NX * NY
GRID = 201
RHO = 1.4
SEED = 0
FREQ_RATIOS = [0.90, 0.95, 1.00, 1.05, 1.10]
PILOT_ROUNDS = 4      # 试点降配（0A 全配 15）：上限估计略保守，注明即可
PILOT_INIT_GRID = 21  # 0A 全配 41
A2_TAG_PREFIXES = ('fail', 'pos')  # A2 只跑失效+位置（关键锚点）；
# ideal/quant/freq 的 A2 因 CLARABEL 单线程耗时过长砍掉，A0/A1 保留

PILOT_REGULAR = ['regular_000', 'regular_004', 'regular_028',
                 'regular_050', 'regular_072']
PILOT_RANDOM = ['random_000', 'random_050']
QUANT_CASES = ['regular_000', 'regular_028', 'regular_072', 'random_000']
FREQ_CASES = ['regular_000', 'regular_050']
FAIL_RATES = [0.05, 0.10, 0.20]


# ---------------- Stage2 冻结权值重建（确定性） ----------------

def rebuild_stage2_weights(cases):
    """按 run_stage2_strict_closure.py 的冻结流程重建 A0 权值。"""
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, 35)
    out = {}
    for case in cases:
        theta0, phi0 = case['theta0_deg'], case['phi0_deg']
        phx, phy = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp, phase = combine_2d_excitation(amp_x, amp_y, phx, phy)
        if case['set'] == 'regular':
            from mylib.sum_diff import capon_nulling_2d
            amp, phase = capon_nulling_2d(
                posx, posy, amp, phase, theta0, phi0, case['null_dirs'])
        out[case['case_id']] = {'amp': amp, 'phase': phase,
                                'sum_method': 'lcmv' if case['set'] == 'regular' else 'taylor'}
    return out


# ---------------- A1：坐标 Taylor + flat LCMV（任意逐元坐标） ----------------

def flat_lcmv(px, py, pz, w_ref, theta0, phi0, null_dirs, active, lamb=1.0):
    """逐元坐标版最小修正 LCMV（照抄 capon_nulling_2d 数学，R=I）。"""
    k = 2 * np.pi / lamb
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    def a_vec(u, v):
        w_dir = np.sqrt(max(1.0 - u * u - v * v, 0.0))
        return np.exp(1j * k * (px[active] * u + py[active] * v
                                + pz[active] * w_dir))

    a_main = a_vec(u0, v0)
    w0 = w_ref[active] / (np.conj(a_main) @ w_ref[active])

    cols = [a_main]
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        cols.append(a_vec(un, vn))
    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0
    residual = f - C.conj().T @ w0
    w_act = w0 + C @ np.linalg.lstsq(C.conj().T @ C, residual, rcond=1e-10)[0]

    w_full = np.zeros(len(px), dtype=complex)
    w_full[active] = w_act
    return w_full


def arm_a1(px, py, pz, theta0, phi0, null_dirs, active, lamb=1.0):
    """A1：扰动孔径坐标 Taylor（活跃元）+ flat LCMV 置零。"""
    amp_x, amp_y = taylor_2d_separable(NX, NY, 35)
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
    w_t[~active] = 0.0
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    w0c = np.cos(np.deg2rad(theta0))
    a_main = np.exp(1j * 2 * np.pi / lamb *
                    (px * u0 + py * v0 + pz * w0c))
    resp = np.conj(a_main[active]) @ w_t[active]
    if abs(resp) > 1e-12:
        w_t = w_t / resp
    w_a1 = flat_lcmv(px, py, pz, w_t, theta0, phi0, null_dirs, active, lamb)
    return w_a1, w_t


# ---------------- 评估 ----------------

def official_sum_sll(w, case, posx, posy, lamb=1.0):
    """官方评估器：和波束 SLL + 零陷。w 为 (n,) 复权值（失效元为 0）。"""
    amp = np.abs(w).reshape(NX, NY)
    if amp.max() > 0:
        amp = amp / amp.max()
    phase = (np.angle(w) % (2 * np.pi)).reshape(NX, NY)
    r = evaluate_official_case(amp, phase, posx, posy,
                               case['theta0_deg'], case['phi0_deg'],
                               null_dirs=case['null_dirs'], n_uv=GRID,
                               lamb=lamb)
    s = r['sum']
    return {'sll_db': float(s['sll_db']),
            'null_worst_db': float(s['null_worst_db'])
            if s.get('null_worst_db') is not None else None,
            'beamwidth_deg': float(s['beamwidth_3db_deg']),
            'pointing_err_deg': float(s['pointing_error_deg'])}


def arm_a0(w_frozen, perturb, case, posx_ideal, posy_ideal):
    """A0：冻结权值 + 扰动（Stage4A 口径）。返回 (w, 评估坐标, lamb)。"""
    amp = w_frozen['amp'].copy()
    phase = w_frozen['phase'].copy()
    posx, posy, lamb = posx_ideal, posy_ideal, 1.0
    if perturb['kind'] == 'failure':
        amp = amp.reshape(-1).copy()
        amp[perturb['mask']] = 0.0
        amp = amp.reshape(NX, NY)
        phase = phase.reshape(-1).copy()
        phase[perturb['mask']] = 0.0
        phase = phase.reshape(NX, NY)
    elif perturb['kind'] == 'position':
        posx, posy = perturb['px'], perturb['py']
    elif perturb['kind'] == 'quant':
        amp = quantize_amplitude(amp)
        phase = quantize_phase(phase)
    elif perturb['kind'] == 'freq':
        lamb = 1.0 / perturb['ratio']
    return amp * np.exp(1j * phase), posx, posy, lamb


# ---------------- 主流程 ----------------

_G = {}


def socp_task(task):
    """A2 臂：子进程内跑密集 SOCP + 官方评估。"""
    import torch
    torch.set_num_threads(1)
    (idx, cid, tag, theta0, phi0, null_dirs, px, py, pz, active,
     w_taylor, lam_s, eval_ex, eval_ey, quant) = task
    case = _G['cases_by_id'][cid]
    t0 = time.time()
    r_t = independent_eval(w_taylor[active], px[active], py[active],
                           pz[active], theta0, phi0, null_dirs)
    w_best, sll_ind = solve_dense_socp(
        px[active], py[active], pz[active], theta0, phi0, null_dirs,
        r_t['sll_db'] if 'sll_db' in r_t else -100.0,
        w_taylor[active], rho=RHO,
        n_init=PILOT_INIT_GRID, rounds=PILOT_ROUNDS)
    a2_time = time.time() - t0
    if w_best is None:
        return {'idx': idx, 'a2': None, 'a2_sll_independent': None,
                'a2_norm_ratio': None, 'a2_time_s': a2_time}
    w_a2 = np.zeros(len(px), dtype=complex)
    w_a2[active] = w_best
    if quant:
        a2_amp = quantize_amplitude(np.abs(w_a2).reshape(NX, NY))
        a2_ph = quantize_phase((np.angle(w_a2) % (2 * np.pi)).reshape(NX, NY))
        w_a2 = (a2_amp * np.exp(1j * a2_ph)).reshape(-1)
    a2 = official_sum_sll(w_a2, case, eval_ex, eval_ey, lam_s)
    r_a2 = independent_eval(w_a2[active], px[active], py[active],
                            pz[active], theta0, phi0, null_dirs)
    return {'idx': idx, 'a2': a2,
            'a2_sll_independent': None if sll_ind is None else float(sll_ind),
            'a2_norm_ratio': r_a2.get('norm_ratio_vs_taylor'),
            'a2_time_s': float(a2_time)}


def prepare_case(case, perturb, weights, posx_ax, posy_ax):
    """A0/A1 臂 + A2 任务打包（快，主进程串行）。"""
    cid = case['case_id']
    theta0, phi0 = case['theta0_deg'], case['phi0_deg']
    null_dirs = [tuple(nd) for nd in case['null_dirs']]
    w_frozen = weights[cid]

    w0, ex, ey, lamb = arm_a0(w_frozen, perturb, case, posx_ax, posy_ax)
    a0 = official_sum_sll(w0.reshape(-1), case, ex, ey, lamb)

    if perturb['kind'] == 'position':
        px = perturb['px'].reshape(-1).copy()
        py = perturb['py'].reshape(-1).copy()
        active = np.ones(N_ELEMENTS, dtype=bool)
    elif perturb['kind'] == 'freq':
        px = np.tile(posx_ax[:, None], (1, NY)).ravel() * perturb['ratio']
        py = np.tile(posy_ax[None, :], (NX, 1)).ravel() * perturb['ratio']
        active = np.ones(N_ELEMENTS, dtype=bool)
    else:
        px = np.tile(posx_ax[:, None], (1, NY)).ravel()
        py = np.tile(posy_ax[None, :], (NX, 1)).ravel()
        active = (~perturb['mask']) if perturb['kind'] == 'failure' \
            else np.ones(N_ELEMENTS, dtype=bool)
    pz = np.zeros(N_ELEMENTS)
    lam_s = 1.0 / perturb['ratio'] if perturb['kind'] == 'freq' else 1.0

    w_a1, w_taylor = arm_a1(px, py, pz, theta0, phi0, null_dirs, active,
                            lamb=lam_s)
    if perturb['kind'] == 'quant':
        a1_amp = quantize_amplitude(np.abs(w_a1).reshape(NX, NY))
        a1_ph = quantize_phase((np.angle(w_a1) % (2 * np.pi)).reshape(NX, NY))
        w_a1 = (a1_amp * np.exp(1j * a1_ph)).reshape(-1)
    a1 = official_sum_sll(w_a1, case, ex if perturb['kind'] == 'position'
                          else posx_ax, ey if perturb['kind'] == 'position'
                          else posy_ax, lam_s)

    task = (cid, perturb['tag'], theta0, phi0, null_dirs, px, py, pz, active,
            w_taylor, lam_s,
            perturb['px'] if perturb['kind'] == 'position' else posx_ax,
            perturb['py'] if perturb['kind'] == 'position' else posy_ax,
            perturb['kind'] == 'quant')
    row = {'tag': perturb['tag'], 'case_id': cid, 'theta0': theta0,
           'phi0': phi0, 'sum_method': w_frozen['sum_method'],
           'a0': a0, 'a1': a1}
    if perturb['kind'] == 'failure':
        row['fail_rate'] = perturb['rate']
    if perturb['kind'] == 'freq':
        row['freq_ratio'] = perturb['ratio']
    return row, task


def main():
    import multiprocessing as mp

    manifest = json.load(open(os.path.join(
        ROOT, 'results', 'stage2_strict_closure', 'baseline',
        'case_manifest.json'), encoding='utf-8'))
    all_cases = manifest['regular'] + manifest['random']
    by_id = {c['case_id']: c for c in all_cases}
    pilot_ids = PILOT_REGULAR + PILOT_RANDOM
    cases = [by_id[i] for i in pilot_ids]
    _G['cases_by_id'] = by_id

    posx_ax = uniform_linear_array_pos(NX)
    posy_ax = uniform_linear_array_pos(NY)

    print('Rebuilding Stage2 frozen weights (A0) ...', flush=True)
    weights = rebuild_stage2_weights(cases)
    print('  done', flush=True)

    failure_json = json.load(open(os.path.join(STAGE4A, 'failure_cases.json')))
    sanity = []

    # ---------- 构建全部场景 ----------
    scenarios = []
    for case in cases:
        scenarios.append((case, {'kind': 'ideal', 'tag': 'ideal'}))
    for rate in FAIL_RATES:
        for case in cases:
            r0 = [r for r in failure_json['realizations']
                  if r['case_id'] == case['case_id']
                  and abs(r['failure_rate'] - rate) < 1e-9 and r['seed'] == SEED][0]
            mask = np.zeros(N_ELEMENTS, dtype=bool)
            mask[np.array(r0['failed_indices'])] = True
            scenarios.append((case, {'kind': 'failure', 'rate': rate,
                                     'mask': mask,
                                     'tag': f'fail{int(rate*100):02d}',
                                     'frozen_a0': r0['sum_sll_db']}))
    for case in cases:
        ppx, ppy, _, _ = apply_position_error(posx_ax, posy_ax, SEED)
        scenarios.append((case, {'kind': 'position', 'px': ppx, 'py': ppy,
                                 'tag': 'pos'}))
    for case in cases:
        if case['case_id'] in QUANT_CASES:
            scenarios.append((case, {'kind': 'quant', 'tag': 'quant'}))
    for case in cases:
        if case['case_id'] in FREQ_CASES:
            for ratio in FREQ_RATIOS:
                scenarios.append((case, {'kind': 'freq', 'ratio': ratio,
                                         'tag': f'freq{ratio:.2f}'}))

    print(f'Scenarios: {len(scenarios)} — A0/A1 fast arms ...', flush=True)
    rows, tasks = [], []
    t_prep = time.time()
    for i, (case, perturb) in enumerate(scenarios):
        row, task = prepare_case(case, perturb, weights, posx_ax, posy_ax)
        rows.append(row)
        if perturb['tag'].startswith(A2_TAG_PREFIXES):
            tasks.append((i,) + task)
        else:
            row['a2'] = None
        if perturb['kind'] == 'failure':
            sanity.append({'case_id': case['case_id'],
                           'ref': 'stage4a_failure',
                           'ratio': perturb['rate'],
                           'rebuilt': row['a0']['sll_db'],
                           'frozen': perturb['frozen_a0'],
                           'diff': row['a0']['sll_db'] - perturb['frozen_a0']})
        print(f"  {perturb['tag']:10s} {case['case_id']:12s} "
              f"A0={row['a0']['sll_db']:7.2f} A1={row['a1']['sll_db']:7.2f}",
              flush=True)
    print(f'  A0/A1 done in {time.time()-t_prep:.0f}s', flush=True)

    # ---------- A2 并行 ----------
    n_proc = min(len(tasks), max(1, (os.cpu_count() or 8) - 2))
    print(f'\nA2 dense SOCP x{len(tasks)} on {n_proc} processes '
          f'(0A frozen config: rho={RHO}, 15 rounds) ...', flush=True)
    t_socp = time.time()
    with mp.get_context('fork').Pool(n_proc) as pool:
        a2_results = pool.map(socp_task, tasks)
    print(f'A2 done in {time.time()-t_socp:.0f}s', flush=True)

    for a2r in a2_results:
        row = rows[a2r['idx']]
        row['a2'] = a2r['a2']
        row['a2_sll_independent'] = a2r['a2_sll_independent']
        row['a2_norm_ratio'] = a2r.get('a2_norm_ratio')
        row['a2_time_s'] = a2r['a2_time_s']
        a2v = 'SOCP-FAIL' if a2r['a2'] is None else f"{a2r['a2']['sll_db']:7.2f}"
        print(f"  {row['tag']:10s} {row['case_id']:12s} "
              f"A0={row['a0']['sll_db']:7.2f} A1={row['a1']['sll_db']:7.2f} "
              f"A2={a2v} ({a2r['a2_time_s']:.0f}s)", flush=True)

    # ---------- sanity: A0 ideal vs Stage2 冻结 ----------
    stage2_sum = json.load(open(os.path.join(
        ROOT, 'results', 'stage2_strict_closure', 'baseline',
        'sum_cases.json'), encoding='utf-8'))
    frozen_by_id = {(r['case_id'], r['method']): r['sll_db'] for r in stage2_sum}
    for row in rows:
        if row['tag'] != 'ideal':
            continue
        ref = frozen_by_id.get((row['case_id'], row['sum_method']))
        if ref is not None:
            sanity.append({'case_id': row['case_id'], 'method': row['sum_method'],
                           'rebuilt': row['a0']['sll_db'], 'frozen': ref,
                           'diff': row['a0']['sll_db'] - ref})

    # ---------- 聚合 ----------
    def agg(tag_prefix):
        sel = [r for r in rows if r['tag'].startswith(tag_prefix)]
        if not sel:
            return None
        out = {}
        for arm in ['a0', 'a1', 'a2']:
            vals = [r[arm]['sll_db'] for r in sel
                    if r.get(arm) is not None]
            out[arm] = ({'mean': float(np.mean(vals)),
                         'worst': float(np.max(vals)), 'n': len(vals)}
                        if vals else None)
        out['recovery_a2_vs_a0'] = (out['a2']['mean'] - out['a0']['mean']
                                    if out['a2'] else None)
        out['recovery_a1_vs_a0'] = out['a1']['mean'] - out['a0']['mean']
        out['n'] = len(sel)
        return out

    aggregate = {t: agg(t) for t in
                 ['ideal', 'fail05', 'fail10', 'fail20', 'pos', 'quant',
                  'freq0.90', 'freq0.95', 'freq1.00', 'freq1.05', 'freq1.10']}

    payload = {
        'stage': 'Stage 0D pilot — perturbed-aperture teacher upper bound',
        'purpose': 'AI conditional re-synthesis ceiling check before v5 training',
        'arms': {
            'a0': 'frozen Stage2 weights under perturbation (Stage4A caliber)',
            'a1': 'coordinate Taylor + flat LCMV re-synthesis on perturbed aperture',
            'a2': 'dense SOCP with norm bound rho=1.4 (teacher upper bound)',
        },
        'protocol': {
            'metric_version': '1.0.0', 'grid': GRID, 'seed': SEED,
            'masks': 'exact reuse of stage4a failure_cases.json failed_indices',
            'socp': {'rho': RHO, 'eps_null_db': -35, 'rounds': PILOT_ROUNDS,
                     'init_grid': PILOT_INIT_GRID, 'recheck_grid': 81,
                     'note': 'pilot-reduced config vs stage0a (15 rounds/41 grid); '
                             'upper-bound estimate is mildly conservative'},
            'frequency_ratios': FREQ_RATIOS,
        },
        'sanity': sanity,
        'rows': rows,
        'aggregate': aggregate,
    }
    with open(OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 78)
    print('AGGREGATE (official sum SLL, mean dB)')
    print(f"{'scenario':14s} {'n':>3s} {'A0':>8s} {'A1':>8s} {'A2':>8s} "
          f"{'rec A1':>7s} {'rec A2':>7s}")
    for t, a in aggregate.items():
        if a is None:
            continue
        a2m = f"{a['a2']['mean']:8.2f}" if a['a2'] else '    n/a '
        r2 = (f"{a['recovery_a2_vs_a0']:+7.2f}"
              if a['recovery_a2_vs_a0'] is not None else '    n/a')
        print(f"{t:14s} {a['n']:3d} {a['a0']['mean']:8.2f} {a['a1']['mean']:8.2f} "
              f"{a2m} {a['recovery_a1_vs_a0']:+7.2f} {r2}")
    max_sanity = max((abs(s['diff']) for s in sanity), default=0.0)
    print(f"\nsanity max |diff| vs frozen: {max_sanity:.2e}")
    print(f'Saved: {OUTPUT}')


if __name__ == '__main__':
    main()
