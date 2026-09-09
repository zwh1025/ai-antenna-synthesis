"""v5 条件式 DeepSets 数据集生成：混合教师策略。

教师来源（0D 试点结论指导）：
  解析教师（零成本，SOCP 无增益的场景）：
    - 理想平面 120（复用 v3 planar_augmented，mask=1）
    - 理想曲面 280（复用 v2，mask=1）
    - 4096 规模 100（复用 v4，mask=1）
    - 位置误差 ±0.05λ 60（坐标 Taylor 教师，mask=1）
  SOCP 教师（重，失效场景有 +4.7~6.9dB 天花板）：
    - 平面失效 5%/10%/20% 各 50（v2 固定掩膜口径 SOCP，rounds=12）

特征扩展：v4 的 9 维 + mask(1) = 10 维
输出参数化：w = mask · (w0 + Δw/scale)

输出:
  outputs/teacher_labels_v5_analytic.npz  (解析教师, 秒级)
  outputs/teacher_labels_v5_failure.npz   (失效 SOCP 教师, ~3h 后台)
  训练时合并。
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('RAYON_NUM_THREADS', '1')

from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
)
from run_curved_verify import coordinate_taylor_3d, eval_dense_3d, uv_to_uvw
from run_deepsets_train import _get_null_dirs
from run_scale_fix_v4 import _normalize_weights_torch
from run_stage4a_robustness_degradation import apply_position_error

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

NX = NY = 32
N_ELEMENTS = NX * NY
SLL = 35
COORD_NORM = 8.0
SEED_ANALYTIC = 314

POS_N = 60
FAIL_RATES = [0.10, 0.20]  # skip 5% (0D 证明≈Taylor, 无增益)
FAIL_N_PER_RATE = 50       # 50 × 2 rates = 100 高质量样本
FAIL_SEED_BASE = 500
FAIL_THETAS = [0.0, 15.0, 30.0, 45.0, 60.0]


# ============ 解析教师 ============

def gen_position_samples(rng, posx_ax, posy_ax, amp_x, amp_y):
    """位置误差 ±0.05λ，坐标 Taylor 教师（解析已近最优）。"""
    samples = []
    for i in range(POS_N):
        theta0 = float(rng.choice(FAIL_THETAS))
        phi0 = float(rng.uniform(0, 360))
        ppx, ppy, _, _ = apply_position_error(posx_ax, posy_ax, int(rng.randint(0, 10000)))
        px = ppx.reshape(-1)
        py = ppy.reshape(-1)
        pz = np.zeros(N_ELEMENTS)

        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0c = np.cos(np.deg2rad(theta0))

        w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y, theta0, phi0)
        w_t = _normalize_weights_torch(w_t, px, py, pz, u0, v0, w0c)
        null_dirs = _get_null_dirs(theta0, phi0)
        sll_t, _, _, _ = eval_dense_3d(w_t, px, py, pz, theta0, phi0, null_dirs)

        samples.append({
            'px': px, 'py': py, 'pz': pz,
            'mask': np.ones(N_ELEMENTS, dtype=bool),
            'w_taylor_re': w_t.real, 'w_taylor_im': w_t.imag,
            'w_socp_re': w_t.real, 'w_socp_im': w_t.imag,
            'sll_taylor': float(sll_t), 'sll_socp': float(sll_t),
            'alpha': 0.0, 'theta0': theta0, 'phi0': phi0,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0c),
            'perturb': 'position',
        })
    return samples


def load_ideal_planar_1024():
    """生成 1024 平面理想样本（零成本，mask=1）。"""
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    rng = np.random.RandomState(SEED_ANALYTIC + 1)
    px0 = np.tile(posx[:, None], (1, NX)).ravel()
    py0 = np.tile(posy[None, :], (NX, 1)).ravel()
    pz0 = np.zeros(N_ELEMENTS)
    out = []
    THETAS = [0.0, 15.0, 30.0, 45.0, 60.0]
    for i in range(120):
        theta0 = float(rng.choice(THETAS)) + rng.uniform(-5, 5)
        theta0 = min(max(theta0, 0.0), 60.0)
        phi0 = float(rng.uniform(0, 360))
        u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
        v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
        w0c = np.cos(np.deg2rad(theta0))
        w_t = coordinate_taylor_3d(px0, py0, pz0, amp_x, amp_y, theta0, phi0)
        w_t = _normalize_weights_torch(w_t, px0, py0, pz0, u0, v0, w0c)
        null_dirs = _get_null_dirs(theta0, phi0)
        sll_t, _, _, _ = eval_dense_3d(w_t, px0, py0, pz0, theta0, phi0,
                                       null_dirs)
        out.append({
            'px': px0, 'py': py0, 'pz': pz0,
            'mask': np.ones(N_ELEMENTS, dtype=bool),
            'w_taylor_re': w_t.real, 'w_taylor_im': w_t.imag,
            'w_socp_re': w_t.real, 'w_socp_im': w_t.imag,
            'sll_taylor': float(sll_t), 'sll_socp': float(sll_t),
            'alpha': 0.0, 'theta0': theta0, 'phi0': phi0,
            'u0': float(u0), 'v0': float(v0), 'w0': float(w0c),
            'perturb': 'planar_ideal', 'split': 0,
        })
    return out


def load_ideal_4096():
    """复用 v4 4096 样本（mask=1）。"""
    p = os.path.join(OUTPUT_DIR, 'teacher_labels_4096.npz')
    if not os.path.exists(p):
        return []
    d = np.load(p, allow_pickle=True)
    out = []
    for j in range(len(d['theta0'])):
        out.append({
            'px': d['px'][j], 'py': d['py'][j], 'pz': d['pz'][j],
            'mask': np.ones(len(d['px'][j]), dtype=bool),
            'w_taylor_re': d['w_taylor_re'][j], 'w_taylor_im': d['w_taylor_im'][j],
            'w_socp_re': d['w_socp_re'][j], 'w_socp_im': d['w_socp_im'][j],
            'sll_taylor': float(d['sll_taylor'][j]), 'sll_socp': float(d['sll_socp'][j]),
            'alpha': float(d['alpha'][j]), 'theta0': float(d['theta0'][j]),
            'phi0': float(d['phi0'][j]),
            'u0': float(d['u0'][j]), 'v0': float(d['v0'][j]), 'w0': float(d['w0'][j]),
            'perturb': 'scale_4096', 'split': 0,
        })
    return out


def stage1_analytic():
    """生成解析教师数据集（秒级）。"""
    out_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v5_analytic.npz')
    if os.path.exists(out_path):
        print(f'[analytic] 已存在, 跳过: {out_path}')
        return

    posx_ax = uniform_linear_array_pos(NX)
    posy_ax = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    rng = np.random.RandomState(SEED_ANALYTIC)

    samples = []
    print('[analytic] 生成 1024 平面理想 ...')
    n0 = len(samples)
    samples.extend(load_ideal_planar_1024())
    print(f'  平面1024: {len(samples) - n0}')

    print('[analytic] 加载 4096 规模 ...')
    n0 = len(samples)
    samples.extend(load_ideal_4096())
    print(f'  4096: {len(samples) - n0}')

    print('[analytic] 生成位置误差 ...')
    n0 = len(samples)
    samples.extend(gen_position_samples(rng, posx_ax, posy_ax, amp_x, amp_y))
    print(f'  位置: {len(samples) - n0}')

    _save_samples(out_path, samples, 'v5_analytic')
    print(f'[analytic] saved: {out_path} ({len(samples)} samples)')


# ============ 失效 SOCP 教师（后台并行） ============

SOCP_ROUNDS = 15  # 15轮（0D: round10 -24.59, round25 -31.66; 15轮预计-28~-29）
SOCP_RHO = 1.4
SOCCP_EPS_DB = -35
SOCP_COARSE_GRID = 31
SOCP_EVAL_GRID = 81


def _socp_cutting_v5(px, py, pz, theta0, phi0, null_dirs,
                     sll_taylor, w_taylor_n, active, w_ref=None,
                     fail_mask=None, posx_ax=None, posy_ax=None):
    """固定掩膜口径 SOCP（0D v2 验证：失效场景 +4.7~6.9dB 增益）。
    rounds=5, recheck 201 网格, best 选择用固定掩膜口径。
    单样本 ~300-600s，可接受。"""
    import run_teacher_stage0d_v2 as v2mod
    case = {'theta0_deg': theta0, 'phi0_deg': phi0,
            'null_dirs': [list(nd) for nd in null_dirs]}
    best_w, best_sll, _ = v2mod.solve_socp_official(
        px, py, pz, case, w_taylor_n, null_dirs,
        posx_ax, posy_ax, fail_mask=fail_mask, w_ref=w_ref,
        rho=SOCP_RHO, eps_db=SOCCP_EPS_DB,
        n_init=SOCP_COARSE_GRID, rounds=SOCP_ROUNDS)
    return best_w, best_sll


def _gen_failure_tasks():
    """生成失效场景任务列表。"""
    tasks = []
    seed = FAIL_SEED_BASE
    for rate in FAIL_RATES:
        for i in range(FAIL_N_PER_RATE):
            rng = np.random.RandomState(seed)
            theta0 = float(rng.choice(FAIL_THETAS))
            phi0 = float(rng.uniform(0, 360))
            mask_seed = int(rng.randint(0, 100000))
            count = int(np.floor(N_ELEMENTS * rate))
            mrng = np.random.RandomState(mask_seed)
            indices = np.sort(mrng.choice(N_ELEMENTS, size=count, replace=False))
            mask = np.zeros(N_ELEMENTS, dtype=bool)
            mask[indices] = True
            tasks.append({
                'rate': rate, 'theta0': theta0, 'phi0': phi0,
                'mask': mask, 'task_id': seed,
            })
            seed += 1
    return tasks


def _socp_worker(task):
    """单样本失效 SOCP 教师（v2 固定掩膜口径）。"""
    import torch
    torch.set_num_threads(1)
    import run_teacher_stage0d_v2 as v2mod
    import run_teacher_stage0d_pilot as v1

    posx_ax = uniform_linear_array_pos(NX)
    posy_ax = uniform_linear_array_pos(NY)
    px = np.tile(posx_ax[:, None], (1, NY)).ravel()
    py = np.tile(posy_ax[None, :], (NX, 1)).ravel()
    pz = np.zeros(N_ELEMENTS)

    case = {'theta0_deg': task['theta0'], 'phi0_deg': task['phi0'],
            'null_dirs': _get_null_dirs(task['theta0'], task['phi0'])}
    null_dirs = [tuple(nd) for nd in case['null_dirs']]
    fail_mask = task['mask']
    active = ~fail_mask

    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)
    w_t = coordinate_taylor_3d(px, py, pz, amp_x, amp_y,
                               task['theta0'], task['phi0'])
    w_t[fail_mask] = 0.0
    u0 = np.sin(np.deg2rad(task['theta0'])) * np.cos(np.deg2rad(task['phi0']))
    v0 = np.sin(np.deg2rad(task['theta0'])) * np.sin(np.deg2rad(task['phi0']))
    w0c = np.cos(np.deg2rad(task['theta0']))
    a_main = np.exp(1j * 2 * np.pi * (px * u0 + py * v0))
    resp = np.conj(a_main[active]) @ w_t[active]
    if abs(resp) > 1e-12:
        w_t = w_t / resp

    # A0 基线权值（掩膜参考）
    from mylib.antenna_calc import (beam_steering_phase_2d, combine_2d_excitation)
    from mylib.sum_diff import capon_nulling_2d
    phx, phy = beam_steering_phase_2d(posx_ax, posy_ax, task['theta0'], task['phi0'])
    amp0, phase0 = combine_2d_excitation(amp_x, amp_y, phx, phy)
    amp0_l, phase0_l = capon_nulling_2d(posx_ax, posy_ax, amp0, phase0,
                                       task['theta0'], task['phi0'],
                                       case['null_dirs'])
    w_ref = (amp0_l * np.exp(1j * phase0_l)).reshape(-1).copy()
    w_ref[fail_mask] = 0.0

    sll_taylor, _, _, _ = eval_dense_3d(w_t, px, py, pz,
                                        task['theta0'], task['phi0'],
                                        null_dirs, n_eval=SOCP_EVAL_GRID)
    best_w, best_sll = _socp_cutting_v5(
        px, py, pz, task['theta0'], task['phi0'], null_dirs,
        float(sll_taylor), w_t, active, w_ref=w_ref,
        fail_mask=fail_mask, posx_ax=posx_ax, posy_ax=posy_ax)

    return {
        'task_id': task['task_id'], 'rate': task['rate'],
        'px': px, 'py': py, 'pz': pz,
        'theta0': task['theta0'], 'phi0': task['phi0'],
        'mask': fail_mask,
        'w_taylor_re': w_t.real, 'w_taylor_im': w_t.imag,
        'w_socp_re': best_w.real, 'w_socp_im': best_w.imag,
        'sll_taylor': float(sll_taylor), 'sll_socp': float(best_sll),
        'alpha': 0.0,
        'u0': float(u0), 'v0': float(v0), 'w0': float(w0c),
        'perturb': f"failure_{task['rate']}",
    }


def stage2_failure():
    """后台并行生成失效 SOCP 教师。"""
    out_path = os.path.join(OUTPUT_DIR, 'teacher_labels_v5_failure.npz')
    if os.path.exists(out_path):
        print(f'[failure] 已存在, 跳过: {out_path}')
        return

    import multiprocessing as mp
    tasks = _gen_failure_tasks()
    n_proc = min(len(tasks), max(1, (os.cpu_count() or 8) - 2))
    print(f'[failure] {len(tasks)} 失效样本 SOCP 教师 (rounds={SOCP_ROUNDS}, '
          f'rho={SOCP_RHO}) on {n_proc} procs', flush=True)
    t0 = time.time()
    with mp.get_context('fork').Pool(n_proc) as pool:
        results = []
        for i, r in enumerate(pool.imap(_socp_worker, tasks)):
            results.append(r)
            if (i + 1) % 10 == 0 or i == 0:
                print(f'[failure] {i+1}/{len(tasks)} '
                      f'({time.time()-t0:.0f}s, rate={r["rate"]}, '
                      f'th={r["theta0"]:.0f}, SLL_t={r["sll_taylor"]:.1f}, '
                      f'SLL_s={r["sll_socp"]:.1f})', flush=True)
    print(f'[failure] done in {time.time()-t0:.0f}s', flush=True)

    _save_samples(out_path, results, 'v5_failure')
    print(f'[failure] saved: {out_path} ({len(results)} samples)')


# ============ 工具 ============

def _save_samples(path, samples, version):
    keys = ['px', 'py', 'pz', 'mask', 'w_taylor_re', 'w_taylor_im',
            'w_socp_re', 'w_socp_im', 'sll_taylor', 'sll_socp',
            'alpha', 'theta0', 'phi0', 'u0', 'v0', 'w0']
    d = {'version': version, 'n_elements': [len(s['px']) for s in samples]}
    for k in keys:
        d[k] = np.array([s[k] for s in samples], dtype=object)
    d['perturb'] = np.array([s.get('perturb', 'unknown') for s in samples], dtype=object)
    d['split'] = np.array([s.get('split', 0) for s in samples], dtype=np.int32)
    np.savez(path, **d)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['analytic', 'failure', 'all'],
                    default='all')
    args = ap.parse_args()

    if args.only in ('analytic', 'all'):
        stage1_analytic()
    if args.only in ('failure', 'all'):
        stage2_failure()


if __name__ == '__main__':
    main()
