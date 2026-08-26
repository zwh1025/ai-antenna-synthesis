"""200+连续随机方向验证 + 抛物线插值指向误差。

- 随机θ∈[0,60], φ∈[0,360)
- 随机4个主瓣外零陷方向
- 峰值附近0.01°细网格 → 抛物线插值 → 真实指向误差
- 报告RMS, P95, 最差值
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    calculate_2d_pattern, angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d
from mylib.evaluation import evaluate_2d_comprehensive

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35
N_RANDOM = 200

def random_null_dirs(theta0, phi0, rng):
    dirs = []
    attempts = 0
    while len(dirs) < 4 and attempts < 100:
        tn = rng.uniform(10, 85)
        pn = rng.uniform(0, 360)
        d = angular_distance_deg(tn, pn, theta0, phi0)
        if d >= 15:
            dirs.append((float(tn), float(pn)))
        attempts += 1
    while len(dirs) < 4:
        dirs.append((85.0, float(rng.uniform(0, 360))))
    return dirs

def fine_peak_search(amp, phase, posx, posy, theta0, phi0):
    """峰值附近0.01°细网格 + 抛物线插值。"""
    # 先在粗网格找峰值
    th_coarse = np.linspace(0, 90, 181)
    ph_coarse = np.linspace(0, 360, 361)
    pat_c = calculate_2d_pattern(amp, phase, posx, posy, th_coarse, ph_coarse).numpy()
    idx = np.unravel_index(np.argmax(pat_c), pat_c.shape)
    t0 = th_coarse[idx[0]]; p0 = ph_coarse[idx[1]]

    # 细网格 ±2°
    dt = 0.02
    th_fine = np.linspace(max(t0-2, 0), min(t0+2, 90), 201)
    ph_fine = np.linspace((p0-2)%360, (p0+2)%360, 201)
    pat_f = calculate_2d_pattern(amp, phase, posx, posy, th_fine, ph_fine).numpy()
    idx_f = np.unravel_index(np.argmax(pat_f), pat_f.shape)
    t_peak = th_fine[idx_f[0]]; p_peak = ph_fine[idx_f[1]]

    # 抛物线插值 (3点)
    if 0 < idx_f[0] < len(th_fine)-1 and 0 < idx_f[1] < len(ph_fine)-1:
        y0, y1, y2 = pat_f[idx_f[0]-1, idx_f[1]], pat_f[idx_f[0], idx_f[1]], pat_f[idx_f[0]+1, idx_f[1]]
        denom = y0 - 2*y1 + y2
        if abs(denom) > 1e-10:
            offset = 0.5 * (y0 - y2) / denom
            t_peak = t_peak + offset * (th_fine[1]-th_fine[0])

        y0, y1, y2 = pat_f[idx_f[0], idx_f[1]-1], pat_f[idx_f[0], idx_f[1]], pat_f[idx_f[0], idx_f[1]+1]
        denom = y0 - 2*y1 + y2
        if abs(denom) > 1e-10:
            offset = 0.5 * (y0 - y2) / denom
            p_peak = p_peak + offset * (ph_fine[1]-ph_fine[0])

    err = angular_distance_deg(t_peak, p_peak % 360, theta0, phi0)
    return err, t_peak, p_peak

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    rng = np.random.RandomState(42)
    print("="*80)
    print(f"Random Direction Validation ({N_RANDOM} directions)")
    print(f"θ∈[0,60], φ∈[0,360), 4 random nulls, parabolic peak interpolation")
    print("="*80)

    taylor_slls = []; lcmv_slls = []
    lcmv_nulls = []; pointing_errs = []; bw3dbs = []

    t0 = time.time()
    for i in range(N_RANDOM):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        null_dirs = random_null_dirs(theta0, phi0, rng)

        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_sum, phase_sum = combine_2d_excitation(amp_x, amp_y, px, py)

        # Taylor
        r_t = evaluate_2d_comprehensive(
            amp_sum, phase_sum, posx, posy, theta0, phi0, null_dirs=null_dirs)
        taylor_slls.append(r_t['sll_first_null'])
        bw3dbs.append(r_t['bw_3db'])

        # LCMV
        amp_l, phase_l = capon_nulling_2d(
            posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)
        r_l = evaluate_2d_comprehensive(
            amp_l, phase_l, posx, posy, theta0, phi0, null_dirs=null_dirs)
        lcmv_slls.append(r_l['sll_first_null'])
        for nr in r_l['null_results']:
            lcmv_nulls.append(nr['max_3deg'])

        # 插值指向误差
        pe, _, _ = fine_peak_search(amp_l, phase_l, posx, posy, theta0, phi0)
        pointing_errs.append(pe)

        if (i+1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{N_RANDOM} ({elapsed:.0f}s): "
                  f"θ={theta0:.1f}° φ={phi0:.1f}° "
                  f"T={r_t['sll_first_null']:.1f} L={r_l['sll_first_null']:.1f} "
                  f"pt={pe:.3f}°")

    t1 = time.time()
    t_arr = np.array(taylor_slls)
    l_arr = np.array(lcmv_slls)
    n_arr = np.array(lcmv_nulls)
    p_arr = np.array(pointing_errs)
    bw_arr = np.array(bw3dbs)

    print(f"\n  Time: {t1-t0:.1f}s")
    print(f"\n{'='*80}")
    print(f"RANDOM DIRECTION RESULTS ({N_RANDOM} directions)")
    print(f"{'='*80}")
    print(f"{'Metric':>25} {'Mean':>8} {'P95':>8} {'Worst':>8} {'Target':>8} {'Pass':>6}")

    for name, vals, target in [
        ('Taylor SLL (1st null)', t_arr, -35),
        ('LCMV SLL (1st null)', l_arr, -35),
        ('LCMV null depth (3°)', n_arr, -30),
        ('Pointing err (°)', p_arr, None),
        ('3dB BW (°)', bw_arr, None),
    ]:
        m = np.mean(vals); p95 = np.percentile(vals, 95); w = np.max(vals)
        if target:
            pr = np.mean(vals <= target) * 100
            print(f"  {name:>25}: {m:>8.2f} {p95:>8.2f} {w:>8.2f} {target:>8} {pr:>5.0f}%")
        else:
            print(f"  {name:>25}: {m:>8.3f} {p95:>8.3f} {w:>8.3f} {'—':>8} {'—':>6}")

    # RMS
    rms = np.sqrt(np.mean(p_arr**2))
    bw_mean = np.mean(bw_arr)
    target_pt = bw_mean / 30
    pt_pass = np.mean(p_arr <= target_pt) * 100
    print(f"\n  Pointing RMS: {rms:.4f}° (target ≤ {target_pt:.4f}° = BW/30)")
    print(f"  Pointing pass: {pt_pass:.0f}%")

    # 汇总
    print(f"\n  Summary:")
    print(f"    Taylor SLL ≤ -35: {np.mean(t_arr <= -35)*100:.0f}% ({np.sum(t_arr <= -35)}/{N_RANDOM})")
    print(f"    LCMV SLL ≤ -35:   {np.mean(l_arr <= -35)*100:.0f}% ({np.sum(l_arr <= -35)}/{N_RANDOM})")
    print(f"    LCMV null ≤ -30:  {np.mean(n_arr <= -30)*100:.0f}% ({np.sum(n_arr <= -30)}/{len(n_arr)}/{4:.0f})")
    print(f"    Pointing ≤ BW/30: {pt_pass:.0f}%")

    results = {
        'taylor_sll': t_arr.tolist(), 'lcmv_sll': l_arr.tolist(),
        'null_depths': n_arr.tolist(), 'pointing_errs': p_arr.tolist(),
        'bw3dbs': bw_arr.tolist(),
    }
    with open(os.path.join(OUTPUT_DIR, 'random_validation.json'), 'w') as f:
        json.dump(results, f)
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
