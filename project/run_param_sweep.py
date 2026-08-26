"""Taylor参数扫描: 寻找严格口径100%达标的最小锥削。

扫描 SLL_design = 35, 38, 40, 42, 45 dB
对每个设计值评估73方向严格第一零点SLL
选择最差值 ≤ -35.5 dB 的最小设计值
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable, taylor_excitation,
    beam_steering_phase_2d, combine_2d_excitation,
)
from mylib.sum_diff import bayliss_excitation, capon_nulling_2d
from mylib.evaluation import evaluate_2d_comprehensive

NX = NY = 32

def get_directions():
    dirs = [(0, 0)]
    for t in [10, 20, 30, 40, 50, 60]:
        for p in range(0, 360, 30):
            dirs.append((t, p))
    return dirs

def get_null_dirs(theta0, phi0):
    from mylib.antenna_calc import angular_distance_deg
    cands = [(30,(phi0+90)%360), (30,(phi0+180)%360),
             (30,(phi0+270)%360), (min(theta0+25,85),(phi0+45)%360)]
    nulls = [(t,p) for t,p in cands if angular_distance_deg(t,p,theta0,phi0)>=15]
    while len(nulls)<4: nulls.append((85,(phi0+len(nulls)*60)%360))
    return nulls[:4]

def main():
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    dirs = get_directions()

    print("="*80)
    print(f"Taylor参数扫描 ({NX}×{NY})")
    print(f"SLL_design = 35, 38, 40, 42, 45 dB")
    print(f"评估: 73方向, 方向自适应第一零点, LCMV")
    print("="*80)

    results = {}
    for sll_design in [35, 38, 40, 42, 45]:
        amp_x, amp_y = taylor_2d_separable(NX, NY, sll_design)

        taylor_slls = []
        lcmv_slls = []
        null_depths_all = []
        bws = []
        worst_dir = None
        worst_sll = 0

        for theta0, phi0 in dirs:
            px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_sum, phase_sum = combine_2d_excitation(amp_x, amp_y, px, py)
            null_dirs = get_null_dirs(theta0, phi0)

            r_t = evaluate_2d_comprehensive(
                amp_sum, phase_sum, posx, posy, theta0, phi0, null_dirs=null_dirs)
            taylor_slls.append(r_t['sll_first_null'])
            bws.append(r_t['bw_3db'])

            amp_l, phase_l = capon_nulling_2d(
                posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)
            r_l = evaluate_2d_comprehensive(
                amp_l, phase_l, posx, posy, theta0, phi0, null_dirs=null_dirs)
            lcmv_slls.append(r_l['sll_first_null'])
            for nr in r_l['null_results']:
                null_depths_all.append(nr['max_3deg'])

            if r_t['sll_first_null'] > worst_sll:
                worst_sll = r_t['sll_first_null']
                worst_dir = (theta0, phi0)

        t_arr = np.array(taylor_slls)
        l_arr = np.array(lcmv_slls)
        n_arr = np.array(null_depths_all)
        bw_arr = np.array(bws)

        t_pass = np.mean(t_arr <= -35) * 100
        l_pass = np.mean(l_arr <= -35) * 100
        n_pass = np.mean(n_arr <= -30) * 100

        results[sll_design] = {
            'taylor_mean': float(np.mean(t_arr)),
            'taylor_worst': float(np.max(t_arr)),
            'taylor_pass': float(t_pass),
            'lcmv_mean': float(np.mean(l_arr)),
            'lcmv_worst': float(np.max(l_arr)),
            'lcmv_pass': float(l_pass),
            'null_worst': float(np.max(n_arr)),
            'null_pass': float(n_pass),
            'bw_mean': float(np.mean(bw_arr)),
            'worst_dir': worst_dir,
        }

        print(f"\n  SLL_design = {sll_design} dB:")
        print(f"    Taylor: mean={np.mean(t_arr):.1f} worst={np.max(t_arr):.1f} "
              f"pass={t_pass:.0f}% worst_dir={worst_dir}")
        print(f"    LCMV:   mean={np.mean(l_arr):.1f} worst={np.max(l_arr):.1f} "
              f"pass={l_pass:.0f}%")
        print(f"    Nulls:  worst={np.max(n_arr):.1f} pass={n_pass:.0f}%")
        print(f"    BW:     mean={np.mean(bw_arr):.2f}°")

    # 选择最优
    print(f"\n{'='*80}")
    print("PARAMETER SELECTION")
    print(f"{'='*80}")
    print(f"{'SLL_design':>10} {'T_worst':>8} {'T_pass':>6} {'L_worst':>8} {'L_pass':>6} {'N_worst':>8} {'BW':>6}")

    best = None
    for sll in [35, 38, 40, 42, 45]:
        r = results[sll]
        print(f"  {sll:>8} {r['taylor_worst']:>8.1f} {r['taylor_pass']:>5.0f}% "
              f"{r['lcmv_worst']:>8.1f} {r['lcmv_pass']:>5.0f}% "
              f"{r['null_worst']:>8.1f} {r['bw_mean']:>5.2f}°")

        if r['taylor_worst'] <= -35.5 and best is None:
            best = sll

    if best:
        print(f"\n  → Recommended: SLL_design = {best} dB")
        print(f"    Taylor worst: {results[best]['taylor_worst']:.1f} dB ≤ -35.5")
        print(f"    Taylor pass: {results[best]['taylor_pass']:.0f}%")
    else:
        print(f"\n  → None reach -35.5 worst, need larger array or different taper")

if __name__ == '__main__':
    main()
