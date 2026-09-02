"""Taylor 基线随机 200 方向 3bw 口径补测（与 acceptance_v3_ai 同口径对照）。

背景：原始 random_validation.json 只保存 first-null（-30dB 连通域）口径，
该口径在大扫描角下存在主瓣被可见域边界截断的伪影（-8 dB 假副瓣），
无法支撑 README 中"200/200 通过 -35 dBc"的表述。本脚本用当前评估器
对 Taylor 基线在与 run_acceptance_v3_ai 完全相同的 200 随机方向
（同种子、同零陷生成顺序）上补测 3×3dB_BW 口径，形成完整三方对照：
Taylor / ai_direct / ai_lcmv。

输出: outputs/random200_taylor_3bw.json
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import uniform_linear_array_pos, taylor_2d_separable
from mylib import evaluation as ev
from run_acceptance_v2 import get_scan_directions, get_null_dirs
from run_random_validation import random_null_dirs
from run_acceptance_v3_ai import (
    _pattern_on_uv_grid_torch, _capon_nulling_2d_torch,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
N_RANDOM = 200
NX = NY = 32
SLL = 35


def main():
    print("=" * 78, flush=True)
    print("Taylor baseline, 200 random directions, 3x3dB_BW caliber", flush=True)
    print("=" * 78, flush=True)

    ev._pattern_on_uv_grid = _pattern_on_uv_grid_torch

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    from mylib.antenna_calc import beam_steering_phase_2d, combine_2d_excitation

    rng = np.random.RandomState(42)
    results = []
    t0 = time.time()
    for i in range(N_RANDOM):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        null_dirs = random_null_dirs(theta0, phi0, rng)

        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp2d, ph2d = combine_2d_excitation(amp_x, amp_y, px, py)

        r_t = ev.evaluate_uv(amp2d, ph2d, posx, posy, theta0, phi0,
                             null_dirs=null_dirs)

        amp_l, ph_l = _capon_nulling_2d_torch(posx, posy, amp2d, ph2d,
                                              theta0, phi0, null_dirs)
        r_l = ev.evaluate_uv(amp_l, ph_l, posx, posy, theta0, phi0,
                             null_dirs=null_dirs)

        nulls_t = [nr['max_3deg'] for nr in r_t['null_results']]
        nulls_l = [nr['max_3deg'] for nr in r_l['null_results']]

        results.append({
            'theta0': float(theta0), 'phi0': float(phi0),
            'taylor_sll_3bw': r_t['sll_3bw'],
            'taylor_sll_fn': r_t['sll_first_null'],
            'taylor_worst_null': max(nulls_t),
            'lcmv_sll_3bw': r_l['sll_3bw'],
            'lcmv_sll_fn': r_l['sll_first_null'],
            'lcmv_worst_null': max(nulls_l),
        })
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1:3d}/{N_RANDOM} ({time.time()-t0:.0f}s) "
                  f"th={theta0:.1f} T={r_t['sll_3bw']:.2f} "
                  f"L={r_l['sll_3bw']:.2f}", flush=True)

    ts = np.array([r['taylor_sll_3bw'] for r in results])
    ls = np.array([r['lcmv_sll_3bw'] for r in results])
    tn = np.array([r['taylor_worst_null'] for r in results])
    ln = np.array([r['lcmv_worst_null'] for r in results])

    print("\n  Taylor  3bw: worst=%.2f  pass<=-35: %d/200" % (ts.max(), (ts <= -35).sum()))
    print("  LCMV    3bw: worst=%.2f  pass<=-35: %d/200" % (ls.max(), (ls <= -35).sum()))
    print("  Taylor null: worst=%.2f  pass<=-30: %d/200" % (tn.max(), (tn <= -30).sum()))
    print("  LCMV   null: worst=%.2f  pass<=-30: %d/200" % (ln.max(), (ln <= -30).sum()))

    out = {
        'description': 'Taylor/LCMV baseline on the same 200 random directions '
                       'as acceptance_v3_ai.json, 3x3dB_BW caliber, current evaluator',
        'summary': {
            'taylor_sll_3bw_worst': float(ts.max()),
            'taylor_sll_3bw_pass': int((ts <= -35).sum()),
            'lcmv_sll_3bw_worst': float(ls.max()),
            'lcmv_sll_3bw_pass': int((ls <= -35).sum()),
            'taylor_null_worst': float(tn.max()),
            'taylor_null_pass': int((tn <= -30).sum()),
            'lcmv_null_worst': float(ln.max()),
            'lcmv_null_pass': int((ln <= -30).sum()),
        },
        'results': results,
    }
    out_path = os.path.join(OUTPUT_DIR, 'random200_taylor_3bw.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
