"""非理想条件实验（修正后）。

激励按中心频率(λ=1)和理想阵列设计后固定，
不针对每次误差重新计算解析权值。

条件:
  1. 0.5dB幅度+5.625°相位量化
  2. ±λ/20位置扰动(每阵元独立2D)
  3. 5%/10%/20%阵元失效
  4. ±10%频率偏移(固定移相器)
每种条件100个随机种子+3个扫描方向
报告: mean, P95, worst
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    calculate_2d_pattern, get_2d_sll, angular_distance_deg,
    calculate_2d_pattern_arbitrary,
)
from mylib.evaluation import evaluate_2d_comprehensive
from mylib.sum_diff import capon_nulling_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35
N_SEEDS = 20  # 时间限制, 先20种子
SCAN_DIRS = [(0, 0), (30, 0), (60, 0)]

def quantize_amp_05db(amp):
    """0.5dB步进量化。"""
    amp = np.clip(amp, 1e-6, 1.0)
    db = 20 * np.log10(amp)
    q_db = np.round(db / 0.5) * 0.5
    return 10 ** (q_db / 20)

def quantize_phase_6bit(phase):
    """5.625°移相量化(6bit)。"""
    step = 2 * np.pi / 64
    return (np.round(phase / step) * step) % (2 * np.pi)

def apply_failure(amp, phase, rate, rng):
    """阵元失效: 幅值置零, 相位不变。"""
    mask = np.zeros(NX * NY, dtype=bool)
    n = int(NX * NY * rate)
    mask[rng.choice(NX * NY, n, replace=False)] = True
    mask = mask.reshape(NX, NY)
    amp_f = amp.copy()
    amp_f[mask] = 0
    return amp_f, phase  # 相位不变

def eval_sll_3bw(amp, phase, posx, posy, theta0, phi0):
    """3×3dB_BW口径SLL，用实际峰值位置排除。"""
    theta = np.linspace(0, 90, 91)
    phi = np.linspace(0, 360, 181)
    pat = calculate_2d_pattern(amp.astype(np.float32), phase.astype(np.float32),
                               posx, posy, theta, phi).numpy()
    # 找实际峰值
    idx_peak = np.unravel_index(np.argmax(pat), pat.shape)
    peak_t = theta[idx_peak[0]]; peak_p = phi[idx_peak[1]]

    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = 3.0 * bw / max(np.cos(np.deg2rad(peak_t)), 0.1)
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
    dist = angular_distance_deg(th2d, ph2d, peak_t, peak_p)
    visible = (np.sin(np.deg2rad(th2d))*np.cos(np.deg2rad(ph2d)))**2 + \
              (np.sin(np.deg2rad(th2d))*np.sin(np.deg2rad(ph2d)))**2 <= 1
    mask = (dist >= exc) & visible
    sll = float(np.max(pat[mask])) if np.any(mask) else float('nan')
    pt_err = angular_distance_deg(peak_t, peak_p, theta0, phi0)
    return sll, pt_err

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    print("="*80)
    print(f"Non-ideal Experiments (fixed weights, {N_SEEDS} seeds × {len(SCAN_DIRS)} dirs)")
    print("="*80)

    conditions = [
        ("ideal", lambda a, p, px, py, t0, p0, rng: (a, p, px, py)),
        ("0.5dB+6bit quant", lambda a, p, px, py, t0, p0, rng: (
            quantize_amp_05db(a), quantize_phase_6bit(p), px, py)),
        ("±λ/20 pos perturb", lambda a, p, px, py, t0, p0, rng: (
            a, p,
            px[:, None] + rng.uniform(-0.05, 0.05, (NX, NY)),
            py[None, :] + rng.uniform(-0.05, 0.05, (NX, NY)))),
        ("5% failure", lambda a, p, px, py, t0, p0, rng: (
            *apply_failure(a, p, 0.05, rng), px, py)),
        ("10% failure", lambda a, p, px, py, t0, p0, rng: (
            *apply_failure(a, p, 0.10, rng), px, py)),
        ("20% failure", lambda a, p, px, py, t0, p0, rng: (
            *apply_failure(a, p, 0.20, rng), px, py)),
        ("-10% freq", lambda a, p, px, py, t0, p0, rng: (a, p, px, py)),
        ("+10% freq", lambda a, p, px, py, t0, p0, rng: (a, p, px, py)),
    ]

    all_results = {}
    t0 = time.time()

    for cond_name, cond_fn in conditions:
        slls = []
        for seed in range(N_SEEDS):
            rng = np.random.RandomState(seed * 1000 + 42)
            for theta0, phi0 in SCAN_DIRS:
                px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
                amp_sum, phase_sum = combine_2d_excitation(amp_x, amp_y, px, py)

                # 按中心频率设计后固定
                if "freq" in cond_name:
                    lamb = 1.0 / (0.9 if "-10%" in cond_name else 1.1)
                    amp_c, phase_c, px_c, py_c = amp_sum, phase_sum, posx, posy
                    theta = np.linspace(0, 90, 91)
                    phi = np.linspace(0, 360, 181)
                    pat = calculate_2d_pattern(
                        amp_c, phase_c, px_c, py_c, theta, phi, lamb=lamb).numpy()
                    bw = 0.886 * 2.0 / NX * 180 / np.pi
                    exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
                    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
                    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
                    visible = (np.sin(np.deg2rad(th2d))*np.cos(np.deg2rad(ph2d)))**2 + \
                              (np.sin(np.deg2rad(th2d))*np.sin(np.deg2rad(ph2d)))**2 <= 1
                    mask = (dist >= exc) & visible
                    sll = float(np.max(pat[mask])) if np.any(mask) else float('nan')
                else:
                    amp_c, phase_c, px_c, py_c = cond_fn(
                        amp_sum, phase_sum, posx, posy, theta0, phi0, rng)

                    if px_c.ndim == 2:  # 位置扰动
                        theta = np.linspace(0, 90, 91)
                        phi = np.linspace(0, 360, 181)
                        pat = calculate_2d_pattern_arbitrary(
                            amp_c.astype(np.float32), phase_c.astype(np.float32),
                            px_c.astype(np.float32), py_c.astype(np.float32),
                            theta, phi).numpy()
                    else:
                        sll, _ = eval_sll_3bw(amp_c, phase_c, px_c, py_c, theta0, phi0)
                        pat = None

                    if pat is not None:
                        bw = 0.886 * 2.0 / NX * 180 / np.pi
                        exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
                        th2d, ph2d = np.meshgrid(
                            np.linspace(0,90,91), np.linspace(0,360,181), indexing='ij')
                        dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
                        visible = (np.sin(np.deg2rad(th2d))*np.cos(np.deg2rad(ph2d)))**2 + \
                                  (np.sin(np.deg2rad(th2d))*np.sin(np.deg2rad(ph2d)))**2 <= 1
                        mask = (dist >= exc) & visible
                        sll = float(np.max(pat[mask])) if np.any(mask) else float('nan')

                sll = float(sll) if not isinstance(sll, float) or not np.isnan(sll) else sll
                if not np.isnan(sll):
                    slls.append(sll)

        slls = np.array(slls)
        mean = np.mean(slls)
        p95 = np.percentile(slls, 95)
        worst = np.max(slls)
        degrade = worst - np.mean(slls[:len(SCAN_DIRS)])  # vs ideal
        pass_rate = np.mean(slls <= -35) * 100

        all_results[cond_name] = {
            'mean': float(mean), 'p95': float(p95), 'worst': float(worst),
            'pass_rate': float(pass_rate), 'n': len(slls)
        }

        print(f"  {cond_name:>20}: mean={mean:>6.1f} P95={p95:>6.1f} "
              f"worst={worst:>6.1f} pass={pass_rate:>4.0f}% ({len(slls)} samples)")

    t1 = time.time()
    print(f"\n  Time: {t1-t0:.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'nonideal_v2.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("NON-IDEAL EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Condition':>20} {'Mean':>8} {'P95':>8} {'Worst':>8} {'≤-35':>6} {'Degrad':>7}")
    ideal_worst = all_results['ideal']['worst']
    for cond_name, r in all_results.items():
        deg = r['worst'] - ideal_worst
        print(f"  {cond_name:>20}: {r['mean']:>8.1f} {r['p95']:>8.1f} "
              f"{r['worst']:>8.1f} {r['pass_rate']:>5.0f}% {deg:>+7.1f}")
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
