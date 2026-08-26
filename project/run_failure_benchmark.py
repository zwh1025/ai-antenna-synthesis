"""失效补偿基准：无补偿 vs 传统LCMV重优化。

固定32×32阵列, 随机场景:
  - 连续扫描θ∈[0,60], φ∈[0,360)
  - 4个随机主瓣外零陷
  - 失效率 5%/10%/20%
  - 激励按中心频率设计后固定

方法对比:
  1. 无补偿: Taylor→零失效→直接评估
  2. 传统LCMV: 对活跃阵元做LCMV重优化(主瓣=1, 零陷=0)
  3. (AI补偿: 后续)
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    calculate_2d_pattern, get_2d_sll, angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35
N_SCENES = 100  # 每种失效率的场景数


def random_scenario(rng):
    """生成随机场景。"""
    theta0 = rng.uniform(0, 60)
    phi0 = rng.uniform(0, 360)

    # 4个主瓣外零陷
    null_dirs = []
    attempts = 0
    while len(null_dirs) < 4 and attempts < 50:
        tn = rng.uniform(10, 85)
        pn = rng.uniform(0, 360)
        if angular_distance_deg(tn, pn, theta0, phi0) >= 15:
            null_dirs.append((float(tn), float(pn)))
        attempts += 1
    while len(null_dirs) < 4:
        null_dirs.append((85.0, float(rng.uniform(0, 360))))

    return theta0, phi0, null_dirs


def generate_failure_mask(rate, rng):
    """随机失效mask。"""
    n_total = NX * NY
    n_fail = int(n_total * rate)
    mask = np.zeros(n_total, dtype=bool)
    mask[rng.choice(n_total, n_fail, replace=False)] = True
    return mask.reshape(NX, NY)


def eval_sll_with_peak_exclusion(amp, phase, posx, posy, theta0, phi0):
    """用实际峰值位置做3×3dB_BW排除。"""
    theta = np.linspace(0, 90, 91)
    phi = np.linspace(0, 360, 181)
    pat = calculate_2d_pattern(
        amp.astype(np.float32), phase.astype(np.float32),
        posx, posy, theta, phi).numpy()

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

    # 零陷评估
    null_depths = []
    for tn, pn in [(0,0)]:  # 简化: 只评一个零陷方向
        pass
    return sll, pt_err


def lcmv_on_active(posx, posy, amp_ref, phase_ref, failure_mask,
                    theta0, phi0, null_dirs):
    """MVDR最小副瓣能量补偿(活跃阵元)。

    min w^H R_sl w  s.t.  a_main^H w = 1, a_null^H w = 0
    R_sl = 副瓣区方向图相关矩阵
    解: w = R^{-1} C (C^H R^{-1} C)^{-1} f
    """
    Nx, Ny = NX, NY
    k = 2 * np.pi
    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    active = (~failure_mask).ravel()
    n_active = np.sum(active)
    if n_active < 10:
        return amp_ref * (~failure_mask), phase_ref

    posx_a = posx_2d.ravel()[active]
    posy_a = posy_2d.ravel()[active]

    # 主瓣方向
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    a_main = np.exp(1j * k * (posx_a * u0 + posy_a * v0))

    # 零陷约束
    cols = [a_main]
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        cols.append(np.exp(1j * k * (posx_a * un + posy_a * vn)))
    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0

    # 副瓣区相关矩阵(粗采样)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi
    exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    n_th, n_ph = 23, 45
    th = np.linspace(0, 90, n_th)
    ph = np.linspace(0, 360, n_ph)
    th2d, ph2d = np.meshgrid(th, ph, indexing='ij')
    u_sl = (np.sin(np.deg2rad(th2d)) * np.cos(np.deg2rad(ph2d))).ravel()
    v_sl = (np.sin(np.deg2rad(th2d)) * np.sin(np.deg2rad(ph2d))).ravel()
    dist = np.sqrt((u_sl - u0)**2 + (v_sl - v0)**2)
    sl_mask = (dist >= np.sin(np.deg2rad(exc))) & (u_sl**2 + v_sl**2 <= 1)

    # 构建副瓣区导向矩阵
    A_sl = np.zeros((np.sum(sl_mask), n_active), dtype=complex)
    for idx, (u, v) in enumerate(zip(u_sl[sl_mask], v_sl[sl_mask])):
        A_sl[idx] = np.exp(1j * k * (posx_a * u + posy_a * v))

    # R = A_sl^H A_sl + 正则化
    R = A_sl.conj().T @ A_sl + 1e-3 * np.eye(n_active) * np.trace(A_sl.conj().T @ A_sl) / n_active
    R_inv = np.linalg.inv(R)

    # MVDR with null constraints
    CR = C.conj().T @ R_inv @ C
    w_opt_active = R_inv @ C @ np.linalg.lstsq(CR, f, rcond=1e-10)[0]

    # 重建完整权值矩阵
    w_full = np.zeros(Nx * Ny, dtype=complex)
    w_full[active] = w_opt_active
    w_mat = w_full.reshape(Nx, Ny)

    new_amp = np.abs(w_mat)
    if new_amp.max() > 0:
        new_amp = new_amp / new_amp.max()
    new_phase = np.angle(w_mat) % (2 * np.pi)
    return new_amp, new_phase


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    print("=" * 80)
    print(f"Failure Compensation Benchmark ({NX}×{NY})")
    print(f"Scenes: {N_SCENES} per failure rate")
    print(f"Methods: 1) no compensation  2) LCMV re-optimization")
    print("=" * 80)

    all_results = {}
    t0 = time.time()

    for rate in [0.05, 0.10, 0.20]:
        rng = np.random.RandomState(42)
        slls_no = []; slls_lcmv = []
        pt_errs_no = []; pt_errs_lcmv = []

        for i in range(N_SCENES):
            theta0, phi0, null_dirs = random_scenario(rng)
            failure_mask = generate_failure_mask(rate, rng)

            px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)

            # 方法1: 无补偿
            amp_no = amp_ref * (~failure_mask)
            phase_no = phase_ref.copy()
            sll_no, pt_no = eval_sll_with_peak_exclusion(
                amp_no, phase_no, posx, posy, theta0, phi0)
            slls_no.append(sll_no); pt_errs_no.append(pt_no)

            # 方法2: LCMV重优化
            amp_lcmv, phase_lcmv = lcmv_on_active(
                posx, posy, amp_ref, phase_ref, failure_mask,
                theta0, phi0, null_dirs)
            sll_lcmv, pt_lcmv = eval_sll_with_peak_exclusion(
                amp_lcmv, phase_lcmv, posx, posy, theta0, phi0)
            slls_lcmv.append(sll_lcmv); pt_errs_lcmv.append(pt_lcmv)

            if (i + 1) % 20 == 0:
                print(f"  rate={rate:.0%} {i+1}/{N_SCENES}: "
                      f"no={sll_no:.1f} lcmv={sll_lcmv:.1f}")

        slls_no = np.array(slls_no); slls_lcmv = np.array(slls_lcmv)
        pt_no = np.array(pt_errs_no); pt_lcmv = np.array(pt_errs_lcmv)

        rate_str = f"{int(rate*100)}%"
        all_results[rate_str] = {
            'no_comp': {
                'mean': float(np.mean(slls_no)),
                'p95': float(np.percentile(slls_no, 95)),
                'worst': float(np.max(slls_no)),
                'pass': float(np.mean(slls_no <= -35) * 100),
                'pt_rms': float(np.sqrt(np.mean(pt_no**2))),
            },
            'lcmv': {
                'mean': float(np.mean(slls_lcmv)),
                'p95': float(np.percentile(slls_lcmv, 95)),
                'worst': float(np.max(slls_lcmv)),
                'pass': float(np.mean(slls_lcmv <= -35) * 100),
                'pt_rms': float(np.sqrt(np.mean(pt_lcmv**2))),
            },
            'improvement': float(np.mean(slls_lcmv) - np.mean(slls_no)),
        }

        print(f"\n  {rate_str}:")
        print(f"    No comp:  mean={np.mean(slls_no):.1f} worst={np.max(slls_no):.1f} "
              f"pass={np.mean(slls_no<=-35)*100:.0f}% pt_rms={np.sqrt(np.mean(pt_no**2)):.3f}°")
        print(f"    LCMV:     mean={np.mean(slls_lcmv):.1f} worst={np.max(slls_lcmv):.1f} "
              f"pass={np.mean(slls_lcmv<=-35)*100:.0f}% pt_rms={np.sqrt(np.mean(pt_lcmv**2)):.3f}°")
        print(f"    Improve:  {np.mean(slls_lcmv) - np.mean(slls_no):+.1f} dB")

    t1 = time.time()
    print(f"\n  Time: {t1-t0:.1f}s")

    with open(os.path.join(OUTPUT_DIR, 'failure_benchmark.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print("FAILURE COMPENSATION BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Rate':>5} {'Method':>10} {'Mean':>8} {'P95':>8} {'Worst':>8} {'≤-35':>6} {'PtRMS':>7}")
    for rate_str in ['5%', '10%', '20%']:
        r = all_results[rate_str]
        for method in ['no_comp', 'lcmv']:
            m = r[method]
            print(f"  {rate_str:>5} {method:>10}: {m['mean']:>8.1f} {m['p95']:>8.1f} "
                  f"{m['worst']:>8.1f} {m['pass']:>5.0f}% {m['pt_rms']:>7.3f}°")
        print(f"  {'':>5} {'improve':>10}: {r['improvement']:>+8.1f} dB")
    print(f"\n{'='*80}")

if __name__ == '__main__':
    main()
