"""教师标签生成：梯度下降对活跃阵元复权值做数值优化。

对每个失效场景:
  1. Taylor+LCMV 基线 → 失效 → 无补偿 SLL
  2. 梯度下降优化活跃阵元权值（CPU, float32）
  3. 损失: 副瓣超限 + 主瓣响应 + 4零陷 + 正则
  4. 验证: 优化后 SLL < 无补偿, 零陷保持

先做 20 个场景验证可行性。
"""

import os, sys, time, json
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d
from mylib.evaluation import evaluate_uv

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL_DESIGN = 35


def optimize_active_weights(posx, posy, amp_lcmv, phase_lcmv,
                            failure_mask, theta0, phi0, null_dirs,
                            n_iter=300, lr=0.005, n_uv=51):
    """梯度下降优化活跃阵元复权值。

    损失 = 副瓣超限 + 主瓣响应 + 零陷 + 正则
    在 CPU 上运行（避免 NPU float64 问题）。
    """
    posx = np.asarray(posx, dtype=np.float32)
    posy = np.asarray(posy, dtype=np.float32)
    Nx, Ny = NX, NY
    k = float(2 * np.pi)

    active = (~failure_mask).astype(np.float32)

    # 初始权值: LCMV 权值在活跃阵元上
    w_init = (amp_lcmv * np.exp(1j * phase_lcmv)).astype(np.complex64)
    w_init = w_init * active  # 失效阵元置零

    # 可训练参数: 活跃阵元的复权值
    wr = torch.tensor(w_init.real, dtype=torch.float32, requires_grad=True)
    wi = torch.tensor(w_init.imag, dtype=torch.float32, requires_grad=True)
    active_t = torch.tensor(active)

    posx_t = torch.tensor(posx)
    posy_t = torch.tensor(posy)

    # uv 网格
    u = np.linspace(-1, 1, n_uv, dtype=np.float32)
    v = np.linspace(-1, 1, n_uv, dtype=np.float32)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    visible = (u_grid**2 + v_grid**2 <= 1.0).astype(np.float32)

    # 主瓣方向
    u0 = float(np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0)))
    v0 = float(np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0)))

    # 副瓣区: 距主瓣 > exc_uv
    bw = 0.886 * 2.0 / Nx * 180 / np.pi
    exc_deg = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    exc_uv = float(np.sin(np.deg2rad(exc_deg)))
    dist_uv = np.sqrt((u_grid - u0)**2 + (v_grid - v0)**2)
    sl_mask = ((dist_uv >= exc_uv) & (visible.astype(bool))).astype(np.float32)
    sl_mask_t = torch.tensor(sl_mask)

    # 零陷方向
    null_uv = []
    for tn, pn in null_dirs:
        un = float(np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn)))
        vn = float(np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn)))
        null_uv.append((un, vn))

    # 预计算方向图矩阵
    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))
    px_flat = torch.tensor(posx_2d.ravel().astype(np.float32))
    py_flat = torch.tensor(posy_2d.ravel().astype(np.float32))

    u_flat = torch.tensor(u_grid.ravel().astype(np.float32))
    v_flat = torch.tensor(v_grid.ravel().astype(np.float32))
    vis_flat = torch.tensor(visible.ravel())

    optimizer = torch.optim.Adam([wr, wi], lr=lr)
    sll_target_lin = 10 ** (-35 / 20)

    # 保存原始 Taylor 权值作为正则参考
    w_taylor = (amp_lcmv * np.exp(1j * phase_lcmv)).astype(np.float32)
    w_taylor *= active  # 只比较活跃阵元
    wr_ref = torch.tensor(w_taylor.real)
    wi_ref = torch.tensor(w_taylor.imag)

    for it in range(n_iter):
        optimizer.zero_grad()

        wr_a = wr * active_t
        wi_a = wi * active_t

        psi = k * (px_flat.unsqueeze(0) * u_flat.unsqueeze(1) +
                   py_flat.unsqueeze(0) * v_flat.unsqueeze(1))

        cos_p = torch.cos(psi)
        sin_p = torch.sin(psi)
        wr_flat = wr_a.reshape(-1)
        wi_flat = wi_a.reshape(-1)

        real = torch.sum(wr_flat.unsqueeze(0) * cos_p + wi_flat.unsqueeze(0) * sin_p, dim=1)
        imag = torch.sum(wr_flat.unsqueeze(0) * sin_p - wi_flat.unsqueeze(0) * cos_p, dim=1)

        pat = torch.sqrt(real**2 + imag**2 + 1e-8)
        peak = pat.max()
        pat_norm = pat / (peak + 1e-8)

        # 1. 副瓣损失: soft-max (峰值近似)
        sl_vals = (pat_norm * vis_flat * sl_mask_t.ravel()).flatten()
        temp = 0.01
        max_sl = sl_vals.max()
        sl_loss = temp * (max_sl + torch.log(torch.sum(torch.exp((sl_vals - max_sl) / temp) + 1e-12)))

        # 2. 主瓣响应
        dist_main = torch.sqrt((u_flat - u0)**2 + (v_flat - v0)**2)
        idx_main = torch.argmin(dist_main)
        main_resp = pat_norm[idx_main]
        main_loss = (1.0 - main_resp)**2

        # 3. 零陷损失
        null_loss = torch.tensor(0.0)
        for un, vn in null_uv:
            dist_n = torch.sqrt((u_flat - un)**2 + (v_flat - vn)**2)
            idx_n = torch.argmin(dist_n)
            null_resp = pat_norm[idx_n]
            null_loss = null_loss + null_resp**2

        # 4. Taylor 正则: 保持接近原始 Taylor 权值
        taylor_reg = torch.mean((wr_a - wr_ref)**2 + (wi_a - wi_ref)**2) * 0.5

        loss = sl_loss + 0.5 * main_loss + 10.0 * null_loss + taylor_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_([wr, wi], max_norm=0.5)
        optimizer.step()

    with torch.no_grad():
        w_final = (wr_a + 1j * wi_a).numpy()
        amp = np.abs(w_final)
        if amp.max() > 0:
            amp = amp / amp.max()
        phase = np.angle(w_final) % (2 * np.pi)

    return amp, phase


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)

    rng = np.random.RandomState(42)
    n_scenes = 20

    print("="*70)
    print(f"Teacher Label Generation ({n_scenes} scenes)")
    print(f"Method: gradient descent on active element weights")
    print(f"Loss: sidelobe + main lobe + nulls + regularization")
    print("="*70)

    results = []
    t0 = time.time()

    for i in range(n_scenes):
        theta0 = rng.uniform(0, 60)
        phi0 = rng.uniform(0, 360)
        rate = rng.choice([0.05, 0.10, 0.20])

        # 随机零陷
        null_dirs = []
        for _ in range(4):
            for _ in range(20):
                tn = rng.uniform(10, 85)
                pn = rng.uniform(0, 360)
                if angular_distance_deg(tn, pn, theta0, phi0) >= 15:
                    null_dirs.append((float(tn), float(pn)))
                    break
        while len(null_dirs) < 4:
            null_dirs.append((85.0, float(rng.uniform(0, 360))))

        # 失效 mask
        n_fail = int(NX * NY * rate)
        mask_flat = np.zeros(NX * NY, dtype=bool)
        mask_flat[rng.choice(NX * NY, n_fail, replace=False)] = True
        failure_mask = mask_flat.reshape(NX, NY)

        # 基线权值
        px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
        amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)
        amp_lcmv, phase_lcmv = capon_nulling_2d(
            posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)

        # 无补偿
        amp_no = amp_lcmv * (~failure_mask)
        r_no = evaluate_uv(amp_no, phase_lcmv, posx, posy, theta0, phi0,
                            null_dirs, n_uv=101)

        # 优化
        amp_opt, phase_opt = optimize_active_weights(
            posx, posy, amp_lcmv, phase_lcmv,
            failure_mask, theta0, phi0, null_dirs,
            n_iter=200, lr=0.005, n_uv=51)
        r_opt = evaluate_uv(amp_opt, phase_opt, posx, posy, theta0, phi0,
                            null_dirs, n_uv=101)

        sll_no = r_no['sll_3bw']
        sll_opt = r_opt['sll_3bw']
        null_no = max(nr['max_3deg'] for nr in r_no['null_results'])
        null_opt = max(nr['max_3deg'] for nr in r_opt['null_results'])
        improve = sll_opt - sll_no

        results.append({
            'i': i, 'theta0': theta0, 'phi0': phi0, 'rate': float(rate),
            'sll_no': sll_no, 'sll_opt': sll_opt, 'improve': improve,
            'null_no': null_no, 'null_opt': null_opt,
            'pt_no': r_no['pointing_err'], 'pt_opt': r_opt['pointing_err'],
        })

        status = "✓" if improve < -2 else "✗"
        print(f"  {i+1:2d}/{n_scenes}: rate={rate:.0%} θ={theta0:.1f}° "
              f"no={sll_no:.1f} opt={sll_opt:.1f} Δ={improve:+.1f}{status} "
              f"null_no={null_no:.1f} null_opt={null_opt:.1f}")

    t1 = time.time()
    sll_no_arr = np.array([r['sll_no'] for r in results])
    sll_opt_arr = np.array([r['sll_opt'] for r in results])
    improve_arr = np.array([r['improve'] for r in results])

    print(f"\n{'='*70}")
    print("TEACHER OPTIMIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Metric':>20} {'No comp':>8} {'Optimized':>10} {'Improve':>8}")
    print(f"  {'SLL mean':>20}: {np.mean(sll_no_arr):>8.1f} {np.mean(sll_opt_arr):>10.1f} "
          f"{np.mean(improve_arr):>+8.1f}")
    print(f"  {'SLL worst':>20}: {np.max(sll_no_arr):>8.1f} {np.max(sll_opt_arr):>10.1f} "
          f"{np.max(improve_arr):>+8.1f}")
    print(f"  {'SLL ≤-35 pass':>20}: {np.mean(sll_no_arr<=-35)*100:>7.0f}% "
          f"{np.mean(sll_opt_arr<=-35)*100:>9.0f}%")
    print(f"  {'Time':>20}: {t1-t0:.1f}s ({(t1-t0)/n_scenes:.1f}s/scene)")

    with open(os.path.join(OUTPUT_DIR, 'teacher_validation.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n{'='*70}")

    if np.mean(improve_arr) < -2:
        print("  → Teacher optimization is effective! Proceed to CNN training.")
    else:
        print("  → Teacher optimization insufficient. Need better loss/hyperparams.")

if __name__ == '__main__':
    main()
