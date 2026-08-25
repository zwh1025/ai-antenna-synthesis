"""2D 方向图综合模块。

阶段 2 核心发现：可分离 Taylor 的 2D SLL 受限于对角方向过渡区乘积，
无法达到 -35 dBc。改用梯度下降法直接优化 2D 激励幅值（相位固定为
波束指向相位），利用已有的可微 calculate_2d_pattern。

生成的优化激励作为网络训练数据。
"""

import numpy as np
import torch

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    angular_distance_deg,
    calculate_2d_pattern,
    get_2d_sll,
)


def synthesize_2d_gradient(Nx, Ny, theta0, phi0, SLL_target,
                           n_iter=500, lr=0.01,
                           n_theta=91, n_phi=181,
                           exclude_angle=3.0,
                           init_amp=None, verbose=False):
    """梯度下降法 2D 方向图综合。

    优化激励幅值（相位固定为波束指向相位），使 2D SLL ≤ SLL_target。
    利用 PyTorch 自动微分计算梯度。

    Args:
        Nx, Ny: 阵元数
        theta0, phi0: 波束指向 (度)
        SLL_target: 目标副瓣电平 (dB, 负值)
        n_iter: 迭代次数
        lr: 学习率
        n_theta, n_phi: 方向图采样点数
        exclude_angle: 主瓣排除半径 (度)
        init_amp: 初始幅值 (Nx, Ny)
        verbose: 打印进度

    Returns:
        (amp_2d, phase_2d): (Nx, Ny) numpy 数组
    """
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)

    if init_amp is None:
        amp_x, amp_y = taylor_2d_separable(Nx, Ny, 25)
        init_amp = np.outer(amp_x, amp_y)

    steer_px, steer_py = beam_steering_phase_2d(posx, posy, theta0, phi0)
    phase_2d = (steer_px[:, None] + steer_py[None, :]) % (2 * np.pi)

    theta_grid = np.linspace(0, 90, n_theta)
    phi_grid = np.linspace(0, 360, n_phi)
    theta_2d, phi_2d = np.meshgrid(theta_grid, phi_grid, indexing='ij')

    dist = angular_distance_deg(theta_2d, phi_2d, theta0, phi0)
    main_lobe_mask = dist < exclude_angle
    sidelobe_mask = ~main_lobe_mask

    amp_t = torch.tensor(init_amp, dtype=torch.float32, requires_grad=True)
    phase_t = torch.tensor(phase_2d, dtype=torch.float32)
    posx_t = torch.tensor(posx, dtype=torch.float32)
    posy_t = torch.tensor(posy, dtype=torch.float32)
    theta_t = torch.tensor(theta_grid, dtype=torch.float32)
    phi_t = torch.tensor(phi_grid, dtype=torch.float32)
    sl_mask_t = torch.tensor(sidelobe_mask, dtype=torch.float32)
    sll_target_lin = 10 ** (SLL_target / 20)

    optimizer = torch.optim.Adam([amp_t], lr=lr)

    for iteration in range(n_iter):
        optimizer.zero_grad()

        pattern = calculate_2d_pattern(
            amp_t, phase_t, posx_t, posy_t, theta_t, phi_t)

        pattern_norm = torch.pow(10, pattern / 20.0)
        peak = pattern_norm.max()
        pattern_norm = pattern_norm / (peak + 1e-12)

        excess = torch.clamp(pattern_norm - sll_target_lin, min=0.0)
        loss = torch.sum(excess ** 2 * sl_mask_t)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            amp_t.clamp_(0.0, 1.0)
            amp_t.div_(amp_t.max())

        if verbose and (iteration + 1) % 50 == 0:
            pat_np = pattern.detach().numpy()
            sll = get_2d_sll(pat_np, theta_grid, phi_grid,
                             theta0, phi0, exclude_angle)
            print(f"  iter {iteration+1}: loss={loss.item():.4f}, SLL={sll:.1f} dB")

        if loss.item() < 1e-6:
            if verbose:
                print(f"  iter {iteration+1}: converged")
            break

    with torch.no_grad():
        amp_2d = amp_t.detach().numpy()
        amp_2d = amp_2d / amp_2d.max()

    return amp_2d, phase_2d


def verify_2d(amp_2d, phase_2d, posx, posy, theta0, phi0,
              theta_grid=None, phi_grid=None, exclude_angle=3.0):
    """验证 2D 综合结果，返回指标字典。"""
    if theta_grid is None:
        theta_grid = np.linspace(0, 90, 181)
    if phi_grid is None:
        phi_grid = np.linspace(0, 360, 361)

    pat = calculate_2d_pattern(
        amp_2d, phase_2d, posx, posy, theta_grid, phi_grid).numpy()

    sll = get_2d_sll(pat, theta_grid, phi_grid, theta0, phi0, exclude_angle)

    idx = np.unravel_index(np.argmax(pat), pat.shape)
    peak_theta = theta_grid[idx[0]]
    peak_phi = phi_grid[idx[1]]
    dphi = abs(peak_phi - phi0)
    dphi = min(dphi, 360 - dphi)
    pointing_error = np.sqrt((peak_theta - theta0) ** 2 + dphi ** 2)

    return {
        'sll': sll,
        'peak_theta': float(peak_theta),
        'peak_phi': float(peak_phi),
        'pointing_error': float(pointing_error),
        'pattern': pat,
    }
