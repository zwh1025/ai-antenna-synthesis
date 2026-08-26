"""正确的评估函数：uv域第一零点排除。

不使用固定倍数的3dB波束宽度，而是在uv域直接检测主瓣第一零点，
只排除主瓣区域，其余全部作为副瓣。
"""

import numpy as np
import torch

from mylib.antenna_calc import (
    angular_distance_deg,
    calculate_2d_pattern,
    _to_tensor,
)
import math


def find_first_null_uv(amp_2d, phase_2d, posx, posy,
                        u0, v0, lamb=1.0, n_points=401):
    """在uv域沿主瓣方向找第一零点距离。

    从主瓣(u0,v0)向外，在多个方向采样方向图，
    找到第一个零点（<-40dB），取最小距离作为排除半径。
    """
    k = 2 * np.pi / lamb
    amp = np.asarray(amp_2d, dtype=np.float64)
    phase = np.asarray(phase_2d, dtype=np.float64)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = amp.shape

    # 从主瓣方向向外扫描
    r_grid = np.linspace(0.01, 0.3, 200)  # uv距离
    angles = np.linspace(0, 2*np.pi, 16, endpoint=False)  # 16个方向

    min_null_radius = 0.3  # 默认很大

    for ang in angles:
        du = np.cos(ang)
        dv = np.sin(ang)

        for r in r_grid:
            u = u0 + r * du
            v = v0 + r * dv
            if u**2 + v**2 > 1.0:
                break

            psi = k * (posx[:, None] * u + posy[None, :] * v) - phase
            real = np.sum(amp * np.cos(psi))
            imag = np.sum(amp * np.sin(psi))
            mag = np.sqrt(real**2 + imag**2)

            # 主瓣峰值
            psi0 = k * (posx[:, None] * u0 + posy[None, :] * v0) - phase
            peak = np.sqrt(np.sum(amp * np.cos(psi0))**2 +
                          np.sum(amp * np.sin(psi0))**2)

            if peak > 0:
                norm_db = 20 * np.log10(mag / peak + 1e-30)
                if norm_db < -40:  # 找到零点
                    min_null_radius = min(min_null_radius, r)
                    break

    return min_null_radius


def evaluate_2d(amp_2d, phase_2d, posx, posy, theta0, phi0,
                theta_grid=None, phi_grid=None, lamb=1.0):
    """正确评估2D方向图。

    使用uv域第一零点排除主瓣，计算真实SLL。
    """
    if theta_grid is None:
        theta_grid = np.linspace(0, 90, 181)
    if phi_grid is None:
        phi_grid = np.linspace(0, 360, 361)

    # 方向图
    pat = calculate_2d_pattern(
        amp_2d, phase_2d, posx, posy, theta_grid, phi_grid, lamb=lamb).numpy()

    # 主瓣方向
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    # uv域第一零点
    null_radius = find_first_null_uv(
        amp_2d, phase_2d, posx, posy, u0, v0, lamb=lamb)

    # 计算每个方向图点的uv距离
    th2d, ph2d = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    u_grid = np.sin(np.deg2rad(th2d)) * np.cos(np.deg2rad(ph2d))
    v_grid = np.sin(np.deg2rad(th2d)) * np.sin(np.deg2rad(ph2d))
    uv_dist = np.sqrt((u_grid - u0)**2 + (v_grid - v0)**2)

    # 主瓣排除 + 可见域
    visible = (u_grid**2 + v_grid**2) <= 1.0
    sl_mask = (uv_dist >= null_radius) & visible

    # SLL
    if np.any(sl_mask):
        sll = float(np.max(pat[sl_mask]))
    else:
        sll = float('nan')

    # 峰值位置（球面角距离）
    idx_peak = np.unravel_index(np.argmax(pat), pat.shape)
    peak_theta = theta_grid[idx_peak[0]]
    peak_phi = phi_grid[idx_peak[1]]
    pointing_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    return {
        'sll': sll,
        'null_radius': null_radius,
        'pointing_err': pointing_err,
        'peak_theta': float(peak_theta),
        'peak_phi': float(peak_phi),
    }


def evaluate_null_depths(amp_2d, phase_2d, posx, posy,
                          theta0, phi0, null_dirs, lamb=1.0):
    """评估零陷深度和宽度。

    对每个零陷方向：
    - 目标点响应
    - ±1°范围内最大值（零陷宽度）
    """
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)
    pat = calculate_2d_pattern(
        amp_2d, phase_2d, posx, posy, theta, phi, lamb=lamb).numpy()
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')

    results = []
    for tn, pn in null_dirs:
        dist = angular_distance_deg(th2d, ph2d, tn, pn)
        # 目标点响应
        idx = np.unravel_index(np.argmin(dist), pat.shape)
        target_response = pat[idx[0], idx[1]]
        # ±1°范围内最大值
        near_1deg = dist <= 1.0
        near_3deg = dist <= 3.0
        max_1deg = float(np.max(pat[near_1deg])) if np.any(near_1deg) else float('nan')
        max_3deg = float(np.max(pat[near_3deg])) if np.any(near_3deg) else float('nan')
        results.append({
            'target_response': float(target_response),
            'max_1deg': max_1deg,
            'max_3deg': max_3deg,
            'null_theta': tn,
            'null_phi': pn,
        })
    return results
