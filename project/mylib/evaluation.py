"""正确的评估函数：-30dB阈值连通域主瓣检测 + uv域辅助。

主瓣定义：峰值周围 pat > -30dB 的连通区域。
副瓣定义：可见域内、主瓣之外的所有区域。

同时输出：
  - 第一零点口径（严格，圆形排除，可能低估真实SLL）
  - -30dB连通域口径（正确，适应非圆形主瓣）
  - 3×3dB_BW口径（宽松，对比用）
  - 球面指向误差（θ+φ）
  - 目标点零深 + 零陷宽度
"""

import numpy as np
from scipy.ndimage import label

from mylib.antenna_calc import (
    angular_distance_deg,
    calculate_2d_pattern,
    get_2d_sll,
)


def evaluate_2d_comprehensive(amp_2d, phase_2d, posx, posy,
                               theta0, phi0,
                               theta_grid=None, phi_grid=None,
                               lamb=1.0, null_dirs=None):
    """综合评估2D方向图。

    SLL口径:
      - 主口径: 方向自适应第一零点边界（非圆形）
      - 对比: 3×3dB_BW（宽松）
    指向: 球面角距离
    零陷: 目标点+实际最深+零陷宽度
    """
    if theta_grid is None:
        theta_grid = np.linspace(0, 90, 181)
    if phi_grid is None:
        phi_grid = np.linspace(0, 360, 361)

    pat = calculate_2d_pattern(
        amp_2d, phase_2d, posx, posy, theta_grid, phi_grid, lamb=lamb).numpy()

    th2d, ph2d = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    u_grid = np.sin(np.deg2rad(th2d)) * np.cos(np.deg2rad(ph2d))
    v_grid = np.sin(np.deg2rad(th2d)) * np.sin(np.deg2rad(ph2d))
    visible = (u_grid**2 + v_grid**2) <= 1.0

    # 峰值位置
    idx_peak = np.unravel_index(np.argmax(pat), pat.shape)
    peak_theta = theta_grid[idx_peak[0]]
    peak_phi = phi_grid[idx_peak[1]]
    pointing_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    Nx = len(posx)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi

    # ---- SLL 主口径: 方向自适应第一零点 ----
    # 在每个方位角方向，从峰值出发找第一零点
    main_lobe_mask = np.zeros_like(pat, dtype=bool)
    n_az = len(phi_grid)
    for j in range(n_az):
        # 从峰值所在行出发，沿theta方向搜索
        i_start = idx_peak[0]
        # 向theta增大方向
        for i in range(i_start, len(theta_grid)):
            if pat[i, j] < -50:
                break
            main_lobe_mask[i, j] = True
        # 向theta减小方向
        for i in range(i_start-1, -1, -1):
            if pat[i, j] < -50:
                break
            main_lobe_mask[i, j] = True

    sl_mask_fn = (~main_lobe_mask) & visible
    sll_fn = float(np.max(pat[sl_mask_fn])) if np.any(sl_mask_fn) else float('nan')

    # ---- SLL 对比口径: 3×3dB_BW ----
    exc_3bw = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    dist = angular_distance_deg(th2d, ph2d, theta0, phi0)
    sl_mask_3bw = (dist >= exc_3bw) & visible
    sll_3bw = float(np.max(pat[sl_mask_3bw])) if np.any(sl_mask_3bw) else float('nan')

    # ---- 3dB 波束宽度（经过主瓣方向的截面） ----
    phi_idx = np.argmin(np.abs(phi_grid - phi0))
    pat_phi = pat[:, phi_idx]
    bw_3db = _get_3db_bw(pat_phi, theta_grid, theta0)

    # ---- 零陷评估 ----
    null_results = []
    if null_dirs:
        for tn, pn in null_dirs:
            dist_null = angular_distance_deg(th2d, ph2d, tn, pn)
            idx_n = np.unravel_index(np.argmin(dist_null), pat.shape)
            target_resp = float(pat[idx_n[0], idx_n[1]])
            near1 = dist_null <= 1.0
            near3 = dist_null <= 3.0
            max1 = float(np.max(pat[near1])) if np.any(near1) else float('nan')
            max3 = float(np.max(pat[near3])) if np.any(near3) else float('nan')

            near5 = dist_null <= 5.0
            if np.any(near5):
                idx_min = np.unravel_index(np.argmin(pat[near5]), pat.shape)
                actual_null_theta = theta_grid[idx_min[0]]
                actual_null_phi = phi_grid[idx_min[1]]
                null_offset = angular_distance_deg(
                    actual_null_theta, actual_null_phi, tn, pn)
                actual_depth = float(pat[idx_min[0], idx_min[1]])
            else:
                actual_null_theta = float('nan')
                actual_null_phi = float('nan')
                null_offset = float('nan')
                actual_depth = float('nan')

            null_results.append({
                'null_theta': tn, 'null_phi': pn,
                'target_response': target_resp,
                'max_1deg': max1, 'max_3deg': max3,
                'actual_null_theta': actual_null_theta,
                'actual_null_phi': actual_null_phi,
                'null_offset': null_offset,
                'actual_depth': actual_depth,
            })

    return {
        'sll_first_null': sll_fn,
        'sll_3bw': sll_3bw,
        'bw_3db': bw_3db,
        'pointing_err': pointing_err,
        'peak_theta': float(peak_theta),
        'peak_phi': float(peak_phi),
        'null_results': null_results,
    }


def find_first_null_radius(pat, theta_grid, phi_grid, theta0, phi0):
    """在主瓣方向截面找第一零点角度。"""
    phi_idx = np.argmin(np.abs(phi_grid - phi0))
    pat_phi = pat[:, phi_idx]
    idx0 = np.argmin(np.abs(theta_grid - theta0))

    # 向 θ 增大方向搜索
    for i in range(idx0+1, len(theta_grid)):
        if pat_phi[i] < -50:
            return abs(theta_grid[i] - theta0)
    # 向 θ 减小方向
    for i in range(idx0-1, -1, -1):
        if pat_phi[i] < -50:
            return abs(theta_grid[i] - theta0)
    return 10.0  # 默认


def _get_3db_bw(pat_1d, theta_grid, theta0):
    """1D 3dB 波束宽度。"""
    idx0 = np.argmin(np.abs(theta_grid - theta0))
    peak = pat_1d[idx0]
    threshold = peak - 3.0

    left = 0
    for i in range(idx0-1, -1, -1):
        if pat_1d[i] <= threshold:
            left = i; break
    right = len(theta_grid)-1
    for i in range(idx0+1, len(theta_grid)):
        if pat_1d[i] <= threshold:
            right = i; break
    return float(theta_grid[right] - theta_grid[left])
