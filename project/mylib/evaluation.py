"""统一评估器：u-v 均匀网格 + 2D 连通域主瓣检测。

在均匀 u-v 网格上计算方向图，用 -30dB 连通域检测主瓣区域，
第一零点之外全部作为副瓣。同时输出：
  - SLL（严格连通域 + 3×3dB_BW 对比）
  - 球面指向误差
  - 主瓣增益
  - 目标点零深 + 零陷宽度
  - 和/差波束分别评估
"""

import numpy as np
from scipy.ndimage import label

from mylib.antenna_calc import (
    angular_distance_deg,
    calculate_2d_pattern,
)


def _pattern_on_uv_grid(amp, phase, posx, posy, n_uv=201, lamb=1.0):
    """在均匀 u-v 网格上计算方向图（dB）。"""
    k = 2 * np.pi / lamb
    amp = np.asarray(amp, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = amp.shape

    u = np.linspace(-1, 1, n_uv)
    v = np.linspace(-1, 1, n_uv)
    u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
    visible = (u_grid**2 + v_grid**2) <= 1.0

    # 向量化计算
    # posx_2d[i,j] = posx[i], posy_2d[i,j] = posy[j]
    posx_2d = np.tile(posx[:, None], (1, Ny))  # (Nx, Ny)
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    # 展开: (Nx*Ny,) 复权值
    w = amp * np.exp(1j * phase)
    w_flat = w.ravel()  # (Nx*Ny,)
    px_flat = posx_2d.ravel()  # (Nx*Ny,)
    py_flat = posy_2d.ravel()  # (Nx*Ny,)

    # 对每个 (u,v) 点: F = sum(w * exp(j*k*(px*u + py*v)))
    # A[u_idx, n] = exp(j*k*(px_flat[n]*u + py_flat[n]*v))
    # 太大: n_uv^2 * Nx*Ny = 201^2 * 1024 ≈ 41M，可行

    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()
    visible_flat = visible.ravel()

    # 物理公式: F(u,v) = sum a * exp(j*(k*(x*u + y*v) - phase))
    # w = a * exp(j*phase), 所以 F = sum conj(w) * exp(j*k*(x*u + y*v))
    w_conj = np.conj(w_flat)
    wc_real = w_conj.real
    wc_imag = w_conj.imag

    pattern = np.zeros(n_uv * n_uv)
    for idx in range(len(u_flat)):
        if not visible_flat[idx]:
            pattern[idx] = -300
            continue
        psi = k * (px_flat * u_flat[idx] + py_flat * v_flat[idx])
        real = np.sum(wc_real * np.cos(psi) - wc_imag * np.sin(psi))
        imag = np.sum(wc_real * np.sin(psi) + wc_imag * np.cos(psi))
        pattern[idx] = np.sqrt(real**2 + imag**2)

    peak = pattern[visible_flat].max()
    pattern_db = np.where(pattern > 0, 20*np.log10(pattern / (peak + 1e-30) + 1e-12), -300)
    pattern_db = pattern_db.reshape(n_uv, n_uv)

    return pattern_db, u_grid, v_grid, visible


def evaluate_uv(amp, phase, posx, posy, theta0, phi0,
                null_dirs=None, lamb=1.0, n_uv=201):
    """在 u-v 均匀网格上综合评估方向图。

    主瓣检测: -30dB 连通域（scipy.ndimage.label, 8连通）
    SLL 口径: 连通域外 + 可见域内最大值
    对比口径: 3×3dB_BW

    返回 dict。
    """
    pat, u_grid, v_grid, visible = _pattern_on_uv_grid(
        amp, phase, posx, posy, n_uv, lamb)

    # 主瓣方向
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))

    # 峰值
    idx_peak = np.unravel_index(np.argmax(np.where(visible, pat, -300)), pat.shape)
    peak_u = u_grid[idx_peak]
    peak_v = v_grid[idx_peak]
    peak_val = pat[idx_peak]

    # 指向误差（球面角距离）
    peak_theta = np.degrees(np.arcsin(np.clip(np.sqrt(peak_u**2 + peak_v**2), 0, 1)))
    peak_phi = np.degrees(np.arctan2(peak_v, peak_u)) % 360
    pointing_err = angular_distance_deg(peak_theta, peak_phi, theta0, phi0)

    # ---- 主瓣: -30dB 连通域 ----
    main_lobe_mask = pat > -30.0
    # 8连通（含对角）
    structure = np.ones((3, 3), dtype=int)
    labels_img, n_labels = label(main_lobe_mask, structure=structure)
    peak_label = labels_img[idx_peak]
    main_lobe = (labels_img == peak_label) & visible

    sl_mask_connected = (~main_lobe) & visible
    sll_connected = float(np.max(pat[sl_mask_connected])) if np.any(sl_mask_connected) else float('nan')

    # ---- 对比口径: 3×3dB_BW ----
    Nx = len(posx)
    bw = 0.886 * 2.0 / Nx * 180 / np.pi
    exc_3bw = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    dist_uv = np.sqrt((u_grid - u0)**2 + (v_grid - v0)**2)
    sl_mask_3bw = (dist_uv >= np.sin(np.deg2rad(exc_3bw))) & visible
    sll_3bw = float(np.max(pat[sl_mask_3bw])) if np.any(sl_mask_3bw) else float('nan')

    # ---- 主瓣增益 ----
    main_lobe_gain = float(peak_val)

    # ---- 零陷评估 ----
    null_results = []
    if null_dirs:
        for tn, pn in null_dirs:
            un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
            vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
            dist_null = np.sqrt((u_grid - un)**2 + (v_grid - vn)**2)

            # 目标点响应
            idx_n = np.unravel_index(np.argmin(dist_null), pat.shape)
            target_resp = float(pat[idx_n])

            # ±1°/±3° 最大值（转换到uv距离）
            d1 = np.sin(np.deg2rad(1.0))
            d3 = np.sin(np.deg2rad(3.0))
            near1 = (dist_null <= d1) & visible
            near3 = (dist_null <= d3) & visible
            max1 = float(np.max(pat[near1])) if np.any(near1) else float('nan')
            max3 = float(np.max(pat[near3])) if np.any(near3) else float('nan')

            # 实际最深零点
            near5 = (dist_null <= np.sin(np.deg2rad(5.0))) & visible
            if np.any(near5):
                idx_min = np.unravel_index(np.argmin(pat[near5]), pat.shape)
                actual_depth = float(pat[idx_min])
            else:
                actual_depth = float('nan')

            null_results.append({
                'null_theta': tn, 'null_phi': pn,
                'target_response': target_resp,
                'max_1deg': max1, 'max_3deg': max3,
                'actual_depth': actual_depth,
            })

    # ---- 3dB 波束宽度（主瓣方向截面） ----
    # 沿 u 方向取截面
    du = u[1] - u[0] if 'u' in dir() else 2.0 / (n_uv - 1)
    du = 2.0 / (n_uv - 1)
    pat_u = pat[:, n_uv//2]  # v=0 截面
    u_axis = np.linspace(-1, 1, n_uv)
    # 找峰值两侧 -3dB 点
    peak_u_idx = np.argmax(pat_u)
    threshold = pat_u[peak_u_idx] - 3.0
    left = 0
    for i in range(peak_u_idx-1, -1, -1):
        if pat_u[i] <= threshold:
            left = i; break
    right = n_uv - 1
    for i in range(peak_u_idx+1, n_uv):
        if pat_u[i] <= threshold:
            right = i; break
    bw_3db = float((right - left) * du * 180 / np.pi)  # uv→度

    return {
        'sll_connected': sll_connected,
        'sll_3bw': sll_3bw,
        'pointing_err': pointing_err,
        'peak_theta': float(peak_theta),
        'peak_phi': float(peak_phi),
        'main_lobe_gain': main_lobe_gain,
        'bw_3db': bw_3db,
        'null_results': null_results,
        'main_lobe_pixels': int(np.sum(main_lobe)),
    }
