"""天线阵列物理计算模块。

约定：
  - 方向图计算中 phase 统一用弧度，符号为 psi = k*x - phase。
  - 激励生成返回 numpy 数组；方向图计算接收 torch 张量或 numpy 数组。
  - 方向图返回 dB 值，归一化到 0 dB 峰值。
  - theta/phi 角度一律用度，内部转弧度。

修正项（相对原 LSTMSynthesis 代码）：
  1. np.math → math，np.int → int，np.float → float。
  2. 相位公式由 linspace(0, -cos(θ)*(N-1), N)*π 修正为 k*cos(θ₀)*pos（弧度），
     使波束指向与中心对称位置坐标系匹配。
  3. tf_calculate_* 中 psi = k*x + phase 的符号错误不再保留，统一为 - phase。
"""

import math

import numpy as np
import torch


# ============================================================
#  激励生成（numpy，数据准备用）
# ============================================================

def uniform_linear_array_pos(N, d=0.5):
    """以中心对称的均匀线阵位置（波长单位）。"""
    return (np.arange(1, N + 1) - (N / 2 + 0.5)) * d


def chebyshev_excitation(N, SLL, dtype=np.float32):
    """Dolph-Chebyshev 等副瓣激励幅值（归一化到 [0, 1]）。"""
    N = int(N)
    R0 = 10 ** (SLL / 20)
    x0 = np.cosh(np.arccosh(R0) / (N - 1))
    _fact = np.vectorize(lambda v: float(math.factorial(int(v))), otypes=[np.float64])

    if N % 2 == 0:
        M = N // 2
        n = np.arange(1, M + 1, dtype=int)
        n, p = np.meshgrid(n, n)
        mask = (p >= n).astype(np.float64)
        p = np.where(p >= n, p, M)
        half = (_fact(p + M - 2) / _fact(p - n) / _fact(p + n - 1)
                / _fact(M - p) * ((-1) ** (M - p)) * x0 ** (2 * p - 1)
                * (2 * M - 1))
        half *= mask
        half = half.sum(axis=0)
        exc = np.hstack((half[::-1], half))
    else:
        M = (N - 1) // 2
        n = np.arange(2, M + 2, dtype=int)
        n, p = np.meshgrid(n, n)
        mask = (p >= n).astype(np.float64)
        p = np.where(p >= n, p, M + 1)
        half = (_fact(p + M - 2) / _fact(p - n) / _fact(p + n - 2)
                / _fact(M - p + 1) * ((-1) ** (M - p + 1))
                * x0 ** (2 * (p - 1)) * M)
        half *= mask
        half = half.sum(axis=0)
        p = np.arange(1, M + 2, dtype=int)
        center = (_fact(p + M - 2) / _fact(p - 1) / _fact(p - 1)
                  / _fact(M - p + 1) * ((-1) ** (M - p + 1))
                  * x0 ** (2 * (p - 1)) * M)
        center = center.sum()
        exc = np.hstack((half[::-1], np.array([center]), half))

    exc = exc / np.max(exc)
    return exc.astype(dtype)


def taylor_excitation(L, x_unit, RdB, n_bar=0):
    """Taylor 分布激励幅值（归一化到 [0, 1]）。

    L       阵列孔径长度（= N*d）
    x_unit  阵元位置坐标（波长单位）
    RdB     目标副瓣电平 (dB)
    n_bar   等副瓣个数（0 = 自动）
    """
    R0 = 10 ** (RdB / 20)
    A = np.arccosh(R0) / np.pi
    n_bar_min = int(np.ceil(2 * (A ** 2) + 0.5))
    if n_bar == 0:
        n_bar = n_bar_min
    elif n_bar < n_bar_min:
        return np.nan * np.ones_like(x_unit)

    m = np.arange(1, n_bar).reshape((-1, 1)).astype(int)
    n = np.arange(1, n_bar).reshape((1, -1))

    def _fact(x):
        return np.array([math.factorial(v) for v in x.ravel()]).reshape(x.shape)

    sigma = n_bar / (A ** 2 + (n_bar - 0.5) ** 2) ** 0.5
    Sm1 = ((math.factorial(n_bar - 1)) ** 2) / _fact(n_bar + m - 1) / _fact(n_bar - m - 1)
    Sm2 = np.prod(1 - (m ** 2 / sigma ** 2 / (A ** 2 + (n - 0.5) ** 2)), axis=1).reshape((-1, 1))
    Sm = Sm1 * Sm2

    x_unit = np.asarray(x_unit).reshape((1, -1))
    exc = 1 + 2 * np.sum(Sm * np.cos(m * x_unit * 2 * np.pi / L), axis=0)
    exc = exc / np.max(exc)
    return exc.ravel()


def beam_steering_phase(pos, theta0_deg, lamb=1.0):
    """波束指向 theta0 的阵元激励相位（弧度）。

    phase = k * cos(theta0) * pos

    原代码用 linspace(0, -cos(theta0)*(N-1), N) 计算相位，假设阵元
    从位置 0 开始排列，与中心对称位置坐标系不匹配。本函数直接用
    k*cos(theta0)*pos 修正此问题。
    """
    pos = np.asarray(pos, dtype=np.float64)
    k = 2 * np.pi / lamb
    return k * np.cos(np.deg2rad(theta0_deg)) * pos


# ============================================================
#  方向图计算（torch，可微可批处理）
# ============================================================

def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    arr = np.asarray(x)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float32)
    return torch.as_tensor(arr)


def calculate_1d_pattern(pos, amp, phase, theta_calc=None, lamb=1.0):
    """一维线阵方向图（dB，归一化到峰值）。

    pos         (N,) 或 (B, N)  阵元位置（波长单位）
    amp         (N,) 或 (B, N)  激励幅值
    phase       (N,) 或 (B, N)  激励相位（弧度）
    theta_calc  (M,)            方向图角度采样点（度），默认 0–180 共 361 点
    lamb        波长（默认 1.0）

    返回 (M,) 或 (B, M) 的 dB 方向图。
    """
    if theta_calc is None:
        theta_calc = np.linspace(0, 180, 361)

    pos = _to_tensor(pos)
    amp = _to_tensor(amp)
    phase = _to_tensor(phase)
    theta_calc = _to_tensor(theta_calc)

    batched = pos.dim() == 2
    theta_rad = theta_calc * (math.pi / 180.0)
    k = 2 * math.pi / lamb
    cos_t = torch.cos(theta_rad)

    if not batched:
        x = cos_t.unsqueeze(1) * pos.unsqueeze(0)
        psi = k * x - phase.unsqueeze(0)
        real = torch.sum(amp.unsqueeze(0) * torch.cos(psi), dim=1)
        imag = torch.sum(amp.unsqueeze(0) * torch.sin(psi), dim=1)
    else:
        x = cos_t.unsqueeze(0).unsqueeze(2) * pos.unsqueeze(1)
        psi = k * x - phase.unsqueeze(1)
        real = torch.sum(amp.unsqueeze(1) * torch.cos(psi), dim=2)
        imag = torch.sum(amp.unsqueeze(1) * torch.sin(psi), dim=2)

    pattern = torch.sqrt(real ** 2 + imag ** 2)
    if batched:
        peak = pattern.max(dim=1, keepdim=True)[0]
    else:
        peak = pattern.max()
    ratio = torch.clamp(pattern / (peak + 1e-30), min=1e-12)
    return 20 * torch.log10(ratio)


def calculate_2d_pattern(amp, phase, posx, posy,
                         theta_calc=None, phi_calc=None, lamb=1.0):
    """二维平面阵方向图（dB，归一化到峰值）。

    amp         (Nx, Ny)       激励幅值矩阵
    phase       (Nx, Ny)       激励相位矩阵（弧度）
    posx        (Nx,)          x 方向阵元位置
    posy        (Ny,)          y 方向阵元位置
    theta_calc  (Nt,)          俯仰角采样（度），默认 0–90 共 91 点
    phi_calc    (Np,)          方位角采样（度），默认 0–360 共 361 点

    返回 (Nt, Np) 的 dB 方向图。
    """
    if theta_calc is None:
        theta_calc = np.linspace(0, 90, 91)
    if phi_calc is None:
        phi_calc = np.linspace(0, 360, 361)

    amp = _to_tensor(amp)
    phase = _to_tensor(phase)
    posx = _to_tensor(posx)
    posy = _to_tensor(posy)
    theta_calc = _to_tensor(theta_calc)
    phi_calc = _to_tensor(phi_calc)

    Nx = posx.shape[0]
    Ny = posy.shape[0]

    theta_rad = theta_calc * (math.pi / 180.0)
    phi_rad = phi_calc * (math.pi / 180.0)
    k = 2 * math.pi / lamb

    sin_t = torch.sin(theta_rad)
    cos_p = torch.cos(phi_rad)
    sin_p = torch.sin(phi_rad)

    x = (posx.reshape(Nx, 1, 1, 1)
         * sin_t.reshape(1, 1, -1, 1)
         * cos_p.reshape(1, 1, 1, -1))
    y = (posy.reshape(1, Ny, 1, 1)
         * sin_t.reshape(1, 1, -1, 1)
         * sin_p.reshape(1, 1, 1, -1))

    psi = k * (x + y) - phase.reshape(Nx, Ny, 1, 1)
    real = torch.sum(amp.reshape(Nx, Ny, 1, 1) * torch.cos(psi), dim=(0, 1))
    imag = torch.sum(amp.reshape(Nx, Ny, 1, 1) * torch.sin(psi), dim=(0, 1))

    pattern = torch.sqrt(real ** 2 + imag ** 2)
    peak = pattern.max()
    ratio = torch.clamp(pattern / (peak + 1e-30), min=1e-12)
    return 20 * torch.log10(ratio)


# ============================================================
#  指标计算（numpy，验收用）
# ============================================================

def get_sll_1d(pattern_db, theta_deg,
               main_lobe_center=None, exclude_half_width=5.0):
    """一维方向图副瓣电平 (dB)。

    在 [main_lobe_center ± exclude_half_width] 之外搜索最大值。
    若 main_lobe_center 未指定则自动取峰值位置。

    旁瓣搜索区间：[0, center-exclude_half_width] ∪ [center+exclude_half_width, 180]
    """
    pattern_db = np.asarray(pattern_db)
    theta_deg = np.asarray(theta_deg)

    if main_lobe_center is None:
        idx_peak = np.argmax(pattern_db)
        main_lobe_center = theta_deg[idx_peak]

    mask = (np.abs(theta_deg - main_lobe_center) >= exclude_half_width)
    if not np.any(mask):
        return float('nan')
    return float(np.max(pattern_db[mask]))


def get_null_depth_1d(pattern_db, theta_deg, main_lobe_center,
                      search_half_width=5.0):
    """一维方向图零深 (dB)。

    在 [main_lobe_center ± search_half_width] 内搜索最小值。
    主要用于差波束零深评估。
    """
    pattern_db = np.asarray(pattern_db)
    theta_deg = np.asarray(theta_deg)

    mask = (np.abs(theta_deg - main_lobe_center) <= search_half_width)
    if not np.any(mask):
        return float('nan')
    return float(np.min(pattern_db[mask]))


def get_3db_beamwidth_1d(pattern_db, theta_deg, main_lobe_center=None):
    """一维方向图 3 dB 波束宽度 (度)。

    找主瓣峰值，向两侧搜索第一个低于 peak-3 dB 的点，返回角度差。
    """
    pattern_db = np.asarray(pattern_db)
    theta_deg = np.asarray(theta_deg)

    if main_lobe_center is None:
        idx_peak = np.argmax(pattern_db)
        main_lobe_center = theta_deg[idx_peak]
    else:
        idx_peak = int(np.argmin(np.abs(theta_deg - main_lobe_center)))

    peak_val = pattern_db[idx_peak]
    threshold = peak_val - 3.0

    left_idx = idx_peak
    for i in range(idx_peak - 1, -1, -1):
        if pattern_db[i] <= threshold:
            left_idx = i
            break
    else:
        left_idx = 0

    right_idx = idx_peak
    for i in range(idx_peak + 1, len(pattern_db)):
        if pattern_db[i] <= threshold:
            right_idx = i
            break
    else:
        right_idx = len(pattern_db) - 1

    return float(theta_deg[right_idx] - theta_deg[left_idx])


def get_2d_pattern_sll(mat, threshold=-10.0):
    """二维方向图副瓣电平 (dB)。

    用 8 邻域局部极大值法找所有局部峰值，取低于 threshold 的最大值。
    移植自原 AntennaCalculation.get_2d_pattern_SLL。
    """
    mat = np.asarray(mat)
    rows, cols = mat.shape
    padded = np.pad(mat, pad_width=1, mode='constant',
                    constant_values=-np.inf)
    center = padded[1:rows + 1, 1:cols + 1]
    top = padded[0:rows, 1:cols + 1]
    bottom = padded[2:rows + 2, 1:cols + 1]
    left = padded[1:rows + 1, 0:cols]
    right = padded[1:rows + 1, 2:cols + 2]
    tl = padded[0:rows, 0:cols]
    tr = padded[0:rows, 2:cols + 2]
    bl = padded[2:rows + 2, 0:cols]
    br = padded[2:rows + 2, 2:cols + 2]

    local_max = ((center > top) & (center > bottom) &
                 (center > left) & (center > right) &
                 (center > tl) & (center > tr) &
                 (center > bl) & (center > br))

    peaks = center[local_max]
    below = peaks[peaks <= threshold]
    if len(below) == 0:
        return float('nan')
    return float(np.max(below))


# ============================================================
#  2D 可分离激励与方向图（阶段 2）
# ============================================================

def taylor_2d_separable(Nx, Ny, SLL):
    """2D Taylor 可分离激励。

    返回 (amp_x, amp_y)，满足 amp_2d[i,j] = amp_x[i] * amp_y[j]。
    """
    posx = uniform_linear_array_pos(Nx)
    posy = uniform_linear_array_pos(Ny)
    amp_x = taylor_excitation(Nx * 0.5, posx, SLL)
    amp_y = taylor_excitation(Ny * 0.5, posy, SLL)
    return amp_x, amp_y


def beam_steering_phase_2d(posx, posy, theta0_deg, phi0_deg, lamb=1.0):
    """2D 波束指向相位（可分离）。

    返回 (phase_x, phase_y)，满足 phase_2d[i,j] = phase_x[i] + phase_y[j]。
    相位为弧度，已归一化到 [0, 2π)。
    """
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    k = 2 * np.pi / lamb
    u0 = np.sin(np.deg2rad(theta0_deg)) * np.cos(np.deg2rad(phi0_deg))
    v0 = np.sin(np.deg2rad(theta0_deg)) * np.sin(np.deg2rad(phi0_deg))
    phase_x = (k * u0 * posx) % (2 * np.pi)
    phase_y = (k * v0 * posy) % (2 * np.pi)
    return phase_x, phase_y


def combine_2d_excitation(amp_x, amp_y, phase_x, phase_y):
    """将 1D 激励组合为 2D 激励矩阵。

    amp_2d[i,j]   = amp_x[i] * amp_y[j]
    phase_2d[i,j] = phase_x[i] + phase_y[j]  (mod 2π)

    返回 (amp_2d, phase_2d)，形状 (Nx, Ny)。
    """
    amp_x = np.asarray(amp_x, dtype=np.float64)
    amp_y = np.asarray(amp_y, dtype=np.float64)
    phase_x = np.asarray(phase_x, dtype=np.float64)
    phase_y = np.asarray(phase_y, dtype=np.float64)

    amp_2d = np.outer(amp_x, amp_y)
    phase_2d = (np.outer(phase_x, np.ones_like(phase_y)) +
                np.outer(np.ones_like(phase_x), phase_y)) % (2 * np.pi)
    return amp_2d, phase_2d


def angular_distance_deg(theta1, phi1, theta2, phi2):
    """两点间的球面角距离（度）。

    用于主瓣排除区域的计算。
    """
    t1, p1 = np.deg2rad(theta1), np.deg2rad(phi1)
    t2, p2 = np.deg2rad(theta2), np.deg2rad(phi2)
    cos_delta = (np.cos(t1) * np.cos(t2) +
                 np.sin(t1) * np.sin(t2) * np.cos(p1 - p2))
    cos_delta = np.clip(cos_delta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_delta))


def get_2d_sll(pattern_db, theta_grid, phi_grid,
               theta0, phi0, exclude_angle=5.0):
    """二维方向图副瓣电平 (dB)，排除主瓣区域。

    Args:
        pattern_db: (Nt, Np) 方向图 (dB)
        theta_grid: (Nt,) 俯仰角采样 (度)
        phi_grid:   (Np,) 方位角采样 (度)
        theta0, phi0: 主瓣指向 (度)
        exclude_angle: 主瓣排除半径 (度)

    Returns:
        SLL (dB)，负值。
    """
    pattern_db = np.asarray(pattern_db)
    theta_grid = np.asarray(theta_grid)
    phi_grid = np.asarray(phi_grid)

    theta_2d, phi_2d = np.meshgrid(theta_grid, phi_grid, indexing='ij')
    dist = angular_distance_deg(theta_2d, phi_2d, theta0, phi0)

    mask = dist >= exclude_angle
    if not np.any(mask):
        return float('nan')
    return float(np.max(pattern_db[mask]))


def scan_angle_to_1d_theta(theta0_2d, phi0_2d):
    """将 2D 扫描角转换为两个等效 1D 波束角度。

    对于可分离阵列：
      x 方向等效 1D 角: θ_x = arccos(u0) = arccos(sin(θ₀)cos(φ₀))
      y 方向等效 1D 角: θ_y = arccos(v0) = arccos(sin(θ₀)sin(φ₀))

    Args:
        theta0_2d: 俯仰扫描角 (度), [0, 90]
        phi0_2d:   方位扫描角 (度), [0, 360]

    Returns:
        (theta_x, theta_y) 两个 1D 等效角度 (度)
    """
    u0 = np.sin(np.deg2rad(theta0_2d)) * np.cos(np.deg2rad(phi0_2d))
    v0 = np.sin(np.deg2rad(theta0_2d)) * np.sin(np.deg2rad(phi0_2d))
    u0 = np.clip(u0, -1.0, 1.0)
    v0 = np.clip(v0, -1.0, 1.0)
    theta_x = np.rad2deg(np.arccos(u0))
    theta_y = np.rad2deg(np.arccos(v0))
    return float(theta_x), float(theta_y)
