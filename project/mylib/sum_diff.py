"""阶段 3: 和差波束与闭式置零。

实现内容：
  1. Bayliss 差波束激励生成（1D 和 2D 可分离）
  2. 和差波束联合方向图评估
  3. Capon 自适应置零（闭式解）
  4. 单脉冲测角指标

竞赛指标：
  - 和波束副瓣 ≤ -35 dBc
  - 差波束零深 ≤ -30 dBc
  - 差波束副瓣 ≤ -20 dBc
  - 自适应置零 ≤ -30 dBc (≥4 点, 主瓣之外)
  - 差波束指向精度 ≤ 1/30 · 3dB BW (RMS)
"""

import numpy as np

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_excitation,
    beam_steering_phase,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_1d_pattern,
    calculate_2d_pattern,
    get_sll_1d,
    get_null_depth_1d,
    get_3db_beamwidth_1d,
    get_2d_sll,
    angular_distance_deg,
)


# ============================================================
#  Bayliss 差波束激励
# ============================================================

_DIFFERENCE_AXIS_PHI_DEG = {
    "azimuth": 0.0,
    "elevation": 90.0,
}


def difference_axis_to_phi_deg(difference_axis):
    """Return the official uv-axis azimuth for a Difference axis.

    ``azimuth`` is the local tangent along +u/x.  ``elevation`` is the
    orthogonal local tangent along +v/y.  These are local array-plane axes,
    not a replacement for the global spherical ``theta``/``phi`` angles.
    """
    try:
        return float(_DIFFERENCE_AXIS_PHI_DEG[difference_axis])
    except (KeyError, TypeError):
        valid = ", ".join(sorted(_DIFFERENCE_AXIS_PHI_DEG))
        raise ValueError(
            f"difference_axis must be one of {{{valid}}}, got {difference_axis!r}"
        ) from None


def difference_null_is_legal(main_lobe, theta_deg, phi_deg):
    """Return whether a null is outside the official Difference main lobes."""
    target_u, target_v = main_lobe["target_uv"]
    axis = np.deg2rad(main_lobe["difference_axis_phi_deg"])
    du, dv = np.cos(axis), np.sin(axis)
    tu, tv = -dv, du
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    u = float(np.sin(theta) * np.cos(phi))
    v = float(np.sin(theta) * np.sin(phi))
    q = (u - target_u) * du + (v - target_v) * dv
    r = (u - target_u) * tu + (v - target_v) * tv
    negative = (
        main_lobe["negative_lobe_q_bounds"][0] <= q <= 0.0 and
        main_lobe["negative_lobe_r_bounds"][0] <= r <=
        main_lobe["negative_lobe_r_bounds"][1]
    )
    positive = (
        0.0 <= q <= main_lobe["positive_lobe_q_bounds"][1] and
        main_lobe["positive_lobe_r_bounds"][0] <= r <=
        main_lobe["positive_lobe_r_bounds"][1]
    )
    return not (negative or positive)

def bayliss_excitation(N, SLL, dtype=np.float32):
    """Bayliss 差波束激励幅值（1D）。

    采用 Taylor × 线性锥削法：Taylor 权乘以归一化位置坐标，
    产生平滑反对称激励，在阵中心自然过零，副瓣由 Taylor 锥削控制。

    Args:
        N: 阵元数（偶数）
        SLL: 目标副瓣电平 (dB, 正值如 25)
    Returns:
        (excitation, split_index) 激励幅值和左右分界
    """
    if N % 2 != 0:
        raise ValueError("Bayliss requires even N")

    pos = uniform_linear_array_pos(N)
    amp_taylor = taylor_excitation(N * 0.5, pos, SLL)

    pos_norm = pos / np.max(np.abs(pos))
    exc = amp_taylor * pos_norm

    M = N // 2
    if np.max(np.abs(exc)) > 0:
        exc = exc / np.max(np.abs(exc))

    return exc.astype(dtype), M


def bayliss_2d_separable(Nx, Ny, SLL, difference_axis="azimuth"):
    """2D Bayliss 可分离差波束激励。

    ``difference_axis='azimuth'`` 在 x/u 方向产生差零点，
    ``difference_axis='elevation'`` 在 y/v 方向产生差零点。
    另一维使用 Taylor 和波束激励。

    Returns:
        (amp_x, amp_y, split_diff), where the Bayliss taper is placed on
        the selected difference axis.
    """
    difference_axis_to_phi_deg(difference_axis)
    posy = uniform_linear_array_pos(Ny)
    if difference_axis == "azimuth":
        if Nx % 2 != 0:
            raise ValueError("Nx must be even for azimuth Bayliss")
        amp_x, split_x = bayliss_excitation(Nx, SLL)
        amp_y = taylor_excitation(Ny * 0.5, posy, SLL)
        return amp_x, amp_y, split_x
    if Ny % 2 != 0:
        raise ValueError("Ny must be even for elevation Bayliss")
    posx = uniform_linear_array_pos(Nx)
    amp_x = taylor_excitation(Nx * 0.5, posx, SLL)
    amp_y, split_y = bayliss_excitation(Ny, SLL)
    return amp_x, amp_y, split_y


# ============================================================
#  和差波束方向图
# ============================================================

def sum_diff_pattern_1d(pos, amp_sum, amp_diff, phase_sum, phase_diff,
                        theta_calc=None):
    """计算和差波束方向图。

    和波束: amp_sum * exp(j*phase_sum)
    差波束: amp_diff * exp(j*phase_diff) (左右反相)

    Returns:
        (sum_pattern_db, diff_pattern_db) 1D 方向图
    """
    if theta_calc is None:
        theta_calc = np.linspace(0, 180, 361)

    sum_pat = calculate_1d_pattern(pos, amp_sum, phase_sum, theta_calc).numpy()
    diff_pat = calculate_1d_pattern(pos, amp_diff, phase_diff, theta_calc).numpy()

    return sum_pat, diff_pat


# ============================================================
#  Capon 自适应置零
# ============================================================

def capon_nulling(pos, amp, phase, theta0,
                  null_directions, theta_calc=None,
                  lamb=1.0):
    """LCMV 置零（1D，以参考权值为中心的最小修正）。

    min ||w - w_ref||^2  s.t.  C^H w = f
    先归一化 w_ref 使主瓣响应 = 1，避免尺度不一致。
    """
    pos = np.asarray(pos, dtype=np.float64)
    N = len(pos)
    k = 2 * np.pi / lamb

    w_ref = amp * np.exp(1j * phase)
    a_main = np.exp(1j * k * np.cos(np.deg2rad(theta0)) * pos)

    # 关键修正：归一化 w_ref 使主瓣响应 = 1
    main_resp = a_main.conj() @ w_ref
    w_ref = w_ref / main_resp

    null_dirs = list(null_directions)
    cols = [a_main]
    for theta_null in null_dirs:
        cols.append(np.exp(1j * k * np.cos(np.deg2rad(theta_null)) * pos))

    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0  # 现在与 w_ref 的主瓣响应一致

    R_inv = np.eye(N, dtype=complex)
    CR = C.conj().T @ R_inv @ C
    residual = f - C.conj().T @ w_ref
    w_opt = w_ref + R_inv @ C @ np.linalg.lstsq(CR, residual, rcond=1e-10)[0]

    new_amp = np.abs(w_opt)
    if new_amp.max() > 0:
        new_amp = new_amp / new_amp.max()
    new_phase = np.angle(w_opt) % (2 * np.pi)
    return new_amp, new_phase


def capon_nulling_2d(posx, posy, amp_2d, phase_2d, theta0, phi0,
                     null_directions, lamb=1.0):
    """2D LCMV 置零（以参考权值为中心的最小修正）。

    先归一化 w_ref 使主瓣响应 = 1，避免尺度不一致。
    """
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = len(posx), len(posy)
    k = 2 * np.pi / lamb

    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    w_ref = (amp_2d * np.exp(1j * phase_2d)).ravel()

    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    a_main = np.exp(1j * k * (posx_2d * u0 + posy_2d * v0)).ravel()

    # 关键修正：归一化 w_ref 使主瓣响应 = 1
    main_resp = a_main.conj() @ w_ref
    w_ref = w_ref / main_resp

    cols = [a_main]
    for tn, pn in null_directions:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        cols.append(np.exp(1j * k * (posx_2d * un + posy_2d * vn)).ravel())

    C = np.column_stack(cols)
    f = np.zeros(len(cols), dtype=complex)
    f[0] = 1.0  # 现在与 w_ref 的主瓣响应一致

    R_inv = np.eye(Nx * Ny, dtype=complex)
    CR = C.conj().T @ R_inv @ C
    residual = f - C.conj().T @ w_ref
    w_opt = w_ref + R_inv @ C @ np.linalg.lstsq(CR, residual, rcond=1e-10)[0]

    w_mat = w_opt.reshape(Nx, Ny)
    new_amp = np.abs(w_mat)
    if new_amp.max() > 0:
        new_amp = new_amp / new_amp.max()
    new_phase = np.angle(w_mat) % (2 * np.pi)
    return new_amp, new_phase


def capon_nulling_difference_2d(posx, posy, amp_2d, phase_2d,
                                 theta0, phi0, null_directions,
                                 lamb=1.0, difference_axis="azimuth"):
    """2D LCMV nulling for a difference beam.

    The difference reference already has a null at ``(theta0, phi0)``;
    unlike :func:`capon_nulling_2d`, it therefore must not be normalized by a
    main-beam response.  The minimum-norm correction is projected onto the
    null space of the target direction plus the requested null directions::

        min ||w - w_ref||^2  subject to  C^H w = 0

    This keeps the intrinsic difference null while adding generic adaptive
    nulls.  Amplitude/phase output is normalized only by a common positive
    amplitude scale, which does not change any null response.

    ``difference_axis`` declares the local Difference axis used to construct
    the input reference: ``azimuth`` is +u/x and ``elevation`` is +v/y.  The
    steering constraints themselves are fully two-dimensional and therefore
    do not duplicate x-only or y-only solver logic.
    """
    difference_axis_to_phi_deg(difference_axis)
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    amp_2d = np.asarray(amp_2d, dtype=np.float64)
    phase_2d = np.asarray(phase_2d, dtype=np.float64)
    if amp_2d.shape != (len(posx), len(posy)):
        raise ValueError("amp_2d shape must equal (len(posx), len(posy))")
    if phase_2d.shape != amp_2d.shape:
        raise ValueError("phase_2d shape must equal amp_2d shape")

    nx, ny = len(posx), len(posy)
    k = 2 * np.pi / float(lamb)
    posx_2d = np.tile(posx[:, None], (1, ny))
    posy_2d = np.tile(posy[None, :], (nx, 1))
    w_ref = (amp_2d * np.exp(1j * phase_2d)).ravel()

    def steering(theta_deg, phi_deg):
        u = np.sin(np.deg2rad(theta_deg)) * np.cos(np.deg2rad(phi_deg))
        v = np.sin(np.deg2rad(theta_deg)) * np.sin(np.deg2rad(phi_deg))
        return np.exp(1j * k * (posx_2d * u + posy_2d * v)).ravel()

    # Include the intrinsic difference-null direction explicitly.  The
    # reference is already zero there, but the constraint protects it from
    # numerical coupling with the adaptive nulls.
    cols = [steering(theta0, phi0)]
    cols.extend(steering(theta_null, phi_null)
                for theta_null, phi_null in null_directions)
    C = np.column_stack(cols)
    gram = C.conj().T @ C
    residual = -C.conj().T @ w_ref
    correction = C @ np.linalg.lstsq(gram, residual, rcond=1e-10)[0]
    w_opt = w_ref + correction

    w_mat = w_opt.reshape(nx, ny)
    new_amp = np.abs(w_mat)
    max_amp = float(np.max(new_amp))
    if max_amp > 0.0:
        new_amp = new_amp / max_amp
    new_phase = np.angle(w_mat) % (2 * np.pi)
    return new_amp, new_phase


# ============================================================
#  单脉冲测角指标
# ============================================================

def monopulse_metrics(sum_pat_db, diff_pat_db, theta_deg, theta0):
    """单脉冲测角指标。

    Args:
        sum_pat_db: (M,) 和波束方向图 (dB)
        diff_pat_db: (M,) 差波束方向图 (dB)
        theta_deg: (M,) 角度采样
        theta0: 主瓣指向 (度)

    Returns:
        dict with:
          - null_depth: 差波束零深 (dB)
          - null_position: 零点位置 (度)
          - null_offset: 零点偏移 (度)
          - slope: 中心斜率 (1/度)
          - linear_region: 线性区宽度 (度)
          - monopulse_ratio: 中心单脉冲比 (dB)
    """
    sum_pat_db = np.asarray(sum_pat_db)
    diff_pat_db = np.asarray(diff_pat_db)
    theta_deg = np.asarray(theta_deg)

    idx_main = int(np.argmin(np.abs(theta_deg - theta0)))

    search_range = 5
    mask = np.abs(theta_deg - theta0) <= search_range
    if not np.any(mask):
        return {'null_depth': float('nan')}

    idx_null = np.argmin(diff_pat_db[mask])
    null_indices = np.where(mask)[0]
    null_idx = null_indices[idx_null]

    null_depth = float(diff_pat_db[null_idx])
    null_position = float(theta_deg[null_idx])
    null_offset = abs(null_position - theta0)

    if null_idx > 0 and null_idx < len(theta_deg) - 1:
        dt = theta_deg[null_idx + 1] - theta_deg[null_idx - 1]
        dr = (diff_pat_db[null_idx + 1] - diff_pat_db[null_idx - 1]) / dt
        slope = dr / max(sum_pat_db[idx_main] - diff_pat_db[null_idx], 1.0)
    else:
        slope = 0.0

    linear_threshold = 3.0
    linear_mask = np.abs(theta_deg - theta0) <= 10
    diff_linear = diff_pat_db[linear_mask]
    sum_linear = sum_pat_db[linear_mask]
    theta_linear = theta_deg[linear_mask]

    ratio = diff_linear - sum_linear
    center_ratio = ratio[len(ratio) // 2]

    above_threshold = np.abs(ratio - center_ratio) < linear_threshold
    if np.any(above_threshold):
        linear_region = float(theta_linear[above_threshold][-1] -
                              theta_linear[above_threshold][0])
    else:
        linear_region = 0.0

    return {
        'null_depth': null_depth,
        'null_position': null_position,
        'null_offset': null_offset,
        'slope': float(slope),
        'linear_region': linear_region,
        'monopulse_ratio': float(center_ratio),
    }


# ============================================================
#  指向精度
# ============================================================

def pointing_accuracy_1d(sum_pat_db, theta_deg, theta0, n_tests=1):
    """和波束指向精度 (RMS)。

    Returns:
        pointing_error_rms (度)
    """
    idx_peak = int(np.argmax(sum_pat_db))
    peak_theta = theta_deg[idx_peak]
    error = abs(peak_theta - theta0)
    return float(error)
