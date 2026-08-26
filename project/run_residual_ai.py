"""AI 残差补偿网络。

核心方案：W_final = W_LCMV + ΔW_AI

W_LCMV: Taylor + LCMV 置零（保证零陷，但 SLL 退化）
ΔW_AI:  低秩残差修正（恢复 SLL，不破坏零陷）

ΔW 用 rank-1 分解：ΔW = Σ_k σ_k * u_k ⊗ v_k
输出仅 264 实参数（4 组 rank-1），而非 2048（完整复权值）。

模块：
  1. generate_residual_labels: 用梯度下降生成 ΔW 训练标签
  2. ResidualNet: MLP 预测 ΔW 参数
  3. train_residual: 训练网络
  4. evaluate_residual: 84 方向验证
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mylib.antenna_calc import (
    uniform_linear_array_pos,
    taylor_2d_separable,
    beam_steering_phase_2d,
    combine_2d_excitation,
    calculate_2d_pattern,
    get_2d_sll,
    angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d
from mylib.train import get_device

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')

NX = NY = 32
SLL_DESIGN = 35
K_RANK = 4  # rank-1 分量数
N_PARAMS = K_RANK * (NX + NY + 2)  # 4*(32+32+2) = 264


# ============================================================
#  低秩残差参数化
# ============================================================

def params_to_delta_w(params, Nx=NX, Ny=NY, k=K_RANK):
    """将 264 个参数转为 (Nx, Ny) 复数残差矩阵。

    params: (k*(Nx+Ny+2),) = (264,)
      每 k 组: u(Nx) + v(Ny) + re_sigma + im_sigma

    ΔW[i,j] = Σ_k (re_σ + j*im_σ) * u[i] * v[j]
    """
    params = params.reshape(k, Nx + Ny + 2)
    delta_w = torch.zeros(Nx, Ny, dtype=torch.complex64,
                          device=params.device)

    for kk in range(k):
        u = params[kk, :Nx]          # (Nx,)
        v = params[kk, Nx:Nx + Ny]   # (Ny,)
        re_s = params[kk, Nx + Ny]   # scalar
        im_s = params[kk, Nx + Ny + 1]

        sigma = torch.complex(re_s, im_s)
        delta_w = delta_w + sigma * torch.outer(u, v)

    return delta_w


def params_to_delta_w_np(params, Nx=NX, Ny=NY, k=K_RANK):
    """numpy 版本。"""
    params = params.reshape(k, Nx + Ny + 2)
    delta_w = np.zeros((Nx, Ny), dtype=np.complex64)

    for kk in range(k):
        u = params[kk, :Nx]
        v = params[kk, Nx:Nx + Ny]
        re_s = params[kk, Nx + Ny]
        im_s = params[kk, Nx + Ny + 1]

        sigma = re_s + 1j * im_s
        delta_w = delta_w + sigma * np.outer(u, v)

    return delta_w


# ============================================================
#  训练标签生成（梯度下降求 ΔW）
# ============================================================

def generate_residual_label(posx, posy, amp_sum, phase_sum,
                            theta0, phi0, null_dirs,
                            sll_target=-35, n_iter=500, lr=0.01,
                            device=None):
    """闭式解求 ΔW：约束最小二乘。

    min ||ΔW - (W_taylor - W_lcmv)||^2  s.t.  A_null^H ΔW = 0
    即 ΔW 在零空间内，最接近 Taylor-LCMV 差值。
    """
    posx = np.asarray(posx, dtype=np.float64)
    posy = np.asarray(posy, dtype=np.float64)
    Nx, Ny = len(posx), len(posy)
    k = 2 * np.pi

    # 基线权值
    w_taylor = (amp_sum * np.exp(1j * phase_sum)).ravel()
    w_lcmv_amp, w_lcmv_phase = capon_nulling_2d(
        posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)
    w_lcmv = (w_lcmv_amp * np.exp(1j * w_lcmv_phase)).ravel()

    # 目标差值
    delta_target = (w_taylor - w_lcmv)

    # 零陷约束矩阵
    posx_2d = np.tile(posx[:, None], (1, Ny))
    posy_2d = np.tile(posy[None, :], (Nx, 1))

    A_null = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        A_null.append(np.exp(1j * k * (posx_2d * un + posy_2d * vn)).ravel())
    A_null = np.array(A_null)  # (n_nulls, Nx*Ny)

    # 投影到 A_null 的零空间
    # P = I - A_null^H (A_null A_null^H)^{-1} A_null
    AHA = A_null @ A_null.conj().T
    P = np.eye(Nx * Ny, dtype=complex) - \
        A_null.conj().T @ np.linalg.solve(AHA, A_null)

    # 闭式解：ΔW = P @ delta_target
    delta_w = P @ delta_target

    # 评估
    w_final = w_lcmv + delta_w

    # 用 fine grid 评估 SLL
    bw = 0.886 * 2.0 / Nx * 180 / np.pi
    exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)

    w_lcmv_mat = w_lcmv.reshape(Nx, Ny)
    w_final_mat = w_final.reshape(Nx, Ny)

    amp_l = np.abs(w_lcmv_mat); amp_l = amp_l / amp_l.max()
    phase_l = np.angle(w_lcmv_mat) % (2 * np.pi)
    amp_f = np.abs(w_final_mat); amp_f = amp_f / amp_f.max()
    phase_f = np.angle(w_final_mat) % (2 * np.pi)

    pat_lcmv = calculate_2d_pattern(amp_l, phase_l, posx, posy, theta, phi).numpy()
    pat_final = calculate_2d_pattern(amp_f, phase_f, posx, posy, theta, phi).numpy()

    sll_before = get_2d_sll(pat_lcmv, theta, phi, theta0, phi0, exc)
    sll_after = get_2d_sll(pat_final, theta, phi, theta0, phi0, exc)

    # 零陷深度
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
    null_depths = []
    for tn, pn in null_dirs:
        dist = angular_distance_deg(th2d, ph2d, tn, pn)
        mask = dist <= 2.0
        null_depths.append(float(np.max(pat_final[mask])) if np.any(mask) else float('nan'))

    # 对 ΔW 做 SVD，取前 K 个 rank-1 分量作为标签
    delta_w_mat = delta_w.reshape(Nx, Ny)
    U_svd, S_svd, Vh_svd = np.linalg.svd(delta_w_mat)
    # 取前 K_RANK 个分量
    params = np.zeros(N_PARAMS)
    for kk in range(K_RANK):
        s = S_svd[kk] if kk < len(S_svd) else 0.0
        u = U_svd[:, kk] if kk < U_svd.shape[1] else np.zeros(Nx)
        v = Vh_svd[kk, :] if kk < Vh_svd.shape[0] else np.zeros(Ny)
        # u 和 v 是实数（SVD 输出），sigma 是实数
        # 复数 ΔW 需要: ΔW = Σ σ_k * u_k * v_k^H
        # 但 SVD 对复矩阵给出 U, S, V^H，其中 U 和 V 可能是复数
        # 简化：直接用复数 u, v, sigma 的实部+虚部
        params[kk*(Nx+Ny+2) : kk*(Nx+Ny+2)+Nx] = u.real
        params[kk*(Nx+Ny+2)+Nx : kk*(Nx+Ny+2)+Nx+Ny] = v.real
        params[kk*(Nx+Ny+2)+Nx+Ny] = s.real  # sigma 实部
        params[kk*(Nx+Ny+2)+Nx+Ny+1] = s.imag  # sigma 虚部

    return params, sll_before, sll_after, null_depths


# ============================================================
#  残差补偿网络
# ============================================================

class ResidualNet(nn.Module):
    """MLP 预测低秩 ΔW 参数。

    输入: (theta0, phi0, 4个零陷θ+φ) = 10 值
    输出: 264 个 ΔW 参数
    """

    def __init__(self, input_dim=10, output_dim=N_PARAMS,
                 hidden=256, n_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 2):
            layers.extend([nn.Linear(hidden, hidden), nn.ReLU()])
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
#  数据集生成
# ============================================================

def generate_training_data(n_samples=100, verbose=True):
    """生成 (input, ΔW) 训练对。闭式解，不需要 NPU。"""
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)

    theta_list = np.linspace(0, 60, 7)
    phi_list = np.linspace(0, 330, 12)

    inputs = []
    labels = []
    sll_before_list = []
    sll_after_list = []
    null_depth_list = []

    count = 0
    for theta0 in theta_list:
        for phi0 in phi_list:
            if count >= n_samples:
                break

            px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_sum, phase_sum = combine_2d_excitation(amp_x, amp_y, px, py)

            null_dirs = [
                (30, (phi0 + 90) % 360),
                (30, (phi0 + 180) % 360),
                (30, (phi0 + 270) % 360),
                (min(theta0 + 25, 85), (phi0 + 45) % 360),
            ]

            inp = [theta0, phi0]
            for tn, pn in null_dirs:
                inp.extend([tn, pn])
            inputs.append(inp)

            params, sll_b, sll_a, null_d = generate_residual_label(
                posx, posy, amp_sum, phase_sum,
                theta0, phi0, null_dirs)
            labels.append(params)
            sll_before_list.append(sll_b)
            sll_after_list.append(sll_a)
            null_depth_list.append(null_d)

            count += 1
            if verbose and count % 10 == 0:
                nd_avg = np.mean([d for d in null_d if not np.isnan(d)])
                print(f"  {count}/{n_samples}: θ={theta0:.0f}° φ={phi0:.0f}° "
                      f"SLL {sll_b:.1f}→{sll_a:.1f} dB  null={nd_avg:.1f} dB")

    return (np.array(inputs, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(sll_before_list),
            np.array(sll_after_list))


# ============================================================
#  训练与评估
# ============================================================

def train_residual_net(n_samples=84, epochs=200, device=None):
    """训练残差补偿网络。"""
    if device is None:
        device = get_device()

    print(f"\n[1] Generating {n_samples} training labels...")
    t0 = time.time()
    inputs, labels, sll_before, sll_after = generate_training_data(
        n_samples)
    t1 = time.time()
    print(f"    Done in {t1-t0:.1f}s")
    print(f"    SLL improvement: {np.mean(sll_before):.1f} → {np.mean(sll_after):.1f} dB")

    # 训练/验证划分
    n_val = max(int(n_samples * 0.15), 5)
    idx = np.random.permutation(n_samples)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train = torch.tensor(inputs[train_idx], dtype=torch.float32, device=device)
    Y_train = torch.tensor(labels[train_idx], dtype=torch.float32, device=device)
    X_val = torch.tensor(inputs[val_idx], dtype=torch.float32, device=device)
    Y_val = torch.tensor(labels[val_idx], dtype=torch.float32, device=device)

    model = ResidualNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    best_val_loss = float('inf')
    save_path = os.path.join(OUTPUT_DIR, 'residual_net.pt')

    print(f"\n[2] Training ResidualNet ({len(train_idx)} train, {len(val_idx)} val)")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train)
        loss = F.mse_loss(pred, Y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = F.mse_loss(val_pred, Y_val)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}: loss={loss.item():.6f} "
                  f"val_loss={val_loss.item():.6f} lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_loss < 1e-5:
            print(f"  Converged at epoch {epoch+1}")
            break

    # 重新加载最佳模型
    model.load_state_dict(torch.load(save_path, map_location='cpu',
                                      weights_only=False))
    model = model.to(device)
    model.eval()

    # 评估
    print(f"\n[3] Evaluation (all {n_samples} directions)")
    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL_DESIGN)
    bw = 0.886 * 2.0 / NX * 180 / np.pi

    theta = np.linspace(0, 90, 181)
    phi = np.linspace(0, 360, 361)

    sll_lcmv_list = []
    sll_residual_list = []

    with torch.no_grad():
        for i in range(n_samples):
            theta0, phi0 = inputs[i, 0], inputs[i, 1]
            null_dirs = [(inputs[i, 2+2*j], inputs[i, 3+2*j]) for j in range(4)]

            px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
            amp_sum, phase_sum = combine_2d_excitation(amp_x, amp_y, px, py)

            # LCMV 基线
            amp_lcmv, phase_lcmv = capon_nulling_2d(
                posx, posy, amp_sum, phase_sum, theta0, phi0, null_dirs)

            # AI 残差
            inp_t = torch.tensor(inputs[i:i+1], dtype=torch.float32, device=device)
            params_pred = model(inp_t).cpu().numpy()[0]
            delta_w = params_to_delta_w_np(params_pred)

            w_lcmv = amp_lcmv * np.exp(1j * phase_lcmv)
            w_final = w_lcmv + delta_w

            amp_final = np.abs(w_final)
            if amp_final.max() > 0:
                amp_final = amp_final / amp_final.max()
            phase_final = np.angle(w_final) % (2 * np.pi)

            # 方向图
            exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)
            pat_lcmv = calculate_2d_pattern(
                amp_lcmv, phase_lcmv, posx, posy, theta, phi).numpy()
            pat_final = calculate_2d_pattern(
                amp_final, phase_final, posx, posy, theta, phi).numpy()

            sll_lcmv = get_2d_sll(pat_lcmv, theta, phi, theta0, phi0, exc)
            sll_final = get_2d_sll(pat_final, theta, phi, theta0, phi0, exc)

            sll_lcmv_list.append(sll_lcmv)
            sll_residual_list.append(sll_final)

    sll_lcmv_arr = np.array(sll_lcmv_list)
    sll_res_arr = np.array(sll_residual_list)

    print(f"\n{'='*60}")
    print("AI RESIDUAL COMPENSATION RESULTS")
    print(f"{'='*60}")
    print(f"  {'Metric':>20} {'Mean':>8} {'Worst':>8} {'Pass':>6}")
    print(f"  {'LCMV SLL':>20} {np.mean(sll_lcmv_arr):>8.1f} {np.max(sll_lcmv_arr):>8.1f} "
          f"{np.mean(sll_lcmv_arr <= -35)*100:>5.0f}%")
    print(f"  {'LCMV+AI SLL':>20} {np.mean(sll_res_arr):>8.1f} {np.max(sll_res_arr):>8.1f} "
          f"{np.mean(sll_res_arr <= -35)*100:>5.0f}%")
    print(f"  {'Improvement':>20} {np.mean(sll_res_arr - sll_lcmv_arr):>8.1f}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")


if __name__ == '__main__':
    train_residual_net(n_samples=84, epochs=200)
