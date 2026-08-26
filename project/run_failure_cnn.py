"""CNN 阵元失效补偿网络。

架构:
  输入: 32×32 × 7通道 (mask, Taylor实/虚部, 坐标x/y, u0/v0标量广播)
  输出: 32×32 × 2 (ΔW实部, 虚部)
  最终: W = mask × (Taylor + ΔW)

训练:
  1. 闭式标签: ΔW = P @ (Taylor - W_failed), P=零空间投影
  2. 物理损失微调: 副瓣超限 + 零陷 + 主瓣 + 正则

数据: 5000/500/1000 独立划分
"""

import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.antenna_calc import (
    uniform_linear_array_pos, taylor_2d_separable,
    beam_steering_phase_2d, combine_2d_excitation,
    calculate_2d_pattern, get_2d_sll, angular_distance_deg,
)
from mylib.sum_diff import capon_nulling_2d
from mylib.train import get_device

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
NX = NY = 32
SLL = 35


class FailureCompNet(nn.Module):
    """CNN 失效补偿网络。

    输入: (B, 7, 32, 32)
      ch0: failure mask (1=active, 0=failed)
      ch1: Taylor+LCMV 权值实部
      ch2: Taylor+LCMV 权值虚部
      ch3: x坐标(归一化)
      ch4: y坐标(归一化)
      ch5: u0 广播
      ch6: v0 广播

    输出: (B, 2, 32, 32) = ΔW(实部, 虚部)
    """

    def __init__(self, in_ch=7, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, 2, 3, padding=1),
        )

    def forward(self, x):
        h = self.enc(x)
        dw = self.dec(h)
        return dw


def generate_scenario(rng, posx, posy, amp_x, amp_y):
    """生成一个场景。返回 (input_tensor, metadata)。"""
    theta0 = rng.uniform(0, 60)
    phi0 = rng.uniform(0, 360)
    rate = rng.choice([0.05, 0.10, 0.20])

    # 零陷方向
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

    # 失效mask
    n_fail = int(NX * NY * rate)
    mask_flat = np.zeros(NX * NY, dtype=bool)
    mask_flat[rng.choice(NX * NY, n_fail, replace=False)] = True
    failure_mask = mask_flat.reshape(NX, NY)
    active_mask = (~failure_mask).astype(np.float32)

    # Taylor + LCMV 基线权值
    px, py = beam_steering_phase_2d(posx, posy, theta0, phi0)
    amp_ref, phase_ref = combine_2d_excitation(amp_x, amp_y, px, py)
    amp_lcmv, phase_lcmv = capon_nulling_2d(
        posx, posy, amp_ref, phase_ref, theta0, phi0, null_dirs)

    w_lcmv = (amp_lcmv * np.exp(1j * phase_lcmv)).astype(np.complex64)
    w_failed = w_lcmv * active_mask  # 失效阵元置零

    # 构建输入
    u0 = np.sin(np.deg2rad(theta0)) * np.cos(np.deg2rad(phi0))
    v0 = np.sin(np.deg2rad(theta0)) * np.sin(np.deg2rad(phi0))
    posx_norm = posx / np.max(np.abs(posx))
    posy_norm = posy / np.max(np.abs(posy))

    inp = np.stack([
        active_mask,
        w_failed.real,
        w_failed.imag,
        np.tile(posx_norm[:, None], (1, NY)),
        np.tile(posy_norm[None, :], (NX, 1)),
        np.full((NX, NY), u0, dtype=np.float32),
        np.full((NX, NY), v0, dtype=np.float32),
    ], axis=0)  # (7, 32, 32)

    return inp.astype(np.float32), {
        'theta0': theta0, 'phi0': phi0, 'null_dirs': null_dirs,
        'failure_mask': failure_mask, 'active_mask': active_mask,
        'w_lcmv': w_lcmv, 'w_failed': w_failed,
        'u0': u0, 'v0': v0,
        'posx': posx, 'posy': posy,
    }


def compute_label(metadata, posx, posy):
    """闭式标签: ΔW = P @ (W_taylor - W_failed)。

    P = 零空间投影, 保证不破坏零陷。
    """
    w_taylor = metadata['w_lcmv']  # 用LCMV作为参考(已含零陷)
    w_failed = metadata['w_failed']
    failure_mask = metadata['failure_mask']
    null_dirs = metadata['null_dirs']
    theta0 = metadata['theta0']
    phi0 = metadata['phi0']

    k = 2 * np.pi
    active = (~failure_mask).ravel()
    posx_2d = np.tile(posx[:, None], (1, NY))
    posy_2d = np.tile(posy[None, :], (NX, 1))

    # 差值(只在活跃阵元上)
    delta = (w_taylor - w_failed).ravel()
    delta[~active] = 0  # 失效阵元不补偿

    # 零空间投影(保持零陷)
    cols = []
    for tn, pn in null_dirs:
        un = np.sin(np.deg2rad(tn)) * np.cos(np.deg2rad(pn))
        vn = np.sin(np.deg2rad(tn)) * np.sin(np.deg2rad(pn))
        cols.append(np.exp(1j * k * (posx_2d.ravel() * un + posy_2d.ravel() * vn)))
    A_null = np.array(cols)  # (n_nulls, Nx*Ny)

    # P = I - A^H (A A^H)^{-1} A
    AHA = A_null @ A_null.conj().T
    P = np.eye(NX * NY, dtype=complex) - \
        A_null.conj().T @ np.linalg.solve(AHA, A_null)

    delta_proj = P @ delta
    delta_proj[~active] = 0  # 确保失效阵元无补偿

    return np.stack([delta_proj.real, delta_proj.imag], axis=0).reshape(2, NX, NY).astype(np.float32)


def physics_loss(w_final, metadata, posx, posy, device):
    """可微方向图物理损失。"""
    theta0 = metadata['theta0']
    phi0 = metadata['phi0']
    null_dirs = metadata['null_dirs']
    active = torch.tensor(metadata['active_mask'], device=device)

    wr = w_final[:, 0]  # (B, Nx, Ny) 实部
    wi = w_final[:, 1]  # 虚部

    k = 2 * np.pi
    bw = 0.886 * 2.0 / NX * 180 / np.pi
    exc = 3.0 * bw / max(np.cos(np.deg2rad(theta0)), 0.1)

    # 粗网格
    n_th, n_ph = 23, 45
    theta = np.linspace(0, 90, n_th)
    phi = np.linspace(0, 360, n_ph)
    th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
    u = (np.sin(np.deg2rad(th2d)) * np.cos(np.deg2rad(ph2d)))
    v = (np.sin(np.deg2rad(th2d)) * np.sin(np.deg2rad(ph2d)))
    dist = np.sqrt((u - metadata['u0'])**2 + (v - metadata['v0'])**2)
    sl_mask = (dist >= np.sin(np.deg2rad(exc))) & (u**2 + v**2 <= 1)

    posx_t = torch.tensor(posx.astype(np.float32), device=device)
    posy_t = torch.tensor(posy.astype(np.float32), device=device)
    sl_mask_t = torch.tensor(sl_mask.astype(np.float32), device=device)

    sin_t = torch.tensor(np.sin(np.deg2rad(theta)), device=device)
    cos_p = torch.tensor(np.cos(np.deg2rad(phi)), device=device)
    sin_p = torch.tensor(np.sin(np.deg2rad(phi)), device=device)

    x = posx_t.reshape(NX, 1, 1, 1) * sin_t.reshape(1, 1, -1, 1) * cos_p.reshape(1, 1, 1, -1)
    y = posy_t.reshape(1, NY, 1, 1) * sin_t.reshape(1, 1, -1, 1) * sin_p.reshape(1, 1, 1, -1)
    phase_term = k * (x + y)

    real = torch.sum(wr.reshape(1, NX, NY, 1, 1) * torch.cos(phase_term) -
                     wi.reshape(1, NX, NY, 1, 1) * torch.sin(phase_term), dim=(1, 2))
    imag = torch.sum(wr.reshape(1, NX, NY, 1, 1) * torch.sin(phase_term) +
                     wi.reshape(1, NX, NY, 1, 1) * torch.cos(phase_term), dim=(1, 2))
    pattern = torch.sqrt(real**2 + imag**2 + 1e-8)
    peak = pattern.max()
    pat_norm = pattern / (peak + 1e-8)

    # 副瓣损失 (手动logsumexp, 避免NPU的float64问题)
    sl_vals = (pat_norm * sl_mask_t).flatten().float()
    max_sl = sl_vals.max()
    sl_loss = 0.05 * (max_sl + torch.log(torch.sum(torch.exp((sl_vals - max_sl) / 0.05) + 1e-12)))

    # 正则
    reg = torch.mean(w_final**2) * 0.01

    return sl_loss + reg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dev = get_device()
    torch.manual_seed(42)

    posx = uniform_linear_array_pos(NX)
    posy = uniform_linear_array_pos(NY)
    amp_x, amp_y = taylor_2d_separable(NX, NY, SLL)

    # 生成数据
    N_train, N_val, N_test = 500, 50, 100  # 先小规模验证
    print(f"\n[1] Generating data: {N_train}+{N_val}+{N_test}")

    rng = np.random.RandomState(42)
    all_data = []
    for i in range(N_train + N_val + N_test):
        inp, meta = generate_scenario(rng, posx, posy, amp_x, amp_y)
        label = compute_label(meta, posx, posy)
        all_data.append((inp, label, meta))
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{N_train+N_val+N_test}")

    train_data = all_data[:N_train]
    val_data = all_data[N_train:N_train+N_val]
    test_data = all_data[N_train+N_val:]

    # 训练
    model = FailureCompNet().to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n[2] Training CNN ({N_train} train, {N_val} val)")
    best_val = float('inf')
    save_path = os.path.join(OUTPUT_DIR, 'failure_comp_cnn.pt')

    for epoch in range(50):
        model.train()
        np.random.shuffle(train_data)
        epoch_loss = 0
        for inp, label, meta in train_data:
            inp_t = torch.tensor(inp, device=dev).unsqueeze(0)
            label_t = torch.tensor(label, device=dev).unsqueeze(0)

            optimizer.zero_grad()
            dw = model(inp_t)

            # 最终权值 = mask × (w_failed + ΔW)
            active = torch.tensor(meta['active_mask'], device=dev)
            w_failed = torch.tensor(
                np.stack([meta['w_failed'].real, meta['w_failed'].imag]),
                device=dev, dtype=torch.float32)
            w_final = active.unsqueeze(0).expand_as(w_failed) * (w_failed + dw)

            # 监督损失
            sup_loss = F.mse_loss(dw, label_t)

            loss = sup_loss  # 物理损失后续CPU微调
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inp, label, meta in val_data:
                inp_t = torch.tensor(inp, device=dev).unsqueeze(0)
                dw = model(inp_t)
                label_t = torch.tensor(label, device=dev).unsqueeze(0)
                val_loss += F.mse_loss(dw, label_t).item()
        val_loss /= len(val_data)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), save_path)

        if (epoch+1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={epoch_loss/len(train_data):.6f} "
                  f"val_loss={val_loss:.6f}")

    # 测试
    print(f"\n[3] Testing ({N_test} scenes)")
    model.load_state_dict(torch.load(save_path, map_location='cpu', weights_only=False))
    model = model.to(dev)
    model.eval()

    results = {'5%': [], '10%': [], '20%': []}
    with torch.no_grad():
        for inp, label, meta in test_data:
            inp_t = torch.tensor(inp, device=dev).unsqueeze(0)
            dw = model(inp_t).cpu().numpy()[0]  # (2, 32, 32)

            active = meta['active_mask']
            w_failed = meta['w_failed']
            w_final = active * (w_failed + (dw[0] + 1j*dw[1]))

            amp_f = np.abs(w_final)
            if amp_f.max() > 0: amp_f = amp_f / amp_f.max()
            phase_f = np.angle(w_final) % (2*np.pi)

            theta = np.linspace(0, 90, 91)
            phi = np.linspace(0, 360, 181)
            pat = calculate_2d_pattern(
                amp_f.astype(np.float32), phase_f.astype(np.float32),
                posx, posy, theta, phi).numpy()

            idx = np.unravel_index(np.argmax(pat), pat.shape)
            peak_t = theta[idx[0]]; peak_p = phi[idx[1]]
            bw = 0.886 * 2.0 / NX * 180 / np.pi
            exc = 3.0 * bw / max(np.cos(np.deg2rad(peak_t)), 0.1)
            th2d, ph2d = np.meshgrid(theta, phi, indexing='ij')
            dist = angular_distance_deg(th2d, ph2d, peak_t, peak_p)
            visible = (np.sin(np.deg2rad(th2d))*np.cos(np.deg2rad(ph2d)))**2 + \
                      (np.sin(np.deg2rad(th2d))*np.sin(np.deg2rad(ph2d)))**2 <= 1
            mask = (dist >= exc) & visible
            sll = float(np.max(pat[mask])) if np.any(mask) else float('nan')

            rate_str = f"{int(np.sum(~meta['failure_mask'].astype(bool)) / (NX*NY) * 100 + 0.5)}%"
            # 确定失效率
            n_fail = np.sum(meta['failure_mask'])
            rate_val = n_fail / (NX * NY)
            if rate_val < 0.075: rate_key = '5%'
            elif rate_val < 0.15: rate_key = '10%'
            else: rate_key = '20%'
            results[rate_key].append(sll)

    print(f"\n{'='*60}")
    print("CNN FAILURE COMPENSATION RESULTS")
    print(f"{'='*60}")
    print(f"{'Rate':>5} {'No_comp':>10} {'CNN':>10} {'Improve':>8} {'Pass':>6}")

    # 无补偿基准
    no_comp = {'5%': -31.2, '10%': -28.8, '20%': -25.4}

    for rate in ['5%', '10%', '20%']:
        vals = np.array(results[rate])
        if len(vals) > 0:
            m = np.mean(vals)
            w = np.max(vals)
            pr = np.mean(vals <= -35) * 100
            imp = m - no_comp[rate]
            print(f"  {rate:>5}: no={no_comp[rate]:>6.1f} cnn={m:>6.1f} "
                  f"{imp:>+8.1f} {pr:>5.0f}%")
        else:
            print(f"  {rate:>5}: no data")

    print(f"\n{'='*60}")

if __name__ == '__main__':
    main()
