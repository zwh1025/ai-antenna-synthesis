"""DeepSets 训练：曲面阵列坐标→权值 AI 综合。

数据流：
  输入 (batch, N, 9): (x, y, z, w_bl_re, w_bl_im, u0, v0, w0, sll)
  目标 ΔW (batch, N, 2): (w_socp_re - w_bl_re, w_socp_im - w_bl_im)
  网络 → ΔW_AI
  最终权值 W = W_baseline + ΔW_AI

训练：
  损失 = MSE(ΔW_AI, ΔW_target) + λ * physics_loss（可选）
  评估 = 三方对比 Taylor vs SOCP vs AI
  退出条件 = AI 恢复 SOCP 改善量的 80% 以上
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mylib.deepsets import DeepSetsModel, count_parameters
from mylib.train import get_device, EarlyStopping
from run_curved_verify import eval_dense_3d

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'outputs')
COORD_NORM = 8.0
SLL_NORM = 50.0
WEIGHT_SCALE = 1024.0


def load_teacher_labels(path=None):
    """加载 SOCP 教师标签，返回训练/验证/测试数据。"""
    if path is None:
        path = os.path.join(OUTPUT_DIR, 'teacher_labels.npz')
    data = np.load(path)

    def make_features(idx):
        px = data['px'][idx]
        py = data['py'][idx]
        pz = data['pz'][idx]
        w_re = data['w_taylor_re'][idx]
        w_im = data['w_taylor_im'][idx]
        u0 = data['u0'][idx]
        v0 = data['v0'][idx]
        w0 = data['w0'][idx]
        sll = np.full(len(idx), 35.0, dtype=np.float32)

        u0_2d = u0[:, None] * np.ones((len(idx), px.shape[1]))
        v0_2d = v0[:, None] * np.ones((len(idx), px.shape[1]))
        w0_2d = w0[:, None] * np.ones((len(idx), px.shape[1]))
        sll_2d = sll[:, None] / SLL_NORM * np.ones((len(idx), px.shape[1]))

        feat = np.stack([
            px / COORD_NORM, py / COORD_NORM, pz / COORD_NORM,
            w_re * WEIGHT_SCALE, w_im * WEIGHT_SCALE,
            u0_2d, v0_2d, w0_2d,
            sll_2d,
        ], axis=-1).astype(np.float32)

        delta_re = ((data['w_socp_re'][idx] - w_re) * WEIGHT_SCALE).astype(np.float32)
        delta_im = ((data['w_socp_im'][idx] - w_im) * WEIGHT_SCALE).astype(np.float32)
        target = np.stack([delta_re, delta_im], axis=-1).astype(np.float32)

        meta = {
            'px': px, 'py': py, 'pz': pz,
            'w_taylor_re': w_re, 'w_taylor_im': w_im,
            'w_socp_re': data['w_socp_re'][idx],
            'w_socp_im': data['w_socp_im'][idx],
            'sll_taylor': data['sll_taylor'][idx],
            'sll_socp': data['sll_socp'][idx],
            'alpha': data['alpha'][idx],
            'theta0': data['theta0'][idx],
            'phi0': data['phi0'][idx],
            'u0': u0, 'v0': v0, 'w0': w0,
        }
        return feat, target, meta

    split = data['split']
    train_idx = np.where(split == 0)[0]
    val_idx = np.where(split == 1)[0]
    test_idx = np.where(split == 2)[0]

    train_feat, train_target, train_meta = make_features(train_idx)
    val_feat, val_target, val_meta = make_features(val_idx)
    test_feat, test_target, test_meta = make_features(test_idx)

    return {
        'train': (train_feat, train_target, train_meta),
        'val': (val_feat, val_target, val_meta),
        'test': (test_feat, test_target, test_meta),
    }


def _get_null_dirs(theta0, phi0):
    """根据扫描方向计算零陷方向（与 run_multi_scan_generate 一致）。"""
    if theta0 < 10:
        return [(30, 0), (30, 90), (30, 180), (30, 270)]
    return [
        (theta0, (phi0 + 90) % 360),
        (theta0, (phi0 + 180) % 360),
        (theta0, (phi0 + 270) % 360),
        (min(theta0 + 25, 85), (phi0 + 45) % 360),
    ]


def evaluate_ai_weights(model, feat, meta, device, n_samples=None):
    """用 AI 权值计算 SLL，返回 (sll_list, taylor_list, socp_list)。"""
    model.eval()
    n = len(feat) if n_samples is None else min(n_samples, len(feat))
    sll_ai_list = []
    sll_taylor_list = list(meta['sll_taylor'][:n])
    sll_socp_list = list(meta['sll_socp'][:n])

    with torch.no_grad():
        for i in range(n):
            x = torch.as_tensor(feat[i:i+1], dtype=torch.float32, device=device)
            delta = model(x)[0].cpu().numpy()

            w_re = meta['w_taylor_re'][i] + delta[:, 0] / WEIGHT_SCALE
            w_im = meta['w_taylor_im'][i] + delta[:, 1] / WEIGHT_SCALE
            w_ai = w_re + 1j * w_im

            th0 = float(meta['theta0'][i])
            ph0 = float(meta['phi0'][i])
            null_dirs = _get_null_dirs(th0, ph0)

            sll, _, _, _ = eval_dense_3d(
                w_ai, meta['px'][i], meta['py'][i], meta['pz'][i],
                th0, ph0, null_dirs)
            sll_ai_list.append(float(sll) if not np.isnan(sll) else -100.0)

    return sll_ai_list, sll_taylor_list, sll_socp_list


def train_model(model, train_feat, train_target, val_feat, val_target,
                device, epochs=200, batch_size=16, lr=1e-3,
                patience_lr=5, patience_stop=20, save_path=None):
    """训练 DeepSets 模型。"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr)
    early_stop = EarlyStopping(patience=patience_stop)
    criterion = nn.MSELoss()

    train_ds = TensorDataset(
        torch.as_tensor(train_feat, dtype=torch.float32),
        torch.as_tensor(train_target, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_x = torch.as_tensor(val_feat, dtype=torch.float32, device=device)
    val_y = torch.as_tensor(val_target, dtype=torch.float32, device=device)

    best_val_loss = float('inf')
    history = {'loss': [], 'val_loss': [], 'lr': []}

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            pred = model(bx)
            loss = criterion(pred, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        avg_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()

        cur_lr = optimizer.param_groups[0]['lr']
        history['loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(cur_lr)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f} val_loss={val_loss:.6f} "
                  f"lr={cur_lr:.2e}")

        if early_stop.step(val_loss):
            print(f"  Early stopping at epoch {epoch+1} "
                  f"(best val_loss={early_stop.best:.6f})")
            break

    return model, history


def main():
    parser = argparse.ArgumentParser(description='DeepSets training')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--save_name', type=str, default='deepsets_model.pt')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device(args.device)
    save_path = os.path.join(OUTPUT_DIR, args.save_name)

    print("=" * 70)
    print("DeepSets Training: Curved Array Weight Synthesis")
    print("=" * 70)

    data = load_teacher_labels(args.data_path)
    train_feat, train_target, train_meta = data['train']
    val_feat, val_target, val_meta = data['val']
    test_feat, test_target, test_meta = data['test']

    print(f"Train: {len(train_feat)} samples, alpha=["
          f"{train_meta['alpha'].min():.3f}, {train_meta['alpha'].max():.3f}]")
    print(f"Val:   {len(val_feat)} samples, alpha=["
          f"{val_meta['alpha'].min():.3f}, {val_meta['alpha'].max():.3f}]")
    print(f"Test:  {len(test_feat)} samples, alpha=["
          f"{test_meta['alpha'].min():.3f}, {test_meta['alpha'].max():.3f}]")

    train_taylor_mean = np.mean(train_meta['sll_taylor'])
    train_socp_mean = np.mean(train_meta['sll_socp'])
    train_delta = train_socp_mean - train_taylor_mean
    print(f"Teacher SLL: taylor={train_taylor_mean:.1f} socp={train_socp_mean:.1f} "
          f"delta={train_delta:+.1f} dB")

    model = DeepSetsModel(input_dim=9, hidden_dim=args.hidden_dim, output_dim=2)
    print(f"Model parameters: {count_parameters(model):,}")

    print(f"\nTraining: epochs={args.epochs} batch_size={args.batch_size} "
          f"lr={args.lr}")
    print("-" * 70)

    t0 = time.time()
    model, history = train_model(
        model, train_feat, train_target, val_feat, val_target,
        device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, save_path=save_path)
    t_train = time.time() - t0
    print(f"\nTraining time: {t_train:.1f}s")

    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)

    print("\n" + "=" * 70)
    print("EVALUATION: Taylor vs SOCP vs AI")
    print("=" * 70)

    for split_name, (feat, target, meta) in [
        ('Val', data['val']), ('Test', data['test'])
    ]:
        sll_ai, sll_t, sll_s = evaluate_ai_weights(
            model, feat, meta, device)

        taylor_mean = np.mean(sll_t)
        socp_mean = np.mean(sll_s)
        ai_mean = np.mean(sll_ai)

        socp_improve = socp_mean - taylor_mean
        ai_improve = ai_mean - taylor_mean
        recovery = ai_improve / socp_improve * 100 if abs(socp_improve) > 0.01 else 0

        print(f"\n  {split_name} ({len(feat)} samples):")
        print(f"    Taylor:  {taylor_mean:.2f} dB")
        print(f"    SOCP:    {socp_mean:.2f} dB (delta={socp_improve:+.2f})")
        print(f"    AI:      {ai_mean:.2f} dB (delta={ai_improve:+.2f})")
        print(f"    Recovery: {recovery:.1f}% of SOCP improvement")
        if recovery >= 80:
            print(f"    → PASS: >=80% recovery target met")
        else:
            print(f"    → Below 80% recovery target")

        n_eval = min(20, len(feat))
        ai_time_per = 0
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_eval):
                x = torch.as_tensor(feat[i:i+1], dtype=torch.float32, device=device)
                _ = model(x)
        ai_time_per = (time.time() - t0) / n_eval * 1000
        print(f"    AI inference: {ai_time_per:.1f} ms/sample")

    hist_path = os.path.join(OUTPUT_DIR, 'deepsets_history.npz')
    np.savez(hist_path,
             loss=history['loss'], val_loss=history['val_loss'],
             lr=history['lr'])

    # 保存评估结果到JSON（可验证恢复率）
    results_json = {
        'model_params': count_parameters(model),
        'training_time_s': t_train,
        'epochs_trained': len(history['loss']),
        'teacher_sll': {'taylor': float(train_taylor_mean), 'socp': float(train_socp_mean)},
        'evaluation': {},
    }
    for split_name, (feat, target, meta) in [('val', data['val']), ('test', data['test'])]:
        sll_ai, sll_t, sll_s = evaluate_ai_weights(model, feat, meta, device)
        results_json['evaluation'][split_name] = {
            'n_samples': len(feat),
            'taylor_mean': float(np.mean(sll_t)),
            'socp_mean': float(np.mean(sll_s)),
            'ai_mean': float(np.mean(sll_ai)),
            'recovery_pct': float((np.mean(sll_ai) - np.mean(sll_t)) / (np.mean(sll_s) - np.mean(sll_t)) * 100)
                                if abs(np.mean(sll_s) - np.mean(sll_t)) > 0.01 else 0,
        }
    import json as json_mod
    results_path = os.path.join(OUTPUT_DIR, 'deepsets_results.json')
    with open(results_path, 'w') as f:
        json_mod.dump(results_json, f, indent=2, default=str)

    print(f"\nHistory saved: {hist_path}")
    print(f"Results saved: {results_path}")
    print(f"Model saved: {save_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
