"""训练循环与辅助函数。

对应原 train.py，含：
  - CustomAccuracy（向量化 numpy 实现）
  - ReduceLROnPlateau / EarlyStopping
  - 手写训练循环（teacher forcing）
  - 评估函数
  - NPU/GPU/CPU 自动设备选择
"""

import numpy as np
import torch
import torch.nn.functional as F


def get_device(prefer='auto'):
    """自动选择计算设备。

    prefer: 'auto' | 'npu' | 'cuda' | 'cpu'
    auto 优先级: npu > cuda > cpu
    """
    if prefer in ('auto', 'npu'):
        try:
            import torch_npu
            if torch.npu.is_available():
                dev = torch.device('npu:0')
                print(f"Device: NPU ({torch.npu.get_device_name(0)})")
                return dev
        except ImportError:
            pass
    if prefer in ('auto', 'cuda'):
        if torch.cuda.is_available():
            dev = torch.device('cuda:0')
            print(f"Device: GPU ({torch.cuda.get_device_name(0)})")
            return dev
    print("Device: CPU")
    return torch.device('cpu')


def custom_accuracy(y_true, y_pred, tolerance=1e-2, separator_threshold=0.5):
    """自定义准确率（向量化，正确维度）。

    输入 (B, T, vocab, 2)：vocab=32（词表大小+separator），2=amp/phase通道。
    分别检查两个通道。padding（目标全零）不参与。
    """
    orig_shape = y_true.shape
    y_true = y_true.reshape(-1, orig_shape[-2], orig_shape[-1])
    y_pred = y_pred.reshape(-1, orig_shape[-2], orig_shape[-1])

    vocab = y_true.shape[-2]  # 32, 不是 2
    x = np.linspace(0.0, 1.0, vocab - 1)  # 31 个坐标

    correct_list = []
    total = 0
    for ch in range(orig_shape[-1]):
        yt = y_true[:, :, ch]  # (N, 32)
        yp = y_pred[:, :, ch]  # (N, 32)

        is_pad = np.all(yt == 0, axis=-1)
        is_sep = yt[:, -1] > separator_threshold

        valid = ~is_pad
        if not np.any(valid):
            continue

        pred_sep = yp[:, -1] > separator_threshold
        sep_correct = (is_sep & pred_sep & valid).astype(np.float64)

        true_val = np.sum(yt[:, :-1] * x, axis=-1)
        pred_val = np.sum(yp[:, :-1] * x, axis=-1)
        diff = np.abs(true_val - pred_val)
        nonsep_correct = (~is_sep & (diff < tolerance) & valid).astype(np.float64)

        correct_list.append((sep_correct.sum() + nonsep_correct.sum()))
        total += np.sum(valid)

    if total == 0:
        return 0.0
    return float(sum(correct_list) / total)


class EarlyStopping:
    """早停（监控 loss，min 模式）。"""

    def __init__(self, patience=12):
        self.patience = patience
        self.best = float('inf')
        self.no_improve = 0
        self.should_stop = False

    def step(self, metric):
        if metric < self.best - 1e-8:
            self.best = metric
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                self.should_stop = True
        return self.should_stop


def masked_mse_loss(pred, target):
    """带 padding mask 的 MSE loss。

    padding 定义：该时间步的全部目标值（32维×2通道）均为零。
    """
    if target.dim() == 4:
        is_pad = (target.abs().sum(dim=(2, 3)) == 0)
    else:
        is_pad = (target.abs().sum(dim=-1) == 0)

    mask = (~is_pad).float().unsqueeze(-1)
    if target.dim() == 4:
        mask = mask.unsqueeze(-1)

    sq_err = (pred - target) ** 2
    masked = sq_err * mask
    n_valid = mask.sum() * target.shape[-1] * target.shape[-2] if target.dim() == 4 \
        else mask.sum() * target.shape[-1]
    return masked.sum() / (n_valid + 1e-8)


def train_model(model, encoder_input, decoder_input, decoder_output,
                batch_size=64, epochs=50, learning_rate=1e-3,
                patience_lr=3, patience_stop=12, verbose=True,
                device=None, save_path=None, save_every=50,
                val_ratio=0.1):
    """训练 Seq2SeqModel。

    返回 (model, history)：
      history: {'loss': [...], 'accuracy': [...], 'lr': [...], 'val_loss': [...], 'val_acc': [...]}
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience_lr)
    early_stop = EarlyStopping(patience=patience_stop)

    n_samples = len(encoder_input)
    n_val = int(n_samples * val_ratio)
    indices = np.random.permutation(n_samples)
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    history = {'loss': [], 'val_loss': [], 'lr': []}
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        np.random.shuffle(train_idx)

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, len(train_idx), batch_size):
            idx = train_idx[start:start + batch_size]
            enc_x = torch.as_tensor(encoder_input[idx], dtype=torch.float32, device=device)
            dec_x = torch.as_tensor(decoder_input[idx], dtype=torch.float32, device=device)
            dec_y = torch.as_tensor(decoder_output[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred = model(enc_x, dec_x)
            loss = masked_mse_loss(pred, dec_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        cur_lr = optimizer.param_groups[0]['lr']

        # 验证集
        val_loss, val_acc = evaluate_model(
            model, encoder_input[val_idx], decoder_input[val_idx],
            decoder_output[val_idx], batch_size=batch_size, device=device,
            verbose=False)

        history['loss'].append(avg_loss)
        history['lr'].append(cur_lr)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f}  val_loss={val_loss:.6f}  "
                  f"lr={cur_lr:.2e}")

        # 保存最佳模型（而非最后模型）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save(model.state_dict(), save_path)
                if verbose:
                    print(f"  → best model saved: {save_path} (val_loss={val_loss:.6f})")

        if early_stop.step(val_loss):
            if verbose:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(best val_loss={early_stop.best:.6f})")
            break

    return model, history


def evaluate_model(model, encoder_input, decoder_input, decoder_output,
                   batch_size=64, device=None, verbose=True):
    """评估模型，返回 (loss, accuracy)。"""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    n_samples = len(encoder_input)

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            enc_x = torch.as_tensor(encoder_input[start:end], dtype=torch.float32, device=device)
            dec_x = torch.as_tensor(decoder_input[start:end], dtype=torch.float32, device=device)
            dec_y = torch.as_tensor(decoder_output[start:end], dtype=torch.float32, device=device)

            pred = model(enc_x, dec_x)
            loss = masked_mse_loss(pred, dec_y)

            total_loss += loss.item()
            total_acc += custom_accuracy(
                dec_y.cpu().numpy(), pred.cpu().numpy())
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches
