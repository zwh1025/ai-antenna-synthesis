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
    """自定义准确率（向量化）。

    对每个 (time_step, amp/phase) 的输出向量：
      - 若目标是 separator（末位 > threshold）：检查预测末位是否 > threshold
      - 若目标非 separator：计算两者的期望值（加权平均），检查绝对差 < tolerance
    """
    y_true = y_true.reshape(-1, y_true.shape[-2])
    y_pred = y_pred.reshape(-1, y_pred.shape[-2])

    dim = y_true.shape[-1]
    x = np.linspace(0.0, 1.0, dim - 1)

    is_sep = y_true[:, -1] > separator_threshold
    pred_sep = y_pred[:, -1] > separator_threshold

    sep_correct = (is_sep & pred_sep).astype(np.float64)

    true_val = np.sum(y_true[:, :-1] * x, axis=-1)
    pred_val = np.sum(y_pred[:, :-1] * x, axis=-1)
    diff = np.abs(true_val - pred_val)
    nonsep_correct = (~is_sep & (diff < tolerance)).astype(np.float64)

    total = len(y_true)
    if total == 0:
        return 0.0
    return float((sep_correct.sum() + nonsep_correct.sum()) / total)


class EarlyStopping:
    """早停（监控 accuracy，max 模式）。"""

    def __init__(self, patience=12):
        self.patience = patience
        self.best = -1.0
        self.no_improve = 0
        self.should_stop = False

    def step(self, metric):
        if metric > self.best + 1e-8:
            self.best = metric
            self.no_improve = 0
        else:
            self.no_improve += 1
            if self.no_improve >= self.patience:
                self.should_stop = True
        return self.should_stop


def train_model(model, encoder_input, decoder_input, decoder_output,
                batch_size=64, epochs=50, learning_rate=1e-3,
                patience_lr=3, patience_stop=12, verbose=True,
                device=None):
    """训练 Seq2SeqModel。

    返回 (model, history)：
      history: {'loss': [...], 'accuracy': [...], 'lr': [...]}
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience_lr)
    early_stop = EarlyStopping(patience=patience_stop)

    history = {'loss': [], 'accuracy': [], 'lr': []}
    n_samples = len(encoder_input)

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(n_samples)

        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = indices[start:start + batch_size]
            enc_x = torch.as_tensor(encoder_input[idx], dtype=torch.float32, device=device)
            dec_x = torch.as_tensor(decoder_input[idx], dtype=torch.float32, device=device)
            dec_y = torch.as_tensor(decoder_output[idx], dtype=torch.float32, device=device)

            optimizer.zero_grad()
            pred = model(enc_x, dec_x)
            loss = F.mse_loss(pred, dec_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += custom_accuracy(
                dec_y.cpu().numpy(), pred.detach().cpu().numpy())
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_acc = epoch_acc / n_batches
        cur_lr = optimizer.param_groups[0]['lr']

        history['loss'].append(avg_loss)
        history['accuracy'].append(avg_acc)
        history['lr'].append(cur_lr)

        scheduler.step(avg_acc)

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs}: "
                  f"loss={avg_loss:.6f}  acc={avg_acc:.4f}  "
                  f"lr={cur_lr:.2e}")

        if early_stop.step(avg_acc):
            if verbose:
                print(f"Early stopping at epoch {epoch+1} "
                      f"(best acc={early_stop.best:.4f})")
            break

    return model, history


def evaluate_model(model, encoder_input, decoder_input, decoder_output,
                   batch_size=64, device=None):
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
            loss = F.mse_loss(pred, dec_y)

            total_loss += loss.item()
            total_acc += custom_accuracy(
                dec_y.cpu().numpy(), pred.cpu().numpy())
            n_batches += 1

    return total_loss / n_batches, total_acc / n_batches
