"""训练管线正确性验证。

验证 loss、accuracy、padding mask、通道分离是否正确工作。
这些测试确保训练管线本身没有 bug，而非模型性能。
"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mylib.train import masked_mse_loss, custom_accuracy
from mylib.dataset import create_dataset, prepare_training_data


def test_perfect_prediction_loss_zero():
    """完美预测的 masked_mse_loss 应为 0。"""
    X, Y = create_dataset([15, 20], [90, 120], [25, 30], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = torch.as_tensor(dec_out, dtype=torch.float32)
    pred = target.clone()

    loss = masked_mse_loss(pred, target)
    assert loss.item() < 1e-10, f"perfect prediction loss={loss.item()}"
    print("PASS: test_perfect_prediction_loss_zero")


def test_random_prediction_loss_nonzero():
    """随机预测的 loss 应显著大于 0。"""
    X, Y = create_dataset([20], [90], [25], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = torch.as_tensor(dec_out, dtype=torch.float32)
    pred = torch.rand_like(target)

    loss = masked_mse_loss(pred, target)
    assert loss.item() > 0.01, f"random prediction loss={loss.item()}, expected > 0.01"
    print(f"PASS: test_random_prediction_loss_nonzero (loss={loss.item():.4f})")


def test_padding_excluded_from_loss():
    """padding 时间步不应影响 loss。"""
    X, Y = create_dataset([15], [90], [25], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = torch.as_tensor(dec_out, dtype=torch.float32)

    # 在 padding 区域放完全不同的值
    pred_good = target.clone()
    pred_bad = target.clone()
    # 找到 padding 时间步
    is_pad = (target.abs().sum(dim=(2, 3)) == 0).squeeze()
    pad_indices = torch.where(is_pad)[0]
    for idx in pad_indices:
        pred_bad[0, idx] = torch.rand(32, 2) * 100

    loss_good = masked_mse_loss(pred_good, target)
    loss_bad = masked_mse_loss(pred_bad, target)
    assert abs(loss_good.item() - loss_bad.item()) < 1e-10, \
        f"padding should not affect loss: good={loss_good.item()}, bad={loss_bad.item()}"
    print("PASS: test_padding_excluded_from_loss")


def test_perfect_prediction_accuracy_one():
    """完美预测的 accuracy 应为 1.0。"""
    X, Y = create_dataset([15, 20], [90, 120], [25, 30], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = dec_out
    pred = target.copy()

    acc = custom_accuracy(target, pred)
    assert abs(acc - 1.0) < 1e-6, f"perfect accuracy={acc}, expected 1.0"
    print("PASS: test_perfect_prediction_accuracy_one")


def test_random_prediction_accuracy_low():
    """随机预测的 accuracy 应很低（< 10%）。"""
    X, Y = create_dataset([20], [90], [25], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = dec_out
    pred = np.random.rand(*target.shape)

    acc = custom_accuracy(target, pred)
    assert acc < 0.1, f"random accuracy={acc}, expected < 0.1"
    print(f"PASS: test_random_prediction_accuracy_low (acc={acc:.4f})")


def test_channels_not_mixed():
    """幅值和相位通道不应混合。"""
    X, Y = create_dataset([20], [90], [25], reference='Taylor')
    _, _, dec_out = prepare_training_data(X, Y)
    target = dec_out

    # 只改幅值通道，不改相位通道
    pred = target.copy()
    pred[:, :, :, 0] = np.random.rand(*pred[:, :, :, 0].shape)

    acc = custom_accuracy(target, pred)
    # 幅值完全错误，相位完全正确
    # accuracy 应在 0~0.5 之间（幅值通道0%，相位通道100%）
    assert 0.0 < acc < 0.6, f"channel-mixed accuracy={acc}, expected 0 < acc < 0.6"
    print(f"PASS: test_channels_not_mixed (acc={acc:.4f}, amp wrong, phase correct)")


def test_position_encoding_consistent():
    """不同 N 值的位置编码应使用相同范围。"""
    from mylib.antenna_calc import uniform_linear_array_pos

    for N in [15, 20, 25, 32, 48]:
        pos = uniform_linear_array_pos(N)
        pos_norm = pos / np.max(np.abs(pos))
        assert abs(pos_norm.min() + 1.0) < 0.01, \
            f"N={N}: pos_norm min={pos_norm.min()}, expected -1"
        assert abs(pos_norm.max() - 1.0) < 0.01, \
            f"N={N}: pos_norm max={pos_norm.max()}, expected 1"

    X15, _ = create_dataset([15], [90], [25], reference='Taylor')
    X48, _ = create_dataset([48], [90], [25], reference='Taylor')
    assert X15.shape[2] == X48.shape[2], "vocab size should be same"
    print("PASS: test_position_encoding_consistent (all N normalized to [-1,1])")


def test_val_loss_tracking():
    """验证集 loss 应与训练 loss 数量级相当。"""
    X, Y = create_dataset([15, 20, 25], [60, 90, 120], [25, 30], reference='Taylor')
    enc, dec_in, dec_out = prepare_training_data(X, Y)
    target = torch.as_tensor(dec_out, dtype=torch.float32)
    pred = target + torch.randn_like(target) * 0.1

    train_loss = masked_mse_loss(pred, target).item()
    val_loss = masked_mse_loss(pred[:len(pred)//10], target[:len(target)//10]).item()
    ratio = max(train_loss, val_loss) / (min(train_loss, val_loss) + 1e-10)
    assert ratio < 2.0, f"train/val loss ratio={ratio}, expected < 2"
    print(f"PASS: test_val_loss_tracking (train={train_loss:.4f}, val={val_loss:.4f})")


if __name__ == '__main__':
    tests = [
        test_perfect_prediction_loss_zero,
        test_random_prediction_loss_nonzero,
        test_padding_excluded_from_loss,
        test_perfect_prediction_accuracy_one,
        test_random_prediction_accuracy_low,
        test_channels_not_mixed,
        test_position_encoding_consistent,
        test_val_loss_tracking,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("=== ALL TESTS PASSED ===")
