"""DeepSets 模型测试：形状、排列等变性、梯度流、权值修正。"""

import sys, os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mylib.deepsets import DeepSetsModel, count_parameters, apply_ai_correction


def test_model_shape():
    """模型输入输出形状正确。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=64, output_dim=2)
    x = torch.randn(4, 1024, 9)
    out = model(x)
    assert out.shape == (4, 1024, 2), f"Expected (4, 1024, 2), got {out.shape}"
    print(f"PASS: test_model_shape (output={out.shape})")


def test_parameter_count():
    """参数量合理（< 1M）。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=128, output_dim=2)
    n = count_parameters(model)
    assert 1000 < n < 1000000, f"Parameter count {n} out of range"
    print(f"PASS: test_parameter_count ({n:,} params)")


def test_batch_size_one():
    """batch_size=1 可运行。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    x = torch.randn(1, 1024, 9)
    out = model(x)
    assert out.shape == (1, 1024, 2)
    print(f"PASS: test_batch_size_one")


def test_variable_n_elements():
    """不同阵元数可运行（排列等变网络支持变长）。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    for n in [100, 500, 1024]:
        x = torch.randn(2, n, 9)
        out = model(x)
        assert out.shape == (2, n, 2), f"N={n}: got {out.shape}"
    print(f"PASS: test_variable_n_elements (N=100,500,1024)")


def test_permutation_equivariance():
    """排列等变性：打乱输入阵元顺序→输出对应打乱。"""
    torch.manual_seed(42)
    model = DeepSetsModel(input_dim=9, hidden_dim=64, output_dim=2)
    model.eval()

    x = torch.randn(1, 256, 9)

    with torch.no_grad():
        out_original = model(x)
        perm = torch.randperm(256)
        x_perm = x[:, perm, :]
        out_perm = model(x_perm)

    inv_perm = torch.argsort(perm)
    out_restored = out_perm[:, inv_perm, :]

    max_diff = (out_original - out_restored).abs().max().item()
    assert max_diff < 1e-5, f"Equivariance violated: max_diff={max_diff:.2e}"
    print(f"PASS: test_permutation_equivariance (max_diff={max_diff:.2e})")


def test_gradient_flow():
    """梯度可回传。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    x = torch.randn(2, 128, 9)
    target = torch.randn(2, 128, 2)
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    has_grad = all(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "Some parameters have no gradient"
    print(f"PASS: test_gradient_flow (loss={loss.item():.6f})")


def test_apply_ai_correction():
    """apply_ai_correction 正确加法。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    model.eval()
    feat = torch.randn(2, 64, 9)
    w_re = torch.randn(2, 64)
    w_im = torch.randn(2, 64)

    out_re, out_im = apply_ai_correction(model, feat, w_re, w_im)

    with torch.no_grad():
        delta = model(feat)
    expected_re = w_re + delta[..., 0]
    expected_im = w_im + delta[..., 1]

    max_err = (out_re - expected_re).abs().max().item()
    assert max_err < 1e-5, f"Correction mismatch: {max_err:.2e}"
    print(f"PASS: test_apply_ai_correction (max_err={max_err:.2e})")


def test_output_finite():
    """输出无 NaN/Inf。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    x = torch.randn(4, 512, 9)
    out = model(x)
    assert torch.isfinite(out).all(), "Output contains NaN/Inf"
    print(f"PASS: test_output_finite")


def test_device_transfer():
    """模型可在 CPU 上运行（NPU/GPU 测试在训练脚本中）。"""
    model = DeepSetsModel(input_dim=9, hidden_dim=32, output_dim=2)
    x = torch.randn(2, 128, 9)
    out = model(x)
    assert out.device == x.device
    print(f"PASS: test_device_transfer (device={out.device})")


if __name__ == '__main__':
    test_model_shape()
    test_parameter_count()
    test_batch_size_one()
    test_variable_n_elements()
    test_permutation_equivariance()
    test_gradient_flow()
    test_apply_ai_correction()
    test_output_finite()
    test_device_transfer()
    print("\n" + "=" * 60)
    print("Results: 9 passed, 0 failed, 9 total")
    print("=== ALL TESTS PASSED ===")
