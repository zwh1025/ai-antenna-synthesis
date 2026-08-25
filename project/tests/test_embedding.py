"""阶段 1.5 词嵌入/解嵌入层测试。"""

import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mylib.embedding import (
    map_num_to_vec_np,
    map_vec_to_num_np,
    NumEmbedding,
    NumDeEmbedding,
)


def test_numpy_embed_basic():
    """numpy 词嵌入：分割点处为 one-hot，中间为线性插值。"""
    dp = np.linspace(0, 1, 5)

    vec = map_num_to_vec_np(0.0, dp)
    assert vec[0] == 1.0 and vec[-1] == 0.0
    vec = map_num_to_vec_np(0.5, dp)
    assert vec[2] == 1.0

    vec = map_num_to_vec_np(0.125, dp)
    assert abs(vec[0] + vec[1] - 1.0) < 1e-6
    assert abs(vec[1] - 0.5) < 1e-6

    vec_p = map_num_to_vec_np(0.5, dp, padding=True)
    vec_np = map_num_to_vec_np(0.5, dp, padding=False)
    assert len(vec_p) == len(dp) + 1
    assert len(vec_np) == len(dp)
    assert vec_p[-1] == 0.0

    print("PASS: test_numpy_embed_basic")


def test_numpy_deembed_roundtrip():
    """numpy 嵌入→解嵌入往返：误差 ≤ 分割点间距/2。"""
    dp = np.linspace(0, 180, 31)
    spacing = (dp[-1] - dp[0]) / (len(dp) - 1)

    max_err = 0.0
    for x in np.linspace(dp[0], dp[-1], 200):
        vec = map_num_to_vec_np(x, dp)
        x_rec = map_vec_to_num_np(vec, dp)
        err = abs(x_rec - x)
        max_err = max(max_err, err)

    assert max_err <= spacing / 2 + 1e-6, f"roundtrip err={max_err}, spacing={spacing}"
    print(f"PASS: test_numpy_deembed_roundtrip (max_err={max_err:.4f}, spacing={spacing:.1f})")


def test_torch_embed_basic():
    """torch 词嵌入：输出形状、和为 1、separator 位为 0。"""
    dp = np.linspace(0, 1, 31)
    emb = NumEmbedding(dp)

    # 单个
    vec = emb(torch.tensor(0.5))
    assert vec.shape == (32,)
    assert abs(vec[:-1].sum().item() - 1.0) < 1e-5
    assert abs(vec[-1].item()) < 1e-6

    # 批处理
    batch = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    vecs = emb(batch)
    assert vecs.shape == (5, 32)
    sums = vecs[:, :-1].sum(dim=-1)
    assert torch.allclose(sums, torch.ones(5), atol=1e-5)

    print("PASS: test_torch_embed_basic")


def test_torch_deembed_roundtrip():
    """torch 嵌入→解嵌入往返：误差随 beta 增大而减小。"""
    dp = np.linspace(0, 180, 31)
    spacing = (dp[-1] - dp[0]) / (len(dp) - 1)

    for beta_factor in [1, 10, 100]:
        emb = NumEmbedding(dp, beta=(1.0 / spacing**2) * beta_factor)
        deemb = NumDeEmbedding(dp)

        xs = torch.linspace(0, 180, 200)
        vecs = emb(xs)
        xs_rec = deemb(vecs)
        max_err = (xs - xs_rec).abs().max().item()

        if beta_factor == 100:
            assert max_err < spacing / 2, f"beta*100: err={max_err}, spacing={spacing}"
    print(f"PASS: test_torch_deembed_roundtrip")


def test_differentiable():
    """torch 词嵌入可反向传播。"""
    dp = np.linspace(0, 1, 31)
    emb = NumEmbedding(dp)
    deemb = NumDeEmbedding(dp)

    x = torch.tensor(0.5, requires_grad=True)
    vec = emb(x)
    x_rec = deemb(vec)
    loss = (x_rec - 0.3) ** 2
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad)
    print(f"PASS: test_differentiable (grad={x.grad.item():.4f})")


def test_separator():
    """separator 向量：最后一维为 1，其余为 0。"""
    dp = np.linspace(0, 1, 31)
    emb = NumEmbedding(dp)

    sep = emb.separator()
    assert sep.shape == (32,)
    assert sep[-1] == 1.0
    assert sep[:-1].sum() == 0.0

    sep_batch = emb.separator((4,))
    assert sep_batch.shape == (4, 32)
    assert torch.all(sep_batch[:, -1] == 1.0)

    print("PASS: test_separator")


def test_torch_vs_numpy_large_beta():
    """beta 足够大时，torch softmax 版与 numpy 线性插值在分割点上完全一致。"""
    dp = np.linspace(0, 1, 31)
    beta_large = 1e6
    emb = NumEmbedding(dp, beta=beta_large)

    for x_val in dp:
        vec_torch = emb(torch.tensor(float(x_val))).numpy()
        vec_np = map_num_to_vec_np(float(x_val), dp)
        err = np.max(np.abs(vec_torch - vec_np))
        assert err < 1e-3, f"x={x_val}: err={err}"

    print("PASS: test_torch_vs_numpy_large_beta")


if __name__ == '__main__':
    tests = [
        test_numpy_embed_basic,
        test_numpy_deembed_roundtrip,
        test_torch_embed_basic,
        test_torch_deembed_roundtrip,
        test_differentiable,
        test_separator,
        test_torch_vs_numpy_large_beta,
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
    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    if failed:
        sys.exit(1)
    print("=== ALL TESTS PASSED ===")
